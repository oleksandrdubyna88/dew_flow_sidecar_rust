use crate::compile_cache::CachePathLease;
use crate::inference::{pin_shape, ruler_text, SETTLE_ATTEMPTS};
use crate::provider::load_dual;
use crate::state::{AppState, Limits};
use anyhow::Context;
use fastembed::Bgem3DualEmbedding;
use std::sync::OnceLock;

// ---------- the build-time canary ----------
//
// Two defects surfaced by the 2026-07-31 parity gate, both the EP's, both invisible to every guard
// this sidecar had:
//   1. A crash mid-compile leaves a CORRUPT .mxr in the compiled-model cache that LOADS fine and
//      then stably produces garbage — full-length, plausibly shaped, reproducibly wrong (two
//      independent runs of the corrupt program matched each other bit-exactly while scoring
//      cosine 0.13 against the model's real output).
//   2. The first run(s) after a FRESH compile can return full-length garbage at small batch shapes.
//      At production shapes (batch 64) the defect manifests as a SHORT batch, which
//      `embed_settling` catches; at (6..8, cap) it sails through every length/shape check.
// The canary closes both: every freshly built engine must reproduce a known text's embedding
// (cosine against a reference captured from the parity-verified build) before it is allowed to
// serve. Retries absorb defect 2; a cache wipe + one clean recompile heals defect 1; anything past
// that fails the request rather than silently indexing garbage.

/// The exact string the stored reference vector was computed from. Changing it invalidates
/// `canary-reference.f32le` and fails every engine until the reference is regenerated from the new text —
/// which `write_reference` can now do (`--write-canary-reference`). Change the two together or not at all.
pub(crate) const CANARY_TEXT: &str =
    "A canary sentence for the bge-m3 engine self-check: the quick brown fox jumps over the lazy dog 0123456789.";

/// Deliberately loose: EP-to-EP arithmetic differences sit near 0.9999, the observed garbage at
/// 0.13. This is a corruption detector, not a numerics test — the parity harness owns exactness.
pub(crate) const CANARY_MIN_COSINE: f32 = 0.99;

/// The dense embedding of `CANARY_TEXT`, captured 2026-07-31 from the unified build that passed the
/// parity gate bit-exact at both caps, and reproduced at cosine 1.000000000 by this repository's own
/// generator on 2026-08-19.
///
/// Regenerate with `--write-canary-reference` (see `write_reference`) only when the MODEL deliberately
/// changes — **never to green a failing canary**. The tool prints the distance from this file before
/// replacing it precisely so that misuse has to be chosen rather than stumbled into.
pub(crate) fn canary_reference() -> &'static [f32] {
    static REFERENCE: OnceLock<Vec<f32>> = OnceLock::new();
    REFERENCE.get_or_init(|| decode_reference(include_bytes!("canary-reference.f32le")))
}

/// The stored form: little-endian `f32`, no header, no length — the file's size IS the dimension.
///
/// Kept as a pair with `encode_reference` and tested as a pair, because a generator that writes a format
/// the reader does not read produces a file that fails every engine on this machine and looks like a
/// hardware fault.
pub(crate) fn decode_reference(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect()
}

pub(crate) fn encode_reference(vector: &[f32]) -> Vec<u8> {
    vector
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

/// Plain cosine similarity; -1.0 on a dimension mismatch (a mismatched dim IS a failed canary, not
/// a panic).
pub(crate) fn cosine(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return -1.0;
    }
    let (mut dot, mut na, mut nb) = (0f32, 0f32, 0f32);
    for (x, y) in a.iter().zip(b) {
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    if na == 0.0 || nb == 0.0 {
        return -1.0;
    }
    dot / (na.sqrt() * nb.sqrt())
}

/// Runs the canary through a fresh engine, retrying up to `SETTLE_ATTEMPTS` times — the same bound,
/// for the same reason, as `embed_settling`: the first run(s) on a fresh session are allowed to be
/// wrong exactly that often. The batch is laid out at the engine's PRODUCTION shape when pinning is
/// on, so on MIGraphX the canary never compiles a shape of its own — and, as a side effect, it
/// absorbs the shape's expensive first run before any real batch pays for it.
pub(crate) fn canary_check(
    engine: &mut Bgem3DualEmbedding,
    limits: Limits,
    pin: bool,
) -> anyhow::Result<()> {
    let (texts, position) = if pin && limits.max_batch >= 2 {
        let (expanded, positions) =
            pin_shape(&[CANARY_TEXT.to_string()], limits.max_batch, ruler_text());
        (expanded, positions[0])
    } else {
        (vec![CANARY_TEXT.to_string()], 0)
    };

    let mut last_cosine = -1.0f32;
    for attempt in 1..=SETTLE_ATTEMPTS {
        match engine.embed(texts.clone(), Some(limits.max_batch)) {
            Ok(rows) => match rows.get(position) {
                Some((dense, _)) => {
                    let cos = cosine(dense, canary_reference());
                    if cos >= CANARY_MIN_COSINE {
                        tracing::info!("canary: engine verified against the reference (cosine {cos:.6}, run {attempt})");
                        return Ok(());
                    }
                    last_cosine = cos;
                    tracing::info!(
                        "canary: run {attempt} scored cosine {cos:.4} — a fresh session's first runs can be full-length garbage; re-running"
                    );
                }
                None => tracing::info!(
                    "canary: run {attempt} came back short ({} row(s), canary at {position}) — re-running",
                    rows.len()
                ),
            },
            Err(e) if attempt < SETTLE_ATTEMPTS => {
                tracing::info!("canary: run {attempt} was rejected ({e}) — re-running");
            }
            Err(e) => return Err(e).context("canary run failed outright"),
        }
    }
    anyhow::bail!(
        "canary cosine {last_cosine:.4} after {SETTLE_ATTEMPTS} run(s) (threshold {CANARY_MIN_COSINE}) — the engine's output does not match the reference"
    )
}

/// `load_dual` + the canary, healing a corrupt compiled-model cache: a persistent canary failure on
/// a cached program means the .mxr on disk is bad (defect 1 above), so the engine's cache slice is
/// wiped and ONE clean recompile gets its own canary. Still failing after that = the engine cannot
/// be trusted at all — fail the request; never serve unverified embeddings.
///
/// The whole of this — build, canary, wipe, rebuild, second canary — runs under the CALLER's cache-path
/// lease. The canary IS this engine's first kernel launch, which is the moment the MIGraphX EP actually
/// reads the path; a lease that ended with the build would let the other engine redirect it in between.
/// See `CachePathLease`.
pub(crate) fn load_validated_dual(
    state: &AppState,
    provider_hint: &str,
    limits: Limits,
    pin: bool,
    cache: &CachePathLease,
) -> anyhow::Result<Bgem3DualEmbedding> {
    let mut engine = load_dual(state, provider_hint, limits.max_length, cache)?;
    let Err(first_failure) = canary_check(&mut engine, limits, pin) else {
        return Ok(engine);
    };

    if cache.dir().is_empty() {
        // No compiled-model cache -> nothing to heal by wiping; the failure is the answer.
        return Err(first_failure.context("canary failed with no compiled-model cache configured"));
    }

    tracing::warn!(
        "canary failed ({first_failure:#}) — wiping `{}` and recompiling once: a crash mid-compile leaves a corrupt program that loads and stably produces garbage",
        cache.dir()
    );
    drop(engine);
    cache.wipe();
    let mut rebuilt = load_dual(state, provider_hint, limits.max_length, cache)?;
    canary_check(&mut rebuilt, limits, pin).context(
        "canary still failing after a clean recompile — refusing to serve garbage embeddings",
    )?;
    Ok(rebuilt)
}

/// Produces a NEW reference vector from the model this build actually loads, and writes it to `path`.
///
/// <b>Why this exists at all.</b> The generator that made the committed `canary-reference.f32le` was a
/// script in the monorepo this crate was carried out of, and it did not travel. That left the canary — the
/// one guard against a wrong-but-plausible vector — with an oracle nobody could reproduce: a deliberate
/// model change had nothing to regenerate it with, so the only available move was to weaken the check.
///
/// <b>What makes the output trustworthy.</b> It goes through `load_dual` and the production shape, not
/// through a bespoke path — the same loader, the same provider selection, the same pinning decision the
/// serving code makes. It deliberately does NOT go through `load_validated_dual`, which would check the new
/// engine against the OLD reference and refuse; that is the circularity this tool has to stand outside of.
///
/// <b>The number that decides whether to keep it.</b> The cosine against the CURRENT reference is computed
/// and reported before anything is written. Near 1.0 means the model did not really change and this file
/// did not need regenerating. Near 0.1 means either the model deliberately moved — a new export, a new
/// checkpoint — or this build is producing garbage, and only the operator knows which. Regenerating to
/// silence a failing canary is exactly the misuse the docstring on `canary_reference` forbids, and printing
/// the distance is what makes that misuse a decision rather than an accident.
///
/// <b>A byte-level diff is NORMAL and means nothing.</b> Measured 2026-08-19 on an R9700 through DirectML,
/// regenerating against the committed file: cosine <b>1.000000000</b>, and yet <b>1012 of 1024</b> elements
/// differ — max element delta 2.868e-07, mean 6.209e-08. That is float32 rounding, not a change: GPU
/// arithmetic is not bit-reproducible across runs, let alone across execution providers. It is the reason
/// the canary's threshold is a COSINE and not a comparison of bytes, and the reason `git diff` showing this
/// file as changed is not evidence of anything. Read the cosine, never the diff.
pub(crate) fn write_reference(
    state: &AppState,
    limits: Limits,
    path: &std::path::Path,
) -> anyhow::Result<()> {
    let provider = crate::provider::pin_provider(state, "");
    let pin = crate::inference::should_pin_shape(&state.config.pin_input_shape, &provider);
    // The same cache slice the serving path would claim for this shape — a reference produced against a
    // differently-keyed compiled program is a reference produced by a different program.
    let cache = CachePathLease::hold(
        &state.config.mxr_cache_base,
        crate::compile_cache::EMBED_CACHE_ENGINE,
        crate::compile_cache::CacheShape::new(limits.max_batch, limits.max_length),
    );

    tracing::info!(
        "canary reference: building the embed session on `{provider}` at ({}, {}), pinning {}",
        limits.max_batch,
        limits.max_length,
        if pin { "on" } else { "off" }
    );
    let mut engine = load_dual(state, "", limits.max_length, &cache)?;

    let (texts, position) = if pin && limits.max_batch >= 2 {
        let (expanded, positions) =
            pin_shape(&[CANARY_TEXT.to_string()], limits.max_batch, ruler_text());
        (expanded, positions[0])
    } else {
        (vec![CANARY_TEXT.to_string()], 0)
    };

    let rows = engine.embed(texts, Some(limits.max_batch))?;
    let (dense, _) = rows.get(position).ok_or_else(|| {
        anyhow::anyhow!(
            "the engine returned {} row(s); the canary sits at {position}",
            rows.len()
        )
    })?;

    let drift = cosine(dense, canary_reference());
    tracing::warn!(
        "canary reference: the new vector scores cosine {drift:.6} against the one committed here \
         ({} dimensions vs {}). Near 1.0 means nothing needed regenerating. Far from it means either the \
         model deliberately moved or THIS BUILD IS WRONG — and only you know which. Do not keep this file \
         to silence a failing canary.",
        dense.len(),
        canary_reference().len()
    );

    std::fs::write(path, encode_reference(dense))
        .map_err(|e| anyhow::anyhow!("could not write `{}`: {e}", path.display()))?;
    tracing::info!(
        "canary reference: wrote {} dimension(s) to {}",
        dense.len(),
        path.display()
    );
    Ok(())
}

#[cfg(test)]
mod tests {

    /// The corruption detector's arithmetic: identical vectors score 1, unrelated directions score
    /// low, and a dimension mismatch is a FAILED canary (-1), never a panic — the whole point is
    /// that garbage output must fail the check, whatever shape it arrives in.
    #[test]
    fn cosine_separates_identity_from_garbage_and_never_panics_on_shape() {
        let v = [0.6f32, 0.8, 0.0];
        assert!(
            (super::cosine(&v, &v) - 1.0).abs() < 1e-6,
            "a vector matches itself"
        );
        assert!(
            super::cosine(&[1.0, 0.0], &[0.0, 1.0]).abs() < 1e-6,
            "orthogonal scores ~0"
        );
        assert_eq!(
            super::cosine(&v, &[1.0, 0.0]),
            -1.0,
            "a dimension mismatch fails, not panics"
        );
        assert_eq!(
            super::cosine(&[0.0, 0.0], &[1.0, 0.0]),
            -1.0,
            "a zero vector fails, not NaNs"
        );
    }

    /// The generator and the reader are one pair, and a pair that disagrees produces a file which fails
    /// every healthy engine on the machine and reads as a hardware fault.
    #[test]
    fn what_the_generator_writes_is_what_the_canary_reads() {
        let vector: Vec<f32> = (0..1024).map(|i| (i as f32) * 0.001 - 0.5).collect();

        let decoded = super::decode_reference(&super::encode_reference(&vector));

        assert_eq!(
            decoded, vector,
            "little-endian f32, no header — the file's size IS the dimension"
        );
        assert_eq!(super::encode_reference(&vector).len(), 1024 * 4);
    }

    /// The committed file must survive its own round trip, which is the only check that the format the
    /// generator writes today is the format the file on disk was written in.
    #[test]
    fn the_committed_reference_round_trips_through_the_generators_format() {
        let stored = super::canary_reference();

        let round_tripped = super::decode_reference(&super::encode_reference(stored));

        assert_eq!(round_tripped, stored);
    }

    /// The embedded reference must be exactly one bge-m3 dense vector, finite and L2-normalized —
    /// a truncated or stale file would make the canary fail every healthy engine.
    #[test]
    fn canary_reference_is_one_finite_normalized_bgem3_vector() {
        let reference = super::canary_reference();
        assert_eq!(reference.len(), 1024, "bge-m3 dense dim");
        assert!(reference.iter().all(|v| v.is_finite()));
        let norm: f32 = reference.iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!(
            (norm - 1.0).abs() < 1e-3,
            "the sidecar serves normalized vectors, norm was {norm}"
        );
    }
}
