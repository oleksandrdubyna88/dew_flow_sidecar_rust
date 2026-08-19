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

/// The exact string the stored reference vector was computed from. Change it and every engine fails
/// its canary, because the reference cannot be recomputed here: the generator that produced
/// `canary-reference.f32le` stayed in the monorepo this crate was carried out of, and replacing it is
/// open work (`todo/PLAN_sidecar_product.md`, phase 2).
pub(crate) const CANARY_TEXT: &str =
    "A canary sentence for the bge-m3 engine self-check: the quick brown fox jumps over the lazy dog 0123456789.";

/// Deliberately loose: EP-to-EP arithmetic differences sit near 0.9999, the observed garbage at
/// 0.13. This is a corruption detector, not a numerics test — the parity harness owns exactness.
pub(crate) const CANARY_MIN_COSINE: f32 = 0.99;

/// The dense embedding of `CANARY_TEXT`, captured 2026-07-31 from the unified build that passed the
/// parity gate bit-exact at both caps. Regenerate only when the MODEL deliberately changes — never to
/// green a failing canary; and see `CANARY_TEXT` for why regenerating is not currently possible from
/// this repository alone.
pub(crate) fn canary_reference() -> &'static [f32] {
    static REFERENCE: OnceLock<Vec<f32>> = OnceLock::new();
    REFERENCE.get_or_init(|| {
        include_bytes!("canary-reference.f32le")
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect()
    })
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
