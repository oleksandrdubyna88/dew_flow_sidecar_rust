use std::sync::{Mutex, OnceLock};
use std::time::Instant;
use fastembed::TextRerank;
use crate::bookkeeping::{dense_dimension, record_embed_dimension, record_embed_max_length, record_max_batch, remember_engine};
use crate::canary::{load_validated_dual};
use crate::config::{Config};
use crate::engine_cache::{EngineSlot};
use crate::provider::{effective_provider, load_rerank};
use crate::state::{AppState, Limits, positive_or};
use crate::tokens::{token_usage};
use crate::wedge::{EngineWedged, InFlight, InFlightStamp, Patience, Phase, WedgePolicy, inflight_now};
use crate::wire::{EmbedResponse, PassTimings, RerankResponse, SparseVec, TokenUsage};
use crate::compile_cache::{pass_log_message, CachePathLease, CompileWatch, EMBED_CACHE_ENGINE, RERANK_CACHE_ENGINE};

// ---------- blocking inference ----------

/// The batch a rerank call runs at: the request's value when it carries one, else the configured default.
/// Deliberately mirrors `Limits::resolve` for the embed path — a rerank that silently ignored the operator's
/// envelope is exactly the defect this exists to stop, and the two resolutions must not drift apart.
pub(crate) fn rerank_batch(config: &Config, requested: usize) -> usize {
    positive_or(requested, config.max_batch).max(1)
}

/// Whether this provider recompiles the graph for every distinct input shape. MIGraphX does — one
/// compile is ~2-4 minutes AND ~2.5 GB of on-disk cache per shape, measured on an R9700 — so a run
/// whose batches vary in length spends nearly all its time compiling. CUDA/DirectML/CPU take dynamic
/// shapes in stride and must not pay the padding overhead below.
pub(crate) fn should_pin_shape(setting: &str, provider: &str) -> bool {
    match setting.trim().to_lowercase().as_str() {
        "1" | "true" | "on" | "yes" => true,
        "0" | "false" | "off" | "no" => false,
        _ => provider == "migraphx",
    }
}

/// A sequence guaranteed to exceed any `max_length` we allow (8192 tokens max), so the tokenizer
/// truncates it to EXACTLY the cap. One of these in a batch makes `PaddingStrategy::BatchLongest`
/// pad the whole batch to the cap — which is how `pin_shape` gets a constant shape.
///
/// Built ONCE and shared: it is ~110 KB of constant text, and every pinned request used to allocate a
/// fresh copy and then clone it once per chunk — at `max_batch` 64 that is megabytes of pure
/// allocation per request for a value that never changes.
pub(crate) fn ruler_text() -> &'static str {
    static RULER: OnceLock<String> = OnceLock::new();
    RULER.get_or_init(|| "lorem ipsum dolor sit amet ".repeat(4096)).as_str()
}

/// How many runs of the SAME real batch a request gets before it fails. Measured behaviour is that
/// exactly ONE run after an engine build is bad, so the second settles it; the bound exists because
/// "exactly one" is an observation, not a guarantee, and a genuinely broken engine must fail the
/// request instead of spinning here.
pub(crate) const SETTLE_ATTEMPTS: usize = 3;

/// Runs the batch, re-running the SAME texts (bounded) when the session returns a SHORT batch or
/// rejects one.
///
/// The defect this absorbs: under the MIGraphX EP the FIRST `session.run` on a freshly built session
/// comes back short. Measured 2026-07-28 at `(64, 1024)`: dense returned 80 rows for a 128-row input
/// (64 from the second chunk + only 16 from the first) and the sparse head's first chunk arrived
/// shaped `[16, 1024, 1024]`, tripping the vendored shape guard. Every later run on the same session
/// is correct.
///
/// This RETRIES THE REAL BATCH rather than burning throwaway warm-up passes, and that choice is a
/// hard lesson: the warm-up variant cost one extra full-cap pass on EVERY engine build plus its own
/// retries, which pushed the first request of a pass to ~608s — past the host's 600s HTTP budget —
/// and the whole pass "completed" with 0 methods. Retrying the real batch costs nothing on a settled
/// session and exactly one extra run on a fresh one. Embedding is pure, so re-running the same texts
/// is safe.
///
/// `retry_short` is the same provider test as shape pinning (the quirk is MIGraphX's); other
/// providers keep the raw single-run semantics, short rows and all.
pub(crate) fn embed_settling<T>(
    what: &str,
    expected: usize,
    retry_short: bool,
    mut pass: impl FnMut() -> anyhow::Result<Vec<T>>,
) -> anyhow::Result<Vec<T>> {
    if !retry_short {
        return pass();
    }

    let mut short = 0usize;
    for attempt in 1..=SETTLE_ATTEMPTS {
        match pass() {
            Ok(rows) if rows.len() >= expected => return Ok(rows),
            Ok(rows) => {
                short = rows.len();
                tracing::info!(
                    "{what}: run {attempt} came back short ({short} of {expected} row(s)) — the first run on a freshly built session does this; re-running the same batch"
                );
            }
            Err(e) if attempt < SETTLE_ATTEMPTS => {
                tracing::info!("{what}: run {attempt} was rejected ({e}) — the same first-run defect; re-running the same batch");
            }
            Err(e) => return Err(e),
        }
    }
    anyhow::bail!(
        "{what}: the session kept returning short batches ({short} of {expected} row(s)) after {SETTLE_ATTEMPTS} run(s) — refusing to serve a partial result"
    )
}

/// Lays texts out so every chunk fastembed forms has the SAME shape `(max_batch, max_length)`:
/// a ruler sequence leads each chunk (pinning the padded length) and the tail is filled so the last
/// chunk is full (pinning the batch dimension). Returns the expanded list plus, for each original
/// text, its index inside it. Pure, so the layout contract is testable without a model.
///
/// Costs one wasted row per `max_batch - 1` real rows and computes every row at the full cap — far
/// cheaper than a per-batch recompile. Needs `max_batch >= 2`; with 1 there is no room for a ruler,
/// so the caller keeps the natural (unpinned) layout.
pub(crate) fn pin_shape(texts: &[String], max_batch: usize, ruler: &str) -> (Vec<String>, Vec<usize>) {
    let per_chunk = max_batch - 1;
    let mut expanded: Vec<String> = Vec::with_capacity(texts.len() + texts.len().div_ceil(per_chunk) * 2);
    let mut positions = Vec::with_capacity(texts.len());
    for chunk in texts.chunks(per_chunk) {
        expanded.push(ruler.to_string());
        for text in chunk {
            positions.push(expanded.len());
            expanded.push(text.clone());
        }
        while expanded.len() % max_batch != 0 {
            expanded.push(ruler.to_string());
        }
    }
    (expanded, positions)
}

/// Keeps only the rows `pin_shape` recorded as real, in their original order. Returns an error
/// rather than silently misaligning if the engine gave back fewer rows than the layout expects —
/// a shifted embedding is worse than a failed request.
pub(crate) fn unpin_rows<T>(mut rows: Vec<T>, positions: &[usize]) -> anyhow::Result<Vec<T>> {
    let needed = positions.iter().copied().max().map_or(0, |m| m + 1);
    anyhow::ensure!(
        rows.len() >= needed,
        "pinned embed returned {} row(s), need at least {needed} to unpin",
        rows.len()
    );
    let mut taken: Vec<Option<T>> = rows.drain(..).map(Some).collect();
    Ok(positions.iter().filter_map(|&i| taken[i].take()).collect())
}

/// The sequence cap a request actually runs at.
///
/// A `doc` request is an index pass stating its envelope, so it gets exactly what it asked for. A `query` is
/// a handful of words that fits ANY cap we build — and it arrives interleaved with those passes. Letting it
/// state a cap made it evict the loaded pair and rebuild both engines, which is far more expensive than the
/// query itself: a Fast pass ends on the LOW rung (it walks long→short), so the first search after every pass
/// asked for the ceiling and paid a full rebuild, and the next pass paid another one going back down.
///
/// So a query runs at whatever is already resident and never moves the cap. With nothing loaded it falls back
/// to what it asked for, which is the only moment its choice can matter. Measured on an R9700 at batch 64:
/// 1.6s at a resident 256 cap against 6.8s at 1024 — plus the two rebuilds that no longer happen.
///
/// Pure, so the rule is testable without an ONNX session behind it.
pub(crate) fn cap_for(kind: &str, requested: usize, loaded: Option<usize>) -> usize {
    match (kind, loaded) {
        ("query", Some(resident)) => resident,
        _ => requested,
    }
}

pub(crate) fn embed_blocking(
    state: &AppState,
    texts: Vec<String>,
    kind: &str,
    provider_hint: &str,
    limits: Limits,
    request_id: &str,
) -> anyhow::Result<EmbedResponse> {
    // Read the current cap under its own lock and drop it before record_embed_max_length takes it again.
    let resident = state.loaded_embed_max_length.lock().ok().and_then(|loaded| *loaded);
    let limits = Limits { max_length: cap_for(kind, limits.max_length, resident), ..limits };
    record_embed_max_length(state, limits.max_length);
    record_max_batch(state, limits.max_batch);

    // Counted here and nowhere else: AFTER cap_for has settled the effective cap (a `query` may not move
    // the loaded one) and BEFORE pin_shape can splice ruler rows into the batch, so the numbers line up
    // one-to-one with the texts the CALLER sent.
    let usage = token_usage(state, &texts, limits.max_length);
    let lost = usage.truncated.iter().filter(|&&t| t).count();
    if lost > 0 {
        tracing::warn!(
            "{lost} of {} text(s) exceed max_length {} and were truncated to a prefix — the tail of each \
             was embedded as though it did not exist",
            texts.len(), limits.max_length
        );
    }

    // Shape pinning has to happen BEFORE the engines see the texts, and the provider is only known
    // once one is loaded — so consult the pinned provider (or what this request would pick).
    let provider = state
        .pinned_provider
        .get()
        .cloned()
        .unwrap_or_else(|| effective_provider(&state.config, provider_hint));
    // The same provider test gates BOTH shape pinning and the settling retry: they are two halves of
    // working around one EP. Pinning stops the per-shape recompiles; the retry absorbs the short
    // first run a freshly built session returns.
    let retry_short = should_pin_shape(&state.config.pin_input_shape, &provider);
    if retry_short && limits.max_batch >= 2 {
        let (expanded, positions) = pin_shape(&texts, limits.max_batch, ruler_text());
        tracing::info!(
            "embed request: {} text(s), pinned to {} row(s) of ({}, {}) for {provider}",
            texts.len(), expanded.len(), limits.max_batch, limits.max_length
        );
        let padded = embed_natural(state, expanded, provider_hint, limits, retry_short, request_id)?;
        let dense = unpin_rows(padded.dense, &positions)?;
        return Ok(EmbedResponse {
            // Re-measured from the rows that actually leave, not carried over from the padded batch:
            // the width is the same either way, and reporting it from anything but the returned vectors
            // is how a field starts describing something other than what it names.
            dimension: dense_dimension(&dense),
            dense,
            sparse: unpin_rows(padded.sparse, &positions)?,
            usage,
            request_id: padded.request_id,
            timings: padded.timings,
        });
    }

    Ok(EmbedResponse { usage, ..embed_natural(state, texts, provider_hint, limits, retry_short, request_id)? })
}

/// The unpinned path: hand the texts to fastembed as they are and let `BatchLongest` decide the
/// padded length of each batch.
///
/// `retry_short` re-runs the SAME batch when a freshly built session returns a short one — see
/// `embed_settling` for the measurement and for why a throwaway warm-up was the wrong tool. Gated on
/// the same provider test as shape pinning, because the short-first-run defect is MIGraphX's; the
/// other providers keep single-run semantics.
pub(crate) fn embed_natural(
    state: &AppState,
    texts: Vec<String>,
    provider_hint: &str,
    limits: Limits,
    retry_short: bool,
    request_id: &str,
) -> anyhow::Result<EmbedResponse> {
    let batch = Some(limits.max_batch);

    // ONE forward pass over the official FP32 model: the export returns both heads per run
    // (`sentence_embedding` + `token_embeddings`), so the dense/sparse split lives in the
    // post-processing, not in separate sessions — see Bgem3DualEmbedding and
    // research/PLAN_bge_sidecar_unified_session.md. Still deliberately NOT the INT8-quantized
    // all-in-one Bgem3Embedding — retrieval quality over speed (locked decision).
    set_activity(state, "embed: waiting for the engine");
    // Queue wait is measured around the MUTEX only — another request holding the engine is the
    // caller's infrastructure wait, and it must never blend into the inference span below.
    let waited = Instant::now();
    let mut guard = lock_or_refuse(
        &state.engines.embed,
        &state.engines.embed_inflight,
        "embed",
        state.config.wedge,
        Patience::UntilTheHolderIsWedged,
    )?;
    let queue_wait_ms = waited.elapsed().as_millis() as u64;
    // From here the engine is OURS, and the stamp is what makes a wedge UNDER it observable — /health
    // reads it without ever touching the engine lock. It clears itself on every exit, `?` included.
    let stamp = InFlightStamp::hold(state, &state.engines.embed_inflight);
    let mut session_build_ms = 0u64;
    // Claimed ONLY when this request builds, and then held through the first pass below: the MIGraphX EP
    // reads the compiled-model cache path at the first kernel launch, not at session build, so a claim
    // that ended with the build would leave the rerank engine free to redirect it in between. A resident
    // engine claims nothing — serializing steady-state passes across the two engine types would cost far
    // more than the minutes this costs once, while one of them is cold. See `CachePathLease`.
    let cache_claim = if guard.get_mut(limits.max_length).is_none() {
        stamp.enter(Phase::Building, "embed: building and canary-checking the session (a first-ever shape compiles for minutes; cached shapes load in seconds)");
        let building = Instant::now();
        let cache = CachePathLease::hold(&state.config.mxr_cache_base, EMBED_CACHE_ENGINE);
        let built = load_validated_dual(state, provider_hint, limits, retry_short, &cache)?;
        remember_engine(&mut guard, "embed", limits.max_length, built);
        session_build_ms = building.elapsed().as_millis() as u64;
        Some(cache)
    } else {
        None
    };
    stamp.enter(Phase::Running, format!("embed: embedding {} row(s)", texts.len()));
    // The duration below includes any settling re-runs — that is honest: it is what the caller waited.
    let (compiles, pass) = (CompileWatch::start(&state.config.mxr_cache_base, EMBED_CACHE_ENGINE), Instant::now());
    // INVARIANT: this rung is resident. Either it already was, or the block above built it and
    // `remember_engine` filed it — and `guard` is the only handle that can evict, so nothing could have
    // taken it between those lines. `RungCache` capacity is `max(1)`, so a fresh insert cannot evict
    // itself either. Separate the insert from this `get_mut` and the `expect` becomes a live panic on
    // the request path; keep them under one guard and it cannot fire.
    let engine = guard.get_mut(limits.max_length).expect("just loaded");
    // Rows are (dense, sparse) ZIPPED per text, so the settling retry polices one length and a short
    // first run can never shorten one head without the other.
    let rows = embed_settling("embed", texts.len(), retry_short, || engine.embed(texts.clone(), batch))?;
    let inference = pass.elapsed();
    // The first kernel launch has happened, so the EP has read the cache path and compiled or loaded
    // against it. Everything below this line is arithmetic — release the claim rather than holding it
    // across the unzip and the response.
    drop(cache_claim);
    let compile_cache_grew_mb = compiles.grew_mb();
    tracing::info!("{}", pass_log_message(
        request_id,
        &format!("embedded {} row(s), dense+sparse in one pass", rows.len()),
        inference.as_secs_f32(),
        compile_cache_grew_mb,
    ));

    let (dense, sparse): (Vec<_>, Vec<_>) = rows.into_iter().unzip();
    // The width comes from a row this pass actually produced. Recorded here — the one place a real
    // forward pass has happened — so /models can state it instead of a caller having to know it.
    let dimension = dense_dimension(&dense);
    record_embed_dimension(state, dimension);
    // No usage here on purpose: this function also runs over the PINNED batch, whose ruler rows are not
    // the caller's texts. embed_blocking owns the accounting and stamps it on the way out.
    Ok(EmbedResponse {
        dense,
        dimension,
        sparse: sparse
            .into_iter()
            .map(|s: fastembed::SparseEmbedding| SparseVec {
                indices: s.indices.iter().map(|&i| i as u32).collect(),
                values: s.values,
            })
            .collect(),
        usage: TokenUsage::default(),
        request_id: request_id.to_string(),
        timings: PassTimings {
            queue_wait_ms,
            session_build_ms,
            inference_ms: inference.as_millis() as u64,
            compile_cache_grew_mb,
        },
    })
}

pub(crate) fn rerank_blocking(
    state: &AppState,
    query: String,
    documents: Vec<String>,
    provider_hint: &str,
    max_batch: usize,
    request_id: &str,
) -> anyhow::Result<RerankResponse> {
    set_activity(state, "rerank: waiting for the engine");
    // Queue wait around the MUTEX only, exactly as the embed path measures it.
    let waited = Instant::now();
    let mut guard = lock_or_refuse(
        &state.engines.rerank,
        &state.engines.rerank_inflight,
        "rerank",
        state.config.wedge,
        Patience::UntilTheHolderIsWedged,
    )?;
    let queue_wait_ms = waited.elapsed().as_millis() as u64;
    // Its own stamp, not the embed one: the two engines have separate mutexes and can be in flight at
    // the same time, so a shared record would let one overwrite the other's wedge.
    let stamp = InFlightStamp::hold(state, &state.engines.rerank_inflight);
    let mut session_build_ms = 0u64;
    // The embed path's claim, mirrored — same reason, same lifetime. It is dropped when this function
    // returns, which is after `score_documents` has run the first pass through BOTH the pinned and the
    // natural branch below; the embed engine's build cannot redirect the cache path in between.
    let _cache_claim = if guard.is_none() {
        stamp.enter(Phase::Building, "rerank: building the session (a first-ever shape compiles for minutes; cached shapes load in seconds)");
        let building = Instant::now();
        let cache = CachePathLease::hold(&state.config.mxr_cache_base, RERANK_CACHE_ENGINE);
        *guard = Some(load_rerank(state, provider_hint, &cache)?);
        session_build_ms = building.elapsed().as_millis() as u64;
        Some(cache)
    } else {
        None
    };
    stamp.enter(Phase::Running, format!("rerank: scoring {} document(s)", documents.len()));

    // Shape pinning, exactly as the embed path does it and for the same provider: fastembed forms
    // (query, document) pairs and pads each chunk to ITS longest member, so under MIGraphX every distinct
    // (batch, seq) pair of a live query stream compiled its own program — measured 92–162 s and +2.19 GB
    // of cache apiece (2026-07-30). A ruler document leading every chunk pins the padded length to the
    // cap, the tail fill pins the batch dimension: ONE shape, one compile, cached forever.
    let provider = state
        .pinned_provider
        .get()
        .cloned()
        .unwrap_or_else(|| effective_provider(&state.config, provider_hint));
    if should_pin_shape(&state.config.pin_input_shape, &provider) && max_batch >= 2 {
        let (expanded, positions) = pin_shape(&documents, max_batch, ruler_text());
        tracing::info!(
            "rerank request: {} document(s), pinned to {} row(s) of ({}, {}) for {provider}",
            documents.len(), expanded.len(), max_batch, state.config.rerank_max_length
        );
        let (padded, pass) = score_documents(state, &mut guard, &query, &expanded, max_batch, request_id)?;
        return Ok(RerankResponse {
            scores: unpin_rows(padded, &positions)?,
            request_id: request_id.to_string(),
            timings: PassTimings { queue_wait_ms, session_build_ms, ..pass },
        });
    }

    let (scores, pass) = score_documents(state, &mut guard, &query, &documents, max_batch, request_id)?;
    Ok(RerankResponse {
        scores,
        request_id: request_id.to_string(),
        timings: PassTimings { queue_wait_ms, session_build_ms, ..pass },
    })
}

/// One scoring pass over `documents`, returning sigmoid scores ALIGNED with the input order plus the
/// pass's own timing spans (inference and compile; queue and build belong to the caller, which is why
/// they come back zeroed here). Shared by the pinned and natural paths of `rerank_blocking`.
pub(crate) fn score_documents(
    state: &AppState,
    guard: &mut std::sync::MutexGuard<'_, Option<TextRerank>>,
    query: &str,
    documents: &[String],
    max_batch: usize,
    request_id: &str,
) -> anyhow::Result<(Vec<f32>, PassTimings)> {
    let count = documents.len();
    let (compiles, pass) = (CompileWatch::start(&state.config.mxr_cache_base, RERANK_CACHE_ENGINE), std::time::Instant::now());
    // query.to_string(): fastembed's `rerank` shares one generic across the query and the document slice,
    // so an owned query is what lets `&[String]` documents satisfy it.
    let results = guard
        .as_mut()
        // Same invariant as the embed path: the caller built into this slot under the very guard it
        // handed us, and nothing else can empty an `Option<TextRerank>` while it is held.
        .expect("just loaded")
        .rerank(query.to_string(), documents, false, Some(max_batch))?;
    let inference = pass.elapsed();
    let compile_cache_grew_mb = compiles.grew_mb();
    tracing::info!("{}", pass_log_message(
        request_id,
        &format!("rerank: scored {count} document(s)"),
        inference.as_secs_f32(),
        compile_cache_grew_mb,
    ));
    let timings = PassTimings {
        queue_wait_ms: 0,
        session_build_ms: 0,
        inference_ms: inference.as_millis() as u64,
        compile_cache_grew_mb,
    };
    Ok((aligned_scores(count, results.into_iter().map(|r| (r.index, r.score))), timings))
}

/// Scores raised back into the DOCUMENT order the HTTP contract promises: `rerank()` returns results
/// sorted by score, while the C# `CrossEncoderReranker` pairs by position. Sigmoid-normalized to 0..1
/// for parity with the retired Python sidecar; an out-of-range index is dropped rather than trusted.
pub(crate) fn aligned_scores(count: usize, results: impl IntoIterator<Item = (usize, f32)>) -> Vec<f32> {
    let mut scores = vec![0f32; count];
    for (index, raw) in results {
        if index < count {
            scores[index] = sigmoid(raw);
        }
    }
    scores
}

/// Sets the "what am I doing right now" label surfaced by /health and mirrored to the log by the
/// callers that pass a phase worth announcing.
pub(crate) fn set_activity(state: &AppState, activity: impl Into<String>) {
    if let Ok(mut slot) = state.activity.lock() {
        *slot = activity.into();
    }
}

/// Acquires an engine slot with a CEILING, healing a poisoned mutex and refusing a wedged one.
///
/// Three failures meet here, and they need three different answers:
///
/// 1. **Poisoned** — a panic inside a model load (fastembed/ort) unwound while the guard was held. The
///    old `map_err(_ -> "engine poisoned")` failed EVERY later request until a process restart: a live
///    Fast pass ground through thousands of methods answering "sparse engine poisoned", Succeeded=0.
///    One panic must cost one request, so clear the poison, drop whatever half-built state it left, and
///    let the caller reload. Dropping is the conservative choice — the poison says nothing about WHICH
///    engine the panic touched, nor whether the session itself is broken — but it means a panic in
///    POST-PROCESSING costs a ~60 s rebuild per request, so the inference paths must return errors
///    rather than panic (see the shape guard in vendor-fastembed/src/sparse_text_embedding/impl.rs).
/// 2. **Held by a healthy holder** — a first-ever shape compile is minutes of CORRECT slowness. Wait.
/// 3. **Held by a WEDGED holder** — a thread stuck inside the ONNX Runtime C++ call. It never panics,
///    so case 1 can never see it, and the mutex is simply never released. Before this, every caller
///    queued on `.lock()` forever. Refuse, with the holder's activity and elapsed time in the message.
///
/// `Patience` decides how case 2 ends: the inference path waits as long as the holder is alive, /unload
/// waits a bounded time. A hold that nothing STAMPED falls back to `running_after` under either — the
/// missing ceiling is the defect, so the fallback has to exist even when the stamp does not.
pub(crate) fn lock_or_refuse<'a, S: EngineSlot>(
    engine: &'a Mutex<S>,
    inflight: &Mutex<Option<InFlight>>,
    what: &str,
    policy: WedgePolicy,
    patience: Patience,
) -> anyhow::Result<std::sync::MutexGuard<'a, S>> {
    let waiting_since = Instant::now();
    loop {
        match engine.try_lock() {
            Ok(guard) => return Ok(guard),
            Err(std::sync::TryLockError::Poisoned(poisoned)) => {
                engine.clear_poison();
                let mut guard = poisoned.into_inner();
                guard.discard_all();
                tracing::warn!("{what} engine mutex was poisoned by a panic while it was held — cleared it, rebuilding the engine (~1 min); if this repeats every request, something in the embed path is panicking rather than erroring");
                return Ok(guard);
            }
            Err(std::sync::TryLockError::WouldBlock) => {}
        }

        let holder = inflight_now(inflight);
        let (activity, held_for) = holder
            .as_ref()
            .map_or_else(|| (String::new(), waiting_since.elapsed()), |h| (h.label.clone(), h.since.elapsed()));
        let ceiling = holder.as_ref().map_or(policy.running_after, |h| policy.ceiling(h.phase));
        if held_for >= ceiling {
            anyhow::bail!(EngineWedged { what: what.to_string(), activity, elapsed: held_for, wedged: true });
        }
        if let Patience::AtMost(limit) = patience {
            let waited = waiting_since.elapsed();
            if waited >= limit {
                anyhow::bail!(EngineWedged { what: what.to_string(), activity, elapsed: waited, wedged: false });
            }
        }
        std::thread::sleep(policy.poll);
    }
}

pub(crate) fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::*;

    /// A search query runs at whatever cap is already resident and never moves it. Before this rule, a Fast
    /// pass ended on the low rung and the first search after it asked for the ceiling — evicting both engines
    /// and rebuilding them, then the next pass rebuilt them back. An index (`doc`) request still states its
    /// own envelope: that is how the operator's setting reaches the card at all.
    #[test]
    fn a_query_runs_at_the_resident_cap_while_a_doc_states_its_own() {
        assert_eq!(cap_for("query", 1024, Some(256)), 256, "a query must not evict the pair a pass is using");
        assert_eq!(cap_for("query", 256, Some(1024)), 1024, "and must not drag the cap the other way either");
        assert_eq!(cap_for("query", 1024, None), 1024, "with nothing loaded there is nothing to preserve");
        assert_eq!(cap_for("doc", 1024, Some(256)), 1024, "an index pass still sets the envelope it asked for");
        assert_eq!(cap_for("", 512, Some(256)), 512, "an unset kind is treated as a doc, not as a query");
    }

    /// A rerank runs at the batch the REQUEST carries, falling back to the configured default only when the
    /// request does not set one. The handler used to read `config.max_batch` unconditionally, and because the
    /// AppHost passes MAX_BATCH as an empty string that default is 4 — so an operator envelope of 64 was
    /// silently ignored and 50 candidates were scored in 13 forward passes instead of one.
    #[test]
    fn rerank_uses_the_requests_batch_and_falls_back_to_the_configured_default() {
        assert_eq!(rerank_batch(&config(""), 64), 64);
        assert_eq!(rerank_batch(&config(""), 0), 4, "0 means 'not set' — keep the sidecar's own default");
        assert_eq!(rerank_batch(&config(""), 1), 1);
    }

    /// The HTTP contract is scores in DOCUMENT order; fastembed returns them sorted by score. A wrong
    /// alignment here would silently attach one document's relevance to another — the C# side pairs by
    /// position and would never notice.
    #[test]
    fn rerank_scores_come_back_in_document_order_not_score_order() {
        let scores = aligned_scores(3, [(2usize, 0.0f32), (0, 4.0), (1, -4.0)]);

        assert!(scores[0] > 0.98, "doc 0 got the high raw score, sigmoid-normalized");
        assert!(scores[1] < 0.02, "doc 1 got the low one");
        assert!((scores[2] - 0.5).abs() < 1e-6, "doc 2's raw 0 sits at the sigmoid midpoint");
    }

    /// An index past the document count is dropped, not trusted: writing through it would panic or, worse,
    /// score a ruler row from the pinned layout as if it were a real document.
    #[test]
    fn rerank_scores_ignore_an_out_of_range_index() {
        let scores = aligned_scores(2, [(0usize, 1.0f32), (5, 9.9)]);

        assert_eq!(scores.len(), 2);
        assert!((scores[1] - 0.0).abs() < 1e-6, "the out-of-range result left doc 1 unscored");
    }

    /// Only MIGraphX recompiles per input shape, so only it pays the padding overhead by default;
    /// an explicit setting wins either way (a operator can force it off, or on for a measurement).
    #[test]
    fn shape_pinning_defaults_to_migraphx_only_and_obeys_an_explicit_setting() {
        assert!(should_pin_shape("auto", "migraphx"));
        assert!(!should_pin_shape("auto", "dml"));
        assert!(!should_pin_shape("auto", "cuda"));
        assert!(!should_pin_shape("auto", "cpu"));
        assert!(should_pin_shape("1", "dml"));
        assert!(!should_pin_shape("0", "migraphx"));
    }

    /// EVERY chunk fastembed forms must come out the same shape, or MIGraphX recompiles — which was
    /// the whole defect: a run spent ~2-4 min compiling per batch and wrote ~2.5 GB of cache each
    /// time. So: the layout is a whole number of full batches, every batch carries a ruler (making
    /// BatchLongest pad it to the cap), and the recorded positions still address the real texts.
    #[test]
    fn pinned_layout_is_whole_batches_that_each_carry_a_ruler() {
        let ruler = "RULER";
        // 5 texts at batch 4 => 2 chunks of 4 (ruler + 3 texts, then ruler + 2 texts + 1 filler)
        let texts: Vec<String> = (0..5).map(|i| format!("text{i}")).collect();
        let (expanded, positions) = pin_shape(&texts, 4, ruler);

        assert_eq!(expanded.len() % 4, 0, "layout must be whole batches: {expanded:?}");
        assert_eq!(expanded.len(), 8);
        for (n, batch) in expanded.chunks(4).enumerate() {
            assert!(batch.contains(&ruler.to_string()), "batch {n} has no ruler: {batch:?}");
        }
        assert_eq!(positions.len(), texts.len(), "every text is addressable");
        for (text, &at) in texts.iter().zip(&positions) {
            assert_eq!(&expanded[at], text);
        }
    }

    /// Unpinning must return exactly the caller's texts, in the caller's order — a row shifted by
    /// the padding would hand back another method's embedding, which no test downstream would catch.
    #[test]
    fn unpinning_recovers_the_original_rows_in_order() {
        let texts: Vec<String> = (0..7).map(|i| format!("text{i}")).collect();
        let (expanded, positions) = pin_shape(&texts, 3, "RULER");

        // Stand in for the engine: it returns one row per input row, in input order.
        let rows: Vec<String> = expanded.iter().map(|t| format!("emb({t})")).collect();
        let unpinned = unpin_rows(rows, &positions).expect("full result unpins");

        let expected: Vec<String> = texts.iter().map(|t| format!("emb({t})")).collect();
        assert_eq!(unpinned, expected);

        // A short result must fail loudly rather than misalign.
        assert!(unpin_rows(vec!["only-one".to_string()], &positions).is_err());
    }

    /// The common case — a settled session — must cost exactly ONE run. This is the regression that
    /// retired the throwaway warm-up: it charged an extra full-cap pass on EVERY engine build, which
    /// pushed a pass's first request to ~608s, past the host's 600s HTTP budget, and the pass
    /// "completed" with 0 methods embedded.
    #[test]
    fn a_settled_session_costs_exactly_one_run() {
        let mut calls = 0;
        let out = embed_settling("dense", 64, true, || {
            calls += 1;
            Ok(vec![0u8; 64])
        })
        .expect("full batch");

        assert_eq!((calls, out.len()), (1, 64));
    }

    /// The measured defect: the first run on a freshly built session returns 80 of 128 rows. The SAME
    /// batch must be re-run — serving the short result would silently drop the missing methods.
    #[test]
    fn a_short_first_run_is_rerun_with_the_same_batch() {
        let mut calls = 0;
        let out = embed_settling("dense", 128, true, || {
            calls += 1;
            Ok(vec![0u8; if calls == 1 { 80 } else { 128 }])
        })
        .expect("the second run settles");

        assert_eq!((calls, out.len()), (2, 128));
    }

    /// The sparse head never returns a short batch — the vendored shape guard rejects it — so a
    /// rejected first run is the SAME defect, not a dead engine.
    #[test]
    fn a_rejected_first_run_is_rerun_like_a_short_one() {
        let mut calls = 0;
        let out = embed_settling("sparse", 64, true, || {
            calls += 1;
            if calls == 1 {
                anyhow::bail!("came back shaped [16, 1024, 1024]")
            }
            Ok(vec![0u8; 64])
        })
        .expect("the second run settles");

        assert_eq!((calls, out.len()), (2, 64));
    }

    /// A session that never settles must FAIL the request after the bound — returning a partial
    /// result would misalign every embedding behind the missing rows.
    #[test]
    fn a_session_that_never_settles_fails_instead_of_serving_a_partial_result() {
        let mut calls = 0;
        let err = embed_settling::<u8>("dense", 64, true, || {
            calls += 1;
            Ok(vec![0u8; 16])
        })
        .expect_err("short forever");

        assert_eq!(calls, SETTLE_ATTEMPTS);
        assert!(err.to_string().contains("16 of 64"), "the error names the shortfall: {err}");
    }

    /// Providers without the quirk (CUDA/DirectML/CPU) keep raw single-run semantics: no retries, no
    /// short-row policing, and a genuine error propagates immediately.
    #[test]
    fn a_provider_without_the_quirk_runs_exactly_once() {
        let mut calls = 0;
        let out = embed_settling("dense", 64, false, || {
            calls += 1;
            Ok(vec![0u8; 16])
        })
        .expect("passed through untouched");
        assert_eq!((calls, out.len()), (1, 16));

        let mut failing_calls = 0;
        let err = embed_settling::<u8>("dense", 64, false, || {
            failing_calls += 1;
            anyhow::bail!("genuine failure")
        })
        .expect_err("no retry masks it");
        assert_eq!(failing_calls, 1);
        assert!(err.to_string().contains("genuine failure"));
    }

    // The default is the "we do not know" state, and it must not read as "nothing was truncated":
    // token_accounting stays false and both vectors stay empty, so a host that checks the flag cannot
    // mistake an absent measurement for a clean one.
    // The configured batch is a FALLBACK, not a report. Every /embed carries the operator's own batch, so
    // a health field that echoed the config read as the running value — which is how "batch 126" and
    // "max_batch 4" and an actual 64 coexisted in one answer, none of them describing the same thing.
    #[test]
    fn a_requests_batch_overrides_the_configured_default() {
        assert_eq!(rerank_batch(&config("dml"), 64), 64, "the request wins");
        assert_eq!(rerank_batch(&config("dml"), 0), config("dml").max_batch, "0 falls back to the config");
    }

    /// ~110 KB of constant text, cloned once per chunk of every pinned request. One allocation, shared.
    #[test]
    fn the_ruler_is_allocated_once_and_shared() {
        assert!(std::ptr::eq(ruler_text(), ruler_text()), "the same buffer, not a fresh copy per request");
        assert!(ruler_text().len() > 100_000, "still long enough to truncate to any cap we allow");
    }

    /// is what lets the pinned path re-measure from the returned rows rather than carrying a number over.
    #[test]
    fn unpinning_leaves_the_width_untouched() {
        let padded = vec![vec![0.0f32; 1024], vec![1.0f32; 1024], vec![2.0f32; 1024]];
        let wide = dense_dimension(&padded);

        let real = unpin_rows(padded, &[1]).expect("row 1 was the caller's");

        assert_eq!(real.len(), 1, "one row of the caller's survives");
        assert_eq!(dense_dimension(&real), wide, "and it is exactly as wide as what the engine returned");
    }
}
