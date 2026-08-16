use std::sync::{Arc, Mutex};
use axum::body::Bytes;
use axum::extract::State;
use axum::Json;
use fastembed::Bgem3DualEmbedding;
use serde::Deserialize;
use crate::config::{DENSE_MODEL, DUAL_MODEL, RERANK_MODEL, SPARSE_MODEL};
use crate::engine_cache::{EngineSlot};
use crate::inference::lock_or_refuse;
use crate::provider::{PROVENANCE, compiled_providers, effective_provider};
use crate::state::AppState;
use crate::tokens::BGE_TOKENIZER;
use crate::wedge::{Patience};
use crate::wire::{HealthResponse, KIND_DENSE_SPARSE, KIND_RERANK, KIND_TOKENIZER_ONLY, LimitsWire, LoadedModels, ModelEntry, ModelNames, ModelsResponse, in_flight_now, join_error_text, tokenizer_available};

pub(crate) async fn health(State(state): State<Arc<AppState>>) -> Json<HealthResponse> {
    let requested = state
        .pinned_provider
        .get()
        .cloned()
        .unwrap_or_else(|| effective_provider(&state.config, ""));
    // try_lock, never lock: /health must not queue behind a multi-minute engine build — the same rule
    // `loaded_now` follows. A busy lock reads as "unknown yet", which is honest.
    let active = state.active_provider.try_lock().ok().and_then(|a| a.clone());
    let last_error = state.last_provider_error.try_lock().ok().and_then(|e| e.clone());
    let in_flight = in_flight_now(&state);
    let wedged = in_flight.iter().any(|held| held.wedged);
    // Read, never compute: the hashes are prewarmed on the blocking pool at startup. This endpoint is
    // a readiness probe, and a probe that SHA-256s every provider library beside the exe on its first
    // call reports nothing, slowly, from a reactor thread.
    let provenance = PROVENANCE.get();
    Json(HealthResponse {
        status: if wedged { "wedged" } else { "ok" },
        activity: state.activity.try_lock().map(|a| a.clone()).unwrap_or_else(|_| "busy".to_string()),
        provider: active.clone().unwrap_or_else(|| requested.clone()),
        requested_provider: requested,
        compiled_providers: compiled_providers(),
        provider_ready: active.is_some(),
        active_provider: active,
        last_provider_error: last_error,
        exe_sha256: provenance.map(|p| p.exe_sha256.clone()).unwrap_or_default(),
        runtime_manifest_sha256: provenance.map(|p| p.runtime_manifest_sha256.clone()).unwrap_or_default(),
        provenance_ready: provenance.is_some(),
        in_flight,
        wedged,
        loaded: LoadedModels {
            // Both heads are served by the ONE dual engine, so the two flags mirror it — the wire
            // shape stays as the C# host (SidecarEmbedder.UnloadAsync) asserts it.
            dense: loaded_now(&state.engines.embed),
            sparse: loaded_now(&state.engines.embed),
            rerank: loaded_now(&state.engines.rerank),
        },
        models: ModelNames { dense: DENSE_MODEL, sparse: SPARSE_MODEL, rerank: RERANK_MODEL },
        limits: LimitsWire {
            embed_max_length: state.config.embed_max_length,
            max_batch: state.config.max_batch,
            rerank_max_length: state.config.rerank_max_length,
            loaded_embed_max_length: state.loaded_embed_max_length.try_lock().ok().and_then(|g| *g),
            loaded_max_batch: state.loaded_max_batch.try_lock().ok().and_then(|g| *g),
            resident_embed_max_lengths: state.engines.embed.try_lock().map(|g| g.caps()).unwrap_or_default(),
            max_body_bytes: state.config.max_body_bytes,
            tokenize_max_texts: state.config.tokenize_max_texts,
        },
        adapter: state.adapter.clone(),
    })
}

/// What this build can embed with, rerank with, and count with — one read, answered without touching an
/// engine lock (`loaded_now` try_locks, exactly as `/health` does).
///
/// It exists so a consumer can validate a corpus recipe BEFORE starting a pass rather than discovering a
/// mismatch in the middle of one. Every fact here is read from what is loaded or configured; nothing is
/// a constant a second repository would have to keep in step.
pub(crate) async fn models(State(state): State<Arc<AppState>>) -> Json<ModelsResponse> {
    Json(ModelsResponse { models: models_now(&state) })
}

pub(crate) fn models_now(state: &AppState) -> Vec<ModelEntry> {
    let served = [
        ModelEntry {
            id: "bge-m3",
            name: DUAL_MODEL,
            kind: KIND_DENSE_SPARSE,
            dimension: state.loaded_embed_dimension.try_lock().ok().and_then(|d| *d),
            // What a pass would run at: the resident rung when there is one, else the configured
            // default. The same resolution the host already performs over /health, moved to the side
            // that owns the fact.
            max_sequence_length: Some(
                state
                    .loaded_embed_max_length
                    .try_lock()
                    .ok()
                    .and_then(|c| *c)
                    .unwrap_or(state.config.embed_max_length),
            ),
            tokenizer: Some(BGE_TOKENIZER),
            available: loaded_now(&state.engines.embed),
            tokenizer_available: tokenizer_available(state, Some(BGE_TOKENIZER)),
        },
        ModelEntry {
            id: "bge-reranker-v2-m3",
            name: RERANK_MODEL,
            kind: KIND_RERANK,
            // A cross-encoder returns scores, not vectors: there is no width to report, ever.
            dimension: None,
            max_sequence_length: Some(state.config.rerank_max_length),
            tokenizer: None,
            available: loaded_now(&state.engines.rerank),
            tokenizer_available: None,
        },
    ];

    // Then every registered tokenizer that no served model already claimed — which is precisely the
    // `tokenizer-only` set, derived rather than listed so it cannot drift from the registry.
    let claimed: Vec<&str> = served.iter().filter_map(|model| model.tokenizer).collect();
    let counting = state.tokenizers.entries.iter().filter(|entry| !claimed.contains(&entry.name)).map(|entry| {
        ModelEntry {
            id: entry.name,
            name: entry.name,
            kind: KIND_TOKENIZER_ONLY,
            dimension: None,
            max_sequence_length: None,
            tokenizer: Some(entry.name),
            // On a tokenizer-only row the two are the same fact, which is the consistency the split
            // buys: a reader never has to know which kind of row it is holding to read either flag.
            available: entry.tokenizer.is_some(),
            tokenizer_available: Some(entry.tokenizer.is_some()),
        }
    });

    served.into_iter().chain(counting).collect()
}

/// Non-blocking engine presence for /health: a busy lock means a load or an inference pass holds
/// the engine RIGHT NOW, so report presence instead of queueing the probe behind model work — a
/// hung first load (the ort load-dynamic version-mismatch deadlock) once froze /health forever
/// behind this lock. A poisoned lock (a panicked load) counts as "nothing loaded".
pub(crate) fn loaded_now<S: EngineSlot>(engine: &Mutex<S>) -> bool {
    match engine.try_lock() {
        Ok(guard) => guard.is_loaded(),
        Err(std::sync::TryLockError::WouldBlock) => true,
        Err(std::sync::TryLockError::Poisoned(_)) => false,
    }
}

/// A scoped `/unload` body (research/PLAN_gpu_search_arbitration.md): the host's budget-aware eviction
/// names exactly what must go — individual embed rungs and/or the reranker — so the rest stays warm.
/// An EMPTY body keeps the original contract: drop everything (the exclusive-LLM handover).
#[derive(Deserialize, Default, Debug, PartialEq)]
pub(crate) struct UnloadRequest {
    /// Sequence caps whose embed engines to drop. Empty = not specified.
    #[serde(default)]
    pub(crate) embed_max_lengths: Vec<usize>,
    /// Drop the reranker too. None = not specified.
    #[serde(default)]
    pub(crate) rerank: Option<bool>,
}

impl UnloadRequest {
    pub(crate) fn is_full_drain(&self) -> bool {
        self.embed_max_lengths.is_empty() && self.rerank.is_none()
    }
}

/// A malformed body drops NOTHING (and logs) — a typo'd partial request silently becoming a full
/// drain would evict engines the caller explicitly asked to keep. `None` = "do not touch anything".
pub(crate) fn parse_unload_request(body: &[u8]) -> Option<UnloadRequest> {
    if body.is_empty() {
        return Some(UnloadRequest::default());
    }
    match serde_json::from_slice(body) {
        Ok(req) => Some(req),
        Err(e) => {
            tracing::warn!("bad /unload body ({e}) — refusing to drop anything");
            None
        }
    }
}

/// Drops loaded ONNX engines so their VRAM is released. With no body: everything (called by the
/// host's GPU-lease coordinator before an exclusive local-LLM session takes the whole card). With a
/// JSON body: only the named rungs/reranker (the budget-aware eviction path). No /load counterpart
/// is needed: the next /embed or /rerank lazily re-creates the engines through the same first-use
/// path, on the provider already pinned in `active_provider`. Responds with the same shape as
/// /health so the caller can assert what is (still) loaded.
pub(crate) async fn unload(State(state): State<Arc<AppState>>, body: Bytes) -> Json<HealthResponse> {
    let Some(req) = parse_unload_request(&body) else {
        return health(State(state)).await;
    };

    // The ACQUISITION moved onto the blocking pool with the teardown, and it acquired a ceiling.
    // Both halves answer one incident class: /unload took the blocking engine mutex directly on a Tokio
    // WORKER thread, so the operator's only recovery tool — which also serves the host's GPU-lease
    // coordinator — queued on the very mutex a hung request was holding. With Tokio's default worker
    // count a handful of such calls starved the whole HTTP server, /health included, so the one endpoint
    // that could have explained the freeze went down with it. The same file had already solved exactly
    // this for /health (`loaded_now`); /unload was left behind.
    let worker = state.clone();
    let drained = tokio::task::spawn_blocking(move || drain_engines(&worker, &req))
        .await
        .unwrap_or_else(|e| {
            tracing::error!("unload task panicked: {}", join_error_text(e));
            Drained::default()
        });

    let resident: Vec<usize> = state.engines.embed.try_lock().map(|g| g.caps()).unwrap_or_default();
    tracing::info!(
        "unloaded engines (embed rung(s) {:?}, rerank: {}) — resident rung(s) now: {:?}",
        drained.embed_rungs, drained.rerank, resident
    );
    if !drained.refused.is_empty() {
        // Loud, because a PARTIAL handover read as a complete one is how an exclusive LLM ends up
        // sharing the card with a session that never let go. The body says so too: `loaded` still
        // reports the engine, and `in_flight` names what is holding it.
        tracing::warn!(
            "/unload could not take {:?} within {}s — those engines are STILL LOADED and this response says so \
             (`loaded`/`in_flight`). Whoever asked for the card must not assume it was released.",
            drained.refused,
            state.config.wedge.unload_wait.as_secs()
        );
    }
    health(State(state)).await
}

/// What one `/unload` actually moved.
#[derive(Default)]
pub(crate) struct Drained {
    pub(crate) embed_rungs: Vec<usize>,
    pub(crate) rerank: bool,
    /// Engines /unload could NOT take before its ceiling. Reported so a caller can never read a partial
    /// handover as a complete one.
    pub(crate) refused: Vec<&'static str>,
}

/// Takes the named engines out under their locks and drops them OUTSIDE those locks.
///
/// Runs entirely on the blocking pool (see `unload`): ort session teardown is not instant, and both the
/// wait and the drop would otherwise sit on a reactor thread. The drop-outside-the-lock shape is the
/// pre-existing correct design and is preserved deliberately — holding an engine mutex through a
/// teardown would block the very /health that reports the handover.
pub(crate) fn drain_engines(state: &AppState, req: &UnloadRequest) -> Drained {
    let policy = state.config.wedge;
    let patience = Patience::AtMost(policy.unload_wait);
    let mut drained = Drained::default();

    // EVERY resident rung goes on a full drain, not just the current one — a rung left behind would keep
    // holding the card the lease is handing to an exclusive LLM.
    if req.is_full_drain() || !req.embed_max_lengths.is_empty() {
        match lock_or_refuse(&state.engines.embed, &state.engines.embed_inflight, "embed", policy, patience) {
            Ok(mut guard) => {
                let taken: Vec<(usize, Bgem3DualEmbedding)> = if req.is_full_drain() {
                    guard.drain()
                } else {
                    req.embed_max_lengths.iter().filter_map(|&cap| guard.remove(cap).map(|e| (cap, e))).collect()
                };
                drop(guard);
                drained.embed_rungs = taken.iter().map(|(cap, _)| *cap).collect();
                drop(taken);
            }
            Err(e) => {
                tracing::warn!("/unload: {e:#}");
                drained.refused.push("embed");
            }
        }
    }

    if req.is_full_drain() || req.rerank == Some(true) {
        match lock_or_refuse(&state.engines.rerank, &state.engines.rerank_inflight, "rerank", policy, patience) {
            Ok(mut guard) => {
                let taken = guard.take();
                drop(guard);
                drained.rerank = taken.is_some();
                drop(taken);
            }
            Err(e) => {
                tracing::warn!("/unload: {e:#}");
                drained.refused.push("rerank");
            }
        }
    }

    drained
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;
    use std::time::{Duration, Instant};
    use axum::extract::State;
    use crate::testing::*;
    use crate::bookkeeping::{dense_dimension, record_embed_dimension};
    use crate::engine_cache::{RungCache};
    use crate::tokens::{BGE_TOKENIZER};
    use crate::wedge::{Phase, write_inflight};
    
    /// /unload hands the whole card to an exclusive local LLM. A rung left behind would keep holding VRAM
    /// while the host believes the sidecar released it — and the host asserts on `loaded: {false,…}`.
    #[test]
    fn unload_drains_every_resident_rung() {
        let mut cache = cache_of(2, &[(1024, 10), (256, 20)]);

        let drained = cache.drain();

        assert_eq!(drained.len(), 2, "both rungs come out");
        assert!(cache.caps().is_empty() && !loaded_now(&Mutex::new(cache)), "and /health reports nothing loaded");
    }

    /// /health must never queue behind the engine mutex: a lock held by a (possibly hung) model
    /// load or a running inference pass reports as "present" without blocking. The regression was
    /// a first load frozen inside ort's load-dynamic version check holding this lock forever, so
    /// /health hung with it — with the old blocking `lock()` the third assertion deadlocks.
    #[test]
    fn health_engine_flag_answers_instantly_while_the_lock_is_held() {
        let slot: Mutex<Option<u8>> = Mutex::new(None);
        assert!(!loaded_now(&slot), "empty and unlocked reports not loaded");

        *slot.lock().expect("fresh lock") = Some(1);
        assert!(loaded_now(&slot), "loaded and unlocked reports loaded");

        let held = slot.lock().expect("fresh lock");
        assert!(loaded_now(&slot), "held lock reports busy-as-present, without blocking");
        drop(held);

        // The same guarantee for the rung-keyed slots: the cache changed WHICH engine the guard hands
        // back, and must not have changed how long /health waits for it.
        let cache: Mutex<RungCache<u8>> = Mutex::new(RungCache::new(2));
        assert!(!loaded_now(&cache), "an empty cache reports not loaded");

        cache.lock().expect("fresh lock").insert(256, 1);
        assert!(loaded_now(&cache), "one resident rung is enough to report loaded");

        let held_cache = cache.lock().expect("fresh lock");
        assert!(loaded_now(&cache), "a cache held by a load or a pass reports busy-as-present");
        drop(held_cache);
    }

    /// The /unload body contract (research/PLAN_gpu_search_arbitration.md): an empty body is the
    /// original full drain; a scoped body names rungs and/or the reranker; a MALFORMED body drops
    /// nothing — a typo'd partial request must never silently become a full drain.
    #[test]
    fn unload_body_parses_empty_scoped_and_malformed_correctly() {
        let full = parse_unload_request(b"").expect("empty body is the legacy full drain");
        assert!(full.is_full_drain());

        let scoped = parse_unload_request(br#"{"embed_max_lengths":[1024],"rerank":true}"#)
            .expect("a well-formed scoped body parses");
        assert!(!scoped.is_full_drain());
        assert_eq!(scoped.embed_max_lengths, vec![1024]);
        assert_eq!(scoped.rerank, Some(true));

        let rerank_only = parse_unload_request(br#"{"rerank":true}"#).expect("rerank-only parses");
        assert!(!rerank_only.is_full_drain(), "naming the reranker alone must not drain the rungs");

        assert_eq!(parse_unload_request(b"{not json"), None, "malformed drops NOTHING");
    }

    // ---------- 24/7 reliability: no unbounded wait, no blind probe (2026-08-16) --------------
    /// /unload is the operator's ONLY recovery tool and the host's GPU-lease handover, so it has to
    /// ANSWER while an engine is held — it used to take the blocking mutex directly on a Tokio worker
    /// thread, so a handful of calls against a stuck engine starved every worker and took /health down
    /// with them. Two worker threads: one the handler can occupy, one for the timer that catches it.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn unload_answers_while_the_engine_is_held() {
        let state = app_state();
        let held = HeldEngine::hold(state.clone(), |s| &s.engines.embed);

        // The handler runs as its OWN task on purpose: a future that blocks inside `.lock()` never
        // returns to the executor, so a timeout wrapped straight around it could not fire either —
        // which IS the starvation under test, and would hang this test instead of failing it.
        let handler = tokio::spawn(async move { unload(State(state), Bytes::new()).await });
        let answered = tokio::time::timeout(Duration::from_secs(2), handler).await;

        // Release before asserting, whatever the verdict: a worker still blocked on the mutex would
        // keep the runtime from shutting down and turn a red test into a hung one.
        drop(held);
        let body = answered
            .expect("/unload must answer while the engine is held, never queue on it")
            .expect("the unload task must not panic");
        assert!(body.0.loaded.dense, "and it must answer HONESTLY: an engine it could not take is still loaded");
    }

    /// A thread STUCK inside the ORT/MIGraphX C++ call never panics, so the poison healing can never
    /// reach it: the mutex is not poisoned, it is simply never released. /health must be able to SAY
    /// so — before this, a wedged engine and a healthy idle one answered with the same `status`, which
    /// is exactly why a system-wide freeze was invisible from the outside.
    #[tokio::test]
    async fn health_reports_wedged_once_an_inference_exceeds_the_threshold() {
        let state = app_state();
        let idle = health(State(state.clone())).await.0;

        // A cold compile: MINUTES here are correct, and must keep reading as "ok". This is the case the
        // ceilings are chosen around, and the one a naive watchdog would have called a hang.
        write_inflight(
            &state.engines.embed_inflight,
            Some(stamped(Phase::Building, "embed: building and canary-checking the session", Duration::from_millis(50))),
        );
        let building = health(State(state.clone())).await.0;
        assert_eq!(building.status, "ok", "a build inside its ceiling is slow but alive");
        assert!(!building.wedged && !building.in_flight[0].wedged);

        // The same phase, past its ceiling.
        write_inflight(
            &state.engines.embed_inflight,
            Some(stamped(Phase::Building, "embed: building and canary-checking the session", Duration::from_secs(9))),
        );
        let wedged = health(State(state)).await.0;

        assert_ne!(wedged.status, idle.status, "a wedged engine must not answer /health exactly as an idle one");
        assert_eq!(wedged.status, "wedged");
        assert!(wedged.wedged, "the one boolean a host can route on");
        assert_eq!(wedged.in_flight[0].engine, "embed");
        assert_eq!(wedged.in_flight[0].phase, "building");
        assert!(wedged.in_flight[0].activity.contains("canary-checking"), "the operator sees WHAT is stuck");
        assert!(wedged.in_flight[0].elapsed_seconds >= 9, "and for how long: {:?}", wedged.in_flight[0].elapsed_seconds);
        assert!(idle.in_flight.is_empty(), "an idle sidecar reports nothing in flight");
    }

    /// /health is a readiness probe: it must do ZERO blocking work inline. The first call used to
    /// SHA-256 the executable AND every provider library beside it (cuDNN alone is often >500 MB) on
    /// a Tokio reactor thread — a probe that hashes gigabytes is a probe that reports nothing.
    #[tokio::test]
    async fn the_first_health_probe_never_hashes_the_binaries_inline() {
        let state = app_state();

        let started = Instant::now();
        let answer = health(State(state)).await.0;
        let elapsed = started.elapsed();

        assert!(
            elapsed < Duration::from_millis(50),
            "the first /health took {elapsed:?} — it hashed the executable and every library beside it on the probe path"
        );
        // Nothing prewarmed it in a test process, so the honest answer is "not yet", never a made-up hash.
        assert!(!answer.provenance_ready, "an unhashed provenance says so instead of pretending");
        assert!(answer.exe_sha256.is_empty(), "and leaves the field empty rather than inventing a value");
    }

    /// The cap is on `/health` because a limit a client cannot read is a limit it will guess at. The host
    /// batches under it (`SidecarClient.RequestByteBudget`) and had to learn the number by bisection.
    #[tokio::test]
    async fn health_reports_the_body_limit_it_enforces() {
        let mut config = config("");
        config.max_body_bytes = 4096;
        let state = app_state_with(config);

        let reported = health(State(state.clone())).await.limits.max_body_bytes;

        assert_eq!(reported, state.config.max_body_bytes, "what /health says is what the router enforces");
    }

    /// `bge` is the embedder's tokenizer AND a registered counting name. It must appear ONCE, on the model
    /// it belongs to — a duplicate row would offer a caller two ways to name one thing.
    #[test]
    fn a_tokenizer_claimed_by_a_model_is_not_also_listed_on_its_own() {
        let state = app_state_with_tokenizers(&[BGE_TOKENIZER, "qwen"]);

        let models = models_now(&state);

        assert_eq!(models.iter().filter(|entry| entry.id == BGE_TOKENIZER).count(), 0, "not a row of its own");
        assert_eq!(
            model_row(&models, "bge-m3").tokenizer,
            Some(BGE_TOKENIZER),
            "it is named where it is used, on the model it counts for"
        );
    }

    /// UNKNOWN is a value. A dimension nobody has measured yet must not read as `0`, and must not be
    /// guessed from a constant either — a caller sizing a vector collection from a guess sizes it wrong,
    /// and that failure does not surface until something tries to store into it.
    #[test]
    fn an_unmeasured_dimension_is_unknown_rather_than_zero_or_a_constant() {
        let state = app_state_with_tokenizers(&[BGE_TOKENIZER]);

        let cold = models_now(&state);
        assert_eq!(model_row(&cold, "bge-m3").dimension, None, "nothing has been embedded on this process");

        // A pass reports a real row, and only then does the field carry a number.
        record_embed_dimension(&state, dense_dimension(&[vec![0.0f32; 1024]]));

        let warm = models_now(&state);
        assert_eq!(model_row(&warm, "bge-m3").dimension, Some(1024), "measured from a row, not from a const");
    }

    /// "Engine cold, tokenizer ready" is the state a consumer is in while it validates a recipe BEFORE
    /// starting a pass — so the two facts cannot share one flag.
    ///
    /// Found by reading the real response rather than by a test: the log said `bge token counting enabled
    /// from …tokenizer.json` while `/models` reported the bge-m3 row `available: false`, which is true of
    /// the engine and says nothing about the tokenizer. One field, two meanings by row kind.
    #[test]
    fn a_loaded_tokenizer_is_visible_even_while_its_engine_is_cold() {
        let cache = model_cache_with_a_tokenizer("cold-engine");
        let mut config = config("");
        config.cache_dir = cache.clone();
        let state = app_state_with(config);

        let models = models_now(&state);
        let bge = model_row(&models, "bge-m3");

        assert!(!bge.available, "no engine has been built on this state");
        assert_eq!(bge.tokenizer_available, Some(true), "but the tokenizer counted fine, and that is readable");
        assert_eq!(
            model_row(&models, "bge-reranker-v2-m3").tokenizer_available,
            None,
            "a row that names no tokenizer reports null, not false"
        );

        std::fs::remove_dir_all(&cache).ok();
    }

    /// A registered name whose file is missing is still a ROW, marked unavailable. Hiding it would make a
    /// deployment problem look like a name that does not exist.
    #[test]
    fn a_tokenizer_with_no_file_is_listed_and_marked_unavailable() {
        let state = app_state_with_tokenizers(&[BGE_TOKENIZER, "qwen"]);

        let models = models_now(&state);

        assert!(!model_row(&models, "qwen").available, "the test rows carry no files");
    }

    /// model work it describes is the defect `/health` already had fixed (`loaded_now` try_locks).
    #[test]
    fn models_answers_while_an_engine_is_held() {
        let state = app_state_with_tokenizers(&[BGE_TOKENIZER]);
        let held = HeldEngine::hold(state.clone(), |s| &s.engines.embed);

        let models = models_now(&state);

        assert_eq!(models.len(), 2, "the full answer, not a degraded one");
        assert!(model_row(&models, "bge-m3").available, "a busy lock reads as resident, exactly as /health does");
        drop(held);
    }

    /// A counting tokenizer is its own KIND, not a model with missing fields. A consumer that cannot see
    /// the difference between "a model you can embed with" and "a tokenizer you can count with" will
    /// eventually ask this process to embed with the second one.
    #[test]
    fn models_reports_a_counting_tokenizer_as_its_own_kind_never_as_an_embedder() {
        let state = app_state_with_tokenizers(&[BGE_TOKENIZER, "qwen"]);

        let models = models_now(&state);
        let qwen = model_row(&models, "qwen");

        assert_eq!(qwen.kind, "tokenizer-only");
        assert_eq!(qwen.dimension, None, "there is nothing to embed with, so there is no width");
        assert_eq!(qwen.max_sequence_length, None, "and a counting tokenizer imposes no cap here");
        assert_eq!(qwen.tokenizer.expect("it IS a tokenizer"), "qwen", "and /tokenize takes exactly this id");
    }

    /// A cross-encoder returns scores, not vectors — so it never has a width, cold or warm.
    #[test]
    fn the_reranker_never_reports_a_dimension() {
        let state = app_state_with_tokenizers(&[BGE_TOKENIZER]);
        record_embed_dimension(&state, Some(1024));

        let models = models_now(&state);
        let rerank = model_row(&models, "bge-reranker-v2-m3");

        assert_eq!(rerank.kind, "rerank");
        assert_eq!(rerank.dimension, None, "an embedder's width must never leak onto the reranker's row");
        assert_eq!(rerank.tokenizer, None, "and this process registers no counter for it — null, not a guess");
    }
}
