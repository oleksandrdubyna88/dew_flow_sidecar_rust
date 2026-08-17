use axum::http::StatusCode;
use axum::Json;
use serde::{Deserialize, Serialize};
use crate::adapters;
use crate::state::{AppState};
use crate::wedge::{EngineWedged, InFlight, WedgePolicy, inflight_now};

// ---------- wire types ----------

#[derive(Deserialize)]
pub(crate) struct EmbedRequest {
    pub(crate) texts: Vec<String>,
    /// "doc" | "query". BGE-M3 is symmetric, so both embed identically — but a QUERY is also never allowed to
    /// move the loaded sequence cap (see `cap_for`), because it arrives interleaved with index passes.
    #[serde(default)]
    pub(crate) kind: String,
    /// Optional provider hint forwarded from the operator's DB setting (used only before first load).
    #[serde(default)]
    pub(crate) provider: String,
    /// Optional per-request token cap (operator setting). 0/absent = the configured default. Changing it
    /// evicts and rebuilds the engines, so the operator sees the new memory envelope without a restart.
    #[serde(default)]
    pub(crate) max_length: usize,
    /// Optional per-request batch size (operator setting). 0/absent = the configured default.
    #[serde(default)]
    pub(crate) max_batch: usize,
    /// Optional caller correlation id: echoed verbatim in the response and prefixed to this request's
    /// pass log lines. Opaque here — without it, two concurrent requests are indistinguishable in
    /// either place.
    #[serde(default)]
    pub(crate) request_id: String,
}

#[derive(Serialize)]
pub(crate) struct SparseVec {
    pub(crate) indices: Vec<u32>,
    pub(crate) values: Vec<f32>,
}

/// What the tokenizer actually did to this request's texts.
///
/// It exists because truncation here is SILENT: an input longer than `max_length` is cut to a prefix
/// by the tokenizer and embedded as though that prefix were the whole text. No error, no warning, no
/// counter — the caller receives a perfectly well-formed vector for a document it never sent. Measured
/// 2026-08-13 on an R9700, real source tokenizes at 2.99–3.50 chars/token, so a host budgeting 4
/// chars/token overshoots the window by ~34 % and loses the tail of every text it fills.
///
/// The host cannot compute any of this: it has no tokenizer. So the sidecar says it.
#[derive(Serialize, Default)]
pub(crate) struct TokenUsage {
    /// Tokens each input text costs, special tokens included, measured BEFORE the cap is applied — so a
    /// value above `max_length` is exactly the overflow that was thrown away.
    pub(crate) token_count: Vec<usize>,
    /// Per text: the model saw a PREFIX of it. Parallel to the request's `texts`.
    pub(crate) truncated: Vec<bool>,
    /// The EFFECTIVE cap those were judged against (after `cap_for`), not the requested one.
    pub(crate) max_length: usize,
    /// False when the tokenizer could not be loaded at all. The two vectors are then EMPTY, and the
    /// caller must treat "no truncation reported" as UNKNOWN rather than as proof of none — which is
    /// the whole difference between a guard and a decoration.
    pub(crate) token_accounting: bool,
}

/// Where a request's wall-clock went inside the sidecar, on the wire so the CALLER can attribute it.
///
/// Every number here used to die in this process's own log file, and the worst one was never measured
/// at all: the pass timer starts only after the engine mutex is held, so a request that waited 8 s
/// behind another caller's pass and then ran 0.4 s looked, to its caller, like a slow model. Queue
/// wait and session build stay separate fields — both are infrastructure wait, but the remedies
/// differ (concurrency vs warm-up), and a bucket that mixes two causes explains neither.
#[derive(Serialize, Default, Clone, Copy)]
pub(crate) struct PassTimings {
    /// Waiting for the engine mutex behind another request — infrastructure wait, never model speed.
    pub(crate) queue_wait_ms: u64,
    /// Building + canary-checking the session; 0 on a warm engine.
    pub(crate) session_build_ms: u64,
    /// The forward pass(es), settling re-runs included — what this request's inference actually cost.
    pub(crate) inference_ms: u64,
    /// `>0` = MIGraphX compiled this input shape during the pass. The EP saves its cache LAZILY, so
    /// growth measured across the pass is the only moment a compile is observable.
    pub(crate) compile_cache_grew_mb: u64,
}

#[derive(Serialize)]
pub(crate) struct EmbedResponse {
    pub(crate) dense: Vec<Vec<f32>>,
    pub(crate) sparse: Vec<SparseVec>,
    /// The width of the dense rows in THIS response, read from one of them.
    ///
    /// Free — the length of a row already computed — and it removes the last model constant a caller had
    /// to already know. A vector store creating a collection from the response it is holding cannot then
    /// create it at the wrong width. `null` when the batch was empty: never `0`.
    pub(crate) dimension: Option<usize>,
    #[serde(flatten)]
    pub(crate) usage: TokenUsage,
    /// Echoed from the request; empty when the caller sent none.
    pub(crate) request_id: String,
    pub(crate) timings: PassTimings,
}

/// Ask what a text really costs, WITHOUT embedding it.
///
/// It exists because the host cannot answer this for itself and every other route to the answer is worse.
/// Ollama has no `/api/tokenize` (404 on 0.32.9); its `/api/embed` does report `prompt_eval_count`, but
/// only AFTER embedding — and that endpoint silently truncates past `num_ctx`, so the number arrives once
/// the damage is done. `Microsoft.ML.Tokenizers` cannot read a HuggingFace `tokenizer.json` at all. This
/// sidecar already owns the reference implementation, so it answers for BOTH models: bge for its own
/// chunks, qwen for the semantic channel it never embeds.
#[derive(Deserialize)]
pub(crate) struct TokenizeRequest {
    pub(crate) texts: Vec<String>,
    /// `"bge"` (default) or `"qwen"`. Unknown names are refused rather than silently served by the wrong
    /// tokenizer — a count from the wrong model is worse than no count.
    #[serde(default)]
    pub(crate) model: String,
}

#[derive(Serialize, Debug)]
pub(crate) struct TokenizeResponse {
    /// Tokens each text costs, special tokens included, with NO truncation applied — the caller is asking
    /// precisely so it can split BEFORE anything is capped.
    pub(crate) token_count: Vec<usize>,
    /// Which tokenizer answered.
    pub(crate) model: String,
    /// False when that tokenizer is not loadable here; `token_count` is then empty and the caller must
    /// treat the size as unknown rather than as zero.
    pub(crate) available: bool,
}

#[derive(Deserialize)]
pub(crate) struct RerankRequest {
    pub(crate) query: String,
    pub(crate) documents: Vec<String>,
    #[serde(default)]
    pub(crate) provider: String,
    /// Sequences per forward pass, forwarded by the host exactly as `/embed` does. 0/absent keeps the
    /// configured default. Before this existed the handler read `config.max_batch` — and the AppHost passes
    /// MAX_BATCH as an empty string, so every rerank ran at the env default of 4 while the operator's stored
    /// envelope said 64: scoring 50 candidates cost 13 forward passes instead of one.
    #[serde(default)]
    pub(crate) max_batch: usize,
    /// Optional caller correlation id, exactly as on `EmbedRequest`.
    #[serde(default)]
    pub(crate) request_id: String,
}

#[derive(Serialize)]
pub(crate) struct RerankResponse {
    pub(crate) scores: Vec<f32>,
    /// Echoed from the request; empty when the caller sent none.
    pub(crate) request_id: String,
    pub(crate) timings: PassTimings,
}

#[derive(Serialize)]
pub(crate) struct HealthResponse {
    /// `"ok"`, or `"wedged"` once an engine has been held past its phase's ceiling. Deliberately not a
    /// constant: a health endpoint that always says "ok" cannot report the one failure that matters
    /// here (reliability.md § "Health endpoints tell the truth and never block").
    pub(crate) status: &'static str,
    /// What the sidecar is doing right now: "idle", "dense: building session…", "sparse:
    /// embedding N row(s)". Lets the host UI show "compiling models" instead of a dead card
    /// during a multi-minute MIGraphX first build.
    pub(crate) activity: String,
    /// LEGACY, kept so existing callers (the C# host) do not break: the active provider once one
    /// exists, otherwise the requested one. Ambiguous by construction — read the four fields below
    /// instead, which is why they exist.
    pub(crate) provider: String,
    /// What was ASKED for: ORT_PROVIDER, else the first request's hint, else "auto". Says nothing
    /// about whether it works.
    pub(crate) requested_provider: String,
    /// The EPs this binary was COMPILED with. A provider absent here can never become active, however
    /// it is configured — the failure is in the build flavor, not the settings.
    pub(crate) compiled_providers: Vec<&'static str>,
    /// The provider of a successfully created ORT session. `null` until one exists.
    pub(crate) active_provider: Option<String>,
    /// A session has been created on `active_provider`. False means no inference has ever
    /// succeeded on this process, whatever `requested_provider` says.
    pub(crate) provider_ready: bool,
    /// The last EP registration failure, verbatim. `null` when none was recorded.
    pub(crate) last_provider_error: Option<String>,
    /// SHA-256 of the executable serving this response. Empty when `current_exe()` is unreadable — or
    /// when `provenance_ready` is false and it has simply not been computed yet. A benchmark records it
    /// so a later run can prove it measured the same binary; an installed sidecar older than its commit
    /// is invisible to every other field here.
    pub(crate) exe_sha256: String,
    /// SHA-256 over the sorted `name:sha256` manifest of the dynamic libraries beside the executable.
    /// Empty when none were found (or not yet computed — see `provenance_ready`). Identical executables
    /// with different provider libraries are different runtimes, and this is the field that says so.
    pub(crate) runtime_manifest_sha256: String,
    /// The two hashes above are FINAL. False = the startup task is still hashing, so an empty hash means
    /// "not yet", not "unreadable". They are computed on the blocking pool at startup precisely because
    /// the first /health used to compute them inline — 1.4 s over 67 MB of test binaries, and a CUDA
    /// deployment is gigabytes (cuDNN alone often >500 MB) — on a Tokio reactor thread.
    pub(crate) provenance_ready: bool,
    /// The engine work in flight right now, one entry per engine that is held. THE window into the one
    /// wait this process cannot cancel: without it a wedged inference and a healthy multi-minute build
    /// looked identical from outside.
    pub(crate) in_flight: Vec<InFlightWire>,
    /// Any engine past its ceiling. The single boolean a host can route on.
    pub(crate) wedged: bool,
    pub(crate) loaded: LoadedModels,
    pub(crate) models: ModelNames,
    /// The memory envelope in force — the defaults plus the cap the loaded engines actually carry, so the
    /// host can show what is running rather than what was requested.
    pub(crate) limits: LimitsWire,
    /// The DXGI adapter this sidecar's DirectML EP targets — the ground truth the host UI labels
    /// devices with. Null = mapping unavailable (non-DML build / DXGI failure / id out of range).
    pub(crate) adapter: Option<adapters::ResolvedAdapter>,
}

#[derive(Serialize)]
pub(crate) struct LoadedModels {
    pub(crate) dense: bool,
    pub(crate) sparse: bool,
    pub(crate) rerank: bool,
}

/// One held engine, as /health reports it.
#[derive(Serialize)]
pub(crate) struct InFlightWire {
    /// `"embed"` | `"rerank"`.
    pub(crate) engine: &'static str,
    /// `"building"` | `"running"` — the ceilings differ by an order of magnitude, so the phase has to
    /// travel with the elapsed time or the number cannot be judged.
    pub(crate) phase: &'static str,
    /// What the holder said it was doing, verbatim from `activity`.
    pub(crate) activity: String,
    pub(crate) elapsed_seconds: u64,
    /// The ceiling this phase is judged against, so a reader needs no access to the configuration.
    pub(crate) ceiling_seconds: u64,
    /// Past that ceiling: no longer "slow but alive". A first-ever MIGraphX shape compile is minutes of
    /// CORRECT slowness and must never read as this — which is why `building` gets an hour.
    pub(crate) wedged: bool,
}

impl InFlightWire {
    pub(crate) fn of(engine: &'static str, holder: &InFlight, policy: WedgePolicy) -> Self {
        let elapsed = holder.since.elapsed();
        let ceiling = policy.ceiling(holder.phase);
        Self {
            engine,
            phase: holder.phase.name(),
            activity: holder.label.clone(),
            elapsed_seconds: elapsed.as_secs(),
            ceiling_seconds: ceiling.as_secs(),
            wedged: elapsed >= ceiling,
        }
    }
}

/// Every engine currently held, with its verdict. Reads only the tiny stamp mutexes, never an engine
/// lock — a probe that queued behind the wedge it is meant to report would be the defect itself.
pub(crate) fn in_flight_now(state: &AppState) -> Vec<InFlightWire> {
    [("embed", &state.engines.embed_inflight), ("rerank", &state.engines.rerank_inflight)]
        .into_iter()
        .filter_map(|(engine, slot)| inflight_now(slot).map(|holder| InFlightWire::of(engine, &holder, state.config.wedge)))
        .collect()
}

#[derive(Serialize)]
pub(crate) struct LimitsWire {
    pub(crate) embed_max_length: usize,
    /// The CONFIGURED default — what a request that carries no batch of its own falls back to. It is NOT
    /// what the last embed ran at: every request carries the operator's own batch, so this field described
    /// an intention and was read as a fact ("why 15 methods/s when the batch is 126?" — three different
    /// numbers, none of them this one). Read `loaded_max_batch` for what actually happened.
    pub(crate) max_batch: usize,
    pub(crate) rerank_max_length: usize,
    /// None = no embedding engine has been built yet, so no cap is committed to.
    pub(crate) loaded_embed_max_length: Option<usize>,
    /// The batch the most recent embed ACTUALLY ran at, request override included. `None` until one has
    /// run. The twin of `loaded_embed_max_length`, and the field whose absence made the configured default
    /// above look authoritative — the same requested-vs-active split the provider fields already carry.
    pub(crate) loaded_max_batch: Option<usize>,
    /// EVERY cap whose engines are currently resident, least-recently-used first — the cache's
    /// occupancy, which `loaded_embed_max_length` (the current rung alone) cannot show. Empty also
    /// when the probe found the cache busy: /health must never queue behind model work.
    pub(crate) resident_embed_max_lengths: Vec<usize>,
    /// The largest request body any route accepts. Here because a limit a client cannot read is a limit
    /// it will guess at — the host had to find this one by bisection after a pass died nine minutes in,
    /// and the rejection names the socket rather than the cause.
    pub(crate) max_body_bytes: usize,
    /// The most texts one `/tokenize` call may carry, for the same reason.
    pub(crate) tokenize_max_texts: usize,
}

#[derive(Serialize)]
pub(crate) struct ModelNames {
    pub(crate) dense: &'static str,
    pub(crate) sparse: &'static str,
    pub(crate) rerank: &'static str,
}

/// A model serving both an embedding head and a learned-sparse head from ONE forward pass.
pub(crate) const KIND_DENSE_SPARSE: &str = "dense+sparse";
/// A cross-encoder: it scores pairs and returns no vectors at all.
pub(crate) const KIND_RERANK: &str = "rerank";
/// A tokenizer you can COUNT with and cannot embed with — exactly what the qwen row is here.
///
/// A real kind rather than a hack. A consumer that cannot see the difference between "a model you can
/// embed with" and "a tokenizer you can count with" will eventually ask this process to embed with the
/// second one, and the clearest moment to prevent that is the read where it chooses.
pub(crate) const KIND_TOKENIZER_ONLY: &str = "tokenizer-only";

/// What this build can do, per model — `GET /models`.
#[derive(Serialize, Debug)]
pub(crate) struct ModelsResponse {
    pub(crate) models: Vec<ModelEntry>,
}

/// One thing this build can serve or count with.
#[derive(Serialize, Debug)]
pub(crate) struct ModelEntry {
    /// What a caller names it: `bge-m3`, `bge-reranker-v2-m3`, `qwen`. Short and stable — the long
    /// descriptive strings `/health` reports under `models` stay there, unchanged, because a consumer
    /// already reads them as its model id.
    pub(crate) id: &'static str,
    /// The full name of what actually loads, for a human reading this response.
    pub(crate) name: &'static str,
    /// `dense+sparse` | `rerank` | `tokenizer-only`. Present from the first version deliberately: a
    /// sparse-only model is already defined in the vendored library and wired to nothing, so the field
    /// that would have to be retrofitted exists now instead.
    pub(crate) kind: &'static str,
    /// The width of the vectors this model produces, MEASURED from a row it returned.
    ///
    /// `null` in two situations, and `kind` is what tells them apart: a row that has no vectors at all
    /// (`rerank`, `tokenizer-only`) never has one, while an embedding row reports `null` until a pass
    /// has actually produced a vector. Never `0` — see `dense_dimension`.
    pub(crate) dimension: Option<usize>,
    /// The cap a pass would run at right now: the loaded one when an engine is resident, else the
    /// configured default. `null` where the concept does not apply (a counting tokenizer imposes none).
    pub(crate) max_sequence_length: Option<usize>,
    /// The registered tokenizer name that counts for this model — the id `/tokenize` takes.
    ///
    /// `null` = this process registers no counter for it. That is the reranker's honest answer: its
    /// tokenizer lives inside the vendored library and is never read here, and claiming `bge` because
    /// the models look related is the kind of confident guess this endpoint exists to remove.
    pub(crate) tokenizer: Option<&'static str>,
    /// Whether this row can answer RIGHT NOW. For a model: an engine is resident (a busy lock reads as
    /// resident, as `/health` does). For a tokenizer row: its file loaded at startup.
    pub(crate) available: bool,
    /// Whether the tokenizer named above can COUNT right now — a fact about the file, and deliberately
    /// not folded into `available`.
    ///
    /// One flag could not carry both. On a model row `available` reports the ENGINE, so a perfectly
    /// loaded tokenizer behind an unloaded engine was invisible — and "engine cold, tokenizer ready" is
    /// exactly the state a consumer is in while it validates a recipe *before* starting a pass, which is
    /// what this endpoint exists for. `null` where the row names no tokenizer.
    pub(crate) tokenizer_available: Option<bool>,
}

/// Whether a named tokenizer can count right now. `None` when the row names none.
pub(crate) fn tokenizer_available(state: &AppState, tokenizer: Option<&str>) -> Option<bool> {
    tokenizer.map(|name| state.tokenizers.entry(name).is_some_and(|row| row.tokenizer.is_some()))
}

#[derive(Serialize, Debug)]
pub(crate) struct ErrorResponse {
    pub(crate) error: String,
}

pub(crate) type ApiError = (StatusCode, Json<ErrorResponse>);

pub(crate) fn internal_error(err: anyhow::Error) -> ApiError {
    tracing::error!("request failed: {err:#}");
    (StatusCode::INTERNAL_SERVER_ERROR, Json(ErrorResponse { error: format!("{err:#}") }))
}

/// A caller mistake, not a sidecar failure — refused with the reason so it is fixable from the message.
pub(crate) fn bad_request(error: String) -> ApiError {
    (StatusCode::BAD_REQUEST, Json(ErrorResponse { error }))
}

/// Maps an inference failure to its status code. A wedged or busy engine is **503**, not 500: nothing
/// is wrong with the request, the card is unavailable RIGHT NOW, and a host that degrades or retries on
/// 503 while treating 500 as a hard failure can only act on the difference if we make it.
pub(crate) fn engine_error(err: anyhow::Error) -> ApiError {
    if err.downcast_ref::<EngineWedged>().is_some() {
        tracing::warn!("refusing a request: {err:#}");
        return (StatusCode::SERVICE_UNAVAILABLE, Json(ErrorResponse { error: format!("{err:#}") }));
    }
    internal_error(err)
}

/// The message out of a `spawn_blocking` panic — the payload is the only place the actual reason
/// (a failed expect deep in ort/fastembed, an allocation failure) survives, and it belongs in the
/// operator's log instead of an opaque "task panicked".
pub(crate) fn join_error_text(e: tokio::task::JoinError) -> String {
    if !e.is_panic() {
        return e.to_string();
    }
    let payload = e.into_panic();
    payload
        .downcast_ref::<&str>()
        .map(|s| (*s).to_string())
        .or_else(|| payload.downcast_ref::<String>().cloned())
        .unwrap_or_else(|| "panic with a non-string payload".to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    
    
    
    
    
    
    
    use crate::bookkeeping::{dense_dimension};
    

    /// The four field names are a WIRE contract (consumed by the benchmark's telemetry and any other
    /// HTTP caller): renaming one here silently breaks a consumer that deserializes by name.
    #[test]
    fn pass_timings_serialize_under_their_contract_names() {
        let json = serde_json::to_value(PassTimings {
            queue_wait_ms: 1,
            session_build_ms: 2,
            inference_ms: 3,
            compile_cache_grew_mb: 4,
        })
        .expect("PassTimings is serializable");

        assert_eq!(json["queue_wait_ms"], 1);
        assert_eq!(json["session_build_ms"], 2);
        assert_eq!(json["inference_ms"], 3);
        assert_eq!(json["compile_cache_grew_mb"], 4);
    }

    /// embed_blocking's natural path returns `EmbedResponse { usage, ..inner }` — the struct update
    /// must keep carrying the inner call's timings and echo. This goes red if a refactor ever rebuilds
    /// that response without the new fields.
    #[test]
    fn the_struct_update_path_keeps_the_inner_timings_and_echo() {
        let inner = EmbedResponse {
            dense: vec![vec![0.0f32; 1024]],
            sparse: vec![],
            dimension: Some(1024),
            usage: TokenUsage::default(),
            request_id: "leg-7/q3".to_string(),
            timings: PassTimings {
                queue_wait_ms: 5,
                session_build_ms: 0,
                inference_ms: 9,
                compile_cache_grew_mb: 0,
            },
        };

        let outer = EmbedResponse { usage: TokenUsage::default(), ..inner };

        assert_eq!(outer.timings.inference_ms, 9, "the inner pass's timing survives the update");
        assert_eq!(outer.request_id, "leg-7/q3", "and so does the caller's echo");
        assert_eq!(outer.dimension, dense_dimension(&outer.dense), "and the width still describes THESE rows");
    }

    /// The panic PAYLOAD is the only place the real reason survives (a failed expect deep inside
    /// ort/fastembed); the log must carry it, not an opaque "task panicked".
    #[tokio::test]
    async fn a_spawn_blocking_panic_surfaces_its_message() {
        let err = tokio::task::spawn_blocking(|| panic!("the real reason"))
            .await
            .expect_err("the task panicked");
        assert!(join_error_text(err).contains("the real reason"));
    }

    #[test]
    fn the_default_usage_admits_it_measured_nothing() {
        let usage = TokenUsage::default();

        assert!(!usage.token_accounting);
        assert!(usage.token_count.is_empty());
        assert!(usage.truncated.is_empty());
    }
}
