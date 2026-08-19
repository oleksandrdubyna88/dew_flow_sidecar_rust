use crate::config::Config;
use crate::inference::{embed_blocking, rerank_batch, rerank_blocking, set_activity};
use crate::state::{AppState, Limits};
use crate::tokens::{count_tokens, BGE_TOKENIZER};
use crate::wire::{
    bad_request, engine_error, internal_error, join_error_text, ApiError, EmbedRequest,
    EmbedResponse, PassTimings, RerankRequest, RerankResponse, TokenUsage, TokenizeRequest,
    TokenizeResponse,
};
use axum::extract::State;
use axum::Json;
use std::sync::Arc;

// ---------- handlers ----------

/// Records that somebody asked this process to do something.
///
/// Read by exactly one thing — the idle unloader, which is off unless an operator turned it on. Written
/// here rather than in the inference path so a REFUSED request counts too: a caller hammering a wedged
/// engine is not an idle sidecar, and unloading underneath them would turn one problem into two.
fn touch(state: &AppState) {
    if let Ok(mut at) = state.last_request.try_lock() {
        *at = std::time::Instant::now();
    }
}

/// Runs one blocking pass and turns its TWO failure modes into the two different HTTP answers they
/// deserve.
///
/// `/embed` and `/rerank` carried an identical copy of this — spawn, clear the activity, unwrap the join,
/// map the engine error — differing only in the closure and one word in the panic message. The pair had
/// already started to matter: `set_activity(idle)` has to happen even when the pass FAILED, or the
/// sidecar reports itself busy forever after one bad request, and that is the kind of line a second copy
/// eventually loses.
///
/// The two failures are not the same thing and must not collapse:
/// a JoinError means the blocking task PANICKED (a bug here, 500), while an `Err` from the pass is the
/// engine refusing — which `engine_error` may turn into a 503 the host can retry.
async fn run_pass<T, F>(state: &Arc<AppState>, what: &str, pass: F) -> Result<Json<T>, ApiError>
where
    F: FnOnce() -> anyhow::Result<T> + Send + 'static,
    T: Send + 'static,
{
    touch(state);
    let idle = state.clone();
    let outcome = tokio::task::spawn_blocking(pass).await;
    set_activity(&idle, "idle");
    let result = outcome
        .map_err(|e| {
            internal_error(anyhow::anyhow!(
                "{what} task panicked: {}",
                join_error_text(e)
            ))
        })?
        .map_err(engine_error)?;
    Ok(Json(result))
}

pub(crate) async fn embed(
    State(state): State<Arc<AppState>>,
    Json(req): Json<EmbedRequest>,
) -> Result<Json<EmbedResponse>, ApiError> {
    let EmbedRequest {
        texts,
        kind,
        provider,
        max_length,
        max_batch,
        request_id,
    } = req;
    if texts.is_empty() {
        return Ok(Json(EmbedResponse {
            dense: vec![],
            sparse: vec![],
            // An empty batch measured nothing. Reporting the last pass's width here would state a fact
            // about vectors this response does not contain — /models is where the process's own width
            // belongs, and it says "unknown" until something has actually been embedded.
            dimension: None,
            usage: TokenUsage::default(),
            request_id,
            timings: PassTimings::default(),
        }));
    }

    let limits = Limits::resolve(&state.config, max_length, max_batch);
    let pass = state.clone();
    run_pass(&state, "embed", move || {
        embed_blocking(&pass, texts, &kind, &provider, limits, &request_id)
    })
    .await
}

/// The name this request counts with: its own, or bge when it named none.
pub(crate) fn requested_tokenizer(model: &str) -> String {
    match model.trim() {
        "" => BGE_TOKENIZER.to_string(),
        named => named.to_string(),
    }
}

/// Refuses a name this build does not register, NAMING the ones it does.
///
/// The refusal is the discoverability: a caller that misspells a tokenizer learns the registered set
/// from the answer instead of reading this file. Serving the request with the wrong tokenizer would
/// answer confidently and wrongly, which is the one outcome a chunker cannot detect.
pub(crate) fn known_tokenizer(state: &AppState, model: &str) -> Result<(), ApiError> {
    match state.tokenizers.entry(model) {
        Some(_) => Ok(()),
        None => Err(bad_request(format!(
            "unknown tokenizer '{model}' — this sidecar counts for {}. Serving the request with the \
             wrong tokenizer would answer confidently and wrongly.",
            state.tokenizers.names()
        ))),
    }
}

/// Refuses a batch past `TOKENIZE_MAX_TEXTS`, stating the cap so the caller can batch to it.
pub(crate) fn tokenize_batch_within_cap(config: &Config, texts: usize) -> Result<(), ApiError> {
    if texts <= config.tokenize_max_texts {
        return Ok(());
    }
    Err(bad_request(format!(
        "{texts} texts in one /tokenize call, and the cap is {} (TOKENIZE_MAX_TEXTS) — split the batch. \
         The encode is pure CPU, and the request body limit does not bound the ROW count: enough short \
         texts fit under it to occupy this process for seconds.",
        config.tokenize_max_texts
    )))
}

/// Counts tokens without embedding: a vocabulary lookup and BPE merges, no session, no GPU, no model
/// weights. That is what makes it safe to call from an index pass — it never queues behind the card.
///
/// The encode runs on the BLOCKING pool even so. "Pure CPU" is a statement about the card, not about the
/// reactor: a batch of ten thousand encodes in front of the async runtime stalls /health and /unload,
/// which are the two endpoints an operator reaches for when something is stalled. /embed has always
/// tokenized inside its own `spawn_blocking`, and this is the same work.
pub(crate) async fn tokenize(
    State(state): State<Arc<AppState>>,
    Json(req): Json<TokenizeRequest>,
) -> Result<Json<TokenizeResponse>, ApiError> {
    touch(&state);
    let TokenizeRequest { texts, model } = req;
    let model = requested_tokenizer(&model);
    known_tokenizer(&state, &model)?;
    tokenize_batch_within_cap(&state.config, texts.len())?;

    let shared = state.clone();
    let counting = model.clone();
    // Resolved again inside the task rather than carried in: the borrow cannot cross `spawn_blocking`,
    // and re-reading a two-row table is free next to an `.expect()` that a later refactor could turn
    // into a panic on the request path.
    let counted = tokio::task::spawn_blocking(move || {
        shared
            .tokenizers
            .entry(&counting)
            .and_then(|entry| entry.tokenizer.as_ref())
            .map(|tokenizer| count_tokens(tokenizer, &texts, "tokenize"))
    })
    .await
    .map_err(|e| {
        internal_error(anyhow::anyhow!(
            "tokenize task panicked: {}",
            join_error_text(e)
        ))
    })?;

    // Same rule as /embed's accounting: one refusal makes the whole answer UNKNOWN rather than zero.
    // A caller asks here precisely so it can split BEFORE anything is capped, and a `0` it cannot tell
    // from an empty text is worse than an honest "not measured". An unloadable tokenizer folds into the
    // same answer: both mean NOT MEASURED, and the wire has one way to say that.
    let Some(token_count) =
        counted.and_then(|counts| counts.into_iter().collect::<Option<Vec<usize>>>())
    else {
        return Ok(Json(TokenizeResponse {
            token_count: vec![],
            model,
            available: false,
        }));
    };
    Ok(Json(TokenizeResponse {
        token_count,
        model,
        available: true,
    }))
}

pub(crate) async fn rerank(
    State(state): State<Arc<AppState>>,
    Json(req): Json<RerankRequest>,
) -> Result<Json<RerankResponse>, ApiError> {
    let RerankRequest {
        query,
        documents,
        provider,
        max_batch,
        request_id,
    } = req;
    if documents.is_empty() {
        return Ok(Json(RerankResponse {
            scores: vec![],
            request_id,
            timings: PassTimings::default(),
        }));
    }

    let max_batch = rerank_batch(&state.config, max_batch);
    let pass = state.clone();
    run_pass(&state, "rerank", move || {
        rerank_blocking(&pass, query, documents, &provider, max_batch, &request_id)
    })
    .await
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::build_router;
    use crate::testing::*;
    use axum::extract::State;
    use axum::http::StatusCode;
    use std::sync::mpsc;
    use std::time::{Duration, Instant};
    use tower::ServiceExt;

    use crate::inference::lock_or_refuse;
    use crate::tokens::BGE_TOKENIZER;
    use crate::wedge::{inflight_now, write_inflight, Patience, Phase};
    use crate::wire::TokenizeRequest;

    /// The request path had no deadline of any kind: every /embed behind a wedged inference queued on
    /// `.lock()` forever, and the daemon's deliberately infinite HTTP timeout turned that into a
    /// system-wide freeze nobody could see. A request must be REFUSED with a reason instead.
    #[test]
    fn a_request_refuses_instead_of_queueing_behind_a_wedged_inference() {
        let state = app_state();
        let _held = HeldEngine::hold(state.clone(), |s| &s.engines.embed);
        write_inflight(
            &state.engines.embed_inflight,
            Some(stamped(
                Phase::Running,
                "embed: embedding 64 row(s)",
                Duration::from_secs(30),
            )),
        );

        let asked = Instant::now();
        let Err(refused) = lock_or_refuse(
            &state.engines.embed,
            &state.engines.embed_inflight,
            "embed",
            state.config.wedge,
            Patience::UntilTheHolderIsWedged,
        ) else {
            panic!("a wedged engine must refuse, not queue");
        };
        let waited = asked.elapsed();

        assert!(
            waited < Duration::from_millis(200),
            "the refusal is immediate, not another wait: {waited:?}"
        );
        let text = format!("{refused:#}");
        assert!(text.contains("WEDGED"), "{text}");
        assert!(
            text.contains("embed: embedding 64 row(s)"),
            "the reason names what is holding it: {text}"
        );
        assert!(text.contains("/unload"), "and how to recover: {text}");
    }

    /// The counter-guarantee, and the reason the ceilings are minutes rather than seconds: a FIRST-EVER
    /// MIGraphX shape compile is minutes of CORRECT slowness (measured 214 s on an R9700). A waiter must
    /// ride that out — refusing it would fail an index pass that was about to succeed.
    #[test]
    fn a_request_still_queues_behind_a_cold_compile_that_is_slow_but_alive() {
        let state = app_state();
        let held = HeldEngine::hold(state.clone(), |s| &s.engines.embed);
        // Well inside the 600 ms building ceiling this test config carries.
        write_inflight(
            &state.engines.embed_inflight,
            Some(stamped(
                Phase::Building,
                "embed: building the session",
                Duration::from_millis(10),
            )),
        );

        let waiter = state.clone();
        let (answered, outcome) = mpsc::channel();
        let thread = std::thread::spawn(move || {
            let taken = lock_or_refuse(
                &waiter.engines.embed,
                &waiter.engines.embed_inflight,
                "embed",
                waiter.config.wedge,
                Patience::UntilTheHolderIsWedged,
            );
            answered.send(taken.is_ok()).ok();
        });

        assert!(
            outcome.recv_timeout(Duration::from_millis(80)).is_err(),
            "a healthy build must NOT be refused — the waiter is supposed to still be waiting"
        );
        drop(held);
        assert_eq!(
            outcome.recv_timeout(Duration::from_secs(2)),
            Ok(true),
            "and it gets the engine once the build ends"
        );
        thread.join().ok();
    }

    /// A hold that nothing stamped still needs a ceiling — a missing stamp is exactly the case where a
    /// wait would otherwise be unbounded again, so the fallback must not depend on the record existing.
    #[test]
    fn an_unstamped_hold_is_still_refused_at_the_running_ceiling() {
        let state = app_state();
        let _held = HeldEngine::hold(state.clone(), |s| &s.engines.rerank);

        let asked = Instant::now();
        let Err(refused) = lock_or_refuse(
            &state.engines.rerank,
            &state.engines.rerank_inflight,
            "rerank",
            state.config.wedge,
            Patience::UntilTheHolderIsWedged,
        ) else {
            panic!("an unstamped hold cannot be waited on forever either");
        };

        assert!(
            asked.elapsed() >= state.config.wedge.running_after,
            "it waited the fallback ceiling out first"
        );
        assert!(format!("{refused:#}").contains("rerank"), "{refused:#}");
        assert!(
            inflight_now(&state.engines.rerank_inflight).is_none(),
            "precondition: nothing had stamped it"
        );
    }

    /// An unknown name is refused — and the refusal names the set this build actually registered, rather
    /// than a sentence somebody has to remember to edit. Three rows, because two can be hardcoded and
    /// three cannot: this is the guarantee that a new tokenizer is a row and not a code change.
    #[tokio::test]
    async fn an_unknown_tokenizer_is_refused_naming_every_registered_name() {
        let state = app_state_with_tokenizers(&["bge", "qwen", "gemma"]);

        let refused = tokenize(
            State(state),
            axum::Json(TokenizeRequest {
                texts: vec!["alpha".to_string()],
                model: "llama".to_string(),
            }),
        )
        .await
        .expect_err("an unknown tokenizer must be refused, never served by the wrong one");

        assert_eq!(refused.0, StatusCode::BAD_REQUEST);
        for registered in ["bge", "qwen", "gemma"] {
            assert!(
                refused.1.error.contains(registered),
                "the refusal must name '{registered}' so the caller can correct itself: {}",
                refused.1.error
            );
        }
    }

    /// A registered name whose file is absent degrades THAT NAME ONLY. The distinction the wire draws is
    /// between "not a name here" (400, the caller is wrong) and "a name here that cannot answer today"
    /// (200 with `available: false`, the deployment is wrong) — collapsing the two would send a chunker
    /// hunting for a typo while a file was simply missing.
    #[tokio::test]
    async fn a_missing_tokenizer_file_degrades_one_name_and_leaves_the_others_answering() {
        let cache = model_cache_with_a_tokenizer("degrade");
        let mut config = config("");
        config.cache_dir = cache.clone();
        config.qwen_tokenizer_path = cache.join("there-is-no-qwen-here.json");
        let state = app_state_with(config);

        let missing = tokenize(
            State(state.clone()),
            axum::Json(TokenizeRequest {
                texts: vec!["alpha beta".to_string()],
                model: "qwen".to_string(),
            }),
        )
        .await
        .expect("a registered name is never a 400, however unloadable it is");
        assert!(!missing.available, "qwen has no file here");
        assert!(
            missing.token_count.is_empty(),
            "and an unavailable counter reports no numbers at all"
        );

        let present = tokenize(
            State(state),
            axum::Json(TokenizeRequest {
                texts: vec!["alpha beta".to_string()],
                model: "bge".to_string(),
            }),
        )
        .await
        .expect("bge answers");
        assert!(
            present.available,
            "one absent file must not take the whole registry down"
        );
        assert_eq!(present.token_count, vec![2], "and it really counted");

        std::fs::remove_dir_all(&cache).ok();
    }

    /// The batch cap: /tokenize used to accept any number of texts and encode them all inline on the
    /// async runtime. The body limit does NOT bound this — 52,617 texts is a real number from a
    /// 10,000-file pass, and enough of them are short that the batch fits well under 2 MB while still
    /// being tens of thousands of encodes in front of the reactor.
    #[tokio::test]
    async fn tokenize_refuses_a_batch_beyond_the_cap() {
        let state = app_state_with_tokenizers(&[BGE_TOKENIZER]);
        let cap = state.config.tokenize_max_texts;

        let refused = tokenize(
            State(state),
            axum::Json(TokenizeRequest {
                texts: vec!["alpha".to_string(); cap + 1],
                model: String::new(),
            }),
        )
        .await
        .expect_err("a batch past the cap is refused rather than encoded");

        assert_eq!(refused.0, StatusCode::BAD_REQUEST);
        assert!(
            refused.1.error.contains(&cap.to_string()),
            "the refusal has to state the cap, or the caller cannot batch to it: {}",
            refused.1.error
        );
    }

    // ---------- the request body limit, stated rather than inherited ----------
    /// The CONFIGURED limit has to be the one actually enforced.
    ///
    /// Every route used to run on axum's default 2 MB — a real limit that was never written down, never
    /// reported, and produced nothing in the log when it fired, because axum rejects the body before any
    /// handler runs. The body below is far under 2 MB and far over the configured cap, so it separates
    /// "a limit exists" from "OUR limit is in force": before the layer was applied this request was
    /// accepted, and the assertion named the status it got instead.
    #[tokio::test]
    async fn a_body_beyond_the_configured_limit_is_refused() {
        let mut config = config("");
        config.max_body_bytes = 1024;
        let app = build_router(app_state_with(config));

        let oversized = format!(r#"{{"texts":["{}"],"model":"bge"}}"#, "x".repeat(4096));
        let response = app
            .oneshot(
                axum::http::Request::builder()
                    .method("POST")
                    .uri("/tokenize")
                    .header("content-type", "application/json")
                    .body(axum::body::Body::from(oversized))
                    .expect("a well-formed request"),
            )
            .await
            .expect("the router answers");

        assert_eq!(
            response.status(),
            StatusCode::PAYLOAD_TOO_LARGE,
            "a 4 KB body must not pass a 1 KB cap just because axum's own default is 2 MB"
        );
    }

    /// A body UNDER the cap still goes through — a limit that refuses everything is not a limit.
    #[tokio::test]
    async fn a_body_within_the_configured_limit_still_reaches_the_handler() {
        let mut config = config("");
        config.max_body_bytes = 1024;
        let app = build_router(app_state_with(config));

        let response = app
            .oneshot(
                axum::http::Request::builder()
                    .method("POST")
                    .uri("/tokenize")
                    .header("content-type", "application/json")
                    .body(axum::body::Body::from(
                        r#"{"texts":["alpha"],"model":"bge"}"#,
                    ))
                    .expect("a well-formed request"),
            )
            .await
            .expect("the router answers");

        assert_eq!(
            response.status(),
            StatusCode::OK,
            "the handler ran and answered"
        );
    }
}
