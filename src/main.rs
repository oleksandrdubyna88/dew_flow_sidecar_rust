//! BGE sidecar: BGE-M3 dense + learned-sparse embeddings (FP32, official BAAI ONNX) and
//! BGE-Reranker-v2-M3 cross-encoder scores, served over HTTP for the v2 code-RAG pipeline.
//!
//! Contract (mirrors research/PLAN_hybrid_search_onnx.md §A):
//!   POST /embed  { texts: [string], kind: "doc"|"query", provider?: string,
//!                  max_length?: usize, max_batch?: usize }
//!                -> { dense: [[f32]], sparse: [{ indices: [u32], values: [f32] }] }
//!   POST /rerank { query: string, documents: [string], provider?: string } -> { scores: [f32] }
//!   GET  /health -> { status, provider, loaded: {dense, sparse, rerank}, models: {...},
//!                     limits: { embed_max_length, max_batch, rerank_max_length, loaded_embed_max_length },
//!                     adapter: { name, vram_mb, luid, requested_device, dml_device_id } | null }
//!
//! VRAM budget: bge-m3 is XLM-RoBERTa-large (24 layers, 16 heads). The attention scores of ONE layer
//! are `batch × 16 × seq² × 4 B`, and softmax holds a second buffer of the same size, so the peak is
//!     batch × 16 × seq² × 4 × 2
//! At the historical defaults (batch 8, seq 8192) that is 64 GiB — DirectML does not OOM, it silently
//! over-commits into shared (system) memory and the pass crawls over PCIe. `max_length` is therefore the
//! single most important knob here, and both it and `max_batch` are settable PER REQUEST so the host's
//! Settings → RAG page can tune them live; the env values below are only the bootstrap defaults.
//!   POST /unload -> same body as /health; drops all loaded engines to free VRAM (GPU lease:
//!                   an exclusive local LLM takes the card; the next /embed//rerank lazily reloads)
//!
//! Execution provider: ORT_PROVIDER env (auto|cuda|dml|migraphx|cpu) wins; else the first request's
//! `provider` hint (the C# side forwards the operator's DB setting); else "auto"
//! (cuda -> migraphx -> dml -> cpu registration chain, uncompiled/unavailable EPs fall through).
//! The migraphx flavor (AMD ROCm on Linux/WSL) is load-dynamic: ort resolves the ONNX Runtime
//! from ORT_DYLIB_PATH at startup — point it at a libonnxruntime.so built with --use_migraphx,
//! version >= 1.{ort::MINOR_VERSION}. Startup PREFLIGHTS that dylib (dlopen + OrtGetApiBase) and
//! exits with an actionable message on mismatch, because ort rc.12's own version check deadlocks
//! instead of erroring (its error path re-enters the API OnceLock it is initializing), which
//! froze the whole sidecar — and every caller — on the first model load.
//! Models lazy-load on first use and stay pinned to the provider chosen at that moment —
//! changing the setting takes effect on sidecar restart (hardware config, set once).
//!
//! Device id: ORT_DEVICE_ID counts adapters in DXGI HIGH-PERFORMANCE order (device 0 = the
//! fastest card — the same numbering the host UI's picker labels use). For DirectML it is
//! translated to the plain EnumAdapters index the legacy EP consumes (see adapters.rs); CUDA
//! keeps the raw id (its own numbering). /health reports the resolved adapter as ground truth.


mod adapters;
mod logging;
mod log_segments;
mod config;
mod wedge;
mod engine_cache;
mod state;
mod tokens;
mod preflight;
mod wire;
mod handlers;
mod introspection;
mod compile_cache;
mod inference;
mod bookkeeping;
mod canary;
mod provider;
mod testing;

use std::net::SocketAddr;
use std::sync::{Arc, Mutex, OnceLock};
use axum::http::StatusCode;
use axum::routing::{get, post};
use axum::Router;

// `logging` is no longer imported here: its one export, day_and_clock, is now used by log_segments,
// which owns the whole question of which file a line goes to.
use crate::config::*;
use crate::wedge::*;
use crate::state::*;
use crate::tokens::*;
use crate::preflight::*;
use crate::handlers::*;
use crate::introspection::*;
use crate::bookkeeping::*;
use crate::provider::*;

#[tokio::main]
pub(crate) async fn main() {
    // Stdout (the Aspire dashboard) PLUS an append-only file: the dashboard's scrollback is transient
    // and unreachable from scripts, which made every incident this week depend on a human pasting log
    // fragments. The file survives restarts and is greppable — analyze-pass-log.mjs reads it directly.
    // Best-effort: an unwritable directory must never keep the sidecar from starting.
    use tracing_subscriber::layer::SubscriberExt;
    use tracing_subscriber::util::SubscriberInitExt;
    let filter = tracing_subscriber::EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| "info,ort=warn".into());
    // A folder per DAY and a file per RUN, matching every .NET host in this product family
    // (.claude/rules/common/logging-serilog.md). Appending every run into one file was the previous shape,
    // and the question actually asked is almost always "what did THAT run do" — with two sidecars on one
    // machine and several restarts a day, one appended file makes that question unanswerable.
    // The device id stays in the name: two sidecars started in the same second differ only by their card.
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    // A run that outlives the day continues in a 00-00-00 segment under the next day's folder, same pid —
    // because "a file per run" and "this process does not restart" are each right and together produce one
    // file growing for months. Not the rolling-by-day sink the rule forbids: that merges DIFFERENT runs,
    // these two files belong to ONE. See log_segments::DaySegments.
    let (segments, log_path) = log_segments::DaySegments::open(
        &env_str("SIDECAR_LOG_DIR", "logs"),
        &format!("bge-sidecar-device{}", env_parse::<i32>("ORT_DEVICE_ID", 0)),
        now,
    );
    // Best-effort: an unwritable directory must never keep the sidecar from starting.
    let file_layer = segments.map(|writer| {
        tracing_subscriber::fmt::layer()
            .with_ansi(false)
            .with_writer(std::sync::Mutex::new(writer))
    });
    tracing_subscriber::registry()
        .with(filter)
        // ANSI on, explicitly: the host captures this stream and renders it in a dashboard, and `tracing`
        // disables colour on its own once stdout is not a terminal — which is that case exactly.
        .with(tracing_subscriber::fmt::layer().with_ansi(true))
        .with(file_layer)
        .init();
    tracing::info!("log file: {log_path} (one per run; SIDECAR_LOG_DIR overrides the directory)");

    #[cfg(feature = "migraphx")]
    {
        preflight_ort_dylib();
        preflight_migraphx_cache();
    }

    let config = Config::from_env();
    seed_model_cache_from_env(&config.cache_dir);

    // Fail-fast, but ONLY for a provider that is already decided. An explicit ORT_PROVIDER is a
    // promise this process can keep or not, and finding out at the first user search — minutes into
    // an operator's query, as a 500 — is the failure mode this replaces. An EMPTY ORT_PROVIDER is not
    // an error: the provider then comes from the first request's hint, and that path runs the same
    // preflight in `load_dual`/`load_rerank` before the first session.
    if !config.provider.trim().is_empty() {
        let requested = effective_provider(&config, "");
        if let Err(error) = preflight_provider(&requested, &exe_dir()) {
            tracing::error!("{error:#}");
            eprintln!("[bge-sidecar] fatal: {error:#}");
            std::process::exit(2);
        }
        tracing::info!(
            "provider preflight: `{requested}` is compiled in ({}) and its runtime libraries are present",
            compiled_providers().join(", ")
        );
    }

    let state_config_preview = config.clone();
    let port: u16 = env_parse("PORT", 5320);
    let adapter = adapters::resolve(config.device_id);
    match &adapter {
        Some(a) => tracing::info!(
            "ORT_DEVICE_ID {} (performance order) -> adapter '{}' ({} MB, LUID {}), DirectML device_id {}",
            a.requested_device, a.name, a.vram_mb, a.luid, a.dml_device_id
        ),
        None => tracing::warn!(
            "ORT_DEVICE_ID {} could not be resolved against the DXGI adapter list — DirectML falls back to the raw id (plain enumeration order)",
            config.device_id
        ),
    }
    tracing::info!(
        "limits: embed max_length {}, max_batch {}, rerank max_length {} (attention peak ~{} MB)",
        state_config_preview.embed_max_length,
        state_config_preview.max_batch,
        state_config_preview.rerank_max_length,
        attention_peak_mb(state_config_preview.max_batch, state_config_preview.embed_max_length)
    );
    tracing::info!(
        "engine cache: up to {} sequence rung(s) stay resident per head — a cap change is a lookup, not a rebuild (EMBED_ENGINE_CACHE_RUNGS raises it when the two-rung ladder is opted back in)",
        state_config_preview.engine_cache_rungs.max(1)
    );
    // Every tokenizer is read HERE, before the listener is up — so no request pays a directory walk and
    // a multi-MB parse, and what this build can count for is in the startup log rather than discovered
    // by a caller getting a 400.
    let tokenizers = TokenizerRegistry::load(&config);
    tracing::info!(
        "tokenizers registered: {} (up to TOKENIZE_MAX_TEXTS={} texts per call) — an unknown name is refused naming this set",
        tokenizers.describe(),
        config.tokenize_max_texts
    );
    let state = Arc::new(AppState {
        engines: Engines::new(config.engine_cache_rungs),
        config,
        activity: Mutex::new("idle".to_string()),
        pinned_provider: OnceLock::new(),
        active_provider: Mutex::new(None),
        last_provider_error: Mutex::new(None),
        loaded_embed_max_length: Mutex::new(None),
        loaded_max_batch: Mutex::new(None),
        loaded_embed_dimension: Mutex::new(None),
        adapter,
        tokenizers,
    });

    tracing::info!(
        "wedge detector: a build is slow-but-alive for up to {}s, a forward pass for {}s; past that /health \
         reports status \"wedged\" and new requests are refused with the reason. Process exit on a wedge is {} \
         (WEDGE_EXIT)",
        state.config.wedge.building_after.as_secs(),
        state.config.wedge.running_after.as_secs(),
        match state.config.wedge.exit_after_wedged {
            Some(after) => format!("ON, {}s after the verdict", after.as_secs()),
            None => "OFF".to_string(),
        }
    );
    prewarm_provenance();
    spawn_wedge_watchdog(state.clone());

    let app = build_router(state);

    let addr = SocketAddr::from(([127, 0, 0, 1], port));
    tracing::info!("bge-sidecar listening on http://{addr}");
    let listener = tokio::net::TcpListener::bind(addr).await.expect("bind failed");
    axum::serve(listener, app)
        .with_graceful_shutdown(async {
            let _ = tokio::signal::ctrl_c().await;
        })
        .await
        .expect("server failed");
}

/// The routes and the layers around them, built apart from `main` so a test can drive them without
/// binding a port — a body limit is a LAYER, and nothing below the router can prove one is applied.
pub(crate) fn build_router(state: Arc<AppState>) -> Router {
    let max_body_bytes = state.config.max_body_bytes;
    Router::new()
        .route("/health", get(health))
        .route("/models", get(models))
        .route("/embed", post(embed))
        .route("/rerank", post(rerank))
        .route("/unload", post(unload))
        .route("/tokenize", post(tokenize))
        .with_state(state)
        // Set EXPLICITLY. The value is axum's own default, so nothing changes but the fact that it is a
        // decision — readable on /health, movable by an operator, and no longer a framework constant
        // nobody could find. The log layer is outermost so it observes the rejection the limit produces.
        .layer(axum::extract::DefaultBodyLimit::max(max_body_bytes))
        .layer(axum::middleware::from_fn(log_body_rejections))
}

/// Says, in the log, when a request was refused for its SIZE.
///
/// axum rejects an oversized body before any handler runs, so a 413 produced nothing at all in
/// `bge-sidecar-*.log` — and it reaches the caller as a socket abort ("an established connection was
/// aborted by the software in your host machine"), which names neither the size nor the cap, because the
/// server rejects while the client is still writing. A 10,000-file repository died nine minutes into an
/// indexing pass this way and it cost an afternoon to find. This is the line that makes it five minutes.
pub(crate) async fn log_body_rejections(request: axum::extract::Request, next: axum::middleware::Next) -> axum::response::Response {
    let route = request.uri().path().to_string();
    let announced = request
        .headers()
        .get(axum::http::header::CONTENT_LENGTH)
        .and_then(|value| value.to_str().ok())
        .unwrap_or("unstated")
        .to_string();
    let response = next.run(request).await;
    if response.status() == StatusCode::PAYLOAD_TOO_LARGE {
        tracing::warn!(
            "{route}: request body refused — {announced} byte(s) announced, over MAX_BODY_BYTES. The caller \
             sees only a socket abort, so it cannot learn this from the response: batch under the cap \
             /health reports as `max_body_bytes`."
        );
    }
    response
}
