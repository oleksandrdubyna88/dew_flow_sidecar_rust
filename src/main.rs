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

use anyhow::Context;
use std::net::SocketAddr;

/// UTC date and clock for the log path, from a unix timestamp.
///
/// Written out rather than pulled from `chrono`: this is the only place the crate needs a calendar date, and
/// a dependency added for one format string is a dependency to audit, licence and keep current forever.
/// The civil-from-days algorithm is Howard Hinnant's, valid for any date this product will see.
fn day_and_clock(unix_seconds: u64) -> (String, String) {
    let days = (unix_seconds / 86_400) as i64;
    let secs = unix_seconds % 86_400;

    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1_460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = if m <= 2 { y + 1 } else { y };

    (
        format!("{year:04}-{m:02}-{d:02}"),
        format!("{:02}-{:02}-{:02}", secs / 3600, (secs % 3600) / 60, secs % 60),
    )
}
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant};

use axum::body::Bytes;
use axum::extract::State;
use axum::http::StatusCode;
use axum::routing::{get, post};
use axum::{Json, Router};
use fastembed::{Bgem3DualEmbedding, Bgem3DualInitOptions, RerankInitOptions, RerankerModel, TextRerank};
use ort::execution_providers::{
    CUDAExecutionProvider, DirectMLExecutionProvider, ExecutionProviderDispatch,
    MIGraphXExecutionProvider,
};
use serde::{Deserialize, Serialize};

const DENSE_MODEL: &str = "BAAI/bge-m3 (dense, FP32)";
const SPARSE_MODEL: &str = "BAAI/bge-m3 (learned sparse, FP32)";
/// What actually loads: ONE session whose run serves both of the above heads.
const DUAL_MODEL: &str = "BAAI/bge-m3 (dense+sparse heads, FP32, one session)";
const RERANK_MODEL: &str = "bge-reranker-v2-m3";

/// Runtime knobs, all env-driven (the AppHost injects them; sane defaults for a bare `cargo run`).
#[derive(Clone)]
struct Config {
    /// auto | cuda | dml | migraphx | cpu. Empty = decide from the first request hint / auto.
    provider: String,
    device_id: i32,
    /// Default embedding batch size when a request does not carry one. Attention is O(batch × seq²) —
    /// see the VRAM budget in the module header before raising it.
    max_batch: usize,
    /// 0 = ONNX Runtime decides.
    intra_threads: usize,
    /// Default token cap per sequence when a request does not carry one. THE memory driver: cost grows
    /// with its SQUARE. 1024 tokens (~4k chars) covers essentially every method body we embed.
    embed_max_length: usize,
    rerank_max_length: usize,
    cache_dir: PathBuf,
    /// Where Qwen's `tokenizer.json` lives (`QWEN_TOKENIZER`). Counting only — no Qwen model is ever loaded.
    qwen_tokenizer_path: PathBuf,
    /// Force every embedding batch into ONE tensor shape (see `pin_shape`). "auto" (the default)
    /// turns it on only for MIGraphX, the provider that compiles per shape; "1"/"0" force it.
    pin_input_shape: String,
    /// The MIGraphX compiled-model cache root as the AppHost passed it (empty = no cache /
    /// non-migraphx flavor). Engine builds redirect the EP into a PER-ENGINE subdirectory of it —
    /// see `with_engine_cache` for why sharing one directory corrupted the sparse engine.
    mxr_cache_base: String,
    /// How many sequence caps keep their engines resident at once (`RungCache` capacity). 2 = the
    /// ladder's own maximum, so a pass never rebuilds a rung it will return to; 1 restores the old
    /// evict-on-change behaviour.
    engine_cache_rungs: usize,
    /// When a wait stops being "slow but alive" — see `WedgePolicy`.
    wedge: WedgePolicy,
}

impl Config {
    fn from_env() -> Self {
        Self {
            provider: env_str("ORT_PROVIDER", ""),
            device_id: env_parse("ORT_DEVICE_ID", 0),
            // 64, raised from 4 on 2026-08-13 to match SidecarMemory.DefaultMaxBatch. At the shipped 256
            // cap a batch of 64 costs 512 MiB of attention; the old 4 was sized for the 1024-token era and
            // silently re-batched a 126-text call into 32 forward passes whenever a request carried no
            // batch of its own. The AppHost passes MAX_BATCH as an EMPTY string, so this default is what a
            // real deployment actually runs on until a request overrides it.
            max_batch: env_parse("MAX_BATCH", 64),
            intra_threads: env_parse("ORT_THREADS", 0),
            // 256 mirrors the host's shipped cap (SidecarMemory.DefaultMaxLength). It is a BOOTSTRAP value
            // only — every /embed request carries the operator's own max_length — but a bare `cargo run`
            // and the window before the first request should not build an engine the host will never ask
            // for. The reranker keeps 1024: its cap is independent, and its documents are prose.
            embed_max_length: env_parse("EMBED_MAX_LENGTH", 256),
            rerank_max_length: env_parse("RERANK_MAX_LENGTH", 1024),
            cache_dir: PathBuf::from(env_str("MODEL_CACHE_DIR", ".model-cache")),
            // Beside this tool by default (tools/qwen-tokenizer/), because it belongs to the SEMANTIC
            // channel rather than to any model this sidecar loads — it is here only to be counted with.
            qwen_tokenizer_path: PathBuf::from(env_str("QWEN_TOKENIZER", "../qwen-tokenizer/tokenizer.json")),
            pin_input_shape: env_str("PIN_INPUT_SHAPE", "auto"),
            mxr_cache_base: env_str("ORT_MIGRAPHX_MODEL_CACHE_PATH", ""),
            // 1, not 2: the host ships a SINGLE 256 rung now (RagSettingLimits.PlanFor), so a second slot
            // would only hold an engine nothing asks for. An operator who opts the ladder back in raises
            // this with the same setting that raises the rung count.
            engine_cache_rungs: env_parse("EMBED_ENGINE_CACHE_RUNGS", 1),
            wedge: WedgePolicy::from_env(),
        }
    }
}

fn env_str(key: &str, default: &str) -> String {
    std::env::var(key).ok().filter(|v| !v.trim().is_empty()).unwrap_or_else(|| default.to_string())
}

fn env_parse<T: std::str::FromStr>(key: &str, default: T) -> T {
    std::env::var(key).ok().and_then(|v| v.trim().parse().ok()).unwrap_or(default)
}

/// An opt-in switch, spelled the way every other boolean knob here is (`PIN_INPUT_SHAPE`). Absent
/// or unrecognised means OFF — a switch whose default is "maybe" is not a switch.
fn env_truthy(key: &str) -> bool {
    matches!(env_str(key, "").trim().to_lowercase().as_str(), "1" | "true" | "on" | "yes")
}

// ---------- the wedge detector: this file's one unbounded wait, made observable ----------
//
// An ORT/MIGraphX forward pass cannot be cancelled. It is a C++ call on a thread we do not own, and a
// thread merely STUCK inside it never panics — so `lock_or_refuse`'s poison healing, which recovers a
// mutex a PANIC poisoned, can never reach it. Before 2026-08-16 that combination had no detector at
// all: every later /embed queued on `.lock()` forever, /health reported the freeze exactly as it
// reports a healthy multi-minute build, and the daemon's deliberately infinite sidecar HTTP timeout
// composed the two into a system-wide freeze nobody could see (the four-repo reliability audit,
// .claude/rules/shared/common/reliability.md § "Every wait has a ceiling").
//
// The remedy is not cancellation — it cannot be — but VISIBILITY plus a ceiling: stamp what holds an
// engine and since when, declare it wedged once that passes the phase's ceiling, refuse new requests
// with a reason instead of queueing them, and say so in the log without waiting to be asked.

/// What an engine's holder is doing. The two phases exist because their ceilings differ by an order
/// of magnitude, and conflating them is what would make a correct cold compile look like a freeze.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Phase {
    /// Building + canary-checking a session. Minutes here are CORRECT.
    Building,
    /// A forward pass on an engine that is already built.
    Running,
}

impl Phase {
    /// The name /health reports.
    fn name(self) -> &'static str {
        match self {
            Phase::Building => "building",
            Phase::Running => "running",
        }
    }
}

/// The ceilings, all env-overridable. Deliberately generous: a false "wedged" is expensive in both
/// directions — it refuses requests a healthy compile would have served, and (with the opt-in exit)
/// killing a process mid-compile is precisely how a corrupt `.mxr` lands in the compiled-model cache,
/// the 2026-07-31 incident the build canary exists for.
#[derive(Clone, Copy)]
struct WedgePolicy {
    /// A forward pass on a built engine. Warm passes are seconds (measured 1.6 s at a 256 cap, 6.8 s at
    /// 1024); the slowest legitimate one on record is ~608 s, when a first request also paid a lazy
    /// MIGraphX compile plus its settling retries, and a first rerank pass compiles 92-162 s with no
    /// canary ahead of it. 900 s is ~1.5x that worst honest case.
    running_after: Duration,
    /// Building + canary-checking a session. This phase legitimately contains the cold compile
    /// (measured 214 s), up to `SETTLE_ATTEMPTS` canary runs, and — on a corrupt cache — a wipe plus one
    /// clean recompile with a canary of its own. An hour, not a quarter of one.
    building_after: Duration,
    /// How long /unload waits for an engine before answering "still loaded". It is the operator's
    /// recovery tool and the host's GPU-lease handover: long enough to ride out a normal in-flight pass,
    /// short enough that the coordinator gets an answer rather than a hang.
    unload_wait: Duration,
    /// How often a waiter re-checks the lock and the holder's stamp.
    poll: Duration,
    /// Recovery of last resort: exit the process — the host restarts the sidecar — once an engine has
    /// been WEDGED (not merely busy) for this long on top of its ceiling. `None`, the DEFAULT, never
    /// exits. Opt in with `WEDGE_EXIT=1`.
    exit_after_wedged: Option<Duration>,
}

impl WedgePolicy {
    fn from_env() -> Self {
        let running_after = Duration::from_secs(env_parse("WEDGE_RUNNING_AFTER_SECONDS", 900));
        Self {
            running_after,
            building_after: Duration::from_secs(env_parse("WEDGE_BUILDING_AFTER_SECONDS", 3600)),
            unload_wait: Duration::from_secs(env_parse("UNLOAD_LOCK_WAIT_SECONDS", 30)),
            poll: Duration::from_millis(env_parse("WEDGE_POLL_MS", 50)),
            exit_after_wedged: env_truthy("WEDGE_EXIT").then(|| {
                Duration::from_secs(env_parse("WEDGE_EXIT_AFTER_SECONDS", running_after.as_secs()))
            }),
        }
    }

    /// How long this phase may run before it stops being "slow but alive".
    fn ceiling(&self, phase: Phase) -> Duration {
        match phase {
            Phase::Building => self.building_after,
            Phase::Running => self.running_after,
        }
    }
}

/// What holds an engine right now, and since when.
///
/// It lives in its OWN tiny mutex, never inside the engine slot, and that is the whole design: a
/// holder wedged UNDER the engine mutex could never be observed THROUGH that same mutex, which is
/// exactly why the freeze was invisible. This lock is only ever held for the length of an assignment.
#[derive(Clone)]
struct InFlight {
    phase: Phase,
    /// The same human label /health already showed as `activity` ("embed: embedding 64 row(s)").
    label: String,
    since: Instant,
}

/// Stamps the engine a request holds, and CLEARS the stamp on drop — including the `?` early return
/// and the panic unwind, which is where a hand-written clear gets forgotten and leaves a phantom wedge
/// that refuses every later request for the life of the process.
struct InFlightStamp<'a> {
    state: &'a AppState,
    slot: &'a Mutex<Option<InFlight>>,
}

impl<'a> InFlightStamp<'a> {
    fn hold(state: &'a AppState, slot: &'a Mutex<Option<InFlight>>) -> Self {
        Self { state, slot }
    }

    /// Enters a phase: re-stamps the record — the clock restarts, because a finished build is not part
    /// of the pass that follows it — and mirrors the label into /health's `activity`, so the operator's
    /// window and the wedge detector can never disagree about what is happening.
    fn enter(&self, phase: Phase, label: impl Into<String>) {
        let label = label.into();
        set_activity(self.state, label.clone());
        write_inflight(self.slot, Some(InFlight { phase, label, since: Instant::now() }));
    }
}

impl Drop for InFlightStamp<'_> {
    fn drop(&mut self) {
        write_inflight(self.slot, None);
    }
}

/// Writes the in-flight record, healing poison. This mutex guards three fields and is never held
/// across anything that can block, so a poisoned one means a panic somewhere else — never a wedge
/// here — and failing to clear the stamp because of it would strand the engine as permanently busy.
fn write_inflight(slot: &Mutex<Option<InFlight>>, record: Option<InFlight>) {
    match slot.lock() {
        Ok(mut guard) => *guard = record,
        Err(poisoned) => {
            slot.clear_poison();
            *poisoned.into_inner() = record;
        }
    }
}

/// A NON-BLOCKING read of the in-flight record. /health's standing rule, and doubly so here: the
/// entire purpose of this slot is to stay readable while something else is stuck.
fn inflight_now(slot: &Mutex<Option<InFlight>>) -> Option<InFlight> {
    slot.try_lock().ok().and_then(|holder| holder.clone())
}

/// The refusal a caller gets rather than an unbounded queue. A distinct error type so the HTTP layer
/// can answer **503** (temporary — retry, degrade, or look at /health) instead of 500 ("your request
/// was wrong"): the host's degradation logic reads that difference, and a wedge is not the caller's
/// fault.
#[derive(Debug)]
struct EngineWedged {
    what: String,
    /// What the holder said it was doing; empty when nothing had stamped it.
    activity: String,
    elapsed: Duration,
    /// True = the holder passed its own ceiling (a real wedge). False = it is still legitimately busy
    /// and THIS caller ran out of patience, which is a different sentence for the operator to read.
    wedged: bool,
}

impl std::fmt::Display for EngineWedged {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let seconds = self.elapsed.as_secs();
        if self.wedged {
            return write!(
                f,
                "the {} engine is WEDGED: `{}` has held it for {seconds}s, past its ceiling. An ONNX Runtime \
                 call cannot be cancelled from outside, so this request is refused instead of queueing behind \
                 it forever — see /health (status \"wedged\"), then POST /unload or restart the sidecar",
                self.what, self.activity
            );
        }
        write!(
            f,
            "the {} engine is busy (`{}`) and did not come free within {seconds}s — refusing rather than \
             queueing without a ceiling",
            self.what,
            if self.activity.is_empty() { "no activity recorded" } else { &self.activity }
        )
    }
}

impl std::error::Error for EngineWedged {}

/// How long a caller is willing to wait for an engine somebody else holds.
#[derive(Clone, Copy)]
enum Patience {
    /// The inference path: wait as long as the holder is legitimately alive, however long that is —
    /// a first-ever shape compile is minutes of CORRECT slowness, and failing a pass that would have
    /// succeeded is worse than waiting. This is the "documented pair" reliability.md allows: an
    /// unbounded-looking wait plus the detector that ends it the moment the holder stops being alive.
    UntilTheHolderIsWedged,
    /// /unload: answer the caller within a bound, whatever the holder is doing.
    AtMost(Duration),
}

/// What the watchdog does about a phase that has been in flight this long. Pure, so the policy is
/// testable without a clock, a GPU, or a process to kill.
#[derive(Debug, PartialEq, Eq)]
enum WedgeAction {
    /// Slow but alive.
    Nothing,
    /// Past its ceiling: /health says `wedged` and new requests are refused with the reason.
    Report,
    /// Past the OPT-IN exit ceiling on top of that: leave loudly, so the host's restart is the recovery.
    Exit,
}

/// The exit ceiling is measured FROM the wedge verdict, never from the phase start — otherwise a
/// `WEDGE_EXIT_AFTER_SECONDS` shorter than the (deliberately hour-long) build ceiling would kill the
/// process in the middle of a legitimate compile, which is the one action guaranteed to leave a
/// corrupt program in the compiled-model cache.
fn wedge_action(phase: Phase, elapsed: Duration, policy: WedgePolicy) -> WedgeAction {
    let ceiling = policy.ceiling(phase);
    if elapsed < ceiling {
        return WedgeAction::Nothing;
    }
    match policy.exit_after_wedged {
        Some(after) if elapsed >= ceiling + after => WedgeAction::Exit,
        _ => WedgeAction::Report,
    }
}

/// How often the watchdog looks. A wedge lasts forever by definition, so the tick only decides how
/// soon it reaches the log — never whether it does.
const WEDGE_WATCHDOG_TICK: Duration = Duration::from_secs(30);

/// The exit code a deliberate wedge exit leaves behind, distinct from the two startup preflights
/// (1, 2) so a supervisor's restart log says WHY this process left.
const WEDGE_EXIT_CODE: i32 = 3;

/// The staleness watchdog — the LOG half of the detector.
///
/// /health can only tell someone who asks, and by construction the party that would have asked (the
/// daemon, on an infinite sidecar HTTP timeout) is the one already blocked. So the wedge has to reach
/// the log on its own, once, with the phase and the elapsed time, or the incident is again only
/// visible to whoever thinks to poll a port.
fn spawn_wedge_watchdog(state: Arc<AppState>) {
    tokio::spawn(async move {
        let policy = state.config.wedge;
        // One announcement per wedge, not one per tick: an hour of a 30-second tick is 120 identical
        // ERROR lines, which is how a real incident gets scrolled past.
        let mut announced = [false, false];
        loop {
            tokio::time::sleep(WEDGE_WATCHDOG_TICK).await;
            let engines: [(&str, &Mutex<Option<InFlight>>); 2] = [
                ("embed", &state.engines.embed_inflight),
                ("rerank", &state.engines.rerank_inflight),
            ];
            for (at, (engine, slot)) in engines.into_iter().enumerate() {
                let Some(holder) = inflight_now(slot) else {
                    announced[at] = false;
                    continue;
                };
                let elapsed = holder.since.elapsed();
                match wedge_action(holder.phase, elapsed, policy) {
                    WedgeAction::Nothing => announced[at] = false,
                    WedgeAction::Report if announced[at] => {}
                    WedgeAction::Report => {
                        announced[at] = true;
                        tracing::error!(
                            "{engine} engine WEDGED: `{}` has held it for {}s, past the {}s ceiling for phase \
                             `{}`. An ONNX Runtime call cannot be cancelled from outside, so /health now reports \
                             status \"wedged\" and new requests are refused with this reason instead of queueing. \
                             Recovery: POST /unload, or restart the sidecar (WEDGE_EXIT=1 makes this process \
                             exit on its own).",
                            holder.label,
                            elapsed.as_secs(),
                            policy.ceiling(holder.phase).as_secs(),
                            holder.phase.name()
                        );
                    }
                    WedgeAction::Exit => {
                        tracing::error!(
                            "{engine} engine wedged for {}s in `{}` — WEDGE_EXIT is set, so this process is \
                             exiting with code {WEDGE_EXIT_CODE} for the host to restart it. Nothing inside this \
                             process can free a thread stuck in the ONNX Runtime C++ call.",
                            elapsed.as_secs(),
                            holder.label
                        );
                        std::process::exit(WEDGE_EXIT_CODE);
                    }
                }
            }
        }
    });
}

/// Engines that have been built, keyed by the sequence cap baked into them.
///
/// `max_length` is compiled into an ort session AND into the EP's program, so it used to be a reason to
/// EVICT: a cap change dropped both embedding engines and the next request rebuilt them. Measured
/// 2026-07-30, that cost 156-173 s per crossing — of which only ~13 s was session building, the rest
/// being MIGraphX materialising its ~2.4 GB compiled program at the FIRST `session.run`, which no
/// amount of eager loading can move. And a Fast pass crosses the boundary TWICE (it walks the ladder
/// down, then the next pass starts at the ceiling again), so the toll was ~5.5 min per pass, forever.
/// Keeping one engine per rung turns a crossing into a lookup.
///
/// An insertion-ordered `Vec` rather than a `HashMap`: the ladder has at most two rungs
/// (`SidecarRungPlan`), and the order IS the eviction policy — least-recently-used first, so a map
/// would need a second structure to carry it.
struct RungCache<T> {
    capacity: usize,
    rungs: Vec<(usize, T)>,
}

impl<T> RungCache<T> {
    /// `capacity` is the operator's `EMBED_ENGINE_CACHE_RUNGS`; 1 reproduces the pre-cache behaviour
    /// exactly (every cap change evicts), which is the escape hatch if the VRAM budget ever demands it.
    fn new(capacity: usize) -> Self {
        Self { capacity: capacity.max(1), rungs: Vec::new() }
    }

    /// The engine built for this cap, if any — marking it most-recently-used, so eviction always takes
    /// the rung the pass is LEAST likely to come back to.
    fn get_mut(&mut self, cap: usize) -> Option<&mut T> {
        let at = self.rungs.iter().position(|(resident, _)| *resident == cap)?;
        let entry = self.rungs.remove(at);
        self.rungs.push(entry);
        self.rungs.last_mut().map(|(_, engine)| engine)
    }

    /// Stores a freshly built engine as most-recently-used and returns whatever had to make room for
    /// it. The caller drops the evicted engine — ort session teardown is not instant and the caller
    /// already knows whether it is on a blocking thread.
    fn insert(&mut self, cap: usize, engine: T) -> Option<(usize, T)> {
        self.rungs.retain(|(resident, _)| *resident != cap);
        self.rungs.push((cap, engine));
        (self.rungs.len() > self.capacity).then(|| self.rungs.remove(0))
    }

    /// Every resident engine, emptying the cache — what `/unload` hands to the GPU lease.
    fn drain(&mut self) -> Vec<(usize, T)> {
        std::mem::take(&mut self.rungs)
    }

    /// ONE rung's engine, if resident — the partial `/unload`'s per-rung eviction (the host's
    /// budget-aware planner drops the largest unnecessary rung and keeps the rest warm).
    fn remove(&mut self, cap: usize) -> Option<T> {
        let at = self.rungs.iter().position(|(resident, _)| *resident == cap)?;
        Some(self.rungs.remove(at).1)
    }

    /// The caps currently resident, least-recently-used first. Reported by `/health` and logged on
    /// every build so the occupancy is observable rather than inferred.
    fn caps(&self) -> Vec<usize> {
        self.rungs.iter().map(|(cap, _)| *cap).collect()
    }
}

/// What an engine slot can be asked, independently of whether it holds one engine (rerank — a single
/// fixed cap) or one per rung (dense/sparse). Exists so `loaded_now` and `lock_healing` keep serving
/// both shapes without either growing a branch.
trait EngineSlot {
    /// Whether anything is loaded at all — `/health`'s per-model boolean.
    fn is_loaded(&self) -> bool;

    /// Drop everything, because a panic left state we cannot vouch for. See `lock_healing`.
    fn discard_all(&mut self);
}

impl<T> EngineSlot for Option<T> {
    fn is_loaded(&self) -> bool {
        self.is_some()
    }

    fn discard_all(&mut self) {
        *self = None;
    }
}

impl<T> EngineSlot for RungCache<T> {
    fn is_loaded(&self) -> bool {
        !self.rungs.is_empty()
    }

    fn discard_all(&mut self) {
        self.rungs.clear();
    }
}

/// Lazily-loaded model engines. Each is guarded by its own mutex: the GPU serializes inference
/// anyway, and the lock makes the first-use load race-free. Loads happen inside spawn_blocking.
/// BOTH embedding heads live in ONE `Bgem3DualEmbedding` per rung (see `RungCache` and
/// research/PLAN_bge_sidecar_unified_session.md — the official export returns both heads from one
/// forward pass, so two sessions doubled every cost for nothing); rerank runs at one fixed cap, so
/// it has nothing to key on.
struct Engines {
    embed: Mutex<RungCache<Bgem3DualEmbedding>>,
    /// What holds `embed` right now — read WITHOUT taking `embed`, which is the only way a wedged
    /// holder can ever be observed. See `InFlight`.
    embed_inflight: Mutex<Option<InFlight>>,
    rerank: Mutex<Option<TextRerank>>,
    /// The same, for the reranker: the two engines have separate mutexes and can be in flight at once,
    /// so one shared stamp would let a rerank overwrite the record of a wedged embed.
    rerank_inflight: Mutex<Option<InFlight>>,
}

impl Engines {
    fn new(cache_rungs: usize) -> Self {
        Self {
            embed: Mutex::new(RungCache::new(cache_rungs)),
            embed_inflight: Mutex::new(None),
            rerank: Mutex::new(None),
            rerank_inflight: Mutex::new(None),
        }
    }
}

struct AppState {
    config: Config,
    engines: Engines,
    /// What the sidecar is doing RIGHT NOW ("idle", "dense: building session…"), surfaced by
    /// /health — the operator's window into multi-minute engine builds that otherwise look like
    /// a hang from the outside.
    activity: Mutex<String>,
    /// The provider every engine PINS to, decided once (ORT_PROVIDER, else the first request's hint,
    /// else auto) and reused until restart. This is a REQUEST, not an outcome: it is set the moment the
    /// choice is made, before any session exists. Shape pinning keys off it — that decision has to be
    /// taken before the engines see any text, so it cannot wait for a session.
    pinned_provider: OnceLock<String>,
    /// The provider an ORT session was ACTUALLY created with — written only after a build SUCCEEDS.
    /// `None` while no session has been built, which is the state the old single field could not
    /// express: it reported the request as though it were the outcome, so a binary whose CUDA EP
    /// failed every registration still answered `provider: "cuda"` (measured 2026-08-08).
    active_provider: Mutex<Option<String>>,
    /// Why the last EP registration failed, verbatim from ort. Kept so /health can explain a
    /// `provider_ready: false` instead of merely asserting it.
    last_provider_error: Mutex<Option<String>>,
    /// The cap the MOST RECENTLY USED embedding pair was built with. Engines are kept per rung, so this
    /// no longer decides what gets evicted — it is what a `query` runs at (`cap_for`) and what /health
    /// reports as current. See `record_embed_max_length`.
    loaded_embed_max_length: Mutex<Option<usize>>,
    /// The batch the most recent embed actually ran at — the request's override, not the config default.
    /// Reported by /health as `loaded_max_batch`, which is what makes the configured value readable as a
    /// default rather than as a fact: every request carries the operator's own batch, so the configured
    /// number described an intention nobody was running.
    loaded_max_batch: Mutex<Option<usize>>,
    /// The DXGI adapter ORT_DEVICE_ID resolves to (None = mapping unavailable — raw id fallback).
    adapter: Option<adapters::ResolvedAdapter>,
    /// BGE-M3's own tokenizer, read from the model cache on first use purely to COUNT tokens for
    /// `TokenUsage` — it never feeds inference, which stays entirely fastembed's. `None` means the
    /// file was not found or would not parse, and every response then says `token_accounting: false`
    /// rather than quietly reporting zero truncations.
    token_counter: OnceLock<Option<tokenizers::Tokenizer>>,
    /// Qwen's tokenizer, for the SEMANTIC channel — which this sidecar never embeds. It lives here anyway
    /// because this is the only process that already speaks the HuggingFace `tokenizer.json` format
    /// natively: `Microsoft.ML.Tokenizers` cannot read that format at all (no regex pre-tokenizer, no NFC
    /// normalizer), so counting in C# would have meant transcribing Qwen's pre-tokenizer regex and its
    /// byte-level rules by hand — precisely the silent near-miss this accounting exists to prevent. Here
    /// the reference implementation reads the file verbatim. `None` when the file is absent, and
    /// `/tokenize` then says so instead of guessing.
    qwen_counter: OnceLock<Option<tokenizers::Tokenizer>>,
}

/// The per-request memory envelope: how long a single sequence may get and how many run together.
#[derive(Clone, Copy)]
struct Limits {
    max_length: usize,
    max_batch: usize,
}

impl Limits {
    /// A request may override either knob; 0/absent means "use the configured default". `max_length` is
    /// clamped to the model's own 8192 ceiling so a bad setting cannot ask for a shape it cannot run.
    fn resolve(config: &Config, max_length: usize, max_batch: usize) -> Self {
        Self {
            max_length: positive_or(max_length, config.embed_max_length).clamp(16, 8192),
            max_batch: positive_or(max_batch, config.max_batch).max(1),
        }
    }
}

fn positive_or(value: usize, fallback: usize) -> usize {
    if value == 0 {
        fallback
    } else {
        value
    }
}

impl AppState {
    /// The index the DirectML EP receives: the mapped plain-enumeration index when the DXGI
    /// resolution succeeded, else the raw configured id (the pre-mapping behaviour).
    fn dml_device_id(&self) -> i32 {
        self.adapter.as_ref().map_or(self.config.device_id, |a| a.dml_device_id)
    }

    /// The token counter, loaded once. Deliberately best-effort: a missing tokenizer must not stop a
    /// sidecar from embedding, it must only stop it from CLAIMING nothing was truncated.
    fn token_counter(&self) -> Option<&tokenizers::Tokenizer> {
        self.token_counter
            .get_or_init(|| load_token_counter(&self.config.cache_dir))
            .as_ref()
    }

    /// Qwen's counter, loaded once from `QWEN_TOKENIZER` (default `../qwen-tokenizer/tokenizer.json`,
    /// beside this tool). Same best-effort contract: absent means `/tokenize` reports it, never guesses.
    fn qwen_counter(&self) -> Option<&tokenizers::Tokenizer> {
        self.qwen_counter
            .get_or_init(|| load_named_tokenizer("qwen", &self.config.qwen_tokenizer_path))
            .as_ref()
    }
}

/// Loads a tokenizer from an explicit file path. Shared by every model this sidecar can COUNT for — which
/// is deliberately a wider set than the models it embeds: the semantic channel runs on Ollama, but nothing
/// on that side can read a HuggingFace `tokenizer.json`, so the count has to happen where the reference
/// implementation already lives.
fn load_named_tokenizer(name: &str, path: &Path) -> Option<tokenizers::Tokenizer> {
    if !path.is_file() {
        tracing::warn!(
            "no {name} tokenizer at `{}` — /tokenize will report it unavailable for that model rather \
             than estimate a count",
            path.display()
        );
        return None;
    }
    match tokenizers::Tokenizer::from_file(path) {
        Ok(t) => {
            tracing::info!("{name} token counting enabled from `{}`", path.display());
            Some(t)
        }
        Err(e) => {
            tracing::warn!("{name} tokenizer at `{}` would not parse ({e}) — counting is off", path.display());
            None
        }
    }
}

/// Finds `tokenizer.json` for BGE-M3 inside the HuggingFace-layout model cache. The snapshot folder is
/// a content hash that changes when the model is re-pulled, so it is discovered rather than hardcoded.
fn find_tokenizer_file(cache_dir: &Path) -> Option<PathBuf> {
    let snapshots = cache_dir.join("models--BAAI--bge-m3").join("snapshots");
    let entries = std::fs::read_dir(snapshots).ok()?;
    entries
        .filter_map(|e| e.ok())
        .map(|e| e.path().join("tokenizer.json"))
        .find(|p| p.is_file())
}

fn load_token_counter(cache_dir: &Path) -> Option<tokenizers::Tokenizer> {
    let Some(path) = find_tokenizer_file(cache_dir) else {
        tracing::warn!(
            "no BGE-M3 tokenizer.json under `{}` — /embed will report token_accounting: false, and the \
             host cannot then prove that no input was silently truncated",
            cache_dir.display()
        );
        return None;
    };
    match tokenizers::Tokenizer::from_file(&path) {
        Ok(tokenizer) => {
            tracing::info!("token accounting enabled from `{}`", path.display());
            Some(tokenizer)
        }
        Err(e) => {
            tracing::warn!("BGE-M3 tokenizer at `{}` would not parse ({e}) — token accounting is off", path.display());
            None
        }
    }
}

/// Counts what each text really costs and flags the ones whose tail the cap will discard.
///
/// The count is taken with truncation OFF on purpose: a truncating tokenizer reports `max_length` for
/// every over-long text, which is precisely the information that hides the problem. Special tokens are
/// included because they occupy the same window the content competes for.
fn token_usage(state: &AppState, texts: &[String], max_length: usize) -> TokenUsage {
    let Some(tokenizer) = state.token_counter() else {
        return TokenUsage { max_length, ..TokenUsage::default() };
    };
    usage_from_counts(count_tokens(tokenizer, texts, "embed"), max_length)
}

/// Counts each text, keeping a REFUSAL as `None` instead of folding it to `0` — and saying so in the
/// log.
///
/// The fold used to be `.map(|e| e.len()).unwrap_or(0)`, silently, in both places that count. A text
/// the tokenizer refused was then reported as 0 tokens and `truncated: false` — "measured, and
/// definitely not truncated", which is the exact inversion of the only guarantee this accounting
/// exists to give. The host owns no tokenizer, so it has no second opinion to check that against.
fn count_tokens(tokenizer: &tokenizers::Tokenizer, texts: &[String], what: &str) -> Vec<Option<usize>> {
    texts
        .iter()
        .map(|text| match tokenizer.encode(text.as_str(), true) {
            Ok(encoded) => Some(encoded.len()),
            Err(e) => {
                tracing::warn!(
                    "{what}: the tokenizer refused a text of {} char(s) ({e}) — reporting the count as UNKNOWN \
                     rather than as zero, which a caller reads as proof that nothing was truncated",
                    text.len()
                );
                None
            }
        })
        .collect()
}

/// Folds per-text counts into the wire shape, keeping UNKNOWN distinct from CLEAN.
///
/// One refusal turns accounting off for the WHOLE response, because that is the distinction this file
/// already models one level up (`token_accounting: false` = NOT MEASURED, both arrays empty) and the
/// only one the host understands. A per-text hole would have to be invented on the wire, and every
/// caller that did not learn about it would read the hole as a zero — the defect again, with an extra
/// field. Conservative in the right direction: the batch reads as unmeasured, never as proven clean.
fn usage_from_counts(counts: Vec<Option<usize>>, max_length: usize) -> TokenUsage {
    let Some(measured) = counts.into_iter().collect::<Option<Vec<usize>>>() else {
        return TokenUsage { max_length, ..TokenUsage::default() };
    };
    TokenUsage {
        truncated: measured.iter().map(|&n| n > max_length).collect(),
        token_count: measured,
        max_length,
        token_accounting: true,
    }
}

#[tokio::main]
async fn main() {
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
    let (day, clock) = day_and_clock(now);
    let log_dir = format!("{}/{day}", env_str("SIDECAR_LOG_DIR", "logs"));
    let log_path = format!(
        "{log_dir}/bge-sidecar-device{}-{clock}-{}.log",
        env_parse::<i32>("ORT_DEVICE_ID", 0),
        std::process::id()
    );
    // Best-effort: an unwritable directory must never keep the sidecar from starting.
    let log_file = std::fs::create_dir_all(&log_dir)
        .ok()
        .and_then(|_| std::fs::OpenOptions::new().create(true).append(true).open(&log_path).ok());
    let file_layer = log_file.map(|file| {
        tracing_subscriber::fmt::layer()
            .with_ansi(false)
            .with_writer(std::sync::Mutex::new(file))
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
    let state = Arc::new(AppState {
        engines: Engines::new(config.engine_cache_rungs),
        config,
        activity: Mutex::new("idle".to_string()),
        pinned_provider: OnceLock::new(),
        active_provider: Mutex::new(None),
        last_provider_error: Mutex::new(None),
        loaded_embed_max_length: Mutex::new(None),
        loaded_max_batch: Mutex::new(None),
        adapter,
        token_counter: OnceLock::new(),
        qwen_counter: OnceLock::new(),
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

    let app = Router::new()
        .route("/health", get(health))
        .route("/embed", post(embed))
        .route("/rerank", post(rerank))
        .route("/unload", post(unload))
        .route("/tokenize", post(tokenize))
        .with_state(state);

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

// ---------- ONNX Runtime dylib preflight (load-dynamic flavor) ----------

/// Fail-fast guard for the load-dynamic (migraphx) flavor. ort rc.12's own version check DEADLOCKS
/// instead of erroring when ORT_DYLIB_PATH points at an ONNX Runtime older than the crate requires:
/// building the "not compatible" error calls `ortsys![CreateStatus]` → `ort::api()` → re-enters the
/// same `G_ORT_API` OnceLock the thread is already initializing → futex wait forever, while holding
/// our engine mutex — /health and every /embed then hang too. Probe the dylib up front through the
/// stable C ABI and exit with an actionable message before any model load can freeze the process.
#[cfg(feature = "migraphx")]
fn preflight_ort_dylib() {
    let path = env_str("ORT_DYLIB_PATH", "libonnxruntime.so");
    match probe_ort_dylib(&path).and_then(|(api_ok, version)| dylib_verdict(api_ok, &version, &path)) {
        Ok(message) => tracing::info!("{message}"),
        Err(message) => {
            tracing::error!("{message}");
            std::process::exit(1);
        }
    }
}

/// Second fail-fast guard for the load-dynamic (migraphx) flavor. ROCm 7.x's MIGraphX EP ALWAYS
/// writes the compiled model to its cache path; when that path is unset it saves to `""`, the write
/// fails and takes the kernel call down with it, so every /embed answered 500 after a ~2-minute
/// compile while the GPU sat idle. An unwritable path is a machine-config error the operator must
/// see once at startup, not on every request.
#[cfg(feature = "migraphx")]
fn preflight_migraphx_cache() {
    match cache_dir_verdict(&env_str("ORT_MIGRAPHX_MODEL_CACHE_PATH", ""), |dir| {
        std::fs::create_dir_all(dir).and_then(|()| {
            let probe = std::path::Path::new(dir).join(".bge-sidecar-write-probe");
            std::fs::write(&probe, b"probe").map(|()| std::fs::remove_file(&probe).ok().unwrap_or(()))
        })
        .map_err(|e| e.to_string())
    }) {
        Ok(message) => tracing::info!("{message}"),
        Err(message) => {
            tracing::error!("{message}");
            std::process::exit(1);
        }
    }
}

/// Seeds the model cache from `MODEL_CACHE_SEED_DIR` when that env var is set (the AppHost's WSL
/// launch line points it at the repo's `.model-cache` on /mnt).
///
/// Why: under WSL the repo lives on DrvFs, which reads the 2.27 GB ONNX weights at ~123 MB/s —
/// ~19 s of EVERY session build (measured 2026-07-28). `MODEL_CACHE_DIR` therefore points at ext4
/// (~NVMe speed), and this startup step keeps that ext4 copy in sync automatically, so no machine
/// ever needs a manual copy. Idempotent (same-size files are skipped — the models are immutable HF
/// blobs) and best-effort: any failure still starts the sidecar, and fastembed then reads whatever
/// the cache holds or downloads from HF exactly as before.
fn seed_model_cache_from_env(cache_dir: &std::path::Path) {
    let seed = env_str("MODEL_CACHE_SEED_DIR", "");
    if seed.trim().is_empty() {
        return; // Windows flavor / manual runs: the cache dir is used as-is.
    }

    let seed_path = std::path::PathBuf::from(&seed);
    if !seed_path.is_dir() {
        tracing::info!("model-cache seed dir `{seed}` does not exist — models will download on first use");
        return;
    }

    let started = std::time::Instant::now();
    let copied = copy_missing_files(&seed_path, cache_dir);
    if copied.files > 0 {
        tracing::info!(
            "model cache seeded: {} file(s), {} MB from `{seed}` -> `{}` in {:.1}s (one-time; later starts verify and skip)",
            copied.files,
            copied.bytes / (1024 * 1024),
            cache_dir.display(),
            started.elapsed().as_secs_f32()
        );
    } else {
        tracing::info!("model cache already seeded: `{}` matches `{seed}`", cache_dir.display());
    }
}

/// What one seeding pass actually moved.
#[derive(Default)]
struct SeededFiles {
    files: u64,
    bytes: u64,
}

/// Recursively copies every file under `from` that is missing (or size-mismatched — an interrupted
/// earlier copy) under `to`. Never deletes anything, never overwrites a same-size file, and treats
/// every per-file error as a warning rather than a failure — a half-seeded cache still works, because
/// fastembed falls back to downloading whatever is unreadable.
fn copy_missing_files(from: &std::path::Path, to: &std::path::Path) -> SeededFiles {
    let mut seeded = SeededFiles::default();
    let entries = match std::fs::read_dir(from) {
        Ok(entries) => entries,
        Err(e) => {
            tracing::warn!("model-cache seed: cannot read `{}`: {e}", from.display());
            return seeded;
        }
    };

    for entry in entries.flatten() {
        let source = entry.path();
        let target = to.join(entry.file_name());
        if source.is_dir() {
            let nested = copy_missing_files(&source, &target);
            seeded.files += nested.files;
            seeded.bytes += nested.bytes;
            continue;
        }

        let source_size = source.metadata().map(|m| m.len()).unwrap_or(0);
        let up_to_date = target.metadata().map(|m| m.len() == source_size).unwrap_or(false);
        if up_to_date {
            continue;
        }

        let copied = std::fs::create_dir_all(to)
            .map_err(|e| e.to_string())
            .and_then(|()| std::fs::copy(&source, &target).map_err(|e| e.to_string()));
        match copied {
            Ok(bytes) => {
                seeded.files += 1;
                seeded.bytes += bytes;
            }
            Err(e) => tracing::warn!("model-cache seed: `{}` -> `{}` failed: {e}", source.display(), target.display()),
        }
    }

    seeded
}

/// Pure verdict for the cache preflight — the writability probe is injected so the message contract
/// is testable without touching the filesystem.
#[cfg_attr(not(feature = "migraphx"), allow(dead_code))]
fn cache_dir_verdict(dir: &str, probe: impl FnOnce(&str) -> Result<(), String>) -> Result<String, String> {
    if dir.trim().is_empty() {
        return Err("ORT_MIGRAPHX_MODEL_CACHE_PATH is not set. ROCm's MIGraphX EP always saves the \
                    compiled model, and with no path it writes to \"\" — the write fails and fails \
                    every /embed with it. Point it at a writable Linux directory (the AppHost sets \
                    Aspire:BgeSidecar:WslMigraphxCacheDir; see README \"AMD on Linux/WSL\")."
            .to_string());
    }
    probe(dir)
        .map(|()| format!("MIGraphX compiled-model cache is writable: {dir} (first call per input shape compiles and saves, later ones load)"))
        .map_err(|e| format!(
            "MIGraphX compiled-model cache `{dir}` is not writable: {e}. The EP saves every compiled \
             model there, so an unwritable path fails every /embed. Fix the path or its permissions."
        ))
}

/// Head of ONNX Runtime's stable C ABI vtable (`OrtApiBase` in onnxruntime_c_api.h): only the two
/// members the preflight needs, in their fixed order.
#[cfg(feature = "migraphx")]
#[repr(C)]
struct OrtApiBaseAbi {
    get_api: unsafe extern "C" fn(u32) -> *const std::ffi::c_void,
    get_version_string: unsafe extern "C" fn() -> *const std::ffi::c_char,
}

/// Loads the dylib and asks it for (does it serve our API version, its version string). The dlopen
/// here is refcounted — ort's own later load reuses the mapping, so the probe costs nothing extra.
#[cfg(feature = "migraphx")]
fn probe_ort_dylib(path: &str) -> Result<(bool, String), String> {
    let lib = unsafe { libloading::Library::new(path) }
        .map_err(|e| format!("cannot load ONNX Runtime dylib `{path}`: {e}"))?;
    let get_base: libloading::Symbol<unsafe extern "C" fn() -> *const OrtApiBaseAbi> =
        unsafe { lib.get(b"OrtGetApiBase") }
            .map_err(|_| format!("`{path}` exports no OrtGetApiBase — not an ONNX Runtime library"))?;
    let base = unsafe { get_base() };
    if base.is_null() {
        return Err(format!("OrtGetApiBase in `{path}` returned null"));
    }
    let version = unsafe { std::ffi::CStr::from_ptr(((*base).get_version_string)()) }
        .to_string_lossy()
        .into_owned();
    let api_ok = !unsafe { ((*base).get_api)(ort::MINOR_VERSION) }.is_null();
    Ok((api_ok, version))
}

/// Pure verdict for the preflight, split out so the message contract is testable without a dylib:
/// `api_ok` is whether `GetApi(ORT_API_VERSION)` returned a vtable (ORT serves all older API
/// versions too, so a NEWER dylib passes; only an older one fails).
#[cfg_attr(not(feature = "migraphx"), allow(dead_code))]
fn dylib_verdict(api_ok: bool, version: &str, path: &str) -> Result<String, String> {
    if api_ok {
        return Ok(format!(
            "ONNX Runtime dylib preflight OK: `{path}` is version {version} (serves API v{})",
            ort::MINOR_VERSION
        ));
    }
    Err(format!(
        "ONNX Runtime at `{path}` is version {version}, which cannot serve API v{minor} required by this \
         build — ort needs ONNX Runtime >= 1.{minor}. Rebuild it from tag v1.{minor}.x with --use_migraphx \
         and reinstall (README \"AMD on Linux/WSL\"), then restart the sidecar.",
        minor = ort::MINOR_VERSION
    ))
}

// ---------- wire types ----------

#[derive(Deserialize)]
struct EmbedRequest {
    texts: Vec<String>,
    /// "doc" | "query". BGE-M3 is symmetric, so both embed identically — but a QUERY is also never allowed to
    /// move the loaded sequence cap (see `cap_for`), because it arrives interleaved with index passes.
    #[serde(default)]
    kind: String,
    /// Optional provider hint forwarded from the operator's DB setting (used only before first load).
    #[serde(default)]
    provider: String,
    /// Optional per-request token cap (operator setting). 0/absent = the configured default. Changing it
    /// evicts and rebuilds the engines, so the operator sees the new memory envelope without a restart.
    #[serde(default)]
    max_length: usize,
    /// Optional per-request batch size (operator setting). 0/absent = the configured default.
    #[serde(default)]
    max_batch: usize,
    /// Optional caller correlation id: echoed verbatim in the response and prefixed to this request's
    /// pass log lines. Opaque here — without it, two concurrent requests are indistinguishable in
    /// either place.
    #[serde(default)]
    request_id: String,
}

#[derive(Serialize)]
struct SparseVec {
    indices: Vec<u32>,
    values: Vec<f32>,
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
struct TokenUsage {
    /// Tokens each input text costs, special tokens included, measured BEFORE the cap is applied — so a
    /// value above `max_length` is exactly the overflow that was thrown away.
    token_count: Vec<usize>,
    /// Per text: the model saw a PREFIX of it. Parallel to the request's `texts`.
    truncated: Vec<bool>,
    /// The EFFECTIVE cap those were judged against (after `cap_for`), not the requested one.
    max_length: usize,
    /// False when the tokenizer could not be loaded at all. The two vectors are then EMPTY, and the
    /// caller must treat "no truncation reported" as UNKNOWN rather than as proof of none — which is
    /// the whole difference between a guard and a decoration.
    token_accounting: bool,
}

/// Where a request's wall-clock went inside the sidecar, on the wire so the CALLER can attribute it.
///
/// Every number here used to die in this process's own log file, and the worst one was never measured
/// at all: the pass timer starts only after the engine mutex is held, so a request that waited 8 s
/// behind another caller's pass and then ran 0.4 s looked, to its caller, like a slow model. Queue
/// wait and session build stay separate fields — both are infrastructure wait, but the remedies
/// differ (concurrency vs warm-up), and a bucket that mixes two causes explains neither.
#[derive(Serialize, Default, Clone, Copy)]
struct PassTimings {
    /// Waiting for the engine mutex behind another request — infrastructure wait, never model speed.
    queue_wait_ms: u64,
    /// Building + canary-checking the session; 0 on a warm engine.
    session_build_ms: u64,
    /// The forward pass(es), settling re-runs included — what this request's inference actually cost.
    inference_ms: u64,
    /// >0 = MIGraphX compiled this input shape during the pass. The EP saves its cache LAZILY, so
    /// growth measured across the pass is the only moment a compile is observable.
    compile_cache_grew_mb: u64,
}

#[derive(Serialize)]
struct EmbedResponse {
    dense: Vec<Vec<f32>>,
    sparse: Vec<SparseVec>,
    #[serde(flatten)]
    usage: TokenUsage,
    /// Echoed from the request; empty when the caller sent none.
    request_id: String,
    timings: PassTimings,
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
struct TokenizeRequest {
    texts: Vec<String>,
    /// `"bge"` (default) or `"qwen"`. Unknown names are refused rather than silently served by the wrong
    /// tokenizer — a count from the wrong model is worse than no count.
    #[serde(default)]
    model: String,
}

#[derive(Serialize)]
struct TokenizeResponse {
    /// Tokens each text costs, special tokens included, with NO truncation applied — the caller is asking
    /// precisely so it can split BEFORE anything is capped.
    token_count: Vec<usize>,
    /// Which tokenizer answered.
    model: String,
    /// False when that tokenizer is not loadable here; `token_count` is then empty and the caller must
    /// treat the size as unknown rather than as zero.
    available: bool,
}

#[derive(Deserialize)]
struct RerankRequest {
    query: String,
    documents: Vec<String>,
    #[serde(default)]
    provider: String,
    /// Sequences per forward pass, forwarded by the host exactly as `/embed` does. 0/absent keeps the
    /// configured default. Before this existed the handler read `config.max_batch` — and the AppHost passes
    /// MAX_BATCH as an empty string, so every rerank ran at the env default of 4 while the operator's stored
    /// envelope said 64: scoring 50 candidates cost 13 forward passes instead of one.
    #[serde(default)]
    max_batch: usize,
    /// Optional caller correlation id, exactly as on `EmbedRequest`.
    #[serde(default)]
    request_id: String,
}

#[derive(Serialize)]
struct RerankResponse {
    scores: Vec<f32>,
    /// Echoed from the request; empty when the caller sent none.
    request_id: String,
    timings: PassTimings,
}

#[derive(Serialize)]
struct HealthResponse {
    /// `"ok"`, or `"wedged"` once an engine has been held past its phase's ceiling. Deliberately not a
    /// constant: a health endpoint that always says "ok" cannot report the one failure that matters
    /// here (reliability.md § "Health endpoints tell the truth and never block").
    status: &'static str,
    /// What the sidecar is doing right now: "idle", "dense: building session…", "sparse:
    /// embedding N row(s)". Lets the host UI show "compiling models" instead of a dead card
    /// during a multi-minute MIGraphX first build.
    activity: String,
    /// LEGACY, kept so existing callers (the C# host) do not break: the active provider once one
    /// exists, otherwise the requested one. Ambiguous by construction — read the four fields below
    /// instead, which is why they exist.
    provider: String,
    /// What was ASKED for: ORT_PROVIDER, else the first request's hint, else "auto". Says nothing
    /// about whether it works.
    requested_provider: String,
    /// The EPs this binary was COMPILED with. A provider absent here can never become active, however
    /// it is configured — the failure is in the build flavor, not the settings.
    compiled_providers: Vec<&'static str>,
    /// The provider of a successfully created ORT session. `null` until one exists.
    active_provider: Option<String>,
    /// A session has been created on `active_provider`. False means no inference has ever
    /// succeeded on this process, whatever `requested_provider` says.
    provider_ready: bool,
    /// The last EP registration failure, verbatim. `null` when none was recorded.
    last_provider_error: Option<String>,
    /// SHA-256 of the executable serving this response. Empty when `current_exe()` is unreadable — or
    /// when `provenance_ready` is false and it has simply not been computed yet. A benchmark records it
    /// so a later run can prove it measured the same binary; an installed sidecar older than its commit
    /// is invisible to every other field here.
    exe_sha256: String,
    /// SHA-256 over the sorted `name:sha256` manifest of the dynamic libraries beside the executable.
    /// Empty when none were found (or not yet computed — see `provenance_ready`). Identical executables
    /// with different provider libraries are different runtimes, and this is the field that says so.
    runtime_manifest_sha256: String,
    /// The two hashes above are FINAL. False = the startup task is still hashing, so an empty hash means
    /// "not yet", not "unreadable". They are computed on the blocking pool at startup precisely because
    /// the first /health used to compute them inline — 1.4 s over 67 MB of test binaries, and a CUDA
    /// deployment is gigabytes (cuDNN alone often >500 MB) — on a Tokio reactor thread.
    provenance_ready: bool,
    /// The engine work in flight right now, one entry per engine that is held. THE window into the one
    /// wait this process cannot cancel: without it a wedged inference and a healthy multi-minute build
    /// looked identical from outside.
    in_flight: Vec<InFlightWire>,
    /// Any engine past its ceiling. The single boolean a host can route on.
    wedged: bool,
    loaded: LoadedModels,
    models: ModelNames,
    /// The memory envelope in force — the defaults plus the cap the loaded engines actually carry, so the
    /// host can show what is running rather than what was requested.
    limits: LimitsWire,
    /// The DXGI adapter this sidecar's DirectML EP targets — the ground truth the host UI labels
    /// devices with. Null = mapping unavailable (non-DML build / DXGI failure / id out of range).
    adapter: Option<adapters::ResolvedAdapter>,
}

#[derive(Serialize)]
struct LoadedModels {
    dense: bool,
    sparse: bool,
    rerank: bool,
}

/// One held engine, as /health reports it.
#[derive(Serialize)]
struct InFlightWire {
    /// `"embed"` | `"rerank"`.
    engine: &'static str,
    /// `"building"` | `"running"` — the ceilings differ by an order of magnitude, so the phase has to
    /// travel with the elapsed time or the number cannot be judged.
    phase: &'static str,
    /// What the holder said it was doing, verbatim from `activity`.
    activity: String,
    elapsed_seconds: u64,
    /// The ceiling this phase is judged against, so a reader needs no access to the configuration.
    ceiling_seconds: u64,
    /// Past that ceiling: no longer "slow but alive". A first-ever MIGraphX shape compile is minutes of
    /// CORRECT slowness and must never read as this — which is why `building` gets an hour.
    wedged: bool,
}

impl InFlightWire {
    fn of(engine: &'static str, holder: &InFlight, policy: WedgePolicy) -> Self {
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
fn in_flight_now(state: &AppState) -> Vec<InFlightWire> {
    [("embed", &state.engines.embed_inflight), ("rerank", &state.engines.rerank_inflight)]
        .into_iter()
        .filter_map(|(engine, slot)| inflight_now(slot).map(|holder| InFlightWire::of(engine, &holder, state.config.wedge)))
        .collect()
}

#[derive(Serialize)]
struct LimitsWire {
    embed_max_length: usize,
    /// The CONFIGURED default — what a request that carries no batch of its own falls back to. It is NOT
    /// what the last embed ran at: every request carries the operator's own batch, so this field described
    /// an intention and was read as a fact ("why 15 methods/s when the batch is 126?" — three different
    /// numbers, none of them this one). Read `loaded_max_batch` for what actually happened.
    max_batch: usize,
    rerank_max_length: usize,
    /// None = no embedding engine has been built yet, so no cap is committed to.
    loaded_embed_max_length: Option<usize>,
    /// The batch the most recent embed ACTUALLY ran at, request override included. `None` until one has
    /// run. The twin of `loaded_embed_max_length`, and the field whose absence made the configured default
    /// above look authoritative — the same requested-vs-active split the provider fields already carry.
    loaded_max_batch: Option<usize>,
    /// EVERY cap whose engines are currently resident, least-recently-used first — the cache's
    /// occupancy, which `loaded_embed_max_length` (the current rung alone) cannot show. Empty also
    /// when the probe found the cache busy: /health must never queue behind model work.
    resident_embed_max_lengths: Vec<usize>,
}

#[derive(Serialize)]
struct ModelNames {
    dense: &'static str,
    sparse: &'static str,
    rerank: &'static str,
}

#[derive(Serialize)]
struct ErrorResponse {
    error: String,
}

type ApiError = (StatusCode, Json<ErrorResponse>);

fn internal_error(err: anyhow::Error) -> ApiError {
    tracing::error!("request failed: {err:#}");
    (StatusCode::INTERNAL_SERVER_ERROR, Json(ErrorResponse { error: format!("{err:#}") }))
}

/// A caller mistake, not a sidecar failure — refused with the reason so it is fixable from the message.
fn bad_request(error: String) -> ApiError {
    (StatusCode::BAD_REQUEST, Json(ErrorResponse { error }))
}

/// Maps an inference failure to its status code. A wedged or busy engine is **503**, not 500: nothing
/// is wrong with the request, the card is unavailable RIGHT NOW, and a host that degrades or retries on
/// 503 while treating 500 as a hard failure can only act on the difference if we make it.
fn engine_error(err: anyhow::Error) -> ApiError {
    if err.downcast_ref::<EngineWedged>().is_some() {
        tracing::warn!("refusing a request: {err:#}");
        return (StatusCode::SERVICE_UNAVAILABLE, Json(ErrorResponse { error: format!("{err:#}") }));
    }
    internal_error(err)
}

/// The message out of a `spawn_blocking` panic — the payload is the only place the actual reason
/// (a failed expect deep in ort/fastembed, an allocation failure) survives, and it belongs in the
/// operator's log instead of an opaque "task panicked".
fn join_error_text(e: tokio::task::JoinError) -> String {
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

// ---------- handlers ----------

async fn health(State(state): State<Arc<AppState>>) -> Json<HealthResponse> {
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
        },
        adapter: state.adapter.clone(),
    })
}

/// Non-blocking engine presence for /health: a busy lock means a load or an inference pass holds
/// the engine RIGHT NOW, so report presence instead of queueing the probe behind model work — a
/// hung first load (the ort load-dynamic version-mismatch deadlock) once froze /health forever
/// behind this lock. A poisoned lock (a panicked load) counts as "nothing loaded".
fn loaded_now<S: EngineSlot>(engine: &Mutex<S>) -> bool {
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
struct UnloadRequest {
    /// Sequence caps whose embed engines to drop. Empty = not specified.
    #[serde(default)]
    embed_max_lengths: Vec<usize>,
    /// Drop the reranker too. None = not specified.
    #[serde(default)]
    rerank: Option<bool>,
}

impl UnloadRequest {
    fn is_full_drain(&self) -> bool {
        self.embed_max_lengths.is_empty() && self.rerank.is_none()
    }
}

/// A malformed body drops NOTHING (and logs) — a typo'd partial request silently becoming a full
/// drain would evict engines the caller explicitly asked to keep. `None` = "do not touch anything".
fn parse_unload_request(body: &[u8]) -> Option<UnloadRequest> {
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
async fn unload(State(state): State<Arc<AppState>>, body: Bytes) -> Json<HealthResponse> {
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
struct Drained {
    embed_rungs: Vec<usize>,
    rerank: bool,
    /// Engines /unload could NOT take before its ceiling. Reported so a caller can never read a partial
    /// handover as a complete one.
    refused: Vec<&'static str>,
}

/// Takes the named engines out under their locks and drops them OUTSIDE those locks.
///
/// Runs entirely on the blocking pool (see `unload`): ort session teardown is not instant, and both the
/// wait and the drop would otherwise sit on a reactor thread. The drop-outside-the-lock shape is the
/// pre-existing correct design and is preserved deliberately — holding an engine mutex through a
/// teardown would block the very /health that reports the handover.
fn drain_engines(state: &AppState, req: &UnloadRequest) -> Drained {
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

async fn embed(
    State(state): State<Arc<AppState>>,
    Json(req): Json<EmbedRequest>,
) -> Result<Json<EmbedResponse>, ApiError> {
    let EmbedRequest { texts, kind, provider, max_length, max_batch, request_id } = req;
    if texts.is_empty() {
        return Ok(Json(EmbedResponse {
            dense: vec![],
            sparse: vec![],
            usage: TokenUsage::default(),
            request_id,
            timings: PassTimings::default(),
        }));
    }

    let limits = Limits::resolve(&state.config, max_length, max_batch);
    let shared = state.clone();
    let outcome = tokio::task::spawn_blocking(move || {
        embed_blocking(&state, texts, &kind, &provider, limits, &request_id)
    })
    .await;
    set_activity(&shared, "idle");
    let result = outcome
        .map_err(|e| internal_error(anyhow::anyhow!("embed task panicked: {}", join_error_text(e))))?
        .map_err(engine_error)?;
    Ok(Json(result))
}

/// Pure CPU: a vocabulary lookup and BPE merges, no session, no GPU, no model weights. That is what makes
/// it safe to call from an index pass — it never queues behind the card.
async fn tokenize(
    State(state): State<Arc<AppState>>,
    Json(req): Json<TokenizeRequest>,
) -> Result<Json<TokenizeResponse>, ApiError> {
    let model = if req.model.trim().is_empty() { "bge" } else { req.model.trim() };
    let counter = match model {
        "bge" => state.token_counter(),
        "qwen" => state.qwen_counter(),
        other => {
            return Err(bad_request(format!(
                "unknown tokenizer '{other}' — this sidecar counts for 'bge' and 'qwen'. Serving the \
                 request with the wrong tokenizer would answer confidently and wrongly."
            )))
        }
    };

    let Some(tokenizer) = counter else {
        return Ok(Json(TokenizeResponse { token_count: vec![], model: model.to_string(), available: false }));
    };

    // Same rule as /embed's accounting: one refusal makes the whole answer UNKNOWN rather than zero.
    // A caller asks here precisely so it can split BEFORE anything is capped, and a `0` it cannot tell
    // from an empty text is worse than an honest "not measured".
    let Some(token_count) = count_tokens(tokenizer, &req.texts, "tokenize").into_iter().collect::<Option<Vec<usize>>>()
    else {
        return Ok(Json(TokenizeResponse { token_count: vec![], model: model.to_string(), available: false }));
    };
    Ok(Json(TokenizeResponse { token_count, model: model.to_string(), available: true }))
}

async fn rerank(
    State(state): State<Arc<AppState>>,
    Json(req): Json<RerankRequest>,
) -> Result<Json<RerankResponse>, ApiError> {
    let RerankRequest { query, documents, provider, max_batch, request_id } = req;
    if documents.is_empty() {
        return Ok(Json(RerankResponse {
            scores: vec![],
            request_id,
            timings: PassTimings::default(),
        }));
    }

    let max_batch = rerank_batch(&state.config, max_batch);
    let shared = state.clone();
    let outcome = tokio::task::spawn_blocking(move || {
        rerank_blocking(&state, query, documents, &provider, max_batch, &request_id)
    })
    .await;
    set_activity(&shared, "idle");
    let result = outcome
        .map_err(|e| internal_error(anyhow::anyhow!("rerank task panicked: {}", join_error_text(e))))?
        .map_err(engine_error)?;
    Ok(Json(result))
}

// ---------- blocking inference ----------

/// The batch a rerank call runs at: the request's value when it carries one, else the configured default.
/// Deliberately mirrors `Limits::resolve` for the embed path — a rerank that silently ignored the operator's
/// envelope is exactly the defect this exists to stop, and the two resolutions must not drift apart.
fn rerank_batch(config: &Config, requested: usize) -> usize {
    positive_or(requested, config.max_batch).max(1)
}

/// Whether this provider recompiles the graph for every distinct input shape. MIGraphX does — one
/// compile is ~2-4 minutes AND ~2.5 GB of on-disk cache per shape, measured on an R9700 — so a run
/// whose batches vary in length spends nearly all its time compiling. CUDA/DirectML/CPU take dynamic
/// shapes in stride and must not pay the padding overhead below.
fn should_pin_shape(setting: &str, provider: &str) -> bool {
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
fn ruler_text() -> &'static str {
    static RULER: OnceLock<String> = OnceLock::new();
    RULER.get_or_init(|| "lorem ipsum dolor sit amet ".repeat(4096)).as_str()
}

/// How many runs of the SAME real batch a request gets before it fails. Measured behaviour is that
/// exactly ONE run after an engine build is bad, so the second settles it; the bound exists because
/// "exactly one" is an observation, not a guarantee, and a genuinely broken engine must fail the
/// request instead of spinning here.
const SETTLE_ATTEMPTS: usize = 3;

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
fn embed_settling<T>(
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
fn pin_shape(texts: &[String], max_batch: usize, ruler: &str) -> (Vec<String>, Vec<usize>) {
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
fn unpin_rows<T>(mut rows: Vec<T>, positions: &[usize]) -> anyhow::Result<Vec<T>> {
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
fn cap_for(kind: &str, requested: usize, loaded: Option<usize>) -> usize {
    match (kind, loaded) {
        ("query", Some(resident)) => resident,
        _ => requested,
    }
}

fn embed_blocking(
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
        return Ok(EmbedResponse {
            dense: unpin_rows(padded.dense, &positions)?,
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
fn embed_natural(
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
    if guard.get_mut(limits.max_length).is_none() {
        stamp.enter(Phase::Building, "embed: building and canary-checking the session (a first-ever shape compiles for minutes; cached shapes load in seconds)");
        let building = Instant::now();
        let built = load_validated_dual(state, provider_hint, limits, retry_short)?;
        remember_engine(&mut guard, "embed", limits.max_length, built);
        session_build_ms = building.elapsed().as_millis() as u64;
    }
    stamp.enter(Phase::Running, format!("embed: embedding {} row(s)", texts.len()));
    // The duration below includes any settling re-runs — that is honest: it is what the caller waited.
    let (cache_before, pass) = (mxr_cache_mb(&state.config.mxr_cache_base), Instant::now());
    let engine = guard.get_mut(limits.max_length).expect("just loaded");
    // Rows are (dense, sparse) ZIPPED per text, so the settling retry polices one length and a short
    // first run can never shorten one head without the other.
    let rows = embed_settling("embed", texts.len(), retry_short, || engine.embed(texts.clone(), batch))?;
    let inference = pass.elapsed();
    let compile_cache_grew_mb = mxr_cache_mb(&state.config.mxr_cache_base).saturating_sub(cache_before);
    tracing::info!("{}", pass_log_message(
        request_id,
        &format!("embedded {} row(s), dense+sparse in one pass", rows.len()),
        inference.as_secs_f32(),
        compile_cache_grew_mb,
    ));

    let (dense, sparse): (Vec<_>, Vec<_>) = rows.into_iter().unzip();
    // No usage here on purpose: this function also runs over the PINNED batch, whose ruler rows are not
    // the caller's texts. embed_blocking owns the accounting and stamps it on the way out.
    Ok(EmbedResponse {
        dense,
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

fn rerank_blocking(
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
    if guard.is_none() {
        stamp.enter(Phase::Building, "rerank: building the session (a first-ever shape compiles for minutes; cached shapes load in seconds)");
        let building = Instant::now();
        *guard = Some(load_rerank(state, provider_hint)?);
        session_build_ms = building.elapsed().as_millis() as u64;
    }
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
fn score_documents(
    state: &AppState,
    guard: &mut std::sync::MutexGuard<'_, Option<TextRerank>>,
    query: &str,
    documents: &[String],
    max_batch: usize,
    request_id: &str,
) -> anyhow::Result<(Vec<f32>, PassTimings)> {
    let count = documents.len();
    let (cache_before, pass) = (mxr_cache_mb(&state.config.mxr_cache_base), std::time::Instant::now());
    // query.to_string(): fastembed's `rerank` shares one generic across the query and the document slice,
    // so an owned query is what lets `&[String]` documents satisfy it.
    let results = guard
        .as_mut()
        .expect("just loaded")
        .rerank(query.to_string(), documents, false, Some(max_batch))?;
    let inference = pass.elapsed();
    let compile_cache_grew_mb = mxr_cache_mb(&state.config.mxr_cache_base).saturating_sub(cache_before);
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
fn aligned_scores(count: usize, results: impl IntoIterator<Item = (usize, f32)>) -> Vec<f32> {
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
fn set_activity(state: &AppState, activity: impl Into<String>) {
    if let Ok(mut slot) = state.activity.lock() {
        *slot = activity.into();
    }
}

/// Total size of the MIGraphX compiled-model cache tree in whole MB (0 = cache not configured,
/// i.e. a non-migraphx flavor). Recursive over the per-engine subdirectories. The EP reads AND
/// writes the cache LAZILY — at the first kernel launch, not at session build — so growth is
/// measured across a PASS, never across a session build (which taught us nothing and lied
/// "served from cache" while the first pass then compiled for two minutes).
fn mxr_cache_mb(base: &str) -> u64 {
    fn tree_bytes(dir: &std::path::Path) -> u64 {
        std::fs::read_dir(dir)
            .map(|entries| {
                entries
                    .flatten()
                    .map(|e| {
                        let path = e.path();
                        if path.is_dir() { tree_bytes(&path) } else { e.metadata().map(|m| m.len()).unwrap_or(0) }
                    })
                    .sum()
            })
            .unwrap_or(0)
    }
    if base.trim().is_empty() {
        return 0;
    }
    tree_bytes(std::path::Path::new(base)) / (1024 * 1024)
}

/// The per-pass log line: plain timing, plus the truth about compilation when the cache grew
/// during the pass — that is the ONLY moment a MIGraphX compile is observable (lazy save).
/// The caller's `request_id` leads the line when one was sent — two concurrent requests are
/// otherwise indistinguishable here.
fn pass_log_message(request_id: &str, action: &str, secs: f32, cache_grew_mb: u64) -> String {
    let prefix = if request_id.is_empty() { String::new() } else { format!("[{request_id}] ") };
    match cache_grew_mb {
        0 => format!("{prefix}{action} in {secs:.1}s"),
        grew => format!("{prefix}{action} in {secs:.1}s — compiled and cached this input shape (+{grew} MB)"),
    }
}

/// The per-engine slice of the compiled-model cache. The dense, sparse, and rerank engines of
/// bge-m3 run the SAME graph at the SAME pinned shape, so the EP's cache key collides across
/// them — a sparse session that loaded a program cached by another engine returned mis-shaped
/// outputs and died on `assertion failed: index < dim` (the stale-cache incident, 2026-07-27).
/// One subdirectory per engine makes a cache hit always mean "MY program".
fn engine_cache_dir(base: &str, engine: &str) -> String {
    format!("{}/{engine}", base.trim_end_matches('/'))
}

/// Redirects the EP's cache into the engine's own subdirectory for the duration of a session
/// build. The path travels via process env (the only knob this ROCm build honors — it ignores the
/// provider-options fields), so builds are serialized by a lock: two engines building at once
/// would race the variable. No-op (straight call) when no cache is configured.
fn with_engine_cache<T>(base: &str, engine: &str, build: impl FnOnce() -> anyhow::Result<T>) -> anyhow::Result<T> {
    if base.trim().is_empty() {
        return build();
    }

    static BUILD_ENV_LOCK: Mutex<()> = Mutex::new(());
    let _serialized = BUILD_ENV_LOCK.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
    let dir = engine_cache_dir(base, engine);
    std::fs::create_dir_all(&dir).ok();
    std::env::set_var("ORT_MIGRAPHX_MODEL_CACHE_PATH", &dir);
    std::env::set_var("ORT_MIGRAPHX_CACHE_PATH", &dir);
    build()
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
fn lock_or_refuse<'a, S: EngineSlot>(
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

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

// ---------- model loading ----------

/// Records the sequence cap this request runs at, so `cap_for` can keep a query on the rung a pass is
/// using and /health can report it.
///
/// It used to EVICT both embedding engines whenever the cap changed, because `max_length` is baked into
/// an ort session at build time. That was the right shape for one engine slot and the wrong price for a
/// two-rung ladder: a pass crosses the boundary twice and each crossing cost 156-173 s of rebuild (see
/// `RungCache`). The engines are now kept per rung, so a change is a lookup and this records, nothing more.
fn record_embed_max_length(state: &AppState, requested: usize) {
    let Ok(mut loaded) = state.loaded_embed_max_length.lock() else {
        return; // poisoned bookkeeping: keep serving at whatever is loaded rather than failing the embed
    };
    *loaded = Some(requested);
}

/// Same bookkeeping for the BATCH, so /health can report what ran rather than what was configured.
fn record_max_batch(state: &AppState, used: usize) {
    let Ok(mut loaded) = state.loaded_max_batch.lock() else {
        return;
    };
    *loaded = Some(used);
}

/// Files a freshly built engine under its rung and reports what the card now holds. The log line is the
/// operator's only window into the cache's occupancy — an eviction that happened silently would look
/// exactly like the rebuild-per-crossing behaviour this cache exists to remove.
fn remember_engine<T>(cache: &mut RungCache<T>, what: &str, cap: usize, engine: T) {
    let capacity = cache.capacity;
    match cache.insert(cap, engine) {
        Some((evicted, _)) => tracing::info!(
            "{what}: built at cap {cap}; rung {evicted} evicted to stay within {capacity} — resident: {:?}",
            cache.caps()
        ),
        None => tracing::info!("{what}: built at cap {cap} — resident rung(s): {:?}", cache.caps()),
    }
}

/// The attention-score peak of ONE layer, in MB: `batch × 16 heads × seq² × 4 B`, doubled for the second
/// softmax buffer. Logged at startup so a misconfigured envelope is visible before the first embed.
fn attention_peak_mb(batch: usize, seq: usize) -> usize {
    batch * 16 * seq * seq * 4 * 2 / (1024 * 1024)
}

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

/// Must match `CANARY_TEXT` in scripts/generate-canary-reference.mjs — the reference vector was
/// computed from exactly this string.
const CANARY_TEXT: &str =
    "A canary sentence for the bge-m3 engine self-check: the quick brown fox jumps over the lazy dog 0123456789.";

/// Deliberately loose: EP-to-EP arithmetic differences sit near 0.9999, the observed garbage at
/// 0.13. This is a corruption detector, not a numerics test — the parity harness owns exactness.
const CANARY_MIN_COSINE: f32 = 0.99;

/// The dense embedding of `CANARY_TEXT`, captured 2026-07-31 from the unified build that passed the
/// parity gate bit-exact at both caps (scripts/generate-canary-reference.mjs). Regenerate only when
/// the MODEL deliberately changes — never to green a failing canary.
fn canary_reference() -> &'static [f32] {
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
fn cosine(a: &[f32], b: &[f32]) -> f32 {
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
fn canary_check(engine: &mut Bgem3DualEmbedding, limits: Limits, pin: bool) -> anyhow::Result<()> {
    let (texts, position) = if pin && limits.max_batch >= 2 {
        let (expanded, positions) = pin_shape(&[CANARY_TEXT.to_string()], limits.max_batch, ruler_text());
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
fn load_validated_dual(
    state: &AppState,
    provider_hint: &str,
    limits: Limits,
    pin: bool,
) -> anyhow::Result<Bgem3DualEmbedding> {
    let mut engine = load_dual(state, provider_hint, limits.max_length)?;
    let Err(first_failure) = canary_check(&mut engine, limits, pin) else {
        return Ok(engine);
    };

    if state.config.mxr_cache_base.trim().is_empty() {
        // No compiled-model cache -> nothing to heal by wiping; the failure is the answer.
        return Err(first_failure.context("canary failed with no compiled-model cache configured"));
    }

    let dir = engine_cache_dir(&state.config.mxr_cache_base, "dual");
    tracing::warn!(
        "canary failed ({first_failure:#}) — wiping `{dir}` and recompiling once: a crash mid-compile leaves a corrupt program that loads and stably produces garbage"
    );
    drop(engine);
    std::fs::remove_dir_all(&dir).ok();
    let mut rebuilt = load_dual(state, provider_hint, limits.max_length)?;
    canary_check(&mut rebuilt, limits, pin)
        .context("canary still failing after a clean recompile — refusing to serve garbage embeddings")?;
    Ok(rebuilt)
}

/// Builds the ONE session both heads share. Its compiled-model cache slice is `dual/` — its own,
/// never `dense/` or `sparse/`: a cache hit must always mean "MY program" (the 2026-07-27 stale-cache
/// incident), and the per-head slices belong to the retired two-session binaries.
fn load_dual(state: &AppState, provider_hint: &str, max_length: usize) -> anyhow::Result<Bgem3DualEmbedding> {
    let provider = pin_provider(state, provider_hint);
    // The hint path's preflight: when ORT_PROVIDER was empty at startup the provider is only known
    // now, so the same check runs here — before the first session, not at the first user-visible
    // failure. Startup already covered the explicit case.
    if let Err(error) = preflight_provider(&provider, &exe_dir()) {
        return record_session_outcome::<Bgem3DualEmbedding>(state, &provider, Err(error));
    }

    tracing::info!("loading {DUAL_MODEL} (provider {provider}, max_length {max_length})");
    let mut options = Bgem3DualInitOptions::default()
        .with_max_length(max_length)
        .with_cache_dir(state.config.cache_dir.clone())
        .with_execution_providers(execution_providers(&provider, state.config.device_id, state.dml_device_id()));
    if state.config.intra_threads > 0 {
        options = options.with_intra_threads(state.config.intra_threads);
    }
    let built = with_engine_cache(&state.config.mxr_cache_base, "dual", || {
        let started = std::time::Instant::now();
        let engine = Bgem3DualEmbedding::try_new(options)?;
        tracing::info!(
            "{DUAL_MODEL}: session ready in {:.1}s (the EP compiles or loads its cache lazily, on the first pass)",
            started.elapsed().as_secs_f32()
        );
        Ok(engine)
    });
    record_session_outcome(state, &provider, built)
}

fn load_rerank(state: &AppState, provider_hint: &str) -> anyhow::Result<TextRerank> {
    let provider = pin_provider(state, provider_hint);
    if let Err(error) = preflight_provider(&provider, &exe_dir()) {
        return record_session_outcome::<TextRerank>(state, &provider, Err(error));
    }

    tracing::info!("loading {RERANK_MODEL} (provider {provider}, max_length {})", state.config.rerank_max_length);
    let mut options = RerankInitOptions::new(RerankerModel::BGERerankerV2M3)
        .with_max_length(state.config.rerank_max_length)
        .with_cache_dir(state.config.cache_dir.clone())
        .with_execution_providers(execution_providers(&provider, state.config.device_id, state.dml_device_id()));
    if state.config.intra_threads > 0 {
        options = options.with_intra_threads(state.config.intra_threads);
    }
    let built = with_engine_cache(&state.config.mxr_cache_base, "rerank", || {
        let started = std::time::Instant::now();
        let engine = TextRerank::try_new(options)?;
        tracing::info!(
            "{RERANK_MODEL}: session ready in {:.1}s (the EP compiles or loads its cache lazily, on the first pass)",
            started.elapsed().as_secs_f32()
        );
        Ok(engine)
    });
    record_session_outcome(state, &provider, built)
}

/// The provider all engines pin to: ORT_PROVIDER env wins, else the first request's hint, else auto.
/// Stored once so later loads and shape pinning reuse the same choice until restart.
/// <para>This is the REQUEST. It says nothing about whether a session can be built on it — that is
/// `AppState::active_provider`, written only after one succeeds.</para>
fn pin_provider(state: &AppState, hint: &str) -> String {
    state.pinned_provider.get_or_init(|| effective_provider(&state.config, hint)).clone()
}

/// The execution providers compiled into THIS binary flavor. Derived from the cargo features rather
/// than from a list someone maintains by hand, so it cannot disagree with the build.
fn compiled_providers() -> Vec<&'static str> {
    let mut providers = Vec::new();
    if cfg!(feature = "cuda") {
        providers.push("cuda");
    }
    if cfg!(feature = "dml") {
        providers.push("dml");
    }
    if cfg!(feature = "migraphx") {
        providers.push("migraphx");
    }
    // Always available: ort falls through to CPU when no EP is registered.
    providers.push("cpu");
    providers
}

/// The provider DLLs an EP needs BESIDE THE EXE. ORT's provider bridge resolves them from the
/// executable's directory — not from PATH — so a package that ships only the exe fails at the first
/// inference with `Error 126`, minutes into a user's search rather than at startup.
fn required_provider_libraries(provider: &str) -> &'static [&'static str] {
    match provider {
        "cuda" => &["onnxruntime_providers_shared.dll", "onnxruntime_providers_cuda.dll"],
        _ => &[],
    }
}

/// Refuses a provider this process cannot possibly serve, with the reason a reader can act on.
///
/// Two failures are indistinguishable at the first inference and must not be: an EP absent from the
/// BUILD (wrong flavor — rebuild) and an EP whose runtime libraries were not deployed (wrong package —
/// re-run the install script). `auto` is exempt: it is a request to try whatever is present, so there
/// is nothing to refuse.
fn preflight_provider(provider: &str, exe_dir: &Path) -> anyhow::Result<()> {
    if provider == "auto" || provider == "cpu" {
        return Ok(());
    }

    let compiled = compiled_providers();
    if !compiled.contains(&provider) {
        anyhow::bail!(
            "execution provider `{provider}` was requested, but this binary was built without it \
             (compiled: {}). Rebuild with `--features {provider}` — no configuration change can add \
             an EP that is not in the binary.",
            compiled.join(", ")
        );
    }

    let missing: Vec<&str> = required_provider_libraries(provider)
        .iter()
        .copied()
        .filter(|lib| !exe_dir.join(lib).exists())
        .collect();
    if !missing.is_empty() {
        anyhow::bail!(
            "execution provider `{provider}` is compiled in, but its runtime libraries are missing \
             from `{}`: {}. ONNX Runtime resolves these from the EXECUTABLE'S directory, not PATH — \
             deploy the full package (tools/bge-sidecar/scripts/build-cuda-windows.ps1), not just the exe.",
            exe_dir.display(),
            missing.join(", ")
        );
    }

    Ok(())
}

/// The directory the running executable lives in — where ORT looks for provider DLLs.
fn exe_dir() -> PathBuf {
    std::env::current_exe()
        .ok()
        .and_then(|p| p.parent().map(Path::to_path_buf))
        .unwrap_or_else(|| PathBuf::from("."))
}

/// SHA-256 of a file, streamed so a 300 MB provider library never enters memory whole.
/// `None` for anything unreadable — the caller reports absence rather than a plausible-looking zero.
fn sha256_file(path: &Path) -> Option<String> {
    use sha2::Digest;
    let mut file = std::fs::File::open(path).ok()?;
    let mut hasher = sha2::Sha256::new();
    std::io::copy(&mut file, &mut hasher).ok()?;
    Some(hasher.finalize().iter().map(|b| format!("{b:02x}")).collect())
}

/// Build provenance: what binary is answering, and which provider libraries stand behind it.
struct Provenance {
    exe_sha256: String,
    runtime_manifest_sha256: String,
}

/// Computed exactly once, at STARTUP, on the blocking pool — never on a request path. `None` until
/// that task lands, which /health reports as `provenance_ready: false`.
static PROVENANCE: OnceLock<Provenance> = OnceLock::new();

/// Hashes the executable and the libraries beside it. Blocking and unbounded by design: on a
/// CUDA/DirectML deployment this is hundreds of MB to gigabytes (cuDNN alone is often >500 MB).
fn compute_provenance() -> Provenance {
    Provenance { exe_sha256: compute_exe_sha256(), runtime_manifest_sha256: compute_runtime_manifest_sha256() }
}

/// Starts the hashing off the request path.
///
/// The incident class: the first `/health` call used to compute both hashes INLINE on a Tokio reactor
/// thread — measured at 1.4 s over a mere 67 MB of test binaries, and a real CUDA install is orders of
/// magnitude more — which is exactly the blocking work this endpoint's own design forbids a few lines
/// above (`loaded_now`'s try_lock, written after a hung load froze /health forever). A readiness probe
/// that hashes gigabytes reports nothing, slowly, while occupying a thread the server needs.
///
/// The outer task exists so the fault is OBSERVED: a detached `spawn_blocking` whose JoinHandle nobody
/// awaits is a hash that silently never happens, and /health would then say `provenance_ready: false`
/// forever with no line in the log saying why.
fn prewarm_provenance() {
    tokio::spawn(async {
        let started = Instant::now();
        match tokio::task::spawn_blocking(|| PROVENANCE.set(compute_provenance()).is_ok()).await {
            Ok(_) => tracing::info!(
                "build provenance hashed in {:.1}s (off the request path): exe {}, runtime manifest {}",
                started.elapsed().as_secs_f32(),
                short_hash(PROVENANCE.get().map(|p| p.exe_sha256.as_str()).unwrap_or_default()),
                short_hash(PROVENANCE.get().map(|p| p.runtime_manifest_sha256.as_str()).unwrap_or_default()),
            ),
            Err(e) => tracing::warn!(
                "build provenance could not be computed ({}) — /health will keep reporting provenance_ready: \
                 false, which is honest but leaves a benchmark unable to prove which binary it measured",
                join_error_text(e)
            ),
        }
    });
}

/// A hash short enough to read in a log line; `(none)` when there is nothing to shorten.
fn short_hash(hash: &str) -> &str {
    match hash.len() {
        0 => "(none)",
        _ => &hash[..hash.len().min(12)],
    }
}

/// SHA-256 of the executable THIS PROCESS is running from.
///
/// Read via `current_exe()`, never from a configured path or an environment variable: the whole point
/// is to catch the case where the installed binary is not the one the build produced, and a value the
/// launcher supplies describes the binary somebody INTENDED to start.
fn compute_exe_sha256() -> String {
    std::env::current_exe()
        .ok()
        .and_then(|p| sha256_file(&p))
        .unwrap_or_default()
}

/// SHA-256 over the manifest of every dynamic library sitting beside the executable.
///
/// The executable can be byte-identical while the CUDA/cuDNN/DirectML libraries next to it are not,
/// and those decide which execution provider can actually load — so the binary's own hash is not
/// sufficient provenance for a benchmark. The manifest is one `name:sha256` row per library, sorted
/// by name and newline-joined, so the order the filesystem happens to enumerate them in is never
/// mistaken for a change. Names are lower-cased: Windows paths are case-insensitive and a
/// differently-cased listing is not a different runtime.
fn compute_runtime_manifest_sha256() -> String {
    use sha2::Digest;
    let Ok(entries) = std::fs::read_dir(exe_dir()) else {
        return String::new();
    };

    let mut rows: Vec<String> = entries
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| {
            path.extension()
                .and_then(|ext| ext.to_str())
                .is_some_and(|ext| ext.eq_ignore_ascii_case("dll") || ext.eq_ignore_ascii_case("so"))
        })
        .filter_map(|path| {
            let name = path.file_name()?.to_str()?.to_ascii_lowercase();
            Some(format!("{name}:{}", sha256_file(&path)?))
        })
        .collect();

    if rows.is_empty() {
        // No libraries beside the executable is a real fact about a misdeployed install, but it is
        // not a manifest — hashing nothing would let two unrelated deployments agree.
        return String::new();
    }

    rows.sort();
    sha2::Sha256::new()
        .chain_update(rows.join("\n").as_bytes())
        .finalize()
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

/// Records the OUTCOME of a session build. Success promotes the provider to `active`; failure stores
/// the reason and leaves `active` untouched, so `provider_ready` can never be true on a provider that
/// has not actually served anything.
fn record_session_outcome<T>(state: &AppState, provider: &str, built: anyhow::Result<T>) -> anyhow::Result<T> {
    match built {
        Ok(engine) => {
            if let Ok(mut active) = state.active_provider.lock() {
                *active = Some(provider.to_string());
            }
            if let Ok(mut last) = state.last_provider_error.lock() {
                *last = None;
            }
            Ok(engine)
        }
        Err(error) => {
            if let Ok(mut last) = state.last_provider_error.lock() {
                *last = Some(format!("{error:#}"));
            }
            Err(error)
        }
    }
}

fn effective_provider(config: &Config, hint: &str) -> String {
    let value = if config.provider.is_empty() { hint } else { &config.provider };
    match value.trim().to_lowercase().as_str() {
        p @ ("cuda" | "dml" | "migraphx" | "cpu") => p.to_string(),
        _ => "auto".to_string(),
    }
}

/// EP registration chain. An explicit choice fails hard (a broken GPU setup must be visible, not a
/// silent CPU fallback); `auto` tries cuda -> migraphx -> dml and lets ort fall through to CPU. EPs
/// not compiled into this binary flavor simply fail registration. CUDA and MIGraphX get the raw
/// configured id (their own numbering — HIP device order for MIGraphX, so pin the discrete card
/// with HIP_VISIBLE_DEVICES when an iGPU is present); DirectML gets the DXGI-mapped
/// plain-enumeration index (see adapters.rs).
fn execution_providers(provider: &str, cuda_device_id: i32, dml_device_id: i32) -> Vec<ExecutionProviderDispatch> {
    let cuda = CUDAExecutionProvider::default().with_device_id(cuda_device_id);
    let dml = DirectMLExecutionProvider::default().with_device_id(dml_device_id);
    let migraphx = MIGraphXExecutionProvider::default().with_device_id(cuda_device_id);
    match provider {
        "cuda" => vec![cuda.build().error_on_failure()],
        "dml" => vec![dml.build().error_on_failure()],
        "migraphx" => vec![migraphx.build().error_on_failure()],
        "cpu" => vec![],
        _ => vec![cuda.build(), migraphx.build(), dml.build()],
    }
}

#[cfg(test)]
mod tests {
    use super::{
        aligned_scores, cache_dir_verdict, compiled_providers, dylib_verdict, effective_provider, engine_cache_dir,
        execution_providers, copy_missing_files, embed_settling, find_tokenizer_file, health, inflight_now,
        join_error_text, loaded_now, lock_or_refuse, cap_for, load_token_counter, parse_unload_request,
        pass_log_message, pin_shape, preflight_provider, rerank_batch, required_provider_libraries, ruler_text,
        should_pin_shape, unload, unpin_rows, usage_from_counts, wedge_action, with_engine_cache, write_inflight,
        AppState, Config, EmbedResponse, Engines, InFlight, PassTimings, Patience, Phase, Provenance,
        RungCache, TokenUsage, WedgeAction, WedgePolicy, SETTLE_ATTEMPTS,
    };
    use axum::body::Bytes;
    use axum::extract::State;
    use std::path::{Path, PathBuf};
    use std::sync::mpsc;
    use std::sync::{Arc, Mutex, OnceLock};
    use std::time::{Duration, Instant};

    /// Wedge ceilings scaled down to milliseconds so the tests measure the RULE rather than the clock.
    /// The shipped values (15 min / 60 min / 30 s) are asserted separately, from the env defaults.
    fn test_wedge_policy() -> WedgePolicy {
        WedgePolicy {
            running_after: Duration::from_millis(300),
            building_after: Duration::from_millis(600),
            unload_wait: Duration::from_millis(200),
            poll: Duration::from_millis(10),
            exit_after_wedged: None,
        }
    }

    /// An in-flight record that started `ago` in the past — the injected clock these tests need, without
    /// a clock abstraction: `Instant` arithmetic is the whole of it.
    fn stamped(phase: Phase, label: &str, ago: Duration) -> InFlight {
        InFlight {
            phase,
            label: label.to_string(),
            since: Instant::now().checked_sub(ago).expect("the process started after the epoch"),
        }
    }

    /// A sidecar state with no engines and no GPU behind it. Every reliability test here exercises
    /// the LOCKS and the probe path, never the model, so an empty state is the whole subject.
    fn app_state() -> Arc<AppState> {
        Arc::new(AppState {
            engines: Engines::new(2),
            config: config(""),
            activity: Mutex::new("idle".to_string()),
            pinned_provider: OnceLock::new(),
            active_provider: Mutex::new(None),
            last_provider_error: Mutex::new(None),
            loaded_embed_max_length: Mutex::new(None),
            loaded_max_batch: Mutex::new(None),
            adapter: None,
            token_counter: OnceLock::new(),
            qwen_counter: OnceLock::new(),
        })
    }

    /// Holds an engine mutex from another thread until it is told to let go — the shape of a thread
    /// WEDGED inside the ORT/MIGraphX C++ call, which is the one failure the poison healing can never
    /// see: a stuck thread does not panic, so the mutex is never poisoned, only never released.
    struct HeldEngine {
        release: mpsc::Sender<()>,
        thread: Option<std::thread::JoinHandle<()>>,
    }

    impl HeldEngine {
        fn hold<S: Send + 'static>(engine: Arc<AppState>, pick: fn(&AppState) -> &Mutex<S>) -> Self {
            let (release, released) = mpsc::channel::<()>();
            let (holding, held) = mpsc::channel::<()>();
            let thread = std::thread::spawn(move || {
                let _guard = pick(&engine).lock().expect("fresh lock");
                holding.send(()).expect("announce the hold");
                released.recv().ok();
            });
            held.recv().expect("the engine is held");
            Self { release, thread: Some(thread) }
        }
    }

    impl Drop for HeldEngine {
        fn drop(&mut self) {
            self.release.send(()).ok();
            if let Some(thread) = self.thread.take() {
                thread.join().ok();
            }
        }
    }

    fn config(provider: &str) -> Config {
        Config {
            provider: provider.to_string(),
            device_id: 0,
            max_batch: 4,
            intra_threads: 0,
            embed_max_length: 1024,
            rerank_max_length: 1024,
            cache_dir: PathBuf::from(".model-cache"),
            qwen_tokenizer_path: PathBuf::from("../qwen-tokenizer/tokenizer.json"),
            pin_input_shape: "auto".to_string(),
            mxr_cache_base: String::new(),
            engine_cache_rungs: 2,
            wedge: test_wedge_policy(),
        }
    }

    /// A u8 stands in for an ort session throughout the cache tests: the cache never looks inside an
    /// engine, and a real one cannot be built without a GPU.
    fn cache_of(capacity: usize, rungs: &[(usize, u8)]) -> RungCache<u8> {
        let mut cache = RungCache::new(capacity);
        for &(cap, engine) in rungs {
            cache.insert(cap, engine);
        }
        cache
    }

    /// The whole point of the cache: a rung already built is HANDED BACK, not rebuilt. Rebuilding cost
    /// 156-173 s measured, because MIGraphX re-materialises its ~2.4 GB program on the first run.
    #[test]
    fn a_rung_already_built_is_returned_rather_than_rebuilt() {
        let mut cache = cache_of(2, &[(256, 7)]);

        assert_eq!(cache.get_mut(256).copied(), Some(7), "the built engine comes back");
        assert_eq!(cache.get_mut(1024), None, "a rung never built is a miss, not a wrong engine");
        assert_eq!(cache.caps(), vec![256], "a miss builds nothing on its own");
    }

    /// THE regression this cache exists for. A Fast pass walks the ladder DOWN (ceiling first), ends on
    /// the low rung, and the next pass starts at the ceiling again — so the boundary is crossed twice per
    /// pass. Before the cache each crossing evicted both engines: ~5.5 min per pass, forever. At a
    /// capacity of 1 this test goes red on the last assertion, which is exactly the old behaviour.
    #[test]
    fn walking_the_ladder_down_and_back_up_evicts_nothing() {
        let mut cache = cache_of(2, &[(1024, 10), (256, 20)]);

        assert_eq!(cache.get_mut(1024).copied(), Some(10), "the ceiling survived the step down");
        assert_eq!(cache.get_mut(256).copied(), Some(20), "and the low rung survived the step back up");
        assert_eq!(cache.caps().len(), 2, "a two-rung ladder never evicts at capacity 2");
    }

    /// The escape hatch: EMBED_ENGINE_CACHE_RUNGS=1 must reproduce the pre-cache behaviour exactly, so a
    /// VRAM budget that cannot hold two pairs has somewhere to go without a code change.
    #[test]
    fn capacity_one_reproduces_the_evicting_behaviour() {
        let mut cache = cache_of(1, &[(1024, 10), (256, 20)]);

        assert_eq!(cache.caps(), vec![256], "the newcomer displaced the previous rung");
        assert_eq!(cache.get_mut(1024), None, "stepping back up rebuilds, exactly as before the cache");
    }

    /// Eviction order decides whether the cache helps or hurts: dropping the OLDEST would throw away the
    /// rung the pass is actively using and keep one it has moved on from. Least-recently-USED is the rule.
    #[test]
    fn a_third_rung_evicts_the_least_recently_used_not_the_oldest() {
        let mut cache = cache_of(2, &[(1024, 10), (256, 20)]);
        cache.get_mut(1024).expect("1024 is resident and now the most recently used");

        cache.insert(512, 30);

        assert_eq!(cache.caps(), vec![1024, 512], "256 went — it was the least recently USED");
        assert_eq!(cache.get_mut(1024).copied(), Some(10), "the rung in active use survived");
    }

    /// Rebuilding a rung that is already resident must REPLACE it, never leave two entries for one cap —
    /// a duplicate would hold a second session's worth of VRAM that nothing can ever hand back.
    #[test]
    fn rebuilding_a_resident_rung_replaces_it_instead_of_duplicating() {
        let mut cache = cache_of(2, &[(256, 1)]);

        cache.insert(256, 2);

        assert_eq!(cache.caps(), vec![256], "one entry per cap");
        assert_eq!(cache.get_mut(256).copied(), Some(2), "the newest build wins");
    }

    /// /unload hands the whole card to an exclusive local LLM. A rung left behind would keep holding VRAM
    /// while the host believes the sidecar released it — and the host asserts on `loaded: {false,…}`.
    #[test]
    fn unload_drains_every_resident_rung() {
        let mut cache = cache_of(2, &[(1024, 10), (256, 20)]);

        let drained = cache.drain();

        assert_eq!(drained.len(), 2, "both rungs come out");
        assert!(cache.caps().is_empty() && !loaded_now(&Mutex::new(cache)), "and /health reports nothing loaded");
    }

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

    /// ORT_PROVIDER wins over the request hint; every supported token round-trips; unknown tokens
    /// (including the retired "rocm") degrade to auto instead of failing the load.
    #[test]
    fn provider_token_resolution_env_wins_and_unknown_degrades_to_auto() {
        assert_eq!(effective_provider(&config(""), "migraphx"), "migraphx");
        assert_eq!(effective_provider(&config("migraphx"), "cuda"), "migraphx");
        assert_eq!(effective_provider(&config("dml"), ""), "dml");
        assert_eq!(effective_provider(&config(""), "rocm"), "auto");
        assert_eq!(effective_provider(&config(""), ""), "auto");
    }

    /// An explicit provider registers exactly ONE fail-hard EP; `cpu` registers none (ort's own
    /// CPU fallback); `auto` offers every GPU EP so uncompiled ones fall through at registration.
    #[test]
    fn dispatch_registers_one_ep_per_explicit_provider_and_all_for_auto() {
        assert_eq!(execution_providers("cuda", 0, 0).len(), 1);
        assert_eq!(execution_providers("dml", 0, 0).len(), 1);
        assert_eq!(execution_providers("migraphx", 0, 0).len(), 1);
        assert_eq!(execution_providers("cpu", 0, 0).len(), 0);
        assert_eq!(execution_providers("auto", 0, 0).len(), 3);
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

    /// The partial unload's per-rung eviction: only the NAMED rung goes, the rest keep their order,
    /// and a non-resident cap is a no-op — the host may name a rung another eviction already took.
    #[test]
    fn remove_drops_only_the_named_rung() {
        let mut cache: RungCache<u8> = RungCache::new(2);
        cache.insert(1024, 1);
        cache.insert(256, 2);

        assert_eq!(cache.remove(1024), Some(1));
        assert_eq!(cache.caps(), vec![256], "the other rung stays resident");
        assert_eq!(cache.remove(512), None, "a non-resident cap is a no-op");
        assert_eq!(cache.caps(), vec![256]);
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

    /// The corruption detector's arithmetic: identical vectors score 1, unrelated directions score
    /// low, and a dimension mismatch is a FAILED canary (-1), never a panic — the whole point is
    /// that garbage output must fail the check, whatever shape it arrives in.
    #[test]
    fn cosine_separates_identity_from_garbage_and_never_panics_on_shape() {
        let v = [0.6f32, 0.8, 0.0];
        assert!((super::cosine(&v, &v) - 1.0).abs() < 1e-6, "a vector matches itself");
        assert!(super::cosine(&[1.0, 0.0], &[0.0, 1.0]).abs() < 1e-6, "orthogonal scores ~0");
        assert_eq!(super::cosine(&v, &[1.0, 0.0]), -1.0, "a dimension mismatch fails, not panics");
        assert_eq!(super::cosine(&[0.0, 0.0], &[1.0, 0.0]), -1.0, "a zero vector fails, not NaNs");
    }

    /// The embedded reference must be exactly one bge-m3 dense vector, finite and L2-normalized —
    /// a truncated or stale file would make the canary fail every healthy engine.
    #[test]
    fn canary_reference_is_one_finite_normalized_bgem3_vector() {
        let reference = super::canary_reference();
        assert_eq!(reference.len(), 1024, "bge-m3 dense dim");
        assert!(reference.iter().all(|v| v.is_finite()));
        let norm: f32 = reference.iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-3, "the sidecar serves normalized vectors, norm was {norm}");
    }

    /// The dylib preflight verdict names BOTH versions on mismatch (found + required) so the
    /// operator knows exactly what to rebuild; a dylib that serves our API version passes.
    #[test]
    fn dylib_preflight_names_required_and_found_versions_on_mismatch() {
        let err = dylib_verdict(false, "1.23.2", "/opt/onnxruntime-migraphx/lib/libonnxruntime.so")
            .expect_err("older dylib must be rejected");
        assert!(err.contains("1.23.2"), "names the found version: {err}");
        assert!(err.contains(&format!("1.{}", ort::MINOR_VERSION)), "names the required version: {err}");
        assert!(err.contains("--use_migraphx"), "tells the operator how to rebuild: {err}");

        assert!(dylib_verdict(true, "1.24.4", "libonnxruntime.so").is_ok());
    }

    /// An unset or unwritable MIGraphX cache path must be caught at STARTUP: the EP saves every
    /// compiled model there unconditionally, so leaving it empty made each /embed spend ~2 minutes
    /// compiling and then answer 500 ("write_buffer: Failure opening file: \"\"/…mxr"), with the GPU
    /// idle and the indexing stage stuck. Both messages must say what to set.
    #[test]
    fn cache_preflight_rejects_an_unset_or_unwritable_path() {
        let unset = cache_dir_verdict("   ", |_| panic!("must not probe when the path is empty"))
            .expect_err("an empty path must be rejected");
        assert!(unset.contains("ORT_MIGRAPHX_MODEL_CACHE_PATH"), "names the variable: {unset}");
        assert!(unset.contains("WslMigraphxCacheDir"), "names the AppHost knob: {unset}");

        let denied = cache_dir_verdict("/read-only/cache", |_| Err("Permission denied".to_string()))
            .expect_err("an unwritable path must be rejected");
        assert!(denied.contains("/read-only/cache") && denied.contains("Permission denied"), "{denied}");

        let ok = cache_dir_verdict("/var/tmp/mgx", |_| Ok(())).expect("a writable path passes");
        assert!(ok.contains("/var/tmp/mgx"), "{ok}");
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

    /// Dense, sparse, and rerank run the SAME graph at the SAME pinned shape, so the EP's cache
    /// key collides across engines — a sparse session once loaded a program cached by another
    /// engine and died on mis-shaped outputs. Every engine must therefore get its OWN cache slice,
    /// and an unconfigured cache must stay a plain passthrough.
    #[test]
    fn each_engine_gets_its_own_cache_slice() {
        assert_eq!(engine_cache_dir("/cache/device-0", "sparse"), "/cache/device-0/sparse");
        assert_eq!(engine_cache_dir("/cache/device-0/", "dense"), "/cache/device-0/dense");

        let ran = with_engine_cache("", "dense", || Ok::<_, anyhow::Error>(42)).expect("passthrough");
        assert_eq!(ran, 42, "no cache configured -> the build just runs");
    }

    /// MIGraphX compiles and saves LAZILY — on the first pass, not at session build — so the
    /// compile attribution lives on the PASS log line, and only when the cache actually grew.
    /// The old build-time attribution reported "served from the compiled-model cache" while the
    /// first pass then silently compiled for two minutes — worse than no message at all.
    #[test]
    fn pass_log_reports_a_compile_only_when_the_cache_grew() {
        let compiled = pass_log_message("", "sparse: embedded 32 row(s)", 118.3, 2401);
        assert!(
            compiled.contains("compiled and cached") && compiled.contains("+2401 MB"),
            "a grown cache means THIS pass compiled: {compiled}"
        );

        let plain = pass_log_message("", "dense: embedded 32 row(s)", 1.4, 0);
        assert!(!plain.contains("compiled"), "steady state stays plain timing: {plain}");
    }

    /// The request id is a correlation aid: it leads the line when the caller sent one and adds no
    /// noise when none was sent — two concurrent requests were previously indistinguishable in the log.
    #[test]
    fn pass_log_prefixes_the_request_id_only_when_one_was_sent() {
        let tagged = pass_log_message("leg-7/q3", "embedded 8 row(s)", 0.4, 0);
        assert!(tagged.starts_with("[leg-7/q3] "), "the caller's id leads the line: {tagged}");

        let untagged = pass_log_message("", "embedded 8 row(s)", 0.4, 0);
        assert!(!untagged.contains('['), "no id, no prefix: {untagged}");
    }

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
            dense: vec![],
            sparse: vec![],
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

    /// A scratch tree for the seeding tests, unique per test so parallel runs never collide.
    fn seed_scratch(name: &str) -> (std::path::PathBuf, std::path::PathBuf) {
        let base = std::env::temp_dir().join(format!("bge-seed-{name}-{}", std::process::id()));
        std::fs::remove_dir_all(&base).ok();
        let from = base.join("from");
        let to = base.join("to");
        std::fs::create_dir_all(&from).unwrap();
        (from, to)
    }

    /// The whole point: a machine with no ext4 copy gets one automatically, nested layout included —
    /// the HF cache is `models--…/snapshots/<sha>/onnx/…`.
    #[test]
    fn seeding_copies_the_full_tree_to_an_empty_target() {
        let (from, to) = seed_scratch("full");
        std::fs::create_dir_all(from.join("m/onnx")).unwrap();
        std::fs::write(from.join("m/onnx/model.onnx_data"), b"weights").unwrap();
        std::fs::write(from.join("tokenizer.json"), b"tok").unwrap();

        let seeded = copy_missing_files(&from, &to);

        assert_eq!((seeded.files, seeded.bytes), (2, 10));
        assert_eq!(std::fs::read(to.join("m/onnx/model.onnx_data")).unwrap(), b"weights");
    }

    /// Later starts must verify and SKIP — re-copying 4.3 GB on every boot would trade the DrvFs tax
    /// for a copy tax. The models are immutable HF blobs, so same size = same file.
    #[test]
    fn seeding_skips_files_the_target_already_has() {
        let (from, to) = seed_scratch("skip");
        std::fs::write(from.join("model.bin"), b"12345").unwrap();
        std::fs::create_dir_all(&to).unwrap();
        std::fs::write(to.join("model.bin"), b"abcde").unwrap(); // same size, already seeded

        let seeded = copy_missing_files(&from, &to);

        assert_eq!(seeded.files, 0);
        assert_eq!(std::fs::read(to.join("model.bin")).unwrap(), b"abcde", "a same-size file is never rewritten");
    }

    /// A size MISMATCH is an interrupted earlier copy and must be repaired, not trusted.
    #[test]
    fn seeding_repairs_a_truncated_earlier_copy() {
        let (from, to) = seed_scratch("repair");
        std::fs::write(from.join("model.bin"), b"full-content").unwrap();
        std::fs::create_dir_all(&to).unwrap();
        std::fs::write(to.join("model.bin"), b"half").unwrap();

        let seeded = copy_missing_files(&from, &to);

        assert_eq!(seeded.files, 1);
        assert_eq!(std::fs::read(to.join("model.bin")).unwrap(), b"full-content");
    }

    /// A missing seed dir is a quiet no-op — a fresh distro simply downloads from HF as before.
    #[test]
    fn seeding_from_a_missing_dir_copies_nothing() {
        let (from, to) = seed_scratch("missing");
        let seeded = copy_missing_files(&from.join("does-not-exist"), &to);

        assert_eq!((seeded.files, seeded.bytes), (0, 0));
        assert!(!to.exists(), "a no-op seed must not create the target either");
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

    // ---- provider truthfulness + fail-fast (2026-08-08) --------------------------------------
    //
    // The defect these pin: /health reported the REQUESTED provider as though it were the active one,
    // so a binary whose CUDA EP failed every registration still answered `provider: "cuda"` while
    // every /embed and /rerank returned 500. Requested, compiled and active are now three facts.

    #[test]
    fn compiled_providers_always_include_cpu_and_match_the_build_flavor() {
        let compiled = compiled_providers();
        assert!(compiled.contains(&"cpu"), "ort falls through to CPU, so it is always available");
        assert_eq!(compiled.contains(&"cuda"), cfg!(feature = "cuda"));
        assert_eq!(compiled.contains(&"dml"), cfg!(feature = "dml"));
        assert_eq!(compiled.contains(&"migraphx"), cfg!(feature = "migraphx"));
    }

    #[test]
    fn an_uncompiled_provider_is_refused_with_a_rebuild_instruction() {
        // Pick a provider this flavor does NOT have, whatever flavor the test runs under.
        let absent = ["cuda", "dml", "migraphx"]
            .into_iter()
            .find(|p| !compiled_providers().contains(p));
        let Some(absent) = absent else {
            return; // a build with every EP compiled in has nothing to refuse
        };

        let error = preflight_provider(absent, Path::new("."))
            .expect_err("an EP absent from the binary must be refused");
        let text = format!("{error:#}");
        assert!(text.contains("built without it"), "{text}");
        assert!(text.contains("--features"), "the message must say how to fix it: {text}");
    }

    #[test]
    fn a_compiled_provider_with_missing_runtime_libraries_is_refused_at_preflight() {
        if !cfg!(feature = "cuda") {
            return; // only the CUDA EP ships separate provider DLLs
        }
        let empty = std::env::temp_dir().join(format!("bge-preflight-{}", std::process::id()));
        std::fs::create_dir_all(&empty).expect("temp dir");

        let error = preflight_provider("cuda", &empty)
            .expect_err("CUDA without its provider DLLs beside the exe cannot serve anything");
        let text = format!("{error:#}");
        assert!(text.contains("onnxruntime_providers_shared.dll"), "{text}");
        assert!(text.contains("EXECUTABLE'S directory"), "PATH is not where ORT looks: {text}");

        std::fs::remove_dir_all(&empty).ok();
    }

    #[test]
    fn auto_and_cpu_are_never_refused() {
        // `auto` asks for whatever is present, so there is nothing to promise and nothing to break.
        preflight_provider("auto", Path::new("/nonexistent")).expect("auto is always serveable");
        preflight_provider("cpu", Path::new("/nonexistent")).expect("cpu needs no EP libraries");
    }

    #[test]
    fn cuda_declares_exactly_the_two_provider_libraries_ort_resolves_beside_the_exe() {
        assert_eq!(
            required_provider_libraries("cuda"),
            ["onnxruntime_providers_shared.dll", "onnxruntime_providers_cuda.dll"]
        );
        // `..._shared.dll` is the one the live failure named (Error 126) — dropping it from the
        // package is the exact mistake this list exists to prevent.
        assert!(required_provider_libraries("dml").is_empty());
        assert!(required_provider_libraries("auto").is_empty());
    }

    // ---------- token accounting ----------

    // The snapshot folder is a content hash that changes whenever the model is re-pulled, so the path
    // has to be discovered. A hardcoded one would silently stop resolving after an update — and a
    // tokenizer that cannot be found turns the guard off without anyone noticing.
    #[test]
    fn the_tokenizer_is_found_by_scanning_the_snapshot_folder() {
        let root = std::env::temp_dir().join(format!("bge-tok-{}", std::process::id()));
        let snapshot = root.join("models--BAAI--bge-m3").join("snapshots").join("deadbeef");
        std::fs::create_dir_all(&snapshot).unwrap();
        std::fs::write(snapshot.join("tokenizer.json"), b"{}").unwrap();

        let found = find_tokenizer_file(&root).expect("the snapshot holds a tokenizer.json");

        assert_eq!(found, snapshot.join("tokenizer.json"));
        std::fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn a_cache_without_the_model_yields_no_tokenizer() {
        let empty = std::env::temp_dir().join(format!("bge-tok-empty-{}", std::process::id()));
        std::fs::create_dir_all(&empty).unwrap();

        assert!(find_tokenizer_file(&empty).is_none());
        // ...and the loader degrades to "off" rather than failing the sidecar: a missing tokenizer must
        // stop us CLAIMING nothing was truncated, never stop us embedding.
        assert!(load_token_counter(&empty).is_none());
        std::fs::remove_dir_all(&empty).ok();
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

    #[test]
    fn the_default_usage_admits_it_measured_nothing() {
        let usage = TokenUsage::default();

        assert!(!usage.token_accounting);
        assert!(usage.token_count.is_empty());
        assert!(usage.truncated.is_empty());
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

    /// The request path had no deadline of any kind: every /embed behind a wedged inference queued on
    /// `.lock()` forever, and the daemon's deliberately infinite HTTP timeout turned that into a
    /// system-wide freeze nobody could see. A request must be REFUSED with a reason instead.
    #[test]
    fn a_request_refuses_instead_of_queueing_behind_a_wedged_inference() {
        let state = app_state();
        let _held = HeldEngine::hold(state.clone(), |s| &s.engines.embed);
        write_inflight(
            &state.engines.embed_inflight,
            Some(stamped(Phase::Running, "embed: embedding 64 row(s)", Duration::from_secs(30))),
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

        assert!(waited < Duration::from_millis(200), "the refusal is immediate, not another wait: {waited:?}");
        let text = format!("{refused:#}");
        assert!(text.contains("WEDGED"), "{text}");
        assert!(text.contains("embed: embedding 64 row(s)"), "the reason names what is holding it: {text}");
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
            Some(stamped(Phase::Building, "embed: building the session", Duration::from_millis(10))),
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
        assert_eq!(outcome.recv_timeout(Duration::from_secs(2)), Ok(true), "and it gets the engine once the build ends");
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

        assert!(asked.elapsed() >= state.config.wedge.running_after, "it waited the fallback ceiling out first");
        assert!(format!("{refused:#}").contains("rerank"), "{refused:#}");
        assert!(inflight_now(&state.engines.rerank_inflight).is_none(), "precondition: nothing had stamped it");
    }

    /// Poison healing is UNCHANGED by the deadline: a panicked load still costs ONE request, never the
    /// process. Before it existed, a live Fast pass answered "sparse engine poisoned" for hours with
    /// Succeeded=0 — and a stuck thread is a different failure precisely because it never gets here.
    #[test]
    fn a_poisoned_engine_lock_still_heals_under_the_deadline() {
        let engine: Mutex<Option<u8>> = Mutex::new(Some(7));
        let panicked = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _held = engine.lock().expect("fresh lock");
            panic!("load blew up mid-flight");
        }));
        assert!(panicked.is_err() && engine.is_poisoned(), "precondition: the panic poisoned the lock");

        let inflight = Mutex::new(None);
        let guard = lock_or_refuse(&engine, &inflight, "test", test_wedge_policy(), Patience::UntilTheHolderIsWedged)
            .expect("a poisoned lock heals rather than refusing");
        assert!(guard.is_none(), "half-built state is dropped so the caller reloads");
        drop(guard);
        assert!(engine.lock().is_ok(), "poison is cleared for every later request");
    }

    /// The three verdicts, and the ordering that matters most: the OPT-IN exit is measured from the
    /// wedge verdict, never from the phase start. An exit ceiling shorter than the (hour-long) build
    /// ceiling would otherwise kill the process mid-compile — the one action guaranteed to leave a
    /// corrupt program in the compiled-model cache, which is the 2026-07-31 incident the canary exists
    /// for.
    #[test]
    fn the_wedge_verdict_spares_a_cold_compile_and_never_exits_before_it_reports() {
        let off = WedgePolicy { exit_after_wedged: None, ..test_wedge_policy() };
        assert_eq!(wedge_action(Phase::Building, Duration::from_millis(500), off), WedgeAction::Nothing);
        assert_eq!(wedge_action(Phase::Running, Duration::from_millis(500), off), WedgeAction::Report);
        assert_eq!(
            wedge_action(Phase::Building, Duration::from_secs(600), off),
            WedgeAction::Report,
            "the exit is OPT-IN: with it off, a wedge is only ever reported"
        );

        // Opted in, with an exit ceiling far SHORTER than the build ceiling — the dangerous combination.
        let on = WedgePolicy { exit_after_wedged: Some(Duration::from_millis(100)), ..test_wedge_policy() };
        assert_eq!(
            wedge_action(Phase::Building, Duration::from_millis(550), on),
            WedgeAction::Nothing,
            "a build inside its ceiling is never exited on, however short the exit ceiling is"
        );
        assert_eq!(wedge_action(Phase::Building, Duration::from_millis(650), on), WedgeAction::Report);
        assert_eq!(wedge_action(Phase::Building, Duration::from_millis(750), on), WedgeAction::Exit);
    }

    /// The shipped ceilings, asserted from the env defaults rather than from the test config: a build
    /// gets an hour because it legitimately contains a cold compile plus a wipe-and-recompile, a pass
    /// gets 15 minutes (~1.5x the slowest honest one on record), and the process exit is OFF.
    #[test]
    fn the_shipped_ceilings_leave_room_for_a_cold_compile_and_default_to_never_exiting() {
        let shipped = Config::from_env().wedge;

        assert_eq!(shipped.ceiling(Phase::Running), Duration::from_secs(900));
        assert_eq!(shipped.ceiling(Phase::Building), Duration::from_secs(3600));
        assert!(
            shipped.ceiling(Phase::Building) > shipped.ceiling(Phase::Running),
            "a compile is slower than a pass, and conflating them is what would flag correct slowness"
        );
        assert_eq!(shipped.unload_wait, Duration::from_secs(30), "/unload answers the lease coordinator, not never");
        assert_eq!(shipped.exit_after_wedged, None, "the process exit is opt-in (WEDGE_EXIT) and OFF by default");
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

    /// The provenance READER never computes — that is the whole fix. A local cell, so the assertion is
    /// about the rule rather than about which test in this binary happened to run first.
    #[test]
    fn reading_provenance_never_computes_it() {
        let cell: OnceLock<Provenance> = OnceLock::new();

        assert!(cell.get().is_none(), "a fresh cell holds nothing");

        cell.set(Provenance { exe_sha256: "abc".to_string(), runtime_manifest_sha256: "def".to_string() })
            .map_err(|_| "already set")
            .expect("the startup task sets it once");
        assert_eq!(cell.get().map(|p| p.exe_sha256.as_str()), Some("abc"));
    }

    /// The whole point of token accounting is a TRUSTWORTHY truncation signal, so an encode failure
    /// folded to `0` tokens is the exact inversion of the contract: it reads as "measured, and
    /// definitely not truncated". Unknown must stay unknown.
    #[test]
    fn an_unencodable_text_is_reported_as_unknown_rather_than_zero_tokens() {
        let usage = usage_from_counts(vec![Some(10), None, Some(300)], 256);

        assert!(!usage.token_accounting, "one refusal makes the whole answer UNMEASURED");
        assert!(usage.token_count.is_empty() && usage.truncated.is_empty(), "and empties the arrays with it");
        assert_eq!(usage.max_length, 256, "the cap they would have been judged against still travels");

        // The refuted approach, reproduced so the defect stays visible in the suite: the shipped fold was
        // `.map(|e| e.len()).unwrap_or(0)`, which reported the refused text as 0 tokens and NOT truncated —
        // "measured, and definitely nothing lost", from the one signal the host cannot compute itself.
        let folded: Vec<usize> = [Ok(10usize), Err("refused"), Ok(300usize)]
            .into_iter()
            .map(|encoded| encoded.unwrap_or(0))
            .collect();
        assert_eq!((folded[1], folded[1] > 256), (0, false), "which is exactly what it claimed");

        // A clean batch still measures, truncation flags and all.
        let clean = usage_from_counts(vec![Some(10), Some(300)], 256);
        assert!(clean.token_accounting);
        assert_eq!(clean.truncated, vec![false, true]);
    }

    /// ~110 KB of constant text, cloned once per chunk of every pinned request. One allocation, shared.
    #[test]
    fn the_ruler_is_allocated_once_and_shared() {
        assert!(std::ptr::eq(ruler_text(), ruler_text()), "the same buffer, not a fresh copy per request");
        assert!(ruler_text().len() > 100_000, "still long enough to truncate to any cap we allow");
    }
}
