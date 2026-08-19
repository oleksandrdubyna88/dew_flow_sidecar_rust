use crate::adapters;
use crate::config::Config;
use crate::engine_cache::RungCache;
use crate::tokens::{TokenizerRegistry, BGE_TOKENIZER};
use crate::wedge::InFlight;
use fastembed::{Bgem3DualEmbedding, TextRerank};
use std::sync::{Mutex, OnceLock};

/// Lazily-loaded model engines. Each is guarded by its own mutex: the GPU serializes inference
/// anyway, and the lock makes the first-use load race-free. Loads happen inside spawn_blocking.
/// BOTH embedding heads live in ONE `Bgem3DualEmbedding` per rung (see `RungCache` and
/// research/module_inference.md — the official export returns both heads from one
/// forward pass, so two sessions doubled every cost for nothing); rerank runs at one fixed cap, so
/// it has nothing to key on.
pub(crate) struct Engines {
    pub(crate) embed: Mutex<RungCache<Bgem3DualEmbedding>>,
    /// What holds `embed` right now — read WITHOUT taking `embed`, which is the only way a wedged
    /// holder can ever be observed. See `InFlight`.
    pub(crate) embed_inflight: Mutex<Option<InFlight>>,
    pub(crate) rerank: Mutex<Option<TextRerank>>,
    /// The same, for the reranker: the two engines have separate mutexes and can be in flight at once,
    /// so one shared stamp would let a rerank overwrite the record of a wedged embed.
    pub(crate) rerank_inflight: Mutex<Option<InFlight>>,
}

impl Engines {
    pub(crate) fn new(cache_rungs: usize) -> Self {
        Self {
            embed: Mutex::new(RungCache::new(cache_rungs)),
            embed_inflight: Mutex::new(None),
            rerank: Mutex::new(None),
            rerank_inflight: Mutex::new(None),
        }
    }
}

pub(crate) struct AppState {
    pub(crate) config: Config,
    pub(crate) engines: Engines,
    /// What the sidecar is doing RIGHT NOW ("idle", "dense: building session…"), surfaced by
    /// /health — the operator's window into multi-minute engine builds that otherwise look like
    /// a hang from the outside.
    pub(crate) activity: Mutex<String>,
    /// The provider every engine PINS to, decided once (ORT_PROVIDER, else the first request's hint,
    /// else auto) and reused until restart. This is a REQUEST, not an outcome: it is set the moment the
    /// choice is made, before any session exists. Shape pinning keys off it — that decision has to be
    /// taken before the engines see any text, so it cannot wait for a session.
    pub(crate) pinned_provider: OnceLock<String>,
    /// The provider an ORT session was ACTUALLY created with — written only after a build SUCCEEDS.
    /// `None` while no session has been built, which is the state the old single field could not
    /// express: it reported the request as though it were the outcome, so a binary whose CUDA EP
    /// failed every registration still answered `provider: "cuda"` (measured 2026-08-08).
    pub(crate) active_provider: Mutex<Option<String>>,
    /// Why the last EP registration failed, verbatim from ort. Kept so /health can explain a
    /// `provider_ready: false` instead of merely asserting it.
    pub(crate) last_provider_error: Mutex<Option<String>>,
    /// The sequence cap this process is COMMITTED to: the most recently used resident rung, or the one
    /// a build is materialising right now. `0` = none. What a `query` inherits (`cap_for`) and what
    /// /health reports as `loaded_embed_max_length`.
    ///
    /// It is a mirror of `Engines.embed`'s occupancy, maintained under that engine's lock by
    /// `commit_cap` / `settle_cap`, and read WITHOUT it. Both halves are the design:
    ///
    /// - **Read without the lock**, because a query arriving during somebody else's pass is the ordinary
    ///   case, and blocking a cap decision on the engine mutex is the rule /health follows everywhere
    ///   else (`loaded_now`'s try_lock) applied to the one decision that has to happen before the lock.
    /// - **Written only where residency changes**, because this used to be a cell every request stamped
    ///   with what it ASKED for, before any build. A request that was then refused (503) or failed its
    ///   canary still left its cap here, and the next `query` inherited a cap no engine had ever been
    ///   built at — building it, and at the shipped one-rung capacity evicting the rung the pass was
    ///   using. The same conflation the requested/active provider split exists to prevent, one field over.
    pub(crate) committed_embed_cap: std::sync::atomic::AtomicUsize,
    /// The batch the most recent embed actually ran at — the request's override, not the config default.
    /// Reported by /health as `loaded_max_batch`, which is what makes the configured value readable as a
    /// default rather than as a fact: every request carries the operator's own batch, so the configured
    /// number described an intention nobody was running.
    pub(crate) loaded_max_batch: Mutex<Option<usize>>,
    /// The width of the dense vectors this process has actually produced — MEASURED from a returned row,
    /// never a constant.
    ///
    /// `None` until an embed has run, and `/models` reports that as UNKNOWN rather than guessing 1024.
    /// A constant here would be a fact living in two repositories with nothing to keep them equal, and
    /// the failure it produces is a vector collection created at the wrong width — which does not fail
    /// until something tries to store into it.
    pub(crate) loaded_embed_dimension: Mutex<Option<usize>>,
    /// The DXGI adapter ORT_DEVICE_ID resolves to (None = mapping unavailable — raw id fallback).
    pub(crate) adapter: Option<adapters::ResolvedAdapter>,
    /// What building each engine cost on that adapter, where it could be attributed to ONE engine —
    /// see `vram::VramLedger`. Never a residency reading: nothing here re-samples after the build.
    pub(crate) vram: Mutex<crate::vram::VramLedger>,
    /// Every tokenizer this build can COUNT with, resolved at startup — see `TokenizerRegistry`.
    ///
    /// Deliberately a wider set than the models this process embeds: the semantic channel runs on Ollama,
    /// but nothing on that side can read a HuggingFace `tokenizer.json` at all (`Microsoft.ML.Tokenizers`
    /// has no regex pre-tokenizer and no NFC normalizer), so counting in C# would have meant transcribing
    /// a pre-tokenizer regex and its byte-level rules by hand — precisely the silent near-miss this
    /// accounting exists to prevent. Here the reference implementation reads the file verbatim.
    pub(crate) tokenizers: TokenizerRegistry,
}

/// The per-request memory envelope: how long a single sequence may get and how many run together.
#[derive(Clone, Copy)]
pub(crate) struct Limits {
    pub(crate) max_length: usize,
    pub(crate) max_batch: usize,
}

impl Limits {
    /// A request may override either knob; 0/absent means "use the configured default". `max_length` is
    /// clamped to the model's own 8192 ceiling so a bad setting cannot ask for a shape it cannot run.
    pub(crate) fn resolve(config: &Config, max_length: usize, max_batch: usize) -> Self {
        Self {
            max_length: positive_or(max_length, config.embed_max_length).clamp(16, 8192),
            max_batch: positive_or(max_batch, config.max_batch).max(1),
        }
    }
}

pub(crate) fn positive_or(value: usize, fallback: usize) -> usize {
    if value == 0 {
        fallback
    } else {
        value
    }
}

impl AppState {
    /// The index the DirectML EP receives: the mapped plain-enumeration index when the DXGI
    /// resolution succeeded, else the raw configured id (the pre-mapping behaviour).
    pub(crate) fn dml_device_id(&self) -> i32 {
        self.adapter
            .as_ref()
            .map_or(self.config.device_id, |a| a.dml_device_id)
    }

    /// BGE-M3's counter — what `/embed`'s own truncation accounting counts with. Resolved through the
    /// registry so there is exactly ONE place in this process a tokenizer can come from.
    pub(crate) fn token_counter(&self) -> Option<&tokenizers::Tokenizer> {
        self.tokenizers
            .entry(BGE_TOKENIZER)
            .and_then(|entry| entry.tokenizer.as_ref())
    }
}
