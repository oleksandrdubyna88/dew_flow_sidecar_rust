use crate::wedge::WedgePolicy;
use std::path::PathBuf;

pub(crate) const DENSE_MODEL: &str = "BAAI/bge-m3 (dense, FP32)";
pub(crate) const SPARSE_MODEL: &str = "BAAI/bge-m3 (learned sparse, FP32)";
/// What actually loads: ONE session whose run serves both of the above heads.
pub(crate) const DUAL_MODEL: &str = "BAAI/bge-m3 (dense+sparse heads, FP32, one session)";
pub(crate) const RERANK_MODEL: &str = "bge-reranker-v2-m3";

/// Runtime knobs, all env-driven (the AppHost injects them; sane defaults for a bare `cargo run`).
#[derive(Clone)]
pub(crate) struct Config {
    /// auto | cuda | dml | migraphx | cpu. Empty = decide from the first request hint / auto.
    pub(crate) provider: String,
    pub(crate) device_id: i32,
    /// Default embedding batch size when a request does not carry one. Attention is O(batch × seq²) —
    /// see the VRAM budget in the module header before raising it.
    pub(crate) max_batch: usize,
    /// 0 = ONNX Runtime decides.
    pub(crate) intra_threads: usize,
    /// Default token cap per sequence when a request does not carry one. THE memory driver: cost grows
    /// with its SQUARE. 1024 tokens (~4k chars) covers essentially every method body we embed.
    pub(crate) embed_max_length: usize,
    pub(crate) rerank_max_length: usize,
    pub(crate) cache_dir: PathBuf,
    /// Where Qwen's `tokenizer.json` lives (`QWEN_TOKENIZER`). Counting only — no Qwen model is ever loaded.
    pub(crate) qwen_tokenizer_path: PathBuf,
    /// The most texts one `/tokenize` call may carry.
    ///
    /// Not a memory limit — a RUNTIME limit. The encode is pure CPU, and axum's body cap does not bound
    /// the ROW count: a 10,000-file pass sent 52,617 texts in one request, and enough of them are short
    /// that such a batch fits under 2 MB while still being tens of thousands of encodes.
    pub(crate) tokenize_max_texts: usize,
    /// The largest request body any route accepts.
    ///
    /// Explicit on purpose. Every route ran on axum's DEFAULT 2 MB cap, which meant the limit was real,
    /// undocumented and unreadable — and it fails in the worst possible shape: the server rejects the
    /// body while the client is still writing it, so the client sees "an established connection was
    /// aborted by the software in your host machine", which names the socket and not the cause. A
    /// 10,000-file repository died nine minutes into an indexing pass that way. Same number as before,
    /// so nothing changes but the fact that it is now a decision.
    pub(crate) max_body_bytes: usize,
    /// Force every embedding batch into ONE tensor shape (see `pin_shape`). "auto" (the default)
    /// turns it on only for MIGraphX, the provider that compiles per shape; "1"/"0" force it.
    pub(crate) pin_input_shape: String,
    /// The MIGraphX compiled-model cache root as the AppHost passed it (empty = no cache /
    /// non-migraphx flavor). Engine builds redirect the EP into a PER-ENGINE subdirectory of it —
    /// see `with_engine_cache` for why sharing one directory corrupted the sparse engine.
    pub(crate) mxr_cache_base: String,
    /// How many sequence caps keep their engines resident at once (`RungCache` capacity). 2 = the
    /// ladder's own maximum, so a pass never rebuilds a rung it will return to; 1 restores the old
    /// evict-on-change behaviour.
    pub(crate) engine_cache_rungs: usize,
    /// When a wait stops being "slow but alive" — see `WedgePolicy`.
    pub(crate) wedge: WedgePolicy,
}

impl Config {
    pub(crate) fn from_env() -> Self {
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
            qwen_tokenizer_path: PathBuf::from(env_str(
                "QWEN_TOKENIZER",
                "../qwen-tokenizer/tokenizer.json",
            )),
            // 4096 — eight times the host's own per-request row budget (`SidecarClient.RequestRowBudget`
            // = 512, dew_flow_rag_qln), so this is a backstop against a pathological caller and never a
            // wall a normal pass walks into. A cap at the host's number would refuse batches its only
            // caller assembles by design.
            tokenize_max_texts: env_parse("TOKENIZE_MAX_TEXTS", 4096),
            // 2 MiB — axum's own default, kept exactly. Measured against this sidecar from the host side
            // (2026-08-15): 980 KB to /tokenize succeeds, 2.1 MB comes back 413. Writing it down changes
            // no behaviour; it makes the number readable, configurable, and reportable on /health, which
            // is what an operator raising EMBED_MAX_LENGTH toward 8192 needs before they hit it.
            max_body_bytes: env_parse("MAX_BODY_BYTES", 2 * 1024 * 1024),
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

pub(crate) fn env_str(key: &str, default: &str) -> String {
    std::env::var(key)
        .ok()
        .filter(|v| !v.trim().is_empty())
        .unwrap_or_else(|| default.to_string())
}

pub(crate) fn env_parse<T: std::str::FromStr>(key: &str, default: T) -> T {
    std::env::var(key)
        .ok()
        .and_then(|v| v.trim().parse().ok())
        .unwrap_or(default)
}

/// An opt-in switch, spelled the way every other boolean knob here is (`PIN_INPUT_SHAPE`). Absent
/// or unrecognised means OFF — a switch whose default is "maybe" is not a switch.
pub(crate) fn env_truthy(key: &str) -> bool {
    matches!(
        env_str(key, "").trim().to_lowercase().as_str(),
        "1" | "true" | "on" | "yes"
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The cap has to clear the host's own ceiling with room to spare, or the sidecar refuses batches its
    /// only caller assembles by design (`SidecarClient.RequestRowBudget` = 512, dew_flow_rag_qln).
    #[test]
    fn the_default_batch_cap_leaves_room_above_the_hosts_own_row_budget() {
        assert!(
            Config::from_env().tokenize_max_texts >= 512 * 2,
            "a cap at or below the host's 512-row budget turns a normal pass into a wall of 400s"
        );
    }
}
