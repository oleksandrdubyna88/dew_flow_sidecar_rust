use std::sync::Mutex;

pub(crate) const EMBED_CACHE_ENGINE: &str = "dual";
pub(crate) const RERANK_CACHE_ENGINE: &str = "rerank";

/// Watches ONE engine's compiled-model cache across a pass.
///
/// Scoped to that engine's own subdirectory, and both halves of that matter:
///
/// - **Correctness.** `with_engine_cache` has given each engine its own subdirectory since the
///   2026-07-27 stale-cache incident, but the measurement kept summing the WHOLE tree — so a rerank
///   compile running concurrently with an embed pass was reported as that pass's growth. The two engines
///   hold independent mutexes, so concurrent is the ordinary case right after a restart.
/// - **Cost.** On the MIGraphX flavour that tree is multi-GB across engine subdirectories, and this is
///   walked twice per request. One engine's slice is a fraction of it.
///
/// The walk itself stays. Measuring only when "a compile could have happened" was tried and recorded as
/// a lie (`mxr_cache_mb` — the EP saves LAZILY, at the first kernel launch, so a fresh-build flag does
/// not predict when bytes land, and a build-scoped measurement claimed "served from cache" while the
/// first pass then compiled for two minutes). On every non-MIGraphX flavour the base is empty and the
/// whole thing costs a `trim().is_empty()`.
pub(crate) struct CompileWatch {
    /// Empty = no cache configured, i.e. a non-migraphx flavour. `mxr_cache_mb` then answers 0 without
    /// touching the filesystem.
    pub(crate) dir: String,
    pub(crate) before: u64,
}

impl CompileWatch {
    pub(crate) fn start(base: &str, engine: &str) -> Self {
        let dir = if base.trim().is_empty() { String::new() } else { engine_cache_dir(base, engine) };
        Self { before: mxr_cache_mb(&dir), dir }
    }

    /// Megabytes this engine's cache grew during the pass — the ONLY moment a MIGraphX compile is
    /// observable, since the EP writes lazily rather than at session build.
    pub(crate) fn grew_mb(&self) -> u64 {
        mxr_cache_mb(&self.dir).saturating_sub(self.before)
    }
}

pub(crate) fn mxr_cache_mb(base: &str) -> u64 {
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
pub(crate) fn pass_log_message(request_id: &str, action: &str, secs: f32, cache_grew_mb: u64) -> String {
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
pub(crate) fn engine_cache_dir(base: &str, engine: &str) -> String {
    format!("{}/{engine}", base.trim_end_matches('/'))
}

/// Redirects the EP's cache into the engine's own subdirectory for the duration of a session
/// build. The path travels via process env (the only knob this ROCm build honors — it ignores the
/// provider-options fields), so builds are serialized by a lock: two engines building at once
/// would race the variable. No-op (straight call) when no cache is configured.
pub(crate) fn with_engine_cache<T>(base: &str, engine: &str, build: impl FnOnce() -> anyhow::Result<T>) -> anyhow::Result<T> {
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;
    use crate::testing::*;

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

    /// A compile by ONE engine must not be reported as growth by the other.
    ///
    /// `with_engine_cache` has given each engine its own subdirectory since the 2026-07-27 stale-cache
    /// incident, but the pass measurement kept summing the WHOLE tree. `Engines.embed` and
    /// `Engines.rerank` are independent mutexes, so an embed pass running while the rerank engine
    /// compiles is the ordinary situation right after a restart — and it reported the other engine's
    /// megabytes as its own, on the one field that exists to say "a compile happened here".
    #[test]
    fn a_compile_by_one_engine_is_not_reported_as_growth_by_the_other() {
        let base = std::env::temp_dir().join(format!("mxr-scope-{}", std::process::id()));
        write_mb(&base.join(EMBED_CACHE_ENGINE).join("warm.mxr"), 1);
        write_mb(&base.join(RERANK_CACHE_ENGINE).join("warm.mxr"), 1);
        let base = base.to_string_lossy().to_string();

        let embed_pass = CompileWatch::start(&base, EMBED_CACHE_ENGINE);
        // The OTHER engine compiles while this pass runs.
        write_mb(Path::new(&base).join(RERANK_CACHE_ENGINE).join("fresh.mxr").as_path(), 3);

        assert_eq!(embed_pass.grew_mb(), 0, "another engine's compile is not this pass's growth");

        // ...and this engine's own compile still is.
        write_mb(Path::new(&base).join(EMBED_CACHE_ENGINE).join("fresh.mxr").as_path(), 2);
        assert_eq!(embed_pass.grew_mb(), 2, "its own compile is exactly what it reports");

        // The refuted approach, reproduced so the defect stays visible in the suite: summing the whole
        // tree charges this pass with all 7 MB, 5 of which it did not cause.
        assert_eq!(mxr_cache_mb(&base), 7, "which is what the pass used to measure");

        std::fs::remove_dir_all(&base).ok();
    }

    /// Every non-MIGraphX flavour configures no cache, and must not pay a filesystem walk to learn that.
    /// An empty base must stay empty rather than becoming `/dual`, which is a real directory to stat.
    #[test]
    fn a_flavour_with_no_cache_never_walks_anything() {
        let watch = CompileWatch::start("", EMBED_CACHE_ENGINE);

        assert!(watch.dir.is_empty(), "no path was composed from an empty base");
        assert_eq!(watch.grew_mb(), 0, "and nothing is ever reported");
    }
}
