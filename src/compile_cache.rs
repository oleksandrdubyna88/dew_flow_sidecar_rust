use std::sync::{Mutex, MutexGuard};

// The cache subdirectory each engine builds into. Shared by the builder (`CachePathLease`) and the
// measurement (`CompileWatch`) so the two cannot drift: a measurement pointed at a directory nothing
// writes to reports every compile as zero, which is indistinguishable from a healthy warm cache.
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

/// Total size of the MIGraphX compiled-model cache tree in whole MB (0 = cache not configured,
/// i.e. a non-migraphx flavor). Recursive over the per-engine subdirectories. The EP reads AND
/// writes the cache LAZILY — at the first kernel launch, not at session build — so growth is
/// measured across a PASS, never across a session build (which taught us nothing and lied
/// "served from cache" while the first pass then compiled for two minutes).
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

/// Serializes the process-global cache-path variables. Static because what it guards is static: there
/// is exactly one environment per process, however many engines want to point it somewhere.
static CACHE_PATH_LOCK: Mutex<()> = Mutex::new(());

/// A held claim on the MIGraphX compiled-model cache path, for ONE engine.
///
/// **Why the path is an environment variable at all.** It is the only knob this ROCm build honors — the
/// per-session provider-options fields were tried and are IGNORED. That is a property of this build, not
/// of the design: a per-session option would remove this hazard class instead of narrowing it, so
/// **re-test the provider-options fields whenever `ort` or ROCm is bumped**, and update this comment —
/// it is the only place that finding lives.
///
/// **Why the claim outlives the build.** The EP reads and writes that path LAZILY, at the first kernel
/// launch, not at session build. The predecessor of this type (`with_engine_cache`) released at the end of
/// the build, so the window between a build returning and its first pass was unprotected —
/// and `Engines.embed` / `Engines.rerank` are independent mutexes, so an embed build and a rerank build
/// legitimately run at once on two `spawn_blocking` threads, which is the ordinary situation right after a
/// restart when the host hits both endpoints. Whichever one set the variable last owned the directory that
/// BOTH engines then compiled into: the 2026-07-27 stale-cache incident (a session that loaded a program
/// cached by another engine returned mis-shaped outputs and died on `assertion failed: index < dim`),
/// whose entire fix was per-engine subdirectories, reopened through a timing gap the lock did not cover.
///
/// So the lease is taken by the caller that BUILDS, and held across the build **and that engine's first
/// pass** — see `embed_natural` and `rerank_blocking`. It costs concurrency between the two engine types
/// while one of them is cold: minutes, once, and only on a MIGraphX flavour.
///
/// **Lock order.** The engine mutex is always taken first and this lease second. Neither path ever takes
/// the other engine's mutex, so the two cannot cycle.
///
/// **Free everywhere else.** With no cache configured — every non-MIGraphX flavour — there is nothing to
/// claim: no lock is taken and no environment is touched, so holding one across a pass costs nothing.
pub(crate) struct CachePathLease {
    engine: &'static str,
    /// This engine's slice; empty when no cache is configured.
    dir: String,
    /// `None` when no cache is configured — there is no process-global state to serialize on.
    _held: Option<MutexGuard<'static, ()>>,
}

impl CachePathLease {
    pub(crate) fn hold(base: &str, engine: &'static str) -> Self {
        if base.trim().is_empty() {
            return Self { engine, dir: String::new(), _held: None };
        }

        let held = CACHE_PATH_LOCK.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        let dir = engine_cache_dir(base, engine);
        std::fs::create_dir_all(&dir).ok();
        std::env::set_var("ORT_MIGRAPHX_MODEL_CACHE_PATH", &dir);
        std::env::set_var("ORT_MIGRAPHX_CACHE_PATH", &dir);
        Self { engine, dir, _held: Some(held) }
    }

    pub(crate) fn engine(&self) -> &'static str {
        self.engine
    }

    /// This engine's cache slice — empty when none is configured, which is the same test the callers
    /// used to spell as `config.mxr_cache_base.trim().is_empty()` in three places.
    pub(crate) fn dir(&self) -> &str {
        &self.dir
    }

    pub(crate) fn covers(&self, engine: &str) -> bool {
        self.engine == engine
    }

    /// Wipes this engine's slice and re-creates it WITHOUT releasing the claim — the canary's heal path,
    /// where a corrupt compiled program has to go and exactly one clean recompile takes its place. It
    /// composes the path here rather than at the call site on purpose: the caller used to spell the engine
    /// name as a literal `"dual"`, and a name spelled twice is a name that drifts.
    pub(crate) fn wipe(&self) {
        if self.dir.is_empty() {
            return;
        }
        std::fs::remove_dir_all(&self.dir).ok();
        std::fs::create_dir_all(&self.dir).ok();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;
    use std::sync::{Arc, Barrier};
    use std::time::Duration;
    use crate::testing::*;

    /// Dense, sparse, and rerank run the SAME graph at the SAME pinned shape, so the EP's cache
    /// key collides across engines — a sparse session once loaded a program cached by another
    /// engine and died on mis-shaped outputs. Every engine must therefore get its OWN cache slice,
    /// and an unconfigured cache must claim nothing at all.
    #[test]
    fn each_engine_gets_its_own_cache_slice() {
        assert_eq!(engine_cache_dir("/cache/device-0", "sparse"), "/cache/device-0/sparse");
        assert_eq!(engine_cache_dir("/cache/device-0/", "dense"), "/cache/device-0/dense");

        let unconfigured = CachePathLease::hold("", EMBED_CACHE_ENGINE);
        assert!(unconfigured.dir().is_empty(), "no cache configured -> nothing to claim, no lock taken");
        assert!(unconfigured.covers(EMBED_CACHE_ENGINE), "and it still knows which engine it stands for");
    }

    /// The MIGraphX EP reads the cache path at its FIRST KERNEL LAUNCH, not at session build — so a claim
    /// that ends when the build returns protects the wrong interval. `Engines.embed` and `Engines.rerank`
    /// are independent mutexes, so the other engine's build is legitimately running in that window, and
    /// whichever set the variable last owned the directory both then compiled into: the 2026-07-27
    /// stale-cache incident, reopened through a timing gap.
    #[test]
    fn a_concurrent_build_cannot_change_the_cache_path_before_the_first_launch() {
        let base = std::env::temp_dir().join(format!("mxr-race-{}", std::process::id()));
        let base = base.to_string_lossy().to_string();
        let building = Arc::new(Barrier::new(2));

        let embed = {
            let (base, building) = (base.clone(), Arc::clone(&building));
            std::thread::spawn(move || {
                let cache = CachePathLease::hold(&base, EMBED_CACHE_ENGINE);
                building.wait(); // the rerank engine may now start building
                // The window the old scope left open: the session is built, the first kernel launch —
                // which is what actually reads the variable — has not happened yet.
                std::thread::sleep(Duration::from_millis(150));
                let at_first_launch = std::env::var("ORT_MIGRAPHX_MODEL_CACHE_PATH").unwrap_or_default();
                drop(cache);
                at_first_launch
            })
        };
        let rerank = {
            let (base, building) = (base.clone(), Arc::clone(&building));
            std::thread::spawn(move || {
                building.wait();
                let _cache = CachePathLease::hold(&base, RERANK_CACHE_ENGINE);
                std::thread::sleep(Duration::from_millis(50));
            })
        };

        let seen = embed.join().expect("the embed build");
        rerank.join().expect("the rerank build");

        assert_eq!(
            seen,
            engine_cache_dir(&base, EMBED_CACHE_ENGINE),
            "the first kernel launch must still read the slice its own session was built under"
        );

        // The retired narrow scope, reproduced so the defect stays visible in the suite: when the claim
        // ends with the BUILD, the other engine's build flips the variable before the first launch reads
        // it — and the launch then compiles into, or loads from, a directory that is not its own.
        {
            let _built_under = CachePathLease::hold(&base, EMBED_CACHE_ENGINE);
        } // <- where `with_engine_cache` returned
        let _other_engine_builds = CachePathLease::hold(&base, RERANK_CACHE_ENGINE);
        assert_eq!(
            std::env::var("ORT_MIGRAPHX_MODEL_CACHE_PATH").unwrap_or_default(),
            engine_cache_dir(&base, RERANK_CACHE_ENGINE),
            "which is what an embed engine used to read at its first launch"
        );

        std::fs::remove_dir_all(&base).ok();
    }

    /// The heal path wipes a corrupt compiled program and lets exactly one clean recompile take its
    /// place — still holding the claim, because dropping it there would hand the fresh compile to
    /// whichever engine grabbed the variable next.
    #[test]
    fn wiping_a_slice_leaves_it_present_and_empty_and_never_touches_the_other_engine() {
        let base = std::env::temp_dir().join(format!("mxr-wipe-{}", std::process::id()));
        write_mb(&base.join(EMBED_CACHE_ENGINE).join("corrupt.mxr"), 1);
        write_mb(&base.join(RERANK_CACHE_ENGINE).join("healthy.mxr"), 1);
        let base = base.to_string_lossy().to_string();

        let cache = CachePathLease::hold(&base, EMBED_CACHE_ENGINE);
        cache.wipe();

        assert!(Path::new(cache.dir()).is_dir(), "the slice is re-created, so the recompile has somewhere to land");
        assert_eq!(mxr_cache_mb(cache.dir()), 0, "and the corrupt program is gone");
        assert_eq!(mxr_cache_mb(&engine_cache_dir(&base, RERANK_CACHE_ENGINE)), 1, "the other engine is untouched");

        drop(cache);
        std::fs::remove_dir_all(&base).ok();
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
