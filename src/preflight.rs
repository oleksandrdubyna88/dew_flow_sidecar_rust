use crate::config::{env_str};

// ---------- ONNX Runtime dylib preflight (load-dynamic flavor) ----------

/// Fail-fast guard for the load-dynamic (migraphx) flavor. ort rc.12's own version check DEADLOCKS
/// instead of erroring when ORT_DYLIB_PATH points at an ONNX Runtime older than the crate requires:
/// building the "not compatible" error calls `ortsys![CreateStatus]` → `ort::api()` → re-enters the
/// same `G_ORT_API` OnceLock the thread is already initializing → futex wait forever, while holding
/// our engine mutex — /health and every /embed then hang too. Probe the dylib up front through the
/// stable C ABI and exit with an actionable message before any model load can freeze the process.
#[cfg(feature = "migraphx")]
pub(crate) fn preflight_ort_dylib() {
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
pub(crate) fn preflight_migraphx_cache() {
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
pub(crate) fn seed_model_cache_from_env(cache_dir: &std::path::Path) {
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
pub(crate) struct SeededFiles {
    pub(crate) files: u64,
    pub(crate) bytes: u64,
}

/// Recursively copies every file under `from` that is missing (or size-mismatched — an interrupted
/// earlier copy) under `to`. Never deletes anything, never overwrites a same-size file, and treats
/// every per-file error as a warning rather than a failure — a half-seeded cache still works, because
/// fastembed falls back to downloading whatever is unreadable.
pub(crate) fn copy_missing_files(from: &std::path::Path, to: &std::path::Path) -> SeededFiles {
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
pub(crate) fn cache_dir_verdict(dir: &str, probe: impl FnOnce(&str) -> Result<(), String>) -> Result<String, String> {
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
pub(crate) struct OrtApiBaseAbi {
    pub(crate) get_api: unsafe extern "C" fn(u32) -> *const std::ffi::c_void,
    pub(crate) get_version_string: unsafe extern "C" fn() -> *const std::ffi::c_char,
}

/// Loads the dylib and asks it for (does it serve our API version, its version string). The dlopen
/// here is refcounted — ort's own later load reuses the mapping, so the probe costs nothing extra.
#[cfg(feature = "migraphx")]
pub(crate) fn probe_ort_dylib(path: &str) -> Result<(bool, String), String> {
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
pub(crate) fn dylib_verdict(api_ok: bool, version: &str, path: &str) -> Result<String, String> {
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

#[cfg(test)]
mod tests {
    use super::*;
    
    
    
    
    
    
    use crate::testing::*;
    

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
}
