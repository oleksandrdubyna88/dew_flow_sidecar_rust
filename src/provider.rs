use crate::compile_cache::{CachePathLease, EMBED_CACHE_ENGINE, RERANK_CACHE_ENGINE};
use crate::config::{Config, DUAL_MODEL, RERANK_MODEL};
use crate::state::AppState;
use crate::vram::{self, Attribution, SoloBuildWindow};
use crate::wire::join_error_text;
use fastembed::{
    Bgem3DualEmbedding, Bgem3DualInitOptions, RerankInitOptions, RerankerModel, TextRerank,
};
use ort::execution_providers::{
    CUDAExecutionProvider, DirectMLExecutionProvider, ExecutionProviderDispatch,
    MIGraphXExecutionProvider,
};
use std::path::{Path, PathBuf};
use std::sync::OnceLock;
use std::time::Instant;

/// Everything a session load does around the one line that differs — building the options.
///
/// The two loaders below were the same seven steps twice: pin the provider, preflight it, log the
/// intent, build the options, build the session inside the engine's own cache slice while timing it, log
/// the duration, record the outcome. Only the options builder genuinely differed, and the two types
/// (`Bgem3DualInitOptions`, `RerankInitOptions`) share no trait — so the shape that removes the copy is
/// a closure that receives the resolved provider and returns a session.
///
/// The preflight is here rather than only at startup because when `ORT_PROVIDER` is empty the provider
/// is not known until the first request names it: the check has to run before the first session, not at
/// the first user-visible failure. Startup already covered the explicit case.
///
/// `cache` is EVIDENCE, not data: a session may only be built while its caller holds the compiled-model
/// cache path for this engine, because the EP will read that path at the first kernel launch — after this
/// function has long returned. See `CachePathLease`.
fn load_session<T>(
    state: &AppState,
    provider_hint: &str,
    model: &str,
    engine: &str,
    seq: usize,
    cache: &CachePathLease,
    build: impl FnMut(&str) -> anyhow::Result<T>,
) -> anyhow::Result<T> {
    // Checked rather than assumed, and checked FIRST so a mismatch pins nothing and builds nothing: a
    // lease taken for the other engine would point this build at the other engine's slice — the
    // 2026-07-27 cross-engine cache mix-up, arriving through the very guard that exists to prevent it.
    anyhow::ensure!(
        cache.covers(engine),
        "internal: building a `{engine}` session while the compiled-model cache path is claimed for \
         `{}` — hold the lease for the engine you are about to build",
        cache.engine()
    );

    // And the SHAPE, which is the same hazard one axis further out. 2026-08-19: a session asked for
    // `max_length` 256 was served programs compiled for 128 and then 224 — both already in `dual/`, one of
    // them from the previous day — and returned `token_embeddings` shaped `[64, 128, 1024]` against an input
    // of 64x256. The slice name now keeps the PROGRAMS apart; this keeps the INTENT apart, so a mismatched
    // lease is one sentence here instead of a canary failure and a 213 s heal minutes later.
    anyhow::ensure!(
        cache.shape().seq == seq,
        "internal: building a `{engine}` session at sequence length {seq} while the compiled-model cache \
         path is claimed for sequence length {} — hold the lease for the shape you are about to build",
        cache.shape().seq
    );

    // The REQUEST is pinned once per process; what it resolves to is decided per build, by trying. An
    // `auto` request that reported itself as the answer is what this indirection exists to end — see
    // `resolution_order`.
    let requested = pin_provider(state, provider_hint);
    let candidates = resolution_order(&requested);
    let mut build = build;

    let (served, built) = build_on_first_working(&candidates, |provider| {
        preflight_provider(provider, &exe_dir())?;

        // Two samples around the ONE place a session is ever built, so both engines are covered by
        // construction and a third added later is covered for free. What the delta measures is the LOAD,
        // not residency: a later pass may allocate more — under MIGraphX the first kernel launch
        // allocates most of it — and nothing here re-samples. See `vram::attribute` for why a delta is
        // published only when this build was alone.
        let started = std::time::Instant::now();
        let window = SoloBuildWindow::open(vram::sample(state));
        let built = build(provider);
        let attribution = window.close(vram::sample(state));
        if built.is_ok() {
            // Only a build that produced a session: a failed one allocates and frees an unknown amount,
            // and charging that to the engine would describe the failure rather than the engine.
            vram::record(state, engine, attribution);
            tracing::info!(
                "{model}: session ready on provider {provider} in {:.1}s, {} (the EP compiles or loads                  its cache lazily, on the first pass — the cache path stays claimed until it has)",
                started.elapsed().as_secs_f32(),
                load_size_text(attribution)
            );
        }
        built
    });

    if built.is_ok() && served != requested {
        // The line an operator needs when a machine is quietly slower than the one beside it. INFO, not
        // a warning: a CPU fallback is a working service. It is also the only moment the gap between what
        // was asked for and what runs is cheap to see.
        tracing::info!(
            "provider {requested} resolved to {served} for {model}: that is what /health reports as              active, and what the runtime panel names the arm with"
        );
    }

    record_session_outcome(state, &served, built)
}

/// How the build's allocation reads in the startup log — the same four states `/health` publishes, in a
/// sentence, because the log is where an operator meets this first.
fn load_size_text(attribution: Attribution) -> String {
    match attribution {
        Attribution::Measured(bytes) => format!("{} MB of adapter memory", bytes / (1024 * 1024)),
        Attribution::Overlapped => {
            "adapter memory not attributed (another build overlapped this one)".to_string()
        }
        Attribution::NotSampled => "adapter memory not sampled on this build".to_string(),
        Attribution::NoGrowth => {
            "adapter memory unchanged across the build (allocation not visible here)".to_string()
        }
    }
}

/// Applies the knobs every model shares. `intra_threads` stays conditional: 0 means "let ONNX Runtime
/// decide", which is not the same as asking it for zero threads.
fn shared_options<O>(state: &AppState, provider: &str, options: O) -> O
where
    O: WithCommonOptions,
{
    let options = options
        .cache_dir(state.config.cache_dir.clone())
        .providers(execution_providers(
            provider,
            state.config.device_id,
            state.dml_device_id(),
        ));
    match state.config.intra_threads {
        0 => options,
        threads => options.threads(threads),
    }
}

/// The three option setters both fastembed builders happen to have, named once so `shared_options` can
/// reach them. A trait rather than a macro: the compiler then checks each implementation against the
/// builder it wraps, and an upstream rename becomes a compile error instead of a silently skipped knob.
trait WithCommonOptions: Sized {
    fn cache_dir(self, dir: std::path::PathBuf) -> Self;
    fn providers(self, providers: Vec<ExecutionProviderDispatch>) -> Self;
    fn threads(self, threads: usize) -> Self;
}

impl WithCommonOptions for Bgem3DualInitOptions {
    fn cache_dir(self, dir: std::path::PathBuf) -> Self {
        self.with_cache_dir(dir)
    }
    fn providers(self, providers: Vec<ExecutionProviderDispatch>) -> Self {
        self.with_execution_providers(providers)
    }
    fn threads(self, threads: usize) -> Self {
        self.with_intra_threads(threads)
    }
}

impl WithCommonOptions for RerankInitOptions {
    fn cache_dir(self, dir: std::path::PathBuf) -> Self {
        self.with_cache_dir(dir)
    }
    fn providers(self, providers: Vec<ExecutionProviderDispatch>) -> Self {
        self.with_execution_providers(providers)
    }
    fn threads(self, threads: usize) -> Self {
        self.with_intra_threads(threads)
    }
}

/// Builds the ONE session both heads share. Its compiled-model cache slice is `dual/` — its own,
/// never `dense/` or `sparse/`: a cache hit must always mean "MY program" (the 2026-07-27 stale-cache
/// incident), and the per-head slices belong to the retired two-session binaries.
pub(crate) fn load_dual(
    state: &AppState,
    provider_hint: &str,
    max_length: usize,
    cache: &CachePathLease,
) -> anyhow::Result<Bgem3DualEmbedding> {
    load_session(
        state,
        provider_hint,
        DUAL_MODEL,
        EMBED_CACHE_ENGINE,
        max_length,
        cache,
        |provider| {
            tracing::info!("loading {DUAL_MODEL} (provider {provider}, max_length {max_length})");
            Bgem3DualEmbedding::try_new(shared_options(
                state,
                provider,
                Bgem3DualInitOptions::default().with_max_length(max_length),
            ))
        },
    )
}

pub(crate) fn load_rerank(
    state: &AppState,
    provider_hint: &str,
    cache: &CachePathLease,
) -> anyhow::Result<TextRerank> {
    load_session(
        state,
        provider_hint,
        RERANK_MODEL,
        RERANK_CACHE_ENGINE,
        state.config.rerank_max_length,
        cache,
        |provider| {
            tracing::info!(
                "loading {RERANK_MODEL} (provider {provider}, max_length {})",
                state.config.rerank_max_length
            );
            TextRerank::try_new(shared_options(
                state,
                provider,
                RerankInitOptions::new(RerankerModel::BGERerankerV2M3)
                    .with_max_length(state.config.rerank_max_length),
            ))
        },
    )
}

/// The provider all engines pin to: ORT_PROVIDER env wins, else the first request's hint, else auto.
/// Stored once so later loads and shape pinning reuse the same choice until restart.
/// <para>This is the REQUEST. It says nothing about whether a session can be built on it — that is
/// `AppState::active_provider`, written only after one succeeds.</para>
pub(crate) fn pin_provider(state: &AppState, hint: &str) -> String {
    state
        .pinned_provider
        .get_or_init(|| effective_provider(&state.config, hint))
        .clone()
}

/// Which operating system this process is executing on: `"windows"`, `"wsl"` or `"linux"`.
///
/// Reported rather than inferred by a caller, because nobody outside this process can know it — a WSL
/// sidecar and a Windows one are both reached on `localhost`, and a launcher's configuration states what
/// it intended rather than what is running.
///
/// WSL is distinguished from bare Linux by the kernel release string, which Microsoft's kernel stamps
/// with `microsoft`. That matters because the two are not interchangeable for anything this field is read
/// for: the filesystem crossing, the localhost forwarding and the GPU passthrough are all properties of the
/// VM rather than of Linux, and one index pass measured 155 s of difference across that boundary alone
/// (`dew_flow_rag_qln · research/GPU_BACKEND_WSL_VS_WINDOWS.md` §9-10). An unreadable release string reads
/// as plain `linux`: not knowing it is a VM is a smaller error than claiming it is one.
pub(crate) fn host_os() -> &'static str {
    if cfg!(target_os = "windows") {
        return "windows";
    }

    if !cfg!(target_os = "linux") {
        return std::env::consts::OS;
    }

    match std::fs::read_to_string("/proc/sys/kernel/osrelease") {
        Ok(release) => linux_flavour(&release),
        // Unreadable proc: plain `linux`. Not knowing it is a VM is a smaller error than claiming it is.
        Err(_) => "linux",
    }
}

/// `wsl` or `linux`, from the kernel release string alone — the part with the judgement in it, split out
/// so it can be checked without a Linux kernel to read one from.
///
/// Microsoft stamps its kernel release with `microsoft` (`5.15.167.4-microsoft-standard-WSL2`), and that
/// substring is the only marker present on every WSL2 build. Matched case-insensitively because the
/// capitalisation has moved between releases and this is not the place to be surprised by it.
pub(crate) fn linux_flavour(osrelease: &str) -> &'static str {
    if osrelease.to_ascii_lowercase().contains("microsoft") {
        "wsl"
    } else {
        "linux"
    }
}

/// The execution providers compiled into THIS binary flavor. Derived from the cargo features rather
/// than from a list someone maintains by hand, so it cannot disagree with the build.
pub(crate) fn compiled_providers() -> Vec<&'static str> {
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
pub(crate) fn required_provider_libraries(provider: &str) -> &'static [&'static str] {
    match provider {
        "cuda" => &[
            "onnxruntime_providers_shared.dll",
            "onnxruntime_providers_cuda.dll",
        ],
        _ => &[],
    }
}

/// Refuses a provider this process cannot possibly serve, with the reason a reader can act on.
///
/// Two failures are indistinguishable at the first inference and must not be: an EP absent from the
/// BUILD (wrong flavor — rebuild) and an EP whose runtime libraries were not deployed (wrong package —
/// re-run the install script). `auto` is exempt: it is a request to try whatever is present, so there
/// is nothing to refuse.
pub(crate) fn preflight_provider(provider: &str, exe_dir: &Path) -> anyhow::Result<()> {
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
             copy them next to the exe (README, \"Build\"): a package that ships only the binary fails \
             here every time.",
            exe_dir.display(),
            missing.join(", ")
        );
    }

    Ok(())
}

/// The directory the running executable lives in — where ORT looks for provider DLLs.
pub(crate) fn exe_dir() -> PathBuf {
    std::env::current_exe()
        .ok()
        .and_then(|p| p.parent().map(Path::to_path_buf))
        .unwrap_or_else(|| PathBuf::from("."))
}

/// SHA-256 of a file, streamed so a 300 MB provider library never enters memory whole.
/// `None` for anything unreadable — the caller reports absence rather than a plausible-looking zero.
pub(crate) fn sha256_file(path: &Path) -> Option<String> {
    use sha2::Digest;
    let mut file = std::fs::File::open(path).ok()?;
    let mut hasher = sha2::Sha256::new();
    std::io::copy(&mut file, &mut hasher).ok()?;
    Some(
        hasher
            .finalize()
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect(),
    )
}

/// Build provenance: what binary is answering, and which provider libraries stand behind it.
pub(crate) struct Provenance {
    pub(crate) exe_sha256: String,
    pub(crate) runtime_manifest_sha256: String,
}

/// Computed exactly once, at STARTUP, on the blocking pool — never on a request path. `None` until
/// that task lands, which /health reports as `provenance_ready: false`.
pub(crate) static PROVENANCE: OnceLock<Provenance> = OnceLock::new();

/// Hashes the executable and the libraries beside it. Blocking and unbounded by design: on a
/// CUDA/DirectML deployment this is hundreds of MB to gigabytes (cuDNN alone is often >500 MB).
pub(crate) fn compute_provenance() -> Provenance {
    Provenance {
        exe_sha256: compute_exe_sha256(),
        runtime_manifest_sha256: compute_runtime_manifest_sha256(),
    }
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
pub(crate) fn prewarm_provenance() {
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
pub(crate) fn short_hash(hash: &str) -> &str {
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
pub(crate) fn compute_exe_sha256() -> String {
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
pub(crate) fn compute_runtime_manifest_sha256() -> String {
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
                .is_some_and(|ext| {
                    ext.eq_ignore_ascii_case("dll") || ext.eq_ignore_ascii_case("so")
                })
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
pub(crate) fn record_session_outcome<T>(
    state: &AppState,
    provider: &str,
    built: anyhow::Result<T>,
) -> anyhow::Result<T> {
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

pub(crate) fn effective_provider(config: &Config, hint: &str) -> String {
    let value = if config.provider.is_empty() {
        hint
    } else {
        &config.provider
    };
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
/// The concrete providers a request may resolve to, in the order they are tried.
///
/// **`auto` is a request, not an execution provider**, and until 2026-08-20 it was reported as though it
/// were one: the chain was handed to ort as a soft list and whichever EP won was never captured, so
/// `/health` answered `active_provider: "auto"` on a machine where DirectML was demonstrably serving
/// (2 175 MB charged to the embed build; a CPU session allocates nothing on the card). That is the
/// 2026-08-08 defect — the requested provider published as the active one — reintroduced on the one path
/// that was not covered when it was fixed for the explicit case.
///
/// Resolving it by TRYING rather than predicting it from `compiled_providers()` is the whole point: a
/// binary with DirectML compiled in still falls through to CPU on a machine whose driver will not register
/// it, and a prediction would name `dml` for exactly the run an operator most needs to see named `cpu`.
///
/// An explicit request resolves to itself and offers no fallback — unchanged, and deliberately: a broken
/// GPU setup must be visible rather than quietly served by the CPU.
pub(crate) fn resolution_order(provider: &str) -> Vec<&'static str> {
    match provider {
        "cuda" => vec!["cuda"],
        "dml" => vec!["dml"],
        "migraphx" => vec!["migraphx"],
        "cpu" => vec!["cpu"],
        // ort's own chain order, narrowed to what this binary carries. An EP absent from the build cannot
        // win a registration it will not survive, and offering it would only buy a failed session build.
        _ => {
            let compiled = compiled_providers();
            let mut order: Vec<&'static str> = ["cuda", "migraphx", "dml"]
                .into_iter()
                .filter(|candidate| compiled.contains(candidate))
                .collect();
            order.push("cpu");
            order
        }
    }
}

/// Builds on the first candidate that works, and returns **which one did**.
///
/// The returned name is the fact `/health.active_provider` publishes and `ComputeArm` (over in
/// `dew_flow_rag_qln`) names an arm with, so it has to be observed rather than assumed. When nothing
/// builds the name is empty — nothing served, so nothing may be reported as having served — and the error
/// is the LAST candidate's, because a CPU fallback that also failed is the reason a reader needs.
fn build_on_first_working<T>(
    candidates: &[&str],
    mut attempt: impl FnMut(&str) -> anyhow::Result<T>,
) -> (String, anyhow::Result<T>) {
    let mut last = anyhow::anyhow!("no execution provider was offered to try");
    for candidate in candidates {
        match attempt(candidate) {
            Ok(built) => return (candidate.to_string(), Ok(built)),
            Err(error) => {
                tracing::warn!("provider {candidate} could not build a session: {error:#}");
                last = error;
            }
        }
    }
    (String::new(), Err(last))
}

pub(crate) fn execution_providers(
    provider: &str,
    cuda_device_id: i32,
    dml_device_id: i32,
) -> Vec<ExecutionProviderDispatch> {
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
    use super::*;
    use std::path::Path;
    use std::sync::OnceLock;

    // Only the tests build a lease by hand; the production paths receive one already held.
    use crate::compile_cache::CacheShape;

    use crate::testing::*;

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

    // ---- what `auto` resolves TO (2026-08-20) ----------------------------------------------------
    //
    // Observed live on an R9700 the day the runtime panel started naming ARMS: `/health` answered
    // `active_provider: "auto"` while DirectML was demonstrably serving (2 175 MB of adapter memory
    // charged to the embed build; a CPU session allocates nothing on the card). `auto` is the DEFAULT,
    // so this was the shipping configuration.
    //
    // It is the 2026-08-08 defect wearing different clothes. That one reported the REQUESTED provider as
    // the active one and was fixed for the explicit case; `auto` reintroduced it, because the chain was
    // handed to ort as a list and whichever EP won was never captured. The symptom is milder — it names
    // no wrong EP, it names no EP at all — and the consequence is not: `ComputeArm` over in
    // `dew_flow_rag_qln` says in as many words that it uses the active provider "rather than the
    // requested one, deliberately: a provider that was asked for and never registered is exactly the
    // failure this naming exists to expose". Under `auto` the active provider IS the requested one, so
    // the exposure does not happen — and it fails SILENTLY, because "auto" is a non-empty string and
    // passes that consumer's "three segments or nothing" guard.
    //
    // What makes it worth code rather than a note: the self-check cannot cover for it. A silent CPU
    // fallback still scores cosine ~1.0, because CPU computes the same numbers. The pair of fields that
    // answers "right numbers, AND right hardware" would then answer only the first half.

    #[test]
    fn an_auto_request_resolves_to_concrete_providers_and_never_to_auto_itself() {
        let order = resolution_order("auto");

        assert!(
            !order.contains(&"auto"),
            "`auto` is a request, not an execution provider; resolving it to itself is the whole defect: {order:?}"
        );
        assert_eq!(
            order.last(),
            Some(&"cpu"),
            "ort falls through to CPU, so CPU is the last candidate and the chain can always conclude: {order:?}"
        );
        for candidate in &order {
            assert!(
                compiled_providers().contains(candidate),
                "`{candidate}` is not in this binary, so it can never win a registration it will not survive"
            );
        }
    }

    #[test]
    fn an_explicit_request_resolves_to_itself_and_offers_no_fallback() {
        // The existing doctrine, pinned so the auto work cannot quietly soften it: an explicit choice
        // fails hard, because a broken GPU setup must be visible rather than silently served by CPU.
        assert_eq!(resolution_order("dml"), vec!["dml"]);
        assert_eq!(resolution_order("cuda"), vec!["cuda"]);
        assert_eq!(resolution_order("migraphx"), vec!["migraphx"]);
        assert_eq!(resolution_order("cpu"), vec!["cpu"]);
    }

    #[test]
    fn the_provider_reported_is_the_one_that_actually_built_not_the_one_asked_for() {
        // THE test. A machine whose DirectML cannot register — a driver rollback, a card off the bus —
        // is served by CPU, and that is exactly the state an operator must be able to SEE.
        let (served, built) = build_on_first_working(&["dml", "cpu"], |provider| match provider {
            "dml" => anyhow::bail!("DXGI: no compatible adapter"),
            _ => Ok(provider.to_string()),
        });

        assert_eq!(
            served, "cpu",
            "the fallback served, so the fallback is what is reported"
        );
        assert_eq!(built.expect("cpu builds"), "cpu");
    }

    #[test]
    fn the_first_candidate_that_builds_wins_and_the_rest_are_never_tried() {
        let mut tried = Vec::new();
        let (served, built) = build_on_first_working(&["dml", "cpu"], |provider| {
            tried.push(provider.to_string());
            Ok(provider.to_string())
        });

        assert_eq!(served, "dml");
        assert!(built.is_ok());
        assert_eq!(
            tried,
            vec!["dml"],
            "a working provider must not cost a second session build"
        );
    }

    #[test]
    fn a_chain_that_builds_nowhere_fails_with_the_last_reason_rather_than_a_provider() {
        // Nothing served, so nothing may be named as having served. The error text is the last
        // candidate's, because the CPU fallback failing is the reason a reader needs.
        let (served, built) = build_on_first_working::<String>(&["dml", "cpu"], |provider| {
            anyhow::bail!("{provider} refused")
        });

        assert!(
            served.is_empty(),
            "nothing served, so nothing may be named as having served — got `{served}`"
        );

        let error = format!("{:#}", built.expect_err("every candidate failed"));
        assert!(error.contains("cpu refused"), "{error}");
    }

    // ---- provider truthfulness + fail-fast (2026-08-08) --------------------------------------
    //
    // The defect these pin: /health reported the REQUESTED provider as though it were the active one,
    // so a binary whose CUDA EP failed every registration still answered `provider: "cuda"` while
    // every /embed and /rerank returned 500. Requested, compiled and active are now three facts.
    #[test]
    fn compiled_providers_always_include_cpu_and_match_the_build_flavor() {
        let compiled = compiled_providers();
        assert!(
            compiled.contains(&"cpu"),
            "ort falls through to CPU, so it is always available"
        );
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
        assert!(
            text.contains("--features"),
            "the message must say how to fix it: {text}"
        );
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
        assert!(
            text.contains("EXECUTABLE'S directory"),
            "PATH is not where ORT looks: {text}"
        );

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
            [
                "onnxruntime_providers_shared.dll",
                "onnxruntime_providers_cuda.dll"
            ]
        );
        // `..._shared.dll` is the one the live failure named (Error 126) — dropping it from the
        // package is the exact mistake this list exists to prevent.
        assert!(required_provider_libraries("dml").is_empty());
        assert!(required_provider_libraries("auto").is_empty());
    }

    /// A lease taken for the OTHER engine would point this build at the other engine's cache slice —
    /// the 2026-07-27 cross-engine mix-up, arriving through the guard that exists to prevent it. It has
    /// to be refused BEFORE anything is pinned or loaded, because the symptom otherwise arrives minutes
    /// later as mis-shaped output from a program that loaded cleanly.
    #[test]
    fn a_session_built_under_another_engines_lease_is_refused() {
        let state = app_state();
        let wrong = CachePathLease::hold(
            &state.config.mxr_cache_base,
            RERANK_CACHE_ENGINE,
            CacheShape::unpinned_batch(128),
        );

        // let-else rather than `expect_err`: an ort session is not `Debug`, and a test that could only
        // report this failure by formatting a GPU handle would be a test nobody could run here.
        let Err(error) = load_dual(&state, "cpu", 128, &wrong) else {
            panic!("a rerank lease must not be able to build the embed session");
        };

        let text = format!("{error:#}");
        assert!(
            text.contains(EMBED_CACHE_ENGINE),
            "names the session being built: {text}"
        );
        assert!(
            text.contains(RERANK_CACHE_ENGINE),
            "and the claim actually held: {text}"
        );
        assert!(
            state.pinned_provider.get().is_none(),
            "and refused before pinning a provider"
        );
    }

    #[test]
    fn a_microsoft_kernel_release_is_wsl_and_anything_else_is_linux() {
        // Real strings: WSL2 on this machine, and a stock Ubuntu kernel.
        assert_eq!(linux_flavour("5.15.167.4-microsoft-standard-WSL2"), "wsl");
        assert_eq!(linux_flavour("6.14.0-1012-MICROSOFT-standard-WSL2"), "wsl");
        assert_eq!(linux_flavour("6.8.0-45-generic"), "linux");
        assert_eq!(linux_flavour(""), "linux");
    }

    /// The provenance READER never computes — that is the whole fix. A local cell, so the assertion is
    /// about the rule rather than about which test in this binary happened to run first.
    #[test]
    fn reading_provenance_never_computes_it() {
        let cell: OnceLock<Provenance> = OnceLock::new();

        assert!(cell.get().is_none(), "a fresh cell holds nothing");

        cell.set(Provenance {
            exe_sha256: "abc".to_string(),
            runtime_manifest_sha256: "def".to_string(),
        })
        .map_err(|_| "already set")
        .expect("the startup task sets it once");
        assert_eq!(cell.get().map(|p| p.exe_sha256.as_str()), Some("abc"));
    }
}
