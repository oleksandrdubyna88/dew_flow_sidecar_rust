//! Shared test fixtures.
//!
//! One module rather than a copy per test block: `app_state`, the wedge policy and the
//! tokenizer fixture are used from several modules, and a fixture that drifts between
//! copies is a test that stops describing the same thing as its neighbour.

#![cfg(test)]

use crate::config::Config;
use crate::engine_cache::RungCache;
use crate::state::{AppState, Engines};
use crate::tokens::{TokenizerRegistry, TokenizerSource};
use crate::wedge::{InFlight, Phase, WedgePolicy};
use crate::wire::ModelEntry;
use std::path::{Path, PathBuf};
use std::sync::{mpsc, Arc, Mutex, OnceLock};
use std::time::{Duration, Instant};

/// Wedge ceilings scaled down to milliseconds so the tests measure the RULE rather than the clock.
/// The shipped values (15 min / 60 min / 30 s) are asserted separately, from the env defaults.
pub(crate) fn test_wedge_policy() -> WedgePolicy {
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
pub(crate) fn stamped(phase: Phase, label: &str, ago: Duration) -> InFlight {
    InFlight {
        phase,
        label: label.to_string(),
        since: Instant::now()
            .checked_sub(ago)
            .expect("the process started after the epoch"),
    }
}

/// A sidecar state with no engines and no GPU behind it. Every reliability test here exercises
/// the LOCKS and the probe path, never the model, so an empty state is the whole subject.
pub(crate) fn app_state() -> Arc<AppState> {
    app_state_with(config(""))
}

/// The same empty state over a CHOSEN config — what the tokenizer tests need, since the whole
/// subject there is which files the configuration points at.
pub(crate) fn app_state_with(config: Config) -> Arc<AppState> {
    let tokenizers = TokenizerRegistry::load(&config);
    app_state_full(config, tokenizers)
}

/// A state registering the given names with NO file behind any of them. Routing — which name is
/// known, which is refused, and what the refusal says — is decided before a tokenizer is ever
/// touched, so the files are beside the point for those tests.
pub(crate) fn app_state_with_tokenizers(names: &[&'static str]) -> Arc<AppState> {
    let tokenizers = TokenizerRegistry::from_sources(
        names
            .iter()
            .map(|&name| TokenizerSource {
                name,
                path: None,
                consequence: "a test row",
            })
            .collect(),
    );
    app_state_full(config(""), tokenizers)
}

pub(crate) fn app_state_full(config: Config, tokenizers: TokenizerRegistry) -> Arc<AppState> {
    Arc::new(AppState {
        engines: Engines::new(2),
        config,
        activity: Mutex::new("idle".to_string()),
        pinned_provider: OnceLock::new(),
        active_provider: Mutex::new(None),
        last_provider_error: Mutex::new(None),
        loaded_embed_max_length: Mutex::new(None),
        loaded_max_batch: Mutex::new(None),
        loaded_embed_dimension: Mutex::new(None),
        adapter: None,
        tokenizers,
    })
}

/// Holds an engine mutex from another thread until it is told to let go — the shape of a thread
/// WEDGED inside the ORT/MIGraphX C++ call, which is the one failure the poison healing can never
/// see: a stuck thread does not panic, so the mutex is never poisoned, only never released.
pub(crate) struct HeldEngine {
    release: mpsc::Sender<()>,
    thread: Option<std::thread::JoinHandle<()>>,
}

impl HeldEngine {
    pub(crate) fn hold<S: Send + 'static>(
        engine: Arc<AppState>,
        pick: fn(&AppState) -> &Mutex<S>,
    ) -> Self {
        let (release, released) = mpsc::channel::<()>();
        let (holding, held) = mpsc::channel::<()>();
        let thread = std::thread::spawn(move || {
            let _guard = pick(&engine).lock().expect("fresh lock");
            holding.send(()).expect("announce the hold");
            released.recv().ok();
        });
        held.recv().expect("the engine is held");
        Self {
            release,
            thread: Some(thread),
        }
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

pub(crate) fn config(provider: &str) -> Config {
    Config {
        provider: provider.to_string(),
        device_id: 0,
        max_batch: 4,
        intra_threads: 0,
        embed_max_length: 1024,
        rerank_max_length: 1024,
        // Deliberately paths that do NOT exist. A unit state must not resolve differently on a
        // machine that happens to have a model cache beside the source — and now that the registry
        // loads eagerly, pointing at the real `.model-cache` would parse a 17 MB tokenizer in every
        // test that builds a state. The tokenizer tests point this at their own fixture.
        cache_dir: PathBuf::from("no-model-cache-in-tests"),
        qwen_tokenizer_path: PathBuf::from("no-qwen-tokenizer-in-tests/tokenizer.json"),
        pin_input_shape: "auto".to_string(),
        mxr_cache_base: String::new(),
        engine_cache_rungs: 2,
        tokenize_max_texts: 4096,
        max_body_bytes: 2 * 1024 * 1024,
        wedge: test_wedge_policy(),
    }
}

/// A u8 stands in for an ort session throughout the cache tests: the cache never looks inside an
/// engine, and a real one cannot be built without a GPU.
pub(crate) fn cache_of(capacity: usize, rungs: &[(usize, u8)]) -> RungCache<u8> {
    let mut cache = RungCache::new(capacity);
    for &(cap, engine) in rungs {
        cache.insert(cap, engine);
    }
    cache
}

/// A scratch tree for the seeding tests, unique per test so parallel runs never collide.
pub(crate) fn seed_scratch(name: &str) -> (std::path::PathBuf, std::path::PathBuf) {
    let base = std::env::temp_dir().join(format!("bge-seed-{name}-{}", std::process::id()));
    std::fs::remove_dir_all(&base).ok();
    let from = base.join("from");
    let to = base.join("to");
    std::fs::create_dir_all(&from).unwrap();
    (from, to)
}

// ---------- the tokenizer registry ----------
/// A VALID tokenizer.json, minimal: WordLevel over three words, whitespace pre-tokenizer. The real
/// BGE-M3 file is 17 MB and these tests are about WHEN a file is read and WHICH name it answers for,
/// never about what the counts are — so the smallest thing the `tokenizers` crate will actually parse
/// is the right fixture. Verified to load and encode before anything was built on it.
pub(crate) const FIXTURE_TOKENIZER: &[u8] = br#"{
  "version": "1.0",
  "truncation": null,
  "padding": null,
  "added_tokens": [],
  "normalizer": null,
  "pre_tokenizer": {"type": "Whitespace"},
  "post_processor": null,
  "decoder": null,
  "model": {"type": "WordLevel", "vocab": {"[UNK]": 0, "alpha": 1, "beta": 2}, "unk_token": "[UNK]"}
}"#;

/// A model cache holding that fixture in BGE-M3's HuggingFace snapshot layout — the shape
/// `find_tokenizer_file` discovers.
pub(crate) fn model_cache_with_a_tokenizer(tag: &str) -> PathBuf {
    let root = std::env::temp_dir().join(format!("bge-reg-{tag}-{}", std::process::id()));
    let snapshot = root
        .join("models--BAAI--bge-m3")
        .join("snapshots")
        .join("deadbeef");
    std::fs::create_dir_all(&snapshot).expect("the fixture cache");
    std::fs::write(snapshot.join("tokenizer.json"), FIXTURE_TOKENIZER).expect("the fixture file");
    root
}

// ---------- compile growth is measured per engine ----------
pub(crate) fn write_mb(path: &Path, mb: usize) {
    std::fs::create_dir_all(path.parent().expect("a parent")).expect("the cache dir");
    std::fs::write(path, vec![0u8; mb * 1024 * 1024]).expect("the fake compiled program");
}

// ---------- GET /models: what this build can do ----------
pub(crate) fn model_row<'a>(models: &'a [ModelEntry], id: &str) -> &'a ModelEntry {
    models
        .iter()
        .find(|entry| entry.id == id)
        .unwrap_or_else(|| panic!("no '{id}' row in {models:?}"))
}
