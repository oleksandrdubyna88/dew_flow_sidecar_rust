# bge-sidecar

ONNX inference sidecar of the `dew_flow_*` family — the embedding/rerank engine
`dew_flow_rag_qln` talks to over HTTP (design records live in this repo's `research/`; the
retrieval product's side is `dew_flow_rag_qln · research/architecture.md`):

- **BGE-M3 dense + learned-sparse embeddings** — FP32, the official `BAAI/bge-m3` ONNX export,
  two engines over the same model (`TextEmbedding` + `SparseTextEmbedding` from `fastembed`),
  `max_length` pinned to **8192** (fastembed's default 512 silently truncates otherwise).
- **BGE-Reranker-v2-M3** cross-encoder rerank scores (sigmoid-normalized 0..1, aligned with the
  input document order — same contract as the retired `tools/reranker` Python sidecar).

## Build (once per machine, like `tools/ts-parser` needs `npm install`)

Requires Rust (rustup, MSVC toolchain) + VS Build Tools with C++.

```bash
# AMD GPU (DirectML — the default feature)
cargo build --release

# NVIDIA GPU (CUDA)
cargo build --release --no-default-features --features cuda

# Both EPs in one binary
cargo build --release --features cuda

# CPU only
cargo build --release --no-default-features

# AMD on Linux/WSL (ROCm via the MIGraphX EP) — run INSIDE WSL, see the dedicated section below
cargo build --release --no-default-features --features migraphx --target-dir target-wsl
```

The Windows binary lands at `target/release/bge-sidecar.exe` (the WSL flavor at
`target-wsl/release/bge-sidecar` — separate target dir so the two OS flavors never clobber each
other). The Aspire AppHost registers it via `AddExecutable` when `Aspire:BgeSidecar:Enabled` is set
**and** the binary exists (graceful-off otherwise), and injects its URL into the API/worker as
`Agent__Embedding__SidecarUrl` / `Agent__Reranker__SidecarUrl`.

Model files (~2.2 GB for bge-m3 FP32 + the reranker) download from Hugging Face on first use into
`.model-cache/` (override with `MODEL_CACHE_DIR`). First `/embed` and `/rerank` calls are slow
(lazy load); subsequent calls are warm.

## Endpoints

| Endpoint | Body | Response |
|---|---|---|
| `POST /embed` | `{ "texts": ["..."], "kind": "doc"\|"query", "provider": "auto\|cuda\|dml\|migraphx\|cpu"?, "request_id": "..."? }` | `{ "dense": [[f32]], "sparse": [{ "indices": [u32], "values": [f32] }], "dimension": usize\|null, "token_count": [usize], "truncated": [bool], "max_length": usize, "token_accounting": bool, "request_id": "...", "timings": {...} }` |
| `POST /rerank` | `{ "query": "...", "documents": ["..."], "provider": ...?, "request_id": "..."? }` | `{ "scores": [f32] (input order), "request_id": "...", "timings": {...} }` |
| `POST /tokenize` | `{ "texts": ["..."], "model": "bge"\|"qwen"? }` | `{ "token_count": [usize], "model": "...", "available": bool }` |
| `GET /models` | — | `{ "models": [{ "id", "name", "kind": "dense+sparse"\|"rerank"\|"tokenizer-only", "dimension": usize\|null, "max_sequence_length": usize\|null, "tokenizer": "..."\|null, "available": bool, "tokenizer_available": bool\|null }] }` |
| `GET /health` | — | `{ "status": "ok"\|"wedged", "wedged", "in_flight": [...], "provider", "provenance_ready", "loaded": {...}, "self_check": { "cosine", "serving", "verified", "serving_threshold", "verified_threshold", "attempts", "checked_seconds_ago" } \| null, "vram_at_load": {...}, "models": {...}, "adapter": { "name", "vram_mb", "luid", "requested_device", "dml_device_id" } \| null }` |

`/embed`'s `dimension` is the width of the dense rows in that same response, read from one of them — so a
caller sizing a vector collection cannot size it wrong. `GET /models` states what this build can do
before a pass starts: `dimension` there is **measured**, so it is `null` until something has been
embedded (unknown is a value, never `0`), and `available` reports the engine while `tokenizer_available`
reports the tokenizer file — "engine cold, tokenizer ready" is a real and useful state.

`kind` is accepted for contract compatibility; BGE-M3 embeds queries and documents identically.

### Per-request timings (`timings`, 2026-08-15)

Both inference responses attribute the request's wall-clock, because every number here used to die in
this process's own log file — and the most important one was never measured at all:

| Field | Meaning |
|---|---|
| `queue_wait_ms` | Waiting for the engine mutex behind **another** request. This is the caller's *infrastructure wait*, never model speed — a request that waited 8 s and ran 0.4 s used to look like a slow model. Measured around the mutex only. |
| `session_build_ms` | Building + canary-checking the ORT session. `0` on a warm engine — a cold first call is otherwise indistinguishable, on the wire, from slow inference. |
| `inference_ms` | The forward pass(es), settling re-runs included — what this request's inference actually cost. |
| `compile_cache_grew_mb` | `> 0` = MIGraphX compiled this input shape during the pass (the compiled-model cache saves lazily, so growth across the pass is the only moment a compile is observable). |

`request_id` is an optional, opaque caller correlation id: echoed verbatim in the response and
prefixed to the request's pass log lines, so two concurrent requests can be told apart in both places.
Absent ⇒ empty echo, unprefixed logs.

## Configuration (env, injected by the AppHost)

| Env | Default | Meaning |
|---|---|---|
| `PORT` | 5320 | HTTP port (Aspire injects the allocated one) |
| `ORT_PROVIDER` | *(empty)* | `auto` \| `cuda` \| `dml` \| `migraphx` \| `cpu`. Empty ⇒ the first request's `provider` hint (the C# side forwards the operator's Settings → Models choice), else `auto` (cuda → migraphx → dml → cpu). Explicit choices fail hard instead of silently falling back to CPU. |
| `ORT_DEVICE_ID` | 0 | GPU device index in DXGI **high-performance order** (0 = the fastest card, matching the host UI's picker numbering). For DirectML it is translated internally (`src/adapters.rs`: HP index → adapter LUID → plain `EnumAdapters` index) because the legacy DML EP counts adapters in plain enumeration order, which usually lists the display/integrated adapter first; feeding the raw id straight in ran inference on the wrong card. CUDA gets the raw id (its own numbering). `/health` reports the resolved adapter (`adapter`), or `null` when DXGI could not resolve it (the raw id is then passed through). |
| `MAX_BATCH` | 4 | Default embedding/rerank batch size. Linear in the attention peak below. **Overridable per request** (`max_batch`). |
| `ORT_THREADS` | 0 | Intra-op threads (0 = ONNX Runtime decides; relevant for CPU) |
| `EMBED_MAX_LENGTH` | 1024 | Default token cap for both embedding engines — **the VRAM driver, squared in the peak below**. **Overridable per request** (`max_length`); a change evicts the loaded engines so they rebuild at the new cap. |
| `RERANK_MAX_LENGTH` | 1024 | Token cap for reranker pairs |
| `MODEL_CACHE_DIR` | `.model-cache` | Where model files download to |
| `QWEN_TOKENIZER` | `../qwen-tokenizer/tokenizer.json` | The file behind the `qwen` row of the tokenizer registry. **Counting only** — no Qwen model is ever loaded here. |
| `TOKENIZE_MAX_TEXTS` | 4096 | Texts one `/tokenize` call may carry; past it the call is refused with a `400` naming the cap. A runtime limit, not a memory one — the request body cap does **not** bound the row count, and a 10,000-file pass once sent 52,617 texts in one request. Eight times the host's own per-request row budget (`SidecarClient.RequestRowBudget` = 512), so it is a backstop rather than a wall. Reported by `/health`. |
| `SIDECAR_IDLE_UNLOAD_SECONDS` | 0 (**off**) | Drop every engine after this long with no request, giving the card back without an operator having to call `/unload`. Off by default because the cost it causes is as real as the one it prevents: a pass whose gap between two batches exceeds this pays a rebuild (60 s and up; minutes on a first-ever MIGraphX shape). Set it longer than your own gaps. Two guards past the clock — nothing may be in flight, and the drain goes through the same ceilinged path `/unload` uses. The incident behind it: three sidecars were once found running here, two holding models nobody was using. |
| `MAX_BODY_BYTES` | 2 MiB | The largest request body any route accepts. **Same number axum defaults to** — the point is that it is now a decision rather than a framework constant: readable on `/health` as `max_body_bytes`, movable by an operator raising `EMBED_MAX_LENGTH` toward 8192, and **logged when it fires**. It rejects before any handler runs, and the caller sees only *"an established connection was aborted by the software in your host machine"* — the server rejects the body while the client is still writing it. A 10,000-file repository died nine minutes into a pass this way, and the log said nothing at all. |
| `WEDGE_RUNNING_AFTER_SECONDS` | 900 | A forward pass held longer than this is **wedged** — see below |
| `WEDGE_BUILDING_AFTER_SECONDS` | 3600 | The same for a session build + canary, which legitimately contains a multi-minute compile |
| `UNLOAD_LOCK_WAIT_SECONDS` | 30 | How long `/unload` waits for an engine before answering "still loaded" |
| `WEDGE_POLL_MS` | 50 | How often a waiter re-checks the lock and the holder's stamp |
| `WEDGE_EXIT` | *(off)* | `1` = exit the process on a wedge so the host restarts it. **Opt-in**, and off by default |
| `WEDGE_EXIT_AFTER_SECONDS` | = `WEDGE_RUNNING_AFTER_SECONDS` | How long **after the wedge verdict** the exit fires (never from the phase start — exiting mid-compile corrupts the `.mxr` cache) |

The provider chosen at the **first model load** is pinned until restart (`/health` reports it) —
execution provider is machine-level hardware config; changing the setting requires a sidecar restart.

### VRAM budget (why the defaults are small)

bge-m3 and bge-reranker-v2-m3 are XLM-RoBERTa-large: **24 layers, 16 attention heads**. One layer's
attention-score tensor is `[batch, heads, seq, seq]` in FP32 and softmax holds a second buffer of the same
shape, so the transient peak is

```
batch × 16 heads × seq² × 4 B × 2
```

| Envelope | Attention peak | Fits a 32-GB card? |
|---|---|---|
| `MAX_BATCH=8`, `EMBED_MAX_LENGTH=8192` (pre-2026-07-19 default) | **64 GiB** | no |
| `MAX_BATCH=4`, `EMBED_MAX_LENGTH=1024` (current default) | **512 MiB** | yes (~6.9 GB with all three models warm) |

`seq` is **squared**, so the token cap is the knob that decides whether a pass fits; 1024 tokens (~4k chars)
covers essentially every method body indexed. DirectML does **not** fail an over-sized allocation — it
over-commits into shared (system) memory, so the old defaults surfaced as a full VRAM bar, tens of GB of
process RAM and an index pass crawling over PCIe, with no out-of-memory error anywhere.

Both knobs ride on every `/embed` request (the host's Settings → RAG → *Sidecar memory budget* page owns
them and shows the arithmetic live), so the env values above matter only for the window before the first
request and for a bare `cargo run`.

> **Read `loaded_*`, not the configured values (2026-08-13).** `/health`'s `embed_max_length` and
> `max_batch` are the CONFIGURED fallbacks — what a request that carries no envelope of its own would get.
> They are not what the last embed ran at, and reading them as such is how three numbers ended up
> describing one pass: a host chunk of **126** texts per HTTP call, an operator batch of **64** sequences
> per forward pass, and a `/health` that said **4**. Only the last was wrong-by-omission — the requests
> carried 64 and `Limits::resolve` honoured them; nothing reported it. `loaded_embed_max_length` and
> `loaded_max_batch` now report what actually ran (`null` until one has), the same requested-vs-active
> split the provider fields already carry. The configured batch default was also raised 4 → 64 to match
> `SidecarMemory.DefaultMaxBatch`: at the shipped 256 cap a batch of 64 costs 512 MiB of attention, and the
> old 4 was sized for the 1024-token era. The cost model lived in one place in the host — at the time
of this note that host was the now-frozen `ClaudeRag` predecessor
(`v2.Web.Contracts/Models/SidecarMemoryModels.cs`, `SidecarMemory`, with its tests); today's host is
`dew_flow_rag_qln`, and the citation is kept as the record of where the decision was made.

## The wedge detector: the one wait this process cannot cancel (2026-08-16)

An ONNX Runtime forward pass **cannot be cancelled**. It is a C++ call on a thread this process does not
own, and a thread merely *stuck* inside it never panics — so the poison-healing lock, which recovers a
mutex a *panic* poisoned, can never reach it. Until this landed, nothing else could either: every later
`/embed` queued on `.lock()` forever, `/health` reported the freeze exactly as it reports a healthy
multi-minute build, and the daemon's deliberately infinite sidecar HTTP timeout composed the two into a
system-wide freeze with no symptom anyone could observe.

The remedy is visibility plus a ceiling, not cancellation:

- **The holder stamps itself** — what it is doing and since when — in a mutex of its own, never inside
  the engine slot. A holder wedged *under* the engine mutex could never be observed *through* it.
- **`/health` says so.** `status` flips to `"wedged"`, `wedged: true`, and `in_flight[]` carries
  `{ engine, phase, activity, elapsed_seconds, ceiling_seconds, wedged }` per held engine.
- **New requests fail fast** with `503` and a message naming the activity, the elapsed time and how to
  recover — instead of queueing behind a mutex that will never be released.
- **The log says it unasked**, once per wedge: the party that would otherwise have polled `/health` is
  precisely the one already blocked.
- **`/unload` still answers.** It waits on the blocking pool with a ceiling, never on a reactor thread,
  and an engine it could not take stays loaded and is reported as such.

### Why the ceilings are minutes, not seconds

A **false** "wedged" is expensive in both directions, so the numbers are deliberately generous:

| Phase | Ceiling | Why |
|---|---|---|
| `running` | 900 s | Warm passes are seconds (1.6 s at a 256 cap, 6.8 s at 1024). The slowest *honest* one on record is ~608 s — a first request that also paid a lazy MIGraphX compile plus its settling retries — and a first rerank pass compiles 92–162 s with no canary ahead of it. 900 s is ~1.5× the worst honest case |
| `building` | 3600 s | This phase legitimately contains the cold compile (**214 s measured**), up to three canary runs, and — on a corrupt cache — a wipe plus one clean recompile with a canary of its own |

**A first-ever MIGraphX shape compile is correct slowness and must never be flagged.** That is also why
the process-exit last resort is opt-in and off: killing a process mid-compile is exactly how a corrupt
`.mxr` reaches the compiled-model cache, which is the 2026-07-31 incident the build canary exists for.
When it *is* enabled, its ceiling is measured from the wedge verdict, never from the phase start.

## Token accounting: why `/embed` reports what it tokenized (2026-08-13)

An input longer than `max_length` is **truncated to a prefix by the tokenizer and embedded as though that
prefix were the whole text** — no error, no warning, no counter. The caller receives a perfectly
well-formed vector for a document it never sent, and the tail of that document simply stops being
searchable. The host cannot detect this on its own: it owns no tokenizer.

So every `/embed` reply now carries, parallel to `texts`:

| Field | Meaning |
|---|---|
| `token_count[]` | What each text really cost, special tokens included, measured **before** the cap — so a value above `max_length` is exactly the overflow that was discarded. |
| `truncated[]` | The model saw a prefix of this text. |
| `max_length` | The **effective** cap they were judged against (after `cap_for`, which forbids a `query` from moving the loaded rung). |
| `token_accounting` | `false` = NOT MEASURED — the tokenizer would not load, **or it refused one of the texts**. The two arrays are then empty, and a caller must not read "no truncation reported" as "nothing was truncated". |

The counter is BGE-M3's own `tokenizer.json`, resolved through the **tokenizer registry** at startup
(see below) purely to count; it never feeds inference. A missing file logs a warning at startup and turns
accounting off rather than failing the sidecar. A text the tokenizer *refuses* does the same for that response: the count used to be
folded to `0`, which reads as "measured, and definitely not truncated" — the exact inversion of the only
guarantee this accounting exists to give, on the one signal the host cannot compute for itself.

**What it is for.** The host (`SidecarEmbedder`) turns any reported truncation into
`EmbedInputTruncatedException`, which fails the index run with an operator-readable diagnosis instead of
storing the mutilated vector. It is deliberately the one embed failure that is not degraded-and-retried:
the same text at the same cap truncates identically forever.

**Measured, on an AMD Radeon AI PRO R9700**, by two independent methods that agree — embedding
`[prefix, prefix + marker]` and finding where the dense vectors become bit-identical, and the tokenizer's
own count:

| `max_length` | C# stages | C# razor.cs | Rust | Markdown |
|---|---|---|---|---|
| 256 | 3.28 | 3.36 | 3.11 | 3.15 |
| 512 | 3.36 | 3.32 | 3.16 | 3.12 |
| 1024 | 3.50 | 3.15 | 3.09 | **2.99** |

Real text runs at **2.99–3.50 chars/token**. The host had been budgeting 4 — a ~34 % overshoot: 1024
characters of C# is 300 tokens against a cap of 256, so 44 tokens of every filled chunk were thrown away.
`EmbedBudget` (host side) now spends the measured floor less a safety margin.

## The tokenizer registry: counting for models this process never loads (2026-08-16)

This sidecar can **count** for a wider set of models than it can **embed**, and that is deliberate. A
chunker has to size a chunk with the serving model's *own* tokenizer — a chunk fitted to a window the
model does not have is a chunk that gets truncated — and nothing on the .NET side can read a HuggingFace
`tokenizer.json` at all (`Microsoft.ML.Tokenizers` has no regex pre-tokenizer and no NFC normalizer).
Counting therefore happens where the reference implementation already lives.

The names `/tokenize` answers for are a **table, not a match arm**:

| Row | File | Kind |
|---|---|---|
| `bge` | discovered under `MODEL_CACHE_DIR` (the snapshot folder is a content hash, so it is found, not hardcoded) | also what `/embed`'s truncation accounting counts with |
| `qwen` | `QWEN_TOKENIZER` | counting only — no Qwen model is ever loaded |

Three properties are the contract:

- **Registered ≠ available.** An unknown name is a `400` that **names every registered row**, so a caller
  corrects itself from the answer instead of reading this file. A *registered* name whose file is missing
  is a `200` with `available: false` — the caller's name was right and the deployment is what is missing,
  and collapsing the two would send a chunker hunting for a typo while a file was simply absent.
- **Loaded at startup, never on the first request.** The counters used to be `OnceLock`s filled inside the
  first `/tokenize` call, so that request paid a directory walk and a multi-MB parse *on the async
  runtime*, and a model cache swapped under a running sidecar silently changed the answer. The startup log
  names every row and the file behind it — a count whose tokenizer nobody can name afterwards is a count
  nobody can reproduce.
- **The encode runs on the blocking pool, under a cap.** "Pure CPU" is a statement about the card, not
  about the reactor. `TOKENIZE_MAX_TEXTS` bounds the rows because the body limit does not: enough short
  texts fit under 2 MB to be tens of thousands of encodes in front of `/health` and `/unload`, which are
  the two endpoints an operator reaches for when something is stalled.

Adding a third model is a row and a path. **What** each row is — kind, embedding dimension, max sequence
length, and whether it can answer right now — is [`GET /models`](#endpoints), so a consumer validates a
recipe before a pass rather than discovering a mismatch inside one.

## AMD on Linux/WSL (ROCm via the MIGraphX EP)

> Background (2026-07): the ONNX Runtime ROCm EP was removed upstream (ORT 1.23; ROCm 7.0 was the
> last release carrying it) — **MIGraphX is the only AMD GPU EP** in current ORT. DirectML stays the
> AMD path on native Windows; this section is the direct-ROCm path through WSL2.

Nobody ships ONNX Runtime with the MIGraphX EP as a prebuilt **C library** (AMD publishes Python
wheels only; pyke/Ubuntu builds have no MIGraphX), so this flavor is `ort` in **load-dynamic** mode:
the sidecar dlopens a machine-local `libonnxruntime.so` named by `ORT_DYLIB_PATH` at startup.

> **The dylib version is coupled to the `ort` crate:** `ort =2.0.0-rc.12` builds with `api-24`
> enabled and refuses anything older than **ONNX Runtime 1.24** (`ort::MINOR_VERSION`). Worse,
> the crate's own version check **deadlocks instead of erroring** on an older dylib (rc.12 bug:
> the error path re-enters the API OnceLock it is initializing), which froze the whole sidecar —
> and the C# indexing stage — on the first `/embed` when a 1.23.2 build was installed
> (diagnosed 2026-07-27). The sidecar now preflights the dylib at startup and exits with the
> required version instead of hanging. When bumping `fastembed`/`ort`, re-check `ort::MINOR_VERSION`
> and rebuild the dylib to match.

One-time setup inside WSL (Ubuntu; assumes ROCm + `amdrocm-migraphx`/`-dev` installed and
`rocminfo` shows the GPU):

```bash
# 1. build ONNX Runtime with the MIGraphX EP — tag must satisfy ort::MINOR_VERSION (>= v1.24 for rc.12)
git clone --recursive --depth 1 -b v1.24.4 https://github.com/microsoft/onnxruntime.git ~/onnxruntime
cd ~/onnxruntime
./build.sh --config Release --build_shared_lib --parallel --skip_tests \
  --use_migraphx --migraphx_home /opt/rocm --rocm_home /opt/rocm \
  --cmake_extra_defines CMAKE_HIP_ARCHITECTURES=gfx1201 \
  --compile_no_warning_as_error         # GCC 15+: also patch missing <cstdint> includes as reported

# 2. install the libs to the stable path the AppHost defaults to. On an UPGRADE, move the old
#    install aside first so stale .so.<oldver> files and symlinks cannot linger next to the new ones.
sudo mv /opt/onnxruntime-migraphx /opt/onnxruntime-migraphx.bak 2>/dev/null || true
sudo mkdir -p /opt/onnxruntime-migraphx/lib
sudo cp -a build/Linux/Release/libonnxruntime*.so* \
           build/Linux/Release/libonnxruntime_providers_*.so /opt/onnxruntime-migraphx/lib/

# 3. build the sidecar's Linux flavor (rustup toolchain inside WSL).
#    pkg-config + libssl-dev: on Linux fastembed's HF downloader links system OpenSSL
#    (Windows uses Schannel, so the Windows build never needed them).
sudo apt install -y pkg-config libssl-dev
cd /mnt/<drive>/<repo>/tools/bge-sidecar
cargo build --release --no-default-features --features migraphx --target-dir target-wsl
```

AppHost config (`Aspire:BgeSidecar:*`): set `WslDistro` (e.g. `Ubuntu`) to flip the sidecar into WSL
mode — it then launches `wsl.exe -d <distro> --cd <repo>/tools/bge-sidecar -- bash -lc "… exec
./target-wsl/release/bge-sidecar"` instead of the Windows exe. `WSLENV` carries `PORT`/`ORT_*` across
the Windows→Linux boundary (Aspire sets them on the wsl.exe process; WSL drops them otherwise).
Related keys: `WslOrtDylib` (default `/opt/onnxruntime-migraphx/lib/libonnxruntime.so`),
`WslHipVisibleDevices` (default `0` — pins the discrete card in HIP order so an iGPU is never
picked), `Provider` (defaults to `migraphx` in WSL mode).

### The compiled-model cache is MANDATORY (`ORT_MIGRAPHX_MODEL_CACHE_PATH`)

ROCm 7.x's MIGraphX EP **always** saves the compiled model. With no cache path it writes to `""`,
the write fails, and the failure propagates into the kernel call — so **every** `/embed` answers
`500` after a ~2-minute compile, with the GPU idle in between:

```
migraphx_save: Error: file_buffer.cpp:77: write_buffer: Failure opening file: ""/21000-…-0.mxr
Non-zero status code returned while running MGXKernel_… Status Message: Failed to call function
```

That is the whole symptom of "the index stage runs for ages and nothing happens" (diagnosed
2026-07-27; the EP ignores `ORT_MIGRAPHX_SAVE_COMPILED_MODEL` — the path is the only knob). With the
path set, measured on the R9700: first call **214 s** (compile + save), every later call at the same
input shape **~0 s**. `main.rs` therefore preflights the directory at startup and exits with the fix
instead of failing per request; the AppHost sets and creates it (`Aspire:BgeSidecar:WslMigraphxCacheDir`,
default `$HOME/.cache/bge-sidecar-migraphx/device-N`).

Keep it on the **Linux** filesystem, never under `/mnt` — each entry is about the model's own size
(**~2.3 GB**) and DrvFs would crawl. Budget disk accordingly and prune the directory when you change
the memory envelope (`EMBED_MAX_LENGTH` / `MAX_BATCH` produce new shapes, hence new entries).

### Shape pinning (`PIN_INPUT_SHAPE`, on by default under MIGraphX)

**MIGraphX compiles per input SHAPE**, and fastembed pads with `PaddingStrategy::BatchLongest`
(`fastembed/src/common.rs`), so a batch's padded length follows its longest text. Real indexing
therefore hits a new shape almost every batch — measured on a live Fast pass: ~2-4 minutes of
compile per batch with the GPU otherwise idle, and a **~2.5 GB cache entry each time** (two batches
had already written 5 GB). Over thousands of methods that is days of compiling and hundreds of GB.

So `embed` normalizes the layout before fastembed sees it (`pin_shape`): a "ruler" sequence that
truncates to `max_length` leads every chunk, and the tail is filled so the last chunk is full —
giving exactly ONE shape `(max_batch, max_length)`, hence **one** compile and **one** cache entry per
run. The ruler rows are dropped from the response (`unpin_rows`, which fails loudly rather than
return a misaligned row).

The trade is one wasted row per `max_batch - 1` real rows, and every row computed at the full token
cap instead of its natural length — worth it only where a recompile costs minutes. `PIN_INPUT_SHAPE`
is therefore `auto` by default: **on for `migraphx`, off for `cuda`/`dml`/`cpu`**, which take dynamic
shapes in stride. Force it with `PIN_INPUT_SHAPE=1` / `0`. It needs `MAX_BATCH >= 2` (no room for a
ruler otherwise) and is skipped below that.

Changing `EMBED_MAX_LENGTH` / `MAX_BATCH` changes the pinned shape — expect one fresh compile and a
new cache entry, and prune the old ones.
- The model cache (`.model-cache/`) is shared with the Windows flavor via `/mnt/<drive>` — no
  re-download; reads over DrvFs are just slower on first load.
- `/health`'s `adapter` is `null` on Linux (DXGI is Windows-only); the raw `ORT_DEVICE_ID` semantics
  apply (HIP order, after the `HIP_VISIBLE_DEVICES` mask).

## Regenerating the canary reference

Every freshly built engine is checked against `src/canary-reference.f32le` before it is allowed to serve —
a crash mid-compile can leave a cached program that loads fine and stably produces garbage, and that is the
one failure no length or shape check catches. The oracle is regenerated by the binary itself:

```bash
# writes over the committed file; add a path to write somewhere else and compare first
bge-sidecar --write-canary-reference [path]
```

It builds the embed session through the ordinary loader on the configured provider, embeds `CANARY_TEXT` at
the production shape, and **prints the cosine against the file it is about to replace**. Read that number:

- **~1.0** — nothing needed regenerating. The model has not moved.
- **far from 1.0** — either the model deliberately changed (a new export, a new checkpoint) or this build is
  producing garbage. Only you know which, and regenerating to silence a failing canary is the one misuse
  this tool must not be put to.

**A byte-level diff is normal.** Measured on an R9700 through DirectML: cosine 1.000000000 and yet 1012 of
1024 elements differ, max delta 2.868e-07. GPU arithmetic is not bit-reproducible across runs, which is why
the check is a cosine and not a comparison of bytes — and why `git diff` calling this file changed proves
nothing on its own.

## Licence

This repository is **public and proprietary**, which is a pair that surprises people, so it is stated
rather than implied: the source is visible so a customer can audit what runs on their own machine, and
visibility grants no right to use it. The terms are in [LICENSE](LICENSE).

Three third-party facts matter more here than the boilerplate, and each was resolved from the artefacts
rather than from memory — see [THIRD-PARTY-NOTICES.md](THIRD-PARTY-NOTICES.md):

- **`vendor-fastembed/` is a full vendored copy** of the upstream crate (Apache-2.0) carrying one marked
  patch, kept byte-identical to upstream otherwise so the patch stays reviewable.
- **One shipped dependency is MPL-2.0** — `option-ext`, four levels down through `hf-hub`. File-level
  copyleft, unmodified by us; its source stays available and the notice names it.
- **DirectML, CUDA and cuDNN are used and never redistributed.** They are vendor-licensed, and one cached
  copy in a registry, an image layer or an offline bundle would make us a redistributor. That is the reason
  this binary is built on the machine that runs it rather than shipped to it.

[NOTICE](NOTICE) carries the attribution that must travel with any build.
