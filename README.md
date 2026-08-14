# bge-sidecar

ONNX inference sidecar for the v2 code-RAG pipeline (see `research/PLAN_hybrid_search_onnx.md`):

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
| `POST /embed` | `{ "texts": ["..."], "kind": "doc"\|"query", "provider": "auto\|cuda\|dml\|migraphx\|cpu"? }` | `{ "dense": [[f32]], "sparse": [{ "indices": [u32], "values": [f32] }], "token_count": [usize], "truncated": [bool], "max_length": usize, "token_accounting": bool }` |
| `POST /rerank` | `{ "query": "...", "documents": ["..."], "provider": ...? }` | `{ "scores": [f32] }` (input order) |
| `GET /health` | — | `{ "status", "provider", "loaded": {...}, "models": {...}, "adapter": { "name", "vram_mb", "luid", "requested_device", "dml_device_id" } \| null }` |

`kind` is accepted for contract compatibility; BGE-M3 embeds queries and documents identically.

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
> old 4 was sized for the 1024-token era. The cost model lives in one place in the host —
`v2.Web.Contracts/Models/SidecarMemoryModels.cs` (`SidecarMemory`) — with tests in
`tests/v2.Tests/Rag/SidecarMemoryTests.cs`.

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
| `token_accounting` | `false` = NOT MEASURED (the tokenizer would not load). The two arrays are then empty, and a caller must not read "no truncation reported" as "nothing was truncated". |

The counter is BGE-M3's own `tokenizer.json`, read from the model cache on first use purely to count; it
never feeds inference. A missing file logs a warning at startup and turns accounting off rather than
failing the sidecar.

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
