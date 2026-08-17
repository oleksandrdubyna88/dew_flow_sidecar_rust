# PLAN — VRAM per engine, or an honest refusal to guess it

> Status: **plan only, nothing implemented yet, 2026-08-17.** Scope: `src/adapters.rs`,
> `src/provider.rs`, `src/wire.rs`, `src/introspection.rs`.
>
> Raised from `dew_flow_rag_qln`, whose runtime panel has labelled the split unavailable since it shipped
> and whose promoted plan (`dew_flow_rag_qln · research/PLAN_runtime_panel.md`) records that this item
> **was owned by no repository at all** — its status line said it had been raised here, and it had not.
> This plan closes that.

## The symptom

The panel can say *"dense, sparse and rerank are resident; the card holds 31.7 GB"* and cannot say how
much of that each one holds. `/health` reports `adapter.vram_mb` — DXGI's **static capacity**, sampled
once at startup (`src/adapters.rs:21`) — and `loaded: { dense, sparse, rerank }`, three booleans
(`src/wire.rs:213`). Nothing between them. `research/architecture.md` lists it under *What does not exist
yet*: "No live GPU telemetry."

The operational question it blocks is not academic. This machine has run three sidecars at once holding
VRAM between them, and the RAG repository has a measured incident of an index pass co-loading a coder and
an embedder past a 32 GB card. "Which engine should I evict" currently has no answer on any surface.

## The premise this plan was handed, and why it is FALSE

The item arrived worded as: *DXGI's `QueryVideoMemoryInfo` gives the process's current usage, and the
sidecar builds sessions one at a time, so a before/after delta per role would be a real measurement.*

The first half is right. **The second half is wrong on the default flavour**, and it is the whole
difficulty of this plan.

Session construction is serialized only where a compiled-model cache is configured — that is, only on
MIGraphX. `CachePathLease::hold` takes the process-global `CACHE_PATH_LOCK` when a cache base is set and
takes **nothing** otherwise (`src/compile_cache.rs:130-141`); its own doc comment says so in as many
words:

> *"Free everywhere else. With no cache configured — every non-MIGraphX flavour — there is nothing to
> claim: no lock is taken"* … *"`Engines.embed` / `Engines.rerank` are independent mutexes, so an embed
> build and a rerank build legitimately run at once on two `spawn_blocking` threads, which is the
> ordinary situation right after a restart when the host hits both endpoints."*

So on DirectML — the default feature, and what the R9700 this is measured on actually runs — a delta
taken around the embed build can contain the rerank build's allocation. That does not produce a slightly
noisy number; it produces a number that is confidently wrong about which engine is expensive, which is
the exact question the field exists to answer.

Writing the naive delta and calling it measured would reproduce, in this repository, the failure
`research/module_runtime_preflight.md` and the RAG panel's `FactSource` rule were both built to prevent:
a field that reports a request as though it were an observation.

## Design

### One funnel, two samples, and a claim about attribution

Every session in this process is built through one function — `load_session`
(`src/provider.rs:29`), called at `:118` (the dual embed model) and `:129` (the reranker). That is where
the measurement goes; there is no second path to keep in step.

```
sample()  ->  build(provider)  ->  sample()
   |                                  |
   +------------ delta ---------------+   attributable ONLY if nothing else allocated meanwhile
```

`sample()` is `IDXGIAdapter3::QueryVideoMemoryInfo` for the local memory segment, which reports **this
process's** current usage on that adapter. The `windows` crate is already a dependency with
`Win32_Graphics_Dxgi` (`Cargo.toml:41`), and `IDXGIAdapter3` lives in that namespace — this needs no new
feature and no new crate.

### Attribution is a decision, not a subtraction

A process-wide counter sampled twice measures the process, not the engine. So the delta is recorded
**only when it can be attributed**, and otherwise the field is absent:

- A process-global build counter is incremented for the duration of every `load_session`.
- The delta is kept only if that counter was 1 for the whole window — this build was alone.
- If any other build overlapped, the sample is discarded and the engine's figure stays `None`.

`None` is not a failure state here; it is the honest answer, and this service already has the vocabulary
for it (`adapter: Option<ResolvedAdapter>`, `src/wire.rs:209`).

**The rejected alternative: serialize every build.** Taking the lease unconditionally — not just on
MIGraphX — would make every delta attributable. It was rejected because it buys the measurement with a
cold-start regression on the flavour everybody runs: the lease is held across the build *and that
engine's first pass*, so the second engine would wait minutes behind the first, every restart, to
populate a diagnostic field. A number is not worth making the product slower to obtain. If the counter
turns out to discard most samples in practice, this decision is the one to revisit — and the counter's
own discard rate is what should decide it, which is why step 4 below reports it.

### What is measured is the LOAD, not the residency

The delta says what building this engine cost, once. It is not a live reading and must not be presented
as one: a later pass may allocate more, and nothing here re-samples. So the field is named for what it
is — the allocation observed at load — and unload is used to check it rather than to track it.

### Not Windows, not measured

DXGI exists on Windows. The MIGraphX (WSL/ROCm) and CUDA-on-Linux flavours get `None` from this path, and
that is a stated limit rather than a gap to fill later with a second mechanism. `src/adapters.rs` already
has exactly this shape: a `#[cfg(windows)]` DXGI module and an `None` for everything else.

## Build order

1. **`adapters::process_vram_bytes() -> Option<u64>`** beside the existing DXGI code in
   `src/adapters.rs`, `#[cfg(windows)]`, resolving the same adapter the EP targets (the module already
   resolves `ORT_DEVICE_ID` to an adapter — reuse that resolution, do not re-enumerate).
2. **The build counter and the two samples in `load_session`** (`src/provider.rs:29`) — one place, so
   both engines are covered by construction and a third engine added later is covered for free.
3. **`LoadedModels` gains the figures** (`src/wire.rs:213`). Three booleans become three entries each
   carrying `loaded: bool` and `load_bytes: Option<u64>`; `/health` consumers that only read the boolean
   keep working, which matters because the RAG panel ships separately from this binary.
4. **The discard count on `/health`** — how many samples were thrown away for overlap. Without it,
   "unavailable" and "never attributable on this machine" look identical, and the second is what would
   send this plan back to the rejected alternative.
5. **The unload cross-check** — `drain_engines` (`src/introspection.rs:241`) samples before and after too,
   and logs the freed delta beside the recorded load delta. Not a wire field: a log line, because it is
   evidence about the measurement rather than about the engine.

## Test plan

- The delta arithmetic is a pure function over two samples and a counter — tested without DXGI, the way
  `src/adapters.rs` already tests its index mapping OS-independently.
- Two overlapping builds: both engines report `None`, and the discard count is 2. This is the test that
  matters, because a naive implementation passes every other one.
- A single build with the counter at 1: the delta is kept.
- A sample that fails (no DXGI, non-Windows, adapter unresolved) yields `None` and never a `0` — the
  distinction `FactSource` exists for, and the one `research/module_runtime_preflight.md` records the
  cost of losing.
- `/health` still deserializes in a consumer that only knows the three booleans.
- On a real R9700: load dense+sparse, read the figure, unload, and confirm the freed delta is within a
  stated tolerance of the recorded one. This is the only step that proves the number means anything, and
  it cannot run in CI.

## Definition of Done

- [ ] `/health` reports per-engine load bytes, or `None` with a reason it can be told apart from zero.
- [ ] A figure is published only when no other build overlapped it; the discard count is on the wire.
- [ ] The overlap case has a test that fails against the naive delta.
- [ ] Non-Windows and DXGI-failure paths report absent, never zero.
- [ ] The unload cross-check is logged, and its agreement with the load figure is recorded here.
- [ ] `research/architecture.md`'s *What does not exist yet* entry is updated — it currently says "No live
      GPU telemetry", and this is not live telemetry either. Say precisely what now exists.
- [ ] `dew_flow_rag_qln · research/module_runtime.md` is told: its "Nobody owns this" note is why this plan
      exists, and its panel can stop labelling the split permanently unavailable.
