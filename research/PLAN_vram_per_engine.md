# PLAN — VRAM per engine, or an honest refusal to guess it

> Status: **IMPLEMENTED, 2026-08-19.** Scope as built: `src/vram.rs` (new), `src/adapters.rs`,
> `src/provider.rs`, `src/wire.rs`, `src/state.rs`, `src/introspection.rs`.
>
> Raised from `dew_flow_rag_qln`, whose runtime panel has labelled the split unavailable since it shipped
> and whose promoted plan (`dew_flow_rag_qln · research/PLAN_runtime_panel.md`) records that this item
> **was owned by no repository at all** — its status line said it had been raised here, and it had not.
> This plan closed that.
>
> **What it delivered, measured on the R9700 the day it shipped:** the dual bge-m3 embed session's build
> allocates **2 175 MB**, and tearing it down returned **2 183 MB** — a 0.37 % disagreement, which is the
> pass's own transient buffers going with the session. `/health` carries it as `vram_at_load`, alongside
> the count of samples thrown away for overlap. See [What shipped differently](#what-shipped-differently)
> before reading the design below: three of its decisions changed under measurement, and one of its
> premises about the wire was simply wrong.

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
2. **The build counter and the two samples in `load_session`** (`src/provider.rs:32`; the samples now sit at `:79`/`:81`, inside the per-candidate attempt, since `auto` resolves by trying) — one place, so
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

## What shipped differently

Four deviations, and the first is the one worth reading.

### 1. The wire change this plan asked for would have broken the only consumer

Build order step 3 said: *"Three booleans become three entries each carrying `loaded: bool` and
`load_bytes: Option<u64>`; `/health` consumers that only read the boolean keep working, which matters
because the RAG panel ships separately from this binary."*

The reasoning is right and the conclusion is false. Checked against the consumer before writing anything:
`dew_flow_rag_qln · src/Rag.Infrastructure/Runtime/RuntimeInspector.cs` reads
`loaded.TryGetProperty(role, out var isLoaded) && isLoaded.ValueKind == JsonValueKind.True` in
`ResidentModels`, and `LoadedRoles` does the same. A JSON object is `ValueKind.Object`, not `.True` — so
turning the booleans into entries would have emptied the runtime panel of every model, on the very
surface this figure exists to fill, in a repository that ships separately and would not have been
updated in the same change.

**Shipped instead:** `loaded` is untouched, and `vram_at_load` is a SIBLING object on `/health`. A test
(`wire.rs`, `the_loaded_flags_stay_booleans_so_a_consumer_that_only_knows_them_keeps_working`) asserts the
three fields serialize as booleans, so the next person to have this idea meets a red test rather than a
blank panel.

### 2. Absent needed four states, not two

The DoD asked for "`None` with a reason it can be told apart from zero", and two states could not carry
it. `Attribution` is `Measured | Overlapped | NotSampled | NoGrowth`, each counted separately on the wire.
`NoGrowth` is the one that had to be invented: a build sampled alone whose adapter usage did not grow is
not a measurement of zero — it is evidence the allocation was invisible to this sampler — and publishing
it as `0` would have read as "this engine is free", which is the single most misleading thing this field
could say.

### 3. A build counter alone cannot detect overlap

The design said: keep the delta only if the build counter was 1 for the whole window. A counter tells a
build whether it STARTED alone; it cannot tell a build that started alone that somebody joined it
half-way. So there are two statics: `BUILDS_IN_FLIGHT`, and `OVERLAP_MARKS` which any build bumps when it
finds another already in flight. A window compares the mark it opened with against the mark at close. The
read order matters and is argued at `SoloBuildWindow::open`: the mark is read BEFORE counting in, so a
joiner's bump can never land in the gap unobserved.

### 4. What the delta covers, precisely

It covers **session construction**, because that is what `load_session` brackets. On MIGraphX most of the
allocation happens later, at the first kernel launch — the same lazy behaviour `CachePathLease` exists
for — and that memory is not in this number. On DirectML, where it was measured, construction is where
the weights land, which is why 2 175 MB is a credible figure for a 2.27 GB FP32 model. Not a defect;
a stated boundary, repeated at the sampler and in `module_http_surface.md`.

Also minor: `adapters::process_vram_bytes` is gated on a RESOLVED adapter rather than on the configured
device id. With no resolution the sidecar passes the raw id to the EP, and an id DXGI could not map is
one this sampler must not pretend to understand either.

## Definition of Done

- [x] `/health` reports per-engine load bytes, or `None` with a reason it can be told apart from zero —
      `vram_at_load`, with `unavailable_reason` and three discard counters.
- [x] A figure is published only when no other build overlapped it; the discard count is on the wire.
- [x] The overlap case has a test that fails against the naive delta —
      `vram.rs · two_overlapping_builds_both_refuse_to_publish_and_are_counted_as_discards`, two real
      threads held open on a barrier.
- [x] Non-Windows and DXGI-failure paths report absent, never zero.
- [x] The unload cross-check is logged, and its agreement with the load figure is recorded here:
      **2 175 MB recorded at build, 2 183 MB freed at teardown** (R9700, DirectML, 2026-08-19). The 8 MB
      excess is the inference pass's own buffers, freed with the session; a tolerance of ±5 % is what this
      cross-check should be read against, and nothing tighter, because the sample is process-wide and any
      other consumer of the card moves it.
- [x] `research/architecture.md`'s *What does not exist yet* entry is updated — it said "No live GPU
      telemetry", and this is not live telemetry either. It now says what exists and what still does not.
- [x] `dew_flow_rag_qln · research/module_runtime.md` is told.

## The open tail

The **panel still shows 0**. This repository now publishes the figure; nothing reads it yet.
`RuntimeInspector.ResidentModels` hardcodes `new LoadedModelVm(Text(models, role), 0, cap)` with the
comment *"The sidecar does not report per-model VRAM; 0 says 'unknown', never 'none'"* — a comment that
became false on 2026-08-19. Reading `vram_at_load.embed_bytes` / `rerank_bytes` (bytes here, MB there)
is owned by `dew_flow_rag_qln` and is not tracked by this plan.

Two smaller ones, both deliberate: MIGraphX/WSL and CUDA-on-Linux report `discarded_not_sampled` forever
(DXGI is Windows), and the number is a LOAD, never residency — a second pass that allocates more is
invisible to it.
