# PLAN — the sidecar: from a binary that runs here to one a customer can build

> Status: **IMPLEMENTED, 2026-08-19**. Its console half was closed the same day in `dew_flow_rag_qln`; one
> item is left, and it is an operator's decision rather than work — see
> [The open tail](#the-open-tail). Scope as built: `src/` (`canary`, `recipes`, `introspection`, `config`,
> `state`, `wire`, `handlers`), `build-recipes.json`, `.github/workflows/{ci,release}.yml`, `LICENSE`,
> `NOTICE`, `THIRD-PARTY-NOTICES.md`, `Cargo.toml` and the README.
>
> **What it delivered:** the build recipe as data with a test that keeps it honest against the manifest in
> both directions; the verification gate on the wire (`/health.self_check`, two thresholds, measured at
> 1.0000004 on the reference build); the licence position, resolved from the artefacts rather than from
> memory and turning up a shipped MPL-2.0 dependency nobody had noticed; a generator for the canary's own
> oracle; a tag-driven release workflow for the one flavour that may ship; and an opt-in idle unload for the
> incident where three sidecars sat holding models nobody was using.
>
> **The phase-4 audit's own finding** is worth stating up front, because it is the kind of thing a plan
> hides from itself: three of its four ergonomics items were ALREADY DONE when the audit ran, and the plan
> text had simply gone stale. `/health.adapter.luid` tells two sidecars on two cards apart; `session_build_ms`
> on every pass is the sidecar saying it rebuilt; `max_body_bytes` was stated in an earlier task. Only the
> idle unload was real work.
>
> Related: the RAG repo's `todo/PLAN_rag_product.md` phase 4 (the settings and the compile button that drive
> this) and the MCP repo's `todo/PLAN_mcp_product.md` (the surface eventually served by these vectors).

## Where this stands

The engine is real and measured. On an AMD Radeon AI PRO R9700 through DirectML it embedded a repository's
463 members into 611 points in 33 seconds, dense and sparse from **one** forward pass through a vendored
`fastembed` fork. `/health` reports the distinction that matters — the providers the binary was **compiled**
with, the one that is **active**, and whether a session was ever successfully created — because an earlier
version reported the *requested* provider as though it were the outcome, and a binary that could not register
CUDA at all still answered `provider: "cuda"` while every embed failed in 4 ms.

`/tokenize` exists for a reason worth restating: the host cannot count tokens for this model. The .NET
tokenizer libraries cannot read a HuggingFace `tokenizer.json`, and asking an embedding endpoint costs the
embedding. So the chunker on the other side splits against **measured** counts, and there is deliberately no
fallback estimate — a chunker that quietly reverted to guessing would build a differently-shaped index with
nothing recording which algorithm produced it.

**What does not exist is everything between "it runs on this machine" and "a customer has one."**

## The distribution problem, which is the whole plan

The execution provider is a **compile-time** feature. That single fact decides the shape of everything else:

- `default = ["dml"]` — DirectML, Windows only.
- `cuda` needs a CUDA toolkit and an NVCC that satisfies both NVCC and the prebuilt ONNX Runtime's STL.
- `migraphx` needs ROCm **and** an ONNX Runtime built from source with `--use_migraphx`.

And the licences forbid the easy answer. **DirectML and NVIDIA CUDA/cuDNN are vendor-licensed and must never
be mirrored** into our registry, an image layer, or an offline bundle — one cached copy makes us a
redistributor. So we cannot ship a fat binary that works everywhere, and we are not going to try.

**Therefore: the customer's machine builds it, and the product drives that build.** That is the button in the
RAG settings, and this repository owes it three things.

### Phase 1 — a build recipe that is data, not prose

A machine-readable description of what to install and what to run, per platform and per provider: the
toolchain, the ONNX Runtime requirement, the feature flags, the expected build time. The RAG side detects the
card and picks a row; a human reading the same table gets the same answer.

The failure mode to design against is the one already recorded here: an ONNX Runtime whose ABI did not match
the `ort` crate deadlocked inside a `OnceLock` during the version check, so the sidecar hung instead of
saying why. **A version mismatch must fail loudly at startup**, before any model is loaded.

### Phase 2 — the verification gate

A build that compiles proves nothing about the vectors. Before a freshly built sidecar is trusted:

- embed a fixed reference text and compare against a stored reference vector — **cosine ≥ 0.999**;
- confirm the active provider is the one that was asked for, not a silent CPU fallback;
- report both in a form the RAG console can show.

Without this, a customer's DirectML build that silently ran on CPU would look identical to a working one,
except forty times slower, and the complaint would arrive as "your product is slow" rather than "my build is
wrong".

### Phase 3 — being a public repository

- **README**: what this is, what it is not, and how to build the CPU flavour in three commands.
- ~~**LICENSE and notices.**~~ **Done 2026-08-19.** `LICENSE` mirrors the position `dew_flow_mcp` already
  carries — public and proprietary, with the copyright holder still a placeholder awaiting counsel, exactly as
  there — plus `NOTICE` and `THIRD-PARTY-NOTICES.md`. `Cargo.toml` gained `license-file` and `publish = false`:
  the audit found that ours was the ONE crate in a 348-crate graph that did not say what it was.

  Three findings, each resolved from the artefacts rather than from memory. Of the 348 crates in the Windows
  build graph every one is permissive **except `option-ext` (MPL-2.0)**, which arrives four levels down through
  `dirs → hf-hub → fastembed` and ships — file-level copyleft, unmodified by us, so the whole obligation is to
  name it and its source, which the notices do. `r-efi` (MIT OR Apache OR LGPL) is in the metadata but in NO
  real target's graph, so its LGPL option never arrives. And the vendor providers are recorded as used and
  never redistributed, which is the sentence the whole distribution story rests on.
- **Release artefacts** for the flavours we *may* ship: the CPU build ships freely.
- ~~**The `cargo fmt` gate**, deferred deliberately~~ — **done 2026-08-19.** The deferral's reason had
  expired without anyone noticing: it rested on preserving the diff against the sources this crate was
  carried from, and the 2026-08-16 split into 17 modules had already destroyed that diff. Meanwhile the
  drift spread from "790 lines of `main.rs`" to **233 hunks across 19 files** — waiting made it worse
  monotonically. Applied as its own mechanical commit, exactly as the deferral asked, and the gate is on
  (`cargo fmt --package bge-sidecar --check`, ubuntu only; `--package` so the vendored fork stays
  byte-identical to upstream).
- ~~**The canary's reference vector cannot be regenerated from this repository**~~ — **done 2026-08-19.**
  The generator that made `src/canary-reference.f32le` was a script in the monorepo this crate was carried
  out of, and it did not travel; the canary — the one guard against a wrong-but-plausible vector — was left
  with an oracle nobody could reproduce.

  It is now a MODE of the binary rather than a script beside it: `--write-canary-reference [path]`, which
  goes through `load_dual`, the real provider selection and the production shape, and deliberately NOT
  through `load_validated_dual` — checking a new engine against the old reference is the circularity this
  tool has to stand outside of. It reports the cosine against the current file **before** writing, because
  regenerating to silence a failing canary is the misuse, and a printed distance makes that a decision
  rather than an accident.

  Verified on the card the day it was written: the regenerated vector scores cosine **1.000000000** against
  the committed one — this build reproduces the oracle exactly. It is also **not byte-identical**: 1012 of
  1024 elements differ, max delta 2.868e-07, which is float32 rounding on a GPU that is not bit-reproducible
  across runs. That measurement is the answer to the question a future regeneration will raise, and it is
  why the canary's threshold is a cosine.

### Phase 4 — the ergonomics the host already needs

- **Device selection** that reads as ground truth. The adapter mapping is solved — DXGI's high-performance
  ordering is not the plain enumeration the DirectML EP indexes, and `/health` reports the resolved adapter
  precisely because of that mismatch. The remaining work is two cards, two sidecars, and the host knowing
  which is which without guessing.
- **The sequence-cap ladder**: the engine rebuilds when the cap moves, so an unsorted pass thrashes a rebuild
  per batch. The host sorts descending and crosses each rung once; the sidecar should say when it rebuilt, so
  that cost is attributable instead of mysterious.
- **A shutdown that releases VRAM promptly.** Three sidecars were found running on this machine during
  development, two of them holding models nobody was using — the RAG runtime panel now surfaces that, and the
  sidecar should make the fix easy.
- **The body limit should say what it is** (2026-08-15, found from the host side). Every route runs on axum's
  DEFAULT 2 MB request cap — nothing here sets `DefaultBodyLimit`, so nothing states it either. Measured:
  980 KB to `/tokenize` succeeds, 2.1 MB returns `413`. The host now batches under it
  (`SidecarClient.RequestByteBudget`, `dew_flow_rag_qln`), so this is no longer a defect — but it cost an
  afternoon to find, because of HOW it presents: the server rejects the body while the client is still
  writing it, and the client raises "an established connection was aborted by the software in your host
  machine", which names the socket and nothing else. A 10,000-file repository died nine minutes into an
  indexing pass this way. Two things would have made it a five-minute diagnosis: reporting the cap in
  `/health` alongside `max_batch` (a limit a client cannot read is a limit it will guess at), and setting it
  explicitly so it is a decision rather than a framework default.

## Definition of Done

- [x] A build recipe exists as data, covering DirectML, CUDA, MIGraphX and CPU, with toolchain requirements
      and expected build times — `build-recipes.json`, kept honest by `src/recipes.rs` (test-only) in both
      directions: a recipe naming a feature the crate lacks, and a feature the crate gained that no recipe
      builds with. Build times are MEASURED (CPU: 4m29s cold, this machine) or ABSENT (CUDA, MIGraphX —
      no vendor toolchain here, and a guessed number is worse than none on the field an operator uses to
      decide whether to start a build now or after lunch).
- [x] An ABI or provider mismatch fails at startup with a message naming the mismatch — never a hang.
      Already true before this task and verified rather than assumed: `preflight_ort_dylib` probes the dylib
      through the stable C ABI and exits naming both versions (ort's own check DEADLOCKS instead of
      erroring), `preflight_migraphx_cache` refuses an unset or unwritable cache path, and
      `preflight_provider` refuses an EP absent from the build with the `--features` line that would add it.
- [x] A freshly built sidecar verifies itself against a reference vector at cosine ≥ 0.999 and reports its
      active provider. `/health.self_check` carries the cosine and BOTH thresholds (0.99 to serve, 0.999 to
      be called verified), and the provider is answered three ways beside it. **Closed on both sides the same
      day** (2026-08-19): `dew_flow_rag_qln` reads it in `RuntimeInspector.SelfCheck` and renders it inside
      the runtime panel's Provider cell — three badge states, so the build that serves and is not trusted
      stays visible — pinned by `SidecarSelfCheckTests` and six `RuntimePageTests` cases, each of which was
      mutation-checked.
- [x] README, LICENSE and notices exist, with the vendored fork's licence and the never-redistributed vendor
      providers both stated (2026-08-19). The one thing they say that was not anticipated: a shipped MPL-2.0
      dependency, `option-ext`.
- [~] The CPU flavour has a published release artefact — the workflow exists and packages it for Windows
      and Linux on a `v*` tag, with LICENSE, NOTICE, THIRD-PARTY-NOTICES.md and build-recipes.json inside
      every archive, and a `workflow_dispatch` rehearsal path that publishes nothing. **No tag has been
      cut**, so nothing is published yet: that is an operator's decision, not code.
- [x] Formatting is decided on its own merits, in its own commit, and the CI gate is switched back on
      (2026-08-19).
- [x] The canary reference has a generator that lives in THIS repository, so a deliberate model change can
      produce a new one (raised and built 2026-08-19: `--write-canary-reference`, verified on the card at
      cosine 1.000000000 against the committed oracle).

## The open tail

One item, and it is a decision rather than work:

1. ~~**The RAG console has to render the self-check.**~~ **Closed 2026-08-19**, the same day it was written
   down — which is the whole point of having written it down: the `vram_at_load` hand-off took a day to be
   noticed because nothing recorded that it was owed. `dew_flow_rag_qln` now maps `/health.self_check` onto
   `SidecarSelfCheckVm` and renders it in the runtime panel's Provider cell, beside the active provider,
   because the failure the gate exists for — a DirectML build that silently ran on CPU — has correct
   numbers and wrong hardware. Recorded there in `research/module_runtime.md`.

2. **The first release tag.** Cutting `v0.1.0` publishes the CPU archives. Nobody has decided to, and the
   `{{COPYRIGHT_HOLDER}}` placeholder in LICENSE and NOTICE should be resolved by counsel before a binary
   carries them to a third party.

Everything else this plan asked for is in the repository and tested.
