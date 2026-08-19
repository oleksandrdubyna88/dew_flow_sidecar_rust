# PLAN — the sidecar: from a binary that runs here to one a customer can build

> Status: **plan; the engine works and is verified on an R9700, the distribution story is not built.**
> Scope: `src/`, `.github/workflows/ci.yml`, and this repository's public presentation.
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
- **LICENSE and notices.** `vendor-fastembed/` is an upstream fork carried under its own Apache-2.0 licence
  and kept byte-identical to upstream; its patches are marked in place so the provenance survives. The vendor
  provider terms above belong in the notices too — stating that we never redistribute them is part of the
  reason we may use them.
- **Release artefacts** for the flavours we *may* ship: the CPU build ships freely.
- ~~**The `cargo fmt` gate**, deferred deliberately~~ — **done 2026-08-19.** The deferral's reason had
  expired without anyone noticing: it rested on preserving the diff against the sources this crate was
  carried from, and the 2026-08-16 split into 17 modules had already destroyed that diff. Meanwhile the
  drift spread from "790 lines of `main.rs`" to **233 hunks across 19 files** — waiting made it worse
  monotonically. Applied as its own mechanical commit, exactly as the deferral asked, and the gate is on
  (`cargo fmt --package bge-sidecar --check`, ubuntu only; `--package` so the vendored fork stays
  byte-identical to upstream).
- **The canary's reference vector cannot be regenerated from this repository** (found 2026-08-19 while
  auditing the code's own citations). `canary.rs` cited `scripts/generate-canary-reference.mjs`; that
  script stayed in the monorepo this crate was carried out of, so `src/canary-reference.f32le` has no
  reproducible provenance here. It works — every fresh engine is checked against it — but a deliberate
  model change has nothing to regenerate it with, which makes phase 2 partly unimplementable until it is
  replaced. The docstrings now say this instead of naming a file that is not there.

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

- [ ] A build recipe exists as data, covering DirectML, CUDA, MIGraphX and CPU, with toolchain requirements
      and expected build times.
- [ ] An ABI or provider mismatch fails at startup with a message naming the mismatch — never a hang.
- [ ] A freshly built sidecar verifies itself against a reference vector at cosine ≥ 0.999 and reports its
      active provider, and the RAG console shows both.
- [ ] README, LICENSE and notices exist, with the vendored fork's licence and the never-redistributed vendor
      providers both stated.
- [ ] The CPU flavour has a published release artefact.
- [x] Formatting is decided on its own merits, in its own commit, and the CI gate is switched back on
      (2026-08-19).
- [ ] The canary reference has a generator that lives in THIS repository, so a deliberate model change can
      produce a new one (raised 2026-08-19 — phase 2 cannot be completed without it).
