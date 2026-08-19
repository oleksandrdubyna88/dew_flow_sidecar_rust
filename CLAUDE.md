# Claude Code — Project Rules for dew_flow_sidecar_rust

These rules apply to all code in this repository and override Claude's defaults. The family-wide
doctrine lives in [.claude/rules/shared](.claude/rules/shared) (a submodule of
`dew_flow_conventions`); the Rust rules are
[.claude/rules/shared/rust/doctrine.md](.claude/rules/shared/rust/doctrine.md) — written FROM this
crate's own practices, so a change here that contradicts it is either a defect or a doctrine edit,
never silently both.

## Project Overview

`bge-sidecar` is the family's inference engine: a single-binary Rust HTTP service over ONNX Runtime
serving BGE-M3 dense + learned-sparse embeddings, BGE-Reranker-v2-M3 rerank scores and tokenization
on the one local GPU. It has no database, no queue, and no knowledge of any .NET repo —
`dew_flow_rag_qln` speaks to it over HTTP with DTOs defined independently on each side (`wire.rs`
here), which is the deliberate shape of a cross-language boundary.

**Read first:** [README.md](README.md) (build flavors, the memory model, the wedge detector), then
`research/` for the design records.

## Commands

```bash
# Build — the execution provider is a COMPILE-TIME feature; pick the machine's flavor
cargo build --release                                    # AMD on Windows (DirectML, the default)
cargo build --release --no-default-features              # CPU only
cargo build --release --no-default-features --features cuda   # NVIDIA
# AMD on Linux/WSL (ROCm via MIGraphX): build INSIDE WSL with the separate target dir —
# see README.md; the two OS flavors must never clobber each other's target/.

# Tests (CI runs exactly this, per flavor)
cargo test --release --locked
cargo test --release --locked --no-default-features      # the CPU leg
```

`--locked` always: a lockfile drift is a build failure, not a surprise. `cargo fmt --check` runs in
CI on ubuntu only — see the workflow's own comment before "fixing" that.

## Repository-specific rules

1. **Request paths never panic the process.** Inference runs on `spawn_blocking`; a `JoinError` is a
   bug answered as 500 with the panic payload surfaced (`join_error_text`). Production `.expect()`
   only with the invariant stated inline at the call site.
2. **Locks refuse and heal.** `lock_or_refuse` + wedge policy on the engines (`/health` says
   `wedged`, new requests are refused, never queued behind a stuck forward pass); every recovery
   path calls `clear_poison()` AND logs — the silent `let Ok(..) = lock() else { return }` skip is
   the named anti-pattern (`bookkeeping.rs` carries its incident).
3. **The GPU is shared.** Anything that loads or runs a model competes with the rest of the family —
   the lease rule is [.claude/rules/shared/common/gpu-lease.md](.claude/rules/shared/common/gpu-lease.md);
   this process is usually the SERVER side of that story, but a test or script that drives it still
   takes the lease.
4. **`/health` tells the truth**: wedge state, loaded engines, real limits (`max_body_bytes`,
   `loaded_max_batch`), build provenance hashes. A new failure mode worth seeing gets a field there,
   never only a log line.
5. **The canary guards numbers.** Embedding output is compared against the committed
   `canary-reference.f32le` — a wrong-but-plausible vector is a red check, not a quiet quality drop.
   Touch the canary only when the model or the numeric contract deliberately changes.

## Definition of Done

- [ ] `cargo build --release --locked` and `cargo test --release --locked` green for the flavor(s)
      the change touches; the CPU leg too when the change is not EP-specific.
- [ ] New behaviour has tests beside the code (`#[cfg(test)]`); a fix has a test watched failing.
- [ ] The relevant `research/module_*.md` moved with the code; a finished plan was promoted
      (`node .claude/rules/shared/tools/plan-lifecycle.mjs` is CI's check).
