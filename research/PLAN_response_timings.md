# PLAN — per-request timings on the wire: queue wait, session build, inference, compile

> Status: **IMPLEMENTED, 2026-08-15** — authored and shipped the same day. Scope: `src/main.rs`
> (request/response structs and the two blocking inference paths), `README.md` endpoint contract,
> `todo/README.md` index.
>
> **Its `src/main.rs:NNNN` references are historical.** They were written when `main.rs` was one
> 4 744-line file; [PLAN_reliability_tail.md](PLAN_reliability_tail.md) split it into 17 modules on
> 2026-08-16, so those numbers now point past the end of a 400-line file. They are left as written
> rather than renumbered — the code did not move within a file, it moved to different files, and a
> design record is a statement about the day it was made. Follow the NAMES, not the numbers.
>
> **Deviations.** One: the DoD's original `cargo fmt --check` gate was dropped — the checkout carries
> 109 pre-existing rustfmt diffs (the fmt gate is `todo/PLAN_sidecar_product.md`'s open work), so the
> new code matches the surrounding style instead of introducing a second one. Everything else shipped
> as planned: 51 tests green, three of them new (`pass_log_prefixes_the_request_id_only_when_one_was_sent`,
> `pass_timings_serialize_under_their_contract_names`, `the_struct_update_path_keeps_the_inner_timings_and_echo`).
>
> **Open tail.** The manual observation (two concurrent `/embed` calls against one warm engine — the
> second's `queue_wait_ms` ≈ the first's `inference_ms`) has not been run live yet; it needs the GPU
> host free and is worth doing on the next occasion the sidecar is up anyway. Until then the
> queue-wait number is correct by construction (measured around the mutex only) but unobserved under
> real contention.
>
> Requested by the benchmark programme (`dew_flow_benchmark · todo/PLAN_rag_bench_repo.md` §5.2/§5.3):
> a run's wall-clock must be attributable to three buckets — tools · thinking · **infrastructure
> wait** — and today every number that could feed the third bucket dies in this sidecar's own log file.

## Symptom

The sidecar measures its own passes and then keeps the numbers to itself:

- The per-pass wall time is captured (`pass_log_message`, `src/main.rs:1565`) from `embed_natural`
  (`src/main.rs:1426-1435`) and `score_documents` (`src/main.rs:1502-1513`) — but only as a log line.
  The HTTP caller receives vectors and token accounting, never a duration.
- **Queue wait is invisible to everyone.** Concurrent requests serialize on the engine mutex
  (`lock_healing`, `src/main.rs:1418` for embed, `:1461` for rerank); the pass timer starts only
  *after* the lock is held, so a request that waited 8 s behind another caller's pass and then ran
  0.4 s reports nothing anywhere — the caller sees 8.4 s and has no way to learn that 8 s of it was
  contention, not model speed. This is exactly the "busy card reads as a slow model" failure the
  benchmark's three-bucket rule exists for.
- Session build time (`load_validated_dual` at `src/main.rs:1421`, `load_rerank` at `:1464`) is logged
  but not returned — a cold first call is indistinguishable, on the wire, from a slow model.
- Concurrent requests cannot be told apart even in the log: no request id is read, generated, or
  echoed anywhere.

The one per-request datum that already reaches the caller is token accounting
(`TokenUsage`, `src/main.rs:807-820`) — the precedent this plan extends.

## The contract (additive — no version bump, no consumer breakage)

`POST /embed` and `POST /rerank` requests gain one optional field; both responses gain two.

Request:

```jsonc
{ "request_id": "…" }   // optional, default empty; opaque to the sidecar
```

Response:

```jsonc
{
  "request_id": "…",              // echoed verbatim; empty when the caller sent none
  "timings": {
    "queue_wait_ms": 0,           // waiting for the engine mutex behind another request —
                                  // the caller's INFRASTRUCTURE WAIT, never model speed
    "session_build_ms": 0,        // building + canary-checking the session; 0 on a warm engine
    "inference_ms": 0,            // the forward pass(es), settling re-runs included — honest,
                                  // it is what the engine cost (same rationale as src/main.rs:1425)
    "compile_cache_grew_mb": 0    // >0 = MIGraphX compiled this shape during the pass (lazy save,
                                  // measured across the pass per src/main.rs:1538-1542)
  }
}
```

Decisions, stated rather than implied:

- **`queue_wait_ms` and `session_build_ms` stay separate.** Both are infrastructure wait, but the
  remedies differ (concurrency vs warm-up), and folding them would repeat the `admit` mistake the
  benchmark's lessons record — a bucket that mixes two causes explains neither.
- **`inference_ms` includes settling re-runs** — it is what the caller's request actually spent in
  the engine, mirroring the existing log line's honesty.
- **`/tokenize` and `/health` are unchanged.** Tokenize is a pure CPU vocabulary lookup with no
  queue and no session; health must never wait on anything (`try_lock` only) and so has nothing
  truthful to report per-request.
- The pass log lines gain the request id as a prefix when one was sent — today two concurrent
  requests interleave in the log with no way to attribute a line to either.

The existing .NET consumer (`dew_flow_rag_qln · src/Rag.Infrastructure/Embedding/SidecarClient.cs`)
deserializes a subset of the wire shape and `System.Text.Json` ignores unknown members by default —
additive fields cannot break it.

## Build order

1. `PassTimings` (serialize, `Default`, `Clone, Copy`) + `request_id` on `EmbedRequest`
   (`src/main.rs:773`), `RerankRequest` (`:859`), `EmbedResponse` (`:822`), `RerankResponse` (`:873`).
   Empty-input early returns (`src/main.rs:1131-1133`, `:1182-1184`) carry defaults.
2. `embed_natural` measures the three spans (lock at `:1418`, build at `:1419-1423`, pass already at
   `:1426`) and returns them on `EmbedResponse.timings`; `embed_blocking` carries them through both
   the pinned (`:1385-1391`) and natural (`:1393`) paths — the struct-update return already does.
3. `score_documents` returns `(scores, timings)` with the pass span it already measures (`:1502`);
   `rerank_blocking` adds the lock (`:1461`) and build (`:1462-1465`) spans on top.
4. Handlers echo `request_id`; pass log lines prefix it when non-empty.
5. `README.md`: the endpoint table (`README.md:46-48`) and a short section beside token accounting
   (`README.md:105`) documenting the four fields and the queue-wait semantics.
6. `todo/README.md` *Currently open* table gains this plan's row (and drops it on promotion).

## Test plan

Inline `mod tests` (`src/main.rs:2067`), matching the existing style — the timing spans themselves are
wall-clock and stay untested, but every pure seam is:

- `pass_log_message` with a request-id prefix: with and without an id, with and without cache growth.
- `PassTimings` serialization shape: the four field names are a wire contract, pinned by a test.
- Struct-update propagation: an `EmbedResponse` built the way `embed_blocking` builds it keeps the
  inner call's timings (guards the `..` against a later refactor dropping the field).
- Manual observation (documented in the promoted plan, not CI): two concurrent `/embed` calls against
  one warm engine — the second's `queue_wait_ms` ≈ the first's `inference_ms`.

## Definition of Done

- [ ] Both `/embed` and `/rerank` responses carry `timings` + `request_id`; a warm pass reports
      `session_build_ms = 0`; an uncontended pass reports `queue_wait_ms ≈ 0`.
- [ ] Queue wait is measured around the engine mutex only — never around `spawn_blocking` scheduling.
- [ ] `README.md` documents the fields and states that `queue_wait_ms` is infrastructure wait.
- [ ] The QLN consumer builds and runs unchanged (additive fields only).
- [ ] `cargo test` green and the compiled flavor builds. (No `cargo fmt` gate: the checkout carries
      109 pre-existing rustfmt diffs — the fmt gate is `PLAN_sidecar_product.md`'s open work, and new
      code matches the surrounding style instead of introducing a second one.)
