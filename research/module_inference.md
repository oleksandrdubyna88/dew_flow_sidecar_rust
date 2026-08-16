# Module — inference

> `src/inference.rs` (`embed_blocking`, `embed_natural`, `rerank_blocking`, `score_documents`,
> `pin_shape`, `embed_settling`, `lock_or_refuse`), `src/engine_cache.rs` (`RungCache`),
> `src/state.rs` (`Engines`, `Limits`), `src/compile_cache.rs` (`CompileWatch`, `CachePathLease`)
> and `src/bookkeeping.rs` (the recorders, `remember_engine`). The system as it is, 2026-08-16.

## Purpose

Turn text into vectors and query–document pairs into scores, on a GPU, without rebuilding a model
session more often than the hardware forces.

Almost every non-obvious decision here is a workaround for one execution provider or one measured
failure. They are recorded in place because none of them is guessable from the code's shape.

## Flow

```mermaid
flowchart TD
    REQ["/embed request"] --> CAP["cap_for(kind, requested, loaded)"]
    CAP --> USAGE["token_usage — before any padding"]
    USAGE --> PIN{"pin shapes?<br/>(MIGraphX, batch ≥ 2)"}
    PIN -->|yes| RULER["pin_shape — ruler rows to one constant (batch, seq)"]
    PIN -->|no| NATURAL
    RULER --> NATURAL["embed_natural"]

    NATURAL --> WAIT["lock_or_refuse(engines.embed)<br/>⏱ queue_wait_ms"]
    WAIT -->|holder past its ceiling| REFUSE["503 EngineWedged<br/>names the activity + elapsed"]
    WAIT --> STAMP["InFlightStamp — what holds it, since when"]
    STAMP --> RESIDENT{"rung resident?"}
    RESIDENT -->|no| CLAIM["CachePathLease::hold — this engine's cache slice"]
    RESIDENT -->|yes| RUN
    CLAIM --> BUILD["load_validated_dual<br/>⏱ session_build_ms — phase `building`"]
    BUILD --> REMEMBER["remember_engine — LRU insert;<br/>evicted rung torn down OFF the lock"]
    REMEMBER --> RUN["embed_settling → engine.embed<br/>⏱ inference_ms — phase `running`"]
    RUN --> RELEASE["drop the lease — the first kernel launch<br/>has read the path"]
    RELEASE --> GROWTH["CompileWatch delta over THIS engine's<br/>cache subdir ⇒ compile_cache_grew_mb"]
    GROWTH --> UNZIP["unzip (dense, sparse) per text"]
    UNZIP --> UNPIN{"was pinned?"}
    UNPIN -->|yes| STRIP["unpin_rows — drop ruler rows"]
    UNPIN -->|no| OUT
    STRIP --> OUT["EmbedResponse + dimension + usage + timings"]
```

## Core structures

| Type | Role |
|---|---|
| `Engines` | Two mutex-guarded slots — `embed: Mutex<RungCache<Bgem3DualEmbedding>>`, `rerank: Mutex<Option<TextRerank>>` — each paired with its own `Mutex<Option<InFlight>>` stamp, read **without** taking the engine |
| `InFlight` / `InFlightStamp` | The holder's phase, label and start instant; RAII, so `?` and panics clear it. The only thing that can see a wedge, because it lives outside the mutex the wedge is under |
| `WedgePolicy` | The ceilings: `building` 3600 s, `running` 900 s, `/unload` 30 s, poll 50 ms, opt-in process exit (default off) |
| `RungCache<T>` | One engine per `max_length` ("rung"), least-recently-**used** eviction, capacity from `EMBED_ENGINE_CACHE_RUNGS` |
| `Limits` | `max_length` (clamped 16…8192) and `max_batch` (≥ 1), resolved per request over the configured defaults |
| `PassTimings` | `queue_wait_ms`, `session_build_ms`, `inference_ms`, `compile_cache_grew_mb` |
| `CompileWatch` | A pass's growth in **one engine's own** cache subdirectory (`EMBED_CACHE_ENGINE` / `RERANK_CACHE_ENGINE`, shared with `CachePathLease` so builder and measurement cannot drift). Summing the whole tree charged a rerank compile to a concurrent embed pass — the engines hold independent mutexes, so concurrent is the ordinary case after a restart |
| `CachePathLease` | An RAII claim on the process-global MIGraphX cache path, for ONE engine. Taken by the caller that BUILDS and held across that engine's **first pass**, because the EP reads the path at the first kernel launch and not at session build. `load_dual`/`load_rerank` require one by type, and a lease held for the other engine is refused before anything is pinned or loaded. A resident engine claims nothing |
| `TokenUsage` | `token_count[]`, `truncated[]`, effective `max_length`, `token_accounting` |
| `Bgem3DualEmbedding` | **One** session yielding both BGE-M3 heads per forward pass |
| `TextRerank` | The `bge-reranker-v2-m3` cross-encoder |

## Entry points

| Function | Purpose |
|---|---|
| `embed_blocking` | Owns capping, token accounting and shape pinning; delegates the pass |
| `embed_natural` | The pass itself: lock, build if needed, run, time each span |
| `remember_engine` | Files a built engine under its rung and hands the EVICTED one to `teardown_off_the_lock` — an ort teardown done inline is paid by whoever is queued, and lands in their `queue_wait_ms` |
| `rerank_blocking` / `score_documents` | The same shape for the cross-encoder; scores come back **in input order** |
| `cap_for(kind, requested, loaded)` | A `query` may never move the loaded cap — it arrives interleaved with index passes |
| `pin_shape` / `unpin_rows` | Splice ruler rows to one constant shape, then strip them |
| `embed_settling` | Re-runs the same batch when a freshly built session returns a short one |
| `lock_or_refuse` | Every engine acquisition: heals poison, waits while the holder is alive, refuses (`503`) once it is wedged |
| `wedge_action` / `spawn_wedge_watchdog` | The verdict, and the log line that reports it without being asked |

## Measured decisions

| Decision | The measurement behind it |
|---|---|
| One session for both heads | The official FP32 export returns `sentence_embedding` and `token_embeddings` per run, so the dense/sparse split is post-processing rather than two sessions |
| **Not** the INT8 all-in-one model | Retrieval quality over speed — a locked decision |
| LRU cache, capacity ≥ 1 | A Fast lane walks the ladder down then up, crossing the boundary twice per pass; before the cache each crossing evicted both engines — ~5.5 min per pass, forever. Rebuilding one rung costs 156–173 s because MIGraphX re-materialises a ~2.4 GB program |
| Shape pinning under MIGraphX | Each distinct `(batch, seq)` compiles its own program: 92–162 s and +2.19 GB of cache apiece, measured 2026-07-30 |
| Settling retry, bounded to 3 | The first `session.run` on a fresh MIGraphX session returns short: at `(64, 1024)`, dense gave 80 rows for a 128-row input and the sparse head arrived shaped `[16, 1024, 1024]`. Every later run is correct |
| No throwaway warm-up | It charged an extra full-cap pass on every engine build, pushing a pass's first request to ~608 s past the host's 600 s budget — the pass "completed" with 0 methods embedded |
| Per-engine EP cache directory | The dense, sparse and rerank engines run the same graph at the same pinned shape, so the EP's cache key collided; a sparse session loaded another engine's program and died on `index < dim` (2026-07-27) |
| Rows zipped per text | The settling retry polices one length, so a short first run cannot shorten one head without the other |
| Timing spans measured separately | Queue wait and session build are both infrastructure wait, but the remedies differ — concurrency against warm-up — and one bucket for both explains neither |
| Wedge ceilings 900 s / 3600 s, not seconds | A first-ever shape compile is 214 s of CORRECT slowness, a first rerank pass 92–162 s, and the slowest honest first request on record ~608 s. A ceiling below those would refuse work that was about to succeed |
| The process-exit last resort is opt-in and OFF | Exiting mid-compile is exactly how a corrupt `.mxr` reaches the cache (2026-07-31), so the exit ceiling is measured **from the wedge verdict**, never from the phase start |
| The ruler string is one `OnceLock` allocation | ~110 KB rebuilt per pinned request and cloned per padding row — megabytes of allocator work per request, in the flavour that exists to avoid expensive work |

## Dependencies

- `fastembed` (vendored, one marked patch in `sparse_text_embedding/impl.rs` — output selection by name
  for the MIGraphX EP), `ort`, `tokenizers`, `anyhow`, `tracing`.
- The GPU, through whichever execution provider was compiled in.
- `MODEL_CACHE_DIR` for weights; `ORT_MIGRAPHX_MODEL_CACHE_PATH` for compiled programs.

## Tests

Inline `#[cfg(test)] mod tests` in `main.rs`: the rung cache's hand-back, ladder walk, capacity-1
behaviour and LRU eviction order; `pin_shape`/`unpin_rows` round-tripping; `embed_settling` costing
exactly one run when settled; `aligned_scores` returning document order; `cap_for`; the pass log line;
and the `PassTimings` wire field names.

The wedge guards are tested with a held mutex and an `Instant` moved into the past — no GPU, no clock
abstraction: a wedged holder is refused within milliseconds and the message names it; a **healthy** cold
compile is still waited out (the counter-guarantee, and the reason the ceilings are minutes); an
unstamped hold still hits the fallback ceiling; poison still heals under the deadline; and the shipped
ceilings are asserted from the env defaults so a stray override cannot green them.
