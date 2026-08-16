# todo/

Plans for work that is **not finished**. Documentation of the system as it *is* belongs in `research/`.

The test is one question: **is someone still supposed to build this?** If yes it lives here; once it ships,
the plan moves to `research/` with its status changed to `IMPLEMENTED <date>` and its deviations recorded.

Every plan starts with a status line on line 2–3 and carries: the symptom or goal before any solution,
references to real code as `file.rs:line` (verified, not guessed), a build order, a test plan, and a
Definition of Done.

## Currently open

| plan | status | scope |
|---|---|---|
| [PLAN_sidecar_product.md](PLAN_sidecar_product.md) | engine works and is measured on an R9700; distribution is not built | build recipe as data, self-verification against a reference vector, public-repository hygiene, host ergonomics |
| [PLAN_tokenizer_registry.md](PLAN_tokenizer_registry.md) | **step 1 implemented 2026-08-16** — the registry; steps 2–3 open | ✔ tokenizers by name instead of a two-arm `match`, loaded at startup, an unknown name refused naming the registered set. Open: a `GET /models` read stating kind · dimension · max sequence · tokenizer per entry (with unknown as its own state), and the embedding dimension on `/embed`. Driven by the consumer's need to chunk each embedding model with **its own** tokenizer. Landed together with [PLAN_reliability_tail.md](PLAN_reliability_tail.md) item 2, as both plans asked. Consumer: `dew_flow_rag_qln · todo/PLAN_tokenizer_contract_and_chunk_coverage.md` |
| [PLAN_reliability_tail.md](PLAN_reliability_tail.md) | partially implemented, 2026-08-16 — items 2 (`/tokenize` off the reactor) and 3 (the ruler) are done; 1, 4, 5, 6, 7, 8 open. **Re-anchored against `d0139b1`**, because the fix commit moved every line it cited | what the 24/7 audit found and the same-day fixes did not take: the MIGraphX cache path race between build and first launch, the twice-per-request cache walk, an engine dropped inside the lock its waiters queue on, and the 3 821-line `main.rs` |

Implemented plans live in [`../research/`](../research/) — most recently
[PLAN_response_timings.md](../research/PLAN_response_timings.md) (additive `timings` + `request_id`
on `/embed` and `/rerank`, 2026-08-15).

## The constraint that shapes every plan here

The execution provider is a **compile-time** feature, and the vendor providers — DirectML, NVIDIA CUDA and
cuDNN — are licensed such that we may use them and must never redistribute them. One cached copy in a
registry, an image layer or an offline bundle makes us a redistributor.

So there is no fat binary that works everywhere, and no plan here should propose one. The customer's machine
builds what it needs, and the product drives that build.

## Sibling repositories

- `dew_flow_rag_qln` — the product that drives the build and consumes the vectors (private)
- `dew_flow_mcp` — the tool surface those vectors eventually answer through (public)
