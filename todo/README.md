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
| [PLAN_tokenizer_registry.md](PLAN_tokenizer_registry.md) | plan only, 2026-08-16 | tokenizers by name instead of a two-arm `match`, a `GET /models` read stating kind · dimension · max sequence · tokenizer per entry (with unknown as its own state), and the embedding dimension on `/embed`. Driven by the consumer's need to chunk each embedding model with **its own** tokenizer: `/tokenize` already counts with the model's real `tokenizer.json` and with truncation off, but a third model is a code change in three places and a caller cannot discover what this build can count for. Shares a handler with [PLAN_reliability_tail.md](PLAN_reliability_tail.md) item 2 — startup loading subsumes its pre-warm. Consumer: `dew_flow_rag_qln · todo/PLAN_tokenizer_contract_and_chunk_coverage.md` |
| [PLAN_reliability_tail.md](PLAN_reliability_tail.md) | partially implemented, 2026-08-16 — item 3 (the ruler) landed with the CRITICAL/HIGH fixes; the rest is open | what the 24/7 audit found and the same-day fixes did not take: the MIGraphX cache path race between build and first launch, `/tokenize` on the reactor, the twice-per-request cache walk, and the 2 887-line `main.rs` |

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
