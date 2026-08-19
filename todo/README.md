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
| [PLAN_sidecar_product.md](PLAN_sidecar_product.md) | engine works and is measured on an R9700; distribution is not built. The `cargo fmt` gate closed 2026-08-19 | build recipe as data, self-verification against a reference vector, public-repository hygiene (LICENSE and notices are the outstanding blocker), host ergonomics |

Implemented plans live in [`../research/`](../research/).

| Promoted | Plan | What it delivered |
|---|---|---|
| 2026-08-19 | [PLAN_vram_per_engine.md](../research/PLAN_vram_per_engine.md) | How much of the card each engine's BUILD took, published only when it could be attributed to that build alone. Measured on the R9700 the day it shipped: the dual embed session costs **2 175 MB**, and teardown returned **2 183 MB**. Two statics decide attribution rather than one — a build counter can tell you whether you STARTED alone and not whether somebody joined half-way — and absent has four distinguishable causes rather than a bare `None`, because a `0` here reads as "this engine is free". The plan's own step 3 was refuted before it was written: turning `loaded`'s three booleans into objects would have emptied the consumer's runtime panel, so `vram_at_load` is a sibling and a test now pins the booleans |
| 2026-08-16 | [PLAN_reliability_tail.md](../research/PLAN_reliability_tail.md) | The whole tail of the 24/7 audit, all eight items. `/tokenize` off the reactor and under a cap; the ruler allocated once; compile growth measured per ENGINE (it was charging one engine's compile to the other); the evicted engine torn down off the lock; the body limit stated · reported · logged; poisoned bookkeeping healed and logged instead of going quiet forever; `main.rs` **split** into 17 modules (largest ~700 lines) with the copy-paste the split exposed removed. Last: the MIGraphX cache-path race — `with_engine_cache` became `CachePathLease`, an RAII claim held across the build **and the first kernel launch**, required by type at both loaders. |
| 2026-08-16 | [PLAN_tokenizer_registry.md](../research/PLAN_tokenizer_registry.md) | Tokenizers are a startup-built registry rather than a two-arm `match`, so a model is a row and a path; an unknown name is refused naming the registered set; `GET /models` states kind · dimension · max sequence · tokenizer per row with unknown as its own state; `/embed` reports the width of the vectors it just returned. Took [PLAN_reliability_tail.md](../research/PLAN_reliability_tail.md) item 2 with it — `/tokenize` off the reactor, under a cap. |
| 2026-08-15 | [PLAN_response_timings.md](../research/PLAN_response_timings.md) | Additive `timings` + `request_id` on `/embed` and `/rerank`. |

## The constraint that shapes every plan here

The execution provider is a **compile-time** feature, and the vendor providers — DirectML, NVIDIA CUDA and
cuDNN — are licensed such that we may use them and must never redistribute them. One cached copy in a
registry, an image layer or an offline bundle makes us a redistributor.

So there is no fat binary that works everywhere, and no plan here should propose one. The customer's machine
builds what it needs, and the product drives that build.

## Sibling repositories

- `dew_flow_rag_qln` — the product that drives the build and consumes the vectors (private)
- `dew_flow_mcp` — the tool surface those vectors eventually answer through (public)
