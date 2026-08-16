# research/

Documentation of the system **as it is** — plus the design records of plans that already shipped.

The test is one question: **does this describe code that exists today?** If yes it lives here. Work
still to be built lives in [`../todo/`](../todo/), and a plan moves across when it ships, with its
status changed to `IMPLEMENTED <date>` and its **deviations recorded**.

The [README](../README.md) at the repository root is the operator's document — how to build each
flavour, what the endpoints are, the VRAM arithmetic, the WSL/MIGraphX recipe. This folder holds the
*why* behind the shapes it describes.

## What is here

| Document | What it is |
|---|---|
| [architecture.md](architecture.md) | The whole service: what it is, the container diagram, one embed end to end, the cross-cutting rules, and an explicit list of what does **not** exist yet |
| [module_inference.md](module_inference.md) | Engines, the rung cache, shape pinning, the settling retry, the timing spans — and the measurement behind each |
| [module_http_surface.md](module_http_surface.md) | The five routes, their wire shapes, and the fields that exist to prevent a specific wrong reading |
| [module_runtime_preflight.md](module_runtime_preflight.md) | Config, provider resolution, the DXGI device mapping, startup checks |
| [PLAN_reliability_tail.md](PLAN_reliability_tail.md) | Design record, IMPLEMENTED 2026-08-16 — the whole tail of the 24/7 audit, eight items, each with the observation that made it a defect. Also the record of a plan whose own line references decayed against its own work twice, and what that cost |
| [PLAN_tokenizer_registry.md](PLAN_tokenizer_registry.md) | Design record, IMPLEMENTED 2026-08-16 — tokenizers as a startup-built registry, and `GET /models` |
| [PLAN_response_timings.md](PLAN_response_timings.md) | Design record, IMPLEMENTED 2026-08-15 — `timings` and `request_id` on `/embed` and `/rerank` |

## Sibling repositories

Citations across repositories are written as **paths, not links** — a relative link that resolves on one
machine is worse than a citation that names its source.

- `dew_flow_rag_qln` — the product that drives the build and consumes the vectors (private)
- `dew_flow_mcp` — the tool surface those vectors eventually answer through (public)
- `dew_flow_benchmark` — the measurement harness the response timings were added for
