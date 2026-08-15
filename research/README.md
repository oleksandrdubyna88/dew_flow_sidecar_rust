# research/

Documentation of the system **as it is** — plus the design records of plans that already shipped.

The test is one question: **does this describe code that exists today?** If yes it lives here. Work
still to be built lives in [`../todo/`](../todo/), and a plan moves across when it ships, with its
status changed to `IMPLEMENTED <date>` and its **deviations recorded**.

The authoritative description of the running contract is the repository
[README](../README.md) — endpoints, configuration, the VRAM budget and the MIGraphX mechanics. This
folder holds the *why* behind changes to it.

## What is here

| Document | What it is |
|---|---|
| [PLAN_response_timings.md](PLAN_response_timings.md) | Design record, IMPLEMENTED 2026-08-15 — additive `timings` (queue wait · session build · inference · compile) and `request_id` on `/embed` and `/rerank`, so infrastructure wait stops dying in the log file |

## Sibling repositories

- `dew_flow_rag_qln` — the product that drives the build and consumes the vectors (private)
- `dew_flow_mcp` — the tool surface those vectors eventually answer through (public)
- `dew_flow_benchmark` — the measurement harness these timings were added for
