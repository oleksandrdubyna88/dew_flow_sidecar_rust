# `http/` — the sidecar's contract suite

One folder per route group, per
[`.claude/rules/shared/common/http-contracts.md`](../.claude/rules/shared/common/http-contracts.md).
The groups are the ones the source already uses.

| Folder | Routes |
|---|---|
| [`introspection/`](introspection) | `GET /health`, `GET /models` — what the process says about itself |
| [`inference/`](inference) | `POST /embed`, `/rerank`, `/tokenize`, `/unload` |

## The one thing this suite will not do

**It never runs a real inference.** A non-empty `/embed` or `/rerank` builds an engine on the GPU, and
[`gpu-lease.md`](../.claude/rules/shared/common/gpu-lease.md) says the card is claimed through a lease
and never grabbed — a suite that must be runnable by anyone, at any time, in CI, cannot take it. So the
suite exercises routing, validation, the body limit, the batch caps and the refusals, and leaves the
card alone.

That is a smaller claim than "the sidecar embeds correctly", and it is the honest one. Embedding
quality is the benchmark's subject; this tier's subject is whether a request reaches the right handler
and comes back in the shape a client can read.

## Running it

```bash
npm ci --prefix http                                  # once per machine

PORT=5320 ./target/release/bge-sidecar &              # or cargo run --release
node .claude/rules/shared/tools/http-run.mjs --env local --target http://127.0.0.1:5320
```

No configuration and no tokens: the sidecar binds loopback and is spoken to by one host on the same
machine. The verdict is the exit code — `0` pass · `1` contract regression · `3` environment ·
`4` configuration · `5` no valid report.

Two requests build large bodies in a pre-request script rather than carrying them in the file: a
4097-text batch (the tokenize cap, refused with a message naming the number) and a 2.2 MiB body (the
router's limit, refused before any handler runs). Neither belongs in git as a literal.
