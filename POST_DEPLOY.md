# Post-deploy checks — bge-sidecar

Per [`.claude/rules/shared/common/post-deploy-checks.md`](.claude/rules/shared/common/post-deploy-checks.md).

Nothing is deployed anywhere: this binary is **built and installed**, and the host that speaks to it
spawns or connects to whatever is on the machine. So "prod" is the sidecar an operator actually has
running, and this list is run at every release against it — the rule's second row, the one people skip.

Everything here is decided by the machine rather than by the code: which execution provider the
binary can actually reach, whether the model cache survived, what the running process was compiled
with, and whether the card is free.

Target: the running sidecar, as an origin — `--target http://127.0.0.1:5320`
Last verified: 2026-09-03 · http://127.0.0.1:5399 (a local run of `target/release/bge-sidecar.exe`) · all four automated items PASS; item 2 was watched FAIL against a wrong `EXPECTED_EXE_SHA256`.

| # | What a person loses if this is broken | Check | Auto |
|---|---|---|---|
| 1 | Retrieval stops entirely and the host has nothing to fall back to: no embeddings, no reranking, no search | `node -e "fetch(process.env.TARGET+'/health').then(r=>r.json()).then(h=>process.exitCode=+(h.status==='ok'?0:1))"` | auto |
| 2 | An older binary is running than the one you built — the measurements you are about to take describe a build nobody is looking at | `node -e "fetch(process.env.TARGET+'/health').then(r=>r.json()).then(h=>process.exitCode=+(h.exe_sha256===process.env.EXPECTED_EXE_SHA256?0:1))"` | auto |
| 3 | It fell back to CPU. Everything still *works*, an indexing pass takes hours instead of minutes, and nothing anywhere says the word "CPU" — the GPU-as-code-31 failure, which is invisible from inside | `node -e "fetch(process.env.TARGET+'/health').then(r=>r.json()).then(h=>process.exitCode=+(h.compiled_providers.some(p=>p!=='cpu')?0:1))"` | auto |
| 4 | The model cache did not survive, so the first real request fails minutes into a compile instead of at startup | `node -e "fetch(process.env.TARGET+'/models').then(r=>r.json()).then(m=>process.exitCode=+(m.models.length>0?0:1))"` | auto |
| 5 | The card is held by a process nobody is watching, and every pass queues behind it forever | `nvidia-smi` / Task Manager, or the family's GPU lease: confirm nothing else holds the card before a campaign starts | manual |

## What "provider" says, and what it does not

Item 3 asserts the binary was **compiled** with something other than CPU — that is what `/health`
can answer before any model is loaded. Whether the accelerated provider actually *initialises* is
`provider_ready` / `active_provider`, and both stay null until the first inference builds an engine.
Asserting them here would mean taking the GPU from a checklist, which
[`gpu-lease.md`](.claude/rules/shared/common/gpu-lease.md) says a tool does not do. So the
honest split: the checklist proves the build can, the first real pass proves it did — and the pass
already logs which provider it got.

## Running it

```bash
export EXPECTED_EXE_SHA256=$(sha256sum target/release/bge-sidecar.exe | cut -d' ' -f1)
node .claude/rules/shared/tools/post-deploy-check.mjs --target http://127.0.0.1:5320
```

Item 2 fails without that variable, deliberately: a check that cannot say **which** build you meant
can only confirm that *a* sidecar is running.
