# PLAN — the reliability tail the 24/7 audit left open

> Status: **plan only, nothing implemented yet, 2026-08-16.** Scope: `src/main.rs` throughout.
> The CRITICAL/HIGH defects of the same audit — the blocking lock in `/unload`, the invisible
> inference wedge, the DLL hashing on the `/health` path and the silently zeroed token counts — are
> being fixed in a separate task and are **not** in this plan.
>
> Related: `.claude/rules/shared/common/reliability.md` (the doctrine this audit produced),
> [README.md](../README.md) (the incident history most of this file's machinery answers).

## Why this document exists

On 2026-08-16, the eve of the first long unattended runs, all four `dew_flow_*` repositories were
audited against one mission: **24/7 operation, no leaks, no hangs, every failure legible in the log
afterwards.** This sidecar came out of it well — the canary check, the poison-healing lock and the
startup preflights are each a real past incident converted into a guard, and there are no unbounded
caches anywhere. What follows is the remainder: one genuine correctness hazard, some avoidable hot
path cost, and the file-size violation that makes all of it harder to change.

## The symptom, per item

### 1. The MIGraphX cache path is protected across the build but not the first inference — HIGH

`src/main.rs:1680-1692`, `with_engine_cache`, serializes on a `BUILD_ENV_LOCK` while it sets the
process-global `ORT_MIGRAPHX_MODEL_CACHE_PATH` / `ORT_MIGRAPHX_CACHE_PATH` and builds the session.
But this file's own comments (`:1630-1634`, `:1899-1901`) record that the execution provider reads
and writes that cache **lazily, at the first kernel launch** — which happens *after*
`with_engine_cache` has returned and released the lock: in `canary_check` via `load_validated_dual`
(`:1876-1877`) for embed, and in `score_documents` (`:1519-1536`) for rerank.

`Engines.embed` and `Engines.rerank` are two independent mutexes, so an embed build and a rerank
build legitimately run concurrently on two `spawn_blocking` threads — the ordinary situation right
after a restart when the host hits both endpoints. If the other engine's build flips the env var in
that window, engine A compiles or reads against engine B's cache directory. That is the 2026-07-27
stale-cache incident (`:1667-1671`) whose entire fix was per-engine subdirectories, reopened through
a timing gap the lock does not cover.

**Fix, in order of preference:**

1. **Per-session options.** If the `ort` version in use exposes the MIGraphX cache path as a session
   option rather than only an env var, set it there and delete the process-global mechanism outright.
   This is the only fix that removes the hazard class rather than narrowing it.
2. **Widen the serialization** to cover the first inference: hold `BUILD_ENV_LOCK` through the
   canary's real `embed()` call (and the rerank equivalent), so no other build can flip the variable
   before the compile has read it. Costs concurrency between the two engine types during a build —
   which is minutes, once, and only when both are cold.

Feature-gated to `migraphx` builds. Verify which of the two is possible before choosing; record the
finding here either way, because the next reader will ask the same question.

### 2. `/tokenize` does CPU work — and first-call file I/O — on the async runtime — MEDIUM

`src/main.rs:1191-1216` encodes every text inline in the handler, and on the first call
`state.token_counter()` also walks a directory and loads a tokenizer from disk inside the
`OnceLock::get_or_init`. Unlike `/embed`, whose tokenization happens inside `embed_blocking` under
`spawn_blocking` (`:1178`), nothing moves this off the reactor, and unlike `/embed` there is no
`Limits::resolve` cap on how many texts or how long. The in-line comment calls it "pure CPU… safe to
call from an index pass", which holds for one short text and is not guaranteed for a large batch.

**Fix:** `spawn_blocking` for the encode, a cap on the batch matching the one `/embed` already
enforces, and — cheapest of all — pre-warm the tokenizers at startup so no request pays the load.

### 3. The ruler string is rebuilt, then cloned, on every pinned request — MEDIUM

`src/main.rs:1269-1271`, `ruler_text()`, returns `"lorem ipsum dolor sit amet ".repeat(4096)` — about
114 KB, allocated fresh at `:1426` (embed), `:1552` (rerank) and `:1830` (canary) whenever shape
pinning is on, which is the MIGraphX default. `pin_shape` (`:1336-1351`) then `.to_string()`-clones it
per padding row: at `max_batch = 64`, a single-text request can churn ~7 MB. Nothing leaks, but it is
pure allocator work in the hot path of the flavour that exists precisely to avoid expensive work.

**Fix:** `static RULER: OnceLock<String>` and hand out `&str`/`Arc<str>` clones instead of fresh
allocations.

### 4. The compiled-model cache tree is walked twice per request — LOW-MEDIUM

`src/main.rs:1635-1653`, `mxr_cache_mb`, recursively sums the directory tree, and it is called before
and after inference at `:1482`/`:1488` (embed) and `:1585`/`:1593` (rerank) to report
`compile_cache_grew_mb`. On the MIGraphX flavour that tree is multi-GB across engine subdirectories,
and on the steady-state path — no compile, which is the overwhelming majority of requests — the
answer is always zero. Real I/O, per request, to learn nothing.

**Fix:** only measure when a compile could have happened (a fresh build this call), or track bytes
written and invalidate on write. Keep the field: it earned its place during the cache incidents.

### 5. Two silent-degradation corners — LOW

- `:1483` and `:1590` — `.expect("just loaded")`. Sound today: the same `MutexGuard` that inserted
  the entry is the only handle that can evict it, and `RungCache` capacity is `max(1)`. It is a
  latent panic the day a refactor separates the insert from the `get_mut`. **Fix:** a comment
  anchoring the invariant, so the next editor meets it.
- `:1732-1745` — `record_embed_max_length` / `record_max_batch` use `let Ok(mut x) = … else { return; }`
  with no log and no `clear_poison()`, unlike the engine mutexes which `lock_healing` explicitly
  un-poisons. If either is ever poisoned, `/health` stops reporting those two fields forever with no
  diagnostic trail. **Fix:** log the degradation at minimum; heal it like the engine locks if cheap.

### 6. `queue_wait_ms` silently includes engine teardown — LOW

`remember_engine` (`:1750-1759`) drops the evicted engine synchronously while still holding the
engine mutex other requests are queued behind — the opposite of `/unload`'s own explicit "drop
OUTSIDE the lock, ort teardown takes a moment" design (`:1121-1122`). At the shipped default
`EMBED_ENGINE_CACHE_RUNGS=1`, that eviction fires on every cap change. The teardown time then lands
in the next waiter's `queue_wait_ms`, which is the field the README introduced specifically to stop
misattributing waiting.

**Fix:** hand the evicted engine to `spawn_blocking` to drop, as `/unload` does; if the attribution
still matters, give teardown its own field rather than letting it hide in the queue wait.

### 7. The implicit 2 MB body limit is invisible when it fires — LOW

The router (`:555-561`) adds no layers, so axum's default `DefaultBodyLimit` (2 MB) applies. It is a
useful accidental backstop against unbounded-body memory growth — but it is not in the README's
configuration table, and axum rejects the request **before** any handler runs, so a 413 produces
nothing at all in `bge-sidecar-*.log`. An operator raising `EMBED_MAX_LENGTH` toward 8192 with a real
batch can start hitting it with no sidecar-side explanation.

**Fix:** set it explicitly, document it beside the other env vars, and log rejections.

### 8. `src/main.rs` is 2 887 lines — LOW severity, high friction

The family's limit is 800 (`.claude/rules/shared/common/coding-style.md`), typical 200–400. The file
already marks its own seams with `// ---------- X ----------` banners, and they map onto a clean
split: `logging` (`:45-69`, `:446-488`), `config` (`:91-160`), `engine_cache` (`:162-270`), `state`
(`:272-444`), `preflight` (`:574-770`), `wire` (`:771-1020`), `handlers` (`:1021-1244`), `inference`
(`:1245-1722`), `loading` + `provider` (`:1723-1766`, `:1930-2160`), `canary` (`:1767-1897`),
`provenance` (`:2032-2110`). The ~725-line test module (`:2162-2886`) distributes into the new
modules beside what it tests, as is idiomatic in Rust.

**Deliberately last.** Done first, it would bury every fix above in an unreviewable diff. Do it as its
own commit, mechanically, with no behaviour change — and `cargo test` green on both sides of it.

## Build order

1. **(1) the cache race** — the only correctness hazard here; investigate the per-session option
   first, because it decides whether the rest of the item exists.
2. **(2) `/tokenize`** and **(3) the ruler** — same hot path, both small.
3. **(4) cache walk**, **(6) teardown attribution**, **(7) body limit** — measurable hygiene.
4. **(5) the two comments/logs** — trivial, any time.
5. **(8) the module split** — last, alone, no behaviour change.

## Test plan

Per `.claude/rules/shared/common/testing.md`, each item starts with a RED test named for the
guarantee, observed failing for the real symptom. The existing test module (`:2162-2886`) is the
idiom to follow — fake clocks and held mutexes rather than a real GPU:

| item | test name |
|---|---|
| 1 | `a_concurrent_build_cannot_change_the_cache_path_before_the_first_launch` |
| 2 | `tokenize_refuses_a_batch_beyond_the_cap` |
| 3 | `the_ruler_is_allocated_once` |
| 4 | `the_cache_is_not_walked_when_no_build_happened` |
| 6 | `an_evicted_engine_is_dropped_outside_the_lock` |
| 7 | `a_body_beyond_the_limit_is_refused_and_logged` |

Item 8 asserts nothing new: the guarantee is that the existing suite is green before and after, and
the diff contains no behaviour change.

Note honestly which items cannot be verified on this machine: the MIGraphX path is feature-gated and
needs the AMD toolchain, so item 1's test may only be compilable, not runnable, here. Say which,
rather than implying a green run that did not happen.

## Definition of Done

- [ ] Item 1 is resolved by the per-session option or by widened serialization, and the finding about
      which was possible is recorded here.
- [ ] Every other item is implemented, or explicitly declined here with the reason recorded.
- [ ] Each implemented item has a RED-then-GREEN test, both observations quoted; items that could not
      run on this hardware are named as such.
- [ ] Every new guard carries a comment naming the incident class it prevents — the convention this
      file already follows and the reason it is defensible.
- [ ] `cargo build` and `cargo test` green; the logging contract (ANSI stdout, plain file, UTC day
      folder, level from `RUST_LOG`) still holds.
- [ ] After the split, no file exceeds 800 lines.
- [ ] On completion the plan is promoted to `research/` with its deviations recorded, and the
      *Currently open* table in [README.md](README.md) is updated in the same task.
