# PLAN — the reliability tail the 24/7 audit left open

> Status: **partially implemented, 2026-08-16 — items 2 and 3 are done; items 1, 4, 5, 6, 7, 8 remain
> open.** Scope: `src/main.rs` throughout.
> The CRITICAL/HIGH defects of the same audit — the blocking lock in `/unload`, the invisible
> inference wedge, the DLL hashing on the `/health` path and the silently zeroed token counts — were
> fixed in a separate task (2026-08-16) and are **not** in this plan. That task also took item 3 on the
> way past, because it was one line in a function it was already editing.
>
> **Re-anchored 2026-08-16 against `d0139b1` (`src/main.rs`, 3 821 lines).** Every reference below was
> re-read at that revision, not re-derived from the original audit. `a44e00e` leaves that file
> byte-identical, so the references hold there too — but item 2 shipped after the re-anchoring and moved
> the numbers again; see the note under item 8.
>
> Related: `.claude/rules/shared/common/reliability.md` (the doctrine this audit produced),
> [README.md](../README.md) (the incident history most of this file's machinery answers),
> [PLAN_tokenizer_registry.md](PLAN_tokenizer_registry.md) (item 2 is that plan's step 1 — see the
> overlap note there; the two are the same twenty lines and should land together).

## Why the line numbers moved, and what it cost

This plan was written against `34ff2ce`, where `main.rs` was 2 886 lines. The very next commit —
`2c1a853`, the CRITICAL/HIGH fixes above — added 1 253 lines to that file. **Every `src/main.rs:NNNN`
reference in the original draft pointed somewhere else within a day of being written**, and item 8's
headline number was understated by a third.

Worth stating rather than quietly correcting, because the convention that produced it is a good one: a
plan that cites `file.rs:line` is a plan whose claims can be checked. The cost is that the citations
decay against the very work the plan describes, and a stale reference is worse than a vague one — it
sends the reader confidently to the wrong function. **Re-anchor before taking an item, not after.**

Three items also changed in *substance*, not just position, and those corrections are marked
**REVISED** below. In two of them the file already contained the answer to a question this plan had
left open — which is its own lesson: the audit read the code, but not every comment in it.

## The symptom, per item

### 1. The MIGraphX cache path is protected across the build but not the first inference — HIGH

`src/main.rs:2193-2205`, `with_engine_cache`, serializes on a `BUILD_ENV_LOCK` while it sets the
process-global `ORT_MIGRAPHX_MODEL_CACHE_PATH` / `ORT_MIGRAPHX_CACHE_PATH` and builds the session.
But this file's own comments (`:2143-2147`, `:2470`, `:2496`) record that the execution provider reads
and writes that cache **lazily, at the first kernel launch** — which happens *after*
`with_engine_cache` has returned and released the lock: in `canary_check` via `load_validated_dual`
(`:2418-2447`) for embed, and in `score_documents` (`:2089-2120`) for rerank.

`Engines.embed` and `Engines.rerank` are two independent mutexes, so an embed build and a rerank
build legitimately run concurrently on two `spawn_blocking` threads — the ordinary situation right
after a restart when the host hits both endpoints. If the other engine's build flips the env var in
that window, engine A compiles or reads against engine B's cache directory. That is the 2026-07-27
stale-cache incident (`:2183`, `:2447`) whose entire fix was per-engine subdirectories, reopened
through a timing gap the lock does not cover.

**REVISED — the choice between the two fixes is already made, by a comment this plan's own audit
walked past.** `with_engine_cache`'s doc comment (`:2189-2192`) reads:

> *The path travels via process env (the only knob this ROCm build honors — **it ignores the
> provider-options fields**), so builds are serialized by a lock: two engines building at once would
> race the variable.*

That comment was present at `34ff2ce` — it predates this plan. So **fix 1 (per-session options) is
ruled out on the ROCm build in use**: somebody already tried the provider-options fields and found
them ignored. The original draft asked the reader to "verify which of the two is possible"; the answer
was in the function it was citing.

**Therefore the fix is (2): widen the serialization** to cover the first inference — hold
`BUILD_ENV_LOCK` through the canary's real `embed()` call (and the rerank equivalent), so no other
build can flip the variable before the compile has read it. Costs concurrency between the two engine
types during a build — which is minutes, once, and only when both are cold.

Keep fix 1 recorded as the preferred shape for a **future `ort` or ROCm upgrade**: it removes the
hazard class rather than narrowing it, and the reason it is unavailable is a property of this build,
not of the design. Re-test the provider-options fields whenever either is bumped, and update the
comment at `:2189-2192` — it is the only place that finding lives.

Feature-gated to `migraphx` builds.

### 2. `/tokenize` does CPU work — and first-call file I/O — on the async runtime — MEDIUM · **DONE 2026-08-16**

> Landed with [PLAN_tokenizer_registry.md](PLAN_tokenizer_registry.md) step 1, as this plan's build
> order asked — they were the same twenty lines. All three fixes shipped:
>
> - **Pre-warm.** `TokenizerRegistry::load` reads every row at startup, before the listener is up. The
>   `OnceLock`s are gone, so no request pays a directory walk or a multi-MB parse.
> - **`spawn_blocking`.** The encode runs on the blocking pool, as `/embed`'s always has.
> - **A batch cap.** `TOKENIZE_MAX_TEXTS`, default **4096** — *not* `/embed`'s `max_batch` as drafted:
>   `/embed` does not refuse at that number, it re-batches, and the host assembles `/tokenize` calls of
>   up to 512 rows on purpose. The deviation is recorded in the sibling plan.
>
> Tests: `a_tokenizer_present_at_startup_still_counts_after_its_file_is_gone` (RED first — *"a loader
> that reads on first use instead answers None here"*) and `tokenize_refuses_a_batch_beyond_the_cap`
> (RED first — returned `Ok` having counted all 1025 texts inline).

*(Original symptom, for the record: the handler encoded every text inline on the async runtime, and the
first call also loaded a tokenizer from disk inside an `OnceLock::get_or_init`. Its own comment called
it "pure CPU… safe to call from an index pass" — true of the card, never of the reactor — and nothing
bounded how many texts one call could carry.)*

### 3. The ruler string is rebuilt, then cloned, on every pinned request — MEDIUM · **DONE 2026-08-16**

> Landed with the CRITICAL/HIGH reliability fixes: `ruler_text()` (`:1763`) now returns `&'static str`
> from a `static RULER: OnceLock<String>`, and the three call sites pass it straight through. Test:
> `the_ruler_is_allocated_once_and_shared` (pointer identity, so a future `String` return goes red).
> `pin_shape` (`:1831-1850`) still clones per padding row — that is the layout's own contract, and the
> allocation it now clones from is one, not one per request.

*(Original symptom, for the record: `ruler_text()` returned `"lorem ipsum dolor sit amet ".repeat(4096)`
— about 114 KB, allocated fresh at every pinned embed, rerank and canary call, then `.to_string()`-cloned
per padding row; at `max_batch = 64` a single-text request could churn ~7 MB of pure allocator work in
the hot path of the flavour that exists precisely to avoid expensive work.)*

### 4. The compiled-model cache tree is walked twice per request — LOW-MEDIUM

`src/main.rs:2148-2165`, `mxr_cache_mb`, recursively sums the directory tree, and it is called before
and after inference at `:1986`/`:1992` (embed) and `:2098`/`:2106` (rerank) to report
`compile_cache_grew_mb`. On the MIGraphX flavour that tree is multi-GB across engine subdirectories,
and on the steady-state path — no compile, which is the overwhelming majority of requests — the
answer is always zero. Real I/O, per request, to learn nothing.

**REVISED, twice — the item is narrower than drafted, and one of its two proposed fixes is
contradicted by the record.**

- **Scope: MIGraphX only.** `mxr_cache_mb` returns `0` immediately when `base` is empty (`:2162-2164`),
  and `mxr_cache_base` is empty on every non-MIGraphX flavour. On DirectML, CUDA and CPU this costs a
  `trim().is_empty()` and nothing else. The "real I/O per request" claim holds on one flavour.
- **"Only measure when a compile could have happened" is the shape that was already tried and
  abandoned.** `mxr_cache_mb`'s own doc (`:2143-2147`) says growth is measured across a **pass**, never
  across a session build, because the build-scoped measurement *"taught us nothing and lied 'served from
  cache' while the first pass then compiled for two minutes"* — the EP saves lazily, so a fresh-build
  flag does not predict when bytes land. Gating on "a fresh build this call" reintroduces exactly that
  lie.

**So the remaining route is the second one drafted:** track bytes written and invalidate on write, or
otherwise learn about growth without walking the tree. Keep the field — it earned its place during the
cache incidents, and the doc comment above is why it is shaped the way it is. If nothing cheap works,
**declining this item is a legitimate outcome**: record that the walk is the price of the only honest
signal there is.

### 5. Two silent-degradation corners — LOW

- `:1987` and `:2103` — `.expect("just loaded")`. Sound today: the same `MutexGuard` that inserted
  the entry is the only handle that can evict it, and `RungCache` capacity is `max(1)`. It is a
  latent panic the day a refactor separates the insert from the `get_mut`. **Fix:** a comment
  anchoring the invariant, so the next editor meets it. *(Still bare at both sites.)*
- `:2279-2292` — `record_embed_max_length` / `record_max_batch` use `let Ok(mut x) = … else { return; }`
  with no log and no `clear_poison()`, unlike the engine mutexes which `lock_or_refuse` (`:2227`)
  explicitly un-poisons. If either is ever poisoned, `/health` stops reporting those two fields forever
  with no diagnostic trail. **Fix:** log the degradation at minimum; heal it like the engine locks if
  cheap.
  **REVISED — half of this landed.** `record_embed_max_length` now carries the reasoning inline
  (`:2281`: *"poisoned bookkeeping: keep serving at whatever is loaded rather than failing the embed"*),
  which answers *why* it swallows. `record_max_batch` (`:2288-2290`) is still a bare `else { return; }`.
  **Neither logs, and neither heals** — the open half is the diagnostic trail, not the rationale.

### 6. `queue_wait_ms` silently includes engine teardown — LOW

`remember_engine` (`:2297-2306`) drops the evicted engine synchronously — the `Some((evicted, _))`
binding at `:2300` drops it at the end of the match arm — while its caller `embed_blocking` (`:1981`)
is still holding the engine mutex other requests are queued behind. That is the opposite of
`/unload`'s own explicit design, whose `drain_engines` doc (`:1599-1604`) states that engines are
taken under their locks and dropped **outside** them, *"preserved deliberately — holding an engine
mutex through a teardown would block the very /health that reports the handover."* At the shipped
default `EMBED_ENGINE_CACHE_RUNGS=1`, that eviction fires on every cap change. The teardown time then
lands in the next waiter's `queue_wait_ms`, which is the field the README introduced specifically to
stop misattributing waiting.

**REVISED — this violates a contract written down two hundred lines away, and the drafted fix names
the wrong axis.** `RungCache::insert`'s own doc (`:495-497`) reads:

> *Stores a freshly built engine as most-recently-used and returns whatever had to make room for it.
> **The caller drops the evicted engine** — ort session teardown is not instant and the caller already
> knows whether it is on a blocking thread.*

The API hands the evicted engine back **on purpose**, so the caller can drop it somewhere sensible.
`remember_engine` is that caller and drops it in the worst available place. So this is not an
oversight in a function nobody thought about; it is one half of a design whose other half was
documented.

The original fix — *"hand the evicted engine to `spawn_blocking` to drop, as `/unload` does"* — is
aimed at the wrong problem: `remember_engine` is **already** on a blocking thread (`embed_blocking`
runs under `spawn_blocking`, `:1666`). Nothing here sits on the reactor. **The fix is to return the
evicted engine up to a scope where the guard has been released, and drop it there** — the same
lock-shaped move `drain_engines` makes, not a thread-shaped one.

Scope: the **embed path only**. `remember_engine` has exactly one call site (`:1981`); rerank holds an
`Option<TextRerank>` slot with no rung cache and no eviction.

### 7. The implicit 2 MB body limit is invisible when it fires — LOW

The router (`:912-918`) adds no layers, so axum's default `DefaultBodyLimit` (2 MB) applies. It is a
useful accidental backstop against unbounded-body memory growth — but it is not in the README's
configuration table, and axum rejects the request **before** any handler runs, so a 413 produces
nothing at all in `bge-sidecar-*.log`. An operator raising `EMBED_MAX_LENGTH` toward 8192 with a real
batch can start hitting it with no sidecar-side explanation.

Measured from the host side (2026-08-15): 980 KB to `/tokenize` succeeds, 2.1 MB returns `413`, and the
client sees only *"an established connection was aborted by the software in your host machine"* — a
10 000-file repository died nine minutes into an indexing pass this way. The host now batches under it
(`SidecarClient.RequestByteBudget`, `dew_flow_rag_qln`), so this is no longer a live defect; it is an
afternoon that would not have been spent.

**Fix:** set it explicitly, report it in `/health` beside `max_batch` (a limit a client cannot read is
a limit it will guess at), document it beside the other env vars, and log rejections. Shares its `/health`
half with [PLAN_sidecar_product.md](PLAN_sidecar_product.md) phase 4, which records the same finding.

### 8. `src/main.rs` is 3 821 lines — LOW severity, high friction

**REVISED — the number, twice in one day.** The original draft said 2 887; the CRITICAL/HIGH fix commit
added 1 253 lines to reach **3 821**; item 2's registry then added ~308 more, so the file stands at
**4 129** against a family limit of 800 (`.claude/rules/shared/common/coding-style.md`, typical
200–400).

That is worth stating plainly rather than quietly re-typing: this plan defers the split so its fixes stay
reviewable, and every deferral makes the next fix land in a bigger file. The trade is still the right one
per fix and clearly wrong in aggregate — **the split should happen after item 1, not after item 8's turn
comes around.** Re-derive the table below when it does; do not trust these numbers, trust the groupings.

The file marks its own seams with `// ---------- X ----------` banners, though fewer than the draft
implied: seven survive (`:172`, `:931`, `:1128`, `:1446`, `:1735`, `:2270`, `:2314`). Re-derived split,
at `d0139b1`:

| module | lines | contents |
|---|---|---|
| `logging` | 44–85 | `day_and_clock` and the log-path helpers (the subscriber wiring is inline in `main`, `:791-830`, and moves with it) |
| `config` | 86–171 | model-name consts, `Config`, `env_str`/`env_parse`/`env_truthy` |
| `wedge` | 172–472 | `Phase`, `WedgePolicy`, `InFlight`, `EngineWedged`, `wedge_action`, the watchdog |
| `engine_cache` | 473–558 | `RungCache`, `EngineSlot` |
| `state` | 559–680 | `Engines`, `AppState`, `Limits` |
| `tokens` | 681–789 | tokenizer loading, `count_tokens`, `token_usage`, `usage_from_counts` — **the module [PLAN_tokenizer_registry.md](PLAN_tokenizer_registry.md) reshapes** |
| `main` | 790–930 | startup, logging init, router |
| `preflight` | 931–1127 | ORT dylib + MIGraphX cache probes, model-cache seeding |
| `wire` | 1128–1445 | request/response types, `ApiError` constructors |
| `handlers` | 1446–1734 | `/health`, `/unload`, `/embed`, `/tokenize`, `/rerank` |
| `inference` | 1735–2269 | pinning, settling, `embed_blocking`, `rerank_blocking`, `score_documents`, `mxr_cache_mb`, `with_engine_cache`, `lock_or_refuse` |
| `loading` | 2270–2313 | the bookkeeping recorders, `remember_engine`, `attention_peak_mb` |
| `canary` | 2314–2447 | reference vector, cosine, `canary_check`, `load_validated_dual` |
| `provider` | 2448–2755 | `load_dual`/`load_rerank`, provider preflight, provenance, `execution_providers` |
| tests | 2756–3821 | **1 066 lines** — distributes into the new modules beside what it tests, as is idiomatic in Rust |

`inference` at 535 lines and `provider` at 308 are the only ones that stay large; both are under the
limit and both are coherent.

**Deliberately last.** Done first, it would bury every fix above in an unreviewable diff. Do it as its
own commit, mechanically, with no behaviour change — and `cargo test` green on both sides of it. Note
that the ranges above will themselves have moved by then: re-derive from the banners, do not trust
this table's numbers, trust its groupings.

## Build order

*(Items 2 and 3 are done. Item 2 went first because it was the same twenty lines as
[PLAN_tokenizer_registry.md](PLAN_tokenizer_registry.md)'s step 1, which is on the critical path of two
other repositories — `dew_flow_rag_qln`, then `dew_flow_benchmark`.)*

1. **(1) the cache race** — the only correctness hazard left here. The investigation the draft asked
   for is already answered (see the REVISED note): implement the widened serialization. Feature-gated,
   and see the honesty clause in the test plan about what can be run on this machine.
2. **(6) teardown attribution**, **(7) body limit**, **(4) cache walk** — measurable hygiene, in that
   order: (6) is a documented contract violation, (7) is one line plus a `/health` field, and (4) may
   end in a recorded decline.
3. **(5) the two comments/logs** — trivial, any time.
4. **(8) the module split** — last, alone, no behaviour change.

## Test plan

Per `.claude/rules/shared/common/testing.md`, each item starts with a RED test named for the
guarantee, observed failing for the real symptom. The existing test module (`:2756-3821`) is the
idiom to follow — fake clocks and held mutexes rather than a real GPU:

| item | test name |
|---|---|
| 1 | `a_concurrent_build_cannot_change_the_cache_path_before_the_first_launch` |
| 2 | shipped as `tokenize_refuses_a_batch_beyond_the_cap` **+** `a_tokenizer_present_at_startup_still_counts_after_its_file_is_gone` — the second one is the pre-warm half, and it is the only way to assert "loaded at startup" without watching syscalls: a tokenizer that was on disk when the process started keeps counting after the file is taken away |
| 3 | ~~`the_ruler_is_allocated_once`~~ → shipped as `the_ruler_is_allocated_once_and_shared` |
| 4 | `the_cache_is_not_walked_when_no_build_happened` |
| 5 | `poisoned_bookkeeping_is_logged_rather_than_silent` |
| 6 | `an_evicted_engine_is_dropped_outside_the_lock` |
| 7 | `a_body_beyond_the_limit_is_refused_and_logged` |

Item 8 asserts nothing new: the guarantee is that the existing suite is green before and after, and
the diff contains no behaviour change.

Note honestly which items cannot be verified on this machine: the MIGraphX path is feature-gated and
needs the AMD toolchain — and that toolchain currently needs an ONNX Runtime built from source with
`--use_migraphx` — so item 1's test may only be compilable, not runnable, here. Say which, rather than
implying a green run that did not happen.

## Definition of Done

- [ ] Item 1 is resolved by the widened serialization, and the recorded finding — that this ROCm build
      ignores the provider-options fields, so the per-session option is unavailable — is carried into
      the code comment at `:2189-2192` so the next `ort` bump knows to re-test it.
- [ ] Every other item is implemented, or explicitly declined here with the reason recorded (item 4 is
      the likely decline; it must be a written decision, not an omission).
- [ ] Each implemented item has a RED-then-GREEN test, both observations quoted; items that could not
      run on this hardware are named as such.
- [ ] Every new guard carries a comment naming the incident class it prevents — the convention this
      file already follows and the reason it is defensible.
- [ ] `cargo build` and `cargo test` green; the logging contract (ANSI stdout, plain file, UTC day
      folder, level from `RUST_LOG`) still holds.
- [ ] After the split, no file exceeds 800 lines.
- [ ] On completion the plan is promoted to `research/` with its deviations recorded, and the
      *Currently open* table in [README.md](README.md) is updated in the same task.
