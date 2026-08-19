# PLAN — the reliability tail the 24/7 audit left open

> Status: **IMPLEMENTED, 2026-08-16.** All eight items shipped. Item 1 (the MIGraphX cache race) landed last,
> as planned, and the honesty clause below is narrower than the plan expected: the SERIALIZATION is proved by a
> test that runs on this machine (RED first), while what remains unverifiable here is the EP's own lazy read —
> a property this codebase recorded during the 2026-07-27 incident, not a claim made by the fix. Scope:
> `src/main.rs` throughout — which, by item 8, is now seventeen modules.
> The CRITICAL/HIGH defects of the same audit — the blocking lock in `/unload`, the invisible
> inference wedge, the DLL hashing on the `/health` path and the silently zeroed token counts — were
> fixed in a separate task (2026-08-16) and are **not** in this plan. That task also took item 3 on the
> way past, because it was one line in a function it was already editing.
>
> **Re-anchored twice.** First on 2026-08-16 against `d0139b1`, because the CRITICAL/HIGH fix commit had
> moved every line the original audit cited. Then again after item 8 split `main.rs` into 17 modules —
> the two remaining items now point at `compile_cache.rs`, `canary.rs`, `inference.rs` and
> `bookkeeping.rs`, because the file they were written against no longer holds that code. Re-anchoring
> the open items is part of finishing a change here, not a later tidy-up.
>
> Related: `.claude/rules/shared/common/reliability.md` (the doctrine this audit produced),
> [README.md](../README.md) (the incident history most of this file's machinery answers),
> [PLAN_tokenizer_registry.md](PLAN_tokenizer_registry.md) (**implemented** — item 2 was its
> step 1, and the two landed together as both plans asked, because they were the same twenty lines).

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

### 1. The MIGraphX cache path is protected across the build but not the first inference — HIGH · **DONE 2026-08-16**

> **The claim outlives the build now.** `with_engine_cache` — a function whose scope ENDED where the hazard
> began — is replaced by `CachePathLease` (`src/compile_cache.rs:95-169`), an RAII claim taken by the caller
> that BUILDS and held across that engine's **first pass**. The two build sites are the only two places that
> take one: `embed_natural` (`src/inference.rs:267-294`, dropped immediately after `embed_settling` returns —
> everything below that line is arithmetic) and `rerank_blocking` (`src/inference.rs:357-366`, dropped at
> return so it spans `score_documents` through both the pinned and the natural branch). A **resident** engine
> claims nothing: serializing steady-state passes across the two engine types would cost far more than the
> minutes this costs once, while one of them is cold.
>
> Three things came with it that the draft did not ask for, each because the widening made them visible:
>
> - **The lease is EVIDENCE, and the evidence is checked.** `load_dual` / `load_rerank` now require a
>   `&CachePathLease`, so a session cannot be built without one — the hazard is a compile error rather than a
>   convention. `load_session` refuses a lease taken for the OTHER engine, first thing, before a provider is
>   pinned or anything is loaded: that mismatch would point a build at the other engine's slice, which is the
>   2026-07-27 incident arriving through the guard meant to prevent it. Test:
>   `a_session_built_under_another_engines_lease_is_refused`.
> - **The canary's heal path stopped spelling the engine name by hand.** It composed
>   `engine_cache_dir(base, "dual")` as a literal and did its own `mxr_cache_base.trim().is_empty()` check;
>   both now come off the lease (`cache.dir()`, `cache.wipe()`). A name spelled twice is a name that drifts —
>   the same defect item 4 fixed on the measurement side.
> - **The recorded finding moved with the code.** The DoD asked for it, and it is now the opening paragraph of
>   `CachePathLease`'s doc: this ROCm build honors only the process env, the per-session provider-options
>   fields were tried and are IGNORED, and **they must be re-tested whenever `ort` or ROCm is bumped** —
>   because that fix removes the hazard class rather than narrowing it.
>
> **Tests, and what they can and cannot prove here.**
> `a_concurrent_build_cannot_change_the_cache_path_before_the_first_launch` — **RED first**, with exactly the
> real symptom: the embed engine's first launch read `…/mxr-race-48440/rerank`, the other engine's slice,
> against its own `…/dual`. Teeth proved by temporarily releasing the claim at the end of the build (the
> retired scope) and restoring it. The test also reproduces that retired scope inline, single-threaded, so the
> defect stays visible in the suite the way item 4's does. Plus
> `wiping_a_slice_leaves_it_present_and_empty_and_never_touches_the_other_engine`.
>
> **What is still unverified on this machine, precisely:** that the MIGraphX EP reads the variable at the
> first kernel launch rather than at session build. That is not this fix's claim — it is the codebase's
> pre-existing record, cited in three doc comments and paid for during the 2026-07-27 incident. The
> `migraphx` feature build was type-checked here (`cargo check --no-default-features --features migraphx`,
> clean); it cannot be RUN without ONNX Runtime 1.24.x built `--use_migraphx` (see
> `src/preflight.rs:184-198`). When that toolchain first works, verify this on the wire in the same session —
> `dew_flow_benchmark · todo/PLAN_compute_backend_axis.md` §6 is blocked on exactly the same prerequisite and
> should be measured then.
>
> *Deviation from the drafted fix:* the draft said "hold `BUILD_ENV_LOCK` through the canary's real `embed()`
> call". That is what happens, but not by widening a lock inside `with_engine_cache` — the lock lives in a
> function whose whole shape was the problem. Handing the claim to the caller is what makes the lifetime
> visible at the call site, and what let the compiler enforce that a build has one.

*(Original symptom, for the record, below.)*

`src/compile_cache.rs:87-99`, `with_engine_cache`, serializes on a `BUILD_ENV_LOCK` while it sets the
process-global `ORT_MIGRAPHX_MODEL_CACHE_PATH` / `ORT_MIGRAPHX_CACHE_PATH` and builds the session.
But the codebase's own comments (`compile_cache.rs:18`, `compile_cache.rs:36`, `provider.rs:35`,
`provider.rs:61`) record that the execution provider reads and writes that cache **lazily, at the first
kernel launch** — which happens *after* `with_engine_cache` has returned and released the lock: in
`canary.rs:70` `canary_check`, reached via `canary.rs:113` `load_validated_dual`, for embed; and in
`inference.rs:380` `score_documents` for rerank.

`Engines.embed` and `Engines.rerank` are two independent mutexes, so an embed build and a rerank
build legitimately run concurrently on two `spawn_blocking` threads — the ordinary situation right
after a restart when the host hits both endpoints. If the other engine's build flips the env var in
that window, engine A compiles or reads against engine B's cache directory. That is the 2026-07-27
stale-cache incident (`compile_cache.rs:80`, `provider.rs:12`) whose entire fix was per-engine
subdirectories, reopened
through a timing gap the lock does not cover.

**REVISED — the choice between the two fixes is already made, by a comment this plan's own audit
walked past.** `with_engine_cache`'s doc comment (`compile_cache.rs:83-86`) reads:

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
comment at `compile_cache.rs:83-86` — it is the only place that finding lives.

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

### 4. The compiled-model cache tree is walked twice per request — LOW-MEDIUM · **DONE 2026-08-16**

> **Resolved by scoping, not by skipping** — and it turned out to be a correctness fix as much as a cost
> one. `CompileWatch` replaces the two inline before/after pairs and measures **the engine's own cache
> subdirectory** (`EMBED_CACHE_ENGINE` / `RERANK_CACHE_ENGINE`, now constants shared with
> `with_engine_cache` so the builder and the measurement cannot drift onto different directories).
>
> `with_engine_cache` has given each engine its own subdirectory since the 2026-07-27 stale-cache
> incident; the measurement never followed and kept summing the whole tree. `Engines.embed` and
> `Engines.rerank` are independent mutexes, so **a rerank compile running during an embed pass was
> reported as that pass's growth** — on the one field that exists to say "a compile happened here".
> Measured by the test: 3 MB of the other engine's compile charged to this pass.
>
> Cost falls out of the same change: one engine's slice instead of a multi-GB tree, twice per request.
> The empty-base early return is preserved explicitly (`a_flavour_with_no_cache_never_walks_anything`),
> because `engine_cache_dir("", "dual")` would otherwise compose `/dual` — a real path to stat.
>
> **The walk itself stays, deliberately.** The draft's other option — measure only when a compile could
> have happened — is refuted by `mxr_cache_mb`'s own record: the EP saves LAZILY, at the first kernel
> launch, so a fresh-build flag does not predict when bytes land, and a build-scoped measurement claimed
> "served from cache" while the first pass then compiled for two minutes.
>
> Tests: `a_compile_by_one_engine_is_not_reported_as_growth_by_the_other` (teeth proven by reverting the
> scope: `left: 3, right: 0`) and `a_flavour_with_no_cache_never_walks_anything`.

*(Original symptom, for the record, below.)*

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

### 5. Two silent-degradation corners — LOW · **DONE 2026-08-16**

> Landed with the deduplication pass, because the two turned out to be the same edit: the poisoned-lock
> corner existed **in triplicate**, and a fix applied to three copies is a fix that gets applied to two.
>
> - The three recorders now call one `record`, which **heals** the poison (as the engine mutexes have
>   since a panicked load made every later request answer "engine poisoned") and **warns** that it did.
>   Healing is safe here for a reason the engines cannot claim: the cell is an `Option<usize>`, so a
>   panic mid-write leaves no half-built state, only a stale number the call is about to overwrite.
>   Test: `a_poisoned_bookkeeping_cell_still_records_and_says_it_was_poisoned`, **RED first** —
>   `left: None, right: Some(64)`, the write dropped and never recovered.
> - Both `.expect("just loaded")` now carry the invariant that makes them sound — the same guard
>   inserted and reads, and `RungCache` capacity is `max(1)` — so the next editor meets it *before*
>   separating the insert from the read, which is the day it would become a live panic.
>
> *Deviation:* the draft asked only for "a comment at minimum; heal it if cheap". Healing was cheap, so
> both happened, and the log line is what the item was actually about — a degradation nobody can see is
> the defect, not the degradation itself.

*(Original symptoms, for the record, below — **including two mid-flight notes the landing
superseded**: "(Still bare at both sites.)" and "REVISED — half of this landed" describe intermediate
states, not the shipped code. Verified against the source during the 2026-08-19 family audit: all
three recorders heal (`clear_poison`) AND warn through the one shared `record`, and both `.expect`
sites carry their invariant inline, exactly as the closure note above records. Kept for the history;
nothing here is open.)*

- `inference.rs:273` and `inference.rs:394` — `.expect("just loaded")`. Sound today: the same `MutexGuard` that inserted
  the entry is the only handle that can evict it, and `RungCache` capacity is `max(1)`. It is a
  latent panic the day a refactor separates the insert from the `get_mut`. **Fix:** a comment
  anchoring the invariant, so the next editor meets it. *(Still bare at both sites.)*
- `bookkeeping.rs:14-39` — `record_embed_max_length` / `record_max_batch` / `record_embed_dimension` use `let Ok(mut x) = … else { return; }`
  with no log and no `clear_poison()`, unlike the engine mutexes which `lock_or_refuse` (`:2227`)
  explicitly un-poisons. If either is ever poisoned, `/health` stops reporting those two fields forever
  with no diagnostic trail. **Fix:** log the degradation at minimum; heal it like the engine locks if
  cheap.
  **REVISED — half of this landed.** `record_embed_max_length` now carries the reasoning inline
  (`:2281`: *"poisoned bookkeeping: keep serving at whatever is loaded rather than failing the embed"*),
  which answers *why* it swallows. `record_max_batch` (`:2288-2290`) is still a bare `else { return; }`.
  **Neither logs, and neither heals** — the open half is the diagnostic trail, not the rationale.

### 6. `queue_wait_ms` silently includes engine teardown — LOW · **DONE 2026-08-16**

> `remember_engine` now hands the evicted engine to `teardown_off_the_lock`, which drops it on a thread
> of its own and logs how long that took. The teardown leaves the critical path, so it stops landing in
> the next waiter's `queue_wait_ms`; the log line exists because once it is off the lock, nothing else
> would ever say it was slow. A spawn failure drops inline — slow beats leaked — and warns that it did.
>
> Test: `an_evicted_engine_is_dropped_outside_the_lock`, which compares the thread the drop ran on
> against the caller's. **RED first**: `left: ThreadId(2), right: ThreadId(2)`.
>
> *Deviation:* the drafted fix — "hand it to `spawn_blocking`, as `/unload` does" — named the wrong axis,
> as the REVISED note below already found. What was needed is a thread that is not the lock holder's, and
> `std::thread::Builder` gives that without requiring a Tokio runtime in context, which matters because
> `remember_engine` is reachable from tests that have none.

*(Original symptom, for the record, below.)*

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

### 7. The implicit 2 MB body limit is invisible when it fires — LOW · **DONE 2026-08-16**

> Four halves, all shipped: the limit is **set explicitly** (`DefaultBodyLimit::max`, from
> `MAX_BODY_BYTES`, defaulting to the same 2 MiB so nothing changes but the fact that it is a decision);
> it is **reported** on `/health` as `max_body_bytes` beside the new `tokenize_max_texts`; rejections are
> **logged** by a `log_body_rejections` middleware that names the route and the announced size; and the
> router moved into `build_router` so a test can drive it.
>
> Tests: `a_body_beyond_the_configured_limit_is_refused` (**RED first**: `left: 200, right: 413` — a 4 KB
> body passed a configured 1 KB cap, because only axum's own 2 MB default was in force),
> `a_body_within_the_configured_limit_still_reaches_the_handler`, `health_reports_the_body_limit_it_enforces`.
>
> **Verified on the wire**, since the layer and the middleware are exactly what a unit test reaches least
> well: with `MAX_BODY_BYTES=4096`, `/health` reported `max_body_bytes: 4096`, an 8 KB POST came back
> `413`, and the log carried the line that did not exist before —
> *"/tokenize: request body refused — 8030 byte(s) announced, over MAX_BODY_BYTES."*
>
> One dev-dependency: `tower` (with `util`), for `ServiceExt::oneshot`. Already in the tree at that exact
> version via axum — no new supply chain — and a layer cannot be proven applied from below the router.

*(Original symptom, for the record, below.)*

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
half with [PLAN_sidecar_product.md](../todo/PLAN_sidecar_product.md) phase 4, which records the same finding.

### 8. `src/main.rs` was 4 744 lines — LOW severity, high friction · **DONE 2026-08-16**

> **Split into 17 modules. The largest file is now 702 lines; `main.rs` is 264.** Done as its own
> commit, mechanically, with `cargo test` green on both sides — 82 passed before, 82 passed after, and
> zero compiler warnings in both the binary and the test build.
>
> | module | lines | what it owns |
> |---|---|---|
> | `logging` | 26 | the UTC day/clock the log path is named from |
> | `config` | 135 | `Config`, the env readers, the model-name constants |
> | `wedge` | 372 | `Phase`, `WedgePolicy`, `InFlight`, `EngineWedged`, the watchdog |
> | `engine_cache` | 211 | `RungCache`, `EngineSlot` |
> | `state` | 124 | `Engines`, `AppState`, `Limits` |
> | `tokens` | 292 | the tokenizer registry and the counting path |
> | `preflight` | 297 | ORT dylib + MIGraphX cache probes, model-cache seeding |
> | `wire` | 476 | request/response types, `ApiError` |
> | `introspection` | 565 | `/health`, `/models`, `/unload` — the routes that only READ |
> | `handlers` | 391 | `/embed`, `/tokenize`, `/rerank` — the routes that COMPUTE |
> | `inference` | 702 | pinning, settling, the blocking passes, `lock_or_refuse` |
> | `compile_cache` | 208 | `CompileWatch`, `with_engine_cache`, the per-engine cache paths |
> | `bookkeeping` | 122 | the recorders, `remember_engine`, `teardown_off_the_lock` |
> | `canary` | 174 | reference vector, `cosine`, `canary_check` |
> | `provider` | 435 | engine loading, provider preflight, provenance |
> | `testing` | 193 | the shared fixtures (`app_state`, the wedge policy, the tokenizer fixture) |
> | `main` | 264 | crate docs, the module list, `main`, `build_router`, the body-limit middleware |
>
> **Deviations from the table this plan drafted.** Two modules were split again rather than shipped over
> the limit: `handlers` became `handlers` + `introspection` (compute routes against read-only routes) and
> `inference` became `inference` + `compile_cache`. The drafted `loading` module is `bookkeeping`, and
> `testing` did not exist in the draft — the 77 tests distributed to the module each one tests, as the
> plan asked, but `app_state`/`config`/the tokenizer fixture are used from six modules and a fixture
> copied per module is a fixture that drifts.
>
> **How it was verified as a pure move**, rather than asserted: every non-import statement of the original
> 4 744-line file was counted and matched against the union of the new files — **0 lost**. The test names
> were diffed the same way, which is how two tests dropped by the extraction script were caught and
> restored (`unpinning_leaves_the_width_untouched`, `models_answers_while_an_engine_is_held`; both were
> the last block in their file, and the parser was eating the closing brace). `/health`, `/models` and
> `/tokenize` were then called against the rebuilt binary and answered byte-identically.
>
> Visibility is `pub(crate)` throughout — this is a binary crate with no external API, so the alternative
> was a per-item audit that would have changed nothing observable.
>
> *(The growth record, kept because it is the argument for not deferring again: 2 887 at the audit →
> 3 821 after the CRITICAL/HIGH fixes → 4 129 after the registry → 4 744 after `/models` and items 4, 6
> and 7. Every one of those lines was added by this plan's own items, and each deferral was individually
> defensible.)*

**REVISED — the number, four times in two days.** The original draft said 2 887; the CRITICAL/HIGH fix
commit reached **3 821**; item 2's registry **4 129**; `/models` plus items 4, 6 and 7 **4 744** — against
a family limit of 800 (`.claude/rules/shared/common/coding-style.md`, typical 200–400). The file has grown
by 64 % while this plan sat open, and every one of those lines was added by this plan's own items.

That is worth stating plainly rather than quietly re-typing. The deferral is defensible per fix — a split
done first would have buried each one in an unreviewable diff — and indefensible in aggregate: the file is
now 5.9× the limit and every remaining item lands in something harder to review than when it was written.
**The split should happen next, before item 1, not when item 8's turn comes around.** The table below is
therefore already stale by construction: re-derive it from the banners, and trust its groupings rather
than its numbers.

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
| `tokens` | 681–789 | tokenizer loading, `count_tokens`, `token_usage`, `usage_from_counts` — **the module [PLAN_tokenizer_registry.md](PLAN_tokenizer_registry.md) reshaped**, so this range has already moved |
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

*(Items 2, 3, 4, 6 and 7 were done first — the verifiable set was closed deliberately ahead of the one item
believed unprovable here, and item 8's split then landed before item 1 so the last fix went into a 208-line
module rather than a 4 744-line file. Both orderings paid: the cache race turned out to be ~80 lines in one
module, and its serialization turned out to be testable here after all.)*

1. **(1) the cache race** — ✅ **done last**, as the order above intended.

## Test plan

Per `.claude/rules/shared/common/testing.md`, each item starts with a RED test named for the
guarantee, observed failing for the real symptom. The existing test module (`:2756-3821`) is the
idiom to follow — fake clocks and held mutexes rather than a real GPU:

| item | test name |
|---|---|
| 1 | shipped under the drafted name, `a_concurrent_build_cannot_change_the_cache_path_before_the_first_launch` — **RED first** (`left: …/rerank`, `right: …/dual`: the embed engine's first launch reading the other engine's slice), plus `a_session_built_under_another_engines_lease_is_refused` and `wiping_a_slice_leaves_it_present_and_empty_and_never_touches_the_other_engine`. The test-plan note below expected this one to be compile-only here; it RUNS — the serialization is `std::sync` and process env, not the EP |
| 2 | shipped as `tokenize_refuses_a_batch_beyond_the_cap` **+** `a_tokenizer_present_at_startup_still_counts_after_its_file_is_gone` — the second one is the pre-warm half, and it is the only way to assert "loaded at startup" without watching syscalls: a tokenizer that was on disk when the process started keeps counting after the file is taken away |
| 3 | ~~`the_ruler_is_allocated_once`~~ → shipped as `the_ruler_is_allocated_once_and_shared` |
| 4 | shipped as `a_compile_by_one_engine_is_not_reported_as_growth_by_the_other` + `a_flavour_with_no_cache_never_walks_anything` — the guarantee turned out to be SCOPE, not frequency: the walk stays, it just reads one engine's subdirectory |
| 5 | `poisoned_bookkeeping_is_logged_rather_than_silent` |
| 5 | shipped as `a_poisoned_bookkeeping_cell_still_records_and_says_it_was_poisoned` — RED first (`left: None, right: Some(64)`: the write was dropped and never recovered) |
| 6 | shipped as `an_evicted_engine_is_dropped_outside_the_lock` — RED first (`ThreadId(2)` against `ThreadId(2)`) |
| 7 | shipped as `a_body_beyond_the_configured_limit_is_refused` (RED first: `200` against `413`) + `a_body_within_the_configured_limit_still_reaches_the_handler` + `health_reports_the_body_limit_it_enforces`; the log half was verified on the wire, not in a test |

Item 8 asserted nothing new, as planned: the suite was green on both sides (82 before, 82 after) and
the diff carried no behaviour change — proved by counting every statement of the old file against the
union of the new ones (0 lost) and by calling the rebuilt binary's endpoints.

**The honesty clause, settled.** The draft expected item 1's test to be "compilable, not runnable" here. That
turned out to be too pessimistic by one level, and the distinction is worth keeping rather than smoothing
over: the SERIALIZATION is `std::sync` plus process environment, so it runs and it went red first. The
MIGraphX EP's own lazy read is what needs the AMD toolchain — ONNX Runtime built from source with
`--use_migraphx` — and it is a property this repository recorded during an incident, not something the fix
asserts. `cargo check --no-default-features --features migraphx` was run and is clean; that is a type check,
and it is reported as one.

## Definition of Done

- [x] Item 1 is resolved by the widened serialization, and the recorded finding — that this ROCm build
      ignores the provider-options fields, so the per-session option is unavailable — is carried into
      the code comment (now `CachePathLease`'s opening paragraph, `src/compile_cache.rs:95-102`) so the
      next `ort` bump knows to re-test it.
- [x] Every other item is implemented. Nothing was declined: item 4 — flagged in the draft as the likely
      decline — turned out to be a correctness fix rather than a cost one, and shipped.
- [x] Each implemented item has a RED-then-GREEN test, both observations quoted; the one thing that could
      not run on this hardware is named above, and it is narrower than the item.
- [x] Every new guard carries a comment naming the incident class it prevents — the convention this
      file already follows and the reason it is defensible.
- [x] `cargo build` and `cargo test` green (90 passed, 0 failed, 0 warnings, both flavours type-checked);
      the logging contract (ANSI stdout, plain file, UTC day folder, level from `RUST_LOG`) still holds.
- [x] After the split, no file exceeds 800 lines.
- [x] On completion the plan is promoted to `research/` with its deviations recorded, and the
      *Currently open* table in [README.md](../todo/README.md) is updated in the same task.
