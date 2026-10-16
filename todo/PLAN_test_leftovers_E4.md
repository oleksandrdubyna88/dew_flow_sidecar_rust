# PLAN — Test leftovers, E4: a `Drop`-owned, claimed, swept scratch folder for this crate's tests

> Status: **plan only, nothing implemented yet, 2026-10-16; blocked on the E1 release of `dew_flow_conventions`.**
> Stories of epic E4 of `dew_flow_conventions · todo/PLAN_test_leftovers_rollout.md` (§ *Epics*, E4). Scope: a new
> test-only module `src/test_scratch.rs`, the nine `std::env::temp_dir()` sites and their callers, a guard test
> `tests/test_leftovers.rs`, one CI step, one dev-dependency block in `Cargo.toml`, and the submodule pin.
>
> Split by Opus, 2026-10-16. Not a cornerstone. Models per story below.

**Committed together with this document** (one commit, the shared `common/planning-docs.md` § *A plan split into
epics and stories*): its row in `todo/README.md` under *Currently open* —

```
| [PLAN_test_leftovers_E4.md](PLAN_test_leftovers_E4.md) | plan only, blocked on the E1 release | test scratch folders: `TestScratch` (claim, `Drop`, start-of-run sweep), the nine `temp_dir()` sites, the guard |
```

**Parent and contract are cited as paths, never links** (the shared `common/planning-docs.md` § *Across
repositories*): the parent `dew_flow_conventions · todo/PLAN_test_leftovers_rollout.md`; the contract
`dew_flow_conventions · common/test-leftovers.md` (mounted here at `.claude/rules/shared/common/test-leftovers.md`
once S1 moves the pin); the Node reference helper `dew_flow_conventions · tools/lib/owned.mjs`; E1's stories
`dew_flow_conventions · todo/PLAN_test_leftovers_E1.md`. *Rule §n* below is a section of that contract.

## Goal

Every folder a test of this crate creates is a `TestScratch` value named `testsToDelete-sidecar-<what>-<id>`,
claimed by a sibling `<folder>.owner`, removed by its `Drop` on success AND on the unwind of a failed assertion,
never panicking on the way out, and — when a run was killed and `Drop` never ran — collected by the next run's
start-of-run sweep on owner liveness. A guard test keeps it so.

**Done when:** the pin carries the E1 contract (S1); the helper exists with its unwind, kill, open-handle and
sweep proofs (S2); no test leaks a folder on any path, and the 15 tests that leak on EVERY run today stop doing so
(S3); the guard is red on every class of rule §7 that can occur here, on an unregistered token and on a scan of
nothing (S4).

## Facts the split rests on (read on `origin/main` 1c1282e, 2026-10-16)

All nine sites of the parent are where it says they are. What the parent did not say, and every story below is
bound by:

| Site | Old name | Cleanup today | Class | Callers |
|---|---|---|---|---|
| `src/compile_cache.rs:372` | `mxr-race-<pid>` | `:421`, last statement | skipped by a panic | 1 test |
| `src/compile_cache.rs:429` | `mxr-wipe-<pid>` | `:453`, last statement | skipped by a panic | 1 test |
| `src/compile_cache.rs:498` | `mxr-scope-<pid>` | `:535`, last statement | skipped by a panic | 1 test |
| `src/log_segments.rs:204` (fn `scratch`, `:203-207`) | `bge-log-<pid>-<name>` | `:205` **before** use, never after | leaks on EVERY run | 5 direct (`:211`, `:230`, `:259`, `:276`, `:364`) + 6 through `with_days` (`:402-410`; callers `:299`, `:309`, `:323`, `:332`, `:344`, `:377`) = **11 tests** |
| `src/provider.rs:792` | `bge-preflight-<pid>` | `:804`, last statement | skipped by a panic | 1 test — **runs only with `--features cuda`** (`:789-791` returns early otherwise); CI builds cpu and dml only (`.github/workflows/ci.yml`, `build-test` matrix), so CI compiles it and never executes it |
| `src/testing.rs:167` (fn `seed_scratch`, `:166-173`) | `bge-seed-<name>-<pid>` | `:168` **before** use, never after | leaks on EVERY run | 4 tests (`src/preflight.rs:284`, `:302`, `:320`, `:337`) |
| `src/testing.rs:195` (fn `model_cache_with_a_tokenizer`, `:194-203`) | `bge-reg-<tag>-<pid>` | in the CALLERS: `src/handlers.rs:412` and `src/introspection.rs:893` last statement; `src/tokens.rs:337` mid-test, `.expect("take the file away")` — the removal IS that test's subject, not its cleanup | skipped by a panic | 3 tests |
| `src/tokens.rs:248` | `bge-tok-<pid>` | `:259`, last statement | skipped by a panic | 1 test |
| `src/tokens.rs:264` | `bge-tok-empty-<pid>` | `:279`, last statement | skipped by a panic | 1 test |

- **"The model cache the audit named" is not a tenth site.** It is `testing.rs:195`, `model_cache_with_a_tokenizer` —
  a fixture model cache in BGE-M3's snapshot layout. The real model cache (`MODEL_CACHE_DIR`, default
  `.model-cache`, `src/config.rs:85`, git-ignored) is never touched by a test: `src/testing.rs:141` points
  `cache_dir` at `no-model-cache-in-tests`, a path that does not exist, on purpose. A search of `src/`, `tests/`,
  `.github/` and `http/` for `temp_dir`, `tmpdir`, `mkdtemp`, `TEMP`/`TMP`, `docker`, `worktree`, `WithName`,
  `RYUK` and upper-case `DROP`/`TRUNCATE`/`FLUSH*` found the nine sites and nothing else (the only `docker` hits are
  package names in `http/package-lock.json`). The audit row's "missing prefix 11" has no 10th or 11th site on this
  commit.
- **Seven "remove as a last statement" are six inline removals plus one fixture whose three callers remove it.**
  The fixture's removal moves INTO the fixture with this epic; the `tokens.rs:337` removal stays, because it is the
  test.
- **The two pre-clean-only helpers leak on every run, not only on failure** — 15 tests each leave a folder behind
  whenever they pass.
- **The marker must sit beside the folder, never inside** (rule §3) — and here it would break real assertions:
  `log_segments.rs:290` asserts exactly three files under the root, `:304`, `:313`, `:335` and `:351` assert the
  exact listing of day folders under it.
- **Windows open-handle precedent:** `log_segments.rs:371-400` holds a file with
  `OpenOptionsExt::share_mode(0)`; its comment (`:382-384`) records that Rust's `File::open` passes
  `FILE_SHARE_DELETE`, so an ordinary open does NOT block the delete — the first version of that test asserted
  nothing. Every open-handle scenario below uses `share_mode(0)`.
- **`Drop` precedents in this crate:** `HeldEngine` (`testing.rs:120-127`) never panics in `Drop` (`.ok()` on every
  step); `CachePathLease` (`compile_cache.rs`) is the RAII claim the Rust doctrine §6 names.
- **`testing.rs` is `#![cfg(test)]` (`:7`)** and is pulled into 11 test modules by `use crate::testing::*` (or
  `::app_state`); `main.rs:40-59` declares every module, `mod testing;` at `:55`. This is a binary crate — a test in
  `tests/` cannot reach `crate::testing`.
- **Tests and CI.** 147 `#[test]`/`#[tokio::test]` attributes in `src/` and `tests/`, none `#[ignore]`. CI
  (`ci.yml`): `build-test` runs `cargo test --release --locked` with `--no-default-features` on ubuntu (cpu) and the
  default `dml` feature on windows; neither leg fetches `.claude/rules/shared` — only `contract` and `plans` do.
  Required checks (`.github/branch-protection.json`): `ubuntu-latest (cpu)`, `windows-latest (dml)`, `contract`,
  `plans`, `CodeQL`, `pr · semantic title`, `workflows · actionlint`. `vendor-fastembed/` is a `[patch.crates-io]`
  path dependency, not a workspace member, so its own tests (e.g. `vendor-fastembed/tests/text-embeddings.rs:484`)
  never run here, and its bytes stay upstream's (`ci.yml`, the format-check comment).
- **Repository-text tests have a precedent:** `tests/local_cpu_budget.rs` reads `CLAUDE.md`/`README.md` through
  `CARGO_MANIFEST_DIR` and fails when its scan finds nothing (`:86-90`).
- **The pin.** `.claude/rules/shared` is at `e29883f` (branch `release`), whose `common/test-leftovers.md` is the
  pre-E1 text: no owner, no start-of-run sweep, no `## Tokens` table. The `plans` job's *Pin check* requires the pin
  to equal the release tip, and `plans` is required — so the moment E1 is released, every PR here is blocked until
  the pin moves (S1).
- **Crates already in `Cargo.lock`**: `windows 0.62.2` (a `cfg(windows)` dependency, features
  `Win32_Graphics_Dxgi`, `Win32_Foundation`), `regex 1.13.1`, `tempfile 3.27.0`, `libc 0.2.186`, `getrandom`
  (three versions).

## Decisions

**D1 — Where the helper lives.** `src/test_scratch.rs`, `#![cfg(test)]`, declared in `main.rs` beside `mod testing;`
and re-exported from `testing.rs` (`pub(crate) use crate::test_scratch::TestScratch;`) so the existing
`use crate::testing::*` imports reach it. *Loses:* the parent's "in `src/testing.rs`" — the helper, the owner, the
liveness probe and the sweep are ~250 lines of a second concern, and the Rust doctrine §7 says a file splits when it
grows one. Recorded as a deviation from the parent.

**D2 — The name.** `testsToDelete-sidecar-<what>-<id>`; `PREFIX` and `REPO_TOKEN = "sidecar"` are written ONCE, in
`test_scratch.rs` (rule §1; `sidecar` is a row of the contract's `## Tokens` table). `<what>`: non-empty, no `/`,
`\`, NUL or control character, not starting with `-`, spaces and non-ASCII allowed; the whole name ≤ 63, so
`<what>` ≤ 34. `<id>`: 6 characters of `[A-Za-z0-9]` from a `std::collections::hash_map::RandomState` hasher fed a
per-process counter and the clock (std seeds `RandomState` from the OS); the folder is made with `create_dir` (not
`create_dir_all`) and an `AlreadyExists` draws a new id, at most 16 times — `mkdtemp`'s atomicity without a new
crate. *Loses:* `tempfile` as a dev-dependency (in the lock already, but its `TempDir` removes once, swallows the
error and writes no claim — wrapping it buys only the name); the old `<pid>` suffix (unique per process, not per
test — two tests sharing a name was only avoided by hand-picked `<name>`s).

**D3 — The owner, per OS, never spawning** (rule §3; `owned.mjs` is the shape).

| | Linux (CI cpu, WSL) | Windows (CI dml, the default flavour) | anything else |
|---|---|---|---|
| host | `/proc/sys/kernel/hostname` | `GetComputerNameExW(ComputerNameDnsHostname)` | the helper refuses to create (panics naming the OS): this crate is built for Windows and Linux only |
| `<os>` | `linux` | `windows` | — |
| start (integer Unix seconds UTC) | `/proc/<pid>/stat` field 22 (parsed after the LAST `)`) ÷ 100 + `btime` from `/proc/stat`, floored — `owned.mjs` `startFromProcStat` | `GetProcessTimes` creation `FILETIME` → `(ft − 116444736000000000) / 10⁷` | — |
| liveness probe | `/proc/<pid>` exists → alive (+ start); absent while `/proc/self` exists → **gone**; `/proc/self` absent → unknown | `OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION)`: `ERROR_INVALID_PARAMETER` → **gone**; any other failure (access denied) → unknown; opened → `GetExitCodeProcess` ≠ `STILL_ACTIVE` → **gone**, else alive (+ start); the handle is always closed | — |

- The host string is sanitised like `owned.mjs` (`[@:\s]+` → `-`). *Loses:* `%COMPUTERNAME%` — the NetBIOS name,
  upper-case and cut at 15 characters, while the janitor (`tools/janitor.sh:33`, `$(hostname)`) and Node
  (`os.hostname()`) use the DNS host name; a machine name over 15 characters would make every marker this crate
  writes read as a foreign host (`unknown`) to them.
- The Windows calls need three features of the `windows` crate already in the lock. They go into a
  `[target.'cfg(windows)'.dev-dependencies]` block — `windows = { version = "0.62.2", features =
  ["Win32_Foundation", "Win32_System_Threading", "Win32_System_SystemInformation"] }` — with a comment saying why
  (doctrine §8), so the product's own feature set is unchanged in a release build (resolver 2 keeps dev-dependency
  features out of non-test builds). The coder confirms each function and constant name against the 0.62.2 sources
  before writing, and records them in the PR. *Loses:* the two-field form with `unknown` liveness on Windows — safe,
  but then the sweep could never reclaim a claimed folder on the flavour this crate builds by default; and `sysinfo`
  (a new crate).
- A pid that turns out to be a THREAD id on Linux (`/proc/<tid>` resolves) or a zombie reads alive → `live` or
  `unknown`: preserved, the safe direction. A Windows process that exited with code 259 reads alive: preserved.
- `owner_state` is rule §3's order exactly, as `owned.mjs` `ownerState`: unmarked/unreadable → `unknown`; foreign
  `host@os` (case-insensitive) → `unknown`; `Gone` → `gone` (the only `gone`); probe `Unknown` → `unknown`; alive and
  a start missing on either side → `live`; |Δstart| ≤ 2 s → `live`; else → `unknown`.

**D4 — Creation and teardown** (rule §2). `TestScratch::new_in(parent, what)`: validate `what` (a panic before
anything exists) → `create_dir` → construct the `TestScratch` value at once (its `Drop` now owns the folder: that is
"register the cleanup as the very next statement") → write `<folder>.owner` with `create_new(true)`; a claim that
cannot be written panics, and the unwind drops the value, which removes the folder — "a claim that cannot be written
is a resource the run does not create". `Drop` calls `teardown(&path, &mut std::io::stderr())`:

1. `remove_dir_all` the folder, up to **5 tries, 100 ms apart** (the family's figure); `NotFound` is success (an
   already-gone path is swallowed — `tokens.rs:337` removes its folder on purpose);
2. only when the folder is gone, remove `<folder>.owner` (same retries, `NotFound` success);
3. on giving up: return, after writing ONE line `test_scratch: could not remove <path> (<error>) — left for the
   start-of-run sweep` — through `std::io::stderr()` directly, because libtest captures `eprintln!` and shows it only
   for a FAILED test, while a teardown that gives up usually does so in a passing one;
4. never panic, never `unwrap` — a panic in `Drop` during unwinding aborts the process (rule §2).

The folder goes first and its claim last, so a leftover is always prefixed AND claimed. `TestScratch` is the FIRST
binding of every test that uses one, so it drops LAST, after every handle into it (`DaySegments`, `CachePathLease`,
an open file) — locals drop in reverse order of declaration.

**D5 — The start-of-run sweep** (rule §4). There is no `globalSetup` in libtest, so the sweep runs from "a fixture's
constructor", which rule §4 names: the first `TestScratch` constructor in a test process calls it through a
`std::sync::Once`, before creating its own folder. It lists `temp_root()` ONCE and, per entry whose name matches
`^teststodelete[-_]sidecar[-_]` case-insensitively:

- a folder holding a `.git` FILE is never removed (this crate makes no worktrees; one is some clone's — rule §4);
- age = now − mtime; younger than **10 minutes** → not considered, silently;
- claim → `owner_state`: `gone` → removed with its marker; `live` / `unknown` → kept, one line naming it and the
  reason; an UNMARKED folder of the new form, old enough → removed (rule §4: the helper claims within milliseconds
  or does not create);
- a `.owner` file whose folder is absent and which is old enough → removed;
- anything else in the temp root — another token, the old `mxr-*`/`bge-*` names — is not looked at.

The sweep is `sweep(root, now, min_age, probe) -> SweepReport`, pure apart from the file system it is handed, so
every decision is testable on a private root with an injected clock and probe. The `Once` body wraps it in
`std::panic::catch_unwind`: a panic inside `call_once` would poison the `Once` and fail every later test, and a sweep
never fails the run (rule §4) — a caught panic or an I/O error is one reported line. Report lines go to
`std::io::stderr()` directly (D4.3); a sweep that found nothing prints nothing.

**D6 — The old names are not the sweep's business.** None of `mxr-race-`, `mxr-wipe-`, `mxr-scope-`, `bge-log-`,
`bge-preflight-`, `bge-seed-`, `bge-reg-`, `bge-tok-`, `bge-tok-empty-` ever carried the prefix, so (as E1's D5
decided for its own tools) the sweep gets no old-prefix branch and no end date. S3 LISTS the old-form leftovers on the
dev machine before and after and reports them; they are removed only on the owner's word, by hand.

**D7 — The guard is a repository-text test** (rule §7), `tests/test_leftovers.rs`, in the shape of
`tests/local_cpu_budget.rs`: it reads files through `CARGO_MANIFEST_DIR`, so it runs on both CI legs with no new
job. It uses `regex` as a dev-dependency (already in the lock at 1.13.1; `Cargo.lock` gains only the edge, committed
with it). It scans whole files, not "only under `#[cfg(test)]`" — the product has zero temp, container or store
sites today, and a textual `cfg(test)` boundary is a parser nobody needs; a future product site takes an inline
exemption. Classes with no site in this repository (`docker-run`, `docker-name`, `shared-store`, `ryuk-off`) refuse
ANY occurrence — stricter than the rule; the first container or store here extends the guard to rule §5/§6's full
shape in the same change, and the cure sentence says so.

**Growth surface** (the shared planning rule § *A plan that creates something that GROWS*). Scratch folders: 24
tests use one after S3 (11 log, 4 seed, 3 model cache, 3 compile cache, 2 tokens, 1 cuda-only) plus S2's own ~15;
the largest is `compile_cache.rs:498`'s at 7 MB (`write_mb` 1 + 1 + 3 + 2), the run's peak is about 10 MB at
`--test-threads=8`, and a marker is ~40 bytes. Retired by `Drop` on every path that unwinds; by the next run's sweep
after a kill (10 minutes, owner `gone`); by the janitor (`bash .claude/rules/shared/tools/janitor.sh`, one hour)
when this crate's tests do not run again. A teardown that gave up leaves the folder prefixed and claimed by a pid that
dies with the run, so the next sweep collects it.

## Order

| Story | Model (spec · code · review) | Depends on | Lands |
|---|---|---|---|
| S1 — the pin bump to the E1 release | Opus (this section) · main session · Opus | E1 S5 released | first; every other merge waits for it (the required `plans` check) |
| S2 — `TestScratch`: name, claim, `Drop`, liveness, sweep | Opus · Sonnet · Opus | the E1 contract text (not the pin) | after S1 |
| S3 — the nine sites and their callers move to `TestScratch` | Opus · Sonnet · Opus | S2 | after S2 |
| S4 — the guard test, its break-its, the CI submodule step | Opus · Sonnet · Opus | S1 (the mounted `## Tokens` table), S2 (the constant) | written in parallel with S3, merged after it (red until then) |

S2 can be coded before E1 is released — it encodes the contract text, which E1's branch has fixed — but it merges
only after S1. Where this document and the contract differ, the contract wins and the difference is a defect of this
document: report it to the parent, do not resolve it in code.

### Boundary with E1 (both sides — the shared planning rule § *A boundary between two plans*)

| Item | Built by | The other's part |
|---|---|---|
| The contract text, the `## Tokens` table (`sidecar` row), `tools/janitor.sh` | E1 (`dew_flow_conventions`) | E4 reads them through the mount; never edits them here |
| The Node reference helper `tools/lib/owned.mjs` | E1 | E4 mirrors its shape and order, not its code |
| The Rust helper, the start-of-run sweep, the Rust guard, this repository's CI step | E4 | E1 names E4 as the consumer that does it |
| The pin bump | E4 (S1) | E1 S5 releases; consumers bump inside their own epic |

Disjoint otherwise: E4 changes nothing in `dew_flow_conventions`. A contract finding from this epic goes back to the
parent as a note, not into the mounted copy.

## S1 — The pin bump to the E1 release

- **Models.** This section is the brief; the main session makes the commit; Opus reviews the diff.
- **Goal.** `.claude/rules/shared` points at the `release` tip that carries E1, so the mounted
  `common/test-leftovers.md` is the text S2–S4 encode and S4 can read the `## Tokens` table.
- **Files.** `.claude/rules/shared` (the gitlink); `CLAUDE.md` only if `gate-snippet-check.mjs` says the gate block
  changed (a pin move can move the snippet).
- **Change.** `git -C .claude/rules/shared fetch origin release` → `checkout <release sha>`; commit in the existing
  form: `chore(rules): pin shared conventions to release <short sha> — test leftovers: owner, start-of-run sweep,
  guard grammar, tokens table`.
- **Scenarios.**
  1. `node .claude/rules/shared/tools/pin-check.mjs` → 0 (RED first: on the old pin, after the release, it names
     `e29883f` as behind).
  2. `node .claude/rules/shared/tools/gate-snippet-check.mjs`, `plan-lifecycle.mjs`, `build-flags-check.mjs`,
     `post-deploy-check.mjs` → 0.
  3. The mounted rule carries the table and this repository's row:
     `git -C .claude/rules/shared grep -n "^| \`sidecar\` |" common/test-leftovers.md` → one line.
  4. Both cargo legs green (the pin changes no Rust; `tests/local_cpu_budget.rs` reads only `CLAUDE.md`/`README.md`).
- **Commands.** The four node checks above; `node .claude/rules/shared/tools/low-priority.mjs cargo test --release --locked -j 8 -- --test-threads=8`.
- **DoD.** Pin equals the release tip; every required check green; this document's status line still says
  *plan only* (S1 alone ships no leftover change).

## S2 — `TestScratch`: name, claim, `Drop`, liveness, start-of-run sweep

- **Models.** Brief by Opus (this section); code by a Sonnet coder; review by Opus against rule §1–§4, D1–D5 and
  `owned.mjs`.
- **Goal.** One type through which every test-created folder is named, claimed, removed on every unwinding path
  without ever panicking, and — after a kill — collected by the next process's first scratch.
- **Files.** New `src/test_scratch.rs`; `src/main.rs` (one `mod test_scratch;` line, after `mod testing;`);
  `src/testing.rs` (one re-export line); `Cargo.toml` (`[dev-dependencies]` `regex`, and the
  `[target.'cfg(windows)'.dev-dependencies]` block of D3, each with its reason); `Cargo.lock` (the `regex` edge);
  `research/architecture.md:32` (the module list gains `test_scratch`) and `:308-309` (the fixtures sentence names
  both files; the test count is re-counted, not guessed).

**The change.** `src/test_scratch.rs`, opening with a block comment in this crate's voice: what it owns, why the
claim sits beside the folder (the `log_segments.rs` listing assertions), why `Drop` never panics, why its lines bypass
libtest's capture, why the sweep runs from the first constructor.

```rust
pub(crate) const PREFIX: &str = "testsToDelete-";        // rule §1
pub(crate) const REPO_TOKEN: &str = "sidecar";           // rule §1 — the ONLY place the token is written
pub(crate) const MAX_NAME: usize = 63;                   // rule §1
pub(crate) const ID_LENGTH: usize = 6;
pub(crate) const START_TOLERANCE_S: u64 = 2;             // rule §3
pub(crate) const REMOVE_TRIES: u32 = 5;                  // rule §2, the family's figure
pub(crate) const REMOVE_BACKOFF: Duration = Duration::from_millis(100);
pub(crate) const SWEEP_MIN_AGE: Duration = Duration::from_secs(600); // rule §4

pub(crate) struct TestScratch { path: PathBuf }
impl TestScratch {
    pub(crate) fn new(what: &str) -> Self;                  // in temp_root(); the sweep runs once per process first
    pub(crate) fn new_in(parent: &Path, what: &str) -> Self; // a private parent: the tests below, never a checkout
    pub(crate) fn path(&self) -> &Path;
    pub(crate) fn slash_path(&self) -> String;              // `\` → `/`, the form log_segments' tests compose with
}
impl Drop for TestScratch { /* teardown(&self.path, &mut std::io::stderr()); — nothing else */ }

pub(crate) fn temp_root() -> PathBuf;   // the one `std::env::temp_dir()` call, under an inline
                                        // `// testsToDelete-exempt: raw-temp — the helper's own temp root; every scratch is made here`
pub(crate) fn check_what(what: &str) -> Result<(), String>;
pub(crate) fn owned_name(what: &str) -> String;
pub(crate) struct Owner { pub host: String, pub pid: u32, pub start: Option<u64> }
pub(crate) enum Claim { Unmarked, Unreadable, Owner(Owner) }
pub(crate) fn this_owner() -> &'static Owner;             // OnceLock; panics only on an OS with no owner shape (D3)
pub(crate) fn format_owner(owner: &Owner) -> String;      // `<host>@<os>:<pid>[:<start>]`
pub(crate) fn parse_owner(text: &str) -> Claim;           // one trailing \r?\n stripped; `owned.mjs` OWNER_TEXT
pub(crate) fn read_claim(dir: &Path) -> Claim;            // the sibling `<dir>.owner`: absent → Unmarked
pub(crate) enum Probe { Gone, Unknown, Alive { start: Option<u64> } }
pub(crate) fn probe(pid: u32) -> Probe;                   // D3; never spawns a process
pub(crate) enum OwnerState { Live, Gone, Unknown }
pub(crate) fn owner_state(claim: &Claim, me: &Owner, probe: impl Fn(u32) -> Probe) -> OwnerState;
pub(crate) fn remove_with_retries(path: &Path, remover: impl FnMut(&Path) -> io::Result<()>,
                                  on_failed_attempt: impl FnMut(u32)) -> io::Result<()>;
pub(crate) fn teardown(dir: &Path, out: &mut dyn Write) -> bool;  // D4; true when folder and claim are gone
pub(crate) fn sweep(root: &Path, now: SystemTime, min_age: Duration,
                    probe: impl Fn(u32) -> Probe) -> SweepReport;  // D5; SweepReport { removed, kept: Vec<(PathBuf, String)>, errors }
```

- `parse_owner` matches `^([^@:\s]+@(linux|macos|windows)):(\d+)(?::(\d+))?$` (the `regex` dev-dependency); anything
  else — `""`, `added with --lock`, `devbox:1:2`, `devbox@beos:1:2`, `devbox@linux:x:2`, `devbox@linux:1:2:3` — is
  `Unreadable`. A pid that does not fit `u32` is `Unreadable`.
- `remove_with_retries` calls `remover` at most `REMOVE_TRIES` times, sleeps `REMOVE_BACKOFF` between tries (never
  after the last), treats `ErrorKind::NotFound` as success, calls `on_failed_attempt(n)` after each failure (the test
  seam that makes the open-handle scenarios deterministic), and returns the last error.
- No `std::process::Command` in `test_scratch.rs` outside its own `#[cfg(test)] mod tests` (the child harness of
  scenarios 5–6 and the one `hostname` of scenario 13): the writer and the probe never spawn (rule §3).
- Both constructors pass through the sweep's `Once`; `new_in` with a private parent still sweeps the temp root, never
  the parent.

**Scenarios** (`#[cfg(test)] mod tests` inside `test_scratch.rs`; RED first — written against the names above before
the bodies exist, watched failing on each assertion as the module grows; every scenario that counts or lists entries
does so in a PRIVATE parent made with `new_in`, because 8 test threads create scratches in the temp root at once):

1. `a_scratch_is_named_with_the_prefix_the_token_and_a_random_id` — `owned_name("seed")` matches
   `^testsToDelete-sidecar-seed-[A-Za-z0-9]{6}$`; 1 000 calls are all distinct; `TestScratch::new("unicode пробел")`
   exists and its name starts `testsToDelete-sidecar-unicode пробел-`.
2. `a_scratch_is_claimed_beside_itself_and_never_inside` — `<dir>.owner` exists in the same parent and
   `read_claim` equals `this_owner()`; `read_dir(dir)` is empty.
3. `an_invalid_what_creates_nothing` — `""`, `"a/b"`, `"a\\b"`, `"-x"`, `"a\u{1}b"`, and a 35-character `what` each
   panic (`catch_unwind`), and the private parent stays empty; a 34-character `what` succeeds.
4. `a_scratch_is_removed_when_its_test_panics` — inside `catch_unwind`: create, record the path, `panic!`; after: the
   folder and its `.owner` are gone.
5. `a_failed_assertion_in_a_real_test_still_removes_its_scratch` — the child harness: `current_exe()` run with
   `test_scratch::tests::child_scratch --exact --ignored --nocapture --test-threads=1` and the env
   `TEST_SCRATCH_CHILD=assert` + `TEST_SCRATCH_PARENT=<private parent>`; the `#[ignore]`d `child_scratch` returns at
   once unless that env is set (so a person running `--ignored` gets no false failure), else creates
   `new_in(parent, "child")`, prints the path, and `assert!(false)`. The child exits non-zero; its stdout names a path
   that, with its `.owner`, no longer exists.
6. `a_killed_test_leaves_its_scratch_claimed_and_the_next_sweep_collects_it` — the same harness with
   `TEST_SCRATCH_CHILD=abort`: the child creates, prints, and `std::process::abort()`s (no unwind, no `Drop` — the
   SIGKILL case). The folder and `.owner` remain; the claim names the child's pid on this host. `sweep(parent, now +
   11 min, ..)` with a probe answering `Alive` for that pid keeps both with reason `live`; with `Gone` removes both.
   (Never the REAL probe on a dead child's pid: Windows hands exited pids to the next process — E1 met it
   mid-suite.)
7. `removal_retries_then_gives_up_without_panicking` — an injected remover failing twice then succeeding → 3 calls,
   `Ok`; failing always → exactly 5 calls, ≥ 400 ms elapsed, `Err`, no panic; `NotFound` on the first call → 1 call,
   `Ok`.
8. `a_scratch_held_open_stays_prefixed_and_claimed_and_says_so_once` — `#[cfg(windows)]`: outer = `new("held")`;
   inner = `new_in(outer, "inner")`; a file inside inner opened with `share_mode(0)`; `teardown(inner, &mut buf)` →
   `false`, no panic, `buf` holds exactly one line naming inner's path, inner and its `.owner` still exist; release
   the handle; `outer` drops last and takes everything. The same set-up with `remove_with_retries` and an
   `on_failed_attempt` that releases the handle after attempt 1 → `Ok` at attempt 2, inner gone (the retry is real).
9. `a_scratch_that_cannot_be_removed_stays_prefixed_and_claimed` — `#[cfg(unix)]`: inner inside outer, outer
   `chmod 0o555`; teardown → `false`, one line, inner and its `.owner` remain; the premise is ASSERTED (a process
   running as root removes it anyway — the test then fails saying so, never passes for the wrong reason); restore
   `0o755`, outer drops.
10. `an_owner_round_trips_and_every_other_text_is_unreadable` — `parse_owner(format_owner(this_owner()) + "\r\n")`
    round-trips; on Linux and Windows `this_owner()` has a start; `devbox@windows:37560` parses with no start; the six
    unreadable texts of the change list read `Unreadable`.
11. `owner_state_follows_rule_three_in_order` — injected probe: `Unmarked`/`Unreadable` → unknown; `other@<os>` →
    unknown; this host with the other OS tag → unknown; this host in other case → evaluated; `Gone` → gone;
    `Unknown` → unknown; alive with either start absent → live; Δ = 2 s → live; Δ = 3 s either way and Δ ≫ 3 s →
    unknown, never gone.
12. `the_real_probe_sees_this_process_and_not_an_impossible_pid` — `probe(std::process::id())` is `Alive` with a
    start within 2 s of `this_owner().start`; a pid no process can have (Windows: ≡ 2 mod 4, pids are multiples of
    4; Linux: `/proc/sys/kernel/pid_max` + 1 — the premise asserted before and after) → `Gone`.
13. `the_owner_names_this_machine_as_the_janitor_does` — one spawn of `hostname` (this test only), compared
    case-insensitively with `this_owner().host` up to `@`.
14. `the_sweep_removes_only_what_a_gone_owner_left` — on a private root, `now` = real now + 11 min unless stated:
    claimed by this process → kept (`live`); foreign host → kept (`unknown`); unreadable marker → kept; probe-`Gone`
    claim → folder and marker removed; unmarked and old → removed; unmarked at real `now` (young) → untouched and
    unreported; a prefixed folder holding a `.git` FILE → kept; `testsToDelete-mcp-x-abc123` and
    `bge-log-1-first` → untouched and unreported; `teststodelete_sidecar_x_abc123` and
    `TESTSTODELETE-SIDECAR-x-abc123` → considered (rule §1's match); an old orphan `testsToDelete-sidecar-y-abc123.owner`
    whose folder is absent → removed.
15. `the_sweep_never_fails_the_run` — a root that does not exist → empty report, no panic; a probe that panics →
    caught by the `Once` wrapper's `catch_unwind`, one line, and a following `TestScratch::new` still works (the
    `Once` is not poisoned).

- **Commands.** `node .claude/rules/shared/tools/low-priority.mjs cargo test --release --locked -j 8 -- test_scratch --test-threads=8`;
  the whole suite on Windows (dml) and inside WSL (`nice -n 19 cargo test --release --locked -j 8 --no-default-features --target-dir target-wsl -- --test-threads=8`);
  `cargo fmt --package bge-sidecar --check`.
- **DoD.** Every scenario observed RED, then GREEN, on Windows and on the ubuntu CI leg (scenario 8 runs on Windows
  only, 9 on Linux only); the 147 existing tests unchanged and green; the D3 Windows API names recorded in the PR;
  `architecture.md` updated; no `Command` outside the test module (review).

## S3 — The nine sites and their callers move to `TestScratch`

- **Models.** Brief by Opus; code by a Sonnet coder; review by Opus against the facts table and D4.
- **Goal.** No test of this crate leaves a folder behind on any path; the 15 every-run leaks end.
- **Files.** `src/compile_cache.rs`, `src/log_segments.rs`, `src/provider.rs`, `src/testing.rs`, `src/tokens.rs`,
  `src/handlers.rs`, `src/introspection.rs`, `src/preflight.rs`.

**The change** (each site keeps its meaning; `<what>` is the old prefix without its `bge-`/trailing `-`; the scratch
is the FIRST binding of its test, D4):

| Site | Becomes |
|---|---|
| `compile_cache.rs:372`, `:429`, `:498` | `let scratch = TestScratch::new("mxr-race")` / `"mxr-wipe"` / `"mxr-scope"`; `base` from `scratch.path()` as today; the `remove_dir_all` at `:421`, `:453`, `:535` goes |
| `log_segments.rs:203-207` `scratch(name)` | returns `TestScratch::new(&format!("log-{name}"))`; each of the 5 direct callers binds `let scratch = scratch("…"); let root = scratch.slash_path();`; `with_days` (`:402-410`) returns the `TestScratch` (the root string from it), and its 6 callers bind it first |
| `provider.rs:792` | `let scratch = TestScratch::new("preflight")`; `:804` goes. Executed only in a `--features cuda` build — compile-checked on both CI legs, run by hand on an NVIDIA machine if one is available, and the PR says which |
| `testing.rs:166-173` `seed_scratch` | returns `SeedScratch { scratch: TestScratch, from: PathBuf, to: PathBuf }` (`what` = `seed-{name}`); the pre-clean at `:168` goes (a fresh name has nothing to clean); the 4 callers in `preflight.rs` keep the VALUE (`let seed = seed_scratch("full"); … seed.from …`) |
| `testing.rs:194-203` `model_cache_with_a_tokenizer` | returns a `TestScratch` (`what` = `reg-{tag}`) holding the snapshot layout; callers use `cache.path()`; the removals at `handlers.rs:412` and `introspection.rs:893` go; `tokens.rs:337` STAYS (it is the test) and `Drop` then finds the folder gone, which is success |
| `tokens.rs:248`, `:264` | `TestScratch::new("tok")`, `TestScratch::new("tok-empty")`; `:259`, `:279` go |

**The one trap** the review checks at every call site: `let _ = …` and a destructuring that leaves the scratch
unbound (`let (_, from, to) = …`, `let SeedScratch { from, to, .. } = …`) drop the `TestScratch` at the end of the
STATEMENT — the folder is deleted before the test uses it. A scratch is always held by a named binding (`scratch`,
`_scratch`, `seed`, `cache`). S4's guard refuses the `let _ =` shapes textually.

**Scenarios** (RED first — scenarios 1–3 are written against the OLD shapes on `main`, run, and their failure
messages recorded; then the same assertions are carried to the new shapes; this is the rule's "fix landed after the
test" route only in that the shape changes, never the assertion):

1. `a_seed_scratch_is_gone_once_its_test_ends` (in `preflight.rs`'s tests) — `let base = { let seed =
   seed_scratch("gone"); <its base> }; assert!(!base.exists())`. RED on `main`: `bge-seed-gone-<pid>` still exists.
2. `a_log_scratch_is_gone_once_its_test_ends` (in `log_segments.rs`'s tests) — a scratch used by `DaySegments::open`
   in an inner block; after the block the root does not exist. RED on `main`: `bge-log-<pid>-gone` still exists.
3. `a_model_cache_is_gone_even_when_its_test_panics` (in `tokens.rs`'s tests) — `catch_unwind` around
   `model_cache_with_a_tokenizer("panics")` and a `panic!` before any removal; after: the root does not exist. RED on
   `main`: `bge-reg-panics-<pid>` still exists.
4. **Nothing else moved:** the 147 existing tests pass on both legs, with the same names; in particular
   `log_segments.rs`'s exact-listing tests (`:290`, `:304`, `:313`, `:335`, `:351`) and the Windows
   held-file test (`:376`) — the marker beside the root never shows up in them — and `tokens.rs:330` (the folder its
   test removes).
5. **No leftover after a full run** (by hand, recorded in the PR): before S3, list the temp root's old-form names
   (`mxr-race-*`, `mxr-wipe-*`, `mxr-scope-*`, `bge-log-*`, `bge-preflight-*`, `bge-seed-*`, `bge-reg-*`,
   `bge-tok-*`) on Windows (`%TEMP%`) and in WSL (`$TMPDIR` or `/tmp`) and record the counts; after a full run of each
   leg: zero NEW old-form names and zero `testsToDelete-sidecar-*` entries. The pre-existing old-form leftovers are
   reported to the owner and removed only on the owner's word (D6).

- **Commands.** As S2 (both legs, the whole suite); `cargo fmt --package bge-sidecar --check`.
- **DoD.** Scenarios 1–3 observed RED with the symptom above, then GREEN; 4 and 5 recorded; zero
  `std::env::temp_dir()` calls left outside `test_scratch.rs` (S4 proves it).

## S4 — The guard test, its break-its, and the CI step

- **Models.** Brief by Opus; code by a Sonnet coder; review by Opus against rule §1 and §7, D7.
- **Goal.** No test, script or workflow here creates a temp folder, worktree, container or store resource outside
  `TestScratch` (or under a valid inline exemption), the helper's token is registered, and a scan of nothing is red.
- **Files.** New `tests/test_leftovers.rs`; `.github/workflows/ci.yml` (`build-test` gains `run: git submodule update
  --init --depth 1 .claude/rules/shared` after checkout, on both legs — the `contract` job's line); `README.md` only if
  it documents test commands that need the submodule; `research/module_inference.md` (a paragraph beside the
  `tests/local_cpu_budget.rs` one at `:132-136`).

**The guard.** Pure functions over `(relative path, text)` plus one `#[test]` over the real tree.

- **Discovery** (std `read_dir`, no git spawn): `src/**/*.rs`, `tests/**/*.rs`, `.github/workflows/*.yml`,
  `.github/scripts/**/*.mjs`, `http/*.js`, `http/*.mjs` (never `http/node_modules`), `**/*.sh`, `**/*.ps1` outside
  `target*/`, `node_modules/`, `vendor-fastembed/` and `.claude/`. Excluded by name, a frozen list a test pins
  exactly: `tests/test_leftovers.rs` (it carries the violations as fixture strings). `vendor-fastembed/` is out of
  scope by construction (not built as tests here; upstream's bytes). **Scan of nothing:** fewer than one `.rs` under
  `src/`, one under `tests/`, or one workflow ⇒ red with `test_leftovers: scanned nothing under <root> — discovery
  broke; the repository is not clean`.
- **Classes** (rule §7, their Rust/YAML/JS/shell spelling). Whole-line `//` and `#` comments are skipped for matching
  and read for exemptions.
  - `raw-temp`: `\btemp_dir\s*\(`, `\benv::var(_os)?\(\s*"(TEMP|TMP|TMPDIR)"`, `\btempfile::`, `\btempdir\s*\(`,
    `\bmkdtemp(Sync)?\s*\(`, `\btmpdir\s*\(`, `process\.env\.(TEMP|TMP|TMPDIR)\b`, `GetTempPath`, and in `.sh`/`run:`
    blocks `\bmktemp\b`.
  - `raw-worktree`: `\bworktree\s+add\b`, `"worktree",\s*"add"`.
  - `docker-run`, `docker-name`: ANY `\bdocker\s+(run|create|compose)\b`, `\bdocker\s+(volume|network)\s+create\b`,
    `WithName\(` (D7: none exists here; the first one extends this class to rule §5's carve-out check).
  - `shared-store`: `\b(DROP\s+(DATABASE|SCHEMA|TABLE)|TRUNCATE|FLUSHALL|FLUSHDB)\b`, case-sensitive (prose
    "truncated" never matches).
  - `ryuk-off`: `TESTCONTAINERS_RYUK_DISABLED` — never exemptible.
  - `dropped-scratch` (this repository's own, S3's trap): `let\s+_\s*=\s*.*(TestScratch::new(_in)?|seed_scratch|model_cache_with_a_tokenizer)\s*\(`
    and `let\s*\(\s*_\s*,` before `seed_scratch(`.
- **Exemptions — inline only** (rule §7): `testsToDelete-exempt: <class> — <reason>` on the finding's line or the
  line above; the class is one of the above except `ryuk-off`; the reason ≥ 10 characters and ≥ 3 words and not a
  placeholder (`needed`, `see above`, `todo`); `shared-store` needs `lock:`. Anything else carrying
  `testsToDelete-exempt:` is a finding of class `exemption`. The one real exemption: `temp_root()` in
  `src/test_scratch.rs`.
- **The token is registered** (rule §1): exactly one `const REPO_TOKEN: &str = "<t>";` across `src/**/*.rs`, `<t>`
  matching `^[a-z0-9]{2,12}$`, present in the FIRST column (backticks stripped) of the GFM table under `## Tokens` in
  `.claude/rules/shared/common/test-leftovers.md`, whose header is exactly `| token | family | repository |`, with
  no token listed twice. The mounted file missing ⇒ red with `the shared rules are not checked out — git submodule
  update --init .claude/rules/shared` (fail closed: no table registers nothing).
- **Output:** one line per finding, `<file>:<line> — <class> — <text>`, then one cure sentence per class naming rule
  §7, then the assertion fails.

**Scenarios** (RED first: the real-tree test is written and run on the last `main` commit BEFORE S3 merged):

1. `this_repository_creates_nothing_outside_test_scratch` — RED before S3: exactly the nine `raw-temp` findings of the
   facts table (at their then-current lines), nothing else; the counts recorded in the PR. GREEN after S3: zero.
2. `every_class_is_refused_once_and_a_valid_exemption_passes` — fixture strings through the pure function: a raw
   `std::env::temp_dir()` in `src/x.rs` ⇒ 1 `raw-temp` naming the line; `env::var("TMPDIR")` ⇒ 1; `mkdtempSync(` in
   `http/x.mjs` ⇒ 1; `mktemp -d` in a workflow `run:` ⇒ 1; `git worktree add` ⇒ 1; `docker run --rm img` ⇒ 1
   (`docker-run`, D7); `docker volume create v` ⇒ 1; `TRUNCATE TABLE t` ⇒ 1, with a `shared-store` exemption carrying
   `lock:` above ⇒ 0, without `lock:` ⇒ 1 `exemption`; the word `truncated` ⇒ 0; `TESTCONTAINERS_RYUK_DISABLED=true`
   with an exemption above ⇒ still 1; a well-formed `raw-temp` exemption ⇒ 0; `testsToDelete-exempt: raw-temp —
   needed` ⇒ 1 `exemption`; `testsToDelete-exempt: nonsense — a perfectly fine reason here` ⇒ 1 `exemption`;
   `let _ = TestScratch::new("x");` ⇒ 1 `dropped-scratch`; `let (_, from, to) = seed_scratch("x");` ⇒ 1; `let
   _scratch = TestScratch::new("x");` ⇒ 0; a commented-out `// std::env::temp_dir()` ⇒ 0.
3. `a_scan_of_nothing_is_red` — the pure discovery-verdict function over in-memory listings: an empty listing, one
   with no `src/*.rs`, and one with no workflow ⇒ the "scanned nothing" failure each. No folder is made: this file is
   an integration test and cannot reach `crate::testing`'s `TestScratch`, and a second temp helper here would be the
   copy the guard exists to refuse.
4. `the_token_is_a_registered_row` — the real mounted rule: `sidecar` ⇒ no finding; fixture rule texts: no `## Tokens`
   ⇒ 1; a two-column table ⇒ 1; the token only in the `repository` column ⇒ 1; a token listed twice ⇒ 1; `Sidecar`
   and `x` ⇒ 1; two `REPO_TOKEN` constants ⇒ 1.
5. `the_exclusion_list_is_exactly_one_file` — pinned.
6. **Break-it on the real tree** (by hand, recorded in the PR): one `std::env::temp_dir()` added to a test module ⇒
   the guard red naming `file:line`; reverted ⇒ green. The same with `.claude/rules/shared` de-initialised ⇒ red with
   the submodule sentence.

- **Commands.** `node .claude/rules/shared/tools/low-priority.mjs cargo test --release --locked -j 8 --test test_leftovers -- --test-threads=8`;
  the whole suite on both legs; `actionlint` passes on the changed `ci.yml` (the `workflows` job).
- **DoD.** Scenario 1 observed red with exactly the nine sites, then green; every break-it observed red; both CI legs
  green WITH the submodule step; `module_inference.md` updated.

## Test plan

| The parent's "proven by" | Where |
|---|---|
| A panic inside the scope, then the path no longer exists | S2 sc. 4 (in-process `catch_unwind`), sc. 5 (a real failing `#[test]` through the harness). Not the parent's "two tests" — libtest runs tests in parallel and unordered, so a second test cannot observe the first |
| RED-GREEN for the pre-clean-only class | S3 sc. 1, 2; the panic-skipped class S3 sc. 3; the inline sites through S2 sc. 4 and S4 sc. 1 |
| The guard's break-it | S4 sc. 2, 6 |
| An open handle on Windows: no panic, folder gone or a prefixed leftover remains | S2 sc. 8 (`share_mode(0)`), with the POSIX twin sc. 9; the leftover is also CLAIMED, which the parent did not require and the contract does |
| (contract, not in the parent) a killed run's leftover is collected; a live owner's survives | S2 sc. 6, 14, 15 |

## Definition of Done

- [ ] The pin carries the E1 release; the mounted rule lists `sidecar` (S1).
- [ ] Every test-created folder is a `TestScratch` named `testsToDelete-sidecar-<what>-<id>`, claimed beside itself,
      removed by `Drop` on every unwinding path, never panicking; a teardown that gives up leaves it prefixed and
      claimed with one stderr line (S2, S3).
- [ ] The first scratch of a test process sweeps this token's leftovers on owner liveness and never fails the run (S2).
- [ ] The guard is red on every class, on an unregistered token and on a scan of nothing; CI fetches the rules it
      reads (S4).
- [ ] `research/architecture.md` and `research/module_inference.md` describe the helper and the guard.
- [ ] When the last story merges, this document's status becomes `E4 shipped, <date>; promotes with its parent` —
      never *implemented* while it sits in `todo/` — and the parent's E4 entry gets the same sentence. Parent and
      stories documents promote together, after E7.

## Notes for the parent (`dew_flow_conventions · todo/PLAN_test_leftovers_rollout.md`, E4 entry)

1. *"Plus the model cache the audit named"* double-counts: it is `src/testing.rs:195`, one of the nine; no tenth or
   eleventh site exists on 1c1282e (the audit row's "11").
2. The seven "remove as a last statement" are six inline plus one fixture removed by its three callers; the
   pre-clean-only pair leak in 15 tests on every run, not only on failure.
3. The entry encodes rule §2's teardown but not §2's claim or §4's start-of-run sweep; a killed run would leak until
   the janitor. S2 adds both.
4. The guard as written ("`src/**/*.rs` under `#[cfg(test)]`, refuse a `temp_dir()` outside that file") is narrower
   than rule §7 (scripts and CI, every class, exemptions, the token, a scan of nothing) and needs the mounted rule in
   CI — which `build-test` does not fetch today.
5. *"`#[should_panic]` … and, in a second test, asserts the path no longer exists"* cannot work under libtest's
   parallel, unordered execution; replaced by `catch_unwind` and a child-process harness.
6. The warning line `Drop` writes is swallowed by libtest's capture in a passing test unless written to
   `std::io::stderr()` directly; and an open handle blocks removal on Windows only with `share_mode(0)`.
7. `src/provider.rs:792` runs only in a CUDA build, which CI never makes.
8. For the contract (E1), not this repository: rule §3's "the machine name as the OS reports it" has two answers on
   Windows — the DNS host name (`hostname`, Node's `os.hostname()`, the janitor) and the NetBIOS name
   (`%COMPUTERNAME%`, and C#'s `Environment.MachineName`: upper-case, cut at 15 characters). A C# helper written to
   E3's shape with `Environment.MachineName` would write markers the janitor and Node read as a foreign host on any
   machine whose name is longer than 15 characters. One sentence naming the DNS host name would close it before E3
   writes the first C# helper.
