# PLAN — Test leftovers, E4: a `Drop`-owned, claimed, swept scratch folder for this crate's tests

> Status: **IMPLEMENTED, 2026-10-31** (S2, S3 and S4 built on branch `feat/test-leftovers-e4` as the one producer PR, #64;
> the PR merges only under the merge condition of § *Order*, which this promotion does not change). Re-split against release
> `115bbf8`, 2026-10-31; plan-gate round 2 applied the same day. What differs from the text below is recorded in
> § *Deviations*, at the end.
> Stories of epic E4 of `dew_flow_conventions · todo/PLAN_test_leftovers_rollout.md` (§ *Epics*, E4), rewritten against
> the released contract (`dew_flow_conventions` branch `release` at `115bbf8`, the merged core) so a coder builds from
> this document and the mounted rule alone, never from the E1 text. Scope: a new test-only module `src/test_scratch.rs`,
> the nine `std::env::temp_dir()` sites and their callers, a guard test `tests/test_leftovers.rs`, one CI step, three
> dev-dependency lines in `Cargo.toml`. The pin bump is PR #63 and is no longer a story here.
>
> Split by Opus, 2026-10-16; re-split by Fable, 2026-10-31. Not a cornerstone. Models per story below.

**Committed together with this document** (one commit, the shared `common/planning-docs.md` § *A plan split into
epics and stories*): its row in `todo/README.md` under *Currently open* (moved to `research/README.md` on promotion) —

```
| [PLAN_test_leftovers_E4.md](PLAN_test_leftovers_E4.md) | plan only, re-split against release 115bbf8; the producer PR waits for R6 | test scratch folders: `TestScratch` (owner, claim first, `Drop`, start-of-run sweep), the nine `temp_dir()` sites, the guard |
```

**Parent and contract are cited as paths, never links** (the shared `common/planning-docs.md` § *Across
repositories*): the parent `dew_flow_conventions · todo/PLAN_test_leftovers_rollout.md`; the contract
`dew_flow_conventions · common/test-leftovers.md` at `115bbf8` (mounted here at
`.claude/rules/shared/common/test-leftovers.md` once PR #63 is in); its janitor companion
`common/test-leftovers-janitor.md`; the Node reference helper `tools/lib/owned.mjs`, sweep
`tools/lib/leftovers-sweep.mjs` and guard `tools/leftovers-check.mjs`; the conformance vectors
`tools/fixtures/owner-vectors.json`; the merged-core record `research/PLAN_test_leftovers_merged_core.md` (its *Barrier
checklist*); the remainder plan `todo/PLAN_test_leftovers_merged_core_release.md` (R6, the two-stage rule and the
owner's decisions of 2026-10-31). *Rule §n* below is a section of the contract; `O:n`, `S:n`, `G:n` are lines of the
three reference files at `115bbf8`.

## Goal

Every folder a test of this crate creates is a `TestScratch` value named `testsToDelete-<owner>-sidecar-<what>-<id>`,
claimed FIRST by a sibling `<folder>.owner` that carries the run's record, removed by its `Drop` on success AND on the
unwind of a failed assertion, never panicking on the way out, and — when a run was killed and `Drop` never ran —
collected by the next run of the SAME OS user's start-of-run sweep on two keys: scope (the name, the record and the
kernel owner agree it is the caller's) and abandonment (the owner's process is `gone`). A guard test keeps it so.

**Done when:** the helper exists with its claim-first, unwind, kill, open-handle and sweep proofs (S2); no test leaks
a folder on any path, and the 15 tests that leak on EVERY run today stop doing so (S3); the guard is red on every one
of rule §7's eight classes, on an unregistered or foreign token and on a scan of nothing (S4); and the three land as
ONE squash-merged pull request under the merge condition of § *Order*.

## Owner decisions this split encodes (2026-10-31; recorded in R6 of the remainder plan)

1. **The pin bump merges alone** — PR #63, `chore(rules): pin shared conventions to release 115bbf8`, the gitlink only
   (its CLAUDE.md gate block was unchanged). The producer stage — helper, the nine sites, guard, CI step — is ONE
   squash-merged PR; S2–S4 are its reviewable commits on one branch and never merge apart.
2. **Windows kernel owner: BOTH keys.** A folder or claim is the caller's only when its resolved path is inside the
   caller's profile AND the entry's owner SID is the caller's user SID, read with one Win32 call; either unreadable →
   fail closed (listed, never touched). Stricter than `owned.mjs` (`isMine`, O:215-228, profile only — parked there as
   F16). An elevated run's files are owned by `BUILTIN\Administrators`, so they fail closed and are kept; the helper
   says so in its one line.
3. **`owner-blind` in Rust source takes no binding tracer.** ANY literal of the bare prefix — `testsToDelete` or
   `teststodelete` followed by `-` or `_` and NOT the owner form — outside the helper's own file is a finding, so a
   constant can never carry the prefix to a sweep. Scripts (`.sh`, `.ps1`, `.mjs`/`.js`, `.py`, workflow `run:`
   blocks) keep the shipped tracer's semantics: a literal, or a variable the SAME file assigns a value holding the
   prefix. Recorded as a documented deviation from rule §7's "a bare name in script code" (D7).

## Facts the split rests on (read on `origin/main` 1c1282e, 2026-10-16; re-verified line by line 2026-10-31)

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
- **Tests and CI.** 148 test attributes in `src/` and `tests/` (135 `#[test]`, 13 `#[tokio::test]`, one of them
  `flavor = "multi_thread"`), none `#[ignore]`. CI
  (`ci.yml`): `build-test` runs `cargo test --release --locked` with `--no-default-features` on ubuntu (cpu) and the
  default `dml` feature on windows; neither leg fetches `.claude/rules/shared` — only `contract` and `plans` do.
  Required checks (`.github/branch-protection.json`): `ubuntu-latest (cpu)`, `windows-latest (dml)`, `contract`,
  `plans`, `CodeQL`, `pr · semantic title`, `workflows · actionlint`. `vendor-fastembed/` is a `[patch.crates-io]`
  path dependency, not a workspace member, so its own tests never run here, and its bytes stay upstream's.
- **Repository-text tests have a precedent:** `tests/local_cpu_budget.rs` reads `CLAUDE.md`/`README.md` through
  `CARGO_MANIFEST_DIR` and fails when its scan finds nothing (`:86-90`).
- **The pin.** `origin/main` still carries `.claude/rules/shared` at `e29883f` (the pre-E1 text); PR #63 moves it to
  `115bbf8`, whose `common/test-leftovers.md` is the contract below: the owner segment, the `## Tokens` table with the
  `sidecar` row (`:311`), the barrier. The `plans` job's *Pin check* requires the pin to equal the release tip, so
  every PR here is blocked until #63 is in.
- **Stage A (the sweep-only stage) of this repository is EMPTY, measured:** `tools/sweeper-scan.mjs` at `115bbf8` over
  this repository's 104 files lists no owner-blind sweeper (the old names `mxr-*`/`bge-*` are removed by exact name,
  never by prefix). Nothing to merge before the producer PR; the remainder plan records it so.
- **Crates already in `Cargo.lock`** (no new package, only edges): `windows 0.62.2` (a `cfg(windows)` dependency with
  `Win32_Graphics_Dxgi`, `Win32_Foundation`), `sha2 0.10.9` and `serde_json` (direct dependencies, usable in tests as
  they are), `regex 1.13.1`, `libc 0.2.186`, `tempfile 3.27.0`, `getrandom` (three versions).

## Decisions

**D1 — Where the helper lives.** `src/test_scratch.rs`, `#![cfg(test)]`, declared in `main.rs` beside `mod testing;`
and re-exported from `testing.rs` (`pub(crate) use crate::test_scratch::TestScratch;`) so the existing
`use crate::testing::*` imports reach it. *Loses:* the parent's "in `src/testing.rs`" — the helper, the owner, the
liveness probe and the sweep are ~400 lines of a second concern, and the Rust doctrine §7 says a file splits when it
grows one. Recorded as a deviation from the parent.

**D2 — The name** (rule §1). `testsToDelete-<owner>-sidecar-<what>-<id>`; `PREFIX`, `REPO_TOKEN = "sidecar"` and
`REPO_ROW = "the embedding sidecar"` (the row's *repository* column, rule §1's table at `:311`) are written ONCE, in
`test_scratch.rs`.

- `<owner>` is this process's token: `u<uid>` from the REAL uid (`libc::getuid()`); on Windows `w<digits>` — the
  account name without domain from `%USERNAME%` (absent → refuse, never fall back), its ASCII `A`–`Z` lower-cased and
  nothing else changed, the UTF-8 bytes of that name alone through SHA-256 (`sha2`, already a dependency), the first
  four bytes as a big-endian `u32` in decimal (`alice`/`Alice`/`ALICE` → `w735577801`, `Søren` → `w1810497863`, the
  vectors file's `windows-owner`). `TESTSTODELETE_OWNER`, set and non-empty, replaces it after a check against
  `^[uw][0-9]{1,10}$`; a value that fails it panics at the first use (an invalid override refuses; O:102-110). The
  token never widens what the caller may delete: the kernel owner decides (D3).
- `<what>`: non-empty, no `/`, `\`, NUL or control character, not starting with `-`, spaces and non-ASCII allowed. The
  room is DERIVED, never a constant: `63 − len("testsToDelete-") − len(owner) − 1 − len("sidecar") − 1 − 1 − 6` =
  `33 − len(owner)` — 28 under `u1000`, 22 under `w` + ten digits (O:132); a longer `what` panics before anything
  exists. The unit is BYTES (`str::len`), stricter than the rule's "characters" for non-ASCII (`unicode пробел` is 14
  characters, 20 bytes) and never laxer than any reader — a deviation, recorded here. The longest `what` S3 uses is
  `reg-cold-engine` (15).
- `<id>`: 6 characters of `[A-Za-z0-9]` from a `std::collections::hash_map::RandomState` hasher fed a per-process
  counter and the clock (std seeds `RandomState` from the OS); a taken claim or a squatted folder draws again, at most
  **3** draws (O:53), then panics naming `what` and the parent. *Loses:* `tempfile` (its `TempDir` removes once,
  swallows the error, writes no claim); the old `<pid>` suffix (unique per process, not per test).
- The sweep's own pattern is `^teststodelete[-_]<this owner>[-_]sidecar[-_]`, case-insensitive (O:147-152); a name's
  VERSION is read from its ONE second segment against the owner grammar — of it → owner form; anything else →
  **legacy** (`testsToDelete-sidecar-x-abc123`, the form before the owner segment; rule §1 `:51-54`, O:170-182) —
  never by counting separators.

**D3 — The owner record and who may delete** (rule §3). `<owner>:<host>@<os>:<pid>[:<start>[:<ns>]]`, ONE line ending
`\n`, at most 512 bytes; computed once per process (`OnceLock`), never spawning.

| | Linux (CI cpu, WSL) | Windows (CI dml, the default flavour) | anything else |
|---|---|---|---|
| host | `/proc/sys/kernel/hostname`; under WSL (`WSL_DISTRO_NAME` set, non-blank) `<host>~<distro>` — each run of `@`, `:` or whitespace in either written as ONE `-`, nothing trimmed, case kept (O:352-359, `janitor.sh:183-186`) | `GetComputerNameExW(ComputerNameDnsHostname)`, normalised alike — never `%COMPUTERNAME%` (NetBIOS, 15 characters) | the helper refuses to create (panics naming the OS): this crate is built for Windows and Linux only |
| `<os>` | `linux` | `windows` | — |
| start (integer Unix seconds UTC) | `floor(btime + field22 / CLK_TCK)` — field 22 of `/proc/self/stat` after the LAST `)`, `btime` from `/proc/stat`, `CLK_TCK` = `libc::getauxval(libc::AT_CLKTCK)` read once (0 or any read failure → NO start and NO ns; never a guessed 100; O:274-299) | `GetProcessTimes` creation `FILETIME` → `(ft − 116444736000000000) / 10⁷` (rule §3 `:117`, "a native process API") | — |
| ns | the decimal inode of `/proc/self/ns/pid` (`pid:[N]`), written ONLY together with a start (O:310-319, 377-382) | none | — |
| liveness probe | `/proc/<pid>/stat` read → alive (+ its start); `NotFound` → **gone** — unless `/proc/1` is absent (`hidepid`, O:345-347) → unknown; ANY other error (`EACCES`, `EINVAL`, a text that does not parse) → unknown | a pid above `u32::MAX` → unknown, never truncated; `OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION)`: `ERROR_INVALID_PARAMETER` → **gone**; any other failure → unknown; opened → `GetExitCodeProcess` ≠ `STILL_ACTIVE` → **gone**, else alive (+ start); the handle is always closed | — |
| kernel owner, `is_mine(path)` | `symlink_metadata(path)`: a directory or a regular file — never a link — whose `uid()` is `getuid()` | decision 2, on ONE handle: the entry opened with `OpenOptions::custom_flags(FILE_FLAG_OPEN_REPARSE_POINT \| FILE_FLAG_BACKUP_SEMANTICS)` (a junction or symlink is opened ITSELF, never followed); `GetFileInformationByHandle` carrying `FILE_ATTRIBUTE_REPARSE_POINT` → `false`; the resolved path (`canonicalize`) inside the resolved `%USERPROFILE%`, compared case-insensitively by component, AND the owner SID from `GetSecurityInfo(handle, SE_FILE_OBJECT, OWNER_SECURITY_INFORMATION)` (freed with `LocalFree`) `EqualSid` the caller's SID — read once, from the THREAD token first (`OpenThreadToken(GetCurrentThread(), TOKEN_QUERY, true)`, an impersonating thread is the caller) and from `OpenProcessToken` only on `ERROR_NO_TOKEN`, then `GetTokenInformation(TokenUser)`; any failure → `false` | — |

- **`TESTSTODELETE_RUN`** (rule §3 `:112-114`; O:426-435): set and non-empty, it is the record every claim carries —
  but only when it parses, its owner is this process's token and `owner_state` with the REAL probe says `live`;
  anything else writes one warning line and uses this process's own record (a stale value would make every claim
  dead at birth). `run_record_from(given, probe)` is the pure form; `run_record()` calls it once with the
  environment and the real probe.
- **Reading a claim** (O:470-491), `read_claim(dir, mine)` with the kernel-owner question as a seam (O:470 takes it the
  same way): `symlink_metadata` → a regular file only (a FIFO would block for ever); not `mine` → `Unreadable` (a
  stranger's claim is no evidence); at most 513 bytes read; more than 512 → `Unreadable`; one trailing `\r?\n`
  stripped; then the grammar
  `^([uw][0-9]{1,10}):([^@:(?-u:\s)]+@(linux|macos|windows)):([0-9]+)(?::([0-9]+)(?::([0-9]+))?)?$` (O:57) — written
  with `[0-9]` and `(?-u:\s)` because Rust's `\d` and `\s` are Unicode classes: `u1000:h@linux:١٢` (vector r-i25) must
  be `Unreadable`. A second line, a lone `\r`, `added with --lock`, `devbox@linux:4711:1792053744` (no owner segment)
  are `Unreadable`. A number written with a leading zero is read and marked non-canonical (O:389-405); a number that
  does not fit `u64` is read as "no process can have it".
- **`owner_state`, rule §3's order exactly** (`:142-153`; O:531-552), given the claim, the name's owner segment,
  `mine` and `Facts { hidepid }`: `Unmarked`/`Unreadable` → unknown; non-canonical → unknown; record owner ≠ caller's
  token → unknown; name owner ≠ record owner → unknown; not `mine` → unknown; `<host>@<os>` ≠ this machine's
  (case-insensitive) → unknown; on a Linux reader, the record without `ns`, with another `ns`, or the READER's own
  `ns` unreadable → unknown; pid 0 or past 2⁵³ → unknown; `@windows` with pid 1 → unknown; probe `Gone` → **gone**
  (the only gone) — unless `hidepid` → unknown; probe `Unknown` → unknown; alive with a start missing on either side →
  live; |Δstart| ≤ **2 s** → live; else → unknown, never gone.
- The Windows calls need seven features of the `windows` crate already in the lock, in a
  `[target.'cfg(windows)'.dev-dependencies]` block — `windows = { version = "0.62.2", features = ["Win32_Foundation",
  "Win32_Security", "Win32_Security_Authorization", "Win32_Storage_FileSystem", "Win32_System_Threading",
  "Win32_System_SystemInformation", "Win32_System_Diagnostics_ToolHelp"] }` — with a comment saying why (doctrine §8);
  resolver 2 keeps dev-dependency features out of a release build. Linux needs
  `[target.'cfg(unix)'.dev-dependencies] libc = "0.2"` (`getuid`, `getauxval`; in the lock at 0.2.186). The Win32
  names were confirmed in the `windows-rs` 0.62 sources on 2026-10-31 (a feature is its module path); the coder
  re-confirms each name, and `libc::AT_CLKTCK`, against the locked crates before writing and records them in the PR.
  *Loses:* `sysconf(_SC_CLK_TCK)` (on musl a constant 100 — the guess the rule forbids; glibc answers it from
  `AT_CLKTCK` anyway); a profile-only `is_mine` on Windows (decision 2); `sysinfo`.
- A pid that is a THREAD id on Linux or a zombie reads alive → `live` or `unknown`: the safe direction. A Windows
  process that exited with code 259 reads alive: preserved. An elevated Windows shell's new files are expected to be
  owned by `BUILTIN\Administrators` (`S-1-5-32-544`), not the account — UNVERIFIED (the conventions record
  `research/security/test-leftovers-merged-core.md` § *Not verified*), so S2 measures it before any helper code and
  one scenario asserts whatever was measured.

**D4 — Creation and teardown: claim → register → create** (rule §2; O:612-646). `TestScratch { path, written: Vec<u8>,
state: Claimed | Made | Released }`. `TestScratch::new_in(parent, what)`: validate `what` (a panic before anything
exists) → draw an id → write `<folder>.owner` with `create_new(true)` holding `format_owner(run_record()) + "\n"`,
kept as `written` — `AlreadyExists` is somebody else's name, draw again; any other error panics (`could not claim`)
→ construct the value at once in state `Claimed` (its `Drop` is "the cleanup registered as the very next statement")
→ `create_dir(folder)`, never `create_dir_all` (a create that succeeds on an existing folder) — success → `Made`;
`AlreadyExists` is a SQUATTER: one warning line naming it, `remove_claim_if_still(&path, &written)` removes the claim
ONLY while it still holds exactly our bytes (O:444-451; O:638 sets `made.claimed = false` for the same reason), the
value goes to `Released` and its `Drop` then touches nothing, the folder is never adopted and never removed, draw
again; any other error panics and the unwinding `Drop` of a `Claimed` value removes the claim — again only while it
still holds `written`. So nothing the helper made ever exists unclaimed, and an **unmarked** folder of the owner form
is nobody's the helper knows (D5 lists it). `Drop`: `Made` → `teardown(&path, &mut std::io::stderr())`; `Claimed` →
`remove_claim_if_still`; `Released` → nothing. Teardown is by the exact path the helper itself made and needs no
`is_mine`: the kernel-owner key is a sweep's and a claim reader's question, never a remover's of its own folder.

1. `remove_dir_all` the folder, up to **5 tries, 100 ms apart**; `NotFound` is success (`tokens.rs:337` removes its
   folder on purpose);
2. only when the folder is gone, remove `<folder>.owner` (same retries, `NotFound` success);
3. on giving up: return, after writing ONE line `test_scratch: could not remove <path> (<error>) — left for the
   start-of-run sweep` — through `std::io::stderr().lock()` as ONE `write_all`, so eight test threads never
   interleave half-lines, and directly, because libtest captures `eprintln!` and shows it only for a FAILED test,
   while a teardown that gives up usually does so in a passing one;
4. never panic, never `unwrap` — a panic in `Drop` during unwinding aborts the process (rule §2).

The folder goes first and its claim last, so a leftover is always prefixed AND claimed. `TestScratch` is the FIRST
binding of every test that uses one, so it drops LAST, after every handle into it (`DaySegments`, `CachePathLease`,
an open file) — locals drop in reverse order of declaration.

**D5 — The start-of-run sweep: scope selects, liveness authorises, uncertainty preserves** (rule §4; S). There is no
`globalSetup` in libtest, so the sweep runs from "a fixture's constructor": the first `TestScratch` constructor in a
test process calls it through a `std::sync::Once`, before claiming its own name. `sweep(root, now, min_age, probe,
mine, hooks) -> SweepReport` is pure apart from the file system it is handed; it lists `temp_root()` ONCE per pass
and runs three passes, each failure one `errors` entry and the next pass still runs (S:421-453). **A candidate is a
real directory:** an entry whose `symlink_metadata` says directory and NOT symlink (Rust reports a junction as a
symlink; S:345, `janitor.sh:538`) — a link under one of this owner's names is kept `not a directory`, never opened,
its target never touched. **An identity that cannot be read never compares equal** (S:135): a capture that fails is
a verdict of `kept`, never a match.

1. **Quarantine restores** (S:406-419). A directory `.<name>.teststodelete-quarantine-<pid>-<hex>` of this token's
   names, `mine`, whose `<pid>` the probe calls `Gone`, and older than the window by ctime AND mtime (Linux
   `MetadataExt::ctime()`; Windows `GetFileInformationByHandleEx(FileBasicInfo).ChangeTime`) → put back under
   `<name>` **exclusively, never over an existing entry** (rule §4 `:185-187`): on Linux S's placeholder method
   (S:169-191) — `create_dir(<name>)` exclusive, then `rename` over that EMPTY placeholder of our own (a plain
   `rename(2)` of a directory onto an empty directory REPLACES it, so it is never tried against a name that may be
   taken), the placeholder removed with `remove_dir` (refuses anything non-empty) when the rename fails; on Windows a
   plain `rename`, which refuses an existing directory (sc. 18 asserts that it does, so a std change would be seen).
   Restored → `notice: restored <name> …` in `listed`; the name taken → kept `quarantine-conflict` AND
   `quarantine: <q> — restore with: mv -- '<q>' '<name>'` (`Move-Item` on Windows) in `listed`; every later run
   retries it. *Loses:* `renameat2(RENAME_NOREPLACE)` — `EINVAL` on a file system without it, and a fallback is a
   second path.
2. **Folders.** Per directory entry, by the second segment of its name: not prefixed, or another token → not looked
   at; owner form with ANOTHER owner → `foreign: folder <name> — owner <o>` in `listed`, untouched; **legacy** (token
   `sidecar` straight after the prefix) → `legacy: folder <name> — remove with: rm -rf -- '<path>' '<path>.owner'`
   (`Remove-Item -Recurse -Force …` on Windows) in `listed`, never collected — rule §4 allows collecting one whose
   old-grammar record is `gone` on a non-Linux reader; this crate never wrote a legacy name (D6), so it preserves MORE
   and carries no second grammar. Own names, in this order: younger than **10 minutes** by mtime → not considered,
   silently (age SELECTS, never authorises); a `.git` FILE inside → kept `other clone` (this crate makes no
   worktrees); **capture** the identity — Linux `(dev, ino, ctime)` of the entry itself; Windows, from the handle D3
   opens, the volume serial and file index (`GetFileInformationByHandle`) and `ChangeTime` — and the claim's bytes;
   not `mine` → kept `foreign-uid` and a `foreign:` line; no claim → kept `unmarked` and an `unmarked:` line (**never
   collected, however old** — changed from E1, which removed an old unmarked folder); `Unreadable` → kept
   `unreadable`; `owner_state` `live`/`unknown` → kept with that reason (a `live` claim is kept whatever the mtime);
   **`gone` → removed by identity** (S:199-218): re-check identity and claim; atomic `rename` to the sibling
   quarantine name (same parent; a leading `.` so no prefix match reaches it) — **a rename that is refused** (a
   read-only parent on POSIX, a handle held inside on Windows) → kept `could not remove: <error>`, nothing moved;
   check again there; `remove_dir_all` with the retries; then the claim. What changed in between is renamed back
   (`replaced`) or, when its name is taken, left as `quarantine-conflict`; **a removal that gives up in quarantine is
   renamed BACK under its own name** and kept `could not remove: <error>` — a recorded deviation from S, which leaves
   it in quarantine with an `errors` entry: under its own name it stays prefixed, claimed, judged by the next run like
   any leftover and visible to the janitor — never silently dropped.
3. **Orphan claims** (S:376-404). A `<name>.owner` whose folder is absent, of this token, older than the window by the
   claim's mtime: another owner → `foreign:`; legacy → `legacy:` with `rm -f -- '<claim>'`; own → judged by its OWNER
   like any claim (rule §2 `:77-78`): `gone` removes it; `live`, `unknown`, unreadable, not `mine` keep it with the
   reason.

`SweepReport { removed: Vec<PathBuf>, kept: Vec<(PathBuf, String)>, listed: Vec<String>, errors: Vec<String> }`. The
`Once` body is `guarded(run, out)`: it wraps `run` in `std::panic::catch_unwind` — `guarded` never panics, so the
`Once` can never be poisoned (a poisoned `Once` would fail every later test) — and calls `print_report(&report, out)`
with `std::io::stderr()`. `print_report` takes `stderr().lock()` and writes each line with ONE `write_all`, so eight
test threads never interleave half-lines; one line per `kept`, `listed` and `errors` entry, nothing for an empty
report; a sweep never fails the run (rule §4 `:193`). The `read_dir` of the root itself failing is one `errors`
entry per pass.

**D6 — The old names are not the sweep's business; a legacy name is listed.** None of `mxr-race-`, `mxr-wipe-`,
`mxr-scope-`, `bge-log-`, `bge-preflight-`, `bge-seed-`, `bge-reg-`, `bge-tok-`, `bge-tok-empty-` ever carried the
prefix, so the sweep gets no old-prefix branch and no end date; S3 LISTS the old-form leftovers on the dev machine
before and after and reports them; they are removed only on the owner's word, by hand. A `testsToDelete-sidecar-…`
name (legacy, D2) can only be another producer's: listed with its command (D5), never collected.

**D7 — The guard is a repository-text test** (rule §7), `tests/test_leftovers.rs`, in the shape of
`tests/local_cpu_budget.rs`: it reads files through `CARGO_MANIFEST_DIR`, so it runs on both CI legs with no new
job. It uses `regex` as a dev-dependency (in the lock at 1.13.1; `Cargo.lock` gains only the edge). It scans whole
files, not "only under `#[cfg(test)]`" — the product has zero temp, container or store sites today, and a textual
`cfg(test)` boundary is a parser nobody needs; a future product site takes an inline exemption. **It scans its own
file too:** no file is excluded by name — its detectors and fixture strings are built from pieces (`concat!`,
`format!`) so the file holds no prefix word, no temp API name and no exemption marker whole; the frozen exclusion
list is EMPTY and a test pins it so. Workflows are read whole, line by line, with no YAML parser — stricter than the
reference guard, which isolates `run:` blocks: a `mktemp` anywhere in a workflow file is a finding. Whole-line `#`
comments are skipped in workflows and `//` lines in Rust and JS; a `#` line is NEVER skipped in `.rs` (an attribute
starts with `#`). `docker-run` refuses ANY occurrence (D7 has no site and no carve-out check: the first container here
extends the guard to rule §5's full shape in the same change, and the cure sentence says so); `docker-name` accepts
the owner form and `shared-store` the `lock:` exemption, as rule §7 and S4 say. **`owner-blind` is decision 3:** in
`.rs` any bare-prefix literal outside `src/test_scratch.rs` is a finding with no tracer and no exemption; in the
script languages this repository has (`.mjs`, `.js`, `.cjs`, `.ts`, workflow files) the finding is a literal
bare-prefix pattern, or a use (`$P`, `${P…}`, a bare bound name) of a variable the SAME file assigns a value holding
the prefix — the semantics of `tools/lib/prefix-bindings.mjs`'s header, not its implementation (a name from another
file, the environment or an argument is not traced; a prefix split over two literals binds nothing). A `.sh`, `.ps1`
or `.py` file does not exist here and is not read: the first one makes discovery red until the guard reads its
language. Deviation from rule §7 `:255`, recorded: Rust source gets the stricter rule because a Rust `const` is the
shape a tracer would miss.

**Growth surface** (the shared planning rule § *A plan that creates something that GROWS*). Scratch folders: 24
tests use one after S3 (11 log, 4 seed, 3 model cache, 3 compile cache, 2 tokens, 1 cuda-only) plus S2's own ~20;
a claim is ~60 bytes. **The peak, recomputed from the fixtures that can run at once** (verified on 1c1282e): only
the three compile-cache tests write megabytes (`write_mb`, `src/testing.rs:206`) — `mxr-scope` 7 MB
(`compile_cache.rs:499`, `:500` 1 + 1, `:505` 3, `:517` 2), `mxr-wipe` 2 MB (`:430`, `:431`), `mxr-race` none (lease
folders only) — and nothing serialises them (`CACHE_PATH_LOCK`, `compile_cache.rs:162`, serialises the lease, not
the writes). At `--test-threads=8` all three can run at once beside five other scratches of a few KB each: **≈ 9 MB**,
plus S2's sc. 7–8 child of a few KB. The upper bound if every concurrent compile-cache test were the largest
(3 × 7 MB) is ≈ 21 MB; the tests are NOT serialised for this, because 21 MB is not a budget worth a lock. Retired by
`Drop` on every path that unwinds; by the next run's sweep after a kill (10 minutes, owner `gone`, same OS user); by
the janitor — `bash .claude/rules/shared/tools/janitor.sh --token sidecar [--dry-run]`, from Git for Windows' bash on
Windows (never WSL's), as the user whose leftovers they are; 60 minutes by default (`JANITOR_MIN_AGE_MINUTES`); on
Windows a forced dry run unless `JANITOR_WINDOWS_E2E_GREEN` is set; `would run:` on stdout, `skipped:`/`foreign:`/
`legacy:`/`unmarked:` on stderr; exit 0 clean, 3 something of the caller's scope was skipped, 1 a listing failed, 2
usage (`common/test-leftovers-janitor.md:16-68`) — when this crate's tests do not run again. A teardown that gave up
leaves the folder prefixed and claimed by a pid that dies with the run, so the next sweep collects it.

## Order

| Stage | What | Lands |
|---|---|---|
| A — sweep-only | **empty** (measured, § *Facts*) | nothing to merge; recorded in R6 |
| the pin | PR #63, the gitlink to `115bbf8` — the old S1, DONE as a story here | alone, first (the required `plans` check blocks everything else until it is in) |
| B — the producer PR | S2 → S3 → S4 as reviewable commits on ONE branch, squash-merged once | after the merge condition below |

| Story | Model (spec · code · review) | Depends on |
|---|---|---|
| S2 — `TestScratch`: owner, claim first, `Drop`, liveness, sweep | Opus · Sonnet · Opus | the mounted contract (PR #63 in, or the branch rebased on it) |
| S3 — the nine sites and their callers move to `TestScratch` | Opus · Sonnet · Opus | S2 |
| S4 — the guard test, its break-its, the CI submodule step | Opus · Sonnet · Opus | S2 (the constants); written in parallel with S3, red until S3 |

**The merge condition** (rule § *The barrier*; the remainder plan R6): the producer PR may be written, reviewed and
green at any time, and **merges only when** (1) PR #63 is in; (2) the remainder plan
`dew_flow_conventions · todo/PLAN_test_leftovers_merged_core_release.md` R6 records EVERY consumer's sweep-only
stage as merged or empty — other repositories are being recorded in parallel, and the barrier is a machine's, not a
repository's; (3) the merged-core plan's *Barrier checklist* has run on every machine that builds or tests this
crate — the developer's Windows machine and its WSL, and the hosted runners (discarded with their job, no record
beyond that) — with the record in the conventions repository's `research/security/machines.md`. Where this document
and the contract differ, the contract wins and the difference is a defect of this document: report it to the parent,
do not resolve it in code.

### Boundary with the conventions repository (both sides — the shared planning rule § *A boundary between two plans*)

| Item | Built by | The other's part |
|---|---|---|
| The contract text, the `## Tokens` table (`sidecar` row), `tools/janitor.sh`, the vectors | the merged core (`research/PLAN_test_leftovers_merged_core.md`), released as `115bbf8` | E4 reads them through the mount; never edits them here |
| The Node reference helper, sweep and guard | the merged core | E4 mirrors their shape and order, not their code; stricter where decisions 2 and 3 say |
| The Rust helper, the start-of-run sweep, the Rust guard, this repository's CI step | E4 (this document) | the rollout plan names E4 as the consumer that does it |
| The pin bump | PR #63 | the remainder plan R6 orders it before the producer stage |
| The barrier record per machine, the order of all consumers' stages | the remainder plan R1, R2, R6 | E4 merges nothing before it says so |

Disjoint otherwise: E4 changes nothing in `dew_flow_conventions`. A contract finding from this epic goes back to the
parent as a note, not into the mounted copy.

## S1 — The pin bump: DONE by PR #63 (owner decision 1)

The story this document carried as S1 is PR #63, merged alone; nothing of it is built here. Its scenario 5 — *re-check
S2's constants and sweep decisions against the released contract before S2 merges* — is what this re-split did,
2026-10-31, against `115bbf8`:

| Constant / decision | E1 basis | `115bbf8` | Verdict |
|---|---|---|---|
| `PREFIX`, `REPO_TOKEN = "sidecar"`, `MAX_NAME = 63`, `ID_LENGTH = 6` | same | rule §1, `:311` | **same** |
| `START_TOLERANCE_S = 2`, `REMOVE_TRIES = 5` / 100 ms, `SWEEP_MIN_AGE = 10 min` | same | rule §3 `:151`, §2 `:88`, §4 `:176` | **same** |
| the name | `testsToDelete-sidecar-<what>-<id>` | `testsToDelete-<owner>-sidecar-<what>-<id>`; `<what>` room derived | **changed** — D2 |
| the record | `<host>@<os>:<pid>[:<start>]` | `<owner>:<host>@<os>:<pid>[:<start>[:<ns>]]`; WSL `host~distro`; `CLK_TCK` from the auxiliary vector; ≤ 512 bytes | **changed** — D3 |
| creation order | create → register → claim | claim → register → create, exclusive both | **changed** — D4; plan-gate item 6 superseded |
| who may delete | the record's host and pid | two keys: name = record = caller token, kernel owner the caller (decision 2 on Windows) | **changed** — D3, D5 |
| the sweep's pattern | `^teststodelete[-_]sidecar[-_]` | `^teststodelete[-_]<owner>[-_]sidecar[-_]`; foreign and legacy names listed | **changed** — D5 |
| an unmarked folder | removed once old | listed, never collected | **changed** — D5 |
| removal | `remove_dir_all` by the name read earlier | by identity, through a quarantine rename; a killed sweep's quarantine restored | **changed** — D5 |
| `hidepid` | `/proc/self` absent | `/proc/1` absent | **changed** — D3 |
| the guard's classes | six | eight (+ `legacy-name`, `owner-blind`); `owner-blind` takes no exemption; the token's row must be THIS repository's | **changed** — D7 |
| discovery from `git ls-files`, the CI submodule step, `catch_unwind` + the child harness, the direct-stderr line | — | unchanged by the release | **same** |

## S2 — `TestScratch`: owner, claim first, `Drop`, liveness, start-of-run sweep

- **Models.** Brief by Opus (this section); code by a Sonnet coder; review by Opus against rule §1–§4, D1–D6 and
  `owned.mjs` / `leftovers-sweep.mjs`.
- **Goal.** One type through which every test-created folder is named with the caller's owner, claimed first,
  removed on every unwinding path without ever panicking, and — after a kill — collected by the next process of the
  same user's first scratch, on two keys.
- **Files.** New `src/test_scratch.rs`; `src/main.rs` (one `mod test_scratch;` line, after `mod testing;`);
  `src/testing.rs` (one re-export line); `Cargo.toml` (`[dev-dependencies]` `regex`, the `cfg(unix)` `libc` line and
  the `cfg(windows)` `windows` block of D3, each with its reason); `Cargo.lock` (the `regex` and `libc` edges);
  `research/architecture.md:32` (the module list gains `test_scratch`) and `:308-309` (the fixtures sentence names
  both files; the test count is re-counted, not guessed).

**Step 0 — measure the Windows owner, BEFORE any helper code.** The first commit of the branch is one
`#[cfg(windows)]` test, `the_windows_owner_of_a_new_temp_file_is_printed` (kept in the module afterwards): it makes
a `create_new` file in `std::env::temp_dir()` (under the raw-temp exemption of `temp_root()` once that exists; removed
in the same test), prints the file's owner SID (`GetSecurityInfo`, converted with `ConvertSidToStringSidW`) and the
caller's SID (the thread token first, then the process token, D3) as `S-1-5-…` text, whether they are `EqualSid`, and
whether the token is elevated (`GetTokenInformation(TokenElevation)`); it never fails. Run with `--nocapture` once on
the dml leg and once on the developer's shell; both outputs are recorded in the PR and in § *Notes for the parent*.
The expectation — a hosted runner is elevated and its new files are owned by `BUILTIN\Administrators`
(`S-1-5-32-544`) — is UNVERIFIED (D3): sc. 20 asserts whatever was measured, and the consequence of an elevated
runner is that its sweep collects nothing there (the runner is discarded with its job; `Drop` still removes, D4).

**The change.** `src/test_scratch.rs`, opening with a block comment in this crate's voice: what it owns, why the
claim goes FIRST and sits beside the folder (the `log_segments.rs` listing assertions), why the kernel owner and not
the name authorises, why `Drop` never panics, why its lines bypass libtest's capture, why the sweep runs from the
first constructor.

```rust
pub(crate) const PREFIX: &str = "testsToDelete-";        // rule §1 — the ONLY literal of the prefix in src/ (D7)
pub(crate) const REPO_TOKEN: &str = "sidecar";           // rule §1 — the ONLY place the token is written
pub(crate) const REPO_ROW: &str = "the embedding sidecar"; // rule §1 — the token's row, `repository` column
pub(crate) const MAX_NAME: usize = 63;                   // rule §1, in bytes (D2)
pub(crate) const ID_LENGTH: usize = 6;
pub(crate) const MAX_DRAWS: u32 = 3;                     // rule §2 — a taken claim or a squatted name draws again
pub(crate) const START_TOLERANCE_S: u64 = 2;             // rule §3
pub(crate) const MAX_CLAIM_BYTES: usize = 512;           // rule §3 — one line; 513 are read to see one past it
pub(crate) const REMOVE_TRIES: u32 = 5;                  // rule §2, the family's figure
pub(crate) const REMOVE_BACKOFF: Duration = Duration::from_millis(100);
pub(crate) const SWEEP_MIN_AGE: Duration = Duration::from_secs(600); // rule §4

pub(crate) enum ScratchState { Claimed, Made, Released }           // D4
pub(crate) struct TestScratch { path: PathBuf, written: Vec<u8>, state: ScratchState }
impl TestScratch {
    pub(crate) fn new(what: &str) -> Self;                  // in temp_root(); the sweep runs once per process first
    pub(crate) fn new_in(parent: &Path, what: &str) -> Self; // a private parent: the tests below, never a checkout
    pub(crate) fn new_with(parent: &Path, what: &str, seams: &mut Seams<'_>) -> Self; // the scenarios' constructor
    pub(crate) fn path(&self) -> &Path;
    pub(crate) fn slash_path(&self) -> String;              // `\` → `/`, the form log_segments' tests compose with
}
impl Drop for TestScratch { /* Made → teardown(&path, stderr); Claimed → remove_claim_if_still; Released → nothing */ }

/// The constructor's seams; `Seams::real()` is what `new`/`new_in` use.
pub(crate) struct Seams<'a> {
    pub ids: &'a mut dyn FnMut() -> String,                          // the id draw
    pub write_claim: &'a mut dyn FnMut(&Path, &[u8]) -> io::Result<()>, // `create_new(true)` by default
    pub after_claim: &'a mut dyn FnMut(&Path),                       // runs between the claim and the create
    pub create_dir: &'a mut dyn FnMut(&Path) -> io::Result<()>,      // `fs::create_dir` by default
    pub out: &'a mut dyn Write,                                      // the squatter warning; stderr by default
}

pub(crate) fn temp_root() -> PathBuf;   // the one `std::env::temp_dir()` call, under an inline
                                        // `// testsToDelete-exempt: raw-temp — the helper's own temp root; every scratch is made here`
pub(crate) fn owner_token() -> &'static str;             // D2: owner_token_from(env TESTSTODELETE_OWNER), once
pub(crate) fn owner_token_from(override_: Option<&str>) -> String; // panics on an invalid override
pub(crate) fn windows_owner_token(username: &str) -> String;
pub(crate) fn what_room(owner: &str) -> usize;            // D2: 33 − owner.len(), bytes
pub(crate) fn check_what(what: &str) -> Result<(), String>;
pub(crate) fn owned_name(what: &str) -> String;
pub(crate) enum NameForm { Owner { owner: String, token: Option<String> }, Legacy { token: String }, None }
pub(crate) fn classify_name(name: &str) -> NameForm;      // D2: the ONE second segment against the owner grammar
pub(crate) fn host_name(raw: &str) -> String;             // `[@:\s]+` → `-`
pub(crate) fn host_id(raw: &str, distro: &str) -> String; // WSL: `<host>~<distro>`
pub(crate) struct Owner { pub owner: String, pub host: String, pub pid: u64, pub start: Option<u64>, pub ns: Option<u64>, pub canonical: bool }
pub(crate) enum Claim { Unmarked, Unreadable, Owner(Owner) }
pub(crate) fn this_owner() -> &'static Owner;             // OnceLock; panics only on an OS with no owner shape (D3)
pub(crate) fn run_record() -> &'static Owner;             // run_record_from(env TESTSTODELETE_RUN, probe), once
pub(crate) fn run_record_from(given: Option<&str>, probe: impl Fn(u64) -> Probe, out: &mut dyn Write) -> Owner;
pub(crate) fn format_owner(owner: &Owner) -> String;      // `<owner>:<host>@<os>:<pid>[:<start>[:<ns>]]`
pub(crate) fn parse_owner(text: &[u8]) -> Claim;          // D3; > 512 bytes → Unreadable
pub(crate) fn is_mine(path: &Path) -> bool;               // D3: the kernel owner (decision 2 on Windows); false on any failure
#[cfg(windows)] pub(crate) fn caller_sid() -> &'static [u8]; // D3: the thread token first, then the process token; once
pub(crate) fn read_claim(dir: &Path, mine: &dyn Fn(&Path) -> bool) -> Claim; // the sibling `<dir>.owner`; absent → Unmarked
pub(crate) fn remove_claim_if_still(dir: &Path, written: &[u8]);  // D4; never panics
pub(crate) enum Probe { Gone, Unknown, Alive { start: Option<u64> } }
pub(crate) fn probe(pid: u64) -> Probe;                   // D3; never spawns a process; hidepid-aware
pub(crate) struct Facts { pub hidepid: bool }             // `Facts::real()` reads /proc/1
pub(crate) enum OwnerState { Live, Gone, Unknown }
pub(crate) fn owner_state(claim: &Claim, name_owner: Option<&str>, mine: bool, me: &Owner, facts: &Facts,
                          probe: impl Fn(u64) -> Probe) -> OwnerState;
pub(crate) fn remove_with_retries(path: &Path, remover: impl FnMut(&Path) -> io::Result<()>,
                                  on_failed_attempt: impl FnMut(u32)) -> io::Result<()>;
pub(crate) fn teardown(dir: &Path, out: &mut dyn Write) -> bool;  // D4; true when folder and claim are gone
pub(crate) struct Hooks<'a> { pub after_capture: Option<&'a mut dyn FnMut(&Path)>, pub after_rename: Option<&'a mut dyn FnMut(&Path, &Path)> }
pub(crate) fn sweep(root: &Path, now: SystemTime, min_age: Duration, probe: impl Fn(u64) -> Probe,
                    mine: impl Fn(&Path) -> bool, hooks: &mut Hooks<'_>) -> SweepReport;  // D5
pub(crate) fn guarded(run: impl FnOnce() -> SweepReport, out: &mut dyn Write);  // D5: catch_unwind, then print_report; never panics
pub(crate) fn print_report(report: &SweepReport, out: &mut dyn Write);  // D5; one write per line; nothing for an empty report
```

- `remove_with_retries` calls `remover` at most `REMOVE_TRIES` times, sleeps `REMOVE_BACKOFF` between tries (never
  after the last), treats `ErrorKind::NotFound` as success, calls `on_failed_attempt(n)` after each failure (the test
  seam that makes the open-handle scenarios deterministic), and returns the last error.
- No `std::process::Command` in `test_scratch.rs` outside its own `#[cfg(test)] mod tests` (the child harness of
  scenarios 7–8, the one `hostname` of scenario 15, the one `mklink /J` of scenario 17): the writer and the probe
  never spawn (rule §3).
- Both constructors pass through the sweep's `Once`; `new_in` with a private parent still sweeps the temp root, never
  the parent. The sweep's `probe`, `mine`, `Facts` and `Hooks` are the real ones there; the scenarios inject them.
- Only `PREFIX`, the owner-aware pattern and the quarantine sibling (`quarantine` is not a name, G:109) spell the
  prefix in this file: the machine-wide scan and S4's guard read it too.

**Scenarios** (`#[cfg(test)] mod tests` inside `test_scratch.rs`; RED first — written against the names above before
the bodies exist, watched failing on each assertion as the module grows; every scenario that counts or lists entries
does so in a PRIVATE parent made with `new_in`, because 8 test threads create scratches in the temp root at once; a
legacy or foreign name a scenario needs is built at run time from `PREFIX` and pieces, never spelled whole — S4's
`legacy-name` class reads this file):

0. `the_windows_owner_of_a_new_temp_file_is_printed` — step 0 above; the first commit, before any other name exists.
1. `a_scratch_is_named_with_the_owner_the_token_and_a_random_id` — `owned_name("seed")` matches
   `^testsToDelete-[uw][0-9]{1,10}-sidecar-seed-[A-Za-z0-9]{6}$` and its second segment equals `owner_token()`;
   1 000 calls are all distinct; `TestScratch::new("unicode пробел")` exists and its name starts
   `testsToDelete-<owner_token()>-sidecar-unicode пробел-`; `classify_name` says `Owner` for it, `Legacy { token:
   "sidecar" }` for `PREFIX + "sidecar-x-abc123"`, `Owner` with owner `u7` for `PREFIX + "u7-mcp-x-abc123"`, `None`
   for `bge-log-1-first`.
2. `the_owner_token_is_the_users_and_an_invalid_override_refuses` — on Linux `owner_token()` is `u` + `getuid()`;
   on Windows `w` + digits; `windows_owner_token` reproduces every row of the vectors file's `windows-owner` section
   (read through `CARGO_MANIFEST_DIR/.claude/rules/shared/tools/fixtures/owner-vectors.json` with `serde_json`; the
   file missing fails naming `git submodule update --init .claude/rules/shared`); `owner_token_from(Some("u42"))` →
   `u42`; `Some("bob")`, `Some("u")`, `Some("u12345678901")` → a panic (`catch_unwind`), nothing created; `Some("")`
   and `None` → the process's own.
3. `a_scratch_is_claimed_first_beside_itself_and_never_inside` — `<dir>.owner` exists in the same parent, holds
   `format_owner(run_record()) + "\n"`, `read_claim(dir, &is_mine)` equals `this_owner()` and `read_claim(dir, &|_|
   false)` is `Unreadable`; `read_dir(dir)` is empty. **Claim first, observed** through `new_with`: a `create_dir`
   seam that answers `PermissionDenied` → the constructor panics and the private parent is EMPTY afterwards (the
   claim was written, then removed by the unwinding `Drop` of a `Claimed` value); a `write_claim` seam that fails →
   the constructor panics and no folder was ever made.
4. `an_invalid_what_creates_nothing_and_the_room_is_derived` — `""`, `"a/b"`, `"a\\b"`, `"-x"`, `"a\u{1}b"`, and a
   `what` of `what_room(owner_token()) + 1` bytes each panic (`catch_unwind`), and the private parent stays empty; a
   `what` of exactly `what_room(..)` bytes succeeds and the name is 63 bytes long; `what_room("u1000")` is 28,
   `what_room("w4294967295")` is 22; a `what` of `what_room(..) − 1` characters that is `what_room(..) + 1` bytes
   (one `п`) panics — bytes, not characters (D2).
5. `a_taken_claim_draws_again_and_a_squatted_folder_is_never_adopted` — in a private parent through `new_with`, an
   `ids` seam answering `a, b, c`: a file `…-a.owner` written by the test → the scratch is `…-b`, the stranger's
   claim untouched; a FOLDER `…-a` without a claim → the scratch is `…-b`, `…-a` still exists with NO `.owner` beside
   it (our claim was removed again), one warning line through the `out` seam names `…-a`; a `…-a.owner` the test
   overwrites in the `after_claim` seam, before our `create_dir` finds `…-a` squatted → that claim is NOT removed (it
   no longer holds our bytes), the value is `Released` and dropping it touches nothing — RED without D4's state: a
   `Drop` that removes "the claim" takes the stranger's; three squatted draws → a panic naming `what` and the parent,
   the parent holding nothing of ours.
6. `a_scratch_is_removed_when_its_test_panics` — inside `catch_unwind`: create, record the path, `panic!`; after: the
   folder and its `.owner` are gone.
7. `a_failed_assertion_in_a_real_test_still_removes_its_scratch` — the child harness: `current_exe()` run with
   `test_scratch::tests::child_scratch --exact --ignored --nocapture --test-threads=1` and the env
   `TEST_SCRATCH_CHILD=assert` + `TEST_SCRATCH_PARENT=<private parent>`; the `#[ignore]`d `child_scratch` returns at
   once unless that env is set, else creates `new_in(parent, "child")`, prints the path, and `assert!(false)`. The
   child exits non-zero; its stdout names a path that, with its `.owner`, no longer exists.
8. `a_killed_test_leaves_its_scratch_claimed_and_the_next_sweep_collects_it` — the same harness with
   `TEST_SCRATCH_CHILD=abort`: the child creates, prints, and `std::process::abort()`s (no unwind, no `Drop` — the
   SIGKILL case). The folder and `.owner` remain; `read_claim(dir, &|_| true)` names this owner token, the child's
   pid, this host (and on Linux a start and an ns). `sweep(parent, now + 11 min, .., probe, |_| true, ..)` with a
   probe answering `Alive` for that pid keeps both with reason `live`; with `Gone` removes both, `removed` names the
   folder, no quarantine is left. (Never the REAL probe on a dead child's pid: Windows hands exited pids to the next
   process.)
9. `removal_retries_then_gives_up_without_panicking` — an injected remover failing twice then succeeding → 3 calls,
   `Ok`; failing always → exactly 5 calls, ≥ 400 ms elapsed, `Err`, no panic; `NotFound` on the first call → 1 call,
   `Ok`.
10. `a_scratch_held_open_stays_prefixed_and_claimed_and_says_so_once` — `#[cfg(windows)]`: outer = `new("held")`;
    inner = `new_in(outer, "inner")`; a file inside inner opened with `share_mode(0)`; `teardown(inner, &mut buf)` →
    `false`, no panic, `buf` holds exactly one line naming inner's path, inner and its `.owner` still exist; release
    the handle; `outer` drops last and takes everything. The same set-up with `remove_with_retries` and an
    `on_failed_attempt` that releases the handle after attempt 1 → `Ok` at attempt 2, inner gone (the retry is real).
11. `a_scratch_that_cannot_be_removed_stays_prefixed_and_claimed` — `#[cfg(unix)]`: inner inside outer, outer
    `chmod 0o555`; teardown → `false`, one line, inner and its `.owner` remain; the premise is ASSERTED (a process
    running as root removes it anyway — the test then fails saying so, never passes for the wrong reason); restore
    `0o755`, outer drops.
12. `an_owner_round_trips_and_the_shared_vectors_agree` — `parse_owner(format_owner(this_owner()) + "\r\n")`
    round-trips; on Linux `this_owner()` has a start AND an ns, on Windows a start and no ns; `format_owner` of an
    owner with `ns` and no `start` writes neither; every row of the vectors file's `record` section parses to
    `valid` or `Unreadable` as it says (`u1000:h@linux:1`, `u1000:h@linux:1\r\n` valid; `h@linux:1`, `u1000:h@linux:1\r`,
    `u1000:h@linux:1:2:3:4`, `u1000:h@linux:١٢` (r-i25, Unicode digits), the 513-byte text unreadable;
    `u1000:h@linux:007` valid and non-canonical); every row of `hostnames` through `host_name`, of `names` through
    `classify_name`, of `scope` through `owner_state` with its facts, answers as the row says.
13. `owner_state_follows_rule_three_in_order` — injected probe and `Facts`, each step alone: `Unmarked`/`Unreadable`
    → unknown; non-canonical → unknown; the record's owner `u7` under caller `u1000` → unknown; name owner ≠ record
    owner → unknown; `mine = false` → unknown; `other@<os>` → unknown; this host with the other OS tag → unknown; this
    host in other case → evaluated; on Linux a record without `ns`, with `ns + 1`, or a `me` whose own `ns` is `None`
    → unknown; pid 0 → unknown; `@windows` pid 1 → unknown; `Gone` → gone; `Gone` with `Facts { hidepid: true }` →
    unknown; `Unknown` → unknown; alive with either start absent → live; Δ = 2 s → live; Δ = 3 s either way and
    Δ ≫ 3 s → unknown, never gone.
14. `the_real_probe_sees_this_process_and_not_an_impossible_pid` — `probe(std::process::id())` is `Alive` with a
    start within 2 s of `this_owner().start`; `probe(u64::from(u32::MAX) + 1)` on Windows → `Unknown`; on Linux a
    pid of `/proc/sys/kernel/pid_max + 1` (the premise asserted before and after, and `/proc/1` visible — else the
    test says `hidepid`) → `Gone`; on Windows **measured first**: `probe(own pid + 2)` is recorded in the PR — pids
    are handle-table indices and NT ignores the two tag bits, so `4n + 2` may open `4n` (the E1 premise); the dead
    pid is then a `4n` whose four aliases `4n..4n+3` are all absent from a ToolHelp snapshot
    (`CreateToolhelp32Snapshot` / `Process32FirstW`) taken before AND after → `Gone`.
15. `the_owner_names_this_machine_as_the_janitor_does` — one spawn of `hostname` (this test only), normalised with
    `host_name`, compared case-insensitively with `this_owner().host` up to `@` — up to `~` when `WSL_DISTRO_NAME` is
    set, in which case the part after `~` equals `host_name(WSL_DISTRO_NAME)`.
16. `a_claim_is_read_through_the_one_safe_read` — a `.owner` that is a directory → `Unreadable`; a symlink to a
    valid claim → `Unreadable` (`symlink_metadata`, never followed); 512 bytes of a valid record → read; 513 →
    `Unreadable`; `mine` injected `false` → `Unreadable`, `true` → read; `#[cfg(unix)]` a FIFO (`libc::mkfifo`) →
    `Unreadable` within a second, nothing blocks. `run_record_from(Some(..), probe, &mut buf)`: a record of this owner
    and a probe-`Alive` pid → used, `buf` empty; a record of `u7`, of a probe-`Gone` pid, or `bob` → one line in `buf`
    and `this_owner()`; `None` → `this_owner()`, `buf` empty.
17. `the_sweep_removes_only_what_a_gone_owner_left_and_lists_the_rest` — on a private root, `now` = real now +
    11 min unless stated, names built from `PREFIX` and pieces: claimed by this process → kept (`live`); this owner's
    name with a claim of another owner `u7` → kept (`unknown`); foreign host → kept (`unknown`); unreadable claim →
    kept; `mine` injected `false` for one entry → kept (`foreign-uid`) AND a `foreign:` line; probe-`Gone` claim →
    folder and claim removed, by identity (the `after_capture` hook replaces the folder by a fresh one of the same
    name → kept `replaced`, the fresh one and its claim untouched; the `after_rename` hook that drops a file into the
    quarantine → removed anyway, the identity is the directory's); **a link is never a candidate:** a symlink
    (Linux, `symlink`) or a junction (Windows, `cmd /C mklink /J` — a test may spawn) under one of this owner's names,
    beside a claim of a probe-`Gone` pid, pointing at a folder holding a file → kept `not a directory`, the link still
    there, the target and its file intact; **unmarked and old → kept (`unmarked`) and an `unmarked:` line, never
    removed**; unmarked at real `now` (young) → untouched and unreported; a prefixed folder holding a `.git` FILE →
    kept (`other clone`); another owner's name `PREFIX + "u7-sidecar-x-abc123"` → a `foreign:` line, untouched; a
    legacy name `PREFIX + "sidecar-x-abc123"` with a probe-`Gone` old-grammar claim → a `legacy:` line with the exact
    command, untouched; `PREFIX + "u7-mcp-x-abc123"` and `bge-log-1-first` → untouched and unreported; the `_` and
    upper-case spellings of this owner's name → considered; **orphan claims are judged by owner** — four equally old
    orphan `.owner` files of this owner's names, claimed by THIS process (`live`), by a probe-`Gone` pid, by a foreign
    host (`unknown`), and holding unreadable text → only the `gone` one disappears; the other three stay, each in
    `kept` with its reason; a folder claimed by THIS process with `now` injected 10 days ahead → kept (`live`, age
    never authorises). **A removal that teardown gives up on is in `kept`, nothing is lost** — three cases, the
    premise asserted each time (root removes anything): POSIX, the private root `chmod 0o555` → the quarantine
    rename is REFUSED, kept `could not remove: <error>`, folder and claim exactly where they were; POSIX, a
    sub-folder `chmod 0o555` holding a file inside the candidate → the rename succeeds, `remove_dir_all` fails, the
    folder is renamed BACK under its own name (D5's deviation), kept `could not remove:`, absent from `removed`, its
    `.owner` still beside it, no quarantine left; Windows, a file inside held with `share_mode(0)` → the rename is
    refused (asserted: that is all Windows is asked to prove), kept `could not remove:`, nothing moved. Released or
    restored, a second sweep removes each.
18. `a_quarantine_a_killed_sweep_left_is_restored_or_listed` — a directory `.<own name>.teststodelete-quarantine-
    <pid>-abcd` (the literal from pieces) of a probe-`Gone` pid, 11 minutes old by both stamps → renamed back to
    `<own name>`, a `notice: restored …` line; the same with `<own name>` taken by an EMPTY directory → untouched,
    kept `quarantine-conflict`, a `quarantine: … restore with: …` line, the empty directory still there with nothing
    in it (RED first on Linux with a plain `rename`: the empty directory is replaced); taken by a file → the same; a
    second sweep with the name freed → restored (every run retries); of a probe-`Alive` pid, or young by either
    stamp → untouched, unreported; `mine` `false` → untouched.
19. `the_sweep_never_fails_the_run` — **every I/O error is reported, none stops the sweep**: a root that does not
    exist, and a root that is a FILE → no panic, `removed`, `kept` and `listed` empty, and `errors` holds one entry
    per pass naming that root and the `read_dir` error; `print_report` given that report and a `Write` that records
    each call writes exactly those lines, ONE call per line; `guarded(|| panic!("probe"), &mut buf)` → returns, `buf`
    holds one line naming the panic; `guarded` called again afterwards runs its closure. RED first: written before
    `errors` is filled, the file-root case is observed with an empty `errors`. ("The `Once` is not poisoned" is not
    observable in-process and is not asserted: `guarded` never panics, which is what keeps it clean.)
20. `the_kernel_owner_answer_matches_what_was_measured` — `#[cfg(windows)]`, the assertion of step 0: a scratch in a
    private root; the test computes `owner SID == caller SID` for it as step 0 does and asserts `is_mine(path)` equals
    that; `caller_sid()` equals the process token's `TokenUser` (no impersonation is active in a test); then
    `sweep(root, now + 11 min, .., probe = Gone, is_mine, ..)` with the REAL `is_mine`: SIDs equal → removed; not
    equal (an elevated shell) → kept `foreign-uid`, a `foreign:` line, the folder untouched; in both cases the value
    is re-created and its `Drop` removes it — teardown needs no `is_mine` (D4).

- **Commands.** `node .claude/rules/shared/tools/low-priority.mjs cargo test --release --locked -j 8 -- test_scratch --test-threads=8`;
  the whole suite on Windows (dml) and inside WSL (`nice -n 19 cargo test --release --locked -j 8 --no-default-features --target-dir target-wsl -- --test-threads=8`);
  `cargo fmt --package bge-sidecar --check`.
- **DoD.** Step 0's two outputs recorded (the dml runner and the developer's shell, each with its elevation); every
  scenario observed RED, then GREEN, on Windows and on the ubuntu CI leg (scenarios 0, 10 and 20 run on Windows only,
  11 and the FIFO of 16 on Linux only); the 148 existing tests unchanged and green; the D3 Win32 and libc names
  recorded in the PR; `architecture.md` updated; no `Command` outside the test module (review); no literal of the
  prefix in the file but `PREFIX` and the owner-aware pattern (review, and S4).

## S3 — The nine sites and their callers move to `TestScratch`

- **Models.** Brief by Opus; code by a Sonnet coder; review by Opus against the facts table and D4.
- **Goal.** No test of this crate leaves a folder behind on any path; the 15 every-run leaks end.
- **Files.** `src/compile_cache.rs`, `src/log_segments.rs`, `src/provider.rs`, `src/testing.rs`, `src/tokens.rs`,
  `src/handlers.rs`, `src/introspection.rs`, `src/preflight.rs`.

**The change** (each site keeps its meaning; `<what>` is the old prefix without its `bge-`/trailing `-`; the scratch
is the FIRST binding of its test, D4; every `<what>` is at most 15 characters, under the 22 of the longest owner):

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
4. **Nothing else moved:** the 148 existing tests pass on both legs, with the same names; in particular
   `log_segments.rs`'s exact-listing tests (`:290`, `:304`, `:313`, `:335`, `:351`) and the Windows
   held-file test (`:376`) — the claim beside the root never shows up in them — and `tokens.rs:330` (the folder its
   test removes).
5. **No leftover after a full run** (by hand, recorded in the PR): before S3, list the temp root's old-form names
   (`mxr-race-*`, `mxr-wipe-*`, `mxr-scope-*`, `bge-log-*`, `bge-preflight-*`, `bge-seed-*`, `bge-reg-*`,
   `bge-tok-*`) on Windows (`%TEMP%`) and in WSL (`$TMPDIR` or `/tmp`) and record the counts; after a full run of each
   leg: zero NEW old-form names, zero `testsToDelete-<owner>-sidecar-*` entries and zero `.owner` siblings of them;
   `bash .claude/rules/shared/tools/janitor.sh --token sidecar --dry-run` lists nothing of this token. The
   pre-existing old-form leftovers are reported to the owner and removed only on the owner's word (D6).

- **Commands.** As S2 (both legs, the whole suite); `cargo fmt --package bge-sidecar --check`.
- **DoD.** Scenarios 1–3 observed RED with the symptom above, then GREEN; 4 and 5 recorded; zero
  `std::env::temp_dir()` calls left outside `test_scratch.rs` (S4 proves it).

## S4 — The guard test, its break-its, and the CI step

- **Models.** Brief by Opus; code by a Sonnet coder; review by Opus against rule §1 and §7, D7 and decision 3.
- **Goal.** No test, script or workflow here creates a temp folder, worktree, container or store resource outside
  `TestScratch` (or under a valid inline exemption), no source spells a legacy name or an owner-blind match, the
  helper's token is THIS repository's registered row, and a scan of nothing is red.
- **Files.** New `tests/test_leftovers.rs`; `.github/workflows/ci.yml` (`build-test` gains `run: git submodule update
  --init --depth 1 .claude/rules/shared` after checkout, on both legs — the `contract` job's line, S2's vectors test
  needs it too); `README.md` only if it documents test commands that need the submodule; `research/module_inference.md`
  (a paragraph beside the `tests/local_cpu_budget.rs` one at `:132-136`).

**The guard.** Pure functions over `(relative path, text)` plus one `#[test]` over the real tree. Every detector and
every fixture string in the file is built from pieces (`concat!("temp_", "dir(")`, `concat!("teststo", "delete")`,
the exemption marker as `concat!("testsToDelete-", "exempt:")`), because the file scans ITSELF.

- **Discovery — an inventory from git, not hand-listed folders.** The test runs `git ls-files -z --cached --others
  --exclude-standard` once in `CARGO_MANIFEST_DIR` (a test may spawn — the no-spawn rule of D3 binds the helper's
  writer and probe, not the guard) and keeps every listed file whose extension is a language the guard reads,
  WHEREVER it sits: `*.rs`; `*.yml`/`*.yaml` under `.github/workflows/`; `*.mjs`, `*.js`, `*.cjs`, `*.ts`. So every
  script root is covered without naming it — today `.github/scripts/` (`branch-protection.mjs`) and `http/`
  (`httpyac.config.js`), and any new root the day one is added. Four path exclusions come FIRST, by prefix of the
  RELATIVE path: any directory whose name starts with `target` (matched as `^target[^/]*/` on any path segment, so
  every Cargo target directory is skipped even where it is not git-ignored), `node_modules/` (any depth),
  `vendor-fastembed/` (upstream's bytes, not built as tests here) and `.claude/` (another repository). Then **a
  language the guard does not read is red**: a remaining `*.sh`, `*.ps1` or `*.py` ⇒
  `test_leftovers: <file> is a language this guard does not read — extend the guard before adding it` (the trim of
  plan-gate round 2: no tracer is written for a language the repository has no file in). **No file is excluded by
  name**: the frozen exclusion list is EMPTY, pinned by a test, and `tests/test_leftovers.rs` is scanned like any
  other `.rs`. `git` missing or exiting non-zero ⇒ red naming the command, never an empty scan. **Scan of nothing:**
  fewer than one `.rs` under `src/`, one under `tests/`, or one workflow ⇒ red with `test_leftovers: scanned nothing
  under <root> — discovery broke; the repository is not clean`.
- **The inventory is verified against the repository**: on the branch, the PR records the list the guard scanned
  and the output of `git ls-files` filtered by the same extensions, and they are equal; a file in one and not the
  other is a discovery defect fixed before merge.
- **Parsing.** A workflow is read whole, line by line — no YAML parser, no `run:` isolation (stricter than the
  reference guard: a `mktemp` in a workflow comment that is not a whole-line `#` comment is a finding, and the cure
  is to delete it). Whole-line `#` comments are skipped in workflows only; whole-line `//` comments in `.rs` and JS;
  a `#` line in `.rs` is an attribute and is read. Comments are read for exemptions.
- **Classes** (rule §7's eight, their Rust/YAML/JS spelling, G:248-265 for the regexes).
  - `raw-temp`: `\btemp_dir\s*\(`, `\benv::var(_os)?\(\s*"(TEMP|TMP|TMPDIR)"`, `\btempfile::`, `\btempdir\s*\(`,
    `\bmkdtemp(Sync)?\s*\(`, `\btmpdir\s*\(`, `process\.env\.(TEMP|TMP|TMPDIR)\b`, `GetTempPath`,
    `CreateTempSubdirectory`, `GetTempFileName`, and in a workflow file `\bmktemp\b`. Not `RUNNER_TEMP` (rule §7
    `:250`).
  - `raw-worktree`: `\bworktree\s+add\b`, `"worktree",\s*"add"`.
  - `docker-run`: ANY `\bdocker\s+(run|create|compose)\b` (D7: none exists here; the first one extends this class to
    rule §5's carve-out check).
  - `docker-name`: `--name`, `volume create`, `network create`, `WithName(`, compose `-p`/`--project-name`,
    `COMPOSE_PROJECT_NAME`, whose value is not of the OWNER form `^teststodelete[-_][uw][0-9]{1,10}[-_]`
    case-insensitive (G:101) — a value that is wholly a shell expansion is not judged.
  - `legacy-name`: a literal `teststodelete[-_]([a-z0-9]+)` case-insensitive whose captured second segment is not of
    the owner grammar and is not `exempt`, `quarantine` or an upper-case `TESTSTODELETE_[A-Z]+` environment name
    (G:107-109) — in EVERY scanned file, `src/test_scratch.rs` included; and `teststodelete[-_]\$?\{?\s*REPO_TOKEN\b`
    (a constant-built legacy prefix, G:110).
  - `owner-blind` (decision 3): in `.rs` outside `src/test_scratch.rs`, ANY `teststodelete[-_]` case-insensitive not
    followed by `[uw][0-9]{1,10}[-_]`, the three non-names above excepted — no tracer, no exemption; in JS files and
    workflow lines, the same literal, OR a use (`$P`, `${P…}`, a bare bound name in code) of a name the same file
    assigns (`P=` in a workflow line, `const|let|var P =` in JS) a string value holding `teststodelete[-_]` without an
    owner segment right after it, judged on every line whatever the order or scope (a name bound in another file,
    the environment or an argument is not traced; a prefix split over two literal pieces binds nothing). The one
    exception is the shipped janitor's inventory line — not present in this repository, since the janitor is run from
    the mount, never copied.
  - `shared-store`: `\b(DROP\s+(DATABASE|SCHEMA|TABLE)|TRUNCATE|FLUSHALL|FLUSHDB)\b`, case-sensitive (prose
    "truncated" never matches).
  - `ryuk-off`: `TESTCONTAINERS_RYUK_DISABLED\s*[:=]\s*["']?true` — never exemptible.
  - `dropped-scratch` (this repository's own, S3's trap): `let\s+_\s*=\s*.*(TestScratch::new(_in|_with)?|seed_scratch|model_cache_with_a_tokenizer)\s*\(`
    and `let\s*\(\s*_\s*,` before `seed_scratch(`.
- **Exemptions — inline only** (rule §7 `:259-263`): `testsToDelete-exempt: <class> — <reason>` on the finding's line
  or the line above; the class is one of the eight except `ryuk-off` and `owner-blind` (`legacy-name` IS exemptible);
  the reason ≥ 10 characters and ≥ 3 words and not a placeholder (`needed`, `see above`, `todo`); `shared-store` needs
  `lock:`. Anything else carrying `testsToDelete-exempt:` is a finding of class `exemption`. The one real exemption:
  `temp_root()` in `src/test_scratch.rs`.
- **The token is registered AND this repository's** (rule §1; G:695-725): exactly one `const REPO_TOKEN: &str =
  "<t>";` and one `const REPO_ROW` across `src/**/*.rs`, `<t>` matching `^[a-z0-9]{2,12}$` and NOT `^[uw][0-9]{1,10}$`,
  present in the FIRST column (backticks stripped) of the GFM table under `## Tokens` in
  `.claude/rules/shared/common/test-leftovers.md`, whose header is exactly `| token | family | repository |`, with no
  token listed twice, and whose `repository` column for that row equals `REPO_ROW` case-insensitively (`the embedding
  sidecar`, rule `:311`). The mounted file missing ⇒ red with `the shared rules are not checked out — git submodule
  update --init .claude/rules/shared` (fail closed: no table registers nothing).
- **Output:** one line per finding, `<file>:<line> — <class> — <text>`, then one cure sentence per class naming rule
  §7, then the assertion fails.

**Scenarios** (RED first: the real-tree test is written and run on the branch's commit BEFORE S3's):

1. `this_repository_creates_nothing_outside_test_scratch` — RED before S3: exactly the nine `raw-temp` findings of the
   facts table (at their then-current lines), nothing else — in particular no finding in `tests/test_leftovers.rs`
   itself and no `legacy-name` or `owner-blind` finding in `src/test_scratch.rs`'s tests (every such name is built
   from pieces); the counts recorded in the PR. GREEN after S3: zero.
2. `every_class_is_refused_once_and_a_valid_exemption_passes` — fixture strings through the pure function (every
   fixture built from pieces at run time): a raw `std::env::temp_dir()` in `src/x.rs` ⇒ 1 `raw-temp` naming the line;
   `env::var("TMPDIR")` ⇒ 1; `mkdtempSync(` in `http/x.mjs` ⇒ 1; `mktemp -d` in a workflow `run:` line ⇒ 1, and in a
   workflow line that is not a whole-line comment ⇒ 1; `RUNNER_TEMP` ⇒ 0; `git worktree add` ⇒ 1; `docker run --rm img`
   ⇒ 1 (`docker-run`, D7); `docker volume create v` ⇒ 1 `docker-name`; `docker volume create <owner-form name>` ⇒ 0;
   `docker compose -p sidecar-x` ⇒ 1; `TRUNCATE TABLE t` ⇒ 1, with a `shared-store` exemption carrying `lock:` above ⇒
   0, without `lock:` ⇒ 1 `exemption`; the word `truncated` ⇒ 0; `TESTCONTAINERS_RYUK_DISABLED=true` with an exemption
   above ⇒ still 1; a literal `<prefix>sidecar-x-abc123` in `src/x.rs` ⇒ `legacy-name` AND `owner-blind` (two lines,
   one each); the same with a `legacy-name` exemption above ⇒ the `owner-blind` line remains;
   `<prefix>u1000-sidecar-x-abc123` ⇒ 0; the quarantine sibling and `TESTSTODELETE_OWNER` in `src/x.rs` ⇒ 0; a regex
   `^<prefix lower>[-_]sidecar` in `src/x.rs` ⇒ 1 `owner-blind`, in `src/test_scratch.rs` ⇒ 0 `owner-blind`; a
   `#[cfg(test)]` line followed by `temp_dir()` on the next line in `src/x.rs` ⇒ 1 (a `#` line in Rust is read); in
   a workflow, `P=<prefix>` on one `run:` line and `rm -rf "$P"*` nine lines later ⇒ 1 `owner-blind` naming the later
   line; `P="$OTHER"` + `rm -rf "$P"*` ⇒ 0; `P='teststo''delete-'` + `rm -rf "$P"*` ⇒ 0; in `tools/x.mjs` `const P =
   "<prefix>"` and `.filter((n) => n.startsWith(P))` ⇒ 1; a well-formed `raw-temp` exemption ⇒ 0; `testsToDelete-exempt:
   raw-temp — needed` ⇒ 1 `exemption`; `testsToDelete-exempt: owner-blind — a perfectly fine reason here` ⇒ 1
   `exemption`; `testsToDelete-exempt: nonsense — a perfectly fine reason here` ⇒ 1 `exemption`; `let _ =
   TestScratch::new("x");` ⇒ 1 `dropped-scratch`; `let (_, from, to) = seed_scratch("x");` ⇒ 1; `let _scratch =
   TestScratch::new("x");` ⇒ 0; a commented-out `// std::env::temp_dir()` ⇒ 0.
3. `a_scan_of_nothing_is_red` — the pure discovery-verdict function over in-memory listings: an empty listing, one
   with no `src/*.rs`, and one with no workflow ⇒ the "scanned nothing" failure each. No folder is made: this file is
   an integration test and cannot reach `crate::testing`'s `TestScratch`, and a second temp helper here would be the
   copy the guard exists to refuse.
4. `the_token_is_this_repositorys_registered_row` — the real mounted rule: `sidecar` with `REPO_ROW = "the embedding
   sidecar"` ⇒ no finding; fixture rule texts: no `## Tokens` ⇒ 1; a two-column table ⇒ 1; the token only in the
   `repository` column ⇒ 1; a token listed twice ⇒ 1; `Sidecar`, `x` and `u1000` ⇒ 1 each; the real table with
   `REPO_ROW = "the MCP server"` ⇒ 1 (`belongs to … not to this repository`); an empty `REPO_ROW` ⇒ 1 (fails closed);
   two `REPO_TOKEN` constants ⇒ 1.
5. `the_exclusion_list_is_empty_and_pinned` — pinned at zero entries. And
   `discovery_keeps_every_script_root_and_skips_every_target` — the pure filter over an in-memory listing:
   `.github/scripts/a.mjs`, `http/x.js`, `tools/lib/y.mjs`, `deep/new/root/w.ts`, `tests/test_leftovers.rs` ⇒ kept;
   `target/debug/a.rs`, `target-wsl/x.rs`, `target-ci/y.sh`, `sub/target-x/z.rs`, `http/node_modules/p/i.js`,
   `vendor-fastembed/src/a.rs`, `.claude/rules/shared/tools/t.mjs` ⇒ dropped (an excluded path is never red, whatever
   its language); `scripts/z.sh`, `deep/new/root/w.ps1`, `tools/q.py` ⇒ the "language this guard does not read"
   failure, naming the file; `targets.rs` at the root (a FILE whose name starts with `target`) ⇒ kept. RED first:
   written against the old hand-listed discovery, `tools/lib/y.mjs` and `deep/new/root/w.ts` are observed dropped.
6. **Break-it on the real tree** (by hand, recorded in the PR): one `std::env::temp_dir()` added to a test module ⇒
   the guard red naming `file:line`; the same literal written WHOLE inside `tests/test_leftovers.rs` ⇒ red naming the
   guard's own file; one `const OLD: &str = "<prefix>sidecar-";` in `src/tokens.rs`'s tests ⇒ red with `legacy-name`
   and `owner-blind`, and an exemption above it clears only the first; an empty `scripts/x.sh` ⇒ red with the
   unread-language sentence; reverted ⇒ green. The same with `.claude/rules/shared` de-initialised ⇒ red with the
   submodule sentence.

- **Commands.** `node .claude/rules/shared/tools/low-priority.mjs cargo test --release --locked -j 8 --test test_leftovers -- --test-threads=8`;
  the whole suite on both legs; `actionlint` passes on the changed `ci.yml` (the `workflows` job).
- **DoD.** Scenario 1 observed red with exactly the nine sites, then green; every break-it observed red; the scanned
  inventory recorded and equal to the filtered `git ls-files`; both CI legs green WITH the submodule step;
  `module_inference.md` updated (it names the `target*` exclusion, decision 3 and the unread-language red).

## Test plan

| The parent's "proven by" | Where |
|---|---|
| A panic inside the scope, then the path no longer exists | S2 sc. 6 (in-process `catch_unwind`), sc. 7 (a real failing `#[test]` through the harness). Not the parent's "two tests" — libtest runs tests in parallel and unordered, so a second test cannot observe the first |
| RED-GREEN for the pre-clean-only class | S3 sc. 1, 2; the panic-skipped class S3 sc. 3; the inline sites through S2 sc. 6 and S4 sc. 1 |
| The guard's break-it | S4 sc. 2, 6 |
| An open handle on Windows: no panic, folder gone or a prefixed leftover remains | S2 sc. 10 (`share_mode(0)`), with the POSIX twin sc. 11; the leftover is also CLAIMED, which the parent did not require and the contract does |
| (contract, not in the parent) claim first; two keys; a killed run's leftover is collected, a live owner's, a stranger's, a link's and an unmarked one survive; identity removal, exclusive restore; the shared vectors; the measured Windows owner | S2 sc. 0, 3, 5, 8, 12, 13, 16, 17, 18, 19, 20 |

## Definition of Done

- [ ] The pin carries `115bbf8` (PR #63, merged alone); the mounted rule lists `sidecar` as `the embedding sidecar`.
- [ ] Every test-created folder is a `TestScratch` named `testsToDelete-<owner>-sidecar-<what>-<id>`, claimed FIRST
      beside itself with the run's record, removed by `Drop` on every unwinding path, never panicking; a teardown
      that gives up leaves it prefixed and claimed with one stderr line (S2, S3).
- [ ] The first scratch of a test process sweeps this owner's and token's leftovers on the two keys, by identity,
      lists foreign, legacy and unmarked ones, and never fails the run (S2).
- [ ] The guard is red on every one of the eight classes, on `owner-blind` without a tracer in Rust source, on an
      unregistered or foreign token and on a scan of nothing; CI fetches the rules it reads (S4).
- [ ] `research/architecture.md` and `research/module_inference.md` describe the helper and the guard.
- [ ] The Windows owner was MEASURED before the helper was coded (S2 step 0): the owner SID of a new `%TEMP%` file and
      the caller's SID on the dml runner and on the developer's shell, and whether each shell is elevated, recorded in
      the PR and in § *Notes for the parent*.
- [ ] S2–S4 merged as ONE squash-merged PR, only after the merge condition of § *Order* held and was recorded in the
      PR (the R6 entry and the machine records cited).
- [ ] When that PR merges, this document's status becomes `E4 shipped, <date>; promotes with its parent` — never
      *implemented* while it sits in `todo/` — and the parent's E4 entry gets the same sentence. Parent and stories
      documents promote together, after E7.

## Plan-gate round 1 (2026-10-17)

The coai plan gate's first round over this document; judged by Opus. Accepted (applied above), with the re-split's
verdict on each:

1. The guard's discovery is an inventory from `git ls-files` covering every script root by extension, not
   hand-listed folders; the scanned list is verified against the repository — S4 *Discovery*, sc. 5, DoD. **Holds.**
2. The scratch peak is recomputed from the fixtures that can run at once: 0 + 2 + 7 MB (verified), ≈ 9 MB at
   `--test-threads=8`, 21 MB as the all-largest bound; not serialised, and said so — *Growth surface*. **Holds.**
3. S1 re-checks S2's constants and the sweep's decisions against the released contract before S2 merges. **Done by
   this re-split** — the table in § *S1*; the constants are the release's, three sweep decisions were not and are
   rewritten (D4, D5).
4. A removal the sweep attempts and teardown gives up on is recorded in `SweepReport.kept` with its reason — D5,
   S2 sc. 17. **Holds.**
5. `sweep()` reports every I/O error, the root's `read_dir` included, in `SweepReport.errors` and on stderr; a
   scenario injects a `read_dir` failure — D5, S2 sc. 19. **Holds.**
6. ~~The creation order is create → register → claim; an unmarked new-form folder is removed after the 10-minute
   minimum age.~~ **Superseded 2026-10-31:** the released rule §2 (`:64-76`) and `owned.mjs` (O:7-11, 612-646) claim
   FIRST with an exclusive create and then create the folder exclusively, so an unmarked owner-form folder is nobody's
   the helper knows and §4 lists it, never collects it — D4, D5, S2 sc. 3, 5, 17.
7. The discovery exclusion matches any `target*` directory (`target`, `target-wsl`, …) — S4 *Discovery*, sc. 5.
   **Holds.**

Rejected (still rejected):

- CI automation of the leftover count and the break-it — runners are ephemeral, so a count there proves nothing
  about a developer machine, and a planted violation in CI is a permanent defect the family forbids.
- Using a known-exited child's pid as the dead owner — the unmatchable-pid method is measured; pid reuse is the
  defect the contract bans.

## Plan-gate round 2 (2026-10-31)

The coai plan gate's round over the re-split, and the Opus judge's findings. All accepted and applied above; none
rejected:

1. (gate) The caller's SID on Windows comes from the THREAD token first, `OpenThreadToken(GetCurrentThread(),
   TOKEN_QUERY, true)`, and from the process token only on `ERROR_NO_TOKEN` — D3, S2 sc. 20.
2. (gate) Every stderr diagnostic takes `stderr().lock()` and writes each line in ONE `write_all`, so parallel tests
   never interleave — D4.3, D5, S2 sc. 19.
3. (gate) The guard scans its own file: detectors and fixtures are built from pieces, the frozen exclusion list is
   EMPTY and pinned — D7, S4 *Discovery*, sc. 1, 5, 6.
4. (judge, blocker) The Windows owner on an elevated runner is measured BEFORE the helper is coded (S2 step 0);
   `read_claim` takes `mine` as a seam; one Windows scenario asserts the measured answer; teardown needs no `is_mine`
   — D4, S2 sc. 0, 3, 8, 16, 20, DoD.
5. (judge, blocker) A candidate is a real directory, never a link or junction; the Windows handle is opened with
   `FILE_FLAG_OPEN_REPARSE_POINT` and a reparse point is never the caller's — D3, D5, S2 sc. 17.
6. (judge) A quarantine is restored exclusively — S's placeholder on Linux, Windows' refusal of a rename onto a
   directory asserted; while taken it stays listed with its command and every run retries — D5, S2 sc. 18.
7. (judge) `TestScratch` carries its state and the claim bytes it wrote; after the squatter path it is `Released`, and
   a `Claimed` value's `Drop` removes the claim only while it still holds those bytes — D4, S2 sc. 5.
8. (judge) The API block carries every seam the scenarios use (`Seams`, `Facts`, `Hooks`, `run_record_from`,
   `guarded`); the unobservable "the `Once` is not poisoned" claim is replaced by testing `guarded` directly — S2.
9. (judge) A refused quarantine rename is specified (kept, nothing moved); a removal that gives up in quarantine is
   renamed BACK, a recorded deviation from S — D5, S2 sc. 17.
10. (judge) The Windows unmatchable pid is measured first (`probe(own + 2)`), then taken from a ToolHelp snapshot
    (`Win32_System_Diagnostics_ToolHelp`); a pid above `u32::MAX` is `Unknown` there — D3, S2 sc. 14.
11. (judge) The grammar uses `[0-9]` and `(?-u:\s)`, never Unicode `\d`/`\s` — D3, S2 sc. 12 (vector r-i25).
12. (judge) The identity carries ctime; sc. 17 injects `now` instead of an mtime set back — D5, S2 sc. 17.
13. (judge) The fail-closed edges are stated: an unreadable identity never compares equal, the Linux probe is
    `Unknown` on any error but `NotFound`, the reader's own missing `ns` is unknown — D3, D5.
14. (judge) D7 no longer contradicts S4: only `docker-run` refuses any occurrence; `docker-name` and `shared-store`
    accept the owner form and the `lock:` exemption — D7.
15. (judge) 148 tests, not 147 — § *Facts*, S2 DoD, S3 sc. 4.
16. (judge) Workflows are scanned whole, line by line (no YAML parser — stricter); `#` comment skipping never applies
    to `.rs`; `<what>` is measured in BYTES (stricter than the rule's characters, said so) — D2, D7, S4.
17. (judge, trim taken) The script tracer reads only the languages this repository has (`.mjs`/`.js`/`.cjs`/`.ts` and
    workflows); the first `.sh`, `.ps1` or `.py` makes discovery red until the guard is extended — S4 *Discovery*,
    sc. 2, 5.

## Notes for the parent (`dew_flow_conventions · todo/PLAN_test_leftovers_rollout.md`, E4 entry)

The E4 entry is stale against the release; each line below is a correction to carry over:

1. *"Depends on E1"* → depends on the release `115bbf8` (the merged core, R5 of the remainder plan) and on R6: stage A
   of this repository is EMPTY (measured), the pin is PR #63 alone, and the producer stage is ONE squash-merged PR
   that merges only after every consumer's stage A is recorded and the barrier checklist has run per machine.
2. *"named `testsToDelete-sidecar-…`"* → `testsToDelete-<owner>-sidecar-<what>-<id>`; the old form is a `legacy-name`
   the guard refuses and the sweep lists.
3. *"Plus the model cache the audit named"* double-counts: it is `src/testing.rs:195`, one of the nine; no tenth or
   eleventh site exists on 1c1282e.
4. The seven "remove as a last statement" are six inline plus one fixture removed by its three callers; the
   pre-clean-only pair leak in 15 tests on every run, not only on failure.
5. The entry encodes rule §2's teardown but not §2's claim-first order, §3's owner token and two keys, or §4's
   start-of-run sweep; a killed run would leak until the janitor. S2 adds all of them.
6. *"One `TestScratch` type in `src/testing.rs`"* → `src/test_scratch.rs` (D1, a deviation: a second concern of ~400
   lines).
7. The guard as written ("`src/**/*.rs` under `#[cfg(test)]`, refuse a `temp_dir()` outside that file") is narrower
   than rule §7 (scripts and CI, eight classes, exemptions, the token's own row, a scan of nothing) and needs the
   mounted rule in CI — which `build-test` does not fetch today.
8. *"`#[should_panic]` … and, in a second test, asserts the path no longer exists"* cannot work under libtest's
   parallel, unordered execution; replaced by `catch_unwind` and a child-process harness.
9. The warning line `Drop` writes is swallowed by libtest's capture in a passing test unless written to
   `std::io::stderr()` directly; and an open handle blocks removal on Windows only with `share_mode(0)`.
10. `src/provider.rs:792` runs only in a CUDA build, which CI never makes.
11. Two documented deviations from the reference tools, both STRICTER, for the parent's record: the Windows kernel
    owner is profile AND owner SID (decision 2; `owned.mjs` checks the profile only — F16 in the conventions
    follow-ups), so an elevated run's folders fail closed; and `owner-blind` in Rust source takes no binding tracer
    (decision 3), a deviation from rule §7's "a bare name in script code".

## Deviations

Recorded when the plan was promoted, 2026-10-31. Where the contract and this document differ the contract won; each
line says what the plan said, what was built and the evidence.

**S2 — the helper (the S2 coder's, commits `d91ba05…7214662`)**

1. *Layout.* The plan said one `src/test_scratch.rs`; the helper is `src/test_scratch.rs` plus `src/test_scratch/{claim,
   identity,owner,scratch,sweep,win}.rs` and the scenarios in `src/test_scratch/tests/*.rs`, so no file passes 800
   lines (Rust doctrine §7). The guard's helper allowance is therefore the file AND the folder.
2. *The host class of the claim grammar.* The plan wrote `[^@:(?-u:\s)]+`, which is not valid regex syntax; the host is
   an explicit character class anchored `\A…\z`, and vector r-i30 (a host with a Unicode space, read by Node's `\s`)
   forced the explicit class.
3. *Sc. 4 uses two `п`*, not one: the byte room under `u1000` and the character count have to differ by enough to
   cross the 28-byte limit.
4. *Reasons follow `leftovers-sweep.mjs`.* A stranger's claim on a folder of this owner's name is `foreign-uid`
   with a `foreign:` line; "foreign host" is the reason text for a claim of another host.
5. *Quarantine restore for any owner of the token*, as in S (`leftovers-sweep.mjs`), not only this owner's.
6. *The post-rename identity check does not compare the change time*: a rename updates it (measured on tmpfs). The check
   before the rename already compared it; the later check is the device, inode and the claim.
7. *API additions the plan's block did not list:* `Hooks::after_remove`, a third argument of `remove_claim_if_still`,
   and `TestScratch::claim_id`.
8. *(Withdrawn in review round 1; the number stays so the others keep theirs.)* The shared vectors file's `verdicts`
   section was not exercised at first; `the_shared_verdict_vectors_agree` now runs v01-v26 through `owner_state` with a
   seam probe (item 35 below).

**S2 — corrections found by the first CI run of the draft PR (this commit set)**

9. *Windows restore.* The plan said "a plain `rename`, which refuses an existing directory (sc. 18 asserts it)". It does
   not: Rust's `std::fs::rename` on Windows replaces an existing EMPTY directory (replace-if-exists semantics, unlike
   .NET's `Directory.Move`). Sc. 18 failed on `windows-latest` (the folder was restored over somebody's empty directory
   and then judged `unmarked`) — a safety bug against rule §4 `:185-187`. The Windows restore is now `MoveFileExW` with
   no flags (`win::move_without_replace`), which refuses any existing entry. Proven by the next `windows-latest` run
   (not runnable on Linux; type-checked for `x86_64-pc-windows-msvc`).
10. *Sc. 3 reads the claim through the `mine` seam.* The `windows-latest` runner is elevated and its new files belong to
    `BUILTIN\Administrators`, so the REAL kernel-owner check says "not mine" and `read_claim` answers `Unreadable` — the
    measured, correct, fail-closed answer. The scenario asserts the record through the seam, the real read on POSIX, and
    on Windows that the real read agrees with `is_mine`. No other scenario reads a claim with the real `mine`.
11. *CI.* `build-test` fetches `.claude/rules/shared` on BOTH legs (the vectors file and the guard read it; the first
    run failed without it), and the `windows-latest` leg runs Step 0 alone with `--nocapture` in a Git Bash step so its
    SID and elevation lines reach the log (libtest captured them; a pwsh step would have `exit $LASTEXITCODE`
    appended, so the step's own check — that the measurement printed — decides it).

**S3**

12. *Sc. 1–3 were written against the old shapes and watched failing* before the move: `bge-seed-gone-<pid>`,
    `bge-log-<pid>-gone` and `bge-reg-panics-<pid>` still existed. After the move: green, and the 15 tests that
    leaked on every run leave nothing (no `bge-*` or `mxr-*` folder newer than the run; no `testsToDelete-*-sidecar-*`
    in the temp root).
13. *The log "absent" test* (`a_logs_directory_that_does_not_exist_yet_is_not_an_error`) names a root BELOW the empty
    scratch (`<scratch>/logs`): the scratch folder now exists, and the test is about a root that does not.
14. *`SeedScratch`* is a struct with `scratch`, `from` and `to` as the plan said; `model_cache_with_a_tokenizer`
    returns the `TestScratch` itself.
15. *Not run here:* the `--features cuda` test (`provider.rs`, compile-checked on the CPU leg only; an NVIDIA machine
    was not available), every `#[cfg(windows)]` scenario (type-checked for the msvc target, run by the next
    `windows-latest` job), the pre-existing old-form leftovers in the temp root (counted, not removed — D6; the owner's
    word is needed).

**S4**

16. *Layout.* `tests/test_leftovers.rs` is the target (the name the plan gives) and its parts sit in
    `tests/test_leftovers/*.rs` through `#[path]` (a test target's root file looks for children beside itself). The
    guard scans every one of them like any other file.
17. *`owner-blind` in scripts and workflows is the SHIPPED classifier, not a port.* The plan asked for its semantics;
    the guard runs `classifyLines` of the mounted `tools/lib/sweeper-scan.mjs` through `node` (the module path in
    `argv`, the files' text on stdin, a constant driver source — nothing interpolated). The answer is validated: it must
    parse and every finding must name a file that was sent and a line in `1..=count`; node missing, a crash, an
    out-of-range or malformed reply or a missing mount is ONE unjudged `owner-blind` finding (fail closed). A file whose
    text holds no prefix word is not sent (the classifier answers nothing for it by its first line), and with none
    that does node is not started. Both CI legs have node. Lessons carried over from the C# port of the same guard.
18. *`owner-blind` in Rust is wider than "a bare-prefix literal":* ANY spelling of the prefix word outside the helper's
    files not followed by an owner segment (globs, `^…[-_]` regexes, `"word" + "-"`, `format!` interpolation, a bare
    word), AND the helper's prefix constant, AND — as `raw-temp` — the helper's temp-root accessor used outside the
    helper (`create_dir(<accessor>().join("x"))` is a raw-temp bypass). `{PREFIX}{REPO_TOKEN}`-style builds are
    `legacy-name`. Allowed: the owner form, the `exempt`/`quarantine` words, the `.owner` label key and an
    upper-case `TESTSTODELETE_*` environment name. *Corrected in review round 1:* (a) "the helper's files" are
    `src/test_scratch.rs` and `src/test_scratch/` WITHOUT `src/test_scratch/tests.rs` and `src/test_scratch/tests/` —
    the helper's tests are its callers, and a first version let one of them make an unprefixed file in the raw temp
    root; (b) a line is READ, not only matched (item 26): escapes, one-character classes and joined pieces are decoded
    first, and a literal fragment of the word is a finding; (c) the second allowance is one pinned file,
    `tests/test_leftovers/words.rs`; (d) the guard is still a SPELLING check — pieces on different lines, a macro, an
    `include_str!` of a file it does not scan can hide the word.
19. *`dropped-scratch` is wider than the two shapes in the plan and takes no exemption:* also a bare
    `TestScratch::new(..);` statement, `TestScratch::new(..).path()` (a temporary), and a `SeedScratch { .. }` destructuring that does
    not bind `scratch`. Three S2 scenarios that call a constructor expected to panic inside `catch_unwind` now bind the
    value (`let _never_made = …`), and the one legacy name the sweep scenario builds at run time carries a
    `legacy-name` exemption.
20. *The token's identity comes from outside the helper.* The plan compared the helper's `REPO_TOKEN`/`REPO_ROW` with
    the table; a helper copied verbatim into another repository carries both, so that check passes there. The guard also
    maps the crate's package name in `Cargo.toml` (`bge-sidecar`) to the one token and row it may hold (`KNOWN`);
    another package is red.
21. *Smaller extensions of the detectors:* `raw-temp` also reads `var("TMPDIR")` without `env::`, a bare `temp_dir`, `env::temp_dir` as a path and
    `tempfile::`, `["']mktemp` in Rust/JS; `docker-run` also a quoted
    `"docker"` literal and `docker-compose`; `shared-store` also `deleteMany({})` and passes a `DROP`/`TRUNCATE` against a
    name of the owner form; unread languages also `.bash`, `.cmd`, `.bat`; the JavaScript family also `.mts`, `.cts`,
    `.tsx`, `.jsx`.
22. *Sc. 2 is several tests* (one per concern) rather than one; the fixtures are the mutation oracle. Sc. 6 (the
    break-its on the real tree) was done by hand and is recorded below, not a test (a planted violation in CI is a
    permanent defect the family forbids).
23. *Evidence.* Sc. 1 over the unmigrated tree: exactly nine `raw-temp` findings (`compile_cache.rs:372/429/498`,
    `log_segments.rs:204`, `provider.rs:792`, `testing.rs:171/199`, `tokens.rs:248/264`), nothing in the guard's own
    files and no `legacy-name`/`owner-blind` in the helper; after S3: zero. Every detector (48) replaced in turn by a
    never-matching pattern is killed by a fixture, and 9 wiring/logic mutants likewise. Break-its on the real tree: a
    raw temp call added to a test module and written whole in the guard's own file → red naming `file:line`; a legacy
    literal in `src/tokens.rs` → `legacy-name` and `owner-blind`, an exemption above clears only the first; an empty
    `scripts/x.sh` → the unread-language sentence; the mount moved aside → the submodule sentence; each reverted →
    green. The scanned inventory (54 files) equals `git ls-files --cached --others --exclude-standard` filtered by the
    same extensions.
24. *Promotion.* The Definition of Done said the plan becomes `E4 shipped, <date>; promotes with its parent` and is
    never *implemented* while in `todo/`. It was promoted to `research/` with the producer PR on the lead's
    instruction, with the parent's E4 entry to carry the same sentence when the PR merges.

**Review round 1 (E4), 2026-10-31 — what the first review of the producer PR changed**

One line per fix; the contract (`common/test-leftovers.md`, `tools/lib/owned.mjs`, `leftovers-sweep.mjs`,
`tools/janitor.sh`, `tools/fixtures/owner-vectors.json` at the pinned mount) won every disagreement. Every fix was
watched failing first, except where its line says otherwise; the Windows lines are compiled for
`x86_64-pc-windows-msvc` and run by the next `windows-latest` job only.

25. *Quarantine suffix is hex.* `quarantine_path` drew `[a-z0-9]` (2 of 1000 names matched the `[0-9a-f]+` of its own
    pattern, rule §4, `leftovers-sweep.mjs` and `janitor.sh`); it draws 4 random bytes as 8 lower-case hex characters
    (`random_hex`). Sc. 18 gains the producer's real name over 5000 draws and a sweep killed after its rename.
26. *The guard reads literals.* Three of twelve planted spellings were caught. `rustviews.rs` decodes escapes, spells a
    one-character class as the character and joins the pieces of `concat!`/`format!`/`+`; `bypass.rs` judges the
    decoded line, the joined pieces, literal fragments of the word (a whole literal of 5+ characters of it, or any
    literal holding 7), `/tmp` and `/var/tmp` literals, `concat!` inside `env::var`/`var_os`, `drop(<constructor>(..))`
    and a block whose tail is made from the path of the scratch it just bound. `bypass_fixtures.rs` plants every shape,
    built from the pieces of `words.rs`; with the reading switched off five of its six tests are red. Limit: a spelling
    check (item 18d). Three tests used `/tmp` or `/var/tmp` as text and one dropped a scratch at a block's end on
    purpose; they changed text and an explicit `drop`.
27. *The rename into quarantine never replaces.* `std::fs::rename` replaces an existing EMPTY directory on POSIX and on
    Windows. New `rename.rs::move_without_replace`: `renameat2(RENAME_NOREPLACE)` by the raw system call on Linux; on
    `EINVAL`/`ENOSYS` (a file system that does not know the flag) the name is RESERVED with an exclusive `mkdir` and the
    folder renamed over that placeholder of our own, which is removed again only while it is still the directory
    that call made (this is the choice for the no-flag case: a reservation, not a link); `MoveFileExW` with no flags on
    Windows. A collision draws a new name (five draws). The restore uses the same move. `Hooks::quarantine_name` is the
    seam; the test pre-creates an empty directory at the drawn name and finds it, the same one, afterwards.
28. *`win::caller_sid` is read per call* (it was a process-wide `OnceLock`, so the first thread decided for all):
    thread token first, process token on `ERROR_NO_TOKEN`. Windows test: a thread impersonating the anonymous logon
    answers `S-1-5-7` while its neighbours stay the process user.
29. *`ClaimPending`.* The squatter path set `Released` even when taking the claim back failed. It leaves
    `ClaimPending`, whose `Drop` retries; every claim removal (the `Claimed` state's too) is the five tries 100 ms apart.
30. *The quarantine pass applies `owner_state`'s pid checks* (0, a leading zero, past 2^53, Windows' pid 1, a Windows pid
    past `u32::MAX`) before it probes: `pid_is_askable`, shared with `owner_state`.
31. *Windows: one handle.* `is_mine` resolves the profile test with `GetFinalPathNameByHandleW` on the handle that was
    judged (no second open by `canonicalize`); identity is `FILE_ID_INFO` (128 bits: ReFS, Dev Drive), not `nFileIndex`;
    `read_limited` compares the opened handle's identity with the expected one (it ignored it) and requires
    `GetFileType == FILE_TYPE_DISK` and neither a directory nor a reparse point before any read.
32. *POSIX `read_limited`.* The `fstat` of the opened handle (`S_ISREG`, the expected device and inode) was already
    there and untested; it is tested now (a swapped file, a FIFO that must not block) and the test fails without it.
33. *`remove_claim_if_still` compares the whole identity* (device, inode, change time) between its read and its unlink.
34. *The helper's tests are not the helper* (item 18a): Step 0 makes its probe file inside `TestScratch::new("step0")`.
35. *The `verdicts` section runs* (replaces item 8): v01-v26 through `owner_state`, for a Linux and a Windows reader;
    three mutants (pid 0 askable, Windows pid 1 askable, the 2 s tolerance widened) each turn it red. It passed at once.
36. *Windows tests.* Scenario 20 asserts the helper against `measure_owner` (Step 0's own token, handle and
    `GetSecurityInfo`, SIDs compared as strings), not `owner_sid_of`/`sid_equal` again; the POSITIVE real-SID case (the
    owner of a folder and its claim set to the caller's user SID, then the real `is_mine` is true and the sweep removes
    a dead owner's folder); a `share_mode(0)` file inside a folder refuses the rename (kept `could not remove`, nothing
    moved) and the next sweep removes it; a 128-bit identity and final-path test; a junction is never mine; the
    dead-pid pick repeats once with fresh ToolHelp snapshots.
37. *The orphan-claim pass skips a claim whose folder sits in a quarantine* (a sibling quarantine in the listing, or a
    `quarantine-conflict` of this sweep): kept `quarantined`. A killed sweep's claim was removed as an orphan's.
38. *One listing.* The root is listed once and the three passes share the snapshot (a restore updates it; every
    candidate is still re-checked with its own `symlink_metadata`); an entry this sweep removed is skipped, not an
    `errors` entry.
39. *`check_what`'s message* says "multi-byte characters count by bytes" (with the character count) when the two differ.

**Review round 2 (E4), 2026-10-31**

40. *Review round 2: 3 small fixes; 5 findings rejected with reasons in the gate record.* The sweep's passes hold the one snapshot by handle instead of cloning it; the claim-removal stderr line says it is not fatal and that the next run's sweep collects it; the quarantine suffix and `random_id` share one random-bytes helper.
