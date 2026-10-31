//! The structural guard of `common/test-leftovers.md` §7 for this crate: no test, script or workflow here creates a
//! temp folder, worktree, container or store resource outside `TestScratch` (or under a valid inline exemption), no
//! source spells a legacy name or an owner-blind match, the helper's token is THIS repository's registered row, and
//! a scan of nothing is red.
//!
//! It reads files through `CARGO_MANIFEST_DIR`, like `local_cpu_budget.rs`, so it runs on both CI legs; the mounted
//! rules (`git submodule update --init .claude/rules/shared`) are read by two of its parts and a missing mount is
//! red, never a skipped check. The parts live beside this file in `test_leftovers/`:
//!
//! - `discovery` — an inventory from `git ls-files`, four path exclusions, a language it does not read is red, a
//!   scan of nothing is red; the exclusion list by file name is EMPTY and pinned.
//! - `classes` — the classes as pure functions over `(relative path, text)`. `owner-blind` in Rust source takes no
//!   binding tracer (decision 3 of the E4 plan): any spelling of the prefix word outside the helper's files that is
//!   not followed by an owner segment is a finding.
//! - `scripts` — `owner-blind` in scripts and workflows is the SHIPPED classifier's verdict, validated, fail closed.
//! - `token` — `REPO_TOKEN`/`REPO_ROW` against the rule's table and against an identity from outside the helper.
//! - `words` — the words, in pieces: the guard scans its own files, and the machine-wide barrier scan reads them.
//!
//! No test here creates a folder or runs a generated script: this is an integration test of a binary crate and
//! cannot reach `TestScratch`, and a second temp helper here would be the copy the guard exists to refuse.

// The parts sit beside this file in `test_leftovers/`; `#[path]` because a test target's root file looks for its
// children next to itself, not in a folder of its own name (and the target keeps the file name the plan gives it).
#[path = "test_leftovers/classes.rs"]
mod classes;
#[path = "test_leftovers/discovery.rs"]
mod discovery;
#[path = "test_leftovers/repo.rs"]
mod repo;
#[path = "test_leftovers/scripts.rs"]
mod scripts;
#[path = "test_leftovers/token.rs"]
mod token;
#[path = "test_leftovers/words.rs"]
mod words;

#[cfg(test)]
#[path = "test_leftovers/class_fixtures.rs"]
mod class_fixtures;
#[cfg(test)]
#[path = "test_leftovers/discovery_fixtures.rs"]
mod discovery_fixtures;
#[cfg(test)]
#[path = "test_leftovers/script_fixtures.rs"]
mod script_fixtures;
#[cfg(test)]
#[path = "test_leftovers/token_fixtures.rs"]
mod token_fixtures;
#[cfg(test)]
#[path = "test_leftovers/wiring_fixtures.rs"]
mod wiring_fixtures;

use std::path::Path;

/// The crate root: where `git ls-files` runs and every path below is relative to.
fn root() -> &'static Path {
    Path::new(env!("CARGO_MANIFEST_DIR"))
}

/// Sc. 1 — the real tree. Every class, the shipped classifier over scripts and workflows, the unread languages, the
/// scan of nothing and the token, over the files git lists. `--nocapture` prints the inventory it scanned.
#[test]
fn this_repository_creates_nothing_outside_test_scratch() {
    let scan = repo::scan_repository(root());
    println!(
        "test_leftovers: scanned {} file(s): {}",
        scan.scanned.len(),
        scan.scanned.join(", ")
    );
    println!("test_leftovers: {} finding(s)", scan.findings.len());
    assert!(scan.findings.is_empty(), "{}", repo::render(&scan.findings));
}
