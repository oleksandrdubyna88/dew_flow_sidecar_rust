//! Local cargo builds and test runs take at most 8 workers and run at lowered CPU priority — the
//! family rule `common.local-cpu-budget` (`.claude/rules/shared/common/local-cpu-budget.md`).
//!
//! Measured 2026-10-10 on a 24-thread workstation running several agent sessions: a runner left at
//! its default takes one worker per CPU and holds the machine at 100 %. Measured here 2026-10-11 with
//! a 20-test probe crate: libtest ran 20 tests at once by default and 8 with the cap.
//!
//! The cap lives in the DOCUMENTED LOCAL COMMANDS, not in `.cargo/config.toml`: both `jobs` and
//! `RUST_TEST_THREADS` are exact values that would also apply to CI, where they would RAISE a 4-vCPU
//! runner to 8 — and CI is exempt. So this test reads the commands people and agents copy.

use std::path::Path;

const FAMILY_CAP: u32 = 8;

/// Every command line inside a fenced block of `file` that builds or tests with cargo.
fn cargo_commands(file: &str) -> Vec<String> {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join(file);
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let mut fenced = false;
    let mut out = Vec::new();
    for line in text.lines() {
        let trimmed = line.trim_start();
        if trimmed.starts_with("```") {
            fenced = !fenced;
            continue;
        }
        let command = trimmed.split('#').next().unwrap_or("").trim();
        if fenced && (command.contains("cargo build") || command.contains("cargo test")) {
            out.push(command.to_string());
        }
    }
    out
}

/// The value of `-j N` / `--jobs N` / `--test-threads=N` in a command, if present.
fn flag(command: &str, names: &[&str]) -> Option<u32> {
    let tokens: Vec<&str> = command.split_whitespace().collect();
    tokens.iter().enumerate().find_map(|(i, token)| {
        names
            .iter()
            .find_map(|name| match token.strip_prefix(name) {
                Some("") => tokens.get(i + 1).and_then(|v| v.parse().ok()),
                Some(rest) => rest.strip_prefix('=').and_then(|v| v.parse().ok()),
                None => None,
            })
    })
}

fn assert_capped_and_lowered(file: &str, command: &str) {
    let wsl = command.contains("target-wsl");
    let lowered = if wsl {
        command.starts_with("nice -n 19 ")
    } else {
        command.starts_with("node .claude/rules/shared/tools/low-priority.mjs ")
    };
    assert!(lowered, "{file}: `{command}` is not run at lowered priority (low-priority.mjs; `nice -n 19` inside WSL)");
    let jobs = flag(command, &["-j", "--jobs"]);
    assert!(
        matches!(jobs, Some(n) if (1..=FAMILY_CAP).contains(&n)),
        "{file}: `{command}` has no -j of {FAMILY_CAP} or fewer"
    );
    if command.contains("cargo test") {
        let threads = flag(command, &["--test-threads"]);
        assert!(
            matches!(threads, Some(n) if (1..=FAMILY_CAP).contains(&n)),
            "{file}: `{command}` has no --test-threads of {FAMILY_CAP} or fewer"
        );
    }
}

#[test]
fn every_documented_cargo_build_and_test_is_capped_and_lowered() {
    let mut builds = 0;
    let mut tests = 0;
    for file in ["CLAUDE.md", "README.md"] {
        for command in cargo_commands(file) {
            if command.contains("cargo test") {
                tests += 1
            } else {
                builds += 1
            }
            assert_capped_and_lowered(file, &command);
        }
    }
    // A scan that finds nothing passes for ever.
    assert!(
        builds > 0 && tests > 0,
        "the scan no longer finds the documented commands ({builds} builds, {tests} tests)"
    );
}

#[test]
fn the_flag_reader_reads_every_spelling() {
    assert_eq!(flag("cargo build -j 8", &["-j", "--jobs"]), Some(8));
    assert_eq!(flag("cargo build --jobs=4", &["-j", "--jobs"]), Some(4));
    assert_eq!(
        flag("cargo test -- --test-threads=8", &["--test-threads"]),
        Some(8)
    );
    assert_eq!(flag("cargo build --release", &["-j", "--jobs"]), None);
}
