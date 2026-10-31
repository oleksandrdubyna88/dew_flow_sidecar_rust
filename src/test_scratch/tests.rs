//! The scenarios of the E4 plan, S2 (`research/PLAN_test_leftovers_E4.md`), numbered as the plan numbers them.
//!
//! Every scenario that counts or lists entries does so in a PRIVATE parent made with `new_in`, because eight test
//! threads create scratches in the temp root at once. A legacy or foreign name a scenario needs is built at run time
//! from `PREFIX` and pieces, never spelled whole: S4's guard reads this file too.

use super::*;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::path::PathBuf;
use std::time::Duration;

mod owner_scenarios;
mod scratch_scenarios;
mod sweep_edges;
mod sweeping;
mod windows_only;

// ---------- shared helpers ----------

fn panics<R>(f: impl FnOnce() -> R) -> bool {
    catch_unwind(AssertUnwindSafe(f)).is_err()
}

// ---------- helpers for the folder scenarios ----------

use std::collections::BTreeSet;
use std::io;

fn name_with(what: &str, id: &str) -> String {
    name_for(what, id)
}

/// The names in `dir`, sorted — what a scenario means by "the parent is empty".
fn entries(dir: &Path) -> BTreeSet<String> {
    std::fs::read_dir(dir)
        .expect("a readable parent")
        .map(|entry| {
            entry
                .expect("an entry")
                .file_name()
                .to_string_lossy()
                .into_owned()
        })
        .collect()
}

fn panic_text(f: impl FnOnce()) -> String {
    let payload = catch_unwind(AssertUnwindSafe(f)).expect_err("it must panic");
    payload
        .downcast_ref::<String>()
        .cloned()
        .or_else(|| payload.downcast_ref::<&str>().map(|s| s.to_string()))
        .unwrap_or_default()
}

use std::path::Path;

fn sequence(ids: &[&str]) -> impl FnMut() -> String {
    let mut ids: std::collections::VecDeque<String> = ids.iter().map(|id| id.to_string()).collect();
    move || {
        ids.pop_front()
            .expect("the scenario drew more ids than it gave")
    }
}

// ---------- 7 ----------

/// The child of scenarios 7 and 8: `#[ignore]`d, it does nothing unless the harness sets `TEST_SCRATCH_CHILD`.
#[test]
#[ignore = "the child of scenarios 7 and 8; run only by them"]
fn child_scratch() {
    let Ok(mode) = std::env::var("TEST_SCRATCH_CHILD") else {
        return;
    };
    let parent = PathBuf::from(std::env::var("TEST_SCRATCH_PARENT").expect("the parent"));
    let scratch = TestScratch::new_in(&parent, "child");
    // A leading newline: libtest has already printed `test ... ` on this line.
    println!("\nSCRATCH {}", scratch.path().display());
    println!("PID {}", std::process::id());
    match mode.as_str() {
        "assert" => assert!(mode != "assert", "the child fails a real assertion"),
        "abort" => std::process::abort(),
        other => panic!("unknown child mode {other}"),
    }
}

/// Run `child_scratch` as its own test process in `mode`; returns its exit status, the scratch path it printed and its pid.
fn run_child(mode: &str, parent: &Path) -> (std::process::ExitStatus, PathBuf, u64) {
    run_child_with(mode, parent, &[])
}

/// `run_child` with extra environment variables for the child (a shared `TESTSTODELETE_RUN`, say).
fn run_child_with(
    mode: &str,
    parent: &Path,
    envs: &[(&str, &str)],
) -> (std::process::ExitStatus, PathBuf, u64) {
    let output = std::process::Command::new(std::env::current_exe().expect("this test binary"))
        .args([
            "test_scratch::tests::child_scratch",
            "--exact",
            "--ignored",
            "--nocapture",
            "--test-threads=1",
        ])
        .env("TEST_SCRATCH_CHILD", mode)
        .env("TEST_SCRATCH_PARENT", parent)
        .envs(envs.iter().copied())
        .output()
        .expect("the child runs");
    let stdout = String::from_utf8_lossy(&output.stdout);
    let path = stdout
        .lines()
        .find_map(|line| line.strip_prefix("SCRATCH "))
        .unwrap_or_else(|| {
            panic!(
                "the child printed no path:\n{stdout}\n{}",
                String::from_utf8_lossy(&output.stderr)
            )
        });
    let pid = stdout
        .lines()
        .find_map(|line| line.strip_prefix("PID "))
        .and_then(|digits| digits.trim().parse().ok())
        .expect("the child printed its pid");
    (output.status, PathBuf::from(path), pid)
}
