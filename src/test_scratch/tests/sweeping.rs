//! Scenarios 8, 17, 18 and 19 of the E4 plan, S2: the start-of-run sweep.
//!
//! Every scenario sweeps a PRIVATE root with `now` injected, a probe that says which pids are gone, and a `mine` seam
//! (a second uid cannot be had in a test). Names a scenario needs for another owner, another repository or the legacy
//! form are built from `PREFIX` and pieces at run time: S4's guard reads this file too.

use super::*;
use std::time::SystemTime;

/// A pid no live process of this test has: the probe says `Gone` for it, so a record naming it is abandoned.
const GONE_PID: u64 = 999_001;

pub(super) fn later() -> SystemTime {
    SystemTime::now() + Duration::from_secs(11 * 60)
}

/// The probe of these scenarios: `GONE_PID` is gone, this process is alive, everything else cannot be asked.
pub(super) fn probe_of(pid: u64) -> Probe {
    if pid == GONE_PID {
        Probe::Gone
    } else if pid == this_owner().pid {
        Probe::Alive {
            start: this_owner().start,
        }
    } else {
        Probe::Unknown
    }
}

fn other_owner() -> &'static str {
    if owner_token() == "u7" {
        "u8"
    } else {
        "u7"
    }
}

fn record_text(owner: &Owner) -> String {
    format!("{}\n", format_owner(owner))
}

/// A record of this machine and namespace for `pid`, written the way a helper writes one.
fn record_for(pid: u64) -> Owner {
    Owner {
        pid,
        ..this_owner().clone()
    }
}

/// A folder named `name` in `root`, claimed with `claim` (none: unmarked).
pub(super) fn folder(root: &Path, name: &str, claim: Option<&str>) -> PathBuf {
    let dir = root.join(name);
    std::fs::create_dir(&dir).unwrap();
    if let Some(text) = claim {
        std::fs::write(claim_path(&dir), text).unwrap();
    }
    dir
}

pub(super) fn gone_claim() -> String {
    record_text(&record_for(GONE_PID))
}

pub(super) fn run_sweep(root: &Path, now: SystemTime, hooks: &mut Hooks<'_>) -> SweepReport {
    sweep(root, now, SWEEP_MIN_AGE, probe_of, |_| true, hooks)
}

fn kept<'a>(report: &'a SweepReport, path: &Path) -> Option<&'a str> {
    report
        .kept
        .iter()
        .find(|(kept, _)| kept == path)
        .map(|(_, reason)| reason.as_str())
}

fn listed(report: &SweepReport, needle: &str) -> bool {
    report.listed.iter().any(|line| line.contains(needle))
}

// ---------- 8 ----------

#[test]
fn a_killed_test_leaves_its_scratch_claimed_and_the_next_sweep_collects_it() {
    let private = TestScratch::new("killed-parent");
    let (status, path, child_pid) = run_child("abort", private.path());
    assert!(!status.success(), "the child aborts");
    assert!(
        path.is_dir() && claim_path(&path).is_file(),
        "no unwind, no Drop: both remain"
    );

    let Claim::Owner(record) = read_claim(&path, &|_| true) else {
        panic!("the child's claim must be readable");
    };
    assert_eq!(record.owner, owner_token());
    assert_eq!(record.pid, child_pid);
    assert_eq!(record.host, this_owner().host);
    #[cfg(target_os = "linux")]
    assert!(
        record.start.is_some() && record.ns.is_some(),
        "a start and a namespace on Linux: {record:?}"
    );

    // The REAL probe is never asked about a dead child's pid: Windows hands exited pids to the next process.
    let alive = |pid: u64| {
        if pid == child_pid {
            Probe::Alive {
                start: record.start,
            }
        } else {
            Probe::Unknown
        }
    };
    let report = sweep(
        private.path(),
        later(),
        SWEEP_MIN_AGE,
        alive,
        |_| true,
        &mut Hooks::default(),
    );
    assert_eq!(kept(&report, &path), Some("live"), "{report:?}");
    assert!(path.is_dir() && claim_path(&path).is_file());

    let gone = |pid: u64| {
        if pid == child_pid {
            Probe::Gone
        } else {
            Probe::Unknown
        }
    };
    let report = sweep(
        private.path(),
        later(),
        SWEEP_MIN_AGE,
        gone,
        |_| true,
        &mut Hooks::default(),
    );
    assert_eq!(report.removed, std::slice::from_ref(&path), "{report:?}");
    assert!(
        entries(private.path()).is_empty(),
        "no folder, no claim, no quarantine: {:?}",
        entries(private.path())
    );
}

// ---------- 17 ----------

#[test]
fn the_sweep_removes_only_what_a_gone_owner_left_and_lists_the_rest() {
    let scratch = TestScratch::new("sweep-root");
    let root = scratch.path();
    let named = |what: &str| name_with(what, "abc123");

    let live = folder(root, &named("live"), Some(&record_text(this_owner())));
    let stranger_claim = folder(
        root,
        &named("u7claim"),
        Some(&record_text(&Owner {
            owner: other_owner().into(),
            ..record_for(GONE_PID)
        })),
    );
    let foreign_host = folder(
        root,
        &named("host"),
        Some(&record_text(&Owner {
            host: "elsewhere@linux".into(),
            ..record_for(GONE_PID)
        })),
    );
    let garbage = folder(root, &named("garbage"), Some("added with --lock\n"));
    let not_mine = folder(root, &named("notmine"), Some(&gone_claim()));
    let gone = folder(root, &named("gone"), Some(&gone_claim()));
    let upper = folder(
        root,
        &named("up").to_uppercase().replace("ABC123", "abc123"),
        Some(&gone_claim()),
    );
    let underscored = folder(root, &named("under").replace('-', "_"), Some(&gone_claim()));
    let unmarked = folder(root, &named("unmarked"), None);
    let with_git = folder(root, &named("gitfile"), Some(&gone_claim()));
    std::fs::write(with_git.join(".git"), "gitdir: /elsewhere\n").unwrap();
    let swapped = folder(root, &named("swapped"), Some(&gone_claim()));
    let into_quarantine = folder(root, &named("raced"), Some(&gone_claim()));

    let foreign = format!("{PREFIX}{}-{REPO_TOKEN}-x-abc123", other_owner());
    let foreign_dir = folder(root, &foreign, Some(&gone_claim()));
    let legacy = format!("{PREFIX}{REPO_TOKEN}-x-abc123");
    let legacy_dir = folder(root, &legacy, Some("devbox@linux:1\n"));
    let other_token = folder(
        root,
        &format!("{PREFIX}{}-mcp-x-abc123", other_owner()),
        None,
    );
    let not_ours = folder(root, "bge-log-1-first", None);

    let not_mine_name = named("notmine");
    let swapped_name = named("swapped");
    let raced_name = named("raced");
    let aside = root.join("swapped-aside");
    let mut after_capture = |path: &Path| {
        if path.ends_with(&swapped_name) {
            // The folder is replaced by a fresh one of the same name between the capture and the removal; the old
            // one is kept alive beside so the fresh one cannot reuse its inode number.
            std::fs::rename(path, &aside).unwrap();
            std::fs::create_dir(path).unwrap();
        }
    };
    let mut moved = Vec::new();
    let mut after_rename = |_: &Path, quarantine: &Path| {
        moved.push(
            quarantine
                .file_name()
                .unwrap()
                .to_string_lossy()
                .into_owned(),
        );
        if quarantine
            .file_name()
            .is_some_and(|n| n.to_string_lossy().contains(&raced_name))
        {
            std::fs::write(quarantine.join("late"), b"x").unwrap();
        }
    };
    let mut hooks = Hooks {
        after_capture: Some(&mut after_capture),
        after_rename: Some(&mut after_rename),
        after_remove: None,
    };
    let report = sweep(
        root,
        later(),
        SWEEP_MIN_AGE,
        probe_of,
        |path| !path.ends_with(&not_mine_name),
        &mut hooks,
    );
    assert!(
        !moved.iter().any(|name| name.contains(&swapped_name)),
        "the identity is re-checked BEFORE the rename: a folder that was replaced is never moved ({moved:?})"
    );

    assert_eq!(
        kept(&report, &live),
        Some("live"),
        "claimed by this process; age never authorises\n{report:?}"
    );
    assert_eq!(
        kept(&report, &stranger_claim),
        Some("unknown"),
        "a claim of another owner"
    );
    assert_eq!(
        kept(&report, &foreign_host),
        Some("foreign host"),
        "a claim of another machine"
    );
    assert_eq!(kept(&report, &garbage), Some("unreadable"));
    assert_eq!(kept(&report, &not_mine), Some("foreign-uid"));
    assert!(
        listed(&report, &format!("foreign: folder {not_mine_name}")),
        "{:?}",
        report.listed
    );
    assert_eq!(
        kept(&report, &unmarked),
        Some("unmarked"),
        "unmarked and old is NEVER collected"
    );
    assert!(listed(
        &report,
        &format!("unmarked: folder {}", named("unmarked"))
    ));
    assert!(unmarked.is_dir());
    assert_eq!(kept(&report, &with_git), Some("other clone"));
    assert_eq!(kept(&report, &swapped), Some("replaced"), "{report:?}");
    assert!(
        swapped.is_dir() && claim_path(&swapped).is_file(),
        "the fresh folder and its claim are untouched"
    );
    assert!(aside.is_dir(), "and so is the one moved aside");

    for removed in [&gone, &upper, &underscored, &into_quarantine] {
        assert!(
            report.removed.contains(removed),
            "{removed:?} must be removed: {report:?}"
        );
        assert!(
            !removed.exists() && !claim_path(removed).exists(),
            "{removed:?}"
        );
    }
    assert!(
        !entries(root).iter().any(|name| name.starts_with('.')),
        "no quarantine is left behind: {:?}",
        entries(root)
    );

    assert!(
        listed(
            &report,
            &format!("foreign: folder {foreign} — owner {}", other_owner())
        ),
        "{:?}",
        report.listed
    );
    assert!(
        foreign_dir.is_dir() && claim_path(&foreign_dir).is_file(),
        "another owner's scope is never touched"
    );
    #[cfg(unix)]
    {
        let command = format!(
            "legacy: folder {legacy} — remove with: rm -rf -- '{}' '{}'",
            legacy_dir.display(),
            claim_path(&legacy_dir).display()
        );
        assert!(
            report.listed.contains(&command),
            "{:?}\nwanted {command}",
            report.listed
        );
    }
    #[cfg(windows)]
    assert!(
        listed(
            &report,
            &format!("legacy: folder {legacy} — remove with: Remove-Item")
        ),
        "{:?}",
        report.listed
    );
    assert!(
        legacy_dir.is_dir(),
        "a legacy name is listed, never collected"
    );
    for untouched in [&other_token, &not_ours] {
        assert!(untouched.is_dir());
        assert!(
            kept(&report, untouched).is_none()
                && !report.listed.iter().any(|l| l.contains(
                    &untouched
                        .file_name()
                        .unwrap()
                        .to_string_lossy()
                        .into_owned()
                )),
            "{untouched:?} is not this token's business"
        );
    }

    // Unmarked and YOUNG (the real clock): untouched and unreported.
    let young = run_sweep(root, SystemTime::now(), &mut Hooks::default());
    assert!(
        kept(&young, &unmarked).is_none() && !listed(&young, "unmarked"),
        "{young:?}"
    );

    // Age never authorises: ten days on, a folder claimed by this very process is still kept.
    let far = SystemTime::now() + Duration::from_secs(10 * 24 * 3600);
    let report = run_sweep(root, far, &mut Hooks::default());
    assert_eq!(kept(&report, &live), Some("live"));
    assert!(live.is_dir());
    std::fs::remove_dir_all(&aside).unwrap();
}

#[test]
fn a_claim_changed_after_the_capture_stops_the_removal() {
    let scratch = TestScratch::new("claim-change");
    let root = scratch.path();
    let rewritten = folder(root, &name_with("rewritten", "abc123"), Some(&gone_claim()));
    let replaced = folder(root, &name_with("replaced", "abc123"), Some(&gone_claim()));
    let (rewritten_name, replaced_name) = (
        name_with("rewritten", "abc123"),
        name_with("replaced", "abc123"),
    );

    let mut after_capture = |path: &Path| {
        if path.ends_with(&rewritten_name) {
            // The verdict was made on a claim that said `gone`; the claim now names a live process.
            std::fs::write(claim_path(path), record_text(this_owner())).unwrap();
        }
        if path.ends_with(&replaced_name) {
            // The same TEXT, but another file: removed and written again, so its identity is not the captured one.
            // The old file is kept beside, so the new one cannot reuse its inode number.
            let text = std::fs::read(claim_path(path)).unwrap();
            std::fs::rename(claim_path(path), path.with_extension("aside")).unwrap();
            std::fs::write(claim_path(path), text).unwrap();
        }
    };
    let mut hooks = Hooks {
        after_capture: Some(&mut after_capture),
        ..Hooks::default()
    };
    let report = run_sweep(root, later(), &mut hooks);
    for dir in [&rewritten, &replaced] {
        assert_eq!(kept(&report, dir), Some("replaced"), "{report:?}");
        assert!(
            dir.is_dir() && claim_path(dir).is_file(),
            "{dir:?} must be untouched"
        );
    }
    assert!(report.removed.is_empty());
    assert!(
        !entries(root).iter().any(|name| name.starts_with('.')),
        "nothing was moved: {:?}",
        entries(root)
    );
    std::fs::remove_file(replaced.with_extension("aside")).unwrap();
}

#[test]
fn a_claim_changed_after_the_rename_or_the_removal_is_never_removed() {
    let scratch = TestScratch::new("claim-late");
    let root = scratch.path();
    let after_rename_dir = folder(root, &name_with("later", "abc123"), Some(&gone_claim()));
    std::fs::write(after_rename_dir.join("content"), b"x").unwrap();
    let after_remove_dir = folder(root, &name_with("lastly", "abc123"), Some(&gone_claim()));
    let (rename_name, remove_name) = (name_with("later", "abc123"), name_with("lastly", "abc123"));

    // Between the rename and the second check, the claim is rewritten: the folder is renamed BACK, untouched.
    let mut after_rename = |dir: &Path, _: &Path| {
        if dir.ends_with(&rename_name) {
            std::fs::write(claim_path(dir), record_text(this_owner())).unwrap();
        }
    };
    // After the folder is gone, another run's claim takes the name: it is not the captured one and is NOT removed.
    let mut after_remove = |dir: &Path| {
        if dir.ends_with(&remove_name) {
            std::fs::remove_file(claim_path(dir)).unwrap();
            std::fs::write(claim_path(dir), record_text(this_owner())).unwrap();
        }
    };
    let mut hooks = Hooks {
        after_rename: Some(&mut after_rename),
        after_remove: Some(&mut after_remove),
        ..Hooks::default()
    };
    let report = run_sweep(root, later(), &mut hooks);

    assert_eq!(
        kept(&report, &after_rename_dir),
        Some("replaced"),
        "{report:?}"
    );
    assert!(
        after_rename_dir.join("content").is_file(),
        "renamed back with its content"
    );
    assert!(claim_path(&after_rename_dir).is_file());
    assert!(
        !entries(root).iter().any(|name| name.starts_with('.')),
        "no quarantine is left: {:?}",
        entries(root)
    );

    assert!(report.removed.contains(&after_remove_dir), "{report:?}");
    assert!(!after_remove_dir.exists());
    assert_eq!(
        std::fs::read_to_string(claim_path(&after_remove_dir)).unwrap(),
        record_text(this_owner()),
        "a claim that is no longer the captured one is not ours to remove"
    );
}

#[test]
fn a_link_is_never_a_sweep_candidate() {
    let scratch = TestScratch::new("link-root");
    let root = scratch.path();
    let target = root.join("target-folder");
    std::fs::create_dir(&target).unwrap();
    std::fs::write(target.join("precious"), b"x").unwrap();

    let link = root.join(name_with("link", "abc123"));
    #[cfg(unix)]
    std::os::unix::fs::symlink(&target, &link).unwrap();
    #[cfg(windows)]
    {
        let made = std::process::Command::new("cmd")
            .args(["/C", "mklink", "/J"])
            .arg(&link)
            .arg(&target)
            .output()
            .expect("cmd runs");
        assert!(made.status.success(), "mklink /J failed: {made:?}");
    }
    std::fs::write(claim_path(&link), gone_claim()).unwrap();

    let report = run_sweep(root, later(), &mut Hooks::default());
    assert_eq!(kept(&report, &link), Some("not a directory"), "{report:?}");
    assert!(
        std::fs::symlink_metadata(&link).is_ok(),
        "the link is still there"
    );
    assert!(
        target.join("precious").is_file(),
        "the target and its file are intact"
    );
    assert!(report.removed.is_empty());
}

#[test]
fn orphan_claims_are_judged_by_their_owner() {
    let scratch = TestScratch::new("orphan-root");
    let root = scratch.path();
    let orphan = |what: &str, text: &str| {
        let dir = root.join(name_with(what, "abc123"));
        std::fs::write(claim_path(&dir), text).unwrap();
        claim_path(&dir)
    };
    let live = orphan("live", &record_text(this_owner()));
    let gone = orphan("gone", &gone_claim());
    let foreign_host = orphan(
        "host",
        &record_text(&Owner {
            host: "elsewhere@linux".into(),
            ..record_for(GONE_PID)
        }),
    );
    let garbage = orphan("garbage", "nonsense\n");

    let report = run_sweep(root, later(), &mut Hooks::default());
    assert_eq!(report.removed, std::slice::from_ref(&gone), "{report:?}");
    assert!(!gone.exists());
    assert_eq!(kept(&report, &live), Some("live"));
    assert_eq!(kept(&report, &foreign_host), Some("foreign host"));
    assert_eq!(kept(&report, &garbage), Some("unreadable"));
    for kept_claim in [&live, &foreign_host, &garbage] {
        assert!(kept_claim.is_file());
    }

    let young = run_sweep(root, SystemTime::now(), &mut Hooks::default());
    assert!(
        young.removed.is_empty() && young.kept.is_empty(),
        "a young orphan is not considered: {young:?}"
    );

    let stranger = format!("{PREFIX}{}-{REPO_TOKEN}-orphan-abc123.owner", other_owner());
    std::fs::write(root.join(&stranger), gone_claim()).unwrap();
    let report = run_sweep(root, later(), &mut Hooks::default());
    assert!(
        listed(&report, &format!("foreign: claim {stranger}")),
        "{:?}",
        report.listed
    );
    assert!(root.join(&stranger).is_file());
}

#[cfg(unix)]
#[test]
fn a_removal_that_teardown_gives_up_on_is_in_kept_and_nothing_is_lost() {
    use std::os::unix::fs::PermissionsExt;
    let mode = |path: &Path, bits: u32| {
        std::fs::set_permissions(path, std::fs::Permissions::from_mode(bits)).unwrap()
    };
    struct Restore(PathBuf);
    impl Drop for Restore {
        fn drop(&mut self) {
            let _ = std::fs::set_permissions(&self.0, std::fs::Permissions::from_mode(0o755));
        }
    }
    let premise = |dir: &Path| {
        assert!(
            std::fs::File::create(dir.join("probe")).is_err(),
            "this process can write into a 0o555 folder (running as root?) — the scenario cannot prove anything here"
        )
    };

    // 1. The private root is read-only: the quarantine rename is REFUSED — nothing moved.
    let scratch = TestScratch::new("giveup-root");
    let root = scratch.path();
    let refused = folder(root, &name_with("refused", "abc123"), Some(&gone_claim()));
    let _restore_root = Restore(root.to_path_buf());
    mode(root, 0o555);
    premise(root);
    let report = run_sweep(root, later(), &mut Hooks::default());
    let reason = kept(&report, &refused).unwrap_or_else(|| panic!("{report:?}"));
    assert!(reason.starts_with("could not remove:"), "{reason}");
    assert!(
        refused.is_dir() && claim_path(&refused).is_file(),
        "exactly where they were"
    );
    assert!(!report.removed.contains(&refused));
    mode(root, 0o755);

    // 2. A read-only sub-folder with a file in it: the rename succeeds, the removal fails, the folder is renamed BACK
    //    under its own name (a deviation from the reference sweep, which leaves it in quarantine), claim beside it.
    let stuck = folder(root, &name_with("stuck", "abc123"), Some(&gone_claim()));
    let sub = stuck.join("sub");
    std::fs::create_dir(&sub).unwrap();
    std::fs::write(sub.join("file"), b"x").unwrap();
    let _restore_sub = Restore(sub.clone());
    mode(&sub, 0o555);
    premise(&sub);
    std::fs::remove_dir_all(&refused).unwrap();
    std::fs::remove_file(claim_path(&refused)).unwrap();
    let report = run_sweep(root, later(), &mut Hooks::default());
    let reason = kept(&report, &stuck).unwrap_or_else(|| panic!("{report:?}"));
    assert!(reason.starts_with("could not remove:"), "{reason}");
    assert!(!report.removed.contains(&stuck));
    assert!(
        stuck.is_dir() && claim_path(&stuck).is_file(),
        "back under its own name, claim beside it"
    );
    assert!(
        !entries(root).iter().any(|name| name.starts_with('.')),
        "no quarantine is left: {:?}",
        entries(root)
    );

    // Released, a second sweep removes it.
    mode(&sub, 0o755);
    let report = run_sweep(root, later(), &mut Hooks::default());
    assert!(report.removed.contains(&stuck), "{report:?}");
    assert!(!stuck.exists() && !claim_path(&stuck).exists());
}

// ---------- 18 ----------

fn quarantine_name(name: &str, pid: u64) -> String {
    format!(
        ".{name}.{}quarantine-{pid}-abcd",
        PREFIX.to_ascii_lowercase()
    )
}

#[test]
fn a_quarantine_a_killed_sweep_left_is_restored_or_listed() {
    let scratch = TestScratch::new("quarantine-root");
    let root = scratch.path();
    let name = name_with("quar", "abc123");
    let original = root.join(&name);
    let quarantine = root.join(quarantine_name(&name, GONE_PID));
    let make_quarantine = || {
        std::fs::create_dir(&quarantine).unwrap();
        std::fs::write(quarantine.join("kept-file"), b"x").unwrap();
    };

    // Free name: put back, with a notice.
    make_quarantine();
    let report = run_sweep(root, later(), &mut Hooks::default());
    assert!(original.join("kept-file").is_file(), "{report:?}");
    assert!(!quarantine.exists());
    assert!(
        listed(&report, &format!("notice: restored {name}")),
        "{:?}",
        report.listed
    );
    std::fs::remove_dir_all(&original).unwrap();

    // Taken by an EMPTY directory: never replaced (a plain rename(2) of a directory onto an empty one would).
    make_quarantine();
    std::fs::create_dir(&original).unwrap();
    let report = run_sweep(root, later(), &mut Hooks::default());
    assert_eq!(
        kept(&report, &quarantine),
        Some("quarantine-conflict"),
        "{report:?}"
    );
    assert!(
        quarantine.join("kept-file").is_file(),
        "the quarantine is untouched"
    );
    assert!(
        original.is_dir() && entries(&original).is_empty(),
        "the empty directory is still there, still empty"
    );
    assert!(
        listed(&report, &format!("quarantine: {}", quarantine.display())),
        "{:?}",
        report.listed
    );
    #[cfg(unix)]
    assert!(
        listed(
            &report,
            &format!(
                "restore with: mv -- '{}' '{}'",
                quarantine.display(),
                original.display()
            )
        ),
        "{:?}",
        report.listed
    );

    // Taken by a file: the same.
    std::fs::remove_dir(&original).unwrap();
    std::fs::write(&original, b"a file").unwrap();
    let report = run_sweep(root, later(), &mut Hooks::default());
    assert_eq!(
        kept(&report, &quarantine),
        Some("quarantine-conflict"),
        "{report:?}"
    );
    assert_eq!(std::fs::read(&original).unwrap(), b"a file");

    // Every run retries: with the name freed the next sweep restores it.
    std::fs::remove_file(&original).unwrap();
    let report = run_sweep(root, later(), &mut Hooks::default());
    assert!(original.join("kept-file").is_file(), "{report:?}");
    assert!(!quarantine.exists());
    std::fs::remove_dir_all(&original).unwrap();

    // Of a pid that may still be working, or young by either stamp, or not the caller's: untouched and unreported.
    let working = root.join(quarantine_name(&name, this_owner().pid));
    std::fs::create_dir(&working).unwrap();
    let report = run_sweep(root, later(), &mut Hooks::default());
    assert!(working.is_dir() && !original.exists());
    assert!(
        kept(&report, &working).is_none() && !listed(&report, "notice"),
        "{report:?}"
    );
    std::fs::remove_dir(&working).unwrap();

    make_quarantine();
    let young = run_sweep(root, SystemTime::now(), &mut Hooks::default());
    assert!(quarantine.is_dir() && !original.exists());
    assert!(
        young.kept.is_empty() && young.listed.is_empty(),
        "{young:?}"
    );
    let report = sweep(
        root,
        later(),
        SWEEP_MIN_AGE,
        probe_of,
        |_| false,
        &mut Hooks::default(),
    );
    assert!(quarantine.is_dir() && !original.exists());
    assert!(kept(&report, &quarantine).is_none(), "{report:?}");
}

// ---------- 19 ----------

/// A `Write` that records every `write_all` call: one call is one line.
#[derive(Default)]
struct Recorder(Vec<Vec<u8>>);

impl Write for Recorder {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.0.push(buf.to_vec());
        Ok(buf.len())
    }
    fn write_all(&mut self, buf: &[u8]) -> io::Result<()> {
        self.0.push(buf.to_vec());
        Ok(())
    }
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

use std::io::Write;

#[test]
fn the_sweep_never_fails_the_run() {
    let scratch = TestScratch::new("never-fails");
    let missing = scratch.path().join("no-such-root");
    let file_root = scratch.path().join("a-file");
    std::fs::write(&file_root, b"not a directory").unwrap();

    for root in [&missing, &file_root] {
        let report = run_sweep(root, later(), &mut Hooks::default());
        assert!(
            report.removed.is_empty() && report.kept.is_empty() && report.listed.is_empty(),
            "{report:?}"
        );
        assert_eq!(report.errors.len(), 3, "one entry per pass: {report:?}");
        for error in &report.errors {
            assert!(error.contains(&root.display().to_string()), "{error}");
        }

        let mut recorder = Recorder::default();
        print_report(&report, &mut recorder);
        assert_eq!(recorder.0.len(), report.errors.len(), "ONE write per line");
        for call in &recorder.0 {
            let text = String::from_utf8_lossy(call);
            assert!(
                text.ends_with('\n') && text.matches('\n').count() == 1,
                "{text:?}"
            );
        }
    }

    let mut recorder = Recorder::default();
    print_report(&SweepReport::default(), &mut recorder);
    assert!(recorder.0.is_empty(), "nothing for an empty report");

    let mut buf = Vec::new();
    guarded(|| panic!("probe"), &mut buf);
    let text = String::from_utf8(buf).unwrap();
    assert_eq!(text.lines().count(), 1, "{text}");
    assert!(text.contains("probe"), "{text}");

    let mut ran = false;
    let mut buf = Vec::new();
    guarded(
        || {
            ran = true;
            SweepReport::default()
        },
        &mut buf,
    );
    assert!(ran, "guarded runs its closure again afterwards");
    assert!(buf.is_empty());
}
