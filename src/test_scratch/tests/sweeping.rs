//! Scenarios 8, 17, 18 and 19 of the E4 plan, S2: the start-of-run sweep.
//!
//! Every scenario sweeps a PRIVATE root with `now` injected, a probe that says which pids are gone, and a `mine` seam
//! (a second uid cannot be had in a test). Names a scenario needs for another owner, another repository or the legacy
//! form are built from `PREFIX` and pieces at run time: S4's guard reads this file too.

use super::*;
use std::time::SystemTime;

/// A pid no live process of this test has: the probe says `Gone` for it, so a record naming it is abandoned.
pub(super) const GONE_PID: u64 = 999_001;

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

pub(super) fn other_owner() -> &'static str {
    if owner_token() == "u7" {
        "u8"
    } else {
        "u7"
    }
}

pub(super) fn record_text(owner: &Owner) -> String {
    format!("{}\n", format_owner(owner))
}

/// A record of this machine and namespace for `pid`, written the way a helper writes one.
pub(super) fn record_for(pid: u64) -> Owner {
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

pub(super) fn kept<'a>(report: &'a SweepReport, path: &Path) -> Option<&'a str> {
    report
        .kept
        .iter()
        .find(|(kept, _)| kept == path)
        .map(|(_, reason)| reason.as_str())
}

pub(super) fn listed(report: &SweepReport, needle: &str) -> bool {
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
    // testsToDelete-exempt: legacy-name — a legacy name the sweep must only list, built at run time
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
