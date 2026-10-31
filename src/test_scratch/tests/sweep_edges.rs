//! The edges of scenarios 17-19 of the E4 plan, S2: a claim changed late, a link, orphan claims, a removal that gives
//! up, a quarantine a killed sweep left, and a sweep that never fails the run.

use super::sweeping::*;
use super::*;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::time::SystemTime;

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

    let stranger = format!(
        "{}.owner",
        compose_name(other_owner(), REPO_TOKEN, "orphan", "abc123")
    );
    std::fs::write(root.join(&stranger), gone_claim()).unwrap();
    let report = run_sweep(root, later(), &mut Hooks::default());
    assert!(
        listed(&report, &format!("foreign: claim {stranger}")),
        "{:?}",
        report.listed
    );
    assert!(root.join(&stranger).is_file());
}

/// Review round 1: a sweep killed after its rename leaves the folder's claim next to a quarantine. The orphan pass
/// saw "a claim whose folder is absent" and removed it — the claim of a folder that is merely in quarantine.
#[test]
fn an_orphan_claim_beside_a_quarantine_is_kept() {
    let scratch = TestScratch::new("orphan-quarantine");
    let root = scratch.path();
    let name = name_with("held", "abc123");
    let original = folder(root, &name, Some(&gone_claim()));
    let killed = catch_unwind(AssertUnwindSafe(|| {
        let mut die = |_: &Path, _: &Path| panic!("killed after the rename");
        let mut hooks = Hooks {
            after_rename: Some(&mut die),
            ..Hooks::default()
        };
        run_sweep(root, later(), &mut hooks)
    }));
    assert!(killed.is_err() && !original.exists());

    // The renamer is THIS process, alive: the quarantine is not restored, and its claim is not an orphan.
    let report = run_sweep(root, later(), &mut Hooks::default());
    assert!(
        claim_path(&original).is_file(),
        "the claim of a quarantined folder stays: {report:?}"
    );
    assert!(report.removed.is_empty(), "{report:?}");
    assert_eq!(
        kept(&report, &claim_path(&original)),
        Some("quarantined"),
        "{report:?}"
    );

    // Once the quarantine is restored and the folder removed with its claim, nothing is left to be an orphan.
    let left = entries(root)
        .into_iter()
        .find(|entry| quarantine_pattern().is_match(entry))
        .unwrap();
    std::fs::remove_dir_all(root.join(left)).unwrap();
    let report = run_sweep(root, later(), &mut Hooks::default());
    assert_eq!(
        report.removed,
        vec![claim_path(&original)],
        "a true orphan goes: {report:?}"
    );
}

/// Review round 1: the root is listed ONCE and the passes share the snapshot; every candidate is still re-checked, so
/// an entry this very sweep removed (a folder's claim) is skipped, not an `errors` entry.
#[test]
fn the_passes_share_one_listing_of_the_root() {
    let scratch = TestScratch::new("one-listing");
    let root = scratch.path();
    let doomed = folder(root, &name_with("doomed", "abc123"), Some(&gone_claim()));
    let late = claim_path(&root.join(name_with("late", "abc123")));

    // A claim that appears after the listing was taken: this sweep never sees it, the next one does.
    let mut appear = |_: &Path| std::fs::write(&late, gone_claim()).unwrap();
    let mut hooks = Hooks {
        after_capture: Some(&mut appear),
        ..Hooks::default()
    };
    let report = run_sweep(root, later(), &mut hooks);
    assert_eq!(report.removed, vec![doomed.clone()], "{report:?}");
    assert!(
        report.errors.is_empty(),
        "a claim the sweep removed is not an error: {report:?}"
    );
    assert!(late.is_file(), "the snapshot predates it: {report:?}");

    let report = run_sweep(root, later(), &mut Hooks::default());
    assert_eq!(report.removed, vec![late.clone()], "{report:?}");
}

/// Review round 1: the quarantine pass asked the probe about ANY number in the name. A pid that cannot be a process
/// (0, a leading zero, past 2^53, one that overflows) is never asked and never `gone` — `owner_state`'s checks, the
/// reference's `processProbe`, applied before the probe: the quarantine of such a name stays where it is.
#[test]
fn the_quarantine_pass_never_asks_about_a_pid_that_cannot_be_a_process() {
    let scratch = TestScratch::new("quarantine-pids");
    let root = scratch.path();
    let name = name_with("pids", "abc123");
    let original = root.join(&name);
    let beyond = (1u64 << 53).to_string();
    let mut quarantines = Vec::new();
    for pid in ["0", "007", &beyond, "99999999999999999999999"] {
        let quarantine = root.join(quarantine_leaf(&name, pid, "abcd"));
        std::fs::create_dir(&quarantine).unwrap();
        quarantines.push(quarantine);
    }
    let asked = std::cell::RefCell::new(Vec::new());
    let everything_is_gone = |pid: u64| {
        asked.borrow_mut().push(pid);
        Probe::Gone
    };
    let report = sweep(
        root,
        later(),
        SWEEP_MIN_AGE,
        everything_is_gone,
        |_| true,
        &mut Hooks::default(),
    );
    assert!(
        asked.borrow().is_empty(),
        "the probe was asked about {:?}",
        asked.borrow()
    );
    assert!(!original.exists(), "{report:?}");
    for quarantine in &quarantines {
        assert!(quarantine.is_dir(), "{quarantine:?} stays: {report:?}");
    }
    assert!(
        report.listed.is_empty() && report.kept.is_empty(),
        "{report:?}"
    );

    // The same table, for the reader's operating system: pid 1 is no process on Windows, nothing past u32::MAX is.
    assert!(!pid_is_askable(0, false) && !pid_is_askable(1 << 53, false));
    assert!(pid_is_askable(1, false) && pid_is_askable(1 << 40, false));
    assert!(!pid_is_askable(1, true) && !pid_is_askable(u64::from(u32::MAX) + 1, true));
    assert!(pid_is_askable(2, true) && pid_is_askable(u64::from(u32::MAX), true));
}

/// Review round 1: the rename INTO quarantine must not replace anything. POSIX `rename(2)` of a directory onto an EMPTY
/// directory replaces it (and so does Windows' `MoveFileEx` with replace semantics): an empty directory somebody made
/// at the drawn quarantine name would be gone with no trace. The name is drawn again instead.
#[test]
fn the_rename_into_quarantine_never_replaces_an_existing_entry() {
    let scratch = TestScratch::new("no-replace");
    let root = scratch.path();
    let doomed = folder(root, &name_with("doomed", "abc123"), Some(&gone_claim()));
    std::fs::write(doomed.join("inside"), b"x").unwrap();
    let taken = root.join(".taken-quarantine-name");
    let free = root.join(".free-quarantine-name");
    std::fs::create_dir(&taken).unwrap();
    let identity = file_id(&taken);
    assert!(identity.is_some());

    let mut drawn = Vec::new();
    let mut draw = |dir: &Path| {
        assert_eq!(dir, doomed);
        drawn.push(());
        if drawn.len() == 1 {
            taken.clone()
        } else {
            free.clone()
        }
    };
    let mut hooks = Hooks {
        quarantine_name: Some(&mut draw),
        ..Hooks::default()
    };
    let report = run_sweep(root, later(), &mut hooks);
    assert_eq!(report.removed, vec![doomed.clone()], "{report:?}");
    assert!(
        taken.is_dir() && entries(&taken).is_empty() && same_id(file_id(&taken), identity),
        "the empty directory at the drawn name is still there, the same one: {report:?}"
    );
    assert_eq!(drawn.len(), 2, "a collision draws again, once");
    assert!(!doomed.exists() && !free.exists());
}

/// The fallback where a file system refuses the no-replace flag: a name is RESERVED with an exclusive `mkdir`, so a
/// taken one is never touched, and the same placeholder is what the rename then lands on.
#[cfg(unix)]
#[test]
fn the_placeholder_fallback_reserves_a_name_and_refuses_a_taken_one() {
    let scratch = TestScratch::new("placeholder");
    let root = scratch.path();
    let from = root.join("from");
    std::fs::create_dir(&from).unwrap();
    std::fs::write(from.join("inside"), b"x").unwrap();
    let identity = file_id(&from);

    let taken = root.join("taken");
    std::fs::create_dir(&taken).unwrap();
    let taken_id = file_id(&taken);
    let refused = move_via_placeholder(&from, &taken).unwrap_err();
    assert_eq!(refused.kind(), io::ErrorKind::AlreadyExists);
    assert!(same_id(file_id(&taken), taken_id) && from.join("inside").is_file());

    let free = root.join("free");
    move_via_placeholder(&from, &free).unwrap();
    assert!(free.join("inside").is_file() && !from.exists());
    assert!(file_id(&free).is_some_and(|id| identity.is_some_and(|before| id.same_inode(&before))));
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
    quarantine_leaf(name, pid, "abcd")
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

// ---------- 18, the producer's own names ----------

/// The pattern every sweeper of the family reads (`leftovers-sweep.mjs`, `janitor.sh`, rule §4): `<hex>` is
/// `[0-9a-f]+`. Review round 1: the producer drew `[a-z0-9]` and 2 of 1000 names matched.
#[test]
fn the_quarantine_name_the_producer_draws_is_the_one_every_sweeper_reads() {
    let name = name_with("quar", "abc123");
    let dir = Path::new("/somewhere").join(&name);
    for _ in 0..5000 {
        let made = quarantine_path(&dir);
        let leaf = made.file_name().unwrap().to_string_lossy().into_owned();
        let found = quarantine_pattern().captures(&leaf).unwrap_or_else(|| {
            panic!("the producer's quarantine name {leaf} is invisible to the pattern")
        });
        assert_eq!(&found[1], name);
        assert_eq!(found[2].parse::<u32>().unwrap(), std::process::id());
        let hex = &found[3];
        assert!(
            hex.len() == 8
                && hex
                    .bytes()
                    .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b)),
            "the suffix is 8 lower-case hex characters: {hex}"
        );
        // The same pattern as the other sweepers, written out here, independent of `quarantine_pattern`.
        let shared = regex::Regex::new(&format!(
            r"(?i)^\.(.+)\.{}quarantine-(\d+)-([0-9a-f]+)$",
            prefix_lowercase()
        ))
        .unwrap();
        assert!(shared.is_match(&leaf), "{leaf}");
    }
}

#[test]
fn a_sweep_killed_after_its_rename_leaves_a_quarantine_the_next_sweep_restores() {
    let scratch = TestScratch::new("quarantine-real");
    let root = scratch.path();
    let name = name_with("kill", "abc123");
    let original = folder(root, &name, Some(&gone_claim()));
    std::fs::write(original.join("kept-file"), b"x").unwrap();

    // The REAL producer names the quarantine; the sweep dies right after the rename (a panic stands for SIGKILL).
    let killed = catch_unwind(AssertUnwindSafe(|| {
        let mut die = |_: &Path, _: &Path| panic!("killed after the rename");
        let mut hooks = Hooks {
            after_rename: Some(&mut die),
            ..Hooks::default()
        };
        run_sweep(root, later(), &mut hooks)
    }));
    assert!(killed.is_err());
    assert!(!original.exists(), "the folder was renamed away");
    let left: Vec<String> = entries(root)
        .into_iter()
        .filter(|name| quarantine_pattern().is_match(name))
        .collect();
    assert_eq!(
        left.len(),
        1,
        "the killed sweep's quarantine is visible to the pattern: {:?}",
        entries(root)
    );

    // Its renamer is dead: the pid in the REAL name is replaced by a gone one, nothing else is touched.
    let found = quarantine_pattern().captures(&left[0]).unwrap();
    let gone_name = left[0].replacen(
        &format!("quarantine-{}-", &found[2]),
        &format!("quarantine-{GONE_PID}-"),
        1,
    );
    std::fs::rename(root.join(&left[0]), root.join(&gone_name)).unwrap();
    // The folder is judged again once it is back: its owner is a live one now, so it is kept, not collected again.
    std::fs::write(
        claim_path(&original),
        record_text(&record_for(this_owner().pid)),
    )
    .unwrap();
    let report = run_sweep(root, later(), &mut Hooks::default());
    assert!(original.join("kept-file").is_file(), "{report:?}");
    assert!(
        listed(&report, &format!("notice: restored {name}")),
        "{:?}",
        report.listed
    );
    assert_eq!(kept(&report, &original), Some("live"), "{report:?}");
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
