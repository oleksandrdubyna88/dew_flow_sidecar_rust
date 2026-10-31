//! Scenarios 1, 3-7, 9, 11 and 16 of the E4 plan, S2: the name, claim first, teardown on every unwind, the safe read.

use super::*;

// ---------- 1 ----------

#[test]
fn a_scratch_is_named_with_the_owner_the_token_and_a_random_id() {
    let name = owned_name("seed");
    let rest = name.strip_prefix(PREFIX).expect("the prefix");
    let rest = rest
        .strip_prefix(owner_token())
        .expect("the owner follows the prefix")
        .strip_prefix("-sidecar-seed-")
        .expect("the token and `what` follow the owner");
    assert_eq!(rest.len(), ID_LENGTH, "{name}");
    assert!(rest.bytes().all(|b| b.is_ascii_alphanumeric()), "{name}");
    assert_eq!(name.split('-').nth(1), Some(owner_token()));

    let distinct: BTreeSet<String> = (0..1_000).map(|_| owned_name("seed")).collect();
    assert_eq!(distinct.len(), 1_000, "a counter or a clock alone collides");

    let scratch = TestScratch::new("unicode пробел");
    let leaf = scratch
        .path()
        .file_name()
        .and_then(|n| n.to_str())
        .expect("a UTF-8 leaf");
    assert!(
        leaf.starts_with(&format!(
            "{PREFIX}{}-sidecar-unicode пробел-",
            owner_token()
        )),
        "{leaf}"
    );
    assert_eq!(
        classify_name(leaf),
        NameForm::Owner {
            owner: owner_token().to_string(),
            token: Some("sidecar".into())
        }
    );
    assert!(scratch.path().is_dir());
    assert_eq!(
        scratch.slash_path(),
        scratch.path().to_string_lossy().replace('\\', "/")
    );

    assert_eq!(
        classify_name(&format!("{PREFIX}sidecar-x-abc123")),
        NameForm::Legacy {
            token: "sidecar".into()
        }
    );
    assert_eq!(
        classify_name(&format!("{PREFIX}u7-mcp-x-abc123")),
        NameForm::Owner {
            owner: "u7".into(),
            token: Some("mcp".into())
        }
    );
    assert_eq!(classify_name("bge-log-1-first"), NameForm::None);
}

// ---------- 3 ----------

#[test]
fn a_scratch_is_claimed_first_beside_itself_and_never_inside() {
    let parent = TestScratch::new("claim-parent");
    let scratch = TestScratch::new_in(parent.path(), "claimed");
    let claim = claim_path(scratch.path());
    assert_eq!(
        claim.parent(),
        Some(parent.path()),
        "the claim sits beside the folder"
    );
    assert_eq!(
        std::fs::read(&claim).expect("the claim"),
        format!("{}\n", format_owner(run_record())).into_bytes()
    );
    // The record is read through the `mine` seam: whose file it is, is the kernel owner's answer, not this
    // scenario's subject. On POSIX the real answer is "mine" and is asserted; on Windows an ELEVATED run's new files
    // belong to BUILTIN\Administrators (measured on windows-latest), so the real check says "not mine" and the claim
    // reads `Unreadable` — the correct, fail-closed answer, which must agree with `is_mine` itself.
    assert_eq!(
        read_claim(scratch.path(), &|_| true),
        Claim::Owner(run_record().clone())
    );
    #[cfg(unix)]
    assert_eq!(
        read_claim(scratch.path(), &is_mine),
        Claim::Owner(run_record().clone())
    );
    #[cfg(windows)]
    assert_eq!(
        matches!(read_claim(scratch.path(), &is_mine), Claim::Owner(_)),
        is_mine(&claim),
        "the real read agrees with the real kernel-owner answer"
    );
    assert_eq!(read_claim(scratch.path(), &|_| false), Claim::Unreadable);
    assert!(
        entries(scratch.path()).is_empty(),
        "nothing is ever put inside"
    );

    // Claim first, observed: the claim exists when `create_dir` is asked, and the failed create removes it again.
    let private = TestScratch::new("claim-first");
    let mut seen_claims = Vec::new();
    let mut deny = |dir: &Path| -> io::Result<()> {
        seen_claims.push(claim_path(dir).exists());
        Err(io::ErrorKind::PermissionDenied.into())
    };
    let mut seams = Seams::real();
    seams.create_dir = &mut deny;
    let text = panic_text(|| {
        let _never_made = TestScratch::new_with(private.path(), "denied", &mut seams);
    });
    assert!(text.contains("could not create"), "{text}");
    assert_eq!(
        seen_claims,
        [true],
        "the claim was written before the create was tried"
    );
    assert!(
        entries(private.path()).is_empty(),
        "the unwinding Drop removed the claim again"
    );

    // A claim that cannot be written: no folder is ever made.
    let mut made = false;
    let mut fail =
        |_: &Path, _: &[u8]| -> io::Result<()> { Err(io::ErrorKind::PermissionDenied.into()) };
    let mut create = |_: &Path| -> io::Result<()> {
        made = true;
        Ok(())
    };
    let mut seams = Seams::real();
    seams.write_claim = &mut fail;
    seams.create_dir = &mut create;
    let text = panic_text(|| {
        let _never_made = TestScratch::new_with(private.path(), "unclaimed", &mut seams);
    });
    assert!(text.contains("could not claim"), "{text}");
    assert!(!made, "no folder without a claim");
    assert!(entries(private.path()).is_empty());
}

// ---------- 4 ----------

#[test]
fn an_invalid_what_creates_nothing_and_the_room_is_derived() {
    let private = TestScratch::new("what-parent");
    let room = what_room(owner_token());
    let too_long = "a".repeat(room + 1);
    let two_bytes_each = format!("{}пп", "a".repeat(room - 3));
    assert_eq!(two_bytes_each.chars().count(), room - 1);
    assert_eq!(two_bytes_each.len(), room + 1);
    for bad in [
        "",
        "a/b",
        "a\\b",
        "-x",
        "a\u{1}b",
        "a\0b",
        too_long.as_str(),
        two_bytes_each.as_str(),
    ] {
        assert!(
            panics(|| TestScratch::new_in(private.path(), bad)),
            "{bad:?} must refuse"
        );
        assert!(
            entries(private.path()).is_empty(),
            "{bad:?} created something"
        );
    }

    let exact = "a".repeat(room);
    let scratch = TestScratch::new_in(private.path(), &exact);
    let leaf = scratch
        .path()
        .file_name()
        .and_then(|n| n.to_str())
        .expect("a leaf");
    assert_eq!(leaf.len(), MAX_NAME, "{leaf}");

    assert_eq!(what_room("u1000"), 28);
    assert_eq!(what_room("w4294967295"), 22);
}

// ---------- 5 ----------

#[test]
fn a_taken_claim_draws_again_and_a_squatted_folder_is_never_adopted() {
    let private = TestScratch::new("draw-parent");
    let parent = private.path();
    let stranger = b"stranger\n";

    // A claim somebody else wrote under the first name: not ours, left untouched, a new id drawn.
    let taken = parent.join(format!("{}.owner", name_with("draw", "a")));
    std::fs::write(&taken, stranger).unwrap();
    let mut ids = sequence(&["a", "b"]);
    let mut seams = Seams::real();
    seams.ids = &mut ids;
    let scratch = TestScratch::new_with(parent, "draw", &mut seams);
    assert!(
        scratch.path().ends_with(name_with("draw", "b")),
        "{:?}",
        scratch.path()
    );
    assert_eq!(
        std::fs::read(&taken).unwrap(),
        stranger,
        "the stranger's claim is untouched"
    );
    drop(scratch);
    std::fs::remove_file(&taken).unwrap();

    // A folder already there under the claimed name: never adopted, our claim removed again, one warning.
    let squat = parent.join(name_with("squat", "a"));
    std::fs::create_dir(&squat).unwrap();
    let mut ids = sequence(&["a", "b"]);
    let mut warning = Vec::new();
    let mut seams = Seams::real();
    seams.ids = &mut ids;
    seams.out = &mut warning;
    let scratch = TestScratch::new_with(parent, "squat", &mut seams);
    assert!(scratch.path().ends_with(name_with("squat", "b")));
    assert!(squat.is_dir(), "the squatter's folder is never removed");
    assert!(
        !claim_path(&squat).exists(),
        "our claim was removed again: nothing of ours governs it"
    );
    let warning = String::from_utf8(warning).unwrap();
    assert_eq!(warning.lines().count(), 1, "{warning}");
    assert!(warning.contains(&name_with("squat", "a")), "{warning}");
    drop(scratch);
    std::fs::remove_dir(&squat).unwrap();

    // The claim overwritten between our write and our create: it no longer holds our bytes, so it is NOT removed —
    // a Drop that removed "the claim" would take the stranger's.
    let squat = parent.join(name_with("raced", "a"));
    std::fs::create_dir(&squat).unwrap();
    let stranger_claim = claim_path(&squat);
    let mut ids = sequence(&["a", "b"]);
    let mut overwrite = |dir: &Path| {
        if dir.ends_with(name_with("raced", "a")) {
            std::fs::write(claim_path(dir), stranger).unwrap();
        }
    };
    let mut seams = Seams::real();
    seams.ids = &mut ids;
    seams.after_claim = &mut overwrite;
    let scratch = TestScratch::new_with(parent, "raced", &mut seams);
    assert!(scratch.path().ends_with(name_with("raced", "b")));
    drop(scratch);
    assert_eq!(
        std::fs::read(&stranger_claim).unwrap(),
        stranger,
        "the stranger's claim survived the Released value's Drop"
    );
    std::fs::remove_file(&stranger_claim).unwrap();
    std::fs::remove_dir(&squat).unwrap();

    // Three squatted draws: a panic naming `what` and the parent; nothing of ours is left.
    let names = ["a", "b", "c"].map(|id| parent.join(name_with("full", id)));
    for name in &names {
        std::fs::create_dir(name).unwrap();
    }
    let mut ids = sequence(&["a", "b", "c"]);
    let mut quiet = Vec::new();
    let mut seams = Seams::real();
    seams.ids = &mut ids;
    seams.out = &mut quiet;
    let text = panic_text(|| {
        let _never_made = TestScratch::new_with(parent, "full", &mut seams);
    });
    assert!(
        text.contains("full") && text.contains(&parent.display().to_string()),
        "{text}"
    );
    assert_eq!(
        entries(parent).len(),
        3,
        "only the three squatters remain: {:?}",
        entries(parent)
    );
}

// ---------- 2 and 5, the claim's removal is retried ----------

/// A directory that is read-only until this guard drops: its entries cannot be unlinked (a non-root user).
#[cfg(unix)]
struct ReadOnly(PathBuf);

#[cfg(unix)]
impl ReadOnly {
    fn set(dir: &Path, mode: u32) {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(dir, std::fs::Permissions::from_mode(mode)).unwrap();
    }
}

#[cfg(unix)]
impl Drop for ReadOnly {
    fn drop(&mut self) {
        ReadOnly::set(&self.0, 0o700);
    }
}

/// Review round 1: the claim of a value in state `Claimed`, and the claim a squatter made us take back, are removed
/// with the five tries of rule section 2 — and a squatter-path removal that gave up leaves the value `ClaimPending`,
/// whose `Drop` tries the five again, never `Released` with the claim still on disk. The unlink is made to fail
/// (a read-only parent) and the time the attempts take is the evidence: four back-offs of 100 ms per round of tries.
#[cfg(unix)]
#[test]
fn a_claim_that_will_not_go_is_retried_in_both_states() {
    // SAFETY: geteuid has no preconditions and cannot fail.
    if unsafe { libc::geteuid() } == 0 {
        eprintln!("skipped: root unlinks from a read-only directory");
        return;
    }
    let private = TestScratch::new("claim-retry");
    let parent = private.path();
    let guard = ReadOnly(parent.to_path_buf());

    // A squatter: the create says `AlreadyExists`, the parent turns read-only, our claim cannot be taken back. One
    // round of five tries in the squatter path, one more in the Drop of the `ClaimPending` value; then the next
    // draw (the parent is writable again by then) succeeds.
    let first = parent.join(name_with("stuck", "a"));
    let mut ids = {
        let guard_dir = parent.to_path_buf();
        let mut drawn = 0;
        move || {
            drawn += 1;
            if drawn == 2 {
                ReadOnly::set(&guard_dir, 0o700); // the second draw comes after the first value dropped
            }
            ["a", "b"][drawn - 1].to_string()
        }
    };
    let mut calls = 0;
    let mut create = |path: &Path| -> io::Result<()> {
        calls += 1;
        if calls == 1 {
            ReadOnly::set(path.parent().unwrap(), 0o500);
            Err(io::ErrorKind::AlreadyExists.into())
        } else {
            std::fs::create_dir(path)
        }
    };
    let mut quiet = Vec::new();
    let mut seams = Seams::real();
    seams.ids = &mut ids;
    seams.create_dir = &mut create;
    seams.out = &mut quiet;
    let started = std::time::Instant::now();
    let scratch = TestScratch::new_with(parent, "stuck", &mut seams);
    let took = started.elapsed();
    assert!(
        took >= Duration::from_millis(790),
        "two rounds of five tries, 100 ms apart, took only {took:?}"
    );
    assert!(
        claim_path(&first).is_file(),
        "the claim that would not go is left for the sweep, not forgotten"
    );
    drop(scratch);
    std::fs::remove_file(claim_path(&first)).unwrap();

    // A value in state `Claimed` whose create fails for another reason: the unwind drops it, five tries.
    let mut create = |path: &Path| -> io::Result<()> {
        ReadOnly::set(path.parent().unwrap(), 0o500);
        Err(io::ErrorKind::PermissionDenied.into())
    };
    let mut ids = sequence(&["c"]);
    let mut seams = Seams::real();
    seams.ids = &mut ids;
    seams.create_dir = &mut create;
    let started = std::time::Instant::now();
    let text = panic_text(|| {
        let _never_made = TestScratch::new_with(parent, "stuck", &mut seams);
    });
    let took = started.elapsed();
    assert!(text.contains("could not create"), "{text}");
    assert!(
        took >= Duration::from_millis(390),
        "five tries, 100 ms apart, took only {took:?}"
    );
    drop(guard);
    let left = parent.join(format!("{}.owner", name_with("stuck", "c")));
    assert!(left.is_file(), "left for the sweep");
    std::fs::remove_file(left).unwrap();
}

// ---------- 6 ----------

#[test]
fn a_scratch_is_removed_when_its_test_panics() {
    let private = TestScratch::new("panic-parent");
    let mut recorded = None;
    let outcome = catch_unwind(AssertUnwindSafe(|| {
        let scratch = TestScratch::new_in(private.path(), "boom");
        recorded = Some(scratch.path().to_path_buf());
        std::fs::write(scratch.path().join("inside"), b"x").unwrap();
        panic!("a failed assertion");
    }));
    assert!(outcome.is_err());
    let path = recorded.expect("the scratch existed");
    assert!(!path.exists(), "the folder is gone");
    assert!(!claim_path(&path).exists(), "and so is its claim");
}

#[test]
fn a_failed_assertion_in_a_real_test_still_removes_its_scratch() {
    let private = TestScratch::new("child-assert");
    let (status, path, _) = run_child("assert", private.path());
    assert!(!status.success(), "the child must fail");
    assert!(path.starts_with(private.path()));
    assert!(!path.exists(), "the folder survived a failed assertion");
    assert!(!claim_path(&path).exists(), "and so did its claim");
}

// ---------- the lessons of the C# port: records are shared, files are not ----------

#[test]
fn a_claim_with_the_same_text_but_another_identity_is_not_ours_to_remove() {
    let private = TestScratch::new("same-text-parent");
    let parent = private.path();
    let squat = parent.join(name_with("same", "a"));
    std::fs::create_dir(&squat).unwrap();
    let swapped = claim_path(&squat);
    let aside = parent.join("claim-aside");
    let mut ids = sequence(&["a", "b"]);
    let mut swap = |dir: &Path| {
        if dir.ends_with(name_with("same", "a")) {
            // Another run's claim with IDENTICAL text (siblings under one TESTSTODELETE_RUN write identical
            // records); ours is kept alive beside, so the new file cannot reuse its inode number.
            let text = std::fs::read(claim_path(dir)).unwrap();
            std::fs::rename(claim_path(dir), &aside).unwrap();
            std::fs::write(claim_path(dir), text).unwrap();
        }
    };
    let mut quiet = Vec::new();
    let mut seams = Seams::real();
    seams.ids = &mut ids;
    seams.after_claim = &mut swap;
    seams.out = &mut quiet;
    let scratch = TestScratch::new_with(parent, "same", &mut seams);
    assert!(
        swapped.is_file(),
        "the claim now at that path is not the file this run wrote: it stays"
    );
    drop(scratch);
    assert!(swapped.is_file());
}

#[test]
fn siblings_sharing_one_run_record_never_remove_each_others_folders() {
    let private = TestScratch::new("shared-run");
    let record = format_owner(this_owner());
    let sibling = TestScratch::new_in(private.path(), "sibling");
    let envs = [("TESTSTODELETE_RUN", record.as_str())];

    // The child's Drop removes ITS folder and claim, by its own path — never "the folders of the run record".
    let (status, path, _) = run_child_with("assert", private.path(), &envs);
    assert!(!status.success());
    assert!(
        !path.exists() && !claim_path(&path).exists(),
        "the child's own folder is gone"
    );
    assert!(
        sibling.path().is_dir() && claim_path(sibling.path()).is_file(),
        "a sibling holding the very same record is untouched"
    );

    // A killed child leaves a claim carrying the RUN's record (the parent's pid), so the run's liveness governs it:
    // while the run lives both are kept, and nothing is ever removed because two folders share one record.
    let (_, killed, _) = run_child_with("abort", private.path(), &envs);
    let Claim::Owner(theirs) = read_claim(&killed, &|_| true) else {
        panic!("the killed child's claim is readable");
    };
    assert_eq!(
        &theirs,
        this_owner(),
        "the claim carries the run's record, not the child's"
    );
    let report = sweep(
        private.path(),
        std::time::SystemTime::now() + Duration::from_secs(11 * 60),
        SWEEP_MIN_AGE,
        probe,
        |_| true,
        &mut Hooks::default(),
    );
    assert!(report.removed.is_empty(), "{report:?}");
    assert!(killed.is_dir() && sibling.path().is_dir());
}

// ---------- 9 ----------

#[test]
fn removal_retries_then_gives_up_without_panicking() {
    let path = Path::new("unused");
    let mut calls = 0;
    let mut failed = Vec::new();
    let result = remove_with_retries(
        path,
        |_| {
            calls += 1;
            if calls <= 2 {
                Err(io::ErrorKind::PermissionDenied.into())
            } else {
                Ok(())
            }
        },
        |n| failed.push(n),
    );
    assert!(result.is_ok());
    assert_eq!(calls, 3);
    assert_eq!(failed, [1, 2], "a callback after each failed attempt");

    let mut calls = 0;
    let started = std::time::Instant::now();
    let result = remove_with_retries(
        path,
        |_| {
            calls += 1;
            Err(io::ErrorKind::PermissionDenied.into())
        },
        |_| {},
    );
    assert_eq!(calls, REMOVE_TRIES, "exactly five tries");
    assert!(
        started.elapsed() >= Duration::from_millis(400),
        "{:?} — four back-offs of 100 ms",
        started.elapsed()
    );
    assert_eq!(
        result.unwrap_err().kind(),
        io::ErrorKind::PermissionDenied,
        "the last error is returned"
    );

    let mut calls = 0;
    let result = remove_with_retries(
        path,
        |_| {
            calls += 1;
            Err(io::ErrorKind::NotFound.into())
        },
        |_| panic!("NotFound is success, not a failed attempt"),
    );
    assert!(result.is_ok());
    assert_eq!(calls, 1);
}

// ---------- 11 ----------

#[cfg(unix)]
#[test]
fn a_scratch_that_cannot_be_removed_stays_prefixed_and_claimed() {
    use std::os::unix::fs::PermissionsExt;

    /// Gives the folder its write bit back even when an assertion fails, so the scratches inside can be removed.
    struct Restore(PathBuf);
    impl Drop for Restore {
        fn drop(&mut self) {
            let _ = std::fs::set_permissions(&self.0, std::fs::Permissions::from_mode(0o755));
        }
    }

    let outer = TestScratch::new("held-outer");

    // A folder that cannot be emptied while its PARENT stays writable: the claim must stay with it — a teardown that
    // took the claim although the folder stayed would leave an unmarked folder nobody can ever judge.
    let stuck = TestScratch::new_in(outer.path(), "held-stuck");
    let sub = stuck.path().join("sub");
    std::fs::create_dir(&sub).unwrap();
    std::fs::write(sub.join("file"), b"x").unwrap();
    let _restore_sub = Restore(sub.clone());
    std::fs::set_permissions(&sub, std::fs::Permissions::from_mode(0o555)).unwrap();
    assert!(
        std::fs::File::create(sub.join("probe")).is_err(),
        "this process can write into a 0o555 folder (running as root?) — the scenario cannot prove anything here"
    );
    let mut buf = Vec::new();
    assert!(!teardown(stuck.path(), &mut buf));
    assert_eq!(String::from_utf8(buf).unwrap().lines().count(), 1);
    assert!(stuck.path().is_dir());
    assert!(
        claim_path(stuck.path()).is_file(),
        "the folder stayed, so its claim stays"
    );
    drop(_restore_sub);

    let inner = TestScratch::new_in(outer.path(), "held-inner");
    let _restore = Restore(outer.path().to_path_buf());
    std::fs::set_permissions(outer.path(), std::fs::Permissions::from_mode(0o555)).unwrap();
    // The premise: a process running as root removes it anyway, and this scenario would pass for the wrong reason.
    assert!(
        std::fs::File::create(outer.path().join("probe")).is_err(),
        "this process can write into a 0o555 folder (running as root?) — the scenario cannot prove anything here"
    );

    let mut buf = Vec::new();
    assert!(
        !teardown(inner.path(), &mut buf),
        "the teardown must report that it gave up"
    );
    let line = String::from_utf8(buf).unwrap();
    assert_eq!(line.lines().count(), 1, "ONE line: {line}");
    assert!(line.contains(&inner.path().display().to_string()), "{line}");
    assert!(inner.path().is_dir(), "the folder stays");
    assert!(
        claim_path(inner.path()).is_file(),
        "and stays claimed, so the next sweep can judge it"
    );
}

// ---------- 16 ----------

#[test]
fn a_claim_is_read_through_the_one_safe_read() {
    let private = TestScratch::new("read-parent");
    let parent = private.path();
    let dir = parent.join(name_with("read", "a"));
    let claim = claim_path(&dir);
    let record = |bytes: usize| -> Vec<u8> {
        // `u1000:<host>@linux:1\n` padded with the host to exactly `bytes` bytes, line end included.
        let fixed = "u1000:@linux:1\n".len();
        let mut text = format!("u1000:{}@linux:1\n", "h".repeat(bytes - fixed));
        assert_eq!(text.len(), bytes);
        text.shrink_to_fit();
        text.into_bytes()
    };

    assert_eq!(
        read_claim(&dir, &|_| true),
        Claim::Unmarked,
        "no claim at all"
    );

    std::fs::create_dir(&claim).unwrap();
    assert_eq!(
        read_claim(&dir, &|_| true),
        Claim::Unreadable,
        "a directory"
    );
    std::fs::remove_dir(&claim).unwrap();

    let real = parent.join("real-claim");
    std::fs::write(&real, record(40)).unwrap();
    #[cfg(unix)]
    {
        std::os::unix::fs::symlink(&real, &claim).unwrap();
        assert_eq!(
            read_claim(&dir, &|_| true),
            Claim::Unreadable,
            "a link is never followed"
        );
        std::fs::remove_file(&claim).unwrap();
    }

    std::fs::write(&claim, record(MAX_CLAIM_BYTES)).unwrap();
    assert!(
        matches!(read_claim(&dir, &|_| true), Claim::Owner(_)),
        "512 bytes are read"
    );
    assert_eq!(
        read_claim(&dir, &|_| false),
        Claim::Unreadable,
        "a stranger's claim is no evidence"
    );
    std::fs::write(&claim, record(MAX_CLAIM_BYTES + 1)).unwrap();
    assert_eq!(read_claim(&dir, &|_| true), Claim::Unreadable, "513 bytes");
    std::fs::remove_file(&claim).unwrap();

    #[cfg(unix)]
    {
        use std::os::unix::ffi::OsStrExt;
        let c_path = std::ffi::CString::new(claim.as_os_str().as_bytes()).unwrap();
        // SAFETY: `c_path` is a valid NUL-terminated path for the duration of the call.
        assert_eq!(unsafe { libc::mkfifo(c_path.as_ptr(), 0o600) }, 0, "mkfifo");
        let (sender, receiver) = std::sync::mpsc::channel();
        let asked = dir.clone();
        std::thread::spawn(move || sender.send(read_claim(&asked, &|_| true)).ok());
        let answer = receiver
            .recv_timeout(Duration::from_secs(1))
            .expect("a FIFO blocked the read");
        assert_eq!(answer, Claim::Unreadable);
        std::fs::remove_file(&claim).unwrap();
    }

    // The record a helper writes: the run's own when it names a LIVE run of this user, else this process's.
    let me = this_owner();
    let theirs = Owner {
        pid: 4_242,
        ..me.clone()
    };
    let alive = |_: u64| Probe::Alive { start: me.start };
    let mut buf = Vec::new();
    let used = run_record_from(Some(&format_owner(&theirs)), alive, &mut buf);
    assert_eq!(used, theirs);
    assert!(buf.is_empty(), "{}", String::from_utf8_lossy(&buf));

    for (given, probe) in [
        (
            format_owner(&Owner {
                owner: "u7".into(),
                ..theirs.clone()
            }),
            Probe::Alive { start: me.start },
        ),
        (format_owner(&theirs), Probe::Gone),
        ("bob".to_string(), Probe::Alive { start: me.start }),
    ] {
        let mut buf = Vec::new();
        let used = run_record_from(Some(&given), |_| probe, &mut buf);
        assert_eq!(&used, me, "{given}");
        assert_eq!(
            String::from_utf8(buf).unwrap().lines().count(),
            1,
            "{given}: one warning line"
        );
    }
    let mut buf = Vec::new();
    assert_eq!(&run_record_from(None, alive, &mut buf), me);
    assert!(buf.is_empty());
}
