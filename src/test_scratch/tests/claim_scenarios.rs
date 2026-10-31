//! Review round 1 of the E4 producer PR: how the claim is removed and how it is read — five tries per round, the
//! `ClaimPending` state of a squatter path, the whole identity between the read and the unlink, and a safe read that
//! judges the opened handle and not the path it was given.

use super::*;

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

/// Review round 1: between the read of the claim and its unlink the file is compared by its WHOLE identity (device,
/// inode, change time), not by the inode alone — a file re-written or re-moded in that instant has the same inode
/// number and the same bytes, and is no longer the file that was read.
#[cfg(unix)]
#[test]
fn a_claim_changed_between_the_read_and_the_unlink_is_not_removed() {
    use std::os::unix::fs::PermissionsExt;
    let private = TestScratch::new("ctime-parent");
    let dir = private.path().join(name_with("ctime", "a"));
    let claim = claim_path(&dir);
    let text = b"the bytes this run wrote\n";
    write_claim_exclusive(&claim, text).unwrap();
    // The change-time moves, the inode and the bytes do not.
    let mut touch = || {
        std::thread::sleep(Duration::from_millis(20));
        std::fs::set_permissions(&claim, std::fs::Permissions::from_mode(0o640)).unwrap();
    };
    assert!(remove_claim_with_hook(&dir, text, None, &mut touch));
    assert!(
        claim.is_file(),
        "a claim touched between the read and the unlink stays"
    );

    // Untouched, the same call removes it.
    assert!(remove_claim_with_hook(
        &dir,
        text,
        file_id(&claim),
        &mut || {}
    ));
    assert!(!claim.exists());
}

/// Review round 1: the safe read judges the OPENED handle, not the path it was told about. A path swapped between the
/// type test and the open — for a FIFO, or for another regular file — is unreadable, and never blocks.
#[test]
fn the_safe_read_judges_the_opened_handle_not_the_path() {
    let private = TestScratch::new("handle-parent");
    let parent = private.path();
    let claim = parent.join("claim");
    std::fs::write(&claim, b"the file that was typed\n").unwrap();
    let typed = file_id(&claim);
    assert_eq!(
        read_limited(&claim, typed).unwrap(),
        b"the file that was typed\n"
    );

    // Swapped for another regular file (a fresh inode, the old one kept alive beside): not the file that was typed.
    let aside = parent.join("aside");
    std::fs::rename(&claim, &aside).unwrap();
    std::fs::write(&claim, b"another file\n").unwrap();
    assert!(
        read_limited(&claim, typed).is_err(),
        "the file was replaced"
    );
    std::fs::remove_file(&claim).unwrap();

    // Swapped for a FIFO: the open must not block, and the answer is an error.
    #[cfg(unix)]
    fifo_is_refused(&claim, typed);
    // Swapped for a device (Windows): not `FILE_TYPE_DISK`.
    #[cfg(windows)]
    assert!(
        read_limited(Path::new(r"\\.\NUL"), typed).is_err(),
        "a device is not a disk file"
    );
}

#[cfg(unix)]
fn fifo_is_refused(claim: &Path, typed: Option<FileId>) {
    use std::os::unix::ffi::OsStrExt;
    std::fs::remove_file(claim).ok();
    let c_path = std::ffi::CString::new(claim.as_os_str().as_bytes()).unwrap();
    // SAFETY: `c_path` is a valid NUL-terminated path for the duration of the call.
    assert_eq!(unsafe { libc::mkfifo(c_path.as_ptr(), 0o600) }, 0, "mkfifo");
    let (sender, receiver) = std::sync::mpsc::channel();
    let (path, expected) = (claim.to_path_buf(), typed);
    std::thread::spawn(move || sender.send(read_limited(&path, expected).is_err()).ok());
    let refused = receiver
        .recv_timeout(Duration::from_secs(2))
        .expect("a FIFO blocked the read");
    assert!(refused, "a FIFO is not the regular file that was typed");
}
