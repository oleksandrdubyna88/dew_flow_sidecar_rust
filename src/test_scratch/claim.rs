//! The claim file beside a folder: written first, read through ONE safe read, removed only while it is ours.
//!
//! The claim sits BESIDE the folder (`<folder>.owner`), never inside it: a test that makes a git repository at the
//! folder, or asserts the exact listing of the folder, must see exactly what it made (rule §3).

use super::identity::{file_id, handle_id, is_plain_file, FileId};
use super::owner::{parse_owner, Claim};
use super::{remove_with_retries, write_line, StderrOut, MAX_CLAIM_BYTES};
use std::ffi::OsString;
use std::fs::{File, OpenOptions};
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};

const OWNER_SUFFIX: &str = ".owner";

/// `<dir>.owner`, in the same parent.
pub(crate) fn claim_path(dir: &Path) -> PathBuf {
    let mut name: OsString = dir.as_os_str().to_owned();
    name.push(OWNER_SUFFIX);
    PathBuf::from(name)
}

/// What the one safe read found. A stranger's claim (`foreign_uid`) is no evidence and is never opened.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum RawClaim {
    Absent,
    Unreadable { foreign_uid: bool },
    Text(Vec<u8>),
}

/// The claim's bytes and the identity of the file they came from, taken together.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct ClaimSnapshot {
    pub raw: RawClaim,
    pub id: Option<FileId>,
}

/// Open `path` without following a link (POSIX `O_NOFOLLOW`; a Windows reparse point is opened ITSELF) and without
/// ever blocking on a FIFO or a named pipe swapped in after the type test, and read at most `MAX_CLAIM_BYTES + 1`
/// bytes — one past the limit, to see it. The OPENED HANDLE is judged, not the path: it must be a regular file
/// (POSIX `fstat` `S_ISREG`; Windows `GetFileType` is `FILE_TYPE_DISK` and the handle is neither a directory nor a
/// reparse point) and the file `expected` named (device and inode; Windows the 128-bit file id). No identity to
/// compare with, or one that differs, is an error: the entry was swapped, and nothing is read.
pub(crate) fn read_limited(path: &Path, expected: Option<FileId>) -> io::Result<Vec<u8>> {
    let mut options = OpenOptions::new();
    options.read(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK);
    }
    #[cfg(windows)]
    {
        use std::os::windows::fs::OpenOptionsExt;
        options.custom_flags(0x0020_0000); // FILE_FLAG_OPEN_REPARSE_POINT
    }
    let file: File = options.open(path)?;
    if !is_plain_file(&file) {
        return Err(io::Error::other("not a regular file"));
    }
    match (handle_id(&file), expected) {
        (Some(opened), Some(expected)) if opened.same_inode(&expected) => {}
        _ => {
            return Err(io::Error::other(
                "the file was replaced while it was opened",
            ))
        }
    }
    let mut bytes = Vec::new();
    file.take(MAX_CLAIM_BYTES as u64 + 1)
        .read_to_end(&mut bytes)?;
    Ok(bytes)
}

/// The sibling claim of `dir` through the one safe read: `symlink_metadata` first (absent only on `NotFound`), a
/// regular file only (a FIFO would block for ever), the caller's only (`mine`), at most 513 bytes. Every failure is
/// `Unreadable` — fail closed.
pub(crate) fn snapshot_claim(dir: &Path, mine: &dyn Fn(&Path) -> bool) -> ClaimSnapshot {
    let claim = claim_path(dir);
    let id = file_id(&claim);
    let raw = match std::fs::symlink_metadata(&claim) {
        Err(e) if e.kind() == io::ErrorKind::NotFound => RawClaim::Absent,
        Err(_) => RawClaim::Unreadable { foreign_uid: false },
        Ok(meta) if !meta.is_file() => RawClaim::Unreadable { foreign_uid: false },
        Ok(_) if !mine(&claim) => RawClaim::Unreadable { foreign_uid: true },
        Ok(_) => match read_limited(&claim, id) {
            Ok(bytes) => RawClaim::Text(bytes),
            Err(_) => RawClaim::Unreadable { foreign_uid: false },
        },
    };
    ClaimSnapshot { raw, id }
}

/// The record in the claim beside `dir`: absent → `Unmarked`; anything the safe read refuses or the grammar does not
/// accept → `Unreadable`. `mine` is the kernel-owner question as a seam: a second uid cannot be had in a test.
pub(crate) fn read_claim(dir: &Path, mine: &dyn Fn(&Path) -> bool) -> Claim {
    match snapshot_claim(dir, mine).raw {
        RawClaim::Absent => Claim::Unmarked,
        RawClaim::Unreadable { .. } => Claim::Unreadable,
        RawClaim::Text(bytes) => parse_owner(&bytes),
    }
}

/// Write `bytes` as a NEW file: `create_new`, so a claim that is already there is somebody else's name. A claim that
/// could not be written whole is removed again — it is ours, the create was exclusive.
pub(crate) fn write_claim_exclusive(claim: &Path, bytes: &[u8]) -> io::Result<()> {
    let mut options = OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    let mut file = options.open(claim)?;
    if let Err(e) = file.write_all(bytes).and_then(|()| file.flush()) {
        drop(file);
        let _ = std::fs::remove_file(claim);
        return Err(e);
    }
    Ok(())
}

/// One attempt to remove the claim: only while it still holds exactly `written` and, when `captured` is the identity
/// the claim had right after this run wrote it, only while the file at the path is STILL that file. A claim that is
/// not ours any more (another text, another file, gone) is `Ok`: there is nothing of ours to remove.
fn remove_claim_once(
    claim: &Path,
    written: &[u8],
    captured: Option<FileId>,
    between: &mut dyn FnMut(),
) -> io::Result<()> {
    let meta = std::fs::symlink_metadata(claim)?;
    if !meta.is_file() {
        return Ok(());
    }
    let before = file_id(claim);
    if read_limited(claim, before)? != written {
        return Ok(());
    }
    between(); // a seam: what happens between the read and the unlink
               // The same file we read (and, when known, the file this run wrote), not a claim swapped in since.
    let now = file_id(claim);
    let ours = match (before, now) {
        (Some(before), Some(now)) => before == now && captured.is_none_or(|id| id == now),
        _ => false,
    };
    if ours {
        std::fs::remove_file(claim)?;
    }
    Ok(())
}

/// Remove the claim beside `dir` only while it still holds exactly `written` — a claim another run put there since
/// is not ours to remove — and, when `captured` is the identity the claim had right after this run wrote it, only
/// while the file at the path is STILL that file: a claim with the same text but another identity is another run's
/// (siblings under one `TESTSTODELETE_RUN` write identical records). Five tries 100 ms apart (rule §2: the family's
/// figure; a Windows scanner holds a fresh file for a moment). Never panics; `true` when nothing of ours is left to
/// remove, `false` after ONE stderr line when it gave up — the claim stays for the sweep, and the caller decides
/// whether to try again.
pub(crate) fn remove_claim_if_still(dir: &Path, written: &[u8], captured: Option<FileId>) -> bool {
    remove_claim_with_hook(dir, written, captured, &mut || {})
}

/// `remove_claim_if_still` with a seam that runs between the read of the claim and the comparison of its identity.
pub(crate) fn remove_claim_with_hook(
    dir: &Path,
    written: &[u8],
    captured: Option<FileId>,
    between: &mut dyn FnMut(),
) -> bool {
    let claim = claim_path(dir);
    match remove_with_retries(
        &claim,
        |claim| remove_claim_once(claim, written, captured, between),
        |_| {},
    ) {
        Ok(()) => true,
        Err(e) => {
            write_line(
                &mut StderrOut,
                &format!(
                    "test_scratch: could not remove the claim {} ({e}) — not fatal: the next run's start-of-run sweep collects it",
                    claim.display()
                ),
            );
            false
        }
    }
}
