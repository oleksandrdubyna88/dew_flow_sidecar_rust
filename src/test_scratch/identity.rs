//! What identifies an entry at the moment of removal: never a name read earlier (rule §4).

use std::path::Path;
use std::time::SystemTime;

/// The device, the inode and the change time of an entry ITSELF (a link is not followed). The change time exists
/// because a freed inode number is reused at once on some file systems: a folder swapped for a fresh one under the
/// same name can carry the same `(dev, ino)`, never the same change time.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct FileId {
    dev: u64,
    ino: u64,
    changed: (i64, i64),
}

impl FileId {
    /// The same device and inode, whatever the change time says. A `rename` updates the change time of what it moves
    /// (measured on tmpfs; POSIX allows it everywhere), so the check AFTER the quarantine rename uses this.
    pub(crate) fn same_inode(&self, other: &FileId) -> bool {
        self.dev == other.dev && self.ino == other.ino
    }
}

/// Two identities are the same only when both could be read and agree on everything, the change time included: one
/// that cannot be read never compares equal (rule §4), not even to another that cannot.
pub(crate) fn same_id(a: Option<FileId>, b: Option<FileId>) -> bool {
    matches!((a, b), (Some(a), Some(b)) if a == b)
}

/// The identity of `path` itself, or none when it cannot be read — which never compares equal to anything (rule §4).
#[cfg(unix)]
pub(crate) fn file_id(path: &Path) -> Option<FileId> {
    use std::os::unix::fs::MetadataExt;
    let meta = std::fs::symlink_metadata(path).ok()?;
    Some(FileId {
        dev: meta.dev(),
        ino: meta.ino(),
        changed: (meta.ctime(), meta.ctime_nsec()),
    })
}

#[cfg(windows)]
pub(crate) fn file_id(path: &Path) -> Option<FileId> {
    let (dev, ino, changed) = super::win::file_identity(path)?;
    Some(FileId {
        dev,
        ino,
        changed: (changed, 0),
    })
}

#[cfg(not(any(unix, windows)))]
pub(crate) fn file_id(_path: &Path) -> Option<FileId> {
    None
}

/// When the entry's inode last changed (a rename counts): the age of a quarantine is its rename, not its content.
#[cfg(unix)]
pub(crate) fn changed_at(path: &Path) -> Option<SystemTime> {
    use std::os::unix::fs::MetadataExt;
    let meta = std::fs::symlink_metadata(path).ok()?;
    let seconds = u64::try_from(meta.ctime()).ok()?;
    let nanos = u32::try_from(meta.ctime_nsec()).ok()?;
    Some(SystemTime::UNIX_EPOCH + std::time::Duration::new(seconds, nanos))
}

#[cfg(windows)]
pub(crate) fn changed_at(path: &Path) -> Option<SystemTime> {
    let (_, _, changed) = super::win::file_identity(path)?;
    // FILETIME ticks (100 ns since 1601) of the ChangeTime.
    let ticks = u64::try_from(changed).ok()?;
    let since_unix = ticks.checked_sub(116_444_736_000_000_000)?;
    Some(SystemTime::UNIX_EPOCH + std::time::Duration::from_nanos(since_unix.checked_mul(100)?))
}

#[cfg(not(any(unix, windows)))]
pub(crate) fn changed_at(_path: &Path) -> Option<SystemTime> {
    None
}
