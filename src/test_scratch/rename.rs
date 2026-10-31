//! Moving a directory under a name nobody holds: the move is EXCLUSIVE or it does not happen (rule §4).
//!
//! `rename(2)` of a directory onto an EMPTY directory replaces it, and so does Rust's `std::fs::rename` on Windows. A
//! quarantine or a restore that used it would destroy an empty directory somebody made at the drawn name, with no trace.
//! So: Linux asks the kernel for `renameat2(RENAME_NOREPLACE)`, which refuses ANY existing entry atomically; where a
//! file system answers `EINVAL` (it does not know the flag) or `ENOSYS`, a name is RESERVED instead with an exclusive
//! `mkdir` and the directory is renamed over that placeholder of our own; Windows calls `MoveFileExW` with no flags.
//! Every refusal because the name is taken is `ErrorKind::AlreadyExists`, which the caller answers with a fresh draw.

use std::io;
use std::path::Path;

/// Move `from` to `to` only if `to` does not exist. `Err(AlreadyExists)` when it does; any other error is the
/// operating system's own and nothing was moved.
pub(crate) fn move_without_replace(from: &Path, to: &Path) -> io::Result<()> {
    #[cfg(target_os = "linux")]
    {
        match renameat2_noreplace(from, to) {
            Err(e) if matches!(e.raw_os_error(), Some(libc::EINVAL | libc::ENOSYS)) => {
                move_via_placeholder(from, to)
            }
            other => other,
        }
    }
    #[cfg(all(unix, not(target_os = "linux")))]
    {
        move_via_placeholder(from, to)
    }
    #[cfg(windows)]
    {
        super::win::move_without_replace(from, to)
    }
    #[cfg(not(any(unix, windows)))]
    {
        let _ = (from, to);
        Err(io::Error::other(
            "no exclusive move on this operating system",
        ))
    }
}

/// `renameat2(AT_FDCWD, from, AT_FDCWD, to, RENAME_NOREPLACE)` by the raw system call: no libc wrapper is assumed.
#[cfg(target_os = "linux")]
fn renameat2_noreplace(from: &Path, to: &Path) -> io::Result<()> {
    use std::ffi::CString;
    use std::os::unix::ffi::OsStrExt;
    const RENAME_NOREPLACE: libc::c_uint = 1;
    let cstring = |path: &Path| {
        CString::new(path.as_os_str().as_bytes())
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "a path holds a NUL"))
    };
    let (from, to) = (cstring(from)?, cstring(to)?);
    // SAFETY: both pointers are NUL-terminated strings that outlive the call; renameat2 reads them and nothing else.
    let status = unsafe {
        libc::syscall(
            libc::SYS_renameat2,
            libc::AT_FDCWD,
            from.as_ptr(),
            libc::AT_FDCWD,
            to.as_ptr(),
            RENAME_NOREPLACE,
        )
    };
    if status == 0 {
        Ok(())
    } else {
        Err(io::Error::last_os_error())
    }
}

/// The fallback: reserve `to` with an exclusive `mkdir` (`AlreadyExists` when anything is there, a directory, a file
/// or a link), then rename `from` over the placeholder of our own. A rename that fails removes the placeholder, but
/// only while it is still the directory this call made (same device, inode and change time) — a name taken over in
/// between is never touched.
#[cfg(unix)]
pub(crate) fn move_via_placeholder(from: &Path, to: &Path) -> io::Result<()> {
    use std::os::unix::fs::DirBuilderExt;
    std::fs::DirBuilder::new().mode(0o700).create(to)?;
    let placeholder = super::identity::file_id(to);
    match std::fs::rename(from, to) {
        Ok(()) => Ok(()),
        Err(e) => {
            if super::identity::same_id(super::identity::file_id(to), placeholder) {
                let _ = std::fs::remove_dir(to);
            }
            Err(e)
        }
    }
}
