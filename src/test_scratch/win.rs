//! The Windows half of the owner: the host, the process start, liveness, and the kernel-owner key (both halves).
//!
//! The kernel owner of an entry is the caller's only when its resolved path is inside the caller's profile AND the
//! owner SID of the entry itself is the caller's user SID, read on ONE handle that is opened on the entry ITSELF —
//! a junction or symlink is never followed (`FILE_FLAG_OPEN_REPARSE_POINT`). An elevated shell's new files belong to
//! `BUILTIN\Administrators`, so they fail closed here: listed, never touched (owner decision 2 of the E4 plan).
//! Every failure on the way to an answer is `false` / `None` / `Unknown`.

use super::owner::Probe;
use std::ffi::c_void;
use std::fs::{File, OpenOptions};
use std::os::windows::fs::OpenOptionsExt;
use std::os::windows::io::AsRawHandle;
use std::path::{Component, Path};
use windows::core::{HRESULT, PCWSTR, PWSTR};
use windows::Win32::Foundation::{
    CloseHandle, LocalFree, ERROR_INVALID_PARAMETER, ERROR_NO_TOKEN, FILETIME, HANDLE, HLOCAL,
    STILL_ACTIVE,
};
use windows::Win32::Security::Authorization::{
    ConvertSidToStringSidW, GetSecurityInfo, SE_FILE_OBJECT,
};
use windows::Win32::Security::{
    EqualSid, GetLengthSid, GetTokenInformation, TokenElevation, TokenUser,
    OWNER_SECURITY_INFORMATION, PSECURITY_DESCRIPTOR, PSID, TOKEN_ELEVATION, TOKEN_QUERY,
    TOKEN_USER,
};
use windows::Win32::Storage::FileSystem::{
    FileBasicInfo, GetFileInformationByHandle, GetFileInformationByHandleEx, MoveFileExW,
    BY_HANDLE_FILE_INFORMATION, FILE_ATTRIBUTE_REPARSE_POINT, FILE_BASIC_INFO,
    FILE_FLAG_BACKUP_SEMANTICS, FILE_FLAG_OPEN_REPARSE_POINT, MOVE_FILE_FLAGS,
};
use windows::Win32::System::SystemInformation::{ComputerNameDnsHostname, GetComputerNameExW};
use windows::Win32::System::Threading::{
    GetCurrentProcess, GetCurrentThread, GetExitCodeProcess, GetProcessTimes, OpenProcess,
    OpenProcessToken, OpenThreadToken, PROCESS_QUERY_LIMITED_INFORMATION,
};

/// Move `from` to `to` ONLY if `to` does not exist: `MoveFileExW` with no flags, which — unlike `std::fs::rename`,
/// whose Windows implementation replaces an existing EMPTY directory (it asks for replace-if-exists semantics) —
/// fails with `ERROR_ALREADY_EXISTS` on any existing entry (`ErrorKind::AlreadyExists`). A move that replaced an empty
/// directory would put a folder over a name somebody else made, which rule §4 forbids; the callers (a restore, the
/// rename into quarantine) answer the refusal with a fresh name.
pub(super) fn move_without_replace(from: &Path, to: &Path) -> std::io::Result<()> {
    use std::os::windows::ffi::OsStrExt;
    let wide =
        |path: &Path| -> Vec<u16> { path.as_os_str().encode_wide().chain(Some(0)).collect() };
    let (from, to) = (wide(from), wide(to));
    // SAFETY: both buffers are NUL-terminated UTF-16 that outlive the call.
    unsafe {
        MoveFileExW(
            PCWSTR(from.as_ptr()),
            PCWSTR(to.as_ptr()),
            MOVE_FILE_FLAGS(0),
        )
    }
    .map_err(|e| {
        // HRESULT_FROM_WIN32 puts the Win32 code in the low 16 bits of 0x8007xxxx.
        let code = e.code().0 as u32;
        let win32 = if code >> 16 == 0x8007 {
            code & 0xFFFF
        } else {
            code
        };
        std::io::Error::from_raw_os_error(win32 as i32)
    })
}

const READ_CONTROL: u32 = 0x0002_0000;
const FILE_READ_ATTRIBUTES: u32 = 0x0000_0080;
/// 100 ns ticks between 1601-01-01 and 1970-01-01.
const UNIX_EPOCH_TICKS: u64 = 116_444_736_000_000_000;

/// The DNS host name — `GetComputerNameExW(ComputerNameDnsHostname)`, never `%COMPUTERNAME%` (NetBIOS: upper-case
/// and cut at 15 characters).
pub(super) fn dns_host_name() -> Option<String> {
    let mut size = 0u32;
    // SAFETY: the first call only reports the size needed (ERROR_MORE_DATA); the second writes at most `size` units.
    unsafe {
        let _ = GetComputerNameExW(ComputerNameDnsHostname, None, &mut size);
        if size == 0 {
            return None;
        }
        let mut buffer = vec![0u16; size as usize];
        GetComputerNameExW(
            ComputerNameDnsHostname,
            Some(PWSTR(buffer.as_mut_ptr())),
            &mut size,
        )
        .ok()?;
        String::from_utf16(&buffer[..size as usize]).ok()
    }
}

fn unix_seconds(time: FILETIME) -> Option<u64> {
    let ticks = (u64::from(time.dwHighDateTime) << 32) | u64::from(time.dwLowDateTime);
    Some(ticks.checked_sub(UNIX_EPOCH_TICKS)? / 10_000_000)
}

/// `GetProcessTimes` creation time of `process` as integer Unix seconds, UTC: the OS's record of that pid.
fn process_start(process: HANDLE) -> Option<u64> {
    let mut creation = FILETIME::default();
    let (mut exit, mut kernel, mut user) = (
        FILETIME::default(),
        FILETIME::default(),
        FILETIME::default(),
    );
    // SAFETY: four valid out-pointers to locals; `process` is a live handle or the current-process pseudo-handle.
    unsafe { GetProcessTimes(process, &mut creation, &mut exit, &mut kernel, &mut user) }.ok()?;
    unix_seconds(creation)
}

pub(super) fn own_start() -> Option<u64> {
    // SAFETY: the pseudo-handle of the current process needs no closing.
    process_start(unsafe { GetCurrentProcess() })
}

/// Is `pid` alive? `ERROR_INVALID_PARAMETER` from `OpenProcess` is "no such pid" — the one `Gone`; any other failure,
/// and a pid above `u32::MAX` (never truncated), is `Unknown`. An opened process whose exit code is not `STILL_ACTIVE`
/// is `Gone`. The handle is always closed.
pub(super) fn probe(pid: u64) -> Probe {
    let Ok(pid) = u32::try_from(pid) else {
        return Probe::Unknown;
    };
    // SAFETY: `handle` is closed exactly once, on every path that opened it.
    unsafe {
        let handle = match OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, false, pid) {
            Ok(handle) => handle,
            Err(e) if e.code() == HRESULT::from_win32(ERROR_INVALID_PARAMETER.0) => {
                return Probe::Gone
            }
            Err(_) => return Probe::Unknown,
        };
        let mut code = 0u32;
        let answer = match GetExitCodeProcess(handle, &mut code) {
            Err(_) => Probe::Unknown,
            Ok(()) if code != STILL_ACTIVE.0 as u32 => Probe::Gone,
            Ok(()) => Probe::Alive {
                start: process_start(handle),
            },
        };
        let _ = CloseHandle(handle);
        answer
    }
}

// ---------- the caller ----------

/// The caller's token: the THREAD token first (an impersonating thread is the caller), the process token only on
/// `ERROR_NO_TOKEN`. The caller closes it.
fn caller_token() -> Option<HANDLE> {
    let mut token = HANDLE::default();
    // SAFETY: valid out-pointer; the pseudo-handles need no closing.
    unsafe {
        match OpenThreadToken(GetCurrentThread(), TOKEN_QUERY, true, &mut token) {
            Ok(()) => Some(token),
            Err(e) if e.code() == HRESULT::from_win32(ERROR_NO_TOKEN.0) => {
                OpenProcessToken(GetCurrentProcess(), TOKEN_QUERY, &mut token).ok()?;
                Some(token)
            }
            Err(_) => None,
        }
    }
}

fn sid_bytes(sid: PSID) -> Vec<u8> {
    // SAFETY: `sid` points at a valid SID; GetLengthSid is its exact byte length.
    unsafe { std::slice::from_raw_parts(sid.0.cast::<u8>(), GetLengthSid(sid) as usize) }.to_vec()
}

/// The user SID of `token` (`TokenUser`), copied out; the token is closed.
fn user_sid_of(token: HANDLE) -> Option<Vec<u8>> {
    let mut needed = 0u32;
    // SAFETY: the first call reports the size; the buffer is u64-aligned for the TOKEN_USER (it holds a pointer) and
    // lives past the copy of the SID out of it; the token is closed once.
    let sid = unsafe {
        let _ = GetTokenInformation(token, TokenUser, None, 0, &mut needed);
        let mut buffer = vec![0u64; (needed as usize).div_ceil(8).max(1)];
        let read = GetTokenInformation(
            token,
            TokenUser,
            Some(buffer.as_mut_ptr().cast::<c_void>()),
            needed,
            &mut needed,
        );
        let sid = read
            .is_ok()
            .then(|| sid_bytes((*buffer.as_ptr().cast::<TOKEN_USER>()).User.Sid));
        let _ = CloseHandle(token);
        sid
    };
    sid.filter(|bytes| !bytes.is_empty())
}

fn read_caller_sid() -> Option<Vec<u8>> {
    user_sid_of(caller_token()?)
}

/// The PROCESS token's user SID, whatever a thread impersonates: what scenario 20 compares `caller_sid` against.
pub(crate) fn process_user_sid() -> Option<Vec<u8>> {
    let mut token = HANDLE::default();
    // SAFETY: valid out-pointer; the pseudo-handle needs no closing.
    unsafe { OpenProcessToken(GetCurrentProcess(), TOKEN_QUERY, &mut token) }.ok()?;
    user_sid_of(token)
}

/// The caller's user SID, read on EVERY call: the thread token first, the process token only on `ERROR_NO_TOKEN`
/// (`caller_token`). A thread that impersonates is the caller for as long as it does, and another thread of the same
/// process may not be — so the answer is never kept process-wide (review round 1: a `OnceLock` kept the first
/// thread's SID for every later thread). Empty when it could not be read — which makes every `is_mine` answer `false`.
pub(crate) fn caller_sid() -> Vec<u8> {
    read_caller_sid().unwrap_or_default()
}

/// Is the caller's token elevated? `None` when it cannot be read.
pub(crate) fn is_elevated() -> Option<bool> {
    let token = caller_token()?;
    let mut elevation = TOKEN_ELEVATION::default();
    let mut returned = 0u32;
    // SAFETY: valid out-pointers to locals of the exact size; the token is closed once.
    let read = unsafe {
        let read = GetTokenInformation(
            token,
            TokenElevation,
            Some(std::ptr::from_mut(&mut elevation).cast::<c_void>()),
            std::mem::size_of::<TOKEN_ELEVATION>() as u32,
            &mut returned,
        );
        let _ = CloseHandle(token);
        read
    };
    read.ok().map(|()| elevation.TokenIsElevated != 0)
}

pub(crate) fn sid_text(sid: &[u8]) -> String {
    let mut text = PWSTR::null();
    // SAFETY: `sid` is a copy of a valid SID; on success `text` is a LocalAlloc'd string, read once, freed once.
    unsafe {
        if ConvertSidToStringSidW(PSID(sid.as_ptr().cast_mut().cast()), &mut text).is_err()
            || text.is_null()
        {
            return "<unprintable SID>".to_string();
        }
        let shown = text.to_string().unwrap_or_else(|_| "<not UTF-16>".into());
        let _ = LocalFree(Some(HLOCAL(text.0.cast())));
        shown
    }
}

pub(crate) fn sid_equal(a: &[u8], b: &[u8]) -> bool {
    if a.is_empty() || b.is_empty() {
        return false;
    }
    // SAFETY: both slices are copies of valid SIDs.
    unsafe {
        EqualSid(
            PSID(a.as_ptr().cast_mut().cast()),
            PSID(b.as_ptr().cast_mut().cast()),
        )
    }
    .is_ok()
}

// ---------- one handle on the entry itself ----------

/// The entry opened ITSELF: a junction or symlink is not followed (`FILE_FLAG_OPEN_REPARSE_POINT`), a directory can
/// be opened (`FILE_FLAG_BACKUP_SEMANTICS`), and only the rights the two questions need are asked for.
fn open_entry(path: &Path) -> std::io::Result<File> {
    OpenOptions::new()
        .access_mode(READ_CONTROL | FILE_READ_ATTRIBUTES)
        .share_mode(7)
        .custom_flags(FILE_FLAG_OPEN_REPARSE_POINT.0 | FILE_FLAG_BACKUP_SEMANTICS.0)
        .open(path)
}

fn handle_of(file: &File) -> HANDLE {
    HANDLE(file.as_raw_handle())
}

fn information(file: &File) -> Option<BY_HANDLE_FILE_INFORMATION> {
    let mut info = BY_HANDLE_FILE_INFORMATION::default();
    // SAFETY: a live handle and a valid out-pointer.
    unsafe { GetFileInformationByHandle(handle_of(file), &mut info) }.ok()?;
    Some(info)
}

/// The volume serial, the file index and the `ChangeTime` (FILETIME ticks) of the entry itself.
pub(super) fn file_identity(path: &Path) -> Option<(u64, u64, i64)> {
    let file = open_entry(path).ok()?;
    let info = information(&file)?;
    let mut basic = FILE_BASIC_INFO::default();
    // SAFETY: a live handle and a buffer of exactly the size of the class asked for.
    unsafe {
        GetFileInformationByHandleEx(
            handle_of(&file),
            FileBasicInfo,
            std::ptr::from_mut(&mut basic).cast::<c_void>(),
            std::mem::size_of::<FILE_BASIC_INFO>() as u32,
        )
    }
    .ok()?;
    let index = (u64::from(info.nFileIndexHigh) << 32) | u64::from(info.nFileIndexLow);
    Some((
        u64::from(info.dwVolumeSerialNumber),
        index,
        basic.ChangeTime,
    ))
}

/// The owner SID of the entry on `file`'s handle, copied out; none when it cannot be read.
fn owner_sid_of_handle(file: &File) -> Option<Vec<u8>> {
    let mut owner = PSID::default();
    let mut descriptor = PSECURITY_DESCRIPTOR::default();
    // SAFETY: a live handle and valid out-pointers; the owner SID points INTO the descriptor, so it is copied before
    // the descriptor is freed, once.
    unsafe {
        let status = GetSecurityInfo(
            handle_of(file),
            SE_FILE_OBJECT,
            OWNER_SECURITY_INFORMATION,
            Some(&mut owner),
            None,
            None,
            None,
            Some(&mut descriptor),
        );
        let copied = (status.0 == 0 && !owner.is_invalid()).then(|| sid_bytes(owner));
        if !descriptor.0.is_null() {
            let _ = LocalFree(Some(HLOCAL(descriptor.0)));
        }
        copied
    }
}

/// The owner SID of the entry at `path` (the entry itself, never a link's target).
pub(crate) fn owner_sid_of(path: &Path) -> Option<Vec<u8>> {
    owner_sid_of_handle(&open_entry(path).ok()?)
}

/// Is `child` strictly inside `parent`? Both already resolved; compared case-insensitively by component, so a longer
/// sibling name (`C:\Users\alice2`) is not inside `C:\Users\alice`.
fn inside(child: &Path, parent: &Path) -> bool {
    let key = |component: Component<'_>| component.as_os_str().to_string_lossy().to_lowercase();
    let child: Vec<String> = child.components().map(key).collect();
    let parent: Vec<String> = parent.components().map(key).collect();
    child.len() > parent.len() && child[..parent.len()] == parent[..]
}

/// Owner decision 2 of the E4 plan: BOTH keys, on one handle — the entry is not a reparse point, its resolved path is
/// inside the resolved `%USERPROFILE%`, and its owner SID is the caller's user SID. Either unreadable → `false`.
pub(super) fn is_mine(path: &Path) -> bool {
    let sid = caller_sid();
    if sid.is_empty() {
        return false;
    }
    let Some(profile) = std::env::var_os("USERPROFILE").and_then(|p| std::fs::canonicalize(p).ok())
    else {
        return false;
    };
    let Ok(file) = open_entry(path) else {
        return false;
    };
    let Some(info) = information(&file) else {
        return false;
    };
    if info.dwFileAttributes & FILE_ATTRIBUTE_REPARSE_POINT.0 != 0 {
        return false;
    }
    let Ok(resolved) = std::fs::canonicalize(path) else {
        return false;
    };
    inside(&resolved, &profile)
        && owner_sid_of_handle(&file).is_some_and(|owner| sid_equal(&owner, &sid))
}
