//! Scenarios that only Windows can run (0, 10, 14 and 20 of the E4 plan, S2). Compiled out elsewhere.

#![cfg(windows)]

use super::sweeping::{folder, gone_claim, later, probe_of};
use super::*;
use std::collections::HashSet;
use std::fs::OpenOptions;
use std::os::windows::fs::OpenOptionsExt;
use windows::Win32::Foundation::CloseHandle;
use windows::Win32::System::Diagnostics::ToolHelp::{
    CreateToolhelp32Snapshot, Process32FirstW, Process32NextW, PROCESSENTRY32W, TH32CS_SNAPPROCESS,
};

/// Every pid a ToolHelp snapshot shows right now.
fn process_ids() -> HashSet<u32> {
    let mut ids = HashSet::new();
    // SAFETY: the snapshot handle is closed once; `entry` is a valid out-pointer with its size set as the API wants.
    unsafe {
        let snapshot = CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0).expect("a process snapshot");
        let mut entry = PROCESSENTRY32W {
            dwSize: std::mem::size_of::<PROCESSENTRY32W>() as u32,
            ..Default::default()
        };
        if Process32FirstW(snapshot, &mut entry).is_ok() {
            loop {
                ids.insert(entry.th32ProcessID);
                if Process32NextW(snapshot, &mut entry).is_err() {
                    break;
                }
            }
        }
        let _ = CloseHandle(snapshot);
    }
    ids
}

// ---------- 10 ----------

#[test]
fn a_scratch_held_open_stays_prefixed_and_claimed_and_says_so_once() {
    let outer = TestScratch::new("held");
    let inner = TestScratch::new_in(outer.path(), "inner");
    // `share_mode(0)`: Rust's ordinary `File::open` passes FILE_SHARE_DELETE, so an ordinary open does NOT block the
    // delete (the first version of the log_segments held-file test asserted nothing).
    let held = OpenOptions::new()
        .write(true)
        .create_new(true)
        .share_mode(0)
        .open(inner.path().join("held.txt"))
        .unwrap();

    let mut buf = Vec::new();
    assert!(
        !teardown(inner.path(), &mut buf),
        "the teardown must report that it gave up"
    );
    let line = String::from_utf8(buf).unwrap();
    assert_eq!(line.lines().count(), 1, "ONE line: {line}");
    assert!(line.contains(&inner.path().display().to_string()), "{line}");
    assert!(inner.path().is_dir(), "the folder stays");
    assert!(claim_path(inner.path()).is_file(), "and stays claimed");
    drop(held);

    // The retry is real: the handle is released after the first failed attempt, the second attempt removes the folder.
    let mut held = Some(
        OpenOptions::new()
            .write(true)
            .create_new(true)
            .share_mode(0)
            .open(inner.path().join("held-again.txt"))
            .unwrap(),
    );
    let mut attempts = 0;
    let result = remove_with_retries(
        inner.path(),
        |p| {
            attempts += 1;
            std::fs::remove_dir_all(p)
        },
        |_| drop(held.take()),
    );
    assert!(result.is_ok(), "{result:?}");
    assert_eq!(
        attempts, 2,
        "released after attempt 1, removed at attempt 2"
    );
    assert!(!inner.path().exists());
}

// ---------- 14, the Windows half ----------

#[test]
fn the_windows_probe_refuses_a_truncated_pid_and_finds_a_dead_one() {
    let own = u64::from(std::process::id());
    assert_eq!(
        probe(u64::from(u32::MAX) + 1),
        Probe::Unknown,
        "never truncated to a pid that exists"
    );

    // MEASURED: pids are handle-table indices and NT ignores the two tag bits, so `4n + 2` may open `4n`.
    println!("measured: probe(own pid {own} + 2) = {:?}", probe(own + 2));

    // A dead pid is a `4n` whose four aliases `4n..4n+3` are all absent from a snapshot taken before AND after.
    let before = process_ids();
    let dead = (10_000u32..4_000_000)
        .step_by(4)
        .find(|n| (*n..*n + 4).all(|alias| !before.contains(&alias)))
        .expect("an unused pid exists");
    let after = process_ids();
    assert!(
        (dead..dead + 4).all(|alias| !after.contains(&alias)),
        "the premise holds after, too"
    );
    assert_eq!(probe(u64::from(dead)), Probe::Gone, "pid {dead}");
}

// ---------- 20 ----------

#[test]
fn the_kernel_owner_answer_matches_what_was_measured() {
    let scratch = TestScratch::new("kernel-owner");
    let path = scratch.path();
    let owner = win::owner_sid_of(path).expect("the owner SID of a fresh folder is readable");
    let caller = win::caller_sid();
    assert_eq!(
        Some(caller.to_vec()),
        win::process_user_sid(),
        "no impersonation is active in a test: the thread token is the process token"
    );
    let equal = win::sid_equal(&owner, caller);
    let profile = std::env::var_os("USERPROFILE").and_then(|p| std::fs::canonicalize(p).ok());
    let inside = profile.is_some_and(|profile| {
        std::fs::canonicalize(path).is_ok_and(|real| {
            let lower = |p: &Path| p.to_string_lossy().to_lowercase();
            lower(&real).starts_with(&format!("{}\\", lower(&profile).trim_end_matches('\\')))
        })
    });
    println!(
        "measured: owner {} / caller {} / equal {equal} / inside the profile {inside} / elevated {:?}",
        win::sid_text(&owner),
        win::sid_text(caller),
        win::is_elevated()
    );
    assert_eq!(
        is_mine(path),
        equal && inside,
        "the helper's answer is exactly the two keys"
    );

    // The sweep, with the REAL kernel-owner question: removed when the keys agree, kept as `foreign-uid` when not
    // (an elevated shell's files belong to BUILTIN\Administrators).
    let root = TestScratch::new_in(path, "sweep-root");
    let leftover = folder(
        root.path(),
        &name_with("left", "abc123"),
        Some(&gone_claim()),
    );
    let report = sweep(
        root.path(),
        later(),
        SWEEP_MIN_AGE,
        probe_of,
        is_mine,
        &mut Hooks::default(),
    );
    if equal && inside {
        assert!(report.removed.contains(&leftover), "{report:?}");
    } else {
        assert!(report.removed.is_empty(), "{report:?}");
        assert!(
            report
                .kept
                .iter()
                .any(|(p, why)| p == &leftover && why == "foreign-uid"),
            "{report:?}"
        );
        assert!(
            report
                .listed
                .iter()
                .any(|l| l.starts_with("foreign: folder")),
            "{:?}",
            report.listed
        );
        assert!(leftover.is_dir(), "untouched");
    }

    // In both cases the value is re-created and its Drop removes it: teardown needs no `is_mine`.
    let again = TestScratch::new_in(root.path(), "again");
    let again_path = again.path().to_path_buf();
    drop(again);
    assert!(!again_path.exists() && !claim_path(&again_path).exists());
}

// ---------- 0 ----------

/// Step 0 of the E4 plan (`todo/PLAN_test_leftovers_E4.md`, S2): what does Windows say owns a file this
/// process just created in the temp root, who is the caller, and is the token elevated? The rule's second key
/// (the kernel owner is the caller) is only worth anything if a normal run's new files are the caller's; the
/// expectation that a hosted (elevated) runner's files belong to `BUILTIN\Administrators` (`S-1-5-32-544`) was
/// UNVERIFIED when the plan was written, so this prints the facts and never fails. Run it once with
/// `--nocapture` on the dml leg and once on a developer's shell; scenario 20 asserts whatever was measured.
#[test]
fn the_windows_owner_of_a_new_temp_file_is_printed() {
    use std::ffi::c_void;
    use std::os::windows::fs::OpenOptionsExt;
    use std::os::windows::io::AsRawHandle;
    use windows::core::{HRESULT, PWSTR};
    use windows::Win32::Foundation::{CloseHandle, LocalFree, ERROR_NO_TOKEN, HANDLE, HLOCAL};
    use windows::Win32::Security::Authorization::{
        ConvertSidToStringSidW, GetSecurityInfo, SE_FILE_OBJECT,
    };
    use windows::Win32::Security::{
        EqualSid, GetTokenInformation, TokenElevation, TokenUser, OWNER_SECURITY_INFORMATION, PSID,
        TOKEN_ELEVATION, TOKEN_QUERY, TOKEN_USER,
    };
    use windows::Win32::System::Threading::{
        GetCurrentProcess, GetCurrentThread, OpenProcessToken, OpenThreadToken,
    };

    fn sid_text(sid: PSID) -> String {
        let mut text = PWSTR::null();
        // SAFETY: `sid` points at a valid SID for the duration of the call; on success `text` is a
        // LocalAlloc'd string that is read once and freed once below.
        unsafe {
            if ConvertSidToStringSidW(sid, &mut text).is_err() || text.is_null() {
                return "<ConvertSidToStringSidW failed>".to_string();
            }
            let shown = text
                .to_string()
                .unwrap_or_else(|_| "<not UTF-16>".to_string());
            let _ = LocalFree(Some(HLOCAL(text.0.cast())));
            shown
        }
    }

    // SAFETY: every pointer handed to a Win32 call below is to a local that outlives the call; handles are
    // closed exactly once; this test only reads and prints.
    unsafe {
        // The caller's token: the THREAD token first (an impersonating thread is the caller), the process
        // token only when the thread has none (D3).
        let mut token = HANDLE::default();
        let source = match OpenThreadToken(GetCurrentThread(), TOKEN_QUERY, true, &mut token) {
            Ok(()) => "thread token",
            Err(e) if e.code() == HRESULT::from_win32(ERROR_NO_TOKEN.0) => {
                match OpenProcessToken(GetCurrentProcess(), TOKEN_QUERY, &mut token) {
                    Ok(()) => "process token",
                    Err(e) => {
                        eprintln!("step0: OpenProcessToken failed: {e}");
                        return;
                    }
                }
            }
            Err(e) => {
                eprintln!("step0: OpenThreadToken failed: {e}");
                return;
            }
        };

        let mut needed = 0u32;
        let _ = GetTokenInformation(token, TokenUser, None, 0, &mut needed);
        // u64 elements: the buffer holds a TOKEN_USER, which carries a pointer and wants 8-byte alignment.
        let mut buffer = vec![0u64; (needed as usize).div_ceil(8).max(1)];
        let got = GetTokenInformation(
            token,
            TokenUser,
            Some(buffer.as_mut_ptr().cast::<c_void>()),
            needed,
            &mut needed,
        );
        let caller_sid = if got.is_ok() {
            (*buffer.as_ptr().cast::<TOKEN_USER>()).User.Sid
        } else {
            PSID::default()
        };

        let mut elevation = TOKEN_ELEVATION::default();
        let mut returned = 0u32;
        let elevated = GetTokenInformation(
            token,
            TokenElevation,
            Some(std::ptr::from_mut(&mut elevation).cast::<c_void>()),
            std::mem::size_of::<TOKEN_ELEVATION>() as u32,
            &mut returned,
        )
        .map(|()| elevation.TokenIsElevated != 0);

        // The file: made like any test makes one, in the raw temp root, removed before this test returns.
        let path = temp_root().join(format!("bge-step0-owner-probe-{}.tmp", std::process::id()));
        let file = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .share_mode(7)
            .open(&path);
        match file {
            Err(e) => eprintln!("step0: could not create {}: {e}", path.display()),
            Ok(file) => {
                let mut owner = PSID::default();
                let mut descriptor = windows::Win32::Security::PSECURITY_DESCRIPTOR::default();
                let status = GetSecurityInfo(
                    HANDLE(file.as_raw_handle()),
                    SE_FILE_OBJECT,
                    OWNER_SECURITY_INFORMATION,
                    Some(&mut owner),
                    None,
                    None,
                    None,
                    Some(&mut descriptor),
                );
                if status.0 != 0 || owner.is_invalid() {
                    eprintln!("step0: GetSecurityInfo failed with status {}", status.0);
                } else {
                    let equal = !caller_sid.is_invalid() && EqualSid(owner, caller_sid).is_ok();
                    eprintln!("step0: caller token      = {source}");
                    eprintln!("step0: caller SID        = {}", sid_text(caller_sid));
                    eprintln!("step0: file owner SID    = {}", sid_text(owner));
                    eprintln!("step0: owner == caller   = {equal}");
                    eprintln!("step0: token is elevated = {elevated:?}");
                }
                if !descriptor.0.is_null() {
                    let _ = LocalFree(Some(HLOCAL(descriptor.0)));
                }
                drop(file);
            }
        }
        if let Err(e) = std::fs::remove_file(&path) {
            if e.kind() != std::io::ErrorKind::NotFound {
                eprintln!("step0: could not remove {}: {e}", path.display());
            }
        }
        let _ = CloseHandle(token);
    }
}
