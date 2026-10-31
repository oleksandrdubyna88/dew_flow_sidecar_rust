//! Scenarios that only Windows can run (0, 10, 14 and 20 of the E4 plan, S2). Compiled out elsewhere.

#![cfg(windows)]

use super::sweeping::{folder, gone_claim, kept, later, probe_of, run_sweep};
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

    // A dead pid is a `4n` whose four aliases `4n..4n+3` are all absent from a snapshot taken before AND after. A
    // process that starts between the snapshots can still take the number, so the whole pick is made once more with
    // fresh snapshots before the scenario fails.
    let pick = || -> u32 {
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
        dead
    };
    let dead = pick();
    if probe(u64::from(dead)) != Probe::Gone {
        let again = pick();
        assert_eq!(
            probe(u64::from(again)),
            Probe::Gone,
            "pid {again} (after {dead})"
        );
    }
}

// ---------- the independent measurement ----------

/// What the raw Win32 API says about `path` — a folder or a file — by a code path that shares NOTHING with `win::*`:
/// its own token read, its own handle, its own `GetSecurityInfo`, and the SIDs compared by their STRING forms
/// (`ConvertSidToStringSidW`) instead of `EqualSid`. Step 0 prints it; scenario 20 asserts the helper against it, so
/// the assertion is not the helper agreeing with itself.
struct Measure {
    source: &'static str,
    caller: String,
    owner: String,
    elevated: Option<bool>,
}

fn measure_owner(path: &Path) -> Result<Measure, String> {
    use std::ffi::c_void;
    use std::os::windows::io::AsRawHandle;
    use windows::core::{HRESULT, PWSTR};
    use windows::Win32::Foundation::{CloseHandle, LocalFree, ERROR_NO_TOKEN, HANDLE, HLOCAL};
    use windows::Win32::Security::Authorization::{
        ConvertSidToStringSidW, GetSecurityInfo, SE_FILE_OBJECT,
    };
    use windows::Win32::Security::{
        GetTokenInformation, TokenElevation, TokenUser, OWNER_SECURITY_INFORMATION,
        PSECURITY_DESCRIPTOR, PSID, TOKEN_ELEVATION, TOKEN_QUERY, TOKEN_USER,
    };
    use windows::Win32::System::Threading::{
        GetCurrentProcess, GetCurrentThread, OpenProcessToken, OpenThreadToken,
    };

    struct Token(HANDLE);
    impl Drop for Token {
        fn drop(&mut self) {
            // SAFETY: the handle was opened by this measurement and is closed once.
            let _ = unsafe { CloseHandle(self.0) };
        }
    }

    fn text(sid: PSID) -> String {
        let mut text = PWSTR::null();
        // SAFETY: `sid` points at a valid SID for the duration of the call; on success `text` is a LocalAlloc'd
        // string that is read once and freed once below.
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

    // SAFETY: every pointer handed to a Win32 call below is to a local that outlives the call; handles are closed
    // exactly once; this only reads.
    unsafe {
        // The caller's token: the THREAD token first (an impersonating thread is the caller), the process token only
        // when the thread has none (D3).
        let mut raw = HANDLE::default();
        let source = match OpenThreadToken(GetCurrentThread(), TOKEN_QUERY, true, &mut raw) {
            Ok(()) => "thread token",
            Err(e) if e.code() == HRESULT::from_win32(ERROR_NO_TOKEN.0) => {
                OpenProcessToken(GetCurrentProcess(), TOKEN_QUERY, &mut raw)
                    .map_err(|e| format!("OpenProcessToken failed: {e}"))?;
                "process token"
            }
            Err(e) => return Err(format!("OpenThreadToken failed: {e}")),
        };
        let token = Token(raw);

        let mut needed = 0u32;
        let _ = GetTokenInformation(token.0, TokenUser, None, 0, &mut needed);
        // u64 elements: the buffer holds a TOKEN_USER, which carries a pointer and wants 8-byte alignment.
        let mut buffer = vec![0u64; (needed as usize).div_ceil(8).max(1)];
        GetTokenInformation(
            token.0,
            TokenUser,
            Some(buffer.as_mut_ptr().cast::<c_void>()),
            needed,
            &mut needed,
        )
        .map_err(|e| format!("GetTokenInformation(TokenUser) failed: {e}"))?;
        let caller = text((*buffer.as_ptr().cast::<TOKEN_USER>()).User.Sid);

        let mut elevation = TOKEN_ELEVATION::default();
        let mut returned = 0u32;
        let elevated = GetTokenInformation(
            token.0,
            TokenElevation,
            Some(std::ptr::from_mut(&mut elevation).cast::<c_void>()),
            std::mem::size_of::<TOKEN_ELEVATION>() as u32,
            &mut returned,
        )
        .ok()
        .map(|()| elevation.TokenIsElevated != 0);

        // READ_CONTROL on the entry itself; BACKUP_SEMANTICS so a folder opens too.
        let file = std::fs::OpenOptions::new()
            .access_mode(0x0002_0000)
            .share_mode(7)
            .custom_flags(0x0200_0000)
            .open(path)
            .map_err(|e| format!("could not open {}: {e}", path.display()))?;
        let mut owner = PSID::default();
        let mut descriptor = PSECURITY_DESCRIPTOR::default();
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
        let result = if status.0 != 0 || owner.is_invalid() {
            Err(format!("GetSecurityInfo failed with status {}", status.0))
        } else {
            Ok(Measure {
                source,
                caller,
                owner: text(owner),
                elevated,
            })
        };
        if !descriptor.0.is_null() {
            let _ = LocalFree(Some(HLOCAL(descriptor.0)));
        }
        result
    }
}

/// Is `path` (resolved) strictly inside the resolved `%USERPROFILE%`? A string comparison of the two resolved paths,
/// not `win::inside`.
fn inside_the_profile(path: &Path) -> bool {
    let profile = std::env::var_os("USERPROFILE").and_then(|p| std::fs::canonicalize(p).ok());
    profile.is_some_and(|profile| {
        std::fs::canonicalize(path).is_ok_and(|real| {
            let lower = |p: &Path| p.to_string_lossy().to_lowercase();
            lower(&real).starts_with(&format!("{}\\", lower(&profile).trim_end_matches('\\')))
        })
    })
}

/// Hand `path` to `sid` as its owner (`SetNamedSecurityInfoW`, owner only): allowed to the caller for its own user SID,
/// which is in its token.
fn set_owner(path: &Path, sid: &[u8]) {
    use std::os::windows::ffi::OsStrExt;
    use windows::core::PCWSTR;
    use windows::Win32::Security::Authorization::{SetNamedSecurityInfoW, SE_FILE_OBJECT};
    use windows::Win32::Security::{OWNER_SECURITY_INFORMATION, PSID};
    let wide: Vec<u16> = path.as_os_str().encode_wide().chain(Some(0)).collect();
    // SAFETY: the path is NUL-terminated UTF-16 and `sid` a copy of a valid SID; both outlive the call.
    let status = unsafe {
        SetNamedSecurityInfoW(
            PCWSTR(wide.as_ptr()),
            SE_FILE_OBJECT,
            OWNER_SECURITY_INFORMATION,
            Some(PSID(sid.as_ptr().cast_mut().cast())),
            None,
            None,
            None,
        )
    };
    assert_eq!(
        status.0,
        0,
        "SetNamedSecurityInfoW({}) failed",
        path.display()
    );
}

// ---------- 20 ----------

#[test]
fn the_kernel_owner_answer_matches_what_was_measured() {
    let scratch = TestScratch::new("kernel-owner");
    let path = scratch.path();
    let owner = win::owner_sid_of(path).expect("the owner SID of a fresh folder is readable");
    let caller = win::caller_sid();
    assert_eq!(
        Some(caller.clone()),
        win::process_user_sid(),
        "no impersonation is active in a test: the thread token is the process token"
    );
    // Review round 1: the expectation comes from Step 0's code path, which shares nothing with the helper's.
    let measured = measure_owner(path).expect("the independent measurement works");
    assert_eq!(
        win::sid_text(&owner),
        measured.owner,
        "the helper reads the owner the raw API reads"
    );
    assert_eq!(win::sid_text(&caller), measured.caller, "and the caller");
    assert_eq!(win::is_elevated(), measured.elevated, "and the elevation");
    let equal = measured.owner == measured.caller;
    let inside = inside_the_profile(path);
    println!(
        "measured: owner {} / caller {} ({}) / equal {equal} / inside the profile {inside} / elevated {:?}",
        measured.owner, measured.caller, measured.source, measured.elevated
    );
    assert_eq!(
        is_mine(path),
        equal && inside,
        "the helper's answer is exactly the two keys, as the independent path reads them"
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

/// Review round 1, the POSITIVE case: whatever account the run is (an elevated runner's new files belong to
/// `BUILTIN\Administrators`), a folder and its claim whose owner is set to the caller's own user SID are the
/// caller's by the REAL `is_mine`, and the sweep REMOVES a folder of a dead owner there — the key that opens the lock.
#[test]
fn a_folder_the_caller_owns_by_sid_is_mine_and_a_gone_owners_is_swept() {
    let scratch = TestScratch::new("positive-sid");
    let path = scratch.path();
    if !inside_the_profile(path) {
        eprintln!("skipped: the temp root {} is outside the user profile, so no folder there is ever mine", path.display());
        return;
    }
    let root = TestScratch::new_in(path, "sweep-root");
    let leftover = folder(
        root.path(),
        &name_with("mine", "abc123"),
        Some(&gone_claim()),
    );
    let caller = win::caller_sid();
    for entry in [&leftover, &claim_path(&leftover)] {
        set_owner(entry, &caller);
        assert_eq!(
            measure_owner(entry).expect("measured").owner,
            win::sid_text(&caller),
            "the owner is the caller's now, by the independent read"
        );
        assert!(
            is_mine(entry),
            "{} is mine by the real kernel-owner key",
            entry.display()
        );
    }
    let report = sweep(
        root.path(),
        later(),
        SWEEP_MIN_AGE,
        probe_of,
        is_mine,
        &mut Hooks::default(),
    );
    assert!(report.removed.contains(&leftover), "{report:?}");
    assert!(!leftover.exists() && !claim_path(&leftover).exists());
}

/// Review round 1: a folder a held file keeps from being renamed is NOT lost. `share_mode(0)` on a file inside makes
/// the rename into quarantine refuse: the folder is kept `could not remove`, nothing moved (no quarantine appears), and
/// once the handle is released the next sweep removes it.
#[test]
fn a_sweep_that_cannot_move_a_held_folder_keeps_it_and_the_next_one_removes_it() {
    let scratch = TestScratch::new("sweep-held");
    let root = scratch.path();
    let doomed = folder(root, &name_with("held", "abc123"), Some(&gone_claim()));
    let held = OpenOptions::new()
        .write(true)
        .create_new(true)
        .share_mode(0)
        .open(doomed.join("held.txt"))
        .unwrap();

    let report = run_sweep(root, later(), &mut Hooks::default());
    let reason = kept(&report, &doomed).unwrap_or_else(|| panic!("not kept: {report:?}"));
    assert!(reason.starts_with("could not remove"), "{reason}");
    assert!(report.removed.is_empty(), "{report:?}");
    assert!(doomed.join("held.txt").is_file() && claim_path(&doomed).is_file());
    assert!(
        entries(root).iter().all(|name| !name.starts_with('.')),
        "nothing moved: {:?}",
        entries(root)
    );

    drop(held);
    let report = run_sweep(root, later(), &mut Hooks::default());
    assert!(report.removed.contains(&doomed), "{report:?}");
    assert!(!doomed.exists() && !claim_path(&doomed).exists());
}

// ---------- 4, the caller is read per call ----------

/// Reverts the impersonation of this thread when dropped, so a failed assertion cannot leave it anonymous.
struct Anonymous;

impl Anonymous {
    fn begin() -> Anonymous {
        use windows::Win32::Security::ImpersonateAnonymousToken;
        use windows::Win32::System::Threading::GetCurrentThread;
        // SAFETY: the pseudo-handle of the current thread has every access; it needs no closing.
        unsafe { ImpersonateAnonymousToken(GetCurrentThread()) }
            .expect("a thread may impersonate the anonymous logon");
        Anonymous
    }
}

impl Drop for Anonymous {
    fn drop(&mut self) {
        // SAFETY: no arguments; it only removes this thread's impersonation token.
        let _ = unsafe { windows::Win32::Security::RevertToSelf() };
    }
}

/// Review round 1: `caller_sid` was a process-wide `OnceLock`, so the first thread to ask decided for every later one
/// and "the thread token first" was only ever true for that thread. A thread that impersonates the anonymous logon
/// (S-1-5-7) is that user for as long as it does; its neighbours, and this thread afterwards, are still the process
/// user. Two threads asking in turn must each get their own answer.
#[test]
fn the_caller_is_the_threads_token_first_on_every_call() {
    let process_user = win::process_user_sid().expect("the process token's user");
    assert_eq!(win::caller_sid(), process_user, "no impersonation yet");

    let anonymous = std::thread::spawn(|| {
        let _impersonating = Anonymous::begin();
        win::sid_text(&win::caller_sid())
    })
    .join()
    .unwrap();
    assert_eq!(
        anonymous, "S-1-5-7",
        "an impersonating thread is the anonymous logon"
    );

    // The thread that asked first before is not what a later thread inherits, whichever of them asks first.
    assert_eq!(
        win::caller_sid(),
        process_user,
        "this thread was never impersonating"
    );
    let later = std::thread::spawn(win::caller_sid).join().unwrap();
    assert_eq!(later, process_user, "a fresh thread is the process user");
}

// ---------- 7, one handle, 128-bit ids ----------

/// Review round 1: the identity of an entry is `FILE_ID_INFO`'s 128 bits (ReFS and a Dev Drive have ids the 64-bit
/// `nFileIndex` cannot tell apart), read on the handle; and the profile test of `is_mine` resolves the path ON THE SAME
/// HANDLE (`GetFinalPathNameByHandleW`), which agrees with the resolved path a second open would give.
#[test]
fn an_entry_has_a_128_bit_identity_and_one_handle_gives_its_final_path() {
    let scratch = TestScratch::new("handle-identity");
    let (a, b) = (scratch.path().join("a"), scratch.path().join("b"));
    std::fs::write(&a, b"a").unwrap();
    std::fs::write(&b, b"b").unwrap();
    let (id_a, id_b) = (file_id(&a), file_id(&b));
    assert!(
        id_a.is_some() && id_b.is_some(),
        "FileIdInfo answers on a local volume"
    );
    assert!(same_id(id_a, file_id(&a)), "stable for one entry");
    assert!(!same_id(id_a, id_b), "two entries differ");
    let opened = std::fs::File::open(&a).unwrap();
    assert!(id_a.is_some_and(|id| handle_id(&opened).is_some_and(|h| h.same_inode(&id))));
    assert!(is_plain_file(&opened));
    let directory = std::fs::File::open(scratch.path());
    assert!(
        directory.map_or(true, |d| !is_plain_file(&d)),
        "a directory is not a plain file"
    );

    let handle = std::fs::OpenOptions::new()
        .read(true)
        .share_mode(7)
        .open(&a)
        .unwrap();
    let from_handle = win::final_path_of(&handle).expect("a final path");
    let from_name = std::fs::canonicalize(&a).unwrap();
    let lower = |p: &Path| p.to_string_lossy().to_lowercase();
    assert_eq!(lower(&from_handle), lower(&from_name));
}

/// An entry reached THROUGH a junction whose target is outside the profile is not the caller's, whoever owns it; the
/// junction itself is a reparse point and never is.
#[test]
fn a_junction_is_never_mine_and_nothing_reached_through_one_outside_the_profile_is() {
    let scratch = TestScratch::new("junction");
    let link = scratch.path().join("link");
    let target = std::path::PathBuf::from(
        std::env::var_os("SystemRoot").unwrap_or_else(|| r"C:\Windows".into()),
    )
    .join("System32");
    let made = std::process::Command::new("cmd")
        .args(["/C", "mklink", "/J"])
        .arg(&link)
        .arg(&target)
        .output()
        .expect("cmd runs");
    assert!(
        made.status.success(),
        "{}",
        String::from_utf8_lossy(&made.stdout)
    );
    assert!(!is_mine(&link), "a reparse point is never mine");
    assert!(
        !is_mine(&link.join("drivers")),
        "its target is outside the profile: not mine, whoever owns it"
    );
    // `rmdir` removes the junction itself and never what it points at.
    std::fs::remove_dir(&link).unwrap();
}

// ---------- 0 ----------

/// Step 0 of the E4 plan (`research/PLAN_test_leftovers_E4.md`, S2): what does Windows say owns a file this
/// process just created, who is the caller, and is the token elevated? The rule's second key (the kernel owner is the
/// caller) is only worth anything if a normal run's new files are the caller's; the expectation that a hosted
/// (elevated) runner's files belong to `BUILTIN\Administrators` (`S-1-5-32-544`) was UNVERIFIED when the plan was
/// written, so this prints the facts and never fails. Run it once with `--nocapture` on the dml leg and once on a
/// developer's shell; scenario 20 asserts the helper against the same measurement (`measure_owner`). The file is made
/// INSIDE a `TestScratch` (review round 1: the unprefixed probe file in the raw temp root was a leak with a test's
/// shape, and the helper's own tests are not exempt from its rules).
#[test]
fn the_windows_owner_of_a_new_temp_file_is_printed() {
    let scratch = TestScratch::new("step0");
    let path = scratch.path().join("owner-probe.tmp");
    let file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .share_mode(7)
        .open(&path);
    match file {
        Err(e) => eprintln!("step0: could not create {}: {e}", path.display()),
        Ok(file) => {
            match measure_owner(&path) {
                Err(e) => eprintln!("step0: {e}"),
                Ok(m) => {
                    eprintln!("step0: caller token      = {}", m.source);
                    eprintln!("step0: caller SID        = {}", m.caller);
                    eprintln!("step0: file owner SID    = {}", m.owner);
                    eprintln!("step0: owner == caller   = {}", m.owner == m.caller);
                    eprintln!("step0: token is elevated = {:?}", m.elevated);
                }
            }
            drop(file);
        }
    }
}
