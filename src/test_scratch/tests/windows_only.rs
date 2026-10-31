//! Scenarios that only Windows can run (0, 10, 14 and 20 of the E4 plan, S2).

#![cfg(windows)]

use super::super::*;

/// Step 0 of the E4 plan (`todo/PLAN_test_leftovers_E4.md`, S2): what does Windows say owns a file this
/// process just created in the temp root, who is the caller, and is the token elevated? The rule's second key
/// (the kernel owner is the caller) is only worth anything if a normal run's new files are the caller's; the
/// expectation that a hosted (elevated) runner's files belong to `BUILTIN\Administrators` (`S-1-5-32-544`) was
/// UNVERIFIED when the plan was written, so this prints the facts and never fails. Run it once with
/// `--nocapture` on the dml leg and once on a developer's shell; scenario 20 asserts whatever was measured.
#[cfg(windows)]
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
        // testsToDelete-exempt: raw-temp — step 0 measures the owner of a file in the raw temp root before any helper exists
        let path =
            std::env::temp_dir().join(format!("bge-step0-owner-probe-{}.tmp", std::process::id()));
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
