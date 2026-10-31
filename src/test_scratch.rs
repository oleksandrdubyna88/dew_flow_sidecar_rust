//! Every folder a test of this crate creates comes from here, and every one of them is owned.
//!
//! What it owns. A `TestScratch` is a folder named with the shared prefix, the caller's owner token, this
//! repository's token, a `<what>` the test chooses and a random id (`common/test-leftovers.md` §1), claimed by a
//! sibling `<folder>.owner` that carries the run's record (§3), removed by its `Drop`, and — when the run was killed
//! and `Drop` never ran — collected by the next run of the same OS user (§4). The shared rule is the contract; this
//! file is its Rust encoding, and `tools/lib/owned.mjs` and `leftovers-sweep.mjs` in the mounted rules are its
//! reference.
//!
//! Why the claim goes FIRST. Between "the folder exists" and "the folder says whose it is" a kill leaves a folder
//! nobody can judge. So the claim is written before the folder, exclusively; the value that will clean up exists in
//! state `Claimed` before the folder does (the cleanup is registered as the very next statement); and only then is the
//! folder made, with `create_dir`, never a create that succeeds on an existing one. A folder already there under the
//! claimed name is somebody else's: never adopted, never removed, our claim taken back only while it still holds our
//! bytes. Nothing this helper made ever exists unclaimed.
//!
//! Why the claim sits BESIDE the folder. `log_segments.rs` asserts the exact listing of the folders under its root,
//! and a test that makes a git repository at its folder must see exactly what it made: an owner file inside would show.
//!
//! Why the kernel owner and not the name authorises. The owner segment of a name and the owner in a record are text
//! anyone can type into a shared temp root; a sweep that believed either would delete another user's folders. So the
//! name SELECTS, and the entry must be the caller's by the file system's own account of it (the real uid; on Windows
//! the profile AND the owner SID) before a record is even read. An elevated Windows run's files belong to
//! `BUILTIN\Administrators`, so they fail closed: listed, never touched.
//!
//! Why `Drop` never panics. A panic in `Drop` during the unwind of a failed assertion aborts the process and hides the
//! assertion. Every step has its own failure path: five tries 100 ms apart, then ONE line and a folder that stays
//! prefixed AND claimed, so the next run's sweep can judge it. The folder goes first and its claim last, so a leftover
//! is never an unclaimed folder.
//!
//! Why its lines bypass libtest's capture. libtest captures `eprintln!` and shows it only for a FAILED test, while a
//! teardown that gave up usually did so in a passing one. Lines go to `std::io::stderr()` directly, each in one
//! `write_all` under the lock, so eight test threads never interleave half-lines.
//!
//! Why the sweep runs from the first constructor. libtest has no global setup; the first `TestScratch` of a test
//! process is the earliest code that runs once. The sweep deletes on two keys (scope: name, record and caller agree and
//! the kernel owner is the caller; abandonment: the record's process is `gone`), lets age only select, removes by the
//! identity captured before the verdict through a quarantine rename, and never fails the run.
//!
//! No `std::process::Command` outside the test modules: the writer and the probe never spawn (rule §3).

#![cfg(test)]

mod claim;
mod identity;
mod owner;
mod rename;
mod scratch;
mod sweep;
mod tests;
#[cfg(windows)]
mod win;

use std::io::Write;

pub(crate) use claim::*;
pub(crate) use identity::*;
pub(crate) use owner::*;
pub(crate) use rename::*;
pub(crate) use scratch::*;
pub(crate) use sweep::*;

pub(crate) const PREFIX: &str = "testsToDelete-";
pub(crate) const REPO_TOKEN: &str = "sidecar";
// Read by the guard (`tests/test_leftovers.rs`, S4) as TEXT of this file, never by code: it is the token's row in the
// mounted rule's `## Tokens` table, `repository` column, which the guard compares this constant against.
#[allow(dead_code)]
pub(crate) const REPO_ROW: &str = "the embedding sidecar";
pub(crate) const MAX_NAME: usize = 63;
pub(crate) const ID_LENGTH: usize = 6;
pub(crate) const START_TOLERANCE_S: u64 = 2;
pub(crate) const MAX_CLAIM_BYTES: usize = 512;

/// A `Write` that is `std::io::stderr()` locked for ONE `write_all` at a time: libtest captures `eprintln!` and
/// shows it only for a FAILED test, while a teardown that gives up usually does so in a passing one.
pub(crate) struct StderrOut;

impl Write for StderrOut {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        std::io::stderr().lock().write(buf)
    }
    fn write_all(&mut self, buf: &[u8]) -> std::io::Result<()> {
        std::io::stderr().lock().write_all(buf)
    }
    fn flush(&mut self) -> std::io::Result<()> {
        std::io::stderr().lock().flush()
    }
}

/// One line, one `write_all`: eight test threads never interleave half-lines. A failing writer is ignored — a
/// diagnostic must never be the reason a test run fails.
pub(crate) fn write_line(out: &mut dyn Write, line: &str) {
    let mut bytes = Vec::with_capacity(line.len() + 1);
    bytes.extend_from_slice(line.as_bytes());
    bytes.push(b'\n');
    let _ = out.write_all(&bytes);
}

/// The room for `<what>`, in BYTES, under `owner`: 63 less the prefix, the owner, the token, the three separators
/// and the id — derived, never a constant, because the owner differs from `u1` to `w4294967295`.
pub(crate) fn what_room(owner: &str) -> usize {
    MAX_NAME - PREFIX.len() - owner.len() - 1 - REPO_TOKEN.len() - 1 - 1 - ID_LENGTH
}

/// What a name says about its version, read from its ONE second segment against the owner grammar.
#[derive(Debug, PartialEq, Eq)]
pub(crate) enum NameForm {
    Owner {
        owner: String,
        token: Option<String>,
    },
    Legacy {
        token: String,
    },
    None,
}

/// Read `name` against the prefix in every case and either separator: the segment after the prefix is the OWNER when
/// it is of the owner grammar (the segment after that is then the token), anything else is a LEGACY name's token —
/// never by counting separators. Anything not prefixed, or with nothing after the prefix, is `None`.
pub(crate) fn classify_name(name: &str) -> NameForm {
    let word = &PREFIX[..PREFIX.len() - 1];
    let bytes = name.as_bytes();
    let separator = |b: &u8| *b == b'-' || *b == b'_';
    let Some(head) = bytes.get(..word.len()) else {
        return NameForm::None;
    };
    if !head.eq_ignore_ascii_case(word.as_bytes()) || !bytes.get(word.len()).is_some_and(separator)
    {
        return NameForm::None;
    }
    let mut segments = bytes[word.len() + 1..]
        .split(separator)
        .map(|segment| String::from_utf8_lossy(segment).to_ascii_lowercase());
    let second = segments.next().filter(|segment| !segment.is_empty());
    let Some(second) = second else {
        return NameForm::None;
    };
    if owner::is_owner_token(&second) {
        let token = segments.next().filter(|segment| !segment.is_empty());
        NameForm::Owner {
            owner: second,
            token,
        }
    } else {
        NameForm::Legacy { token: second }
    }
}

/// The one `temp_dir()` call of the crate's tests: every scratch is made here.
pub(crate) fn temp_root() -> std::path::PathBuf {
    // testsToDelete-exempt: raw-temp — the helper's own temp root; every scratch is made here
    std::env::temp_dir()
}

/// `<what>` obeys the file system and the budget: non-empty, no path separator, NUL or control character, no leading
/// `-`, and no more BYTES than `what_room` leaves (stricter than the rule's characters for non-ASCII, never laxer than
/// any reader). The owner is budgeted BEFORE `what`; a finished name is never truncated.
pub(crate) fn check_what(what: &str) -> Result<(), String> {
    if what.is_empty() {
        return Err("`what` is empty".to_string());
    }
    if what
        .chars()
        .any(|c| c == '/' || c == '\\' || c.is_control())
    {
        return Err(format!(
            "`what` may not hold a path separator or a control character: {what:?}"
        ));
    }
    if what.starts_with('-') {
        return Err(format!("`what` may not start with `-`: {what:?}"));
    }
    let owner = owner_token();
    let room = what_room(owner);
    if what.len() > room {
        return Err(format!(
            "the whole name would pass {MAX_NAME} bytes — `what` is {} bytes, the most is {room} under the owner `{owner}`: {what:?}",
            what.len()
        ));
    }
    Ok(())
}

/// `<prefix><owner>-<token>-<what>-<id>`: the name of a scratch, from a validated `what` and a drawn id.
pub(crate) fn name_for(what: &str, id: &str) -> String {
    format!("{PREFIX}{}-{REPO_TOKEN}-{what}-{id}", owner_token())
}

/// A name with a fresh random id. Panics on an invalid `what`.
pub(crate) fn owned_name(what: &str) -> String {
    if let Err(problem) = check_what(what) {
        panic!("test_scratch: {problem}");
    }
    name_for(what, &random_id())
}

/// `bytes` random bytes as `2 x bytes` lower-case hex characters (`[0-9a-f]`): the suffix of a quarantine name, which
/// every sweeper of the family reads with `[0-9a-f]+` (rule section 4). Same entropy source as `random_id`.
pub(crate) fn random_hex(bytes: usize) -> String {
    use std::collections::hash_map::RandomState;
    use std::fmt::Write as _;
    use std::hash::{BuildHasher, Hasher};
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let mut hex = String::with_capacity(bytes * 2);
    let mut drawn = 0;
    while drawn < bytes {
        let mut hasher = RandomState::new().build_hasher();
        hasher.write_u64(COUNTER.fetch_add(1, Ordering::Relaxed));
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |elapsed| elapsed.as_nanos());
        hasher.write_u128(now);
        for byte in hasher.finish().to_le_bytes() {
            if drawn < bytes {
                let _ = write!(hex, "{byte:02x}");
                drawn += 1;
            }
        }
    }
    hex
}

/// Six characters of `[A-Za-z0-9]`. Std seeds `RandomState` from the OS; each id is a hasher fed a per-process
/// counter and the clock, so two ids never come from the same input, and a byte is taken only below 248 (4 x 62) so
/// the alphabet is drawn without modulo bias.
pub(crate) fn random_id() -> String {
    use std::collections::hash_map::RandomState;
    use std::hash::{BuildHasher, Hasher};
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    const ALPHABET: &[u8; 62] = b"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
    let mut id = String::with_capacity(ID_LENGTH);
    while id.len() < ID_LENGTH {
        let mut hasher = RandomState::new().build_hasher();
        hasher.write_u64(COUNTER.fetch_add(1, Ordering::Relaxed));
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |elapsed| elapsed.as_nanos());
        hasher.write_u128(now);
        for byte in hasher.finish().to_le_bytes() {
            if byte < 248 && id.len() < ID_LENGTH {
                id.push(char::from(ALPHABET[usize::from(byte) % ALPHABET.len()]));
            }
        }
    }
    id
}
