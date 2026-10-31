//! The words the guard looks for, and the words its fixtures are made of — every one of them in pieces.
//!
//! The guard scans its own files like any other (`discovery::EXCLUDED_FILES` is empty and pinned), and the
//! machine-wide barrier scan of the shared rules reads them too: a file that spelled the prefix, a temp API, a
//! container command or an exemption marker whole would be a finding of the very class it tests. So no line of
//! `tests/test_leftovers.rs` or of this folder holds one whole; they are joined at compile time here, and a
//! fixture that needs a whole spelling gets it at run time from these constants.

/// The prefix word in lower case — what every case-insensitive detector is built from.
pub const WORD: &str = concat!("teststo", "delete");
/// The prefix word as the helper writes it, without its separator.
pub const WORD_CASED: &str = concat!("testsTo", "Delete");
/// The prefix as the helper writes it (`PREFIX` of `src/test_scratch.rs`).
pub const PREFIX_TEXT: &str = concat!("testsTo", "Delete-");
/// The exemption marker.
pub const MARK: &str = concat!("testsTo", "Delete-exempt:");

/// A raw temp call, as a fixture line.
pub const TEMP_CALL: &str = concat!("std::env::temp_", "dir()");
/// The helper's own temp-root accessor.
pub const TEMP_ROOT: &str = concat!("temp", "_root");
/// A worktree creation.
pub const WORKTREE_ADD: &str = concat!("git work", "tree add ../w");
/// A container start without any carve-out.
pub const DOCKER_RUN_CMD: &str = concat!("dock", "er run --rm img");
pub const DOCKER_VOLUME: &str = concat!("dock", "er volume cre", "ate ");
pub const DOCKER_COMPOSE: &str = concat!("dock", "er compose -p ");
/// A delete-all.
pub const STORE_WIPE: &str = concat!("TRUN", "CATE TABLE t");
/// The reaper switched off.
pub const RYUK_OFF_TEXT: &str = concat!("TESTCONTAINERS_RYUK_", "DISABLED=true");
/// The helper's prefix constant.
pub const PREFIX_CONST: &str = concat!("PRE", "FIX");

/// A whole name of the owner form, for a fixture: `<prefix>u1000-sidecar-x-abc123`.
pub fn owner_name() -> String {
    format!("{PREFIX_TEXT}u1000-sidecar-x-abc123")
}

/// A legacy name (no owner segment), for a fixture: `<prefix>sidecar-x-abc123`.
pub fn legacy_name() -> String {
    format!("{PREFIX_TEXT}sidecar-x-abc123")
}
