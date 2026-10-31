//! `TestScratch`: claim → register → create, and a `Drop` that never panics.
//!
//! The order is the whole safety (rule §2). The claim is written FIRST under the generated name with an exclusive
//! create, the cleanup is registered as the very next statement (the value exists in state `Claimed` before the folder
//! does), and only then is the folder made, exclusively too. So nothing this helper made ever exists unclaimed.

use super::claim::{claim_path, remove_claim_if_still, write_claim_exclusive};
use super::identity::{file_id, FileId};
use super::{
    check_what, format_owner, name_for, random_id, run_record, sweep_at_start, temp_root,
    write_line, StderrOut,
};
use std::io::{self, Write};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::path::{Path, PathBuf};
use std::time::Duration;

/// A taken claim or a squatted folder draws a new id, a few times (rule §2).
pub(crate) const MAX_DRAWS: u32 = 3;
/// Five tries 100 ms apart: the family's figure (rule §2).
pub(crate) const REMOVE_TRIES: u32 = 5;
pub(crate) const REMOVE_BACKOFF: Duration = Duration::from_millis(100);

/// Where a value is on its way: `Claimed` (the claim is ours, the folder is not made), `Made` (both are ours),
/// `Released` (the claim was taken back after a squatter — `Drop` touches nothing).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ScratchState {
    Claimed,
    Made,
    Released,
}

/// A folder named with the prefix, the caller's owner token, this repository's token, a `<what>` and a random id, and
/// its sibling claim, removed when the value drops — on a return and on the unwind of a failed assertion. It is the
/// FIRST binding of every test that uses one, so it drops LAST, after every handle into it (locals drop in reverse
/// order of declaration).
pub(crate) struct TestScratch {
    path: PathBuf,
    written: Vec<u8>,
    /// The identity of the claim file right after this run wrote it: the claim is removed only while it is STILL
    /// that file — siblings under one `TESTSTODELETE_RUN` write identical records, so the text alone proves nothing.
    /// None when it could not be read: then the text and the inode between the read and the unlink are all that is
    /// checked.
    claim_id: Option<FileId>,
    state: ScratchState,
}

/// The constructor's seams. `Seams::real()` is what `new` and `new_in` use; the scenarios swap one or two.
pub(crate) struct Seams<'a> {
    /// The id draw.
    pub ids: &'a mut dyn FnMut() -> String,
    /// Writes the claim; `create_new` by default.
    pub write_claim: &'a mut dyn FnMut(&Path, &[u8]) -> io::Result<()>,
    /// Runs between the claim and the create, with the folder's path.
    pub after_claim: &'a mut dyn FnMut(&Path),
    /// `fs::create_dir` by default — never `create_dir_all`, a create that succeeds on an existing folder.
    pub create_dir: &'a mut dyn FnMut(&Path) -> io::Result<()>,
    /// The squatter warning; stderr by default.
    pub out: &'a mut dyn Write,
}

fn real_ids() -> String {
    random_id()
}

/// Exclusive: `create_dir`, never `create_dir_all` (a create that succeeds on an existing folder). On POSIX the folder
/// is private to its owner.
#[cfg(unix)]
fn real_create_dir(path: &Path) -> io::Result<()> {
    use std::os::unix::fs::DirBuilderExt;
    std::fs::DirBuilder::new().mode(0o700).create(path)
}

#[cfg(not(unix))]
fn real_create_dir(path: &Path) -> io::Result<()> {
    std::fs::create_dir(path)
}

fn no_hook(_: &Path) {}

impl<'a> Seams<'a> {
    /// The real seams. Each closure here is a zero-sized function item, so leaking its box allocates nothing.
    pub(crate) fn real() -> Seams<'a> {
        Seams {
            ids: Box::leak(Box::new(real_ids)),
            write_claim: Box::leak(Box::new(write_claim_exclusive)),
            after_claim: Box::leak(Box::new(no_hook)),
            create_dir: Box::leak(Box::new(real_create_dir)),
            out: Box::leak(Box::new(StderrOut)),
        }
    }
}

impl TestScratch {
    /// A scratch in the temp root. The first one in a test process sweeps the previous runs' leftovers first.
    pub(crate) fn new(what: &str) -> Self {
        Self::new_in(&temp_root(), what)
    }

    /// A scratch in `parent`: a private parent for the helper's own scenarios. The sweep still covers the temp root,
    /// never `parent`.
    pub(crate) fn new_in(parent: &Path, what: &str) -> Self {
        if let Err(problem) = check_what(what) {
            panic!("test_scratch: {problem}");
        }
        sweep_at_start();
        Self::new_with(parent, what, &mut Seams::real())
    }

    /// The scenarios' constructor: claim → register → create, with every seam injectable.
    pub(crate) fn new_with(parent: &Path, what: &str, seams: &mut Seams<'_>) -> Self {
        if let Err(problem) = check_what(what) {
            panic!("test_scratch: {problem}");
        }
        let text = format!("{}\n", format_owner(run_record())).into_bytes();
        for _ in 0..MAX_DRAWS {
            let path = parent.join(name_for(what, &(seams.ids)()));
            match (seams.write_claim)(&claim_path(&path), &text) {
                Ok(()) => {}
                // Somebody else's claim on this name: draw again.
                Err(e) if e.kind() == io::ErrorKind::AlreadyExists => continue,
                Err(e) => panic!(
                    "test_scratch: could not claim {}: {e}",
                    claim_path(&path).display()
                ),
            }
            // The cleanup is registered by the value existing: its Drop removes the claim if the create never happens.
            let mut scratch = TestScratch {
                claim_id: file_id(&claim_path(&path)),
                path,
                written: text.clone(),
                state: ScratchState::Claimed,
            };
            (seams.after_claim)(&scratch.path);
            match (seams.create_dir)(&scratch.path) {
                Ok(()) => {
                    scratch.state = ScratchState::Made;
                    return scratch;
                }
                // A SQUATTER: never adopted, never removed. Our claim goes first, only while it still holds our
                // bytes, so a claim of a run that dies later can never govern a folder this run did not create.
                Err(e) if e.kind() == io::ErrorKind::AlreadyExists => {
                    write_line(
                        seams.out,
                        &format!(
                            "test_scratch: {} already exists under the name this run claimed — it is not this run's and is left alone; drawing another name",
                            scratch.path.display()
                        ),
                    );
                    remove_claim_if_still(&scratch.path, &scratch.written, scratch.claim_id);
                    scratch.state = ScratchState::Released;
                }
                Err(e) => panic!(
                    "test_scratch: could not create {}: {e}",
                    scratch.path.display()
                ),
            }
        }
        panic!(
            "test_scratch: no free name for {what:?} in {} after {MAX_DRAWS} draws",
            parent.display()
        );
    }

    pub(crate) fn path(&self) -> &Path {
        &self.path
    }

    /// The path with `\` written as `/`: the form `log_segments`' tests compose their own paths with.
    pub(crate) fn slash_path(&self) -> String {
        self.path.to_string_lossy().replace('\\', "/")
    }
}

impl Drop for TestScratch {
    fn drop(&mut self) {
        // A panic in Drop during unwinding aborts the process (rule §2): nothing below can panic, and this is the
        // belt over those braces.
        let _ = catch_unwind(AssertUnwindSafe(|| match self.state {
            ScratchState::Made => {
                teardown(&self.path, &mut StderrOut);
            }
            ScratchState::Claimed => {
                remove_claim_if_still(&self.path, &self.written, self.claim_id)
            }
            ScratchState::Released => {}
        }));
    }
}

/// Call `remover` at most `REMOVE_TRIES` times, `REMOVE_BACKOFF` apart (never after the last). `NotFound` is success
/// (a test may remove its own folder on purpose). `on_failed_attempt(n)` runs after each failed attempt — the seam
/// that makes the open-handle scenarios deterministic. The last error is returned.
pub(crate) fn remove_with_retries(
    path: &Path,
    mut remover: impl FnMut(&Path) -> io::Result<()>,
    mut on_failed_attempt: impl FnMut(u32),
) -> io::Result<()> {
    let mut last = io::Error::other("no attempt was made");
    for attempt in 1..=REMOVE_TRIES {
        match remover(path) {
            Ok(()) => return Ok(()),
            Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok(()),
            Err(e) => {
                on_failed_attempt(attempt);
                last = e;
            }
        }
        if attempt < REMOVE_TRIES {
            std::thread::sleep(REMOVE_BACKOFF);
        }
    }
    Err(last)
}

/// Remove the folder `dir` made by this helper, then its claim — in that order, so a leftover is always prefixed AND
/// claimed. By the exact path, never by the prefix, and without asking `is_mine`: the kernel-owner key is a sweep's
/// and a claim reader's question, never a remover's of its own folder. On giving up it writes ONE line, directly to
/// `out` (libtest shows captured output only for a FAILED test), and returns `false`. Never panics.
pub(crate) fn teardown(dir: &Path, out: &mut dyn Write) -> bool {
    let gave_up = |what: &Path, error: &io::Error, out: &mut dyn Write| {
        write_line(
            out,
            &format!(
                "test_scratch: could not remove {} ({error}) — left for the start-of-run sweep",
                what.display()
            ),
        );
        false
    };
    if let Err(e) = remove_with_retries(dir, |p| std::fs::remove_dir_all(p), |_| {}) {
        return gave_up(dir, &e, out);
    }
    let claim = claim_path(dir);
    if let Err(e) = remove_with_retries(&claim, |p| std::fs::remove_file(p), |_| {}) {
        return gave_up(&claim, &e, out);
    }
    true
}
