//! The start-of-run sweep (rule §4): scope selects, liveness authorises, uncertainty preserves.
//!
//! A folder is removed only on TWO keys, both required: its name's owner, its record's owner and the caller's token
//! are ONE and the kernel owner of the entry is the caller (scope), and the record's process is `gone` (abandonment).
//! Age only SELECTS: a folder younger than ten minutes is left alone silently, and an old one is never removed for
//! being old. Everything else is kept and named — `live`, `unknown`, `unreadable`, `foreign-uid`, `unmarked` (a folder
//! of the owner form with no claim is nobody's the helper knows: listed, never collected), `other clone`, `replaced`.
//!
//! Removal is by what identifies the folder at the moment of removal, never by a name read earlier: the identity and
//! the claim are captured BEFORE the verdict, re-checked just before an atomic rename to a sibling quarantine, and
//! checked again there; what changed in between is renamed back and kept. A candidate is a real directory — a link or
//! junction under one of this owner's names is kept `not a directory`, never opened, its target never touched.
//!
//! The sweep never panics and never fails the run: every I/O error is one `errors` entry and the next pass still runs.

use super::claim::{claim_path, snapshot_claim, ClaimSnapshot, RawClaim};
use super::identity::{changed_at, file_id, same_id, FileId};
use super::owner::{owner_state, parse_owner, Claim, Facts, Owner, OwnerState, Probe};
use super::scratch::remove_with_retries;
use super::{
    classify_name, is_mine, probe, temp_root, this_owner, write_line, NameForm, StderrOut, PREFIX,
    REPO_TOKEN,
};
use regex::Regex;
use std::cell::RefCell;
use std::collections::HashMap;
use std::io::Write;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::path::{Path, PathBuf};
use std::sync::{Once, OnceLock};
use std::time::{Duration, SystemTime};

/// A folder younger than this is not considered: age SELECTS, it never authorises (rule §4).
pub(crate) const SWEEP_MIN_AGE: Duration = Duration::from_secs(600);

/// What one sweep did. `listed` holds the lines for what a sweep may not touch but a person should see: `foreign:`,
/// `legacy:`, `unmarked:`, `quarantine:` and `notice:`.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub(crate) struct SweepReport {
    pub removed: Vec<PathBuf>,
    pub kept: Vec<(PathBuf, String)>,
    pub listed: Vec<String>,
    pub errors: Vec<String>,
}

/// Seams for the changes between a capture and a removal: a scenario replaces the folder after the capture, drops a
/// file into the quarantine or rewrites the claim after the rename, or swaps the claim after the folder is gone.
#[derive(Default)]
pub(crate) struct Hooks<'a> {
    /// With the folder's path, after its identity and claim were captured and before the verdict.
    pub after_capture: Option<PathHook<'a>>,
    /// With the folder's original path and its quarantine, right after the rename.
    pub after_rename: Option<RenameHook<'a>>,
    /// With the folder's original path, once the quarantine is removed and before its claim is.
    pub after_remove: Option<PathHook<'a>>,
}

pub(crate) type PathHook<'a> = &'a mut dyn FnMut(&Path);
pub(crate) type RenameHook<'a> = &'a mut dyn FnMut(&Path, &Path);

/// What a name says about whose it is, for THIS repository's token.
enum Relation {
    Own { name_owner: String },
    Foreign { name_owner: String },
    Legacy,
    Other,
}

fn relation_of(name: &str, owner: &str) -> Relation {
    match classify_name(name) {
        NameForm::Owner {
            owner: name_owner,
            token: Some(token),
        } if token == REPO_TOKEN => {
            if name_owner == owner {
                Relation::Own { name_owner }
            } else {
                Relation::Foreign { name_owner }
            }
        }
        NameForm::Legacy { token } if token == REPO_TOKEN => Relation::Legacy,
        _ => Relation::Other,
    }
}

/// `.<name>.<prefix, lower-case>quarantine-<pid>-<hex>`: a leading `.` so no prefix match ever reaches it.
fn quarantine_pattern() -> &'static Regex {
    static PATTERN: OnceLock<Regex> = OnceLock::new();
    PATTERN.get_or_init(|| {
        let word = PREFIX.to_ascii_lowercase();
        Regex::new(&format!(
            r"(?i)\A\.(.+)\.{word}quarantine-([0-9]+)-([0-9a-f]+)\z"
        ))
        .expect("the quarantine pattern is valid")
    })
}

fn quarantine_path(dir: &Path) -> PathBuf {
    let name = dir
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_default();
    let token = super::random_id().to_ascii_lowercase();
    dir.with_file_name(format!(
        ".{name}.{}quarantine-{}-{token}",
        PREFIX.to_ascii_lowercase(),
        std::process::id()
    ))
}

fn shell_quote(path: &Path) -> String {
    let text = path.to_string_lossy();
    if cfg!(windows) {
        format!("'{}'", text.replace('\'', "''"))
    } else {
        format!("'{}'", text.replace('\'', "'\\''"))
    }
}

fn is_real_dir(meta: &std::fs::Metadata) -> bool {
    meta.is_dir() && !meta.file_type().is_symlink()
}

fn same_claim(now: &ClaimSnapshot, captured: &ClaimSnapshot) -> bool {
    now.raw == captured.raw && same_id(now.id, captured.id)
}

/// Put `quarantined` back under `original` ONLY if that name is free — never over an existing entry (rule §4). POSIX:
/// a placeholder of our own is created exclusively and the quarantine is renamed over that EMPTY directory (a plain
/// `rename(2)` of a directory onto an empty directory REPLACES it, so it is never tried against a name that may be
/// taken); when the rename fails the placeholder goes with `remove_dir`, which refuses anything non-empty. Windows: a
/// plain `rename`, which refuses an existing directory.
fn exclusive_restore(quarantined: &Path, original: &Path, mine: &dyn Fn(&Path) -> bool) -> bool {
    #[cfg(unix)]
    {
        use std::os::unix::fs::DirBuilderExt;
        if std::fs::DirBuilder::new()
            .mode(0o700)
            .create(original)
            .is_err()
        {
            return false;
        }
        if std::fs::rename(quarantined, original).is_ok() {
            return true;
        }
        let ours = std::fs::symlink_metadata(original).is_ok_and(|m| is_real_dir(&m));
        if ours && mine(original) {
            let _ = std::fs::remove_dir(original);
        }
        false
    }
    #[cfg(not(unix))]
    {
        let _ = mine;
        std::fs::rename(quarantined, original).is_ok()
    }
}

enum Verdict {
    Keep(String),
    Remove,
}

enum Outcome {
    Removed,
    Kept { path: PathBuf, reason: String },
}

struct Sweeper<'a, 'h> {
    root: &'a Path,
    now: SystemTime,
    min_age: Duration,
    me: &'a Owner,
    facts: Facts,
    mine: &'a dyn Fn(&Path) -> bool,
    hooks: &'a mut Hooks<'h>,
    report: SweepReport,
}

impl Sweeper<'_, '_> {
    fn keep(&mut self, path: &Path, reason: impl Into<String>) {
        self.report.kept.push((path.to_path_buf(), reason.into()));
    }

    fn list_foreign(&mut self, kind: &str, name: &str, owner: &str) {
        self.report
            .listed
            .push(format!("foreign: {kind} {name} — owner {owner}"));
    }

    fn list_conflict(&mut self, quarantine: &Path, original: &Path) {
        let (from, to) = (shell_quote(quarantine), shell_quote(original));
        let command = if cfg!(windows) {
            format!("Move-Item -LiteralPath {from} -Destination {to}")
        } else {
            format!("mv -- {from} {to}")
        };
        self.report.listed.push(format!(
            "quarantine: {} — restore with: {command}",
            quarantine.display()
        ));
    }

    /// The entries of the root, once per pass; the failure to list is ONE `errors` entry naming the root.
    fn entries(&mut self, pass: &str) -> Vec<(String, PathBuf)> {
        let listing = match std::fs::read_dir(self.root) {
            Ok(listing) => listing,
            Err(e) => {
                self.report
                    .errors
                    .push(format!("{pass}: cannot list {} ({e})", self.root.display()));
                return Vec::new();
            }
        };
        let mut found = Vec::new();
        for entry in listing {
            match entry {
                Ok(entry) => {
                    if let Some(name) = entry.file_name().to_str() {
                        found.push((name.to_string(), entry.path()));
                    }
                }
                Err(e) => self.report.errors.push(format!(
                    "{pass}: cannot read an entry of {} ({e})",
                    self.root.display()
                )),
            }
        }
        found
    }

    /// Is the entry as old as the window? A time that cannot be read is not old.
    fn old_enough(&self, stamp: Option<SystemTime>) -> bool {
        stamp
            .and_then(|stamp| self.now.duration_since(stamp).ok())
            .is_some_and(|age| age >= self.min_age)
    }

    // ---------- pass 1: quarantines a killed sweep left ----------

    fn restore_quarantines(&mut self, probe: &dyn Fn(u64) -> Probe) {
        for (name, path) in self.entries("quarantines") {
            let Some(found) = quarantine_pattern().captures(&name) else {
                continue;
            };
            let original_name = found[1].to_string();
            if matches!(relation_of(&original_name, &self.me.owner), Relation::Other) {
                continue;
            }
            let Ok(meta) = std::fs::symlink_metadata(&path) else {
                continue;
            };
            let renamer_is_gone = found[2]
                .parse::<u64>()
                .is_ok_and(|pid| probe(pid) == Probe::Gone);
            // Another uid's, or its renamer may still be working, or too young by either stamp (a renamer in another
            // pid namespace has a pid this process cannot see: it is "gone" here and may be mid-delete).
            if !is_real_dir(&meta)
                || !(self.mine)(&path)
                || !renamer_is_gone
                || !self.old_enough(meta.modified().ok())
                || !self.old_enough(changed_at(&path))
            {
                continue;
            }
            let original = self.root.join(&original_name);
            if exclusive_restore(&path, &original, self.mine) {
                self.report.listed.push(format!(
                    "notice: restored {original_name} from a quarantine a killed sweep left"
                ));
            } else {
                self.keep(&path, "quarantine-conflict");
                self.list_conflict(&path, &original);
            }
        }
    }

    // ---------- pass 2: folders ----------

    fn sweep_folders(&mut self, probe: &dyn Fn(u64) -> Probe) {
        for (name, path) in self.entries("folders") {
            let relation = relation_of(&name, &self.me.owner);
            if matches!(relation, Relation::Other) {
                continue;
            }
            let meta = match std::fs::symlink_metadata(&path) {
                Ok(meta) => meta,
                Err(e) => {
                    self.report
                        .errors
                        .push(format!("folders: cannot stat {} ({e})", path.display()));
                    continue;
                }
            };
            if meta.file_type().is_symlink() {
                // A link (a junction too) under one of this owner's names: never opened, its target never touched.
                if matches!(relation, Relation::Own { .. }) {
                    self.keep(&path, "not a directory");
                }
                continue;
            }
            if !meta.is_dir() {
                continue; // a claim file, or noise
            }
            match relation {
                Relation::Foreign { name_owner } => self.list_foreign("folder", &name, &name_owner),
                Relation::Legacy => {
                    let command = if cfg!(windows) {
                        format!(
                            "Remove-Item -Recurse -Force {}, {}",
                            shell_quote(&path),
                            shell_quote(&claim_path(&path))
                        )
                    } else {
                        format!(
                            "rm -rf -- {} {}",
                            shell_quote(&path),
                            shell_quote(&claim_path(&path))
                        )
                    };
                    self.report
                        .listed
                        .push(format!("legacy: folder {name} — remove with: {command}"));
                }
                Relation::Own { name_owner } => {
                    // Younger than the window: not considered, silently — age SELECTS, never authorises.
                    if self.old_enough(meta.modified().ok()) {
                        self.judge_folder(&name, &path, &name_owner, probe);
                    }
                }
                Relation::Other => {}
            }
        }
    }

    fn judge_folder(
        &mut self,
        name: &str,
        path: &Path,
        name_owner: &str,
        probe: &dyn Fn(u64) -> Probe,
    ) {
        if std::fs::symlink_metadata(path.join(".git")).is_ok_and(|m| m.is_file()) {
            self.keep(path, "other clone"); // this crate makes no worktrees: some other clone's
            return;
        }
        // Identity and claim are captured BEFORE the verdict, so what is judged is what is removed.
        let captured_id = file_id(path);
        let claim = snapshot_claim(path, self.mine);
        if let Some(hook) = self.hooks.after_capture.as_deref_mut() {
            hook(path);
        }
        if !(self.mine)(path) {
            self.keep(path, "foreign-uid");
            self.list_foreign("folder", name, name_owner);
            return;
        }
        match self.verdict(&claim.raw, name_owner, probe) {
            Verdict::Keep(reason) => {
                match reason.as_str() {
                    "unmarked" => self.report.listed.push(format!("unmarked: folder {name}")),
                    "foreign-uid" => self.list_foreign("folder", name, name_owner),
                    _ => {}
                }
                self.keep(path, reason);
            }
            Verdict::Remove => {
                let Some(id) = captured_id else {
                    self.keep(path, "identity unreadable");
                    return;
                };
                match self.remove_by_identity(path, id, &claim) {
                    Outcome::Removed => self.report.removed.push(path.to_path_buf()),
                    Outcome::Kept { path, reason } => self.keep(&path, reason),
                }
            }
        }
    }

    /// Rule §4's table for one own-scope claim (the kernel owner of the folder is already settled).
    fn verdict(&self, raw: &RawClaim, name_owner: &str, probe: &dyn Fn(u64) -> Probe) -> Verdict {
        let text = match raw {
            RawClaim::Absent => return Verdict::Keep("unmarked".into()),
            RawClaim::Unreadable { foreign_uid: true } => {
                return Verdict::Keep("foreign-uid".into())
            }
            RawClaim::Unreadable { .. } => return Verdict::Keep("unreadable".into()),
            RawClaim::Text(text) => text,
        };
        let claim = parse_owner(text);
        let Claim::Owner(record) = &claim else {
            return Verdict::Keep("unreadable".into());
        };
        match owner_state(&claim, Some(name_owner), true, self.me, &self.facts, probe) {
            OwnerState::Gone => Verdict::Remove,
            OwnerState::Live => Verdict::Keep("live".into()),
            OwnerState::Unknown if record.host.to_lowercase() == self.me.host.to_lowercase() => {
                Verdict::Keep("unknown".into())
            }
            OwnerState::Unknown => Verdict::Keep("foreign host".into()),
        }
    }

    /// Re-check, rename atomically to the sibling quarantine, check again there, remove, then the claim. What changed
    /// in between is renamed back and kept. Nothing is ever lost: a removal that gives up in the quarantine is renamed
    /// BACK under its own name (a deviation from the reference sweep, which leaves it in quarantine with an `errors`
    /// entry) — it stays prefixed and claimed, judged by the next run like any leftover.
    fn remove_by_identity(&mut self, dir: &Path, id: FileId, claim: &ClaimSnapshot) -> Outcome {
        let kept = |reason: String| Outcome::Kept {
            path: dir.to_path_buf(),
            reason,
        };
        if !same_id(file_id(dir), Some(id)) || !same_claim(&snapshot_claim(dir, self.mine), claim) {
            return kept("replaced".into());
        }
        let quarantine = quarantine_path(dir);
        if let Err(e) = std::fs::rename(dir, &quarantine) {
            return kept(format!("could not remove: {e}")); // refused: nothing moved
        }
        if let Some(hook) = self.hooks.after_rename.as_deref_mut() {
            hook(dir, &quarantine);
        }
        // The change time is not compared here: a rename updates it (measured on tmpfs), and it exists to catch a
        // reused inode number, which the check before the rename already did.
        let still_it = file_id(&quarantine).is_some_and(|there| there.same_inode(&id))
            && same_claim(&snapshot_claim(dir, self.mine), claim);
        if !still_it {
            return self.put_back(&quarantine, dir, "replaced".into());
        }
        if let Err(e) = remove_with_retries(&quarantine, |p| std::fs::remove_dir_all(p), |_| {}) {
            return self.put_back(&quarantine, dir, format!("could not remove: {e}"));
        }
        if let Some(hook) = self.hooks.after_remove.as_deref_mut() {
            hook(dir);
        }
        // The claim goes only while its identity AND its text are the captured ones.
        if same_claim(&snapshot_claim(dir, self.mine), claim) {
            let claim_file = claim_path(dir);
            if let Err(e) = remove_with_retries(&claim_file, |p| std::fs::remove_file(p), |_| {}) {
                self.report
                    .errors
                    .push(format!("claim {}: {e}", claim_file.display()));
            }
        }
        Outcome::Removed
    }

    fn put_back(&mut self, quarantine: &Path, original: &Path, reason: String) -> Outcome {
        if exclusive_restore(quarantine, original, self.mine) {
            Outcome::Kept {
                path: original.to_path_buf(),
                reason,
            }
        } else {
            self.list_conflict(quarantine, original);
            Outcome::Kept {
                path: quarantine.to_path_buf(),
                reason: "quarantine-conflict".into(),
            }
        }
    }

    // ---------- pass 3: claims whose folder is absent ----------

    fn sweep_orphan_claims(&mut self, probe: &dyn Fn(u64) -> Probe) {
        for (name, claim_file) in self.entries("claims") {
            let Some(stem) = name.strip_suffix(".owner") else {
                continue;
            };
            let relation = relation_of(stem, &self.me.owner);
            if matches!(relation, Relation::Other) {
                continue;
            }
            let folder = self.root.join(stem);
            let Ok(meta) = std::fs::symlink_metadata(&claim_file) else {
                continue;
            };
            if !meta.is_file()
                || std::fs::symlink_metadata(&folder).is_ok()
                || !self.old_enough(meta.modified().ok())
            {
                continue;
            }
            match relation {
                Relation::Foreign { name_owner } => self.list_foreign("claim", &name, &name_owner),
                Relation::Legacy => {
                    let command = if cfg!(windows) {
                        format!("Remove-Item -Force {}", shell_quote(&claim_file))
                    } else {
                        format!("rm -f -- {}", shell_quote(&claim_file))
                    };
                    self.keep(&claim_file, "legacy");
                    self.report
                        .listed
                        .push(format!("legacy: claim {name} — remove with: {command}"));
                }
                Relation::Own { name_owner } => {
                    self.judge_orphan(&name, &claim_file, &folder, &name_owner, probe)
                }
                Relation::Other => {}
            }
        }
    }

    /// An orphan claim — left by a run that died between the claim and the create — is judged by its owner like any
    /// claim: `gone` removes it, everything else keeps it with the reason.
    fn judge_orphan(
        &mut self,
        name: &str,
        claim_file: &Path,
        folder: &Path,
        name_owner: &str,
        probe: &dyn Fn(u64) -> Probe,
    ) {
        let captured = snapshot_claim(folder, self.mine);
        match self.verdict(&captured.raw, name_owner, probe) {
            Verdict::Keep(reason) => {
                if reason == "foreign-uid" {
                    self.list_foreign("claim", name, name_owner);
                }
                self.keep(claim_file, reason);
            }
            Verdict::Remove => {
                let unchanged = std::fs::symlink_metadata(folder).is_err()
                    && same_claim(&snapshot_claim(folder, self.mine), &captured);
                if !unchanged {
                    self.keep(claim_file, "replaced");
                    return;
                }
                match remove_with_retries(claim_file, |p| std::fs::remove_file(p), |_| {}) {
                    Ok(()) => self.report.removed.push(claim_file.to_path_buf()),
                    Err(e) => self
                        .report
                        .errors
                        .push(format!("claim {}: {e}", claim_file.display())),
                }
            }
        }
    }
}

/// Collect what a killed run left in `root`: three passes (quarantine restores, folders, orphan claims), each
/// listing the root once. `probe` is the liveness query (a seam), asked at most once per pid; `mine` the kernel-owner
/// question (a seam: a second uid cannot be had in a test); `hooks` the changes between a capture and a removal.
/// Every I/O failure is one `errors` entry and the next pass still runs.
pub(crate) fn sweep(
    root: &Path,
    now: SystemTime,
    min_age: Duration,
    probe: impl Fn(u64) -> Probe,
    mine: impl Fn(&Path) -> bool,
    hooks: &mut Hooks<'_>,
) -> SweepReport {
    let asked: RefCell<HashMap<u64, Probe>> = RefCell::new(HashMap::new());
    let cached = |pid: u64| *asked.borrow_mut().entry(pid).or_insert_with(|| probe(pid));
    let mut sweeper = Sweeper {
        root,
        now,
        min_age,
        me: this_owner(),
        facts: Facts::real(),
        mine: &mine,
        hooks,
        report: SweepReport::default(),
    };
    sweeper.restore_quarantines(&cached);
    sweeper.sweep_folders(&cached);
    sweeper.sweep_orphan_claims(&cached);
    sweeper.report
}

/// Run `run` and print its report; a panic inside it is ONE line, never a failed run. `guarded` never panics, which
/// is what keeps the `Once` that calls it from ever being poisoned (a poisoned `Once` would fail every later test).
pub(crate) fn guarded(run: impl FnOnce() -> SweepReport, out: &mut dyn Write) {
    match catch_unwind(AssertUnwindSafe(run)) {
        Ok(report) => print_report(&report, out),
        Err(payload) => {
            let text = payload
                .downcast_ref::<String>()
                .map(String::as_str)
                .or_else(|| payload.downcast_ref::<&str>().copied())
                .unwrap_or("a panic with no text");
            write_line(
                out,
                &format!("test_scratch: the start-of-run sweep panicked ({text}) — skipped"),
            );
        }
    }
}

/// One line per `kept`, `listed` and `errors` entry — each in ONE `write_all`, so eight test threads never
/// interleave half-lines — and nothing for an empty report.
pub(crate) fn print_report(report: &SweepReport, out: &mut dyn Write) {
    for (path, reason) in &report.kept {
        write_line(
            out,
            &format!("test_scratch: kept {} — {reason}", path.display()),
        );
    }
    for line in &report.listed {
        write_line(out, &format!("test_scratch: {line}"));
    }
    for error in &report.errors {
        write_line(out, &format!("test_scratch: sweep error — {error}"));
    }
}

/// The sweep of the previous runs' leftovers in the temp root, once per test process, from the first constructor
/// (libtest has no global setup). The temp root is listed, never a private parent. A sweep never fails the run.
pub(crate) fn sweep_at_start() {
    static ONCE: Once = Once::new();
    ONCE.call_once(|| {
        guarded(
            || {
                sweep(
                    &temp_root(),
                    SystemTime::now(),
                    SWEEP_MIN_AGE,
                    probe,
                    is_mine,
                    &mut Hooks::default(),
                )
            },
            &mut StderrOut,
        );
    });
}
