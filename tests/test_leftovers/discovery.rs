//! Discovery: an inventory from git, never a hand-listed folder.
//!
//! `git ls-files -z --cached --others --exclude-standard` in the crate root lists every file git knows, tracked or
//! untracked and not ignored. Four path exclusions come first (every Cargo `target*` directory, `node_modules/` at
//! any depth, `vendor-fastembed/` — upstream's bytes — and `.claude/`, another repository). Of what is left the
//! guard reads Rust, workflows and the JavaScript family wherever they sit, and a language it does NOT read is red:
//! no tracer is written for a language the repository has no file in, and the first such file makes the guard say
//! so until it is extended. No file is excluded by name — the list is empty and a test pins it.

use std::path::Path;
use std::process::Command;

use crate::classes::{finding, is_workflow, language_of, Finding, DISCOVERY};

/// Files the scan never reads, by exact path. EMPTY: the guard scans its own files, whose detectors and fixtures are
/// built from pieces (`words.rs`), and a test pins the list at zero entries.
pub const EXCLUDED_FILES: [&str; 0] = [];

/// Extensions of languages the guard does not read: a file of one of them (not excluded by path) is red.
const UNREAD: [&str; 6] = ["sh", "ps1", "py", "bash", "cmd", "bat"];

/// `git ls-files` for the crate root, as sorted unique relative paths. `Err` names the command: git missing or a
/// non-zero exit is red, never an empty scan.
pub fn git_listing(root: &Path) -> Result<Vec<String>, String> {
    let command = "git ls-files -z --cached --others --exclude-standard";
    let output = Command::new("git")
        .args([
            "ls-files",
            "-z",
            "--cached",
            "--others",
            "--exclude-standard",
        ])
        .current_dir(root)
        .output()
        .map_err(|error| {
            format!(
                "`{command}` could not be run in {}: {error}",
                root.display()
            )
        })?;
    if !output.status.success() {
        return Err(format!(
            "`{command}` exited with {} in {}: {}",
            output.status,
            root.display(),
            String::from_utf8_lossy(&output.stderr).trim()
        ));
    }
    Ok(parse_listing(&output.stdout))
}

pub fn parse_listing(bytes: &[u8]) -> Vec<String> {
    let mut paths: Vec<String> = bytes
        .split(|b| *b == 0)
        .filter(|piece| !piece.is_empty())
        .map(|piece| String::from_utf8_lossy(piece).to_string())
        .collect();
    paths.sort();
    paths.dedup();
    paths
}

/// Whether a path is outside the guard's reach: a `target*` directory on any segment, `node_modules/` at any depth,
/// `vendor-fastembed/` and `.claude/` as prefixes. A FILE whose name starts with `target` is not a directory.
pub fn is_excluded(path: &str) -> bool {
    if path.starts_with("vendor-fastembed/") || path.starts_with(".claude/") {
        return true;
    }
    let mut segments: Vec<&str> = path.split('/').collect();
    segments.pop(); // the file name
    segments
        .iter()
        .any(|segment| segment.starts_with("target") || *segment == "node_modules")
}

#[derive(Debug, Default, PartialEq, Eq)]
pub struct Discovery {
    /// The files the guard reads.
    pub scanned: Vec<String>,
    /// The files of a language it does not read.
    pub unread: Vec<String>,
}

pub fn discover(listing: &[String]) -> Discovery {
    let mut found = Discovery::default();
    for path in listing {
        if is_excluded(path) || EXCLUDED_FILES.contains(&path.as_str()) {
            continue;
        }
        if language_of(path).is_some() {
            found.scanned.push(path.clone());
        } else if let Some((_, extension)) = path.rsplit_once('.') {
            if UNREAD.contains(&extension.to_ascii_lowercase().as_str()) {
                found.unread.push(path.clone());
            }
        }
    }
    found
}

/// The red lines for the languages the guard does not read.
pub fn unread_findings(discovered: &Discovery) -> Vec<Finding> {
    discovered
        .unread
        .iter()
        .map(|path| {
            finding(
                path,
                1,
                DISCOVERY,
                format!(
                    "test_leftovers: {path} is a language this guard does not read — extend the guard before adding it"
                ),
            )
        })
        .collect()
}

/// A scan of nothing is red: at least one `.rs` under `src/`, one under `tests/` and one workflow must be scanned.
type Root<'a> = (&'a str, &'a dyn Fn(&str) -> bool);

pub fn scanned_nothing(scanned: &[String]) -> Vec<Finding> {
    let roots: [Root<'_>; 3] = [
        ("src/", &|p: &str| {
            p.starts_with("src/") && p.ends_with(".rs")
        }),
        ("tests/", &|p: &str| {
            p.starts_with("tests/") && p.ends_with(".rs")
        }),
        (".github/workflows/", &|p: &str| is_workflow(p)),
    ];
    roots
        .iter()
        .filter(|(_, matches)| !scanned.iter().any(|path| matches(path)))
        .map(|(root, _)| {
            finding(
                root,
                1,
                DISCOVERY,
                format!(
                    "test_leftovers: scanned nothing under {root} — discovery broke; the repository is not clean"
                ),
            )
        })
        .collect()
}
