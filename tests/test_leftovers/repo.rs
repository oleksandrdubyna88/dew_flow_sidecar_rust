//! The guard over the real tree: discovery, every class, the shipped classifier for scripts, the token — and the
//! report. The unit scenarios in the sibling modules prove each part on fixtures; this is the part that reads files.

use std::path::{Path, PathBuf};

use crate::classes::{
    finding, language_of, scan_text, split_lines, Finding, Lang, DISCOVERY, DOCKER_NAME,
    DOCKER_RUN, DROPPED_SCRATCH, EXEMPTION, LEGACY_NAME, OWNER_BLIND, RAW_TEMP, RAW_WORKTREE,
    RYUK_OFF, SHARED_STORE, TOKEN,
};
use crate::discovery::{discover, git_listing, scanned_nothing, unread_findings};
use crate::scripts::{script_owner_blind, ScriptSource, CLASSIFIER};
use crate::token::{constants_in, package_name, token_findings, RULE};

/// What a scan of the real tree found, and what it read.
pub struct Scan {
    pub scanned: Vec<String>,
    pub findings: Vec<Finding>,
}

/// A file with a NUL byte in its first 8 KiB is binary (git's own test): not text to be read for names.
fn is_binary(bytes: &[u8]) -> bool {
    bytes.iter().take(8192).any(|b| *b == 0)
}

/// Everything a scan reads, handed in: the real tree gives it from disk and git, the wiring scenarios from memory
/// (no folder is made here, so a synthetic tree cannot be a folder).
pub struct Inputs<'a> {
    pub listing: Vec<String>,
    pub read: &'a dyn Fn(&str) -> Option<Vec<u8>>,
    pub classifier: PathBuf,
    pub rule: Option<String>,
    pub cargo_toml: Option<String>,
}

/// Every part of the guard over `inputs`: discovery, the unread languages, the scan of nothing, every class, the
/// shipped classifier over scripts and workflows, and the token.
pub fn scan_inputs(inputs: &Inputs<'_>) -> Scan {
    let discovered = discover(&inputs.listing);
    let mut findings = unread_findings(&discovered);
    findings.extend(scanned_nothing(&discovered.scanned));

    // Each file is read once; its text is shared by every class.
    let mut texts: Vec<(String, String)> = Vec::new();
    for path in &discovered.scanned {
        let Some(bytes) = (inputs.read)(path) else {
            continue; // listed by git, gone from disk (deleted, not yet staged)
        };
        if is_binary(&bytes) {
            continue;
        }
        texts.push((path.clone(), String::from_utf8_lossy(&bytes).to_string()));
    }
    for (path, text) in &texts {
        findings.extend(scan_text(path, text));
    }

    let scripts: Vec<ScriptSource<'_>> = texts
        .iter()
        .filter_map(|(path, text)| match language_of(path) {
            Some(lang @ (Lang::Js | Lang::Workflow)) => Some(ScriptSource::new(path, lang, text)),
            _ => None,
        })
        .collect();
    findings.extend(script_owner_blind(&scripts, &inputs.classifier));

    let constants: Vec<_> = texts
        .iter()
        .filter(|(path, _)| path.starts_with("src/") && path.ends_with(".rs"))
        .flat_map(|(path, text)| constants_in(path, &split_lines(text)))
        .collect();
    let package = inputs.cargo_toml.as_deref().and_then(package_name);
    findings.extend(token_findings(
        inputs.rule.as_deref(),
        &constants,
        package.as_deref(),
    ));

    findings.sort_by(|a, b| {
        (a.file.as_str(), a.line, a.class).cmp(&(b.file.as_str(), b.line, b.class))
    });
    Scan {
        scanned: discovered.scanned,
        findings,
    }
}

/// The real tree: the files git lists, read from disk, the mounted rule and the crate's manifest.
pub fn scan_repository(root: &Path) -> Scan {
    let listing = match git_listing(root) {
        Ok(listing) => listing,
        Err(why) => {
            return Scan {
                scanned: Vec::new(),
                findings: vec![finding("git", 1, DISCOVERY, why)],
            }
        }
    };
    let read = |path: &str| std::fs::read(root.join(path)).ok();
    scan_inputs(&Inputs {
        listing,
        read: &read,
        classifier: root.join(CLASSIFIER),
        rule: std::fs::read_to_string(root.join(RULE)).ok(),
        cargo_toml: std::fs::read_to_string(root.join("Cargo.toml")).ok(),
    })
}

/// The sentence that tells what to do about a class (one per class, naming the rule's section).
fn cure(class: &str) -> &'static str {
    match class {
        RAW_TEMP => "Make test folders with TestScratch (src/test_scratch.rs) — common/test-leftovers.md §7 (raw-temp), §2.",
        RAW_WORKTREE => "Make worktrees through the helper that claims and removes them — common/test-leftovers.md §7 (raw-worktree), §3.",
        DOCKER_RUN => "No container start exists in this repository's tests: the first one extends this guard to the complete carve-out of §5 in the same change — common/test-leftovers.md §5, §7 (docker-run).",
        DOCKER_NAME => "Give the container, volume, network or compose project the owner form <prefix><owner>-<repo>-<what>-<id> — common/test-leftovers.md §1, §7 (docker-name).",
        LEGACY_NAME => "Build every name with the owner segment through the helper; an exemption may clear this class, never owner-blind — common/test-leftovers.md §1, §7 (legacy-name).",
        OWNER_BLIND => "Match the caller's owner after the prefix or delete the sweep and use the shipped janitor; this class takes no exemption — common/test-leftovers.md §7 (owner-blind), the barrier.",
        SHARED_STORE => "Isolate the store per run, or add an exemption of this class with a reason that carries `lock: …` — common/test-leftovers.md §6, §7 (shared-store).",
        RYUK_OFF => "Leave Testcontainers' reaper on; this class takes no exemption — common/test-leftovers.md §5, §7 (ryuk-off).",
        DROPPED_SCRATCH => "Hold a TestScratch in a NAMED binding (`let scratch = …`, `let seed = …`): `let _ =` and a destructuring that leaves it unbound drop it at the end of the statement, and the folder is deleted before the test uses it — this repository's S3 trap, common/test-leftovers.md §2.",
        EXEMPTION => "Write an exemption as the rule's comment form with a class of §7 (not ryuk-off, not owner-blind) and a real reason — common/test-leftovers.md §7 (the exemption grammar).",
        TOKEN => "Register this repository's token and row in the rule's `## Tokens` table and write them once in the helper — common/test-leftovers.md §1, §7.",
        _ => "The discovery of the files to read broke or met a language it does not read — common/test-leftovers.md §7 (a scan of nothing is red).",
    }
}

/// One line per finding, `<file>:<line> — <class> — <text>`, then one cure sentence per class that appeared.
pub fn render(findings: &[Finding]) -> String {
    let mut out = format!(
        "test_leftovers: {} finding(s) — see common/test-leftovers.md §7\n",
        findings.len()
    );
    for item in findings {
        out.push_str(&format!(
            "  {}:{} — {} — {}\n",
            item.file, item.line, item.class, item.text
        ));
    }
    let mut seen: Vec<&str> = Vec::new();
    for item in findings {
        if !seen.contains(&item.class) {
            seen.push(item.class);
        }
    }
    out.push('\n');
    for class in seen {
        out.push_str(&format!("  {class}: {}\n", cure(class)));
    }
    out
}
