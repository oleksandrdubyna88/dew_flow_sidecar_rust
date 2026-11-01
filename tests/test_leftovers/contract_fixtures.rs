//! The Rust choice of the rule file is held against the mount's own selector, `resolveRule` of
//! `tools/leftovers-check.mjs`, over the same fixture trees — and a rule file that exists but cannot be read is ONE
//! fail-closed finding, never a panic.
//!
//! The shared selector picks by EXISTENCE (the detail, else the core); this guard picks by CONTENT (the file that has a
//! `## Tokens` section). They agree wherever the file that exists is the file that has the section; they differ, on
//! purpose, in two cases that are asserted below by name: a detail that exists WITHOUT the section while the core has
//! it (the guard reads the core), and files that exist with the section in neither (the guard says so once).
//!
//! The fixture trees are made by one Node process with the shipped helper of the mount (`ownedDir` of
//! `tools/lib/owned.mjs`: owner-named, claimed, removed when the process ends), because this integration test of a
//! binary crate cannot reach `TestScratch` and a second temp helper here is what the guard refuses. The process builds
//! the trees from the cases it is handed, prints one JSON line, then waits for its stdin to close and removes them: the
//! Rust side reads the same trees while it waits. The script is a constant; the mount path and the cases reach it
//! through `process.argv`, never through the text of the script. Node missing, or the mount's module not loading, FAILS
//! the test with the reason.

use std::io::{BufRead, BufReader, Read};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use serde_json::{json, Value};

use crate::classes::TOKEN;
use crate::token::*;

const SCRIPT: &str = r#"
import * as fs from "node:fs";
import * as path from "node:path";
import { pathToFileURL } from "node:url";

const [mount, casesJson] = process.argv.slice(1);
const check = await import(pathToFileURL(path.join(mount, "tools", "leftovers-check.mjs")).href);
const owned = await import(pathToFileURL(path.join(mount, "tools", "lib", "owned.mjs")).href);
const cleanups = [];
const scope = { after: (fn) => cleanups.push(fn) };
const out = [];
const canUnreadable = process.platform !== "win32" && process.getuid() !== 0;
try {
  for (const item of JSON.parse(casesJson)) {
    if (item.needsUnreadable && !canUnreadable) continue;
    const root = owned.ownedDir(scope, "rule-pick");
    const shared = path.join(root, ".claude", "rules", "shared");
    fs.mkdirSync(path.join(shared, "common"), { recursive: true });
    for (const [name, file] of [["test-leftovers-detail.md", item.detail], ["test-leftovers.md", item.core]]) {
      if (file.kind === "missing") continue;
      const target = path.join(shared, "common", name);
      fs.writeFileSync(target, file.text ?? "");
      if (file.kind === "unreadable") fs.chmodSync(target, 0);
    }
    const picked = check.resolveRule(shared);
    out.push({ name: item.name, root, shared: picked === undefined ? null : picked });
  }
  process.stdout.write(JSON.stringify({ cases: out, canUnreadable }) + "\n");
  await new Promise((done) => { process.stdin.on("end", done); process.stdin.on("close", done); process.stdin.resume(); });
} finally {
  for (const fn of cleanups.reverse()) fn();
}
"#;

const TABLE: &str = "# Rule\n\n## Tokens\n\n| token | family | repository |\n|---|---|---|\n| `sidecar` | dew_flow | the embedding sidecar |\n\n## Never\n";
const PROSE: &str = "# Core\n\nThe `## Tokens` table of the detail registers it.\n";
const NO_TABLE: &str = "# Detail\n\nNo section here.\n";

fn mount() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join(".claude/rules/shared")
}

fn text(body: &str) -> Value {
    json!({ "kind": "text", "text": body })
}

fn missing() -> Value {
    json!({ "kind": "missing" })
}

fn unreadable() -> Value {
    json!({ "kind": "unreadable" })
}

type Lookup<'a> = &'a dyn Fn(&str) -> Option<(PathBuf, Option<String>)>;

/// Run the Node side over `cases` and hand `(root, shared choice)` per case name to `judge` while the trees exist.
fn with_trees(cases: &Value, judge: impl FnOnce(Lookup<'_>, bool)) {
    let mut child = Command::new("node")
        .args(["--input-type=module", "-e", SCRIPT])
        .arg(mount())
        .arg(cases.to_string())
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap_or_else(|e| panic!("the contract test needs `node` and cannot start it: {e}"));
    let mut line = String::new();
    BufReader::new(child.stdout.take().expect("piped stdout"))
        .read_line(&mut line)
        .expect("the node side's answer");
    if line.trim().is_empty() {
        let mut why = String::new();
        let _ = child
            .stderr
            .take()
            .expect("piped stderr")
            .read_to_string(&mut why);
        let status = child.wait().expect("node's status");
        panic!("the mount's selector could not be loaded or run ({status}): {why}");
    }
    let answer: Value =
        serde_json::from_str(&line).unwrap_or_else(|e| panic!("not JSON ({e}): {line}"));
    let can_unreadable = answer["canUnreadable"].as_bool().unwrap_or(false);
    let found = |name: &str| -> Option<(PathBuf, Option<String>)> {
        answer["cases"]
            .as_array()?
            .iter()
            .find(|c| c["name"] == name)
            .map(|c| {
                (
                    PathBuf::from(c["root"].as_str().expect("root")),
                    c["shared"].as_str().map(str::to_string),
                )
            })
    };
    judge(&found, can_unreadable);
    drop(child.stdin.take()); // the end of input is the signal to remove the trees
    let status = child.wait().expect("node's status");
    assert!(
        status.success(),
        "the node side failed while removing its trees: {status}"
    );
}

fn base(path: &str) -> &str {
    path.rsplit(['/', '\\']).next().unwrap_or(path)
}

fn constants() -> Vec<Constant> {
    let make = |name: &str, value: &str| Constant {
        file: "src/test_scratch.rs".to_string(),
        line: 10,
        name: name.to_string(),
        value: value.to_string(),
    };
    vec![
        make("REPO_TOKEN", "sidecar"),
        make("REPO_ROW", "the embedding sidecar"),
    ]
}

#[test]
fn the_rust_choice_equals_the_mounts_selector_where_they_share_semantics_and_differs_where_documented(
) {
    let cases = json!([
        { "name": "split", "detail": text(TABLE), "core": text(PROSE) },
        { "name": "pre_split", "detail": missing(), "core": text(TABLE) },
        { "name": "both", "detail": text(TABLE), "core": text(TABLE) },
        { "name": "detail_without_section", "detail": text(NO_TABLE), "core": text(TABLE) },
        { "name": "no_section_anywhere", "detail": text(NO_TABLE), "core": text(PROSE) },
        { "name": "neither_file", "detail": missing(), "core": missing() },
    ]);
    with_trees(&cases, |case, _| {
        let choice = |name: &str| {
            let (root, shared) = case(name).unwrap_or_else(|| panic!("no case {name}"));
            (RuleFiles::read(&root), shared)
        };
        // the shared semantics apply: the file that exists is the file that has the section
        for name in ["split", "pre_split", "both"] {
            let (rules, shared) = choice(name);
            let (path, _) = rules
                .resolve()
                .unwrap_or_else(|| panic!("{name}: {:?}", rules.outcome()));
            assert_eq!(
                Some(base(path)),
                shared.as_deref().map(base),
                "{name}: the Rust choice is the mount's choice"
            );
        }
        // the documented difference: the detail exists without the section, the core has it: the mount's selector
        // takes the detail (it exists), this guard takes the core (it has the table)
        let (rules, shared) = choice("detail_without_section");
        let (path, _) = rules.resolve().expect("the core holds the table");
        assert_eq!(base(path), "test-leftovers.md");
        assert_eq!(
            shared.as_deref().map(base),
            Some("test-leftovers-detail.md")
        );
        // the section in neither: the mount's selector still names the detail; this guard says so once
        let (rules, shared) = choice("no_section_anywhere");
        assert_eq!(rules.outcome(), &Outcome::NoSection);
        assert_eq!(
            shared.as_deref().map(base),
            Some("test-leftovers-detail.md")
        );
        // no file in the fixture: the mount's selector falls back to the mount's own files, outside the fixture
        let (root, shared) = case("neither_file").expect("case");
        let rules = RuleFiles::read(&root);
        assert_eq!(rules.outcome(), &Outcome::NotCheckedOut);
        assert!(
            shared.is_none_or(|picked| !Path::new(&picked).starts_with(&root)),
            "a file outside the fixture was picked"
        );
    });
}

#[test]
fn a_rule_file_that_exists_and_cannot_be_read_is_one_finding_naming_it() {
    let cases = json!([
        { "name": "detail_unreadable", "needsUnreadable": true, "detail": unreadable(), "core": text(TABLE) },
        { "name": "core_unreadable_after_a_detail_with_the_table", "needsUnreadable": true, "detail": text(TABLE), "core": unreadable() },
        { "name": "core_unreadable_after_a_detail_without", "needsUnreadable": true, "detail": text(NO_TABLE), "core": unreadable() },
    ]);
    with_trees(&cases, |case, can_unreadable| {
        if !can_unreadable {
            println!("a_rule_file_that_exists_and_cannot_be_read: skipped (root, or a platform where chmod 000 does not deny a read)");
            return;
        }
        // the detail would win if it could be read, so the core's table is not a fallback: ONE finding, the file and the error
        let (root, shared) = case("detail_unreadable").expect("case");
        let rules = RuleFiles::read(&root);
        let found = token_findings(&rules, &constants(), Some("bge-sidecar"));
        assert_eq!(found.len(), 1, "{found:?}");
        assert_eq!(found[0].class, TOKEN);
        assert_eq!(found[0].file, RULE_DETAIL);
        assert!(found[0].text.contains(RULE_DETAIL), "{found:?}");
        assert!(
            found[0].text.contains("cannot be read")
                && found[0].text.to_lowercase().contains("permission denied"),
            "the error is named: {found:?}"
        );
        assert_eq!(
            shared.as_deref().map(base),
            Some("test-leftovers-detail.md")
        );

        // a detail that has the table wins before the core is looked at: its unreadable neighbour is not an error
        let (root, _) = case("core_unreadable_after_a_detail_with_the_table").expect("case");
        let rules = RuleFiles::read(&root);
        assert!(token_findings(&rules, &constants(), Some("bge-sidecar")).is_empty());

        // a detail without the section and a core that cannot be read: the table may be in the core: fail closed
        let (root, _) = case("core_unreadable_after_a_detail_without").expect("case");
        let rules = RuleFiles::read(&root);
        let found = token_findings(&rules, &constants(), Some("bge-sidecar"));
        assert_eq!(found.len(), 1, "{found:?}");
        assert_eq!(found[0].file, RULE_CORE);
    });
}

#[test]
fn an_unreadable_candidate_stops_the_choice_whatever_the_error_is() {
    let rules = RuleFiles::new(
        RuleFile::Unreadable("Input/output error (os error 5)".to_string()),
        RuleFile::Text(TABLE.to_string()),
    );
    assert!(rules.resolve().is_none());
    let found = token_findings(&rules, &constants(), Some("bge-sidecar"));
    assert_eq!(found.len(), 1, "{found:?}");
    assert!(
        found[0].text.contains(RULE_DETAIL) && found[0].text.contains("Input/output error"),
        "{found:?}"
    );
    // a missing detail is not an error: the core is read
    let rules = RuleFiles::new(RuleFile::Missing, RuleFile::Text(TABLE.to_string()));
    assert!(token_findings(&rules, &constants(), Some("bge-sidecar")).is_empty());
}
