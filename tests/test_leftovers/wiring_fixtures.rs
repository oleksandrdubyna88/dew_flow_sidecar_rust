//! The guard's parts are wired together: a synthetic tree, in memory, that breaks every part at once, and one that
//! scans nothing. The sibling modules prove each part on fixtures; this proves `scan_inputs` calls them all.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use crate::classes::*;
use crate::repo::{scan_inputs, Inputs};
use crate::scripts::CLASSIFIER;
use crate::token::RULE;
use crate::words::*;

fn real_rule() -> Option<String> {
    std::fs::read_to_string(Path::new(env!("CARGO_MANIFEST_DIR")).join(RULE)).ok()
}

fn classifier() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join(CLASSIFIER)
}

const HELPER: &str = "pub(crate) const REPO_TOKEN: &str = \"sidecar\";\npub(crate) const REPO_ROW: &str = \"the embedding sidecar\";\n";

fn scan_files(files: &[(&str, String)], rule: Option<String>, cargo: Option<&str>) -> Vec<Finding> {
    let map: HashMap<String, Vec<u8>> = files
        .iter()
        .map(|(path, text)| ((*path).to_string(), text.clone().into_bytes()))
        .collect();
    let read = |path: &str| map.get(path).cloned();
    scan_inputs(&Inputs {
        listing: files.iter().map(|(path, _)| (*path).to_string()).collect(),
        read: &read,
        classifier: classifier(),
        rule,
        cargo_toml: cargo.map(str::to_string),
    })
    .findings
}

fn classes_of(findings: &[Finding]) -> Vec<&'static str> {
    let mut classes: Vec<&'static str> = findings.iter().map(|f| f.class).collect();
    classes.sort_unstable();
    classes.dedup();
    classes
}

#[test]
fn a_tree_that_breaks_every_part_is_red_in_every_part() {
    let script = format!("const P = \"{PREFIX_TEXT}\";\nnames.filter((n) => n.startsWith(P));\n");
    let files = [
        ("src/test_scratch.rs", HELPER.to_string()),
        ("src/x.rs", format!("let t = {TEMP_CALL};\n")),
        ("tests/t.rs", "fn t() {}\n".to_string()),
        (".github/workflows/ci.yml", "jobs: {}\n".to_string()),
        ("tools/x.mjs", script),
        ("scripts/run.sh", "echo hi\n".to_string()),
    ];
    let found = scan_files(
        &files,
        real_rule(),
        Some("[package]\nname = \"another-crate\"\n"),
    );
    let classes = classes_of(&found);
    for class in [RAW_TEMP, OWNER_BLIND, TOKEN, DISCOVERY] {
        assert!(classes.contains(&class), "{class} missing from {found:?}");
    }
    assert!(
        found
            .iter()
            .any(|f| f.class == DISCOVERY && f.file == "scripts/run.sh"),
        "the unread language is named: {found:?}"
    );
    assert!(
        found
            .iter()
            .any(|f| f.class == OWNER_BLIND && f.file == "tools/x.mjs" && f.line == 2),
        "the classifier judged the script: {found:?}"
    );
    assert!(
        found
            .iter()
            .any(|f| f.class == RAW_TEMP && f.file == "src/x.rs" && f.line == 1),
        "{found:?}"
    );
}

#[test]
fn a_tree_with_nothing_to_scan_is_red_and_a_clean_one_is_green() {
    let found = scan_files(
        &[],
        real_rule(),
        Some("[package]\nname = \"bge-sidecar\"\n"),
    );
    let nothing = found.iter().filter(|f| f.class == DISCOVERY).count();
    assert_eq!(nothing, 3, "src, tests, workflows: {found:?}");

    let clean = [
        ("src/test_scratch.rs", HELPER.to_string()),
        ("src/x.rs", "fn x() {}\n".to_string()),
        ("tests/t.rs", "fn t() {}\n".to_string()),
        (".github/workflows/ci.yml", "jobs: {}\n".to_string()),
    ];
    let found = scan_files(
        &clean,
        real_rule(),
        Some("[package]\nname = \"bge-sidecar\"\n"),
    );
    assert!(found.is_empty(), "{found:?}");

    // a missing mount is red for the token even on a clean tree
    let found = scan_files(&clean, None, Some("[package]\nname = \"bge-sidecar\"\n"));
    assert_eq!(classes_of(&found), vec![TOKEN], "{found:?}");
}

#[test]
fn a_listed_file_that_is_gone_or_binary_is_skipped_not_fatal() {
    let mut files = vec![("src/test_scratch.rs", HELPER.to_string())];
    files.push(("src/x.rs", "fn x() {}\n".to_string()));
    files.push(("tests/t.rs", "fn t() {}\n".to_string()));
    files.push((".github/workflows/ci.yml", "jobs: {}\n".to_string()));
    files.push(("src/binary.rs", format!("\0{TEMP_CALL}\n")));
    let found = scan_files(
        &files,
        real_rule(),
        Some("[package]\nname = \"bge-sidecar\"\n"),
    );
    assert!(
        found.is_empty(),
        "a binary file is not text to be read: {found:?}"
    );
}
