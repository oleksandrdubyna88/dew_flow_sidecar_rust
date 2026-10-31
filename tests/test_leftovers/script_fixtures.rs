//! `owner-blind` in scripts and workflows: the shipped classifier's verdict, validated, fail closed. These run the
//! real `node` and the mounted classifier — a missing mount or node is red here, never skipped.
//!
//! Fixtures are made of the pieces in `words.rs`. Paths travel in `argv` and the text on stdin (`scripts.rs`);
//! nothing here is interpolated into node's source.

use std::path::{Path, PathBuf};

use crate::classes::{Finding, Lang, OWNER_BLIND};
use crate::scripts::*;
use crate::words::*;

fn classifier() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join(CLASSIFIER)
}

fn judge(name: &str, lang: Lang, text: &str) -> Vec<Finding> {
    let sources = [ScriptSource::new(name, lang, text)];
    script_owner_blind(&sources, &classifier())
}

fn lines_of(findings: &[Finding]) -> Vec<usize> {
    assert!(
        findings.iter().all(|f| f.class == OWNER_BLIND),
        "{findings:?}"
    );
    findings.iter().map(|f| f.line).collect()
}

#[test]
fn a_prefix_bound_on_one_workflow_line_and_swept_nine_lines_later_is_one_finding_on_the_later_line()
{
    let mut text = format!("      - run: |\n          P={PREFIX_TEXT}\n");
    for filler in 0..8 {
        text.push_str(&format!("          echo step {filler}\n"));
    }
    text.push_str("          rm -rf \"$P\"*\n");
    let found = judge(".github/workflows/x.yml", Lang::Workflow, &text);
    assert_eq!(lines_of(&found), vec![11], "{found:?}");
    assert_eq!(found[0].file, ".github/workflows/x.yml");
}

#[test]
fn a_name_from_elsewhere_and_a_prefix_split_in_two_bind_nothing() {
    let elsewhere = "      - run: |\n          P=\"$OTHER\"\n          rm -rf \"$P\"*\n";
    assert!(judge(".github/workflows/x.yml", Lang::Workflow, elsewhere).is_empty());
    let (a, b) = WORD.split_at(7);
    let split = format!("      - run: |\n          P='{a}''{b}-'\n          rm -rf \"$P\"*\n");
    assert!(judge(".github/workflows/x.yml", Lang::Workflow, &split).is_empty());
}

#[test]
fn a_script_constant_holding_the_prefix_and_used_as_a_filter_is_one_finding() {
    let text = format!("const P = \"{PREFIX_TEXT}\";\nnames.filter((n) => n.startsWith(P));\n");
    let found = judge("tools/x.mjs", Lang::Js, &text);
    assert_eq!(lines_of(&found), vec![2], "{found:?}");
}

#[test]
fn a_literal_glob_is_a_finding_and_the_owner_form_and_comments_are_not() {
    let glob = format!("      - run: rm -rf /srv/{WORD_CASED}-*\n");
    assert_eq!(
        lines_of(&judge(".github/workflows/x.yml", Lang::Workflow, &glob)),
        vec![1]
    );

    let owner = format!("      - run: rm -rf /srv/{PREFIX_TEXT}u1000-sidecar-*\n");
    assert!(judge(".github/workflows/x.yml", Lang::Workflow, &owner).is_empty());

    let comment = format!("      # rm -rf /srv/{WORD_CASED}-*\n      - run: echo ok\n");
    assert!(judge(".github/workflows/x.yml", Lang::Workflow, &comment).is_empty());

    let js_comment =
        format!("// names.filter((n) => n.startsWith(\"{PREFIX_TEXT}\"))\nconsole.log(1);\n");
    assert!(judge("http/x.js", Lang::Js, &js_comment).is_empty());

    let js = format!("names.filter((n) => n.startsWith(\"{PREFIX_TEXT}\"));\n");
    assert_eq!(lines_of(&judge("http/x.js", Lang::Js, &js)), vec![1]);
}

#[test]
fn no_file_with_the_prefix_means_node_is_not_started_at_all() {
    // a classifier path that does not exist would be red if anything were sent
    let sources = [ScriptSource::new("a.mjs", Lang::Js, "console.log(1);\n")];
    assert!(script_owner_blind(&sources, Path::new("/no/such/classifier.mjs")).is_empty());
}

#[test]
fn a_classifier_that_is_missing_is_one_unjudged_finding_naming_the_submodule() {
    let text = format!("names.filter((n) => n.startsWith(\"{PREFIX_TEXT}\"));\n");
    let sources = [ScriptSource::new("a.mjs", Lang::Js, &text)];
    let found = script_owner_blind(&sources, Path::new("/no/such/classifier.mjs"));
    assert_eq!(found.len(), 1, "{found:?}");
    assert_eq!(
        (found[0].file.as_str(), found[0].line, found[0].class),
        ("a.mjs", 1, OWNER_BLIND)
    );
    assert!(
        found[0]
            .text
            .contains("git submodule update --init .claude/rules/shared"),
        "{found:?}"
    );
    assert!(found[0].text.contains("nothing was judged"), "{found:?}");
}

#[test]
fn an_answer_that_does_not_hold_is_one_unjudged_finding_never_nothing_found() {
    let text = format!("// {WORD_CASED} in a comment only\nconsole.log(1);\n");
    let sources = [ScriptSource::new("a.mjs", Lang::Js, &text)];
    // a stand-in module whose answer names lines the file does not have
    let stub =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/test_leftovers/stub_classifier.mjs");
    let found = script_owner_blind(&sources, &stub);
    assert_eq!(found.len(), 1, "{found:?}");
    assert!(found[0].text.contains("outside 1..="), "{found:?}");

    // a module that does not load is a node failure, also unjudged
    let not_a_module = Path::new(env!("CARGO_MANIFEST_DIR")).join("Cargo.toml");
    let found = script_owner_blind(&sources, &not_a_module);
    assert_eq!(found.len(), 1, "{found:?}");
    assert!(found[0].text.contains("node exited with"), "{found:?}");
}

#[test]
fn the_answer_is_validated_against_what_was_sent() {
    let sent = [("a.mjs", 3usize), ("b.yml", 10)];
    let ok = validate_answer(
        r#"{"findings":[{"file":"a.mjs","line":3},{"file":"b.yml","line":1}]}"#,
        &sent,
    );
    assert_eq!(
        ok,
        Ok(vec![("a.mjs".to_string(), 3), ("b.yml".to_string(), 1)])
    );
    assert_eq!(validate_answer(r#"{"findings":[]}"#, &sent), Ok(Vec::new()));

    let refused = [
        ("", "not JSON"),
        ("nope", "not JSON"),
        (r#"{}"#, "no `findings`"),
        (r#"{"findings":{}}"#, "no `findings`"),
        (r#"{"findings":[{"line":1}]}"#, "names no file"),
        (r#"{"findings":[{"file":"a.mjs"}]}"#, "names no line"),
        (
            r#"{"findings":[{"file":"a.mjs","line":"1"}]}"#,
            "names no line",
        ),
        (
            r#"{"findings":[{"file":"a.mjs","line":1.5}]}"#,
            "names no line",
        ),
        (r#"{"findings":[{"file":"c.mjs","line":1}]}"#, "not sent"),
        (
            r#"{"findings":[{"file":"a.mjs","line":0}]}"#,
            "outside 1..=3",
        ),
        (
            r#"{"findings":[{"file":"a.mjs","line":4}]}"#,
            "outside 1..=3",
        ),
        (
            r#"{"findings":[{"file":"a.mjs","line":-1}]}"#,
            "names no line",
        ),
        (
            r#"{"findings":[{"file":"a.mjs","line":2},{"file":"b.yml","line":11}]}"#,
            "outside 1..=10",
        ),
    ];
    for (answer, why) in refused {
        let error = validate_answer(answer, &sent).expect_err(answer);
        assert!(error.contains(why), "{answer}: {error}");
    }
}
