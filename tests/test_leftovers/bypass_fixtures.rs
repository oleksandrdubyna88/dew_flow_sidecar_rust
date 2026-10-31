//! Review round 1 of E4 — the planted spellings. The first guard caught three of twelve lines a reviewer planted;
//! each line below is one of those shapes (and the ones the fix added), built from the pieces of `words.rs`, because
//! this file is scanned like any other. Each is refused in a Rust file outside the helper, and its innocent twin is not.
//!
//! The guard is still a SPELLING check: pieces on different lines, a macro or an `include_str!` of a file it does not
//! scan can hide the word. `the_limit_of_a_spelling_check_is_pinned` keeps that sentence honest.

use crate::classes::*;
use crate::words::*;

fn scan(file: &str, text: &str) -> Vec<Finding> {
    scan_text(file, text)
}

fn count(findings: &[Finding], class: &str) -> usize {
    findings.iter().filter(|f| f.class == class).count()
}

fn clean(line: &str) {
    let found = scan("src/x.rs", &format!("{line}\n"));
    assert!(found.is_empty(), "nothing expected for {line:?}: {found:?}");
}

#[test]
fn the_prefix_word_is_found_behind_escapes_pieces_and_classes() {
    let (head, tail) = halves();
    let planted: Vec<(&str, String)> = vec![
        (
            "every character a hex escape",
            format!("let p = \"{}-\";", hex_escaped(WORD_CASED)),
        ),
        (
            "one character a unicode escape",
            format!("let p = \"{}-\";", with_unicode_escape(WORD_CASED, 5)),
        ),
        (
            "a concat! of two literals",
            format!("let p = concat!(\"{head}\", \"{tail}-\");"),
        ),
        (
            "a format! whose arguments are the pieces",
            format!("let p = format!(\"{{}}{{}}-\", \"{head}\", \"{tail}\");"),
        ),
        (
            "a + of a literal and a literal",
            format!("let p = \"{head}\".to_owned() + \"{tail}-\";"),
        ),
        (
            "a String::from + a literal",
            format!("let p = String::from(\"{head}\") + \"{tail}\";"),
        ),
        (
            "a regex with a one-character class",
            format!("let r = Regex::new(r\"^{head}[T]{}\");", &tail[1..]),
        ),
        (
            "a regex with a class of one letter in both cases",
            format!("let r = Regex::new(r\"^{head}[tT]{}\");", &tail[1..]),
        ),
        (
            "a regex whose letter is a regex escape",
            format!("let r = Regex::new(r\"{}{}\");", hex_escaped(head), tail),
        ),
        (
            "a fragment of the word, the first seven characters",
            format!("let p = \"{}\";", &WORD_CASED[..7]),
        ),
        (
            "a fragment in the middle",
            format!("let p = \"{}\";", &WORD_CASED[3..9]),
        ),
        (
            "a fragment at the end",
            format!("let p = \"{}\";", &WORD_CASED[6..]),
        ),
        (
            "a fragment in lower case",
            format!("let p = \"{}\";", WORD[1..8].to_lowercase()),
        ),
        (
            "a longer text that holds a fragment",
            format!("let p = format!(\"x{}\");", &WORD_CASED[..7]),
        ),
    ];
    for (what, line) in &planted {
        let found = scan("src/x.rs", &format!("{line}\n"));
        assert_eq!(count(&found, OWNER_BLIND), 1, "{what}: {line} -> {found:?}");
    }
}

#[test]
fn the_innocent_twins_of_those_spellings_are_not_refused() {
    let (head, tail) = halves();
    // the owner form behind an escape is the owner form; the same name in PIECES is still refused (a name built by
    // hand from pieces is exactly what the helper is for), which is the price of reading fragments
    clean(&format!(
        "let p = \"{}-u1000-sidecar-x\";",
        hex_escaped(WORD_CASED)
    ));
    let pieces = format!("let p = concat!(\"{head}\", \"{tail}-\", \"u1000-\", \"x\");\n");
    assert_eq!(count(&scan("src/x.rs", &pieces), OWNER_BLIND), 1);
    // ordinary text that merely holds a short piece of the word
    clean("let p = \"all the tests pass\"; let q = \"delete it\";");
    clean("let p = \"to delete\"; let q = \"tests to\";");
}

#[test]
fn the_guards_own_piece_builder_is_the_one_pinned_allowance() {
    let (head, tail) = halves();
    let line = format!("pub const X: &str = concat!(\"{head}\", \"{tail}\");\n");
    // the pinned file, by exact path, is not judged for the word; any other file with the same line is
    assert_eq!(WORDS_FILE, "tests/test_leftovers/words.rs");
    assert!(
        scan(WORDS_FILE, &line).is_empty(),
        "{:?}",
        scan(WORDS_FILE, &line)
    );
    for other in [
        "tests/test_leftovers/other.rs",
        "tests/test_leftovers/words2.rs",
        "src/test_leftovers/words.rs",
        "tests/words.rs",
    ] {
        assert_eq!(count(&scan(other, &line), OWNER_BLIND), 1, "{other}");
    }
    // the helper's own files may spell the word, its tests may not (they are the helper's CALLERS)
    assert_eq!(count(&scan("src/test_scratch.rs", &line), OWNER_BLIND), 0);
    assert_eq!(
        count(&scan("src/test_scratch/sweep.rs", &line), OWNER_BLIND),
        0
    );
    assert_eq!(
        count(&scan("src/test_scratch/tests.rs", &line), OWNER_BLIND),
        1
    );
    assert_eq!(
        count(
            &scan("src/test_scratch/tests/sweeping.rs", &line),
            OWNER_BLIND
        ),
        1
    );
}

#[test]
fn a_shared_temp_directory_is_refused_as_a_literal_and_as_a_built_environment_name() {
    for (what, line) in [
        (
            "a shared tmp",
            format!("let d = Path::new(\"{SLASH_TMP}/x\");"),
        ),
        (
            "a shared tmp alone",
            format!("let d = PathBuf::from(\"{SLASH_TMP}\");"),
        ),
        ("a var tmp", format!("let d = Path::new(\"{VAR_TMP}/y\");")),
        (
            "a tmp in pieces",
            format!(
                "let d = concat!(\"{}\", \"{}/x\");",
                &SLASH_TMP[..2],
                &SLASH_TMP[2..]
            ),
        ),
        (
            "a tmp by escapes",
            format!("let d = Path::new(\"{}\");", hex_escaped(SLASH_TMP)),
        ),
        (
            "an environment name built by concat!",
            "let d = std::env::var(concat!(\"TM\", \"PDIR\"));".to_string(),
        ),
        (
            "var_os too",
            "let d = std::env::var_os(concat!(\"TE\", \"MP\"));".to_string(),
        ),
    ] {
        let found = scan("src/x.rs", &format!("{line}\n"));
        assert_eq!(count(&found, RAW_TEMP), 1, "{what}: {line} -> {found:?}");
    }
    // a path that merely has a component of that name is not the shared directory
    clean("let d = Path::new(\"src/tmp/x\"); let e = \"/tmpfoo\"; let f = \"/home/tmp-x\";");
    clean("let a = std::env::var(\"HOME\"); let b = format!(\"{}\", concat!(\"a\", \"b\"));");
    // and an exemption clears it, like the other raw-temp spellings
    let exempt = format!(
        "// {MARK} raw-temp — a path compared as text, no file is made here\nlet d = Path::new(\"{SLASH_TMP}/x\");\n"
    );
    assert!(scan("src/x.rs", &exempt).is_empty());
}

#[test]
fn a_scratch_dropped_by_a_call_or_by_the_end_of_its_block_is_refused() {
    let new = concat!("Test", "Scratch::new");
    let seed = concat!("seed_scr", "atch");
    let cache = concat!("model_cache_with_a_", "tokenizer");
    for (what, text) in [
        ("drop(ctor)", format!("drop({new}(\"x\"));\n")),
        ("drop(ctor_in)", format!("drop({new}_in(&p, \"x\"));\n")),
        ("drop of a seed", format!("drop({seed}(\"x\"));\n")),
        ("drop of a cache", format!("drop({cache}(\"x\"));\n")),
        ("drop with spaces", format!("std::mem::drop( {new}(\"x\") );\n")),
        (
            "a block that returns the path of its scratch (one line)",
            format!("let p = {{ let s = {new}(\"x\"); s.path().to_path_buf() }};\n"),
        ),
        (
            "a block that returns the path of its scratch (lines)",
            format!("let p = {{\n    let s = {new}(\"x\");\n    s.path().to_path_buf()\n}};\n"),
        ),
        (
            "a seed block",
            format!("let p = {{\n    let s = {seed}(\"x\");\n    s.scratch.path().join(\"a\")\n}};\n"),
        ),
        (
            "a nested block",
            format!("fn f() -> PathBuf {{\n    let s = {new}(\"x\");\n    if true {{ return PathBuf::new(); }}\n    s.slash_path().into()\n}}\n"),
        ),
    ] {
        let found = scan("src/x.rs", &text);
        assert_eq!(count(&found, DROPPED_SCRATCH), 1, "{what}: {text} -> {found:?}");
    }
    for (what, text) in [
        ("a named drop", "let scratch = ctor(); drop(scratch);\n".replace("ctor", new)),
        (
            "a block whose scratch outlives its use",
            format!("let n = {{\n    let s = {new}(\"x\");\n    let p = s.path().to_path_buf();\n    p.exists()\n}};\n"),
        ),
        (
            "a block whose tail does not touch the scratch",
            format!("let n = {{\n    let s = {new}(\"x\");\n    let _ = s.path();\n    5\n}};\n"),
        ),
        (
            "the path used and the scratch returned with it",
            format!("fn f() -> (TestScratch, PathBuf) {{\n    let s = {new}(\"x\");\n    let p = s.path().to_path_buf();\n    (s, p)\n}}\n"),
        ),
    ] {
        let found = scan("src/x.rs", &text);
        assert_eq!(count(&found, DROPPED_SCRATCH), 0, "{what}: {text} -> {found:?}");
    }
}

/// The sentence that says what this guard is not, kept next to the code that makes it true: the research file
/// names the same limit, and a guard that claimed more would be a lie the next reader believes.
#[test]
fn the_limit_of_a_spelling_check_is_pinned() {
    let hidden = "let p = include_str!(\"data/names.txt\");\n";
    assert!(scan("src/x.rs", hidden).is_empty());
    let built = "let p = some_macro!(a, b);\n";
    assert!(scan("src/x.rs", built).is_empty());
    let two_lines = format!(
        "let a = \"{}\";\nlet b = \"{}\";\n",
        &WORD_CASED[..4],
        &WORD_CASED[4..8]
    );
    assert!(
        scan("src/x.rs", &two_lines).is_empty(),
        "pieces of 4 characters on different lines are not found: a spelling check, not a proof"
    );
}
