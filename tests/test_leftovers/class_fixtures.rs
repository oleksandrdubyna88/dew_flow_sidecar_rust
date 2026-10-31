//! Sc. 2 — every class is refused once, and a valid exemption passes. Fixture strings through the pure function;
//! every fixture is made of the pieces in `words.rs`, because this file is scanned like any other.
//!
//! The break-it of each detector: remove its pattern from `classes.rs` and the scenario below that names it is red
//! (recorded by hand in the pull request — the scenarios are the mutation oracle).

use crate::classes::*;
use crate::words::*;

fn scan(file: &str, text: &str) -> Vec<Finding> {
    scan_text(file, text)
}

fn count(findings: &[Finding], class: &str) -> usize {
    findings.iter().filter(|f| f.class == class).count()
}

/// Exactly `n` findings of `class` and nothing else.
fn only(findings: &[Finding], class: &str, n: usize) {
    assert_eq!(
        count(findings, class),
        n,
        "{class} x{n} expected, got {findings:?}"
    );
    assert_eq!(findings.len(), n, "and nothing else, got {findings:?}");
}

fn none(findings: &[Finding]) {
    assert!(findings.is_empty(), "nothing expected, got {findings:?}");
}

#[test]
fn a_raw_temp_call_is_refused_in_every_spelling_and_names_its_line() {
    let found = scan("src/x.rs", &format!("fn a() {{}}\nlet t = {TEMP_CALL};\n"));
    only(&found, RAW_TEMP, 1);
    assert_eq!((found[0].file.as_str(), found[0].line), ("src/x.rs", 2));

    let spellings: [(&str, String); 9] = [
        (
            "src/x.rs",
            concat!("std::env::var(\"TM", "PDIR\")").to_string(),
        ),
        (
            "src/x.rs",
            concat!("std::env::var_os(\"TE", "MP\")").to_string(),
        ),
        (
            "src/x.rs",
            concat!("let d = temp", "file::temp", "dir();").to_string(),
        ),
        (
            "http/x.mjs",
            concat!("mkdt", "empSync(join(a, \"x\"))").to_string(),
        ),
        (
            "http/x.js",
            concat!("const d = os.tmp", "dir();").to_string(),
        ),
        (
            "http/x.mjs",
            concat!("const d = process.env.TM", "PDIR;").to_string(),
        ),
        ("src/x.rs", concat!("Get", "TempPath(0, null)").to_string()),
        (
            "src/x.rs",
            concat!("Directory.CreateTemp", "Subdirectory(\"x\")").to_string(),
        ),
        (
            "src/x.rs",
            concat!("Path.GetTemp", "FileName()").to_string(),
        ),
    ];
    for (file, line) in spellings {
        only(&scan(file, &line), RAW_TEMP, 1);
    }
    // each of the temp-dir spellings alone, so no pattern can hide behind another
    let alone: [&str; 4] = [
        concat!("let d = temp_", "dir();"),
        concat!("let f = std::env::temp_", "dir;"),
        concat!("let f = temp", "file::NamedTempFile::new();"),
        concat!("let d = temp", "dir().unwrap();"),
    ];
    for line in alone {
        only(&scan("src/x.rs", line), RAW_TEMP, 1);
    }
    let quoted = concat!("Command::new(\"mk", "temp\").arg(\"-d\")");
    only(&scan("src/x.rs", quoted), RAW_TEMP, 1);
}

#[test]
fn a_temp_path_in_a_workflow_is_refused_anywhere_but_a_whole_line_comment() {
    let make = concat!("mk", "temp -d");
    let workflow = ".github/workflows/x.yml";
    only(
        &scan(workflow, &format!("      - run: DIR=$({make})\n")),
        RAW_TEMP,
        1,
    );
    only(
        &scan(workflow, &format!("      - run: echo ok # {make}\n")),
        RAW_TEMP,
        1,
    );
    none(&scan(workflow, &format!("      # {make} was here once\n")));
    // RUNNER_TEMP is job-scoped: the runner empties it at the start and end of every job (rule §7).
    none(&scan(workflow, "      - run: cd \"$RUNNER_TEMP\" && ls\n"));
    // a '#' line in Rust is an attribute, never a comment: it is read
    let attribute = format!("#[cfg(test)]\n{TEMP_CALL};\n");
    let found = scan("src/x.rs", &attribute);
    only(&found, RAW_TEMP, 1);
    assert_eq!(found[0].line, 2);
    none(&scan("src/x.rs", &format!("// {TEMP_CALL}\n")));
    none(&scan("http/x.mjs", &format!("// {TEMP_CALL}\n")));
}

#[test]
fn the_helpers_temp_root_accessor_is_a_raw_temp_bypass_outside_the_helper() {
    let bypass = format!("std::fs::create_dir({TEMP_ROOT}().join(\"x\")).unwrap();\n");
    only(&scan("src/tokens.rs", &bypass), RAW_TEMP, 1);
    only(&scan("tests/other.rs", &bypass), RAW_TEMP, 1);
    none(&scan("src/test_scratch.rs", &bypass));
    none(&scan("src/test_scratch/tests/x.rs", &bypass));
    // only the whole word is the accessor
    none(&scan("src/x.rs", "let other_temp_root_name = 1;\n"));
}

#[test]
fn a_worktree_is_refused() {
    only(
        &scan(
            ".github/workflows/x.yml",
            &format!("      - run: {WORKTREE_ADD}\n"),
        ),
        RAW_WORKTREE,
        1,
    );
    let array = concat!(
        "Command::new(\"git\").args([\"work",
        "tree\", \"add\", \"..\"])"
    );
    only(&scan("src/x.rs", array), RAW_WORKTREE, 1);
}

#[test]
fn any_container_start_is_refused_until_the_guard_learns_the_carve_out() {
    let ci = ".github/workflows/x.yml";
    only(
        &scan(ci, &format!("      - run: {DOCKER_RUN_CMD}\n")),
        DOCKER_RUN,
        1,
    );
    only(
        &scan(ci, concat!("      - run: dock", "er-compose up -d\n")),
        DOCKER_RUN,
        1,
    );
    let compose = concat!("dock", "er compose up -d");
    let found = scan(ci, compose);
    assert_eq!(count(&found, DOCKER_RUN), 1, "{found:?}");
    only(
        &scan("src/x.rs", concat!("Command::new(\"dock", "er\")")),
        DOCKER_RUN,
        1,
    );
}

#[test]
fn a_container_resource_name_must_be_of_the_owner_form() {
    let ci = ".github/workflows/x.yml";
    only(
        &scan(ci, &format!("      - run: {DOCKER_VOLUME}v\n")),
        DOCKER_NAME,
        1,
    );
    none(&scan(
        ci,
        &format!("      - run: {DOCKER_VOLUME}{}\n", owner_name()),
    ));
    none(&scan(ci, &format!("      - run: {DOCKER_VOLUME}\"$V\"\n")));
    let unnamed = scan(ci, &format!("      - run: {DOCKER_VOLUME}\n"));
    only(&unnamed, DOCKER_NAME, 1);
    let found = scan(ci, &format!("      - run: {DOCKER_COMPOSE}sidecar-x up\n"));
    assert_eq!(count(&found, DOCKER_NAME), 1, "{found:?}");
    let by_flag = format!("      - run: x --na{}", "me box\n");
    only(&scan(ci, &by_flag), DOCKER_NAME, 1);
    none(&scan(ci, &format!("      - run: x --na{} \"$N\"\n", "me")));
    none(&scan(
        ci,
        &format!("      - run: x --na{} {}\n", "me", owner_name()),
    ));
    // a literal default is the name the command gets when V is unset: judged
    only(
        &scan(
            ci,
            &format!("      - run: x --na{} \"${{N:-box}}\"\n", "me"),
        ),
        DOCKER_NAME,
        1,
    );
    // `git diff --name-only` is not `--name`
    none(&scan(ci, "      - run: git diff --name-only\n"));
    let env = concat!("COMPOSE_PROJECT_", "NAME=box");
    only(&scan(ci, &format!("      env: {env}\n")), DOCKER_NAME, 1);
    only(
        &scan("src/x.rs", &format!("builder.With{}(\"box\");\n", "Name")),
        DOCKER_NAME,
        1,
    );
    none(&scan(
        "src/x.rs",
        &format!("builder.With{}(\"{}\");\n", "Name", owner_name()),
    ));
}

#[test]
fn a_delete_all_against_an_unprefixed_store_is_refused_and_prose_is_not() {
    let ci = ".github/workflows/x.yml";
    only(
        &scan(ci, &format!("      - run: psql -c \"{STORE_WIPE}\"\n")),
        SHARED_STORE,
        1,
    );
    only(
        &scan("src/x.rs", concat!("redis.cmd(\"FLUSH", "ALL\");")),
        SHARED_STORE,
        1,
    );
    none(&scan(
        "src/x.rs",
        &format!(
            "let q = \"{} {};\";\n",
            concat!("DR", "OP TABLE"),
            owner_name()
        ),
    ));
    none(&scan(
        "src/x.rs",
        "// the table was truncated and rebuilt\nlet note = \"truncated\";\n",
    ));
    only(
        &scan(
            "http/x.js",
            concat!("await db.collection(\"a\").delete", "Many({});"),
        ),
        SHARED_STORE,
        1,
    );
    // exemption (a): needs its lock
    let exempt = format!("// {MARK} shared-store — lock: the suite-wide store lock held by the fixture\n{STORE_WIPE};\n");
    none(&scan("src/x.rs", &exempt));
    let no_lock = format!("// {MARK} shared-store — a perfectly fine reason here\n{STORE_WIPE};\n");
    let found = scan("src/x.rs", &no_lock);
    assert_eq!(count(&found, EXEMPTION), 1, "{found:?}");
    assert_eq!(
        count(&found, SHARED_STORE),
        1,
        "the invalid exemption clears nothing: {found:?}"
    );
}

#[test]
fn the_reaper_switched_off_is_refused_and_no_exemption_clears_it() {
    only(
        &scan("src/x.rs", &format!("env.set(\"{RYUK_OFF_TEXT}\");\n")),
        RYUK_OFF,
        1,
    );
    let above = format!(
        "// {MARK} ryuk-off — a perfectly fine reason here\nenv.set(\"{RYUK_OFF_TEXT}\");\n"
    );
    let found = scan("src/x.rs", &above);
    assert_eq!(count(&found, RYUK_OFF), 1, "still refused: {found:?}");
    assert_eq!(
        count(&found, EXEMPTION),
        1,
        "and the attempt is itself a finding: {found:?}"
    );
}

#[test]
fn a_legacy_name_is_refused_and_an_exemption_clears_only_that_class() {
    let literal = format!("const OLD: &str = \"{}\";\n", legacy_name());
    let found = scan("src/x.rs", &literal);
    assert_eq!(count(&found, LEGACY_NAME), 1, "{found:?}");
    assert_eq!(
        count(&found, OWNER_BLIND),
        1,
        "two lines, one class each: {found:?}"
    );
    assert_eq!(found.len(), 2);

    let exempt =
        format!("// {MARK} legacy-name — a fixture for the sweep that lists it\n{literal}");
    let found = scan("src/x.rs", &exempt);
    assert_eq!(count(&found, LEGACY_NAME), 0, "{found:?}");
    assert_eq!(
        count(&found, OWNER_BLIND),
        1,
        "the owner-blind line remains: {found:?}"
    );

    // the legacy class is read in the helper's own files too; owner-blind is not
    let found = scan("src/test_scratch/tests/x.rs", &literal);
    only(&found, LEGACY_NAME, 1);
}

#[test]
fn the_owner_form_the_quarantine_sibling_and_the_environment_names_are_not_findings() {
    let owner = format!("const A: &str = \"{}\";\n", owner_name());
    none(&scan("src/x.rs", &owner));
    let quarantine = format!("let q = format!(\".{{name}}.{WORD}-quarantine-{{pid}}-{{hex}}\");\n");
    none(&scan("src/x.rs", &quarantine));
    let env = format!("std::env::var(\"{}_OWNER\");\n", WORD.to_ascii_uppercase());
    none(&scan("src/x.rs", &env));
    let label = format!("let l = \"{WORD_CASED}.owner\".len();\n");
    none(&scan("src/x.rs", &label));
    // upper case with an owner segment: still the owner form
    none(&scan(
        "src/x.rs",
        &format!("let n = \"{}-U7-X-1\";\n", WORD.to_ascii_uppercase()),
    ));
    // a lower-case environment-like name is a legacy name
    let lower = format!("let n = \"{WORD}_owner\";\n");
    assert_eq!(count(&scan("src/x.rs", &lower), LEGACY_NAME), 1);
    // a constant built from the token alone is the legacy form
    let built = format!(
        "let n = format!(\"{{{}}}{{REPO_TOKEN}}-x\");\n",
        concat!("PRE", "FIX")
    );
    assert_eq!(count(&scan("src/x.rs", &built), LEGACY_NAME), 1);
    let spelled = format!("let n = \"{PREFIX_TEXT}{{REPO_TOKEN}}-x\";");
    assert_eq!(count(&scan("src/x.rs", &spelled), LEGACY_NAME), 1);
    let added = format!("let n = {} + REPO_TOKEN;", concat!("PRE", "FIX"));
    assert_eq!(count(&scan("src/x.rs", &added), LEGACY_NAME), 1);
}

#[test]
fn owner_blind_in_rust_takes_no_tracer_and_catches_every_spelling_of_the_prefix() {
    let blind: Vec<(&str, String)> = vec![
        ("a regex", format!("Regex::new(r\"^{WORD}[-_]sidecar\")")),
        ("a glob", format!("let g = \"{WORD_CASED}-*\";")),
        ("a glob of the word", format!("glob(\"{WORD_CASED}*\")")),
        (
            "a regex with no owner",
            format!("Regex::new(\"^{WORD}[-_]\")"),
        ),
        (
            "a concatenation",
            format!("let p = \"{WORD_CASED}\".to_string() + \"-\";"),
        ),
        (
            "an interpolation",
            format!("format!(\"{PREFIX_TEXT}{{owner}}-x\")"),
        ),
        ("a bare word", format!("let w = \"{WORD}\";")),
        (
            "a bare owner letter",
            format!("let w = \"{PREFIX_TEXT}u-x\";"),
        ),
        (
            "a name with no closing separator",
            format!("let w = \"{PREFIX_TEXT}u1000\";"),
        ),
        (
            "the helper's constant",
            format!("name.starts_with({PREFIX_CONST});"),
        ),
        (
            "the helper's constant in a format",
            format!("format!(\"{{{PREFIX_CONST}}}x\")"),
        ),
    ];
    for (what, line) in &blind {
        let found = scan("src/x.rs", line);
        assert_eq!(count(&found, OWNER_BLIND), 1, "{what}: {line} -> {found:?}");
    }
    // the helper's own files may spell the prefix
    for (what, line) in &blind {
        let found = scan("src/test_scratch.rs", line);
        assert_eq!(
            count(&found, OWNER_BLIND),
            0,
            "{what} in the helper: {found:?}"
        );
        let found = scan("src/test_scratch/sweep.rs", line);
        assert_eq!(
            count(&found, OWNER_BLIND),
            0,
            "{what} in the helper folder: {found:?}"
        );
    }
    // an owner segment right after the prefix is the owner form, in every case
    none(&scan(
        "src/x.rs",
        &format!("let n = \"{PREFIX_TEXT}W42-x\";\n"),
    ));
    // a comment is not code
    none(&scan(
        "src/x.rs",
        &format!("// {WORD}-x and {PREFIX_CONST}\n"),
    ));
    // an identifier that merely contains the constant's letters is not the constant
    none(&scan(
        "src/x.rs",
        "let LOG_PREFIX = 1; let PREFIX_LEN = 2;\n",
    ));
}

#[test]
fn a_prefix_bound_in_a_workflow_or_a_script_is_traced_by_the_shipped_classifier_not_here() {
    // The textual classes do not judge owner-blind in scripts: that is `scripts.rs`, validated in script_fixtures.
    let script = format!("const P = \"{PREFIX_TEXT}\";\nnames.filter((n) => n.startsWith(P));\n");
    let found = scan("tools/x.mjs", &script);
    assert_eq!(count(&found, OWNER_BLIND), 0, "{found:?}");
    assert_eq!(count(&found, LEGACY_NAME), 0, "{found:?}");
}

#[test]
fn a_valid_exemption_clears_its_class_on_its_line_or_the_next_and_malformed_ones_are_findings() {
    let raw = format!("{TEMP_CALL};\n");
    let well_formed = format!(
        "// {MARK} raw-temp — the helper's own temp root; every scratch is made here\n{raw}"
    );
    none(&scan("src/x.rs", &well_formed));
    let trailing =
        format!("{TEMP_CALL}; // {MARK} raw-temp — the helper's own temp root, a trailing form\n");
    none(&scan("src/x.rs", &trailing));
    let hash = format!(
        "      - run: mk{} -d # {MARK} raw-temp — a runner we do not control here\n",
        "temp"
    );
    none(&scan(".github/workflows/x.yml", &hash));
    // two lines above is out of reach
    let far = format!(
        "// {MARK} raw-temp — the helper's own temp root; every scratch is made here\n\n{raw}"
    );
    only(&scan("src/x.rs", &far), RAW_TEMP, 1);
    // a class the exemption does not name is not cleared
    let wrong = format!(
        "// {MARK} raw-worktree — the helper's own temp root; every scratch is made here\n{raw}"
    );
    let found = scan("src/x.rs", &wrong);
    only(&found, RAW_TEMP, 1);

    let bad: [(&str, &str); 10] = [
        ("a placeholder reason", "raw-temp — needed"),
        ("owner-blind", "owner-blind — a perfectly fine reason here"),
        (
            "an unknown class",
            "nonsense — a perfectly fine reason here",
        ),
        ("no class", "— a perfectly fine reason here"),
        ("no reason", "raw-temp"),
        ("two words", "raw-temp — two words"),
        ("two long words", "raw-temp — extraordinarily important"),
        ("three short words", "raw-temp — a b c"),
        (
            "a placeholder of enough words",
            "raw-temp — see above for the reason",
        ),
        (
            "the repository's own class",
            "dropped-scratch — a perfectly fine reason here",
        ),
    ];
    for (what, rest) in bad {
        let found = scan("src/x.rs", &format!("// {MARK} {rest}\nlet x = 1;\n"));
        only(&found, EXEMPTION, 1);
        assert_eq!(found[0].line, 1, "{what}");
    }
}

#[test]
fn a_scratch_that_is_not_held_by_a_named_binding_is_refused() {
    let new = |args: &str| format!("{}({args})", concat!("Test", "Scratch::new"));
    let dropped: Vec<String> = vec![
        format!("let _ = {};", new("\"x\"")),
        format!(
            "let _ = {}_in(&parent, \"x\");",
            concat!("Test", "Scratch::new")
        ),
        format!(
            "let (_, from, to) = {}(\"x\");",
            concat!("seed_scr", "atch")
        ),
        format!("let {} {{ from, to, .. }} = x;", concat!("Seed", "Scratch")),
        format!("{};", new("\"x\"")),
        format!("let p = {}.path().to_path_buf();", new("\"x\"")),
        format!(
            "let _ = {}(\"x\");",
            concat!("model_cache_with_a_", "tokenizer")
        ),
    ];
    for line in &dropped {
        let found = scan("src/x.rs", &format!("{line}\n"));
        assert_eq!(count(&found, DROPPED_SCRATCH), 1, "{line} -> {found:?}");
    }
    let held: Vec<String> = vec![
        format!("let _scratch = {};", new("\"x\"")),
        format!("let scratch = {};", new("\"x\"")),
        format!("let seed = {}(\"x\");", concat!("seed_scr", "atch")),
        format!(
            "let {} {{ scratch, from, .. }} = y;",
            concat!("Seed", "Scratch")
        ),
        format!(
            "let cache = {}(\"x\");",
            concat!("model_cache_with_a_", "tokenizer")
        ),
        format!(
            "let scratch = {};\nlet root = scratch.path();",
            new("\"x\"")
        ),
    ];
    for line in &held {
        none(&scan("src/x.rs", &format!("{line}\n")));
    }
}
