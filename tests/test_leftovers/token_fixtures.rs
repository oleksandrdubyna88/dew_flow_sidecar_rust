//! Sc. 4 — the token is this repository's registered row. The real mounted rule must agree; fixture rule texts and
//! constants prove every refusal, including the identity that comes from outside the helper.

use crate::classes::TOKEN;
use crate::token::*;

const TABLE: &str = "# Rule\n\n## Tokens\n\nText.\n\n| token | family | repository |\n|---|---|---|\n| `conventions` | dew_flow | the shared-rules repository |\n| `mcp` | dew_flow | the MCP server |\n| `sidecar` | dew_flow | the embedding sidecar |\n\n<!-- core:begin -->\n## Never\n";

fn constant(name: &str, value: &str) -> Constant {
    Constant {
        file: "src/test_scratch.rs".to_string(),
        line: 10,
        name: name.to_string(),
        value: value.to_string(),
    }
}

fn good() -> Vec<Constant> {
    vec![
        constant("REPO_TOKEN", "sidecar"),
        constant("REPO_ROW", "the embedding sidecar"),
    ]
}

fn red(rule: Option<&str>, constants: &[Constant], package: Option<&str>) -> Vec<String> {
    let found = token_findings(rule, constants, package);
    assert!(found.iter().all(|f| f.class == TOKEN));
    found.into_iter().map(|f| f.text).collect()
}

#[test]
fn the_real_mounted_rule_registers_this_repositorys_token() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let rule = std::fs::read_to_string(root.join(RULE)).unwrap_or_else(|e| {
        panic!("the shared rules are not checked out — git submodule update --init .claude/rules/shared ({e})")
    });
    let cargo = std::fs::read_to_string(root.join("Cargo.toml")).expect("Cargo.toml");
    let package = package_name(&cargo);
    assert_eq!(package.as_deref(), Some("bge-sidecar"));
    // the helper's real constants, read the way the guard reads them
    let helper = std::fs::read_to_string(root.join("src/test_scratch.rs")).expect("the helper");
    let lines: Vec<&str> = helper.split('\n').collect();
    let constants = constants_in("src/test_scratch.rs", &lines);
    assert_eq!(constants.len(), 2, "{constants:?}");
    assert_eq!(
        red(Some(&rule), &constants, package.as_deref()),
        Vec::<String>::new()
    );
}

#[test]
fn the_table_the_constants_and_the_identity_agree() {
    assert!(red(Some(TABLE), &good(), Some("bge-sidecar")).is_empty());
}

#[test]
fn every_way_the_token_can_be_unregistered_is_red() {
    let pkg = Some("bge-sidecar");
    let no_section = "# Rule\n\nNo tokens here.\n";
    assert_eq!(red(Some(no_section), &good(), pkg).len(), 1);
    assert!(red(Some(no_section), &good(), pkg)[0].contains("no `## Tokens` section"));

    let two_columns = TABLE.replace("| token | family | repository |", "| token | family |");
    assert!(red(Some(&two_columns), &good(), pkg)
        .iter()
        .any(|t| t.contains("no table under")
            || t.contains("is not `| token | family | repository |`")));

    let renamed = TABLE.replace(
        "| token | family | repository |",
        "| token | family | description |",
    );
    assert!(red(Some(&renamed), &good(), pkg)
        .iter()
        .any(|t| t.contains("is not `| token | family | repository |`")));

    let only_in_repository_column = TABLE.replace(
        "| `sidecar` | dew_flow | the embedding sidecar |",
        "| `other` | dew_flow | the embedding sidecar |",
    );
    let found = red(Some(&only_in_repository_column), &good(), pkg);
    assert!(
        found.iter().any(|t| t.contains("is not a row")),
        "{found:?}"
    );

    let twice = TABLE.replace(
        "| `mcp` | dew_flow | the MCP server |",
        "| `sidecar` | dew_flow | the MCP server |",
    );
    let found = red(Some(&twice), &good(), pkg);
    assert!(
        found.iter().any(|t| t.contains("is listed twice")),
        "{found:?}"
    );
}

#[test]
fn a_token_of_the_wrong_shape_is_red_each() {
    let pkg = Some("bge-sidecar");
    for bad in ["Sidecar", "x", "u1000", "has space", "waytoolongtoken1"] {
        let constants = vec![
            constant("REPO_TOKEN", bad),
            constant("REPO_ROW", "the embedding sidecar"),
        ];
        let found = red(Some(TABLE), &constants, pkg);
        assert!(!found.is_empty(), "{bad} was accepted");
    }
}

#[test]
fn a_row_that_belongs_to_another_repository_is_red() {
    let pkg = Some("bge-sidecar");
    let other = vec![
        constant("REPO_TOKEN", "sidecar"),
        constant("REPO_ROW", "the MCP server"),
    ];
    let found = red(Some(TABLE), &other, pkg);
    assert!(
        found
            .iter()
            .any(|t| t.contains("belongs to `the embedding sidecar`")
                && t.contains("not to this repository")),
        "{found:?}"
    );

    let empty = vec![constant("REPO_TOKEN", "sidecar"), constant("REPO_ROW", "")];
    let found = red(Some(TABLE), &empty, pkg);
    assert!(
        found.iter().any(|t| t.contains("is empty")),
        "an empty row fails closed: {found:?}"
    );
}

#[test]
fn the_constants_are_declared_exactly_once_each() {
    let pkg = Some("bge-sidecar");
    let mut doubled = good();
    doubled.push(constant("REPO_TOKEN", "sidecar"));
    let found = red(Some(TABLE), &doubled, pkg);
    assert!(
        found.iter().any(|t| t.contains("declared 2 times")),
        "{found:?}"
    );

    let missing_row = vec![constant("REPO_TOKEN", "sidecar")];
    assert!(red(Some(TABLE), &missing_row, pkg)
        .iter()
        .any(|t| t.contains("no `const REPO_ROW")));
    assert!(red(Some(TABLE), &[], pkg).len() >= 2);
}

#[test]
fn a_helper_copied_verbatim_into_another_repository_is_red() {
    // the copy brings its token AND its row; the identity from outside the copied files is the package name
    let copy = good();
    let found = red(Some(TABLE), &copy, Some("another-crate"));
    assert!(
        found
            .iter()
            .any(|t| t.contains("not a repository this guard knows")),
        "{found:?}"
    );
    let found = red(Some(TABLE), &copy, None);
    assert!(
        found
            .iter()
            .any(|t| t.contains("not a repository this guard knows")),
        "{found:?}"
    );

    // in a known repository, a token or row from another one is red even when the table lists it
    let swapped = vec![
        constant("REPO_TOKEN", "mcp"),
        constant("REPO_ROW", "the MCP server"),
    ];
    let found = red(Some(TABLE), &swapped, Some("bge-sidecar"));
    assert!(
        found.iter().any(|t| t.contains("is not this repository's")),
        "{found:?}"
    );
    // each identity check alone: the table agrees with the helper, only the crate's identity does not
    let token_only = vec![
        constant("REPO_TOKEN", "mcp"),
        constant("REPO_ROW", "the embedding sidecar"),
    ];
    let found = red(Some(TABLE), &token_only, Some("bge-sidecar"));
    assert!(
        found
            .iter()
            .any(|t| t.contains("helper's token `mcp` is not this repository's")),
        "{found:?}"
    );
    let row_only = vec![
        constant("REPO_TOKEN", "sidecar"),
        constant("REPO_ROW", "the MCP server"),
    ];
    let found = red(Some(TABLE), &row_only, Some("bge-sidecar"));
    assert!(
        found
            .iter()
            .any(|t| t.contains("helper's `REPO_ROW` `the MCP server` is not this repository's")),
        "{found:?}"
    );
}

#[test]
fn a_mount_that_is_not_checked_out_is_red_with_the_submodule_sentence() {
    let found = red(None, &good(), Some("bge-sidecar"));
    assert_eq!(found.len(), 1, "{found:?}");
    assert!(found[0].contains(
        "the shared rules are not checked out — git submodule update --init .claude/rules/shared"
    ));
}

#[test]
fn the_package_name_is_read_from_the_package_table_only() {
    assert_eq!(
        package_name("[package]\nname = \"bge-sidecar\"\nversion = \"1\"\n").as_deref(),
        Some("bge-sidecar")
    );
    assert_eq!(
        package_name("[dependencies]\nname = \"x\"\n[package]\nname = \"y\"\n").as_deref(),
        Some("y")
    );
    assert_eq!(package_name("[dependencies]\nname = \"x\"\n"), None);
}

#[test]
fn a_constant_in_a_comment_is_not_a_declaration() {
    let lines = [
        "// const REPO_TOKEN: &str = \"x\";",
        "pub(crate) const REPO_TOKEN: &str = \"sidecar\";",
        "#[allow(dead_code)]",
        "pub(crate) const REPO_ROW: &str = \"the embedding sidecar\";",
    ];
    let found = constants_in("src/test_scratch.rs", &lines);
    assert_eq!(found.len(), 2, "{found:?}");
    assert_eq!(
        (
            found[0].line,
            found[0].name.as_str(),
            found[0].value.as_str()
        ),
        (2, "REPO_TOKEN", "sidecar")
    );
}
