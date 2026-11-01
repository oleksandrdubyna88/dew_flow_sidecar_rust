//! The token is registered AND this repository's (rule §1; `tokenFindings` of the shipped `leftovers-check.mjs`).
//!
//! The helper writes two constants, `REPO_TOKEN` and `REPO_ROW`, once, in `src/`. The guard reads them as TEXT and
//! holds them against the `## Tokens` table of the mounted rule — and against an identity that does NOT come from the
//! helper: a helper copied verbatim into another repository carries its token AND its row with it, so a check that
//! trusted the row from the same file would pass there. The crate's package name in `Cargo.toml` is outside the
//! copied files, and `KNOWN` maps it to the one token and row it may use.

use std::path::Path;

use regex::Regex;

use crate::classes::{finding, Finding, TOKEN};

/// The repositories this guard knows: `[package] name` of the `Cargo.toml` it runs in, the token the helper there
/// must hold and the `repository` column of that token's row. A package not listed is red.
pub const KNOWN: [(&str, &str, &str); 1] = [("bge-sidecar", "sidecar", "the embedding sidecar")];

/// The mounted rule file that holds the `## Tokens` table since the split of the rule into a core and a detail
/// (release `105d53c`): the detail. It is the first one tried and the name every message carries when nothing is read.
pub const RULE_DETAIL: &str = ".claude/rules/shared/common/test-leftovers-detail.md";

/// The core of the rule: it held the whole text, table included, before the split, so a mount from before it is read
/// here, after the detail.
pub const RULE_CORE: &str = ".claude/rules/shared/common/test-leftovers.md";

/// One rule file as the mount holds it: absent, present but not readable (permission, I/O, not UTF-8), or its text.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RuleFile {
    Missing,
    Unreadable(String),
    Text(String),
}

impl RuleFile {
    /// Read `root/relative`: a file that is not there is `Missing`; any other failure is `Unreadable` with its message.
    fn read(root: &Path, relative: &str) -> Self {
        match std::fs::read_to_string(root.join(relative)) {
            Ok(text) => Self::Text(text),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => Self::Missing,
            Err(error) => Self::Unreadable(error.to_string()),
        }
    }
}

/// What the two files of the mount come to, decided once when `RuleFiles` is built.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Outcome {
    /// The file that holds the `## Tokens` section, and its text.
    Found { path: &'static str, text: String },
    /// Neither file exists: the submodule is not checked out.
    NotCheckedOut,
    /// A candidate that could hold the table exists and cannot be read: fail closed, name the file and the error.
    Unreadable { path: &'static str, error: String },
    /// Both files that exist were read and neither has a `## Tokens` section.
    NoSection,
}

/// The two rule files of the mount and the file the table is read from, chosen ONCE, by CONTENT: the detail when it has
/// a `## Tokens` section, else the core when it has one (a mount from before the split). The choice is stored; the
/// scan and every message read it, none resolves again. A candidate that is unreadable BEFORE a file with the section
/// is found stops the choice (it could be the file that holds the table, and it would win): ONE finding, never a panic.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RuleFiles {
    outcome: Outcome,
}

impl RuleFiles {
    /// Decide the outcome for the two files (the detail is tried first).
    pub fn new(detail: RuleFile, core: RuleFile) -> Self {
        let mut any = false;
        for (path, file) in [(RULE_DETAIL, detail), (RULE_CORE, core)] {
            match file {
                RuleFile::Missing => {}
                RuleFile::Unreadable(error) => {
                    return Self::of(Outcome::Unreadable { path, error })
                }
                RuleFile::Text(text) => {
                    any = true;
                    if has_tokens_section(&text) {
                        return Self::of(Outcome::Found { path, text });
                    }
                }
            }
        }
        Self::of(if any {
            Outcome::NoSection
        } else {
            Outcome::NotCheckedOut
        })
    }

    fn of(outcome: Outcome) -> Self {
        Self { outcome }
    }

    /// Both files from texts (`None` is a file that is not there).
    pub fn from_texts(detail: Option<&str>, core: Option<&str>) -> Self {
        let file =
            |text: Option<&str>| text.map_or(RuleFile::Missing, |t| RuleFile::Text(t.to_string()));
        Self::new(file(detail), file(core))
    }

    /// Read both files of the mount under `root`.
    pub fn read(root: &Path) -> Self {
        Self::new(
            RuleFile::read(root, RULE_DETAIL),
            RuleFile::read(root, RULE_CORE),
        )
    }

    pub fn outcome(&self) -> &Outcome {
        &self.outcome
    }

    /// The stored choice: the file that holds the table and its text, or None.
    pub fn resolve(&self) -> Option<(&'static str, &str)> {
        match &self.outcome {
            Outcome::Found { path, text } => Some((path, text)),
            _ => None,
        }
    }

    /// The file a message names (`common/<file>`): the one the table is read from, else the detail (where it is meant
    /// to be).
    pub fn rule_name(&self) -> &'static str {
        match &self.outcome {
            Outcome::Found { path, .. } if *path == RULE_CORE => "common/test-leftovers.md",
            _ => "common/test-leftovers-detail.md",
        }
    }
}

/// `## Tokens` on a line of its own — the heading `parse_table` reads, so a file the resolver picks is a file the
/// parser finds a table in (prose that spells the words is not a heading).
fn is_tokens_heading(line: &str) -> bool {
    line.trim_end() == "## Tokens"
}

fn has_tokens_section(text: &str) -> bool {
    text.lines().any(is_tokens_heading)
}

/// A token constant read from a source file: `REPO_TOKEN` or `REPO_ROW`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Constant {
    pub file: String,
    pub line: usize,
    pub name: String,
    pub value: String,
}

/// Every `const REPO_TOKEN: &str = "…";` and `const REPO_ROW: &str = "…";` of a Rust file (comment lines hold none).
pub fn constants_in(file: &str, lines: &[&str]) -> Vec<Constant> {
    let pattern = Regex::new(
        r#"\bconst\s+(REPO_TOKEN|REPO_ROW)\s*:\s*&(?:'static\s+)?str\s*=\s*"([^"]*)"\s*;"#,
    )
    .expect("the constant pattern");
    lines
        .iter()
        .enumerate()
        .filter(|(_, line)| !line.trim_start().starts_with("//"))
        .flat_map(|(index, line)| {
            pattern
                .captures_iter(line)
                .map(|caught| Constant {
                    file: file.to_string(),
                    line: index + 1,
                    name: caught[1].to_string(),
                    value: caught[2].to_string(),
                })
                .collect::<Vec<_>>()
        })
        .collect()
}

/// `[package] name = "…"` of a `Cargo.toml`, or None.
pub fn package_name(cargo_toml: &str) -> Option<String> {
    let mut in_package = false;
    for line in cargo_toml.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with('[') {
            in_package = trimmed == "[package]";
        } else if in_package {
            if let Some(rest) = trimmed.strip_prefix("name") {
                let value = rest.trim_start().strip_prefix('=')?.trim();
                return Some(value.trim_matches('"').to_string());
            }
        }
    }
    None
}

struct Table {
    heading: usize,
    header: Option<Vec<String>>,
    rows: Vec<(String, String, usize)>,
}

fn cells(row: &str) -> Vec<String> {
    row.trim()
        .trim_start_matches('|')
        .trim_end_matches('|')
        .split('|')
        .map(|cell| cell.trim().to_string())
        .collect()
}

fn is_separator(line: &str) -> bool {
    let trimmed = line.trim();
    trimmed.contains('-')
        && trimmed.contains('|')
        && trimmed.chars().all(|c| matches!(c, ' ' | ':' | '|' | '-'))
}

/// The rule's `## Tokens` table: the heading's line, the header cells, the rows `(token, repository, line)`.
fn parse_table(rule: &str) -> Table {
    let lines: Vec<&str> = rule.lines().collect();
    let Some(start) = lines.iter().position(|l| is_tokens_heading(l)) else {
        return Table {
            heading: 0,
            header: None,
            rows: Vec::new(),
        };
    };
    let end = lines[start + 1..]
        .iter()
        .position(|l| l.starts_with("# ") || l.starts_with("## "))
        .map_or(lines.len(), |at| start + 1 + at);
    let section = &lines[start + 1..end];
    let header_at = (0..section.len().saturating_sub(1))
        .find(|&i| section[i].trim_start().starts_with('|') && is_separator(section[i + 1]));
    let Some(header_at) = header_at else {
        return Table {
            heading: start + 1,
            header: None,
            rows: Vec::new(),
        };
    };
    let mut rows = Vec::new();
    for (offset, row) in section[header_at + 2..].iter().enumerate() {
        if !row.trim_start().starts_with('|') {
            break;
        }
        let parts = cells(row);
        let token = parts.first().map_or(String::new(), |c| c.replace('`', ""));
        let repository = parts.get(2).cloned().unwrap_or_default();
        rows.push((token, repository, start + header_at + offset + 4));
    }
    Table {
        heading: start + 1,
        header: Some(cells(section[header_at])),
        rows,
    }
}

/// Findings (class `token`) for the constants, the mounted table and the crate's identity. `rules` is the two files
/// of the mount: the table is read from the one `RuleFiles::resolve` picks; no file at all is red with the submodule
/// sentence, files without a table are ONE finding naming both (no table registers nothing).
pub fn token_findings(
    rules: &RuleFiles,
    constants: &[Constant],
    package: Option<&str>,
) -> Vec<Finding> {
    let at = |file: &str, line: usize, text: String| vec![finding(file, line.max(1), TOKEN, text)];
    let mut found = Vec::new();
    let tokens: Vec<&Constant> = constants
        .iter()
        .filter(|c| c.name == "REPO_TOKEN")
        .collect();
    let rows: Vec<&Constant> = constants.iter().filter(|c| c.name == "REPO_ROW").collect();
    for (name, list) in [("REPO_TOKEN", &tokens), ("REPO_ROW", &rows)] {
        match list.len() {
            1 => {}
            0 => found.extend(at(
                "src",
                1,
                format!(
                    "no `const {name}: &str = \"…\";` in src/**/*.rs — the helper writes it once"
                ),
            )),
            _ => {
                let where_ = list
                    .iter()
                    .map(|c| format!("{}:{}", c.file, c.line))
                    .collect::<Vec<_>>()
                    .join(", ");
                found.extend(at(
                    &list[0].file,
                    list[0].line,
                    format!(
                        "`{name}` is declared {} times ({where_}) — the helper writes it once",
                        list.len()
                    ),
                ));
            }
        }
    }
    let (Some(token), Some(row)) = (tokens.first(), rows.first()) else {
        return found;
    };
    if tokens.len() != 1 || rows.len() != 1 {
        return found;
    }
    let shape = Regex::new("^[a-z0-9]{2,12}$").expect("the token shape");
    let owner_grammar = Regex::new("^[uw][0-9]{1,10}$").expect("the owner grammar");
    if !shape.is_match(&token.value) {
        found.extend(at(
            &token.file,
            token.line,
            format!("the token `{}` is not [a-z0-9]{{2,12}}", token.value),
        ));
    } else if owner_grammar.is_match(&token.value) {
        found.extend(at(&token.file, token.line, format!("the token `{}` matches the owner grammar — no repository may be named like an owner", token.value)));
    }
    if row.value.trim().is_empty() {
        found.extend(at(&row.file, row.line, "`REPO_ROW` is empty, so the token cannot be proven to belong to this repository (fails closed)".to_string()));
    }

    // The identity from OUTSIDE the helper: the crate's package name.
    match package.and_then(|name| KNOWN.iter().find(|(known, _, _)| *known == name)) {
        None => found.extend(at(
            "Cargo.toml",
            1,
            format!(
                "the package `{}` is not a repository this guard knows — a copied helper brings its token and its row with it; register this repository's own row in the rule and in KNOWN",
                package.unwrap_or("(none)")
            ),
        )),
        Some((_, expected_token, expected_row)) => {
            if token.value != *expected_token {
                found.extend(at(&token.file, token.line, format!("the helper's token `{}` is not this repository's (`{expected_token}`)", token.value)));
            }
            if !row.value.trim().eq_ignore_ascii_case(expected_row) {
                found.extend(at(&row.file, row.line, format!("the helper's `REPO_ROW` `{}` is not this repository's (`{expected_row}`)", row.value)));
            }
        }
    }

    let (rule_path, rule) = match rules.outcome() {
        Outcome::Found { path, text } => (*path, text.as_str()),
        Outcome::NotCheckedOut => {
            found.extend(at(
                RULE_DETAIL,
                1,
                format!("the shared rules are not checked out — git submodule update --init .claude/rules/shared (neither {RULE_DETAIL} nor {RULE_CORE} can be read)"),
            ));
            return found;
        }
        Outcome::Unreadable { path, error } => {
            found.extend(at(
                path,
                1,
                format!("{path} exists and cannot be read ({error}) — the token `{}` cannot be proven registered (fails closed)", token.value),
            ));
            return found;
        }
        Outcome::NoSection => {
            found.extend(at(
                RULE_DETAIL,
                1,
                format!(
                    "no `## Tokens` section in {RULE_DETAIL} or in {RULE_CORE} — the token `{}` is not registered",
                    token.value
                ),
            ));
            return found;
        }
    };
    let table = parse_table(rule);
    let Some(header) = &table.header else {
        found.extend(at(
            rule_path,
            table.heading,
            format!(
                "{rule_path} has no table under `## Tokens` — the token `{}` is not registered",
                token.value
            ),
        ));
        return found;
    };
    let expected_header = ["token", "family", "repository"];
    if header.len() != 3
        || !expected_header
            .iter()
            .zip(header)
            .all(|(want, got)| got.eq_ignore_ascii_case(want))
    {
        found.extend(at(rule_path, table.heading, format!("the `## Tokens` table of {rule_path} is not `| token | family | repository |` — the token `{}` is not registered", token.value)));
        return found;
    }
    let mut seen: Vec<&str> = Vec::new();
    for (name, _, line) in &table.rows {
        if seen.contains(&name.as_str()) {
            found.extend(at(
                rule_path,
                *line,
                format!("the token `{name}` is listed twice in {rule_path}"),
            ));
        }
        seen.push(name);
    }
    match table.rows.iter().find(|(name, _, _)| *name == token.value) {
        None => found.extend(at(
            rule_path,
            table.heading,
            format!(
                "the token `{}` is not a row of the `## Tokens` table of {rule_path}",
                token.value
            ),
        )),
        Some((_, repository, line)) => {
            if !row.value.trim().is_empty()
                && !repository.trim().eq_ignore_ascii_case(row.value.trim())
            {
                found.extend(at(rule_path, *line, format!("the token `{}` belongs to `{repository}` in {rule_path}, not to this repository (`{}`)", token.value, row.value)));
            }
        }
    }
    found
}
