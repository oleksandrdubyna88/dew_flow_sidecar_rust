//! The classes of `common/test-leftovers.md` §7, as pure functions over one file's `(relative path, text)`.
//!
//! The rule's eight — `raw-temp`, `raw-worktree`, `docker-run`, `docker-name`, `legacy-name`, `owner-blind`,
//! `shared-store`, `ryuk-off` — plus the two about the guard's own grammar (`exemption`, and `token` in
//! `token.rs`) and one of this repository's own, `dropped-scratch`: a `TestScratch` that is not held by a named
//! binding is dropped at the end of its STATEMENT, and the folder is deleted before the test uses it.
//!
//! `owner-blind` is two judgements. In Rust source it takes no binding tracer (owner decision 3 of the E4 plan): ANY
//! spelling of the prefix word outside the helper's files that is not followed by an owner segment is a finding,
//! because a Rust constant is exactly the shape a tracer would miss. In scripts and workflows it is the shipped
//! classifier's verdict (`scripts.rs`), which this file does not reimplement.
//!
//! Comments: a whole-line `//` is skipped in Rust and JavaScript, a whole-line `#` in workflows and NEVER in Rust
//! (an attribute starts with `#`). Exemptions live in comments, so every line is read for them.

use regex::Regex;
use std::collections::{HashMap, HashSet};
use std::sync::OnceLock;

use crate::words::{MARK, WORD};

pub const RAW_TEMP: &str = "raw-temp";
pub const RAW_WORKTREE: &str = "raw-worktree";
pub const DOCKER_RUN: &str = "docker-run";
pub const DOCKER_NAME: &str = "docker-name";
pub const LEGACY_NAME: &str = "legacy-name";
pub const OWNER_BLIND: &str = "owner-blind";
pub const SHARED_STORE: &str = "shared-store";
pub const RYUK_OFF: &str = "ryuk-off";
pub const DROPPED_SCRATCH: &str = "dropped-scratch";
pub const EXEMPTION: &str = "exemption";
pub const TOKEN: &str = "token";
pub const DISCOVERY: &str = "discovery";

/// The classes an exemption may name. `ryuk-off` and `owner-blind` accept none (rule §7), and neither does the
/// repository's own `dropped-scratch`: a scratch that is dropped at once is a bug, never a choice.
pub const EXEMPTIBLE: [&str; 6] = [
    RAW_TEMP,
    RAW_WORKTREE,
    DOCKER_RUN,
    DOCKER_NAME,
    LEGACY_NAME,
    SHARED_STORE,
];

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Finding {
    pub file: String,
    pub line: usize,
    pub class: &'static str,
    pub text: String,
}

pub fn finding(file: &str, line: usize, class: &'static str, text: impl Into<String>) -> Finding {
    Finding {
        file: file.to_string(),
        line,
        class,
        text: text.into(),
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Lang {
    Rust,
    Js,
    Workflow,
}

const JS_EXTENSIONS: [&str; 8] = ["mjs", "js", "cjs", "ts", "mts", "cts", "tsx", "jsx"];

/// A workflow is a `*.yml`/`*.yaml` directly under `.github/workflows/`; any other YAML is not read.
pub fn is_workflow(file: &str) -> bool {
    file.strip_prefix(".github/workflows/").is_some_and(|rest| {
        !rest.contains('/') && (rest.ends_with(".yml") || rest.ends_with(".yaml"))
    })
}

/// The language the guard reads a file as, from its path alone.
pub fn language_of(file: &str) -> Option<Lang> {
    if is_workflow(file) {
        return Some(Lang::Workflow);
    }
    let extension = file.rsplit_once('.')?.1.to_ascii_lowercase();
    if extension == "rs" {
        Some(Lang::Rust)
    } else if JS_EXTENSIONS.contains(&extension.as_str()) {
        Some(Lang::Js)
    } else {
        None
    }
}

/// The helper's own files: where the prefix may be spelled, the temp root is made and used. Everything else is
/// judged without these allowances.
pub fn is_helper_file(file: &str) -> bool {
    file == "src/test_scratch.rs" || file.starts_with("src/test_scratch/")
}

/// The lines of a text, `\r` dropped, a trailing empty piece kept — the numbering the shipped classifier uses.
pub fn split_lines(text: &str) -> Vec<&str> {
    text.split('\n')
        .map(|line| line.strip_suffix('\r').unwrap_or(line))
        .collect()
}

fn is_comment(lang: Lang, line: &str) -> bool {
    let trimmed = line.trim_start();
    match lang {
        Lang::Rust | Lang::Js => trimmed.starts_with("//"),
        Lang::Workflow => trimmed.starts_with('#'),
    }
}

fn re(parts: &[&str]) -> Regex {
    Regex::new(&parts.concat()).expect("a detector pattern")
}

struct Detectors {
    raw_temp: Vec<Regex>,
    raw_temp_workflow: Regex,
    raw_temp_quoted_tool: Regex,
    accessor: Regex,
    raw_worktree: Vec<Regex>,
    docker_run: Vec<Regex>,
    shared_store: Vec<Regex>,
    ryuk_off: Regex,
    word: Regex,
    legacy_literal: Regex,
    legacy_built: Vec<Regex>,
    owner_after: Regex,
    not_a_name_after: Regex,
    env_name_after: Regex,
    owner_form: Regex,
    prefix_const: Regex,
    marker: Regex,
    marker_rest: Regex,
    placeholder: Regex,
    dropped: Vec<Regex>,
    seed_destructure: Regex,
    docker_name: DockerName,
}

struct DockerName {
    name_flag: Regex,
    create: Regex,
    with_name: Regex,
    compose_env: Regex,
    compose_flag: Regex,
    compose_cmd: Regex,
    wholly_expansion: Regex,
    literal_default: Regex,
}

fn detectors() -> &'static Detectors {
    static DETECTORS: OnceLock<Detectors> = OnceLock::new();
    DETECTORS.get_or_init(|| {
        let owner_form = re(&["(?i-u)^", WORD, r"[-_][uw][0-9]{1,10}[-_]"]);
        Detectors {
            raw_temp: vec![
                re(&[r"\btemp", r"_dir\s*\("]),
                re(&[r"\benv::temp", r"_dir\b"]),
                re(&[r#"\bvar(_os)?\(\s*"(TE"#, r#"MP|TMP|TMPDIR)""#]),
                re(&[r"\btemp", r"file::"]),
                re(&[r"\btemp", r"dir\s*\("]),
                re(&[r"\bmkd", r"temp(Sync)?\s*\("]),
                re(&[r"\btmp", r"dir\s*\("]),
                re(&[r"process\.env\.(TE", r"MP|TMP|TMPDIR)\b"]),
                re(&["Get", "TempPath"]),
                re(&["CreateTemp", "Subdirectory"]),
                re(&["GetTemp", "FileName"]),
            ],
            raw_temp_workflow: re(&[r"\bmk", r"temp\b"]),
            raw_temp_quoted_tool: re(&[r#"["']mk"#, r"temp\b"]),
            accessor: re(&[r"\btemp", r"_root\b"]),
            raw_worktree: vec![
                re(&[r"\bwork", r"tree\s+add\b"]),
                re(&[r#""work"#, r#"tree",\s*"add""#]),
            ],
            docker_run: vec![
                re(&[r"\bdock", r"er\s+(run|create|compose)\b"]),
                re(&[r"\bdock", r"er-compose\b"]),
                re(&[r#"["'`]dock"#, r#"er["'`]"#]),
            ],
            shared_store: vec![
                re(&[
                    r#"\b(DROP\s+(DATABASE|SCHEMA|TABLE)\s+(?:IF\s+EXISTS\s+)?(["'`]?[\w.$-]*)|TRUN"#,
                    r#"CATE\b(?:\s+TABLE)?(?:\s+(["'`]?[\w.$-]*))?|FLUSH(ALL|DB)\b)"#,
                ]),
                re(&[r"deleteMany\(\s*\{\s*\}\s*\)"]),
            ],
            ryuk_off: re(&["(?i)TESTCONTAINERS_RYUK_", r#"DISABLED\s*[:=]\s*["']?true"#]),
            word: re(&["(?i-u)", WORD]),
            legacy_literal: re(&["(?i-u)", WORD, "[-_]([a-z0-9]+)"]),
            legacy_built: vec![
                re(&["(?i-u)", WORD, r"[-_]\$?\{?\s*(REPO_TOKEN|RepoToken|repoToken)\b"]),
                re(&[r"\{PRE", r"FIX\}\{?REPO_TOKEN\b"]),
                re(&[r"\bPRE", r"FIX\s*\+\s*REPO_TOKEN\b"]),
            ],
            owner_after: re(&["(?i-u)^[-_][uw][0-9]{1,10}[-_]"]),
            not_a_name_after: re(&["(?i-u)^[-_](exempt|quarantine)"]),
            env_name_after: re(&["^_[A-Z][A-Z_]*"]),
            owner_form,
            prefix_const: re(&[r"\bPRE", r"FIX\b"]),
            marker: re(&[r"(?://|#|--|/\*|\*|<!--)\s*", MARK, "(.*)$"]),
            marker_rest: re(&[r"^\s*([^\s—]*)\s*(?:—\s*(.*?))?\s*(?:\*/|-->)?\s*$"]),
            placeholder: re(&[
                r"(?i)^(see above|see below|needed|required|necessary|obvious|todo|tbd|n/?a|because|why|reason|it is needed)\b",
            ]),
            dropped: vec![
                re(&[
                    r"let\s+_\s*=\s*.*(Test",
                    r"Scratch::new(_in|_with)?|seed_scr",
                    r"atch|model_cache_with_a_tokenizer)\s*\(",
                ]),
                re(&[r"let\s*\(\s*_\s*,[^=]*=\s*.*seed_scr", r"atch\s*\("]),
                re(&[
                    r"^\s*(Test",
                    r"Scratch::new(_in|_with)?|seed_scr",
                    r"atch|model_cache_with_a_tokenizer)\s*\(.*\)\s*;\s*$",
                ]),
                re(&[
                    r"(Test",
                    r"Scratch::new(_in|_with)?|seed_scr",
                    r"atch|model_cache_with_a_tokenizer)\s*\([^;]*\)\s*\.\s*(path|slash_path|from|to|scratch)\b",
                ]),
            ],
            seed_destructure: re(&[r"let\s+Seed", r"Scratch\s*\{([^}]*)\}"]),
            docker_name: DockerName {
                name_flag: re(&[r#"["']?--na"#, r#"me["']?(?:\s*=\s*|\s*,\s*|\s+)["']?([^\s"',]*)"#]),
                create: re(&[r"\b(volume|network)\s+cre", r"ate\b(.*)$"]),
                with_name: re(&[r"\bWith", r#"Name\(\s*"([^"]*)""#]),
                compose_env: re(&[
                    r#"COMPOSE_PROJECT_NA"#,
                    r#"ME["']?\s*[:=]\s*["']?([^\s"';,}]*)"#,
                ]),
                compose_flag: re(&[r#"(?:^|\s)(?:-p|--project-na"#, r#"me)(?:=|\s+)["']?([^\s"']+)"#]),
                compose_cmd: re(&[r"\bdock", r"er(\s+|-)compose\b"]),
                wholly_expansion: re(&[r"^\$(?:[A-Za-z_]\w*|\{[^{}]*\}|\([^()]*\))$"]),
                literal_default: re(&[r"^\$\{[A-Za-z_]\w*:?[-=]([^${}]+)\}$"]),
            },
        }
    })
}

// ---------- exemptions ----------

pub struct Exemptions {
    pub valid: HashMap<&'static str, HashSet<usize>>,
    pub findings: Vec<Finding>,
}

fn is_reason(reason: &str) -> bool {
    reason.chars().count() >= 10
        && reason.split_whitespace().count() >= 3
        && !detectors().placeholder.is_match(reason)
}

/// `Ok(class)` when the text after the marker is a valid exemption, else `Err(why not)`.
fn exemption_problem(rest: &str) -> Result<&'static str, String> {
    let d = detectors();
    let parsed = d.marker_rest.captures(rest);
    let class = parsed
        .as_ref()
        .and_then(|c| c.get(1))
        .map_or("", |m| m.as_str());
    let reason = parsed
        .as_ref()
        .and_then(|c| c.get(2))
        .map_or("", |m| m.as_str());
    if class.is_empty() {
        return Err("an exemption names no class".to_string());
    }
    if class == RYUK_OFF {
        return Err(format!("`{RYUK_OFF}` accepts no exemption"));
    }
    if class == OWNER_BLIND {
        return Err(format!("`{OWNER_BLIND}` accepts no exemption"));
    }
    let Some(known) = EXEMPTIBLE.iter().find(|name| **name == class) else {
        return Err(format!("an exemption names an unknown class `{class}`"));
    };
    if !is_reason(reason) {
        return Err(format!(
            "the exemption of `{class}` carries no reason (ten characters, three words, not a placeholder)"
        ));
    }
    if class == SHARED_STORE && !reason.contains("lock:") {
        return Err(format!(
            "the exemption of `{SHARED_STORE}` must name its lock (`lock: …`)"
        ));
    }
    Ok(known)
}

pub fn read_exemptions(file: &str, lines: &[&str]) -> Exemptions {
    let mut valid: HashMap<&'static str, HashSet<usize>> = EXEMPTIBLE
        .iter()
        .map(|class| (*class, HashSet::new()))
        .collect();
    let mut findings = Vec::new();
    for (index, line) in lines.iter().enumerate() {
        let Some(marker) = detectors().marker.captures(line) else {
            continue;
        };
        match exemption_problem(&marker[1]) {
            Ok(class) => {
                valid
                    .get_mut(class)
                    .expect("an exemptible class")
                    .insert(index + 1);
            }
            Err(problem) => findings.push(finding(file, index + 1, EXEMPTION, problem)),
        }
    }
    Exemptions { valid, findings }
}

fn is_exempt(exemptions: &Exemptions, class: &str, line: usize) -> bool {
    exemptions
        .valid
        .get(class)
        .is_some_and(|set| set.contains(&line) || (line > 1 && set.contains(&(line - 1))))
}

// ---------- the single-line classes ----------

/// A name that is wholly a shell expansion is not judged (a text scan cannot know what it builds) — except a
/// literal default (`${V:-w}`), which is the name the command gets when V is unset, so `w` itself is judged.
fn docker_name_ok(value: &str) -> bool {
    let d = detectors();
    let bare = value.trim_matches(|c| c == '"' || c == '\'');
    if d.owner_form.is_match(bare) {
        return true;
    }
    if let Some(default) = d.docker_name.literal_default.captures(bare) {
        return d.owner_form.is_match(&default[1]);
    }
    d.docker_name.wholly_expansion.is_match(bare)
}

fn docker_name_hits(line: &str) -> Vec<String> {
    let d = &detectors().docker_name;
    let mut hits = Vec::new();
    for caught in d.name_flag.captures_iter(line) {
        if !docker_name_ok(&caught[1]) {
            hits.push(format!(
                "the name `{}` is not an owner-form leftover name",
                &caught[1]
            ));
        }
    }
    if let Some(caught) = d.create.captures(line) {
        let kind = caught[1].to_string();
        let mut words = caught[2]
            .split_whitespace()
            .map(|w| w.trim_matches(|c| c == '"' || c == '\''));
        let mut name = "";
        while let Some(word) = words.next() {
            if !word.starts_with('-') {
                name = word;
                break;
            }
            if matches!(
                word,
                "-d" | "--driver" | "--label" | "-l" | "-o" | "--opt" | "--subnet"
            ) {
                words.next();
            }
        }
        if name.is_empty() {
            hits.push(format!("`{kind} create` makes an unnamed {kind}"));
        } else if !docker_name_ok(name) {
            hits.push(format!(
                "`{kind} create {name}` is not an owner-form leftover name"
            ));
        }
    }
    for caught in d.with_name.captures_iter(line) {
        if !detectors().owner_form.is_match(&caught[1]) {
            hits.push(format!(
                "a name `{}` is not an owner-form leftover name",
                &caught[1]
            ));
        }
    }
    for caught in d.compose_env.captures_iter(line) {
        if !docker_name_ok(&caught[1]) {
            hits.push(format!(
                "the compose project `{}` is not an owner-form leftover name",
                &caught[1]
            ));
        }
    }
    if d.compose_cmd.is_match(line) {
        for caught in d.compose_flag.captures_iter(line) {
            if !docker_name_ok(&caught[1]) {
                hits.push(format!(
                    "the compose project `{}` is not an owner-form leftover name",
                    &caught[1]
                ));
            }
        }
    }
    hits
}

fn legacy_name_hits(line: &str) -> Vec<String> {
    let d = detectors();
    let mut hits = Vec::new();
    for caught in d.legacy_literal.captures_iter(line) {
        let whole = caught.get(0).expect("the whole match").as_str();
        let second = caught[1].to_ascii_lowercase();
        let env_name = whole.len() > WORD.len()
            && whole == whole.to_ascii_uppercase()
            && whole[WORD.len() + 1..]
                .chars()
                .all(|c| c.is_ascii_uppercase());
        let owner = second.len() >= 2
            && second.len() <= 11
            && matches!(second.as_bytes()[0], b'u' | b'w')
            && second[1..].bytes().all(|b| b.is_ascii_digit());
        if second == "exempt" || second == "quarantine" || env_name || owner {
            continue;
        }
        hits.push(whole.to_string());
    }
    if d.legacy_built.iter().any(|pattern| pattern.is_match(line)) {
        hits.push("a prefix built from the repository token".to_string());
    }
    hits
}

/// Every spelling of the prefix word in Rust source outside the helper that is not followed by an owner segment —
/// and the helper's prefix constant, which would carry the same prefix into a sweep without spelling it.
fn rust_owner_blind_hits(line: &str) -> Vec<String> {
    let d = detectors();
    let mut hits = Vec::new();
    for found in d.word.find_iter(line) {
        let rest = &line[found.end()..];
        let shouted = found.as_str() == found.as_str().to_ascii_uppercase();
        if d.owner_after.is_match(rest)
            || d.not_a_name_after.is_match(rest)
            || rest.starts_with(".owner")
            || (shouted && d.env_name_after.is_match(rest))
        {
            continue;
        }
        let tail: String = rest.chars().take(6).collect();
        hits.push(format!("`{}{tail}`", found.as_str()));
    }
    if d.prefix_const.is_match(line) {
        hits.push("the helper's prefix constant".to_string());
    }
    hits
}

fn shared_store_hit(line: &str) -> Vec<String> {
    let d = detectors();
    let Some(caught) = d.shared_store[0].captures(line) else {
        return d.shared_store[1]
            .find(line)
            .map(|m| vec![m.as_str().to_string()])
            .unwrap_or_default();
    };
    let named = caught.get(3).or_else(|| caught.get(4)).map(|m| m.as_str());
    let bare = named.map(|name| name.trim_start_matches(['"', '\'', '`']));
    if bare.is_some_and(|name| d.owner_form.is_match(name)) {
        return Vec::new();
    }
    vec![caught[1].trim().to_string()]
}

fn dropped_scratch_hit(line: &str) -> Option<String> {
    let d = detectors();
    if let Some(pattern) = d.dropped.iter().find(|pattern| pattern.is_match(line)) {
        return pattern.find(line).map(|m| m.as_str().trim().to_string());
    }
    let caught = d.seed_destructure.captures(line)?;
    let body = &caught[1];
    let binds_scratch = body.split(',').any(|field| {
        let field = field.trim();
        field == "scratch"
            || (field.starts_with("scratch") && !field.replace(' ', "").ends_with(":_"))
    });
    (!binds_scratch).then(|| "a destructuring that leaves the scratch unbound".to_string())
}

/// Every finding of one file's text except a script's `owner-blind` (that one is the shipped classifier's,
/// `scripts.rs`). `file` is the repository-relative path with forward slashes.
pub fn scan_text(file: &str, text: &str) -> Vec<Finding> {
    let Some(lang) = language_of(file) else {
        return Vec::new();
    };
    let lines = split_lines(text);
    let exemptions = read_exemptions(file, &lines);
    let helper = is_helper_file(file);
    let d = detectors();
    let mut found = exemptions.findings.clone();
    for (index, line) in lines.iter().enumerate() {
        if is_comment(lang, line) {
            continue;
        }
        let number = index + 1;
        let mut add = |class: &'static str, hits: Vec<String>| {
            if hits.is_empty() {
                return;
            }
            let never_exempt =
                class == RYUK_OFF || class == OWNER_BLIND || class == DROPPED_SCRATCH;
            if never_exempt || !is_exempt(&exemptions, class, number) {
                let mut unique: Vec<String> = Vec::new();
                for hit in hits {
                    if !unique.contains(&hit) {
                        unique.push(hit);
                    }
                }
                found.push(finding(file, number, class, unique.join(", ")));
            }
        };

        let mut temp: Vec<String> = d
            .raw_temp
            .iter()
            .filter_map(|p| p.find(line).map(|m| m.as_str().trim().to_string()))
            .collect();
        if lang == Lang::Workflow {
            temp.extend(
                d.raw_temp_workflow
                    .find(line)
                    .map(|m| m.as_str().to_string()),
            );
        } else {
            temp.extend(
                d.raw_temp_quoted_tool
                    .find(line)
                    .map(|m| m.as_str().to_string()),
            );
        }
        if !helper {
            temp.extend(d.accessor.find(line).map(|m| {
                format!(
                    "{} (the helper's temp root, used outside the helper)",
                    m.as_str()
                )
            }));
        }
        add(RAW_TEMP, temp);
        add(
            RAW_WORKTREE,
            d.raw_worktree
                .iter()
                .filter_map(|p| p.find(line).map(|m| m.as_str().to_string()))
                .take(1)
                .collect(),
        );
        add(
            DOCKER_RUN,
            d.docker_run
                .iter()
                .filter_map(|p| p.find(line).map(|m| m.as_str().to_string()))
                .take(1)
                .collect(),
        );
        add(DOCKER_NAME, docker_name_hits(line));
        add(LEGACY_NAME, legacy_name_hits(line));
        add(SHARED_STORE, shared_store_hit(line));
        add(
            RYUK_OFF,
            d.ryuk_off
                .is_match(line)
                .then(|| "the reaper is switched off".to_string())
                .into_iter()
                .collect(),
        );
        if lang == Lang::Rust {
            if !helper {
                add(OWNER_BLIND, rust_owner_blind_hits(line));
            }
            add(
                DROPPED_SCRATCH,
                dropped_scratch_hit(line).into_iter().collect(),
            );
        }
    }
    found
}
