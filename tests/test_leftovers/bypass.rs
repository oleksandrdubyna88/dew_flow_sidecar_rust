//! Review round 1 of E4: the spellings the first guard missed, found by reading the literals of a line
//! (`rustviews.rs`) instead of matching its raw text. Pure functions over one line (or one file's lines).
//!
//! STILL a spelling check: pieces on different lines, a macro that builds the word and an `include_str!` of a file
//! the guard does not scan can hide it. The research file of the module says the same.

use regex::Regex;
use std::sync::OnceLock;

use crate::classes::{legacy_name_hits, rust_word_hits, split_lines};
use crate::rustviews::{views, Views};
use crate::words::WORD;

/// The shortest fragment of the prefix word that is a finding when a literal IS it, and the shortest that is one
/// when a literal merely HOLDS it. A whole literal of five characters (`tests`) is no ordinary string in this tree; a
/// longer text that holds seven of the word's characters in a row is not ordinary either.
const WHOLE_FRAGMENT: usize = 5;
const HELD_FRAGMENT: usize = 7;

struct Patterns {
    shared_tmp: Regex,
    env_concat: Regex,
    drop_call: Regex,
    scratch_let: Regex,
    path_use: Regex,
}

fn patterns() -> &'static Patterns {
    static PATTERNS: OnceLock<Patterns> = OnceLock::new();
    PATTERNS.get_or_init(|| {
        let ctor = r"(?:Test\s*Scratch\s*::\s*new(?:_in|_with)?|seed_scr\s*atch|model_cache_with_a_tokenizer)";
        Patterns {
            // `/tmp` and `/var/tmp` as a whole path or the start of one — not a component of another path.
            shared_tmp: Regex::new(r"(?:^|[^\w./-])/(?:var/)?tmp(?:/|\\|$)").expect("the tmp pattern"),
            env_concat: Regex::new(r"\bvar(?:_os)?\s*\(\s*concat!\s*\(").expect("the env pattern"),
            drop_call: Regex::new(&format!(r"\bdrop\s*\(\s*{ctor}\b")).expect("the drop pattern"),
            scratch_let: Regex::new(&format!(r"\blet\s+(?:mut\s+)?(\w+)\s*(?::[^=]*)?=\s*{ctor}\b"))
                .expect("the let pattern"),
            path_use: Regex::new(r"\.\s*(?:path|slash_path)\b").expect("the path pattern"),
        }
    })
}

fn all_substrings_of_word(len: usize) -> Vec<String> {
    (0..=WORD.len().saturating_sub(len))
        .map(|at| WORD[at..at + len].to_string())
        .collect()
}

/// Fragments of the prefix word as literals (a literal that IS one of five characters or more, or HOLDS seven).
pub fn fragment_hits(views: &Views) -> Vec<String> {
    let mut hits = Vec::new();
    for literal in &views.literals {
        let lower = literal.to_ascii_lowercase();
        if lower.contains(WORD) {
            continue; // the whole word: judged as a name (an owner segment after it clears it), not as a piece
        }
        let whole = lower.len() >= WHOLE_FRAGMENT && WORD.contains(lower.as_str());
        let held = all_substrings_of_word(HELD_FRAGMENT)
            .iter()
            .any(|piece| lower.contains(piece.as_str()));
        if whole || held {
            hits.push(format!("a fragment of the prefix word, `{literal}`"));
        }
    }
    hits
}

/// The prefix word in the decoded line and in the joined pieces, judged as the raw line is: an owner segment after it
/// clears it, a bare word, a glob or a regex does not.
pub fn word_hits(views: &Views) -> Vec<String> {
    let mut hits = rust_word_hits(&views.decoded);
    hits.extend(rust_word_hits(&views.joined));
    hits.extend(fragment_hits(views));
    hits
}

/// The legacy-name spellings in the decoded line and in the joined pieces.
pub fn legacy_hits(views: &Views) -> Vec<String> {
    let mut hits = legacy_name_hits(&views.decoded);
    hits.extend(legacy_name_hits(&views.joined));
    hits
}

/// `/tmp` or `/var/tmp` as a literal (or in the joined pieces), and an environment name built by `concat!`.
pub fn temp_hits(views: &Views) -> Vec<String> {
    let p = patterns();
    let mut hits = Vec::new();
    for text in views.literals.iter().chain(std::iter::once(&views.joined)) {
        if let Some(found) = p.shared_tmp.find(text) {
            hits.push(format!(
                "`{}` — a shared temp directory as a literal",
                found
                    .as_str()
                    .trim_start_matches(|c: char| !c.is_alphanumeric() && c != '/')
            ));
            break;
        }
    }
    if p.env_concat.is_match(&views.code) {
        hits.push(
            "an environment name built by concat! (a raw temp variable by another spelling)"
                .to_string(),
        );
    }
    hits
}

/// `drop(<constructor>(…))`: the scratch is held by nothing and dropped by the call.
pub fn drop_call_hits(views: &Views) -> Vec<String> {
    patterns()
        .drop_call
        .find(&views.code)
        .map(|m| format!("{} — a scratch dropped by the call", m.as_str().trim()))
        .into_iter()
        .collect()
}

fn brace_delta(code: &str) -> i64 {
    code.chars().fold(0, |depth, c| match c {
        '{' => depth + 1,
        '}' => depth - 1,
        _ => depth,
    })
}

/// A scratch bound with `let` whose block ends with a TAIL expression made from its path: the scratch drops at the
/// closing brace and the path escapes the block into a folder that is already gone. `(line number, text)` of the tail.
pub fn block_escape_hits(text: &str) -> Vec<(usize, String)> {
    let p = patterns();
    let lines = split_lines(text);
    let codes: Vec<String> = lines.iter().map(|line| views(line).code).collect();
    let mut hits = Vec::new();
    for (index, code) in codes.iter().enumerate() {
        let Some(bound) = p.scratch_let.captures(code) else {
            continue;
        };
        let name = bound[1].to_string();
        let uses_name =
            Regex::new(&format!(r"\b{}\b", regex::escape(&name))).expect("the name pattern");
        // The statement can end on this line or a later one; the block the binding lives in ends where the depth
        // falls below what it was at the binding.
        let mut depth = 0i64;
        let mut last_body: Option<usize> = None;
        let mut closed_at: Option<usize> = None;
        for (at, later) in codes.iter().enumerate().skip(index) {
            let line_code = if at == index {
                // what follows the binding's own statement on its line
                later.split_once(';').map_or("", |(_, rest)| rest)
            } else {
                later.as_str()
            };
            let before = depth;
            depth += brace_delta(line_code);
            let trimmed = line_code.trim();
            if depth < 0 || (before == 0 && trimmed.starts_with('}')) {
                closed_at = Some(at);
                break;
            }
            if before == 0 && !trimmed.is_empty() {
                last_body = Some(at);
            }
        }
        let Some(closed_at) = closed_at else {
            continue;
        };
        if closed_at == index {
            // the one-line block: `{ let s = ctor(); s.path().to_path_buf() }`
            let after = codes[index].split_once(';').map_or("", |(_, rest)| rest);
            let body = after.split_once('}').map_or(after, |(body, _)| body);
            if !body.trim().is_empty() && uses_name.is_match(body) && p.path_use.is_match(body) {
                hits.push((index + 1, lines[index].trim().to_string()));
            }
            continue;
        }
        let Some(tail) = last_body else {
            continue;
        };
        let tail_code = codes[tail].trim();
        if !tail_code.ends_with(';')
            && !tail_code.ends_with('{')
            && !tail_code.ends_with(',')
            && uses_name.is_match(tail_code)
            && p.path_use.is_match(tail_code)
        {
            hits.push((tail + 1, lines[tail].trim().to_string()));
        }
    }
    hits
}
