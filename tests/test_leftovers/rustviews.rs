//! What a line of Rust says once its string literals are READ, not just matched: the escapes decoded, the pieces of
//! a `concat!`, a `format!` or a `+` joined, a one-character regex class spelled as the character, the literals
//! themselves apart from the code around them.
//!
//! Review round 1 of E4 measured the first guard against twelve planted spellings of the prefix, a temp path and a
//! dropped scratch: it caught three. A scan of the raw line sees `"\x74estsToDelete-"`, `concat!("tests", "ToDelete-")`
//! and `r"^tests[T]oDelete-"` as text that does not hold the word. These views are what such a line says when read.
//!
//! The limit, stated where the guard is: this is STILL a spelling check. Pieces on different lines, a macro that builds
//! the word, an `include_str!` of a file the guard does not scan, a constant in another crate — any of them can hide
//! the word. The guard narrows the ways to write it by accident or by habit; it cannot prove a tree clean of a name.

use regex::Regex;
use std::sync::OnceLock;

/// One Rust line, read.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Views {
    /// The line with every string literal's content removed (`""` stays): code, for the shapes of code.
    pub code: String,
    /// Every literal's content, in order, escapes decoded and one-character classes spelled as the character.
    pub literals: Vec<String>,
    /// The line with the escapes of its literals decoded and the classes spelled as the character; code kept.
    pub decoded: String,
    /// The literals joined in order (what a `concat!`, a `+` or the positional arguments of a `format!` build).
    pub joined: String,
}

fn escape_pattern() -> &'static Regex {
    static PATTERN: OnceLock<Regex> = OnceLock::new();
    PATTERN.get_or_init(|| {
        Regex::new(r"\\(?:x\{([0-9a-fA-F]{1,6})\}|x([0-9a-fA-F]{2})|u\{([0-9a-fA-F_]{1,8})\}|u([0-9a-fA-F]{4}))")
            .expect("the escape pattern")
    })
}

fn class_pattern() -> &'static Regex {
    static PATTERN: OnceLock<Regex> = OnceLock::new();
    PATTERN.get_or_init(|| {
        // `[T]`, and `[tT]` (one letter in both cases), as the letter: a regex class of a single character.
        Regex::new(r"\[(?:([^\]\\^\-\[])|([A-Za-z])([A-Za-z]))\]").expect("the class pattern")
    })
}

fn placeholder_pattern() -> &'static Regex {
    static PATTERN: OnceLock<Regex> = OnceLock::new();
    PATTERN.get_or_init(|| Regex::new(r"\{[^{}]*\}").expect("the placeholder pattern"))
}

/// `\xNN`, `\x{N}`, `\u{N}` and `\uNNNN` decoded to their character; the simple escapes too when the literal is not
/// raw (a raw string's backslash is the regex engine's, which reads the same four numeric forms).
fn decode(content: &str, raw: bool) -> String {
    let numeric = escape_pattern().replace_all(content, |caught: &regex::Captures<'_>| {
        let digits = (1..=4)
            .find_map(|i| caught.get(i))
            .map_or("", |m| m.as_str())
            .replace('_', "");
        u32::from_str_radix(&digits, 16)
            .ok()
            .and_then(char::from_u32)
            .map_or_else(|| caught[0].to_string(), |c| c.to_string())
    });
    let simple = if raw {
        numeric.into_owned()
    } else {
        let mut out = String::with_capacity(numeric.len());
        let mut chars = numeric.chars();
        while let Some(c) = chars.next() {
            if c != '\\' {
                out.push(c);
                continue;
            }
            match chars.next() {
                Some('n') => out.push('\n'),
                Some('r') => out.push('\r'),
                Some('t') => out.push('\t'),
                Some('0') => out.push('\0'),
                Some('\\') => out.push('\\'),
                Some('"') => out.push('"'),
                Some('\'') => out.push('\''),
                Some(other) => {
                    out.push('\\');
                    out.push(other);
                }
                None => out.push('\\'),
            }
        }
        out
    };
    class_pattern()
        .replace_all(&simple, |caught: &regex::Captures<'_>| {
            match (caught.get(1), caught.get(2), caught.get(3)) {
                (Some(single), _, _) => single.as_str().to_string(),
                (_, Some(a), Some(b)) if a.as_str().eq_ignore_ascii_case(b.as_str()) => {
                    a.as_str().to_string()
                }
                _ => caught[0].to_string(),
            }
        })
        .into_owned()
}

/// The raw-string opener at `chars[at..]`: `r"`, `r#"`, `br"`, `br##"` … with the number of `#`, and its length.
fn raw_opener(chars: &[char], at: usize) -> Option<(usize, usize)> {
    let mut i = at;
    if chars.get(i) == Some(&'b') {
        i += 1;
    }
    if chars.get(i) != Some(&'r') {
        return None;
    }
    i += 1;
    let mut hashes = 0;
    while chars.get(i) == Some(&'#') {
        hashes += 1;
        i += 1;
    }
    (chars.get(i) == Some(&'"')).then_some((hashes, i + 1 - at))
}

fn is_ident(c: char) -> bool {
    c.is_alphanumeric() || c == '_'
}

/// Split a line into its code (literals emptied) and its literals (content, raw or not). A `//` outside a literal
/// ends the line; a `/* … */` on the line is skipped; `'x'` and `'\n'` are characters, `'a` a lifetime.
fn split(line: &str) -> (String, Vec<(String, bool)>) {
    let chars: Vec<char> = line.chars().collect();
    let mut code = String::new();
    let mut literals = Vec::new();
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        let boundary = i == 0 || !is_ident(chars[i - 1]);
        if c == '/' && chars.get(i + 1) == Some(&'/') {
            break;
        }
        if c == '/' && chars.get(i + 1) == Some(&'*') {
            i += 2;
            while i < chars.len() && !(chars[i] == '*' && chars.get(i + 1) == Some(&'/')) {
                i += 1;
            }
            i += 2;
            continue;
        }
        if boundary && (c == 'r' || c == 'b') {
            if let Some((hashes, length)) = raw_opener(&chars, i) {
                let start = i + length;
                let mut end = start;
                while end < chars.len() {
                    if chars[end] == '"'
                        && (0..hashes).all(|k| chars.get(end + 1 + k) == Some(&'#'))
                    {
                        break;
                    }
                    end += 1;
                }
                literals.push((chars[start..end.min(chars.len())].iter().collect(), true));
                code.push_str("\"\"");
                i = (end + 1 + hashes).min(chars.len());
                continue;
            }
        }
        if c == '"' {
            let mut end = i + 1;
            while end < chars.len() && chars[end] != '"' {
                end += if chars[end] == '\\' { 2 } else { 1 };
            }
            let stop = end.min(chars.len());
            literals.push((chars[i + 1..stop].iter().collect(), false));
            code.push_str("\"\"");
            i = stop + 1;
            continue;
        }
        if c == '\'' {
            // a character literal ('x', '\n', '\u{..}', '"') versus a lifetime ('a)
            let close = if chars.get(i + 1) == Some(&'\\') {
                (i + 3..chars.len().min(i + 12)).find(|&k| chars[k] == '\'')
            } else if chars.get(i + 2) == Some(&'\'') {
                Some(i + 2)
            } else {
                None
            };
            if let Some(close) = close {
                code.push_str("''");
                i = close + 1;
                continue;
            }
        }
        code.push(c);
        i += 1;
    }
    (code, literals)
}

/// `{}` / `{name}` / `{0:?}` of a format string are replaced by the next string-literal ARGUMENTS in order (a
/// `format!("{}{}", "tests", "ToDelete-")` builds the word out of its arguments); a placeholder with none left goes.
fn join(literals: &[String]) -> String {
    let mut used = vec![false; literals.len()];
    let mut out = String::new();
    for index in 0..literals.len() {
        if used[index] {
            continue;
        }
        let mut next = index + 1;
        let piece =
            placeholder_pattern().replace_all(&literals[index], |_: &regex::Captures<'_>| {
                while next < literals.len() && used[next] {
                    next += 1;
                }
                if next < literals.len() {
                    used[next] = true;
                    literals[next].clone()
                } else {
                    String::new()
                }
            });
        out.push_str(&piece);
    }
    out
}

/// The views of one line.
pub fn views(line: &str) -> Views {
    let (code, raw) = split(line);
    let literals: Vec<String> = raw
        .iter()
        .map(|(content, is_raw)| decode(content, *is_raw))
        .collect();
    // The decoded line: the code with each literal put back, decoded.
    let mut decoded = String::new();
    let mut pieces = literals.iter();
    let mut rest = code.as_str();
    while let Some(at) = rest.find("\"\"") {
        decoded.push_str(&rest[..at]);
        decoded.push('"');
        decoded.push_str(pieces.next().map_or("", String::as_str));
        decoded.push('"');
        rest = &rest[at + 2..];
    }
    decoded.push_str(rest);
    let joined = join(&literals);
    Views {
        code,
        literals,
        decoded,
        joined,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn escapes_classes_and_pieces_are_read() {
        // an ordinary word, so no line of this file spells the prefix (the guard scans it like any other)
        let v = views(r#"let p = "\x68ell\u{6f}";"#);
        assert_eq!(v.literals, vec!["hello".to_string()]);
        assert!(v.decoded.contains("\"hello\""));
        assert_eq!(v.code, "let p = \"\";");

        let v = views(r#"let r = Regex::new(r"^hel[l]o[wW]orld");"#);
        assert_eq!(v.literals, vec!["^helloworld".to_string()]);

        let v = views(r#"concat!("a", "b") + "c" // "d""#);
        assert_eq!(v.joined, "abc");

        let v = views(r#"format!("{}{x}-", "a", "b")"#);
        assert_eq!(v.joined, "ab-");

        let v = views(r##"let r = r#"x"y"#; let c = '"'; let l: &'a str = "z";"##);
        assert_eq!(v.literals, vec!["x\"y".to_string(), "z".to_string()]);
    }
}
