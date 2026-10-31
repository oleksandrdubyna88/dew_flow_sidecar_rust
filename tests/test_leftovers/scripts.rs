//! `owner-blind` in scripts and workflows: the SHIPPED classifier's verdict, not a reimplementation.
//!
//! `classifyLines` of the mounted `tools/lib/sweeper-scan.mjs` is the one implementation the family's guard, machine
//! scan and rule-text checks share ("a second copy of these shapes is how one reader comes to call a line safe that
//! another calls a sweep"). It also traces a prefix held in a variable the SAME file assigns. This guard runs it
//! through `node` and takes its answer on four conditions, each one failing closed:
//!
//! - the driver source is a constant; the classifier's path travels in `argv` and the files' text on stdin — nothing
//!   is ever interpolated into source;
//! - the answer is validated: it must parse, every finding must name a file that was SENT and a line in `1..=count`.
//!   An answer that does not hold (node missing, a crash, a malformed or out-of-range reply, a missing mount) is ONE
//!   unjudged `owner-blind` finding — "nothing was judged" is never read as "nothing found";
//! - a file without the prefix word is not sent (the classifier answers nothing for it by its own first line), and
//!   when no file holds the word node is not started;
//! - each file's lines are split once, by the caller, and shared by the comment filter here.

use serde_json::{json, Value};
use std::io::Write;
use std::path::Path;
use std::process::{Command, Stdio};

use crate::classes::{finding, split_lines, Finding, Lang, OWNER_BLIND};
use crate::words::WORD;

/// A file the classifier is asked about: its path, its text and its lines (split once).
pub struct ScriptSource<'a> {
    pub name: &'a str,
    pub lang: Lang,
    pub text: &'a str,
    pub lines: Vec<&'a str>,
}

impl<'a> ScriptSource<'a> {
    pub fn new(name: &'a str, lang: Lang, text: &'a str) -> Self {
        ScriptSource {
            name,
            lang,
            text,
            lines: split_lines(text),
        }
    }
}

/// The driver: a constant. Reads `{ "files": [{ "name", "text" }] }` on stdin, answers
/// `{ "findings": [{ "file", "line" }] }` — the lines the shipped classifier calls broad. `process.argv[1]` is the
/// classifier module's path.
const DRIVER: &str = r#"
import * as fs from "node:fs";
import { pathToFileURL } from "node:url";
const scan = await import(pathToFileURL(process.argv[1]).href);
const request = JSON.parse(fs.readFileSync(0, "utf8"));
const findings = [];
for (const { name, text } of request.files) {
  scan.classifyLines(text, { file: name }).forEach((verdict, index) => {
    if (verdict === "broad") findings.push({ file: name, line: index + 1 });
  });
}
process.stdout.write(JSON.stringify({ findings }));
"#;

/// Where the classifier lives in the mounted rules.
pub const CLASSIFIER: &str = ".claude/rules/shared/tools/lib/sweeper-scan.mjs";

/// The ways an answer can fail to be one. Pure: no process is involved, so every shape is a fixture.
pub fn validate_answer(
    answer: &str,
    sent: &[(&str, usize)],
) -> Result<Vec<(String, usize)>, String> {
    let value: Value =
        serde_json::from_str(answer).map_err(|e| format!("the answer is not JSON ({e})"))?;
    let findings = value
        .get("findings")
        .and_then(Value::as_array)
        .ok_or_else(|| "the answer has no `findings` array".to_string())?;
    let mut valid = Vec::new();
    for entry in findings {
        let file = entry
            .get("file")
            .and_then(Value::as_str)
            .ok_or_else(|| "a finding names no file".to_string())?;
        let line = entry
            .get("line")
            .and_then(Value::as_u64)
            .ok_or_else(|| format!("a finding of `{file}` names no line number"))?;
        let Some((_, count)) = sent.iter().find(|(name, _)| *name == file) else {
            return Err(format!("a finding names `{file}`, which was not sent"));
        };
        let line = usize::try_from(line).unwrap_or(usize::MAX);
        if line == 0 || line > *count {
            return Err(format!(
                "a finding of `{file}` names line {line}, outside 1..={count}"
            ));
        }
        valid.push((file.to_string(), line));
    }
    Ok(valid)
}

/// One unjudged `owner-blind` finding: the classifier's answer could not be used, so nothing was judged.
fn unjudged(first: &str, why: &str) -> Vec<Finding> {
    vec![finding(
        first,
        1,
        OWNER_BLIND,
        format!("the shipped classifier could not judge the scripts ({why}) — nothing was judged, which is not 'nothing found'"),
    )]
}

/// Ask the classifier at `classifier` about `scripts`. A file whose text does not hold the prefix word is not sent.
pub fn script_owner_blind(scripts: &[ScriptSource<'_>], classifier: &Path) -> Vec<Finding> {
    let candidates: Vec<&ScriptSource<'_>> = scripts
        .iter()
        .filter(|script| script.text.to_ascii_lowercase().contains(WORD))
        .collect();
    let Some(first) = candidates.first() else {
        return Vec::new();
    };
    let first_name = first.name;
    if !classifier.is_file() {
        return unjudged(
            first_name,
            "the shared rules are not checked out — git submodule update --init .claude/rules/shared",
        );
    }
    let request = json!({
        "files": candidates.iter().map(|s| json!({ "name": s.name, "text": s.text })).collect::<Vec<_>>(),
    })
    .to_string();
    let spawned = Command::new("node")
        .args(["--input-type=module", "-e", DRIVER, "--"])
        .arg(classifier)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn();
    let mut child = match spawned {
        Ok(child) => child,
        Err(error) => return unjudged(first_name, &format!("node could not be started: {error}")),
    };
    let mut stdin = child.stdin.take().expect("a piped stdin");
    let writer = std::thread::spawn(move || {
        let outcome = stdin.write_all(request.as_bytes());
        drop(stdin);
        outcome
    });
    let output = match child.wait_with_output() {
        Ok(output) => output,
        Err(error) => return unjudged(first_name, &format!("node did not finish: {error}")),
    };
    let written = writer.join().ok().and_then(Result::ok).is_some();
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        let line = stderr
            .lines()
            .find(|l| !l.trim().is_empty())
            .unwrap_or("no message");
        return unjudged(
            first_name,
            &format!("node exited with {}: {line}", output.status),
        );
    }
    if !written {
        return unjudged(first_name, "the request could not be written to node");
    }
    let sent: Vec<(&str, usize)> = candidates
        .iter()
        .map(|s| (s.name, s.text.split('\n').count()))
        .collect();
    let answer = String::from_utf8_lossy(&output.stdout);
    match validate_answer(&answer, &sent) {
        Err(why) => unjudged(first_name, &why),
        Ok(broad) => broad
            .into_iter()
            .filter_map(|(file, line)| {
                let source = candidates.iter().find(|s| s.name == file)?;
                let text = source.lines.get(line - 1).copied().unwrap_or("");
                let trimmed = text.trim_start();
                let comment = match source.lang {
                    Lang::Workflow => trimmed.starts_with('#'),
                    _ => trimmed.starts_with("//"),
                };
                (!comment).then(|| {
                    finding(
                        &file,
                        line,
                        OWNER_BLIND,
                        "matches the prefix with no owner segment after it (the shipped classifier)",
                    )
                })
            })
            .collect(),
    }
}
