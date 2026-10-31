//! The owner: who a scratch folder belongs to, how the record is written and read, and whether its writer lives.
//!
//! Two keys decide what a start-of-run sweep may delete (rule §3, §4): the NAME, the RECORD and the caller's token
//! must be one (scope), and the kernel's owner of the entry must be the caller; then the record's process must be
//! `gone`. Nothing here asks a child process anything — the writer and the probe read `/proc` or call a native API —
//! and every failure on the way to an answer is `unknown`, which preserves.

#[cfg(windows)]
use super::win;
use super::write_line;
use regex::Regex;
use sha2::{Digest, Sha256};
use std::io::Write;
use std::path::Path;
use std::sync::OnceLock;

/// The record of a run: `<owner>:<host>@<os>:<pid>[:<start>[:<ns>]]`. `host` holds `<host>@<os>` as written.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Owner {
    pub owner: String,
    pub host: String,
    pub pid: u64,
    pub start: Option<u64>,
    pub ns: Option<u64>,
    /// False when a number was written with a leading zero: read, but never trusted as that number (rule §3).
    pub canonical: bool,
}

/// What a claim file says. `Unmarked` is an ABSENT claim; everything not exactly the grammar is `Unreadable`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum Claim {
    Unmarked,
    Unreadable,
    Owner(Owner),
}

/// The three answers of a liveness query; only `Gone` can ever authorise a removal.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Probe {
    Gone,
    Unknown,
    Alive { start: Option<u64> },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum OwnerState {
    Live,
    Gone,
    Unknown,
}

/// Facts about the machine a liveness answer depends on. `hidepid`: the process list may hide other users'
/// processes, so a missing pid proves nothing.
pub(crate) struct Facts {
    pub hidepid: bool,
}

impl Facts {
    pub(crate) fn real() -> Facts {
        Facts {
            hidepid: hidepid_active(),
        }
    }
}

// ---------- the owner token (rule §1) ----------

/// `^[uw][0-9]{1,10}$`, by hand: no regex dialect decides what a digit is.
pub(super) fn is_owner_token(text: &str) -> bool {
    let bytes = text.as_bytes();
    matches!(bytes.first(), Some(b'u' | b'w'))
        && (2..=11).contains(&bytes.len())
        && bytes[1..].iter().all(u8::is_ascii_digit)
}

/// The Windows owner token: the account name without its domain, ASCII `A`-`Z` lower-cased and nothing else changed,
/// its UTF-8 bytes through SHA-256, the first four bytes as a big-endian `u32` in decimal (rule §1).
pub(crate) fn windows_owner_token(username: &str) -> String {
    let digest = Sha256::digest(username.to_ascii_lowercase().as_bytes());
    let number = u32::from_be_bytes([digest[0], digest[1], digest[2], digest[3]]);
    format!("w{number}")
}

#[cfg(unix)]
fn own_token() -> String {
    // SAFETY: getuid has no preconditions and cannot fail.
    format!("u{}", unsafe { libc::getuid() })
}

#[cfg(windows)]
fn own_token() -> String {
    match std::env::var("USERNAME") {
        Ok(name) if !name.is_empty() => windows_owner_token(&name),
        _ => panic!("test_scratch: %USERNAME% is not set — the owner token of this account cannot be derived, and the helper never falls back"),
    }
}

#[cfg(not(any(unix, windows)))]
fn own_token() -> String {
    panic!("test_scratch: no owner token for this operating system (the helper is built for Windows and Linux only)")
}

/// The token of `override_` when it is set and non-empty — checked against the grammar first, an invalid value
/// REFUSES and never falls back — else this user's own. The override never widens what the caller may delete: the
/// kernel owner decides (rule §3).
pub(crate) fn owner_token_from(override_: Option<&str>) -> String {
    match override_ {
        Some(value) if !value.is_empty() => {
            assert!(
                is_owner_token(value),
                "test_scratch: TESTSTODELETE_OWNER {value:?} is not an owner token (^[uw][0-9]{{1,10}}$)"
            );
            value.to_string()
        }
        _ => own_token(),
    }
}

/// This process's owner token, computed once from `TESTSTODELETE_OWNER`.
pub(crate) fn owner_token() -> &'static str {
    static TOKEN: OnceLock<String> = OnceLock::new();
    TOKEN.get_or_init(|| match std::env::var_os("TESTSTODELETE_OWNER") {
        None => owner_token_from(None),
        Some(value) => owner_token_from(Some(value.to_str().unwrap_or("\u{fffd}"))),
    })
}

// ---------- the host (rule §3) ----------

/// The whitespace Node's `\s` and the janitor's `host_norm` name — ASCII and Unicode alike, so every reader of the
/// family splits a host on the same characters.
fn is_host_space(c: char) -> bool {
    matches!(
        c,
        '\t' | '\n' | '\u{b}' | '\u{c}' | '\r' | ' ' | '\u{a0}' | '\u{1680}' | '\u{2000}'
            ..='\u{200a}'
                | '\u{2028}'
                | '\u{2029}'
                | '\u{202f}'
                | '\u{205f}'
                | '\u{3000}'
                | '\u{feff}'
    )
}

/// A host as an owner writes it: each run of `@`, `:` or whitespace becomes ONE `-`; nothing is trimmed, the case is kept.
pub(crate) fn host_name(raw: &str) -> String {
    let mut out = String::with_capacity(raw.len());
    let mut in_run = false;
    for c in raw.chars() {
        if c == '@' || c == ':' || is_host_space(c) {
            if !in_run {
                out.push('-');
            }
            in_run = true;
        } else {
            out.push(c);
            in_run = false;
        }
    }
    out
}

/// `host_name(raw)`, and under WSL `~` and the normalised distribution: distributions share one host name but each
/// has its own pid space, so another distribution's claim must read as another host's.
pub(crate) fn host_id(raw: &str, distro: &str) -> String {
    if distro.chars().all(is_host_space) {
        host_name(raw)
    } else {
        format!("{}~{}", host_name(raw), host_name(distro))
    }
}

fn os_tag() -> &'static str {
    if cfg!(target_os = "linux") {
        "linux"
    } else if cfg!(windows) {
        "windows"
    } else {
        panic!("test_scratch: no owner shape for this operating system (the helper is built for Windows and Linux only)")
    }
}

#[cfg(target_os = "linux")]
fn dns_host_name() -> String {
    let from_proc = std::fs::read_to_string("/proc/sys/kernel/hostname")
        .ok()
        .map(|text| text.trim_end_matches(['\r', '\n']).to_string())
        .filter(|text| !text.is_empty());
    from_proc.unwrap_or_else(|| {
        panic!(
            "test_scratch: /proc/sys/kernel/hostname is unreadable — this machine cannot be named"
        )
    })
}

#[cfg(windows)]
fn dns_host_name() -> String {
    win::dns_host_name().unwrap_or_else(|| {
        panic!(
            "test_scratch: GetComputerNameExW(DnsHostname) failed — this machine cannot be named"
        )
    })
}

#[cfg(not(any(target_os = "linux", windows)))]
fn dns_host_name() -> String {
    panic!("test_scratch: no owner shape for this operating system (the helper is built for Windows and Linux only)")
}

// ---------- this process (rule §3) ----------

/// This process: `{ owner, host@os, pid, start?, ns? }`, computed once. `ns` is Linux's pid namespace and rides only
/// with a start. Panics only on an operating system with no owner shape.
pub(crate) fn this_owner() -> &'static Owner {
    static THIS: OnceLock<Owner> = OnceLock::new();
    THIS.get_or_init(|| {
        let os = os_tag();
        let distro = std::env::var("WSL_DISTRO_NAME").unwrap_or_default();
        let pid = u64::from(std::process::id());
        let start = own_start(pid);
        let ns = start.and_then(|_| own_pid_namespace());
        Owner {
            owner: owner_token().to_string(),
            host: format!("{}@{os}", host_id(&dns_host_name(), &distro)),
            pid,
            start,
            ns,
            canonical: true,
        }
    })
}

#[cfg(target_os = "linux")]
fn own_start(pid: u64) -> Option<u64> {
    match procfs::read_stat(pid) {
        procfs::Stat::Found(ticks) => procfs::start_from_ticks(ticks),
        _ => None,
    }
}

#[cfg(windows)]
fn own_start(_pid: u64) -> Option<u64> {
    win::own_start()
}

#[cfg(not(any(target_os = "linux", windows)))]
fn own_start(_pid: u64) -> Option<u64> {
    None
}

/// The decimal inode of `/proc/self/ns/pid` (`pid:[N]`); none where there is none to read.
#[cfg(target_os = "linux")]
fn own_pid_namespace() -> Option<u64> {
    let link = std::fs::read_link("/proc/self/ns/pid").ok()?;
    let link = link.to_str()?;
    let digits = link.strip_prefix("pid:[")?.strip_suffix(']')?;
    (!digits.is_empty() && digits.bytes().all(|b| b.is_ascii_digit()))
        .then(|| digits.parse().ok())
        .flatten()
}

#[cfg(not(target_os = "linux"))]
fn own_pid_namespace() -> Option<u64> {
    None
}

/// The record a helper writes: the run's own, handed down in `TESTSTODELETE_RUN`, when it parses, names this user
/// AND says `live` under the real procedure — a stale value would make every claim dead at birth, so anything else
/// writes one warning line and uses this process's own record.
pub(crate) fn run_record_from(
    given: Option<&str>,
    probe: impl Fn(u64) -> Probe,
    out: &mut dyn Write,
) -> Owner {
    let me = this_owner();
    let Some(text) = given.filter(|text| !text.is_empty()) else {
        return me.clone();
    };
    if let Claim::Owner(record) = parse_owner(text.as_bytes()) {
        let claim = Claim::Owner(record.clone());
        if record.owner == me.owner
            && owner_state(&claim, None, true, me, &Facts::real(), probe) == OwnerState::Live
        {
            return record;
        }
    }
    write_line(
        out,
        "test_scratch: TESTSTODELETE_RUN is not the record of a live run of this user — ignored, this process's own record is used",
    );
    me.clone()
}

/// The record every claim of this process carries, read once from `TESTSTODELETE_RUN`.
pub(crate) fn run_record() -> &'static Owner {
    static RECORD: OnceLock<Owner> = OnceLock::new();
    RECORD.get_or_init(|| {
        let given = std::env::var_os("TESTSTODELETE_RUN").map(|value| {
            value
                .into_string()
                .unwrap_or_else(|_| "not unicode".to_string())
        });
        run_record_from(given.as_deref(), probe, &mut super::StderrOut)
    })
}

// ---------- the record's text (rule §3) ----------

/// `<owner>:<host>:<pid>`, then `:<start>` when there is one, then `:<ns>` — only together with a start.
pub(crate) fn format_owner(owner: &Owner) -> String {
    let mut text = format!("{}:{}:{}", owner.owner, owner.host, owner.pid);
    if let Some(start) = owner.start {
        text.push_str(&format!(":{start}"));
        if let Some(ns) = owner.ns {
            text.push_str(&format!(":{ns}"));
        }
    }
    text
}

/// The grammar of rule §3, anchored at both ends of the text (`\A`..`\z`: a trailing line end is NOT a free pass).
/// Digits are `[0-9]`, never Unicode `\d`; the host excludes `@`, `:` and the whitespace every reader of the family
/// names (`is_host_space`), not only ASCII's.
fn record_grammar() -> &'static Regex {
    static GRAMMAR: OnceLock<Regex> = OnceLock::new();
    GRAMMAR.get_or_init(|| {
        let space = r"\t\n\x0B\x0C\r \x{A0}\x{1680}\x{2000}-\x{200A}\x{2028}\x{2029}\x{202F}\x{205F}\x{3000}\x{FEFF}";
        let pattern = format!(
            r"\A([uw][0-9]{{1,10}}):([^@:{space}]+@(?:linux|macos|windows)):([0-9]+)(?::([0-9]+)(?::([0-9]+))?)?\z"
        );
        Regex::new(&pattern).expect("the record grammar is a valid pattern")
    })
}

/// A number that does not fit `u64` is read as one no process can have.
fn number(digits: &str) -> u64 {
    digits.parse().unwrap_or(u64::MAX)
}

/// The claim a text holds: one line (an LF or CRLF line end) of at most 512 bytes in exactly the grammar, else
/// `Unreadable`. A number with a leading zero is read and marked non-canonical.
pub(crate) fn parse_owner(text: &[u8]) -> Claim {
    if text.len() > super::MAX_CLAIM_BYTES {
        return Claim::Unreadable;
    }
    let Ok(text) = std::str::from_utf8(text) else {
        return Claim::Unreadable;
    };
    let line = text
        .strip_suffix("\r\n")
        .or_else(|| text.strip_suffix('\n'))
        .unwrap_or(text);
    let Some(found) = record_grammar().captures(line) else {
        return Claim::Unreadable;
    };
    let digits = |index: usize| found.get(index).map(|m| m.as_str());
    let leading_zero = |text: &str| text.len() > 1 && text.starts_with('0');
    let canonical = ![3, 4, 5].into_iter().filter_map(digits).any(leading_zero);
    Claim::Owner(Owner {
        owner: found[1].to_string(),
        host: found[2].to_string(),
        pid: number(&found[3]),
        start: digits(4).map(number),
        ns: digits(5).map(number),
        canonical,
    })
}

// ---------- the kernel owner (rule §3: the second key) ----------

/// Is `path` the caller's? POSIX: the entry ITSELF (a link is not followed, so never a link) is a directory or a
/// regular file whose uid is the REAL uid. Windows: profile AND owner SID, on one handle (`win::is_mine`). Any
/// failure is `false`.
#[cfg(unix)]
pub(crate) fn is_mine(path: &Path) -> bool {
    use std::os::unix::fs::MetadataExt;
    match std::fs::symlink_metadata(path) {
        // SAFETY: getuid has no preconditions and cannot fail.
        Ok(meta) => (meta.is_dir() || meta.is_file()) && meta.uid() == unsafe { libc::getuid() },
        Err(_) => false,
    }
}

#[cfg(windows)]
pub(crate) fn is_mine(path: &Path) -> bool {
    win::is_mine(path)
}

#[cfg(not(any(unix, windows)))]
pub(crate) fn is_mine(_path: &Path) -> bool {
    false
}

// ---------- liveness (rule §3) ----------

/// Is the process list possibly hiding other users' processes (`hidepid`)? Process 1 is always there: when even it
/// is invisible, a missing process proves nothing.
fn hidepid_active() -> bool {
    cfg!(target_os = "linux") && std::fs::metadata("/proc/1").is_err()
}

/// `/proc` reads: the start of a process and its parse, with the clock-tick rate from the auxiliary vector.
#[cfg(target_os = "linux")]
mod procfs {
    use std::sync::OnceLock;

    /// `AT_CLKTCK` of the auxiliary vector, read once. `sysconf(_SC_CLK_TCK)` is a constant 100 on musl — the guess
    /// the rule forbids — so a rate the kernel did not hand over is NO rate, and with it no start.
    pub(super) fn clock_ticks() -> Option<u64> {
        static RATE: OnceLock<Option<u64>> = OnceLock::new();
        *RATE.get_or_init(|| {
            // SAFETY: getauxval only reads the process's auxiliary vector; an absent type answers 0.
            let rate = unsafe { libc::getauxval(libc::AT_CLKTCK) };
            (rate > 0).then_some(rate as u64)
        })
    }

    fn boot_time() -> Option<u64> {
        static BOOT: OnceLock<Option<u64>> = OnceLock::new();
        *BOOT.get_or_init(|| {
            let text = std::fs::read_to_string("/proc/stat").ok()?;
            text.lines()
                .find_map(|line| line.strip_prefix("btime "))
                .and_then(|value| value.trim().parse().ok())
        })
    }

    pub(super) enum Stat {
        /// The process exists; field 22 (its start, in clock ticks since boot).
        Found(u64),
        Missing,
        /// A refusal, a failure, or a text that does not parse.
        Unknown,
    }

    /// Field 22 of `/proc/<pid>/stat` — counted after the LAST `)`, because the command name may hold spaces and
    /// parentheses; field 3 is the first one after it.
    pub(super) fn ticks_of(stat: &str) -> Option<u64> {
        let after = &stat[stat.rfind(')')? + 1..];
        let field = after.split_whitespace().nth(19)?;
        field
            .bytes()
            .all(|b| b.is_ascii_digit())
            .then(|| field.parse().ok())?
    }

    pub(super) fn read_stat(pid: u64) -> Stat {
        match std::fs::read(format!("/proc/{pid}/stat")) {
            Ok(bytes) => match ticks_of(&String::from_utf8_lossy(&bytes)) {
                Some(ticks) => Stat::Found(ticks),
                None => Stat::Unknown,
            },
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Stat::Missing,
            Err(_) => Stat::Unknown,
        }
    }

    /// `floor(btime + ticks / CLK_TCK)` as integer Unix seconds.
    pub(super) fn start_from_ticks(ticks: u64) -> Option<u64> {
        boot_time()?.checked_add(ticks / clock_ticks()?)
    }
}

/// Is `pid` alive on this machine? Three NAMED answers; never spawns a process, never signals anything.
#[cfg(target_os = "linux")]
pub(crate) fn probe(pid: u64) -> Probe {
    match procfs::read_stat(pid) {
        procfs::Stat::Found(ticks) => Probe::Alive {
            start: procfs::start_from_ticks(ticks),
        },
        procfs::Stat::Missing if hidepid_active() => Probe::Unknown,
        procfs::Stat::Missing => Probe::Gone,
        procfs::Stat::Unknown => Probe::Unknown,
    }
}

#[cfg(windows)]
pub(crate) fn probe(pid: u64) -> Probe {
    win::probe(pid)
}

#[cfg(not(any(target_os = "linux", windows)))]
pub(crate) fn probe(_pid: u64) -> Probe {
    Probe::Unknown
}

/// The largest integer a pid can be and still be exact in every reader of the family (2^53 - 1).
const MAX_SAFE_PID: u64 = 9_007_199_254_740_991;

/// Rule §3's liveness order exactly, for a claim of the owner form: unmarked or unreadable, a number with a leading
/// zero, a record of another owner than the caller's or the name's, a kernel owner who is not the caller, another
/// machine, another pid namespace (a Linux reader), a pid no process can have — all `Unknown`; then the probe: only
/// "no such process" in a list that hides none is `Gone`; a live process whose start times agree within 2 seconds
/// (or when either is missing) is `Live`; further apart is `Unknown`, never `Gone`.
pub(crate) fn owner_state(
    claim: &Claim,
    name_owner: Option<&str>,
    mine: bool,
    me: &Owner,
    facts: &Facts,
    probe: impl Fn(u64) -> Probe,
) -> OwnerState {
    let Claim::Owner(record) = claim else {
        return OwnerState::Unknown;
    };
    if !record.canonical || record.owner != me.owner {
        return OwnerState::Unknown;
    }
    if name_owner.is_some_and(|name| name != record.owner) || !mine {
        return OwnerState::Unknown;
    }
    let my_host = me.host.to_lowercase();
    if record.host.to_lowercase() != my_host {
        return OwnerState::Unknown;
    }
    // Two pid namespaces can share a host name, never a pid space: only a Linux reader requires one.
    if my_host.ends_with("@linux") && !record.ns.is_some_and(|ns| me.ns == Some(ns)) {
        return OwnerState::Unknown;
    }
    if !pid_is_askable(record.pid, my_host.ends_with("@windows")) {
        return OwnerState::Unknown;
    }
    match probe(record.pid) {
        Probe::Gone if facts.hidepid => OwnerState::Unknown,
        Probe::Gone => OwnerState::Gone,
        Probe::Unknown => OwnerState::Unknown,
        Probe::Alive { start } => match (record.start, start) {
            (Some(written), Some(live)) if written.abs_diff(live) > super::START_TOLERANCE_S => {
                OwnerState::Unknown
            }
            _ => OwnerState::Live,
        },
    }
}

/// Can `pid` name a process the probe may be asked about? Not 0, not past 2^53 (inexact in some reader of the family),
/// and for a reader on Windows not 1 (there is no pid 1: a `gone` answer would be a lie) and not past `u32::MAX` (a
/// Windows pid is 32 bits, never truncated). The caller checks the leading zero in the TEXT (`007` is never 7).
pub(crate) fn pid_is_askable(pid: u64, windows: bool) -> bool {
    if pid == 0 || pid > MAX_SAFE_PID {
        return false;
    }
    !(windows && (pid == 1 || pid > u64::from(u32::MAX)))
}
