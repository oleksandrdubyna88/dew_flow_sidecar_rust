use std::fs::File;
use std::io::{self, Write};
use std::time::{SystemTime, UNIX_EPOCH};

use crate::logging::{day_and_clock, day_start};

/// Seconds in a UTC day. The segment boundary, and the only unit this file needs.
const DAY: u64 = 86_400;

/// How long a day folder is kept. Shared with `RagLogging`, `McpLogging` and `BenchLogging`, whose key is
/// `Serilog:RetentionDays` — the same answer to the same question, spelled the way each runtime spells
/// configuration.
pub(crate) const DEFAULT_RETENTION_DAYS: u64 = 14;

/// Retires day folders past the window, once, at startup. Returns what actually went.
///
/// The OTHER half of the never-restarting problem. `DaySegments` bounds any one file to a day; until this,
/// nothing bounded the total — and a sidecar started once and left alone for months fills a disk with text,
/// where the failure arrives disguised as an inference error rather than a logging one.
///
/// Startup is the right moment rather than a convenient one: it is cheap, it is idempotent, and a process
/// that never restarts is not producing new folders either. The .NET siblings prune at exactly the same
/// point, which is what makes "who owns this directory" answerable across the family instead of per repo.
///
/// **Best effort, and deliberately unable to surprise anyone.** A folder whose name is not a day is never
/// expired and therefore never deleted; a folder that refuses to go is skipped rather than fatal. Retention
/// is not worth failing a start over — the next run tries again.
///
/// It covers `logs/` and nothing else. The engine and compile caches are keyed by content and evicted by
/// their own owners, and a spool, if this process ever writes one, is DRAINED by a consumer that alone
/// knows which records it has taken.
pub(crate) fn retire_day_folders(dir_root: &str, retention_days: u64, now: u64) -> Vec<String> {
    let mut retired = Vec::new();
    // Zero is the explicit off switch — correct when an operator job owns the folder instead. A misread
    // setting that silently deleted a month of logs is the worst failure a retention feature has available,
    // so the ambiguous value does nothing.
    if retention_days == 0 {
        return retired;
    }

    let cutoff = (now / DAY).saturating_sub(retention_days) * DAY;
    let Ok(entries) = std::fs::read_dir(dir_root) else {
        // No logs directory yet: the first run of a fresh checkout has nothing to retire.
        return retired;
    };

    for entry in entries.flatten() {
        let name = entry.file_name().to_string_lossy().into_owned();
        // Strictly older goes; the boundary day itself stays, because "keep 14 days" that quietly keeps 13
        // is a window nobody can reason about.
        let expired = day_folder_start(&name).is_some_and(|start| start < cutoff);

        if expired && entry.path().is_dir() && std::fs::remove_dir_all(entry.path()).is_ok() {
            retired.push(name);
        }
    }

    retired.sort();
    retired
}

/// The unix second a day folder's name stands for, or `None` when the name is not one this product wrote.
///
/// `None` is the safe answer and every unrecognised shape gets it. Anything under `logs/` that is not a day
/// folder was put there by a person.
fn day_folder_start(name: &str) -> Option<u64> {
    let bytes = name.as_bytes();
    if bytes.len() != 10 || bytes[4] != b'-' || bytes[7] != b'-' {
        return None;
    }

    let start = day_start(
        name[0..4].parse().ok()?,
        name[5..7].parse().ok()?,
        name[8..10].parse().ok()?,
    );
    let start = u64::try_from(start).ok()?;

    // The round trip IS the validity check — see `logging::day_start`. It costs one format and removes the
    // need for a second calendar here (leap years, month lengths) that could disagree with the first.
    (day_and_clock(start).0 == name).then_some(start)
}

/// The run's log file, continued in a new segment at every UTC midnight the process lives through.
///
/// Two rules that are each right had never been held against each other. **A file per run** is correct
/// because the question asked of a log is almost always "what did THAT run do". **This process does not
/// restart** — it is started once by the orchestrator and serves until the machine does not. Together they
/// produce one file growing for months.
///
/// So a run starting at 15:00 writes `logs/2026-08-16/bge-sidecar-device0-15-00-00-1234.log` and continues
/// in `logs/2026-08-17/bge-sidecar-device0-00-00-00-1234.log`. Same pid and same device id, so it is still
/// one run — which is what keeps this from being a rolling-by-day sink, the thing the family rule forbids:
/// rolling by day merges DIFFERENT runs into one file, and these two belong to ONE.
///
/// The boundary is the clock rather than twenty-four hours of elapsed time. Elapsed-time segments drift a
/// little each day until the files stop lining up with the folders they live in, and correlating this
/// sidecar with a .NET host becomes arithmetic — which is the one thing the shared UTC rule exists to
/// prevent.
///
/// Matches `DailyRunFileSink` in the .NET repositories; the contract is
/// `.claude/rules/shared/common/logging-serilog.md`.
pub(crate) struct DaySegments {
    dir_root: String,
    /// Everything between the day folder and the timestamp — `bge-sidecar-device0`.
    prefix: String,
    pid: u32,
    /// The first unix second that belongs to the NEXT segment. Comparing against a number costs no
    /// formatting on the overwhelmingly common path where the day has not turned.
    next_boundary: u64,
    file: Option<File>,
}

impl DaySegments {
    /// Opens the first segment. `None` when the directory or the file will not open — best effort, because
    /// an unwritable log directory must never keep the sidecar from starting.
    pub(crate) fn open(dir_root: &str, prefix: &str, started: u64) -> (Option<Self>, String) {
        let (day, clock) = day_and_clock(started);
        let pid = std::process::id();
        let path = Self::path(dir_root, &day, prefix, &clock, pid);

        let file = Self::create(dir_root, &day, &path);
        let segments = file.map(|file| Self {
            dir_root: dir_root.to_string(),
            prefix: prefix.to_string(),
            pid,
            next_boundary: (started / DAY + 1) * DAY,
            file: Some(file),
        });

        (segments, path)
    }

    fn path(dir_root: &str, day: &str, prefix: &str, clock: &str, pid: u32) -> String {
        format!("{dir_root}/{day}/{prefix}-{clock}-{pid}.log")
    }

    fn create(dir_root: &str, day: &str, path: &str) -> Option<File> {
        std::fs::create_dir_all(format!("{dir_root}/{day}")).ok()?;
        std::fs::OpenOptions::new().create(true).append(true).open(path).ok()
    }

    /// Swaps to the next day's segment when the clock has passed the boundary.
    ///
    /// Named `00-00-00` rather than the moment this line arrived: the segment BEGINS at the boundary, and a
    /// reader comparing it with the previous day's file should see the two meet rather than a gap of however
    /// long the sidecar was quiet.
    ///
    /// Forward only, by construction — the boundary only ever moves later, so a clock correction backwards
    /// cannot reopen yesterday's file and orphan today's.
    ///
    /// A file that will not open leaves the writer empty rather than failing the write: a log that cannot be
    /// written is not a reason to break the service it is describing.
    fn roll_if_due(&mut self, now: u64) {
        if now < self.next_boundary {
            return;
        }

        let (day, _) = day_and_clock(now);
        let path = Self::path(&self.dir_root, &day, &self.prefix, "00-00-00", self.pid);

        self.file = Self::create(&self.dir_root, &day, &path);
        self.next_boundary = (now / DAY + 1) * DAY;
    }
}

impl Write for DaySegments {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        let now = SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0);
        self.roll_if_due(now);

        match self.file.as_mut() {
            Some(file) => file.write(buf),
            // Reported as written. The alternative — an error out of a logging writer — reaches `tracing`'s
            // own failure path and says nothing useful to anyone.
            None => Ok(buf.len()),
        }
    }

    fn flush(&mut self) -> io::Result<()> {
        match self.file.as_mut() {
            Some(file) => file.flush(),
            None => Ok(()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 2026-08-16T15:00:00Z, and the midnight after it.
    const AFTERNOON: u64 = 1_786_892_400;
    const NEXT_MIDNIGHT: u64 = 1_786_924_800;

    fn scratch(name: &str) -> String {
        let dir = std::env::temp_dir().join(format!("bge-log-{}-{name}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        dir.to_string_lossy().replace('\\', "/")
    }

    #[test]
    fn the_first_segment_is_named_for_the_moment_the_run_started() {
        let root = scratch("first");

        let (segments, path) = DaySegments::open(&root, "bge-sidecar-device0", AFTERNOON);

        assert!(segments.is_some(), "a writable directory must yield a writer");
        assert!(
            path.ends_with(&format!(
                "2026-08-16/bge-sidecar-device0-15-00-00-{}.log",
                std::process::id()
            )),
            "unexpected path: {path}"
        );
    }

    #[test]
    fn a_run_that_outlives_the_day_continues_in_a_midnight_segment() {
        let root = scratch("roll");
        let (segments, first) = DaySegments::open(&root, "bge-sidecar-device0", AFTERNOON);
        let mut segments = segments.expect("writer");

        segments.roll_if_due(AFTERNOON);
        write_line(&mut segments, "before midnight");
        segments.roll_if_due(NEXT_MIDNIGHT + 1);
        write_line(&mut segments, "after midnight");
        segments.flush().expect("flush");

        // Named 00-00-00, not 00-00-01: the segment BEGINS at the boundary, so a reader comparing it with
        // the previous day's file sees the two meet rather than a gap.
        let second = format!(
            "{root}/2026-08-17/bge-sidecar-device0-00-00-00-{}.log",
            std::process::id()
        );
        assert!(
            std::path::Path::new(&second).exists(),
            "missing second segment: {second}"
        );

        let day_one = std::fs::read_to_string(&first).expect("read first");
        let day_two = std::fs::read_to_string(&second).expect("read second");
        assert!(day_one.contains("before midnight") && !day_one.contains("after midnight"));
        assert!(day_two.contains("after midnight") && !day_two.contains("before midnight"));
    }

    #[test]
    fn a_line_before_the_boundary_does_not_roll() {
        let root = scratch("stay");
        let (segments, first) = DaySegments::open(&root, "bge-sidecar-device0", AFTERNOON);
        let mut segments = segments.expect("writer");

        segments.roll_if_due(NEXT_MIDNIGHT - 1);
        write_line(&mut segments, "still today");
        segments.flush().expect("flush");

        let files = walk(&root);
        assert_eq!(files.len(), 1, "one segment, not {files:?}");
        assert!(std::fs::read_to_string(&first)
            .expect("read")
            .contains("still today"));
    }

    #[test]
    fn every_segment_of_one_run_carries_the_same_pid() {
        let root = scratch("pid");
        let (segments, _) = DaySegments::open(&root, "bge-sidecar-device0", AFTERNOON);
        let mut segments = segments.expect("writer");

        segments.roll_if_due(NEXT_MIDNIGHT + 1);
        write_line(&mut segments, "day two");
        segments.roll_if_due(NEXT_MIDNIGHT + DAY + 1);
        write_line(&mut segments, "day three");
        segments.flush().expect("flush");

        // The property that keeps this from being a rolling-by-day sink: three files, one run, and the name
        // says so.
        let suffix = format!("-{}.log", std::process::id());
        let files = walk(&root);
        assert_eq!(files.len(), 3, "{files:?}");
        assert!(files.iter().all(|f| f.ends_with(&suffix)), "{files:?}");
    }

    /// 2026-08-16T12:00:00Z. Thirty days back is 2026-07-17, fourteen is 2026-08-02.
    const NOON: u64 = 1_786_881_600;

    #[test]
    fn a_day_folder_older_than_the_window_is_retired() {
        let root = with_days("old", &["2026-06-01", "2026-08-15"]);

        let retired = retire_day_folders(&root, 30, NOON);

        assert_eq!(retired, vec!["2026-06-01".to_string()]);
        assert_eq!(walk_days(&root), vec!["2026-08-15".to_string()]);
    }

    #[test]
    fn the_boundary_day_stays_and_so_does_everything_inside_the_window() {
        let root = with_days("boundary", &["2026-07-16", "2026-07-17", "2026-08-16"]);

        retire_day_folders(&root, 30, NOON);

        assert_eq!(
            walk_days(&root),
            vec!["2026-07-17".to_string(), "2026-08-16".to_string()]
        );
    }

    #[test]
    fn the_family_default_keeps_a_fortnight() {
        // Pinned rather than implied: this number is shared with three .NET repositories, and a mirror that
        // drifts is the whole reason the rule exists.
        let root = with_days("default", &["2026-08-01", "2026-08-02"]);

        let retired = retire_day_folders(&root, DEFAULT_RETENTION_DAYS, NOON);

        assert_eq!(retired, vec!["2026-08-01".to_string()]);
    }

    #[test]
    fn a_retention_of_zero_keeps_everything() {
        let root = with_days("off", &["2020-01-01"]);

        let retired = retire_day_folders(&root, 0, NOON);

        assert!(retired.is_empty(), "zero is the off switch, not a sweep");
        assert_eq!(walk_days(&root), vec!["2020-01-01".to_string()]);
    }

    #[test]
    fn a_folder_whose_name_is_not_a_date_is_never_touched() {
        // 2020-02-30 is the sharp case: old, date-SHAPED, and not a date. An implementation comparing the
        // names as strings — which is otherwise sound, since the format sorts chronologically — deletes it.
        let root = with_days(
            "bogus",
            &["2020-01-01", "2020-02-30", "2026-13-45", "keep-this"],
        );

        retire_day_folders(&root, 30, NOON);

        assert_eq!(
            walk_days(&root),
            vec![
                "2020-02-30".to_string(),
                "2026-13-45".to_string(),
                "keep-this".to_string()
            ],
            "anything under logs/ that is not a day folder was put there by a person"
        );
    }

    #[test]
    fn a_logs_directory_that_does_not_exist_yet_is_not_an_error() {
        let root = scratch("absent");

        let retired = retire_day_folders(&root, 30, NOON);

        assert!(retired.is_empty(), "a fresh checkout has nothing to retire");
    }

    /// Windows only, and not for convenience: an open handle is what makes a directory refuse to go THERE.
    /// POSIX unlinks a file another process is reading, so the same setup on Linux tests nothing at all — and
    /// a test that quietly passes for the wrong reason is worse than one that does not run.
    #[test]
    #[cfg(windows)]
    fn a_folder_that_cannot_be_removed_is_skipped_rather_than_fatal() {
        let root = with_days("held", &["2020-01-01"]);

        // A log viewer holding yesterday's file open is the ordinary case, and it must never stop the
        // sidecar from starting — the one thing a retention sweep must not cost.
        //
        // `share_mode(0)` is what makes the handle actually block the delete. Rust's `File::open` passes
        // FILE_SHARE_DELETE, so an ordinary open is removed out from under the reader and this test would
        // have asserted nothing — it failed exactly that way when first written.
        use std::os::windows::fs::OpenOptionsExt;
        let held = std::fs::OpenOptions::new()
            .read(true)
            .share_mode(0)
            .open(format!("{root}/2020-01-01/line.log"))
            .expect("open");

        let retired = retire_day_folders(&root, 30, NOON);

        assert!(retired.is_empty(), "a folder that refused is not a folder that went");
        assert_eq!(walk_days(&root), vec!["2020-01-01".to_string()]);
        drop(held);
    }

    fn with_days(name: &str, days: &[&str]) -> String {
        let root = scratch(name);
        for day in days {
            std::fs::create_dir_all(format!("{root}/{day}")).expect("day folder");
            std::fs::write(format!("{root}/{day}/line.log"), "a line").expect("line");
        }

        root
    }

    fn walk_days(root: &str) -> Vec<String> {
        let mut found: Vec<String> = std::fs::read_dir(root)
            .into_iter()
            .flatten()
            .flatten()
            .map(|e| e.file_name().to_string_lossy().into_owned())
            .collect();
        found.sort();
        found
    }

    /// Writes through `roll_if_due` already having been called, so the tests drive the boundary rather than
    /// the wall clock — `Write::write` reads the real time and cannot be steered.
    fn write_line(segments: &mut DaySegments, line: &str) {
        let file = segments.file.as_mut().expect("an open segment");
        file.write_all(line.as_bytes()).expect("write");
        file.write_all(b"\n").expect("newline");
    }

    fn walk(root: &str) -> Vec<String> {
        let mut found = Vec::new();
        let Ok(days) = std::fs::read_dir(root) else {
            return found;
        };

        for day in days.flatten() {
            if let Ok(entries) = std::fs::read_dir(day.path()) {
                for entry in entries.flatten() {
                    found.push(entry.file_name().to_string_lossy().into_owned());
                }
            }
        }

        found.sort();
        found
    }
}
