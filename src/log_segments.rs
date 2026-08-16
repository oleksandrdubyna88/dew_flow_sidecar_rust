use std::fs::File;
use std::io::{self, Write};
use std::time::{SystemTime, UNIX_EPOCH};

use crate::logging::day_and_clock;

/// Seconds in a UTC day. The segment boundary, and the only unit this file needs.
const DAY: u64 = 86_400;

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
