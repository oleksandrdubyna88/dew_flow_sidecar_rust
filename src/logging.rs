
/// UTC date and clock for the log path, from a unix timestamp.
///
/// Written out rather than pulled from `chrono`: this is the only place the crate needs a calendar date, and
/// a dependency added for one format string is a dependency to audit, licence and keep current forever.
/// The civil-from-days algorithm is Howard Hinnant's, valid for any date this product will see.
pub(crate) fn day_and_clock(unix_seconds: u64) -> (String, String) {
    let days = (unix_seconds / 86_400) as i64;
    let secs = unix_seconds % 86_400;

    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1_460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = if m <= 2 { y + 1 } else { y };

    (
        format!("{year:04}-{m:02}-{d:02}"),
        format!("{:02}-{:02}-{:02}", secs / 3600, (secs % 3600) / 60, secs % 60),
    )
}

/// The inverse: the unix second at which a civil day BEGINS.
///
/// Its only caller is retention, and the pair is what makes a day-folder name *checkable*. Anything that
/// does not survive the round trip back through `day_and_clock` is not a name this product wrote — which
/// is the whole safety property of a routine that deletes directory trees. `2026-13-45` and `2026-02-30`
/// each name some day arithmetically; neither names itself back.
///
/// Hinnant's `days_from_civil`, the companion to the algorithm above. Euclidean division so the era is
/// right for dates before 1970 — unreachable here, but a calendar that is wrong outside its expected range
/// is a calendar someone will trust outside it.
pub(crate) fn day_start(year: i64, month: i64, day: i64) -> i64 {
    let y = year - i64::from(month <= 2);
    let era = y.div_euclid(400);
    let yoe = y - era * 400;
    let mp = if month > 2 { month - 3 } else { month + 9 };
    let doy = (153 * mp + 2) / 5 + day - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;

    (era * 146_097 + doe - 719_468) * 86_400
}
