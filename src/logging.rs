
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
