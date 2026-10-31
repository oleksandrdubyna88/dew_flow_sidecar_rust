//! Sc. 3 and 5 — a scan of nothing is red; the exclusion list is empty and pinned; discovery keeps every script
//! root and skips every `target*`. In-memory listings only: no folder is made here.

use crate::classes::DISCOVERY;
use crate::discovery::*;

fn listing(paths: &[&str]) -> Vec<String> {
    paths.iter().map(|p| (*p).to_string()).collect()
}

#[test]
fn the_exclusion_list_is_empty_and_pinned() {
    assert_eq!(
        EXCLUDED_FILES.len(),
        0,
        "no file is excluded by name — the guard scans its own files, built from pieces"
    );
}

#[test]
fn a_scan_of_nothing_is_red() {
    let healthy = listing(&[
        "src/main.rs",
        "tests/test_leftovers.rs",
        ".github/workflows/ci.yml",
    ]);
    assert!(scanned_nothing(&healthy).is_empty());

    let empty = scanned_nothing(&[]);
    assert_eq!(empty.len(), 3, "{empty:?}");
    assert!(empty.iter().all(|f| f.class == DISCOVERY));

    let no_src = scanned_nothing(&listing(&["tests/a.rs", ".github/workflows/ci.yml"]));
    assert_eq!(no_src.len(), 1);
    assert!(
        no_src[0].text.contains("scanned nothing under src/")
            && no_src[0]
                .text
                .contains("discovery broke; the repository is not clean"),
        "{no_src:?}"
    );

    let no_workflow = scanned_nothing(&listing(&[
        "src/a.rs",
        "tests/a.rs",
        ".github/dependabot.yml",
    ]));
    assert_eq!(
        no_workflow.len(),
        1,
        "a yml outside workflows/ is not a workflow: {no_workflow:?}"
    );
    assert!(no_workflow[0].text.contains(".github/workflows/"));

    // a .rs outside src/ does not stand in for one under it
    let wrong_root = scanned_nothing(&listing(&[
        "build.rs",
        "tests/a.rs",
        ".github/workflows/ci.yml",
    ]));
    assert_eq!(wrong_root.len(), 1, "{wrong_root:?}");
}

#[test]
fn discovery_keeps_every_script_root_and_skips_every_target() {
    let kept = [
        ".github/scripts/a.mjs",
        "http/x.js",
        "tools/lib/y.mjs",
        "deep/new/root/w.ts",
        "tests/test_leftovers.rs",
        "targets.rs",
        ".github/workflows/ci.yml",
        "src/test_scratch/tests/windows_only.rs",
        "scripts/q.cjs",
    ];
    let dropped = [
        "target/debug/a.rs",
        "target-wsl/x.rs",
        "target-ci/y.sh",
        "sub/target-x/z.rs",
        "http/node_modules/p/i.js",
        "node_modules/p/i.js",
        "vendor-fastembed/src/a.rs",
        ".claude/rules/shared/tools/t.mjs",
        ".claude/rules/shared/tools/t.sh",
    ];
    let all: Vec<&str> = kept.iter().chain(dropped.iter()).copied().collect();
    let found = discover(&listing(&all));
    for path in kept {
        assert!(
            found.scanned.iter().any(|p| p == path),
            "{path} was dropped: {found:?}"
        );
    }
    for path in dropped {
        assert!(
            !found.scanned.iter().any(|p| p == path) && !found.unread.iter().any(|p| p == path),
            "{path} was kept (an excluded path is never red, whatever its language): {found:?}"
        );
    }
    assert!(found.unread.is_empty(), "{found:?}");
}

#[test]
fn a_language_the_guard_does_not_read_is_red_naming_the_file() {
    for path in [
        "scripts/z.sh",
        "deep/new/root/w.ps1",
        "tools/q.py",
        "ci/run.bash",
        "ci/run.cmd",
        "ci/RUN.PS1",
    ] {
        let found = discover(&listing(&[path, "src/a.rs"]));
        assert_eq!(found.unread, vec![path.to_string()], "{path}");
        let red = unread_findings(&found);
        assert_eq!(red.len(), 1);
        assert_eq!(red[0].class, DISCOVERY);
        assert!(
            red[0].text
                == format!("test_leftovers: {path} is a language this guard does not read — extend the guard before adding it"),
            "{red:?}"
        );
    }
    // files of no script language at all are not unread
    let other = discover(&listing(&[
        "README.md",
        "Cargo.toml",
        "http/package-lock.json",
        ".github/dependabot.yml",
    ]));
    assert!(
        other.scanned.is_empty() && other.unread.is_empty(),
        "{other:?}"
    );
}

#[test]
fn the_listing_is_parsed_from_nul_separated_git_output_sorted_and_unique() {
    let parsed = parse_listing(b"src/b.rs\0src/a.rs\0src/b.rs\0\0tests/x.rs\0");
    assert_eq!(parsed, vec!["src/a.rs", "src/b.rs", "tests/x.rs"]);
    assert!(parse_listing(b"").is_empty());
}

#[test]
fn a_git_that_cannot_list_is_red_naming_the_command_not_an_empty_scan() {
    let missing = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("no-such-folder-here");
    let why = git_listing(&missing).expect_err("a folder that does not exist cannot be listed");
    assert!(
        why.contains("git ls-files -z --cached --others --exclude-standard"),
        "{why}"
    );
}
