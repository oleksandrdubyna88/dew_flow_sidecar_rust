//! The scenarios of the E4 plan, S2 (`todo/PLAN_test_leftovers_E4.md`), numbered as the plan numbers them.
//!
//! Every scenario that counts or lists entries does so in a PRIVATE parent made with `new_in`, because eight test
//! threads create scratches in the temp root at once. A legacy or foreign name a scenario needs is built at run time
//! from `PREFIX` and pieces, never spelled whole: S4's guard reads this file too.

use super::*;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::path::PathBuf;

mod sweeping;
mod windows_only;

// ---------- helpers ----------

/// The conformance vectors of the mounted rule, read through the checkout. A missing file fails naming the cure.
fn vectors() -> serde_json::Value {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join(".claude/rules/shared/tools/fixtures/owner-vectors.json");
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| {
        panic!(
            "the shared rules are not checked out ({}: {e}) — run: git submodule update --init .claude/rules/shared",
            path.display()
        )
    });
    serde_json::from_str(&text).expect("owner-vectors.json is JSON")
}

fn rows(vectors: &serde_json::Value, section: &str) -> Vec<serde_json::Value> {
    vectors[section]["vectors"]
        .as_array()
        .unwrap_or_else(|| panic!("owner-vectors.json has no `{section}` vectors"))
        .clone()
}

fn owner(owner: &str, host: &str, pid: u64) -> Owner {
    Owner {
        owner: owner.to_string(),
        host: host.to_string(),
        pid,
        start: None,
        ns: None,
        canonical: true,
    }
}

/// A Linux reader: caller `u1000`, machine `box@linux`, its own pid namespace readable.
fn linux_me() -> Owner {
    Owner {
        start: Some(1_000),
        ns: Some(4_242),
        ..owner("u1000", "box@linux", 50)
    }
}

fn their_record() -> Owner {
    Owner {
        start: Some(1_000),
        ns: Some(4_242),
        ..owner("u1000", "box@linux", 777)
    }
}

fn state_of(
    claim: &Claim,
    name_owner: Option<&str>,
    mine: bool,
    me: &Owner,
    hidepid: bool,
    p: Probe,
) -> OwnerState {
    owner_state(claim, name_owner, mine, me, &Facts { hidepid }, |_| p)
}

fn panics<R>(f: impl FnOnce() -> R) -> bool {
    catch_unwind(AssertUnwindSafe(f)).is_err()
}

// ---------- 2 ----------

#[test]
fn the_owner_token_is_the_users_and_an_invalid_override_refuses() {
    let own = owner_token_from(None);
    #[cfg(unix)]
    // SAFETY: getuid has no preconditions and cannot fail.
    assert_eq!(own, format!("u{}", unsafe { libc::getuid() }));
    #[cfg(windows)]
    assert!(
        own.starts_with('w') && own[1..].bytes().all(|b| b.is_ascii_digit()),
        "{own}"
    );

    // The process-wide token is the override when the environment sets one, else the user's own.
    let from_env = std::env::var("TESTSTODELETE_OWNER").ok();
    assert_eq!(owner_token(), owner_token_from(from_env.as_deref()));

    for row in rows(&vectors(), "windows-owner") {
        let name = row["username"].as_str().expect("username");
        assert_eq!(
            windows_owner_token(name),
            row["token"].as_str().expect("token"),
            "{} ({name})",
            row["id"]
        );
    }

    assert_eq!(owner_token_from(Some("u42")), "u42");
    assert_eq!(owner_token_from(Some("w4294967295")), "w4294967295");
    for bad in [
        "bob",
        "u",
        "u12345678901",
        "U42",
        "u4x",
        "u42\n",
        " u42",
        "u-1",
    ] {
        assert!(
            panics(|| owner_token_from(Some(bad))),
            "{bad:?} must refuse"
        );
    }
    assert_eq!(
        owner_token_from(Some("")),
        own,
        "an empty override is no override"
    );
}

// ---------- 12 ----------

#[test]
fn an_owner_round_trips_and_the_shared_vectors_agree() {
    let me = this_owner();
    let text = format!("{}\r\n", format_owner(me));
    assert_eq!(parse_owner(text.as_bytes()), Claim::Owner(me.clone()));
    #[cfg(target_os = "linux")]
    assert!(
        me.start.is_some() && me.ns.is_some(),
        "Linux gives a start and a pid namespace: {me:?}"
    );
    #[cfg(windows)]
    assert!(
        me.start.is_some() && me.ns.is_none(),
        "Windows gives a start and no namespace: {me:?}"
    );

    let ns_only = Owner {
        ns: Some(5),
        ..owner("u1", "h@linux", 7)
    };
    assert_eq!(
        format_owner(&ns_only),
        "u1:h@linux:7",
        "a namespace rides only with a start"
    );
    let full = Owner {
        start: Some(9),
        ..ns_only
    };
    assert_eq!(format_owner(&full), "u1:h@linux:7:9:5");

    let vectors = vectors();
    for row in rows(&vectors, "record") {
        let text = row["text"].as_str().expect("text");
        let valid = row["valid"].as_bool().expect("valid");
        let got = parse_owner(text.as_bytes());
        assert_eq!(
            matches!(got, Claim::Owner(_)),
            valid,
            "{} {text:?}: {}",
            row["id"],
            row["why"]
        );
        if row["id"] == "r-g11" {
            let Claim::Owner(read) = got else {
                unreachable!()
            };
            assert!(!read.canonical, "007 is read and marked non-canonical");
        }
    }
    let long = format!("u1000:{}@linux:1", "h".repeat(MAX_CLAIM_BYTES));
    assert_eq!(
        parse_owner(long.as_bytes()),
        Claim::Unreadable,
        "longer than 512 bytes"
    );
    assert_eq!(
        parse_owner(b"u1000:h\xff@linux:1"),
        Claim::Unreadable,
        "not UTF-8"
    );

    for row in rows(&vectors, "hostnames") {
        let raw = row["raw"].as_str().expect("raw");
        let want = row["host"].as_str().expect("host");
        match row["distro"].as_str() {
            Some(distro) => assert_eq!(host_id(raw, distro), want, "{}", row["id"]),
            None => assert_eq!(host_name(raw), want, "{}", row["id"]),
        }
    }

    for row in rows(&vectors, "names") {
        let name = row["name"].as_str().expect("name");
        let want = match row["form"].as_str().expect("form") {
            "owner" => NameForm::Owner {
                owner: row["owner"].as_str().expect("owner").to_string(),
                token: row["token"].as_str().map(str::to_string),
            },
            "legacy" => NameForm::Legacy {
                token: row["token"].as_str().expect("token").to_string(),
            },
            _ => NameForm::None,
        };
        assert_eq!(classify_name(name), want, "{} {name}", row["id"]);
    }

    for row in rows(&vectors, "scope") {
        let id = row["id"].to_string();
        let caller = row["callerOwner"].as_str().expect("callerOwner");
        let me = Owner {
            owner: caller.to_string(),
            ..linux_me()
        };
        let mut record = Owner {
            owner: row["recordOwner"]
                .as_str()
                .expect("recordOwner")
                .to_string(),
            ..their_record()
        };
        match row["ns"].as_str().expect("ns") {
            "same" => {}
            "absent" => record.ns = None,
            "other" => record.ns = Some(4_243),
            "leading-zero" => record.canonical = false,
            other => panic!("{id}: unknown ns {other}"),
        }
        let probe = match row["pid"].as_str().expect("pid") {
            "absent" => Probe::Gone,
            "present" => Probe::Alive { start: None },
            _ => Probe::Unknown,
        };
        let got = state_of(
            &Claim::Owner(record),
            row["nameOwner"].as_str(),
            row["kernelMine"].as_bool().expect("kernelMine"),
            &me,
            row["hidepid"].as_bool().expect("hidepid"),
            probe,
        );
        let want = match row["expected"].as_str().expect("expected") {
            "gone" => OwnerState::Gone,
            "live" => OwnerState::Live,
            _ => OwnerState::Unknown,
        };
        assert_eq!(got, want, "{id}: {}", row["why"]);
    }
}

// ---------- 13 ----------

#[test]
fn owner_state_follows_rule_three_in_order() {
    let me = linux_me();
    let record = their_record();
    let claim = |o: Owner| Claim::Owner(o);
    let run = |c: &Claim, name: Option<&str>, mine: bool, me: &Owner, p: Probe| {
        state_of(c, name, mine, me, false, p)
    };
    let alive = Probe::Alive { start: Some(1_000) };
    let gone = Probe::Gone;

    assert_eq!(
        run(&claim(record.clone()), Some("u1000"), true, &me, gone),
        OwnerState::Gone,
        "the baseline: every key agrees, no such process"
    );
    assert_eq!(
        run(&claim(record.clone()), Some("u1000"), true, &me, alive),
        OwnerState::Live
    );

    assert_eq!(
        run(&Claim::Unmarked, None, true, &me, gone),
        OwnerState::Unknown,
        "unmarked"
    );
    assert_eq!(
        run(&Claim::Unreadable, None, true, &me, gone),
        OwnerState::Unknown,
        "unreadable"
    );
    let noncanonical = Owner {
        canonical: false,
        ..record.clone()
    };
    assert_eq!(
        run(&claim(noncanonical), None, true, &me, gone),
        OwnerState::Unknown,
        "a leading zero"
    );
    let stranger = Owner {
        owner: "u7".into(),
        ..record.clone()
    };
    assert_eq!(
        run(&claim(stranger), None, true, &me, gone),
        OwnerState::Unknown,
        "record owner u7 under caller u1000"
    );
    assert_eq!(
        run(&claim(record.clone()), Some("u7"), true, &me, gone),
        OwnerState::Unknown,
        "name owner differs from the record's"
    );
    assert_eq!(
        run(&claim(record.clone()), Some("u1000"), false, &me, gone),
        OwnerState::Unknown,
        "the kernel owner is not the caller"
    );
    let other_host = Owner {
        host: "other@linux".into(),
        ..record.clone()
    };
    assert_eq!(
        run(&claim(other_host), None, true, &me, gone),
        OwnerState::Unknown,
        "another machine"
    );
    let other_os = Owner {
        host: "box@windows".into(),
        ..record.clone()
    };
    assert_eq!(
        run(&claim(other_os), None, true, &me, gone),
        OwnerState::Unknown,
        "this host, the other OS tag"
    );
    let upper = Owner {
        host: "BOX@linux".into(),
        ..record.clone()
    };
    assert_eq!(
        run(&claim(upper), None, true, &me, gone),
        OwnerState::Gone,
        "this host in another case is evaluated"
    );

    let no_ns = Owner {
        ns: None,
        ..record.clone()
    };
    assert_eq!(
        run(&claim(no_ns), None, true, &me, gone),
        OwnerState::Unknown,
        "a Linux record without a namespace"
    );
    let other_ns = Owner {
        ns: Some(4_243),
        ..record.clone()
    };
    assert_eq!(
        run(&claim(other_ns), None, true, &me, gone),
        OwnerState::Unknown,
        "another namespace"
    );
    let blind_me = Owner {
        ns: None,
        ..me.clone()
    };
    assert_eq!(
        run(&claim(record.clone()), None, true, &blind_me, gone),
        OwnerState::Unknown,
        "the reader's own namespace is unreadable"
    );

    for pid in [0, 9_007_199_254_740_992, u64::MAX] {
        let odd = Owner {
            pid,
            ..record.clone()
        };
        assert_eq!(
            run(&claim(odd), None, true, &me, gone),
            OwnerState::Unknown,
            "pid {pid} is no process"
        );
    }
    let windows_me = Owner {
        host: "box@windows".into(),
        ns: None,
        ..me.clone()
    };
    let one = Owner {
        host: "box@windows".into(),
        pid: 1,
        ns: None,
        ..record.clone()
    };
    assert_eq!(
        run(&claim(one), None, true, &windows_me, gone),
        OwnerState::Unknown,
        "windows has no pid 1"
    );
    let win_rec = Owner {
        host: "box@windows".into(),
        ns: None,
        ..record.clone()
    };
    assert_eq!(
        run(&claim(win_rec), None, true, &windows_me, gone),
        OwnerState::Gone,
        "a Windows reader needs no namespace"
    );

    assert_eq!(
        state_of(&claim(record.clone()), None, true, &me, true, gone),
        OwnerState::Unknown,
        "hidepid: a missing pid proves nothing"
    );
    assert_eq!(
        run(&claim(record.clone()), None, true, &me, Probe::Unknown),
        OwnerState::Unknown
    );
    let no_start = Owner {
        start: None,
        ..record.clone()
    };
    assert_eq!(
        run(&claim(no_start), None, true, &me, alive),
        OwnerState::Live,
        "the owner carries no start"
    );
    assert_eq!(
        run(
            &claim(record.clone()),
            None,
            true,
            &me,
            Probe::Alive { start: None }
        ),
        OwnerState::Live,
        "the reader cannot read the live start"
    );
    for (delta, want) in [
        (2i64, OwnerState::Live),
        (-2, OwnerState::Live),
        (3, OwnerState::Unknown),
        (-3, OwnerState::Unknown),
        (86_400, OwnerState::Unknown),
    ] {
        let seen = Probe::Alive {
            start: Some((1_000 + delta) as u64),
        };
        assert_eq!(
            run(&claim(record.clone()), None, true, &me, seen),
            want,
            "start delta {delta} s — never gone"
        );
    }
}

// ---------- 14 ----------

#[test]
fn the_real_probe_sees_this_process_and_not_an_impossible_pid() {
    let me = this_owner();
    let Probe::Alive { start } = probe(u64::from(std::process::id())) else {
        panic!("this process must read as alive");
    };
    let (Some(live), Some(written)) = (start, me.start) else {
        panic!(
            "both starts are readable here: live {start:?}, written {:?}",
            me.start
        );
    };
    assert!(
        live.abs_diff(written) <= START_TOLERANCE_S,
        "{live} vs {written}"
    );

    #[cfg(target_os = "linux")]
    {
        assert!(
            std::path::Path::new("/proc/1").exists(),
            "hidepid is on: a missing pid proves nothing here, so this scenario cannot say `Gone`"
        );
        let max: u64 = std::fs::read_to_string("/proc/sys/kernel/pid_max")
            .expect("pid_max")
            .trim()
            .parse()
            .expect("a number");
        let impossible = max + 1;
        assert!(
            !std::path::Path::new(&format!("/proc/{impossible}")).exists(),
            "premise before"
        );
        assert_eq!(probe(impossible), Probe::Gone);
        assert!(
            !std::path::Path::new(&format!("/proc/{impossible}")).exists(),
            "premise after"
        );
    }
}

// ---------- identity (beyond the plan's list: the lessons of the C# port) ----------

#[test]
fn an_identity_carries_the_change_time_and_an_unreadable_one_never_compares() {
    let private = TestScratch::new("identity-parent");
    let file = private.path().join("entry");
    std::fs::write(&file, b"x").unwrap();
    let first = file_id(&file).expect("an identity");
    std::thread::sleep(Duration::from_millis(30));
    // A change of the inode (not of its name or number) moves the change time: a freed inode number that is reused at
    // once on some file systems still carries another change time, so a swapped entry never matches.
    let original = std::fs::metadata(&file).unwrap().permissions();
    let mut readonly = original.clone();
    readonly.set_readonly(true);
    std::fs::set_permissions(&file, readonly).unwrap();
    let second = file_id(&file).expect("an identity");
    std::fs::set_permissions(&file, original).unwrap();
    assert!(first.same_inode(&second), "the same inode");
    assert!(
        !same_id(Some(first), Some(second)),
        "but not the same identity: the change time moved"
    );
    assert!(same_id(Some(first), Some(first)));
    assert!(
        !same_id(None, None),
        "an identity that cannot be read never compares equal"
    );
    assert_eq!(
        file_id(&private.path().join("missing")),
        None,
        "an identity that cannot be read is none"
    );
}

// ---------- 15 ----------

#[test]
fn the_owner_names_this_machine_as_the_janitor_does() {
    let output = std::process::Command::new("hostname")
        .output()
        .expect("hostname runs");
    assert!(output.status.success(), "hostname failed");
    let raw = String::from_utf8(output.stdout).expect("UTF-8");
    let normalised = host_name(raw.trim_end_matches(['\r', '\n']));
    let written = &this_owner().host;
    let (host, os) = written.rsplit_once('@').expect("<host>@<os>");
    assert_eq!(os, if cfg!(windows) { "windows" } else { "linux" });
    match std::env::var("WSL_DISTRO_NAME")
        .ok()
        .filter(|d| !d.trim().is_empty())
    {
        Some(distro) => {
            let (machine, found) = host
                .split_once('~')
                .expect("under WSL the host is <host>~<distro>");
            assert!(
                machine.eq_ignore_ascii_case(&normalised),
                "{machine} vs {normalised}"
            );
            assert_eq!(found, host_name(&distro));
        }
        None => assert!(
            host.eq_ignore_ascii_case(&normalised),
            "{host} vs {normalised}"
        ),
    }
}

// ---------- helpers for the folder scenarios ----------

use std::collections::BTreeSet;
use std::io;

fn name_with(what: &str, id: &str) -> String {
    format!("{PREFIX}{}-{REPO_TOKEN}-{what}-{id}", owner_token())
}

/// The names in `dir`, sorted — what a scenario means by "the parent is empty".
fn entries(dir: &Path) -> BTreeSet<String> {
    std::fs::read_dir(dir)
        .expect("a readable parent")
        .map(|entry| {
            entry
                .expect("an entry")
                .file_name()
                .to_string_lossy()
                .into_owned()
        })
        .collect()
}

fn panic_text(f: impl FnOnce()) -> String {
    let payload = catch_unwind(AssertUnwindSafe(f)).expect_err("it must panic");
    payload
        .downcast_ref::<String>()
        .cloned()
        .or_else(|| payload.downcast_ref::<&str>().map(|s| s.to_string()))
        .unwrap_or_default()
}

use std::path::Path;

fn sequence(ids: &[&str]) -> impl FnMut() -> String {
    let mut ids: std::collections::VecDeque<String> = ids.iter().map(|id| id.to_string()).collect();
    move || {
        ids.pop_front()
            .expect("the scenario drew more ids than it gave")
    }
}

// ---------- 1 ----------

#[test]
fn a_scratch_is_named_with_the_owner_the_token_and_a_random_id() {
    let name = owned_name("seed");
    let rest = name.strip_prefix(PREFIX).expect("the prefix");
    let rest = rest
        .strip_prefix(owner_token())
        .expect("the owner follows the prefix")
        .strip_prefix("-sidecar-seed-")
        .expect("the token and `what` follow the owner");
    assert_eq!(rest.len(), ID_LENGTH, "{name}");
    assert!(rest.bytes().all(|b| b.is_ascii_alphanumeric()), "{name}");
    assert_eq!(name.split('-').nth(1), Some(owner_token()));

    let distinct: BTreeSet<String> = (0..1_000).map(|_| owned_name("seed")).collect();
    assert_eq!(distinct.len(), 1_000, "a counter or a clock alone collides");

    let scratch = TestScratch::new("unicode пробел");
    let leaf = scratch
        .path()
        .file_name()
        .and_then(|n| n.to_str())
        .expect("a UTF-8 leaf");
    assert!(
        leaf.starts_with(&format!(
            "{PREFIX}{}-sidecar-unicode пробел-",
            owner_token()
        )),
        "{leaf}"
    );
    assert_eq!(
        classify_name(leaf),
        NameForm::Owner {
            owner: owner_token().to_string(),
            token: Some("sidecar".into())
        }
    );
    assert!(scratch.path().is_dir());
    assert_eq!(
        scratch.slash_path(),
        scratch.path().to_string_lossy().replace('\\', "/")
    );

    assert_eq!(
        classify_name(&format!("{PREFIX}sidecar-x-abc123")),
        NameForm::Legacy {
            token: "sidecar".into()
        }
    );
    assert_eq!(
        classify_name(&format!("{PREFIX}u7-mcp-x-abc123")),
        NameForm::Owner {
            owner: "u7".into(),
            token: Some("mcp".into())
        }
    );
    assert_eq!(classify_name("bge-log-1-first"), NameForm::None);
}

// ---------- 3 ----------

#[test]
fn a_scratch_is_claimed_first_beside_itself_and_never_inside() {
    let parent = TestScratch::new("claim-parent");
    let scratch = TestScratch::new_in(parent.path(), "claimed");
    let claim = claim_path(scratch.path());
    assert_eq!(
        claim.parent(),
        Some(parent.path()),
        "the claim sits beside the folder"
    );
    assert_eq!(
        std::fs::read(&claim).expect("the claim"),
        format!("{}\n", format_owner(run_record())).into_bytes()
    );
    assert_eq!(
        read_claim(scratch.path(), &is_mine),
        Claim::Owner(run_record().clone())
    );
    assert_eq!(read_claim(scratch.path(), &|_| false), Claim::Unreadable);
    assert!(
        entries(scratch.path()).is_empty(),
        "nothing is ever put inside"
    );

    // Claim first, observed: the claim exists when `create_dir` is asked, and the failed create removes it again.
    let private = TestScratch::new("claim-first");
    let mut seen_claims = Vec::new();
    let mut deny = |dir: &Path| -> io::Result<()> {
        seen_claims.push(claim_path(dir).exists());
        Err(io::ErrorKind::PermissionDenied.into())
    };
    let mut seams = Seams::real();
    seams.create_dir = &mut deny;
    let text = panic_text(|| {
        TestScratch::new_with(private.path(), "denied", &mut seams);
    });
    assert!(text.contains("could not create"), "{text}");
    assert_eq!(
        seen_claims,
        [true],
        "the claim was written before the create was tried"
    );
    assert!(
        entries(private.path()).is_empty(),
        "the unwinding Drop removed the claim again"
    );

    // A claim that cannot be written: no folder is ever made.
    let mut made = false;
    let mut fail =
        |_: &Path, _: &[u8]| -> io::Result<()> { Err(io::ErrorKind::PermissionDenied.into()) };
    let mut create = |_: &Path| -> io::Result<()> {
        made = true;
        Ok(())
    };
    let mut seams = Seams::real();
    seams.write_claim = &mut fail;
    seams.create_dir = &mut create;
    let text = panic_text(|| {
        TestScratch::new_with(private.path(), "unclaimed", &mut seams);
    });
    assert!(text.contains("could not claim"), "{text}");
    assert!(!made, "no folder without a claim");
    assert!(entries(private.path()).is_empty());
}

// ---------- 4 ----------

#[test]
fn an_invalid_what_creates_nothing_and_the_room_is_derived() {
    let private = TestScratch::new("what-parent");
    let room = what_room(owner_token());
    let too_long = "a".repeat(room + 1);
    let two_bytes_each = format!("{}пп", "a".repeat(room - 3));
    assert_eq!(two_bytes_each.chars().count(), room - 1);
    assert_eq!(two_bytes_each.len(), room + 1);
    for bad in [
        "",
        "a/b",
        "a\\b",
        "-x",
        "a\u{1}b",
        "a\0b",
        too_long.as_str(),
        two_bytes_each.as_str(),
    ] {
        assert!(
            panics(|| TestScratch::new_in(private.path(), bad)),
            "{bad:?} must refuse"
        );
        assert!(
            entries(private.path()).is_empty(),
            "{bad:?} created something"
        );
    }

    let exact = "a".repeat(room);
    let scratch = TestScratch::new_in(private.path(), &exact);
    let leaf = scratch
        .path()
        .file_name()
        .and_then(|n| n.to_str())
        .expect("a leaf");
    assert_eq!(leaf.len(), MAX_NAME, "{leaf}");

    assert_eq!(what_room("u1000"), 28);
    assert_eq!(what_room("w4294967295"), 22);
}

// ---------- 5 ----------

#[test]
fn a_taken_claim_draws_again_and_a_squatted_folder_is_never_adopted() {
    let private = TestScratch::new("draw-parent");
    let parent = private.path();
    let stranger = b"stranger\n";

    // A claim somebody else wrote under the first name: not ours, left untouched, a new id drawn.
    let taken = parent.join(format!("{}.owner", name_with("draw", "a")));
    std::fs::write(&taken, stranger).unwrap();
    let mut ids = sequence(&["a", "b"]);
    let mut seams = Seams::real();
    seams.ids = &mut ids;
    let scratch = TestScratch::new_with(parent, "draw", &mut seams);
    assert!(
        scratch.path().ends_with(name_with("draw", "b")),
        "{:?}",
        scratch.path()
    );
    assert_eq!(
        std::fs::read(&taken).unwrap(),
        stranger,
        "the stranger's claim is untouched"
    );
    drop(scratch);
    std::fs::remove_file(&taken).unwrap();

    // A folder already there under the claimed name: never adopted, our claim removed again, one warning.
    let squat = parent.join(name_with("squat", "a"));
    std::fs::create_dir(&squat).unwrap();
    let mut ids = sequence(&["a", "b"]);
    let mut warning = Vec::new();
    let mut seams = Seams::real();
    seams.ids = &mut ids;
    seams.out = &mut warning;
    let scratch = TestScratch::new_with(parent, "squat", &mut seams);
    assert!(scratch.path().ends_with(name_with("squat", "b")));
    assert!(squat.is_dir(), "the squatter's folder is never removed");
    assert!(
        !claim_path(&squat).exists(),
        "our claim was removed again: nothing of ours governs it"
    );
    let warning = String::from_utf8(warning).unwrap();
    assert_eq!(warning.lines().count(), 1, "{warning}");
    assert!(warning.contains(&name_with("squat", "a")), "{warning}");
    drop(scratch);
    std::fs::remove_dir(&squat).unwrap();

    // The claim overwritten between our write and our create: it no longer holds our bytes, so it is NOT removed —
    // a Drop that removed "the claim" would take the stranger's.
    let squat = parent.join(name_with("raced", "a"));
    std::fs::create_dir(&squat).unwrap();
    let stranger_claim = claim_path(&squat);
    let mut ids = sequence(&["a", "b"]);
    let mut overwrite = |dir: &Path| {
        if dir.ends_with(name_with("raced", "a")) {
            std::fs::write(claim_path(dir), stranger).unwrap();
        }
    };
    let mut seams = Seams::real();
    seams.ids = &mut ids;
    seams.after_claim = &mut overwrite;
    let scratch = TestScratch::new_with(parent, "raced", &mut seams);
    assert!(scratch.path().ends_with(name_with("raced", "b")));
    drop(scratch);
    assert_eq!(
        std::fs::read(&stranger_claim).unwrap(),
        stranger,
        "the stranger's claim survived the Released value's Drop"
    );
    std::fs::remove_file(&stranger_claim).unwrap();
    std::fs::remove_dir(&squat).unwrap();

    // Three squatted draws: a panic naming `what` and the parent; nothing of ours is left.
    let names = ["a", "b", "c"].map(|id| parent.join(name_with("full", id)));
    for name in &names {
        std::fs::create_dir(name).unwrap();
    }
    let mut ids = sequence(&["a", "b", "c"]);
    let mut quiet = Vec::new();
    let mut seams = Seams::real();
    seams.ids = &mut ids;
    seams.out = &mut quiet;
    let text = panic_text(|| {
        TestScratch::new_with(parent, "full", &mut seams);
    });
    assert!(
        text.contains("full") && text.contains(&parent.display().to_string()),
        "{text}"
    );
    assert_eq!(
        entries(parent).len(),
        3,
        "only the three squatters remain: {:?}",
        entries(parent)
    );
}

// ---------- 6 ----------

#[test]
fn a_scratch_is_removed_when_its_test_panics() {
    let private = TestScratch::new("panic-parent");
    let mut recorded = None;
    let outcome = catch_unwind(AssertUnwindSafe(|| {
        let scratch = TestScratch::new_in(private.path(), "boom");
        recorded = Some(scratch.path().to_path_buf());
        std::fs::write(scratch.path().join("inside"), b"x").unwrap();
        panic!("a failed assertion");
    }));
    assert!(outcome.is_err());
    let path = recorded.expect("the scratch existed");
    assert!(!path.exists(), "the folder is gone");
    assert!(!claim_path(&path).exists(), "and so is its claim");
}

// ---------- 7 ----------

/// The child of scenarios 7 and 8: `#[ignore]`d, it does nothing unless the harness sets `TEST_SCRATCH_CHILD`.
#[test]
#[ignore = "the child of scenarios 7 and 8; run only by them"]
fn child_scratch() {
    let Ok(mode) = std::env::var("TEST_SCRATCH_CHILD") else {
        return;
    };
    let parent = PathBuf::from(std::env::var("TEST_SCRATCH_PARENT").expect("the parent"));
    let scratch = TestScratch::new_in(&parent, "child");
    // A leading newline: libtest has already printed `test ... ` on this line.
    println!("\nSCRATCH {}", scratch.path().display());
    println!("PID {}", std::process::id());
    match mode.as_str() {
        "assert" => assert!(mode != "assert", "the child fails a real assertion"),
        "abort" => std::process::abort(),
        other => panic!("unknown child mode {other}"),
    }
}

/// Run `child_scratch` as its own test process in `mode`; returns its exit status, the scratch path it printed and its pid.
fn run_child(mode: &str, parent: &Path) -> (std::process::ExitStatus, PathBuf, u64) {
    run_child_with(mode, parent, &[])
}

/// `run_child` with extra environment variables for the child (a shared `TESTSTODELETE_RUN`, say).
fn run_child_with(
    mode: &str,
    parent: &Path,
    envs: &[(&str, &str)],
) -> (std::process::ExitStatus, PathBuf, u64) {
    let output = std::process::Command::new(std::env::current_exe().expect("this test binary"))
        .args([
            "test_scratch::tests::child_scratch",
            "--exact",
            "--ignored",
            "--nocapture",
            "--test-threads=1",
        ])
        .env("TEST_SCRATCH_CHILD", mode)
        .env("TEST_SCRATCH_PARENT", parent)
        .envs(envs.iter().copied())
        .output()
        .expect("the child runs");
    let stdout = String::from_utf8_lossy(&output.stdout);
    let path = stdout
        .lines()
        .find_map(|line| line.strip_prefix("SCRATCH "))
        .unwrap_or_else(|| {
            panic!(
                "the child printed no path:\n{stdout}\n{}",
                String::from_utf8_lossy(&output.stderr)
            )
        });
    let pid = stdout
        .lines()
        .find_map(|line| line.strip_prefix("PID "))
        .and_then(|digits| digits.trim().parse().ok())
        .expect("the child printed its pid");
    (output.status, PathBuf::from(path), pid)
}

#[test]
fn a_failed_assertion_in_a_real_test_still_removes_its_scratch() {
    let private = TestScratch::new("child-assert");
    let (status, path, _) = run_child("assert", private.path());
    assert!(!status.success(), "the child must fail");
    assert!(path.starts_with(private.path()));
    assert!(!path.exists(), "the folder survived a failed assertion");
    assert!(!claim_path(&path).exists(), "and so did its claim");
}

// ---------- the lessons of the C# port: records are shared, files are not ----------

#[test]
fn a_claim_with_the_same_text_but_another_identity_is_not_ours_to_remove() {
    let private = TestScratch::new("same-text-parent");
    let parent = private.path();
    let squat = parent.join(name_with("same", "a"));
    std::fs::create_dir(&squat).unwrap();
    let swapped = claim_path(&squat);
    let aside = parent.join("claim-aside");
    let mut ids = sequence(&["a", "b"]);
    let mut swap = |dir: &Path| {
        if dir.ends_with(name_with("same", "a")) {
            // Another run's claim with IDENTICAL text (siblings under one TESTSTODELETE_RUN write identical
            // records); ours is kept alive beside, so the new file cannot reuse its inode number.
            let text = std::fs::read(claim_path(dir)).unwrap();
            std::fs::rename(claim_path(dir), &aside).unwrap();
            std::fs::write(claim_path(dir), text).unwrap();
        }
    };
    let mut quiet = Vec::new();
    let mut seams = Seams::real();
    seams.ids = &mut ids;
    seams.after_claim = &mut swap;
    seams.out = &mut quiet;
    let scratch = TestScratch::new_with(parent, "same", &mut seams);
    assert!(
        swapped.is_file(),
        "the claim now at that path is not the file this run wrote: it stays"
    );
    drop(scratch);
    assert!(swapped.is_file());
}

#[test]
fn siblings_sharing_one_run_record_never_remove_each_others_folders() {
    let private = TestScratch::new("shared-run");
    let record = format_owner(this_owner());
    let sibling = TestScratch::new_in(private.path(), "sibling");
    let envs = [("TESTSTODELETE_RUN", record.as_str())];

    // The child's Drop removes ITS folder and claim, by its own path — never "the folders of the run record".
    let (status, path, _) = run_child_with("assert", private.path(), &envs);
    assert!(!status.success());
    assert!(
        !path.exists() && !claim_path(&path).exists(),
        "the child's own folder is gone"
    );
    assert!(
        sibling.path().is_dir() && claim_path(sibling.path()).is_file(),
        "a sibling holding the very same record is untouched"
    );

    // A killed child leaves a claim carrying the RUN's record (the parent's pid), so the run's liveness governs it:
    // while the run lives both are kept, and nothing is ever removed because two folders share one record.
    let (_, killed, _) = run_child_with("abort", private.path(), &envs);
    let Claim::Owner(theirs) = read_claim(&killed, &|_| true) else {
        panic!("the killed child's claim is readable");
    };
    assert_eq!(
        &theirs,
        this_owner(),
        "the claim carries the run's record, not the child's"
    );
    let report = sweep(
        private.path(),
        std::time::SystemTime::now() + Duration::from_secs(11 * 60),
        SWEEP_MIN_AGE,
        probe,
        |_| true,
        &mut Hooks::default(),
    );
    assert!(report.removed.is_empty(), "{report:?}");
    assert!(killed.is_dir() && sibling.path().is_dir());
}

// ---------- 9 ----------

#[test]
fn removal_retries_then_gives_up_without_panicking() {
    let path = Path::new("unused");
    let mut calls = 0;
    let mut failed = Vec::new();
    let result = remove_with_retries(
        path,
        |_| {
            calls += 1;
            if calls <= 2 {
                Err(io::ErrorKind::PermissionDenied.into())
            } else {
                Ok(())
            }
        },
        |n| failed.push(n),
    );
    assert!(result.is_ok());
    assert_eq!(calls, 3);
    assert_eq!(failed, [1, 2], "a callback after each failed attempt");

    let mut calls = 0;
    let started = std::time::Instant::now();
    let result = remove_with_retries(
        path,
        |_| {
            calls += 1;
            Err(io::ErrorKind::PermissionDenied.into())
        },
        |_| {},
    );
    assert_eq!(calls, REMOVE_TRIES, "exactly five tries");
    assert!(
        started.elapsed() >= Duration::from_millis(400),
        "{:?} — four back-offs of 100 ms",
        started.elapsed()
    );
    assert_eq!(
        result.unwrap_err().kind(),
        io::ErrorKind::PermissionDenied,
        "the last error is returned"
    );

    let mut calls = 0;
    let result = remove_with_retries(
        path,
        |_| {
            calls += 1;
            Err(io::ErrorKind::NotFound.into())
        },
        |_| panic!("NotFound is success, not a failed attempt"),
    );
    assert!(result.is_ok());
    assert_eq!(calls, 1);
}

use std::time::Duration;

// ---------- 11 ----------

#[cfg(unix)]
#[test]
fn a_scratch_that_cannot_be_removed_stays_prefixed_and_claimed() {
    use std::os::unix::fs::PermissionsExt;

    /// Gives the folder its write bit back even when an assertion fails, so the scratches inside can be removed.
    struct Restore(PathBuf);
    impl Drop for Restore {
        fn drop(&mut self) {
            let _ = std::fs::set_permissions(&self.0, std::fs::Permissions::from_mode(0o755));
        }
    }

    let outer = TestScratch::new("held-outer");

    // A folder that cannot be emptied while its PARENT stays writable: the claim must stay with it — a teardown that
    // took the claim although the folder stayed would leave an unmarked folder nobody can ever judge.
    let stuck = TestScratch::new_in(outer.path(), "held-stuck");
    let sub = stuck.path().join("sub");
    std::fs::create_dir(&sub).unwrap();
    std::fs::write(sub.join("file"), b"x").unwrap();
    let _restore_sub = Restore(sub.clone());
    std::fs::set_permissions(&sub, std::fs::Permissions::from_mode(0o555)).unwrap();
    assert!(
        std::fs::File::create(sub.join("probe")).is_err(),
        "this process can write into a 0o555 folder (running as root?) — the scenario cannot prove anything here"
    );
    let mut buf = Vec::new();
    assert!(!teardown(stuck.path(), &mut buf));
    assert_eq!(String::from_utf8(buf).unwrap().lines().count(), 1);
    assert!(stuck.path().is_dir());
    assert!(
        claim_path(stuck.path()).is_file(),
        "the folder stayed, so its claim stays"
    );
    drop(_restore_sub);

    let inner = TestScratch::new_in(outer.path(), "held-inner");
    let _restore = Restore(outer.path().to_path_buf());
    std::fs::set_permissions(outer.path(), std::fs::Permissions::from_mode(0o555)).unwrap();
    // The premise: a process running as root removes it anyway, and this scenario would pass for the wrong reason.
    assert!(
        std::fs::File::create(outer.path().join("probe")).is_err(),
        "this process can write into a 0o555 folder (running as root?) — the scenario cannot prove anything here"
    );

    let mut buf = Vec::new();
    assert!(
        !teardown(inner.path(), &mut buf),
        "the teardown must report that it gave up"
    );
    let line = String::from_utf8(buf).unwrap();
    assert_eq!(line.lines().count(), 1, "ONE line: {line}");
    assert!(line.contains(&inner.path().display().to_string()), "{line}");
    assert!(inner.path().is_dir(), "the folder stays");
    assert!(
        claim_path(inner.path()).is_file(),
        "and stays claimed, so the next sweep can judge it"
    );
}

// ---------- 16 ----------

#[test]
fn a_claim_is_read_through_the_one_safe_read() {
    let private = TestScratch::new("read-parent");
    let parent = private.path();
    let dir = parent.join(name_with("read", "a"));
    let claim = claim_path(&dir);
    let record = |bytes: usize| -> Vec<u8> {
        // `u1000:<host>@linux:1\n` padded with the host to exactly `bytes` bytes, line end included.
        let fixed = "u1000:@linux:1\n".len();
        let mut text = format!("u1000:{}@linux:1\n", "h".repeat(bytes - fixed));
        assert_eq!(text.len(), bytes);
        text.shrink_to_fit();
        text.into_bytes()
    };

    assert_eq!(
        read_claim(&dir, &|_| true),
        Claim::Unmarked,
        "no claim at all"
    );

    std::fs::create_dir(&claim).unwrap();
    assert_eq!(
        read_claim(&dir, &|_| true),
        Claim::Unreadable,
        "a directory"
    );
    std::fs::remove_dir(&claim).unwrap();

    let real = parent.join("real-claim");
    std::fs::write(&real, record(40)).unwrap();
    #[cfg(unix)]
    {
        std::os::unix::fs::symlink(&real, &claim).unwrap();
        assert_eq!(
            read_claim(&dir, &|_| true),
            Claim::Unreadable,
            "a link is never followed"
        );
        std::fs::remove_file(&claim).unwrap();
    }

    std::fs::write(&claim, record(MAX_CLAIM_BYTES)).unwrap();
    assert!(
        matches!(read_claim(&dir, &|_| true), Claim::Owner(_)),
        "512 bytes are read"
    );
    assert_eq!(
        read_claim(&dir, &|_| false),
        Claim::Unreadable,
        "a stranger's claim is no evidence"
    );
    std::fs::write(&claim, record(MAX_CLAIM_BYTES + 1)).unwrap();
    assert_eq!(read_claim(&dir, &|_| true), Claim::Unreadable, "513 bytes");
    std::fs::remove_file(&claim).unwrap();

    #[cfg(unix)]
    {
        use std::os::unix::ffi::OsStrExt;
        let c_path = std::ffi::CString::new(claim.as_os_str().as_bytes()).unwrap();
        // SAFETY: `c_path` is a valid NUL-terminated path for the duration of the call.
        assert_eq!(unsafe { libc::mkfifo(c_path.as_ptr(), 0o600) }, 0, "mkfifo");
        let (sender, receiver) = std::sync::mpsc::channel();
        let asked = dir.clone();
        std::thread::spawn(move || sender.send(read_claim(&asked, &|_| true)).ok());
        let answer = receiver
            .recv_timeout(Duration::from_secs(1))
            .expect("a FIFO blocked the read");
        assert_eq!(answer, Claim::Unreadable);
        std::fs::remove_file(&claim).unwrap();
    }

    // The record a helper writes: the run's own when it names a LIVE run of this user, else this process's.
    let me = this_owner();
    let theirs = Owner {
        pid: 4_242,
        ..me.clone()
    };
    let alive = |_: u64| Probe::Alive { start: me.start };
    let mut buf = Vec::new();
    let used = run_record_from(Some(&format_owner(&theirs)), alive, &mut buf);
    assert_eq!(used, theirs);
    assert!(buf.is_empty(), "{}", String::from_utf8_lossy(&buf));

    for (given, probe) in [
        (
            format_owner(&Owner {
                owner: "u7".into(),
                ..theirs.clone()
            }),
            Probe::Alive { start: me.start },
        ),
        (format_owner(&theirs), Probe::Gone),
        ("bob".to_string(), Probe::Alive { start: me.start }),
    ] {
        let mut buf = Vec::new();
        let used = run_record_from(Some(&given), |_| probe, &mut buf);
        assert_eq!(&used, me, "{given}");
        assert_eq!(
            String::from_utf8(buf).unwrap().lines().count(),
            1,
            "{given}: one warning line"
        );
    }
    let mut buf = Vec::new();
    assert_eq!(&run_record_from(None, alive, &mut buf), me);
    assert!(buf.is_empty());
}
