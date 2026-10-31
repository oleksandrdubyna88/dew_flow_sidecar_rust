//! Scenarios 2, 12, 13, 14 and 15 of the E4 plan, S2: the owner token, the record, rule 3's liveness order, the probe.

use super::*;

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

// ---------- 12, the `verdicts` section ----------

/// The shared `verdicts` vectors (v01-v26 of `owner-vectors.json`) through `owner_state` with a seam probe, as the Node
/// side runs them: the vector's facts become a record (`host`, `pidText`/`generated pid`, `startDelta` against a live
/// start the probe reports only when `liveStart`) and a probe (`pid`: absent = gone, present = alive, refused =
/// unknown); `ownerText` is raw text behind the caller's owner field, `{HOST}` this machine's `<name>@<os>` and
/// `{NAME}` its name. A vector that is refused before any liveness query must not have asked one. Two readers: a Linux
/// one (its pid namespace readable, the record's the same) and a Windows one (no namespace).
#[test]
fn the_shared_verdict_vectors_agree() {
    const LIVE_START: u64 = 1_790_000_000;
    let vectors = vectors();
    let rows = rows(&vectors, "verdicts");
    assert!(rows.len() >= 26, "v01-v26: {} vectors", rows.len());
    for tag in ["linux", "windows"] {
        let me = Owner {
            host: format!("box@{tag}"),
            start: Some(LIVE_START),
            ns: (tag == "linux").then_some(4_242),
            ..owner("u1000", "", 50)
        };
        for row in &rows {
            let id = row["id"].as_str().expect("id");
            let host = match row["host"].as_str().expect("host") {
                "same" => format!("box@{tag}"),
                "same-case" => format!("BOX@{tag}"),
                "foreign-name" => format!("other-box.example.invalid@{tag}"),
                "foreign-os" => format!("box@{}", if tag == "linux" { "windows" } else { "linux" }),
                other => panic!("{id}: unknown host fact {other}"),
            };
            let pid = row["pidText"].as_str().unwrap_or("4000000");
            let text = match row["ownerText"].as_str() {
                Some(raw) => raw
                    .replace("{HOST}", &format!("box@{tag}"))
                    .replace("{NAME}", "box"),
                None => match row["startDelta"].as_i64() {
                    Some(delta) => {
                        format!("{host}:{pid}:{}", LIVE_START.saturating_add_signed(delta))
                    }
                    None => format!("{host}:{pid}"),
                },
            };
            // The Node harness reads the legacy form and gives it the caller's owner and namespace; so here.
            let claim = match parse_owner(format!("{}:{text}", me.owner).as_bytes()) {
                Claim::Owner(record) => Claim::Owner(Owner {
                    ns: me.ns,
                    ..record
                }),
                other => other,
            };
            let asked = std::cell::RefCell::new(Vec::new());
            let live_start = row["liveStart"].as_bool().expect("liveStart");
            let probe = |pid: u64| {
                asked.borrow_mut().push(pid);
                match row["pid"].as_str().expect("pid") {
                    "absent" => Probe::Gone,
                    "refused" => Probe::Unknown,
                    _ => Probe::Alive {
                        start: live_start.then_some(LIVE_START),
                    },
                }
            };
            let got = owner_state(&claim, None, true, &me, &Facts { hidepid: false }, probe);
            let want = match row["expected"].as_str().expect("expected") {
                "gone" => OwnerState::Gone,
                "live" => OwnerState::Live,
                _ => OwnerState::Unknown,
            };
            assert_eq!(got, want, "{id} as a {tag} reader {text:?}: {}", row["why"]);
            let refused_first = row.get("ownerText").is_some()
                || row.get("pidText").is_some()
                || row["host"]
                    .as_str()
                    .is_some_and(|h| h.starts_with("foreign"));
            if refused_first {
                assert!(
                    asked.borrow().is_empty(),
                    "{id}: asked {:?} before refusing",
                    asked.borrow()
                );
            }
        }
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
