use crate::config::{env_parse, env_truthy};
use crate::inference::set_activity;
use crate::state::AppState;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

// ---------- the wedge detector: this file's one unbounded wait, made observable ----------
//
// An ORT/MIGraphX forward pass cannot be cancelled. It is a C++ call on a thread we do not own, and a
// thread merely STUCK inside it never panics — so `lock_or_refuse`'s poison healing, which recovers a
// mutex a PANIC poisoned, can never reach it. Before 2026-08-16 that combination had no detector at
// all: every later /embed queued on `.lock()` forever, /health reported the freeze exactly as it
// reports a healthy multi-minute build, and the daemon's deliberately infinite sidecar HTTP timeout
// composed the two into a system-wide freeze nobody could see (the four-repo reliability audit,
// .claude/rules/shared/common/reliability.md § "Every wait has a ceiling").
//
// The remedy is not cancellation — it cannot be — but VISIBILITY plus a ceiling: stamp what holds an
// engine and since when, declare it wedged once that passes the phase's ceiling, refuse new requests
// with a reason instead of queueing them, and say so in the log without waiting to be asked.

/// What an engine's holder is doing. The two phases exist because their ceilings differ by an order
/// of magnitude, and conflating them is what would make a correct cold compile look like a freeze.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Phase {
    /// Building + canary-checking a session. Minutes here are CORRECT.
    Building,
    /// A forward pass on an engine that is already built.
    Running,
}

impl Phase {
    /// The name /health reports.
    pub(crate) fn name(self) -> &'static str {
        match self {
            Phase::Building => "building",
            Phase::Running => "running",
        }
    }
}

/// The ceilings, all env-overridable. Deliberately generous: a false "wedged" is expensive in both
/// directions — it refuses requests a healthy compile would have served, and (with the opt-in exit)
/// killing a process mid-compile is precisely how a corrupt `.mxr` lands in the compiled-model cache,
/// the 2026-07-31 incident the build canary exists for.
#[derive(Clone, Copy)]
pub(crate) struct WedgePolicy {
    /// A forward pass on a built engine. Warm passes are seconds (measured 1.6 s at a 256 cap, 6.8 s at
    /// 1024); the slowest legitimate one on record is ~608 s, when a first request also paid a lazy
    /// MIGraphX compile plus its settling retries, and a first rerank pass compiles 92-162 s with no
    /// canary ahead of it. 900 s is ~1.5x that worst honest case.
    pub(crate) running_after: Duration,
    /// Building + canary-checking a session. This phase legitimately contains the cold compile
    /// (measured 214 s), up to `SETTLE_ATTEMPTS` canary runs, and — on a corrupt cache — a wipe plus one
    /// clean recompile with a canary of its own. An hour, not a quarter of one.
    pub(crate) building_after: Duration,
    /// How long /unload waits for an engine before answering "still loaded". It is the operator's
    /// recovery tool and the host's GPU-lease handover: long enough to ride out a normal in-flight pass,
    /// short enough that the coordinator gets an answer rather than a hang.
    pub(crate) unload_wait: Duration,
    /// How often a waiter re-checks the lock and the holder's stamp.
    pub(crate) poll: Duration,
    /// Recovery of last resort: exit the process — the host restarts the sidecar — once an engine has
    /// been WEDGED (not merely busy) for this long on top of its ceiling. `None`, the DEFAULT, never
    /// exits. Opt in with `WEDGE_EXIT=1`.
    pub(crate) exit_after_wedged: Option<Duration>,
}

impl WedgePolicy {
    pub(crate) fn from_env() -> Self {
        let running_after = Duration::from_secs(env_parse("WEDGE_RUNNING_AFTER_SECONDS", 900));
        Self {
            running_after,
            building_after: Duration::from_secs(env_parse("WEDGE_BUILDING_AFTER_SECONDS", 3600)),
            unload_wait: Duration::from_secs(env_parse("UNLOAD_LOCK_WAIT_SECONDS", 30)),
            poll: Duration::from_millis(env_parse("WEDGE_POLL_MS", 50)),
            exit_after_wedged: env_truthy("WEDGE_EXIT").then(|| {
                Duration::from_secs(env_parse(
                    "WEDGE_EXIT_AFTER_SECONDS",
                    running_after.as_secs(),
                ))
            }),
        }
    }

    /// How long this phase may run before it stops being "slow but alive".
    pub(crate) fn ceiling(&self, phase: Phase) -> Duration {
        match phase {
            Phase::Building => self.building_after,
            Phase::Running => self.running_after,
        }
    }
}

/// What holds an engine right now, and since when.
///
/// It lives in its OWN tiny mutex, never inside the engine slot, and that is the whole design: a
/// holder wedged UNDER the engine mutex could never be observed THROUGH that same mutex, which is
/// exactly why the freeze was invisible. This lock is only ever held for the length of an assignment.
#[derive(Clone)]
pub(crate) struct InFlight {
    pub(crate) phase: Phase,
    /// The same human label /health already showed as `activity` ("embed: embedding 64 row(s)").
    pub(crate) label: String,
    pub(crate) since: Instant,
}

/// Stamps the engine a request holds, and CLEARS the stamp on drop — including the `?` early return
/// and the panic unwind, which is where a hand-written clear gets forgotten and leaves a phantom wedge
/// that refuses every later request for the life of the process.
pub(crate) struct InFlightStamp<'a> {
    pub(crate) state: &'a AppState,
    pub(crate) slot: &'a Mutex<Option<InFlight>>,
}

impl<'a> InFlightStamp<'a> {
    pub(crate) fn hold(state: &'a AppState, slot: &'a Mutex<Option<InFlight>>) -> Self {
        Self { state, slot }
    }

    /// Enters a phase: re-stamps the record — the clock restarts, because a finished build is not part
    /// of the pass that follows it — and mirrors the label into /health's `activity`, so the operator's
    /// window and the wedge detector can never disagree about what is happening.
    pub(crate) fn enter(&self, phase: Phase, label: impl Into<String>) {
        let label = label.into();
        set_activity(self.state, label.clone());
        write_inflight(
            self.slot,
            Some(InFlight {
                phase,
                label,
                since: Instant::now(),
            }),
        );
    }
}

impl Drop for InFlightStamp<'_> {
    fn drop(&mut self) {
        write_inflight(self.slot, None);
    }
}

/// Writes the in-flight record, healing poison. This mutex guards three fields and is never held
/// across anything that can block, so a poisoned one means a panic somewhere else — never a wedge
/// here — and failing to clear the stamp because of it would strand the engine as permanently busy.
pub(crate) fn write_inflight(slot: &Mutex<Option<InFlight>>, record: Option<InFlight>) {
    match slot.lock() {
        Ok(mut guard) => *guard = record,
        Err(poisoned) => {
            slot.clear_poison();
            *poisoned.into_inner() = record;
        }
    }
}

/// A NON-BLOCKING read of the in-flight record. /health's standing rule, and doubly so here: the
/// entire purpose of this slot is to stay readable while something else is stuck.
pub(crate) fn inflight_now(slot: &Mutex<Option<InFlight>>) -> Option<InFlight> {
    slot.try_lock().ok().and_then(|holder| holder.clone())
}

/// The refusal a caller gets rather than an unbounded queue. A distinct error type so the HTTP layer
/// can answer **503** (temporary — retry, degrade, or look at /health) instead of 500 ("your request
/// was wrong"): the host's degradation logic reads that difference, and a wedge is not the caller's
/// fault.
#[derive(Debug)]
pub(crate) struct EngineWedged {
    pub(crate) what: String,
    /// What the holder said it was doing; empty when nothing had stamped it.
    pub(crate) activity: String,
    pub(crate) elapsed: Duration,
    /// True = the holder passed its own ceiling (a real wedge). False = it is still legitimately busy
    /// and THIS caller ran out of patience, which is a different sentence for the operator to read.
    pub(crate) wedged: bool,
}

impl std::fmt::Display for EngineWedged {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let seconds = self.elapsed.as_secs();
        if self.wedged {
            return write!(
                f,
                "the {} engine is WEDGED: `{}` has held it for {seconds}s, past its ceiling. An ONNX Runtime \
                 call cannot be cancelled from outside, so this request is refused instead of queueing behind \
                 it forever — see /health (status \"wedged\"), then POST /unload or restart the sidecar",
                self.what, self.activity
            );
        }
        write!(
            f,
            "the {} engine is busy (`{}`) and did not come free within {seconds}s — refusing rather than \
             queueing without a ceiling",
            self.what,
            if self.activity.is_empty() { "no activity recorded" } else { &self.activity }
        )
    }
}

impl std::error::Error for EngineWedged {}

/// How long a caller is willing to wait for an engine somebody else holds.
#[derive(Clone, Copy)]
pub(crate) enum Patience {
    /// The inference path: wait as long as the holder is legitimately alive, however long that is —
    /// a first-ever shape compile is minutes of CORRECT slowness, and failing a pass that would have
    /// succeeded is worse than waiting. This is the "documented pair" reliability.md allows: an
    /// unbounded-looking wait plus the detector that ends it the moment the holder stops being alive.
    UntilTheHolderIsWedged,
    /// /unload: answer the caller within a bound, whatever the holder is doing.
    AtMost(Duration),
}

/// What the watchdog does about a phase that has been in flight this long. Pure, so the policy is
/// testable without a clock, a GPU, or a process to kill.
#[derive(Debug, PartialEq, Eq)]
pub(crate) enum WedgeAction {
    /// Slow but alive.
    Nothing,
    /// Past its ceiling: /health says `wedged` and new requests are refused with the reason.
    Report,
    /// Past the OPT-IN exit ceiling on top of that: leave loudly, so the host's restart is the recovery.
    Exit,
}

/// The exit ceiling is measured FROM the wedge verdict, never from the phase start — otherwise a
/// `WEDGE_EXIT_AFTER_SECONDS` shorter than the (deliberately hour-long) build ceiling would kill the
/// process in the middle of a legitimate compile, which is the one action guaranteed to leave a
/// corrupt program in the compiled-model cache.
pub(crate) fn wedge_action(phase: Phase, elapsed: Duration, policy: WedgePolicy) -> WedgeAction {
    let ceiling = policy.ceiling(phase);
    if elapsed < ceiling {
        return WedgeAction::Nothing;
    }
    match policy.exit_after_wedged {
        Some(after) if elapsed >= ceiling + after => WedgeAction::Exit,
        _ => WedgeAction::Report,
    }
}

/// How often the watchdog looks. A wedge lasts forever by definition, so the tick only decides how
/// soon it reaches the log — never whether it does.
pub(crate) const WEDGE_WATCHDOG_TICK: Duration = Duration::from_secs(30);

/// The exit code a deliberate wedge exit leaves behind, distinct from the two startup preflights
/// (1, 2) so a supervisor's restart log says WHY this process left.
pub(crate) const WEDGE_EXIT_CODE: i32 = 3;

/// The staleness watchdog — the LOG half of the detector.
///
/// /health can only tell someone who asks, and by construction the party that would have asked (the
/// daemon, on an infinite sidecar HTTP timeout) is the one already blocked. So the wedge has to reach
/// the log on its own, once, with the phase and the elapsed time, or the incident is again only
/// visible to whoever thinks to poll a port.
pub(crate) fn spawn_wedge_watchdog(state: Arc<AppState>) {
    tokio::spawn(async move {
        let policy = state.config.wedge;
        // One announcement per wedge, not one per tick: an hour of a 30-second tick is 120 identical
        // ERROR lines, which is how a real incident gets scrolled past.
        let mut announced = [false, false];
        loop {
            tokio::time::sleep(WEDGE_WATCHDOG_TICK).await;
            let engines: [(&str, &Mutex<Option<InFlight>>); 2] = [
                ("embed", &state.engines.embed_inflight),
                ("rerank", &state.engines.rerank_inflight),
            ];
            for (at, (engine, slot)) in engines.into_iter().enumerate() {
                let Some(holder) = inflight_now(slot) else {
                    announced[at] = false;
                    continue;
                };
                let elapsed = holder.since.elapsed();
                match wedge_action(holder.phase, elapsed, policy) {
                    WedgeAction::Nothing => announced[at] = false,
                    WedgeAction::Report if announced[at] => {}
                    WedgeAction::Report => {
                        announced[at] = true;
                        tracing::error!(
                            "{engine} engine WEDGED: `{}` has held it for {}s, past the {}s ceiling for phase \
                             `{}`. An ONNX Runtime call cannot be cancelled from outside, so /health now reports \
                             status \"wedged\" and new requests are refused with this reason instead of queueing. \
                             Recovery: POST /unload, or restart the sidecar (WEDGE_EXIT=1 makes this process \
                             exit on its own).",
                            holder.label,
                            elapsed.as_secs(),
                            policy.ceiling(holder.phase).as_secs(),
                            holder.phase.name()
                        );
                    }
                    WedgeAction::Exit => {
                        tracing::error!(
                            "{engine} engine wedged for {}s in `{}` — WEDGE_EXIT is set, so this process is \
                             exiting with code {WEDGE_EXIT_CODE} for the host to restart it. Nothing inside this \
                             process can free a thread stuck in the ONNX Runtime C++ call.",
                            elapsed.as_secs(),
                            holder.label
                        );
                        std::process::exit(WEDGE_EXIT_CODE);
                    }
                }
            }
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::sync::Mutex;
    use std::time::Duration;

    use crate::config::Config;
    use crate::testing::*;

    use crate::inference::lock_or_refuse;

    /// Poison healing is UNCHANGED by the deadline: a panicked load still costs ONE request, never the
    /// process. Before it existed, a live Fast pass answered "sparse engine poisoned" for hours with
    /// Succeeded=0 — and a stuck thread is a different failure precisely because it never gets here.
    #[test]
    fn a_poisoned_engine_lock_still_heals_under_the_deadline() {
        let engine: Mutex<Option<u8>> = Mutex::new(Some(7));
        let panicked = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _held = engine.lock().expect("fresh lock");
            panic!("load blew up mid-flight");
        }));
        assert!(
            panicked.is_err() && engine.is_poisoned(),
            "precondition: the panic poisoned the lock"
        );

        let inflight = Mutex::new(None);
        let guard = lock_or_refuse(
            &engine,
            &inflight,
            "test",
            test_wedge_policy(),
            Patience::UntilTheHolderIsWedged,
        )
        .expect("a poisoned lock heals rather than refusing");
        assert!(
            guard.is_none(),
            "half-built state is dropped so the caller reloads"
        );
        drop(guard);
        assert!(
            engine.lock().is_ok(),
            "poison is cleared for every later request"
        );
    }

    /// The three verdicts, and the ordering that matters most: the OPT-IN exit is measured from the
    /// wedge verdict, never from the phase start. An exit ceiling shorter than the (hour-long) build
    /// ceiling would otherwise kill the process mid-compile — the one action guaranteed to leave a
    /// corrupt program in the compiled-model cache, which is the 2026-07-31 incident the canary exists
    /// for.
    #[test]
    fn the_wedge_verdict_spares_a_cold_compile_and_never_exits_before_it_reports() {
        let off = WedgePolicy {
            exit_after_wedged: None,
            ..test_wedge_policy()
        };
        assert_eq!(
            wedge_action(Phase::Building, Duration::from_millis(500), off),
            WedgeAction::Nothing
        );
        assert_eq!(
            wedge_action(Phase::Running, Duration::from_millis(500), off),
            WedgeAction::Report
        );
        assert_eq!(
            wedge_action(Phase::Building, Duration::from_secs(600), off),
            WedgeAction::Report,
            "the exit is OPT-IN: with it off, a wedge is only ever reported"
        );

        // Opted in, with an exit ceiling far SHORTER than the build ceiling — the dangerous combination.
        let on = WedgePolicy {
            exit_after_wedged: Some(Duration::from_millis(100)),
            ..test_wedge_policy()
        };
        assert_eq!(
            wedge_action(Phase::Building, Duration::from_millis(550), on),
            WedgeAction::Nothing,
            "a build inside its ceiling is never exited on, however short the exit ceiling is"
        );
        assert_eq!(
            wedge_action(Phase::Building, Duration::from_millis(650), on),
            WedgeAction::Report
        );
        assert_eq!(
            wedge_action(Phase::Building, Duration::from_millis(750), on),
            WedgeAction::Exit
        );
    }

    /// The shipped ceilings, asserted from the env defaults rather than from the test config: a build
    /// gets an hour because it legitimately contains a cold compile plus a wipe-and-recompile, a pass
    /// gets 15 minutes (~1.5x the slowest honest one on record), and the process exit is OFF.
    #[test]
    fn the_shipped_ceilings_leave_room_for_a_cold_compile_and_default_to_never_exiting() {
        let shipped = Config::from_env().wedge;

        assert_eq!(shipped.ceiling(Phase::Running), Duration::from_secs(900));
        assert_eq!(shipped.ceiling(Phase::Building), Duration::from_secs(3600));
        assert!(
            shipped.ceiling(Phase::Building) > shipped.ceiling(Phase::Running),
            "a compile is slower than a pass, and conflating them is what would flag correct slowness"
        );
        assert_eq!(
            shipped.unload_wait,
            Duration::from_secs(30),
            "/unload answers the lease coordinator, not never"
        );
        assert_eq!(
            shipped.exit_after_wedged, None,
            "the process exit is opt-in (WEDGE_EXIT) and OFF by default"
        );
    }
}
