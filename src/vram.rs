use crate::adapters;
use crate::compile_cache::{EMBED_CACHE_ENGINE, RERANK_CACHE_ENGINE};
use crate::state::AppState;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

// ---------- what building an engine COST on the card, when that cost belongs to one engine ----------
//
// The panel one repository over can say "dense, sparse and rerank are resident; the card holds 31.7 GB"
// and cannot say how much of that each one holds. `/health` reported DXGI's static CAPACITY sampled once
// at startup and three booleans, with nothing in between.
//
// The obvious answer — sample the process's usage before and after a build, publish the delta — rests on
// a premise that is FALSE on the flavour everybody runs. Session construction is serialized only where a
// compiled-model cache is configured, i.e. only on MIGraphX (`CachePathLease::hold`): on DirectML the
// embed and rerank builds legitimately run at once on two `spawn_blocking` threads, which is the ordinary
// situation right after a restart when the host hits both endpoints. A delta taken around the embed build
// can then contain the rerank build's allocation — not a slightly noisy number, but a number that is
// confidently wrong about WHICH engine is expensive, which is the only question the field exists to answer.
//
// So attribution is a DECISION, not a subtraction: a figure is published only when this build was alone
// for its whole window, and otherwise the sample is discarded and the count of discards is on the wire.
// `None` is not a failure state here — this service already has the vocabulary for it.
//
// The rejected alternative was to serialize every build by taking the cache lease unconditionally. That
// buys attribution with a cold-start regression on the flavour everybody runs: the lease is held across
// the build AND that engine's first pass, so the second engine would wait minutes behind the first, every
// restart, to populate a diagnostic field. If the discard counters below show that nothing is ever
// attributable on real hardware, that is the decision to revisit — which is why they are reported.

/// Builds in flight process-wide. Static because what it counts is static: there is one GPU and one
/// process, however many engines want to allocate on it.
static BUILDS_IN_FLIGHT: AtomicUsize = AtomicUsize::new(0);

/// Incremented every time a build starts while another is already in flight.
///
/// The counter alone cannot tell a build that STARTED alone that somebody joined it later, and that is
/// exactly the case a naive implementation gets wrong. Comparing this mark before and after closes it:
/// whoever joins must bump it, so the earlier build sees a number it did not start with.
static OVERLAP_MARKS: AtomicU64 = AtomicU64::new(0);

/// What one build's before/after pair is worth. Four states, because "absent" has four different
/// causes and collapsing them would make "unavailable" indistinguishable from "never attributable on
/// this machine" — and the second is what would send this design back to serializing every build.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Attribution {
    /// This build was alone for its whole window and the process's usage grew by this much.
    Measured(u64),
    /// Another build overlapped it. The delta measures the PROCESS, not this engine.
    Overlapped,
    /// Nothing was sampled: not Windows, no resolved adapter, or DXGI refused.
    NotSampled,
    /// Sampled, alone, and usage did not grow. A session build that allocates nothing is not a
    /// measurement of zero — it is evidence the allocation was invisible to this sampler (another heap,
    /// another adapter, or an EP that defers everything to the first kernel launch). Reported apart
    /// from zero, deliberately, because a zero here would be read as "this engine is free".
    NoGrowth,
}

/// The rule, as a pure function over two samples and the two counters — so the one case that matters
/// (two overlapping builds) is testable without DXGI, a GPU, or a build to overlap.
pub(crate) fn attribute(
    alone_at_start: bool,
    marks_at_start: u64,
    marks_at_end: u64,
    before: Option<u64>,
    after: Option<u64>,
) -> Attribution {
    let (Some(before), Some(after)) = (before, after) else {
        return Attribution::NotSampled;
    };
    if !alone_at_start || marks_at_start != marks_at_end {
        return Attribution::Overlapped;
    }
    match after.checked_sub(before).filter(|&grew| grew > 0) {
        Some(grew) => Attribution::Measured(grew),
        None => Attribution::NoGrowth,
    }
}

/// One build's measurement window. Counts itself in for its whole lifetime — including a `?` return
/// and a panic unwind, which is where a hand-written decrement gets forgotten and leaves every later
/// build permanently "overlapped".
pub(crate) struct SoloBuildWindow {
    alone_at_start: bool,
    marks_at_start: u64,
    before: Option<u64>,
}

impl SoloBuildWindow {
    /// The mark is read BEFORE this build counts itself in, and that order is the whole correctness
    /// argument: any build that later observes us must have seen our `fetch_add`, so its own bump of the
    /// mark lands after this read and we cannot miss it.
    pub(crate) fn open(before: Option<u64>) -> Self {
        let marks_at_start = OVERLAP_MARKS.load(Ordering::SeqCst);
        let alone_at_start = BUILDS_IN_FLIGHT.fetch_add(1, Ordering::SeqCst) == 0;
        if !alone_at_start {
            OVERLAP_MARKS.fetch_add(1, Ordering::SeqCst);
        }
        Self {
            alone_at_start,
            marks_at_start,
            before,
        }
    }

    pub(crate) fn close(&self, after: Option<u64>) -> Attribution {
        attribute(
            self.alone_at_start,
            self.marks_at_start,
            OVERLAP_MARKS.load(Ordering::SeqCst),
            self.before,
            after,
        )
    }
}

impl Drop for SoloBuildWindow {
    fn drop(&mut self) {
        BUILDS_IN_FLIGHT.fetch_sub(1, Ordering::SeqCst);
    }
}

/// What this process has managed to attribute, and what it threw away.
///
/// The discard counters are not decoration: without them "unavailable" and "never attributable on this
/// machine" look identical from outside, and only the second is an argument for changing the design.
#[derive(Default, Debug, Clone)]
pub(crate) struct VramLedger {
    /// Bytes the dual embed session's build allocated. `None` until one is attributable.
    pub(crate) embed: Option<u64>,
    pub(crate) rerank: Option<u64>,
    pub(crate) discarded_overlaps: u64,
    pub(crate) discarded_not_sampled: u64,
    pub(crate) discarded_no_growth: u64,
}

impl VramLedger {
    pub(crate) fn record(&mut self, engine: &str, attribution: Attribution) {
        match attribution {
            Attribution::Measured(bytes) if engine == EMBED_CACHE_ENGINE => {
                self.embed = Some(bytes)
            }
            Attribution::Measured(bytes) if engine == RERANK_CACHE_ENGINE => {
                self.rerank = Some(bytes)
            }
            // An engine this ledger has no slot for is still a real build, and silently dropping it
            // would make the counters disagree with the number of builds that happened.
            Attribution::Measured(_) => self.discarded_not_sampled += 1,
            Attribution::Overlapped => self.discarded_overlaps += 1,
            Attribution::NotSampled => self.discarded_not_sampled += 1,
            Attribution::NoGrowth => self.discarded_no_growth += 1,
        }
    }

    /// Why there is nothing to show, when there is nothing to show. `None` once ANY figure exists —
    /// the per-engine `None`s are then explained by the counters beside them rather than by a sentence
    /// that would have to describe two engines at once.
    pub(crate) fn unavailable_reason(&self) -> Option<&'static str> {
        if self.embed.is_some() || self.rerank.is_some() {
            return None;
        }
        Some(match self {
            _ if self.discarded_not_sampled > 0 => {
                "no GPU memory sample is available on this build — the figure is DXGI's, so it needs \
                 Windows and a resolved adapter (MIGraphX/WSL and CUDA-on-Linux report nothing here)"
            }
            _ if self.discarded_overlaps > 0 => {
                "every build so far overlapped another, so no allocation could be attributed to a single \
                 engine — the delta would have measured the process, not the engine"
            }
            _ if self.discarded_no_growth > 0 => {
                "a build was sampled alone and the adapter reported no growth, which is evidence the \
                 allocation was invisible to this sampler rather than a measurement of zero"
            }
            _ => "no session has been built yet",
        })
    }
}

/// Samples the adapter the engines actually run on, or `None` when this process cannot resolve one.
///
/// Gated on a RESOLVED adapter rather than on the configured device id: with no resolution the sidecar
/// passes the raw id through to the EP, and an id that DXGI could not map is an id this sampler must not
/// pretend to understand either.
pub(crate) fn sample(state: &AppState) -> Option<u64> {
    adapters::process_vram_bytes(state.adapter.as_ref()?.dml_device_id)
}

/// Files one build's verdict, healing a poisoned ledger the way the rest of the bookkeeping does: one
/// panic costs one record, never every record until restart.
pub(crate) fn record(state: &AppState, engine: &str, attribution: Attribution) {
    match state.vram.lock() {
        Ok(mut ledger) => ledger.record(engine, attribution),
        Err(poisoned) => {
            tracing::warn!("vram ledger was poisoned by an earlier panic — healing it and recording {attribution:?}");
            poisoned.into_inner().record(engine, attribution);
            state.vram.clear_poison();
        }
    }
}

/// A non-blocking read for `/health`. A ledger busy under another thread reports nothing rather than
/// queueing the probe — the standing rule for every field on that endpoint.
pub(crate) fn snapshot(state: &AppState) -> Option<VramLedger> {
    state.vram.try_lock().ok().map(|ledger| ledger.clone())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::*;
    use std::sync::{Arc, Barrier};

    /// THE test this design exists for. A naive before/after delta passes every other case in this file
    /// and fails only here — two builds in flight at once, which on the DEFAULT (DirectML) flavour is not
    /// an exotic race but what happens on every restart where the host hits both endpoints.
    #[test]
    fn two_overlapping_builds_both_refuse_to_publish_and_are_counted_as_discards() {
        let both_open = Arc::new(Barrier::new(2));
        let verdicts: Vec<Attribution> = std::thread::scope(|scope| {
            let threads: Vec<_> = (0..2)
                .map(|engine| {
                    let both_open = Arc::clone(&both_open);
                    scope.spawn(move || {
                        // Each build "allocates" a plausible amount of its own.
                        let before = 1_000 * (engine + 1);
                        let window = SoloBuildWindow::open(Some(before));
                        both_open.wait(); // neither may close until both are in flight
                        window.close(Some(before + 500))
                    })
                })
                .collect();
            threads
                .into_iter()
                .map(|t| t.join().expect("the build thread"))
                .collect()
        });

        assert_eq!(
            verdicts,
            vec![Attribution::Overlapped, Attribution::Overlapped],
            "a delta taken while another build allocates measures the process, not the engine"
        );

        let mut ledger = VramLedger::default();
        for verdict in verdicts {
            ledger.record(EMBED_CACHE_ENGINE, verdict);
        }
        assert_eq!(
            ledger.discarded_overlaps, 2,
            "and the discards are counted, not swallowed"
        );
        assert_eq!(
            ledger.embed, None,
            "nothing is published from an unattributable window"
        );
    }

    /// A build that really was alone publishes its delta — otherwise the honesty above would just be a
    /// field that is always absent.
    #[test]
    fn a_build_that_was_alone_publishes_its_delta() {
        let window = SoloBuildWindow::open(Some(4_000_000_000));
        let verdict = window.close(Some(6_500_000_000));

        assert_eq!(verdict, Attribution::Measured(2_500_000_000));

        let mut ledger = VramLedger::default();
        ledger.record(EMBED_CACHE_ENGINE, verdict);
        assert_eq!(ledger.embed, Some(2_500_000_000));
        assert_eq!(
            ledger.unavailable_reason(),
            None,
            "a ledger with a figure explains nothing away"
        );
    }

    /// Two builds that did not overlap in TIME are two attributable measurements, however close together
    /// they ran. The overlap mark must not leak from one closed window into the next.
    #[test]
    fn builds_that_merely_follow_each_other_stay_attributable() {
        let first = SoloBuildWindow::open(Some(0)).close(Some(10));
        let second = SoloBuildWindow::open(Some(10)).close(Some(35));

        assert_eq!(
            (first, second),
            (Attribution::Measured(10), Attribution::Measured(25))
        );
    }

    /// Absent is not zero, and the four ways of being absent are told apart. A `0` here would read as
    /// "this engine costs nothing", which is the one thing the number can never mean.
    #[test]
    fn an_unsampled_or_shrinking_window_reports_absent_with_a_reason_never_zero() {
        assert_eq!(
            attribute(true, 0, 0, None, Some(9)),
            Attribution::NotSampled,
            "no `before` sample"
        );
        assert_eq!(
            attribute(true, 0, 0, Some(9), None),
            Attribution::NotSampled,
            "no `after` sample"
        );
        assert_eq!(
            attribute(true, 0, 0, Some(9), Some(9)),
            Attribution::NoGrowth,
            "flat is not a measurement"
        );
        assert_eq!(
            attribute(true, 0, 0, Some(9), Some(4)),
            Attribution::NoGrowth,
            "and neither is shrinking"
        );
        assert_eq!(
            attribute(false, 0, 0, Some(1), Some(9)),
            Attribution::Overlapped,
            "joined at the start"
        );
        assert_eq!(
            attribute(true, 3, 4, Some(1), Some(9)),
            Attribution::Overlapped,
            "joined part-way through"
        );

        for (verdict, expected) in [
            (Attribution::NotSampled, "needs Windows"),
            (Attribution::Overlapped, "attributed to a single"),
            (Attribution::NoGrowth, "measurement of zero"),
        ] {
            let mut ledger = VramLedger::default();
            ledger.record(EMBED_CACHE_ENGINE, verdict);
            let reason = ledger
                .unavailable_reason()
                .expect("an empty ledger owes a reason");
            assert!(
                reason.contains(expected),
                "{verdict:?} must be explained as itself: {reason}"
            );
            assert_eq!(
                ledger.embed, None,
                "and must never publish a 0 in place of the missing figure"
            );
        }

        assert_eq!(
            VramLedger::default().unavailable_reason(),
            Some("no session has been built yet"),
            "a fresh process has not failed to measure anything — it has not measured anything"
        );
    }

    /// The two engines are separate slots: the reranker's build must never be reported as the embedder's.
    #[test]
    fn each_engine_gets_its_own_figure() {
        let mut ledger = VramLedger::default();

        ledger.record(EMBED_CACHE_ENGINE, Attribution::Measured(7));
        ledger.record(RERANK_CACHE_ENGINE, Attribution::Measured(3));

        assert_eq!((ledger.embed, ledger.rerank), (Some(7), Some(3)));
    }

    /// Without a resolved adapter there is nothing to sample, and the sampler says so instead of
    /// guessing an index. Every non-Windows build is permanently in this state, by design.
    #[test]
    fn a_state_with_no_resolved_adapter_never_samples() {
        assert_eq!(
            sample(&app_state()),
            None,
            "no adapter resolved -> no sample, never a 0"
        );
    }
}
