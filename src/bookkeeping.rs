use crate::engine_cache::RungCache;
use crate::state::AppState;
use std::sync::atomic::Ordering;
use std::sync::Mutex;
use std::time::Instant;

// ---------- model loading ----------

/// Writes one bookkeeping cell — and heals it, loudly, when a panic left it poisoned.
///
/// The three recorders below were three copies of the same six lines, each ending
/// `let Ok(..) = lock() else { return; }`: no log, no `clear_poison`. One poisoning therefore made
/// /health stop reporting that field **for the life of the process**, with nothing anywhere saying why —
/// and being three copies, the fix would have had to be made three times.
///
/// Healing matches what the engine mutexes have done since a panicked model load made every later
/// request answer "engine poisoned": one panic costs one operation, never all of them. It is safe here
/// for a reason the engines cannot claim — the cell is a `Option<usize>`, so a panic mid-write can leave
/// no half-built state, only a stale number this call is about to overwrite.
fn record(cell: &Mutex<Option<usize>>, what: &str, value: usize) {
    match cell.lock() {
        Ok(mut slot) => *slot = Some(value),
        Err(poisoned) => {
            tracing::warn!(
                "{what}: bookkeeping lock was poisoned by an earlier panic — healing it and recording \
                 {value}. /health would otherwise have stopped reporting this field until restart."
            );
            *poisoned.into_inner() = Some(value);
            cell.clear_poison();
        }
    }
}

/// The cap this process is committed to, or `None` when no embedding engine is resident or being built.
/// Lock-free by construction — see `AppState::committed_embed_cap` for why the read must not queue.
pub(crate) fn committed_cap(state: &AppState) -> Option<usize> {
    match state.committed_embed_cap.load(Ordering::Relaxed) {
        0 => None,
        cap => Some(cap),
    }
}

/// Declares the cap a build is about to materialise, so a `query` arriving mid-build inherits the cap
/// that build is aiming at instead of asking for one of its own and queueing to evict it.
///
/// Called under the engine lock, and every path that leaves the build MUST follow it with `settle_cap`
/// — a commitment a failed build left behind is exactly the stale intent this mirror replaced.
pub(crate) fn commit_cap(state: &AppState, cap: usize) {
    state.committed_embed_cap.store(cap, Ordering::Relaxed);
}

/// Re-derives the commitment from what the cache actually holds — the most recently used rung, or none.
///
/// The only writer that can ever LOWER the claim, and therefore the one every failure path owes:
/// a build that threw, an eviction, an `/unload`. Takes the cache by reference so the caller must be
/// holding it, which is what makes "mirror" true rather than hopeful.
pub(crate) fn settle_cap<T>(state: &AppState, cache: &RungCache<T>) {
    state
        .committed_embed_cap
        .store(cache.caps().last().copied().unwrap_or(0), Ordering::Relaxed);
}

/// Same bookkeeping for the BATCH, so /health can report what ran rather than what was configured.
pub(crate) fn record_max_batch(state: &AppState, used: usize) {
    record(&state.loaded_max_batch, "max_batch", used);
}

/// The width of the vectors that just came back, so `/models` can state it instead of a caller having
/// to already know it. Recorded from a real row — see `loaded_embed_dimension`.
pub(crate) fn record_embed_dimension(state: &AppState, dimension: Option<usize>) {
    // An empty batch measured nothing; leave whatever an earlier pass established.
    if let Some(dimension) = dimension {
        record(&state.loaded_embed_dimension, "embed dimension", dimension);
    }
}

/// The width of a set of dense vectors, read from a row.
///
/// `None` for an empty set — and that is the whole reason this is an `Option` rather than a `usize`:
/// a `0` is indistinguishable from "a zero-width vector", which is not a thing, and a caller sizing a
/// collection from it would size it wrong.
pub(crate) fn dense_dimension(dense: &[Vec<f32>]) -> Option<usize> {
    dense.first().map(Vec::len)
}

/// Files a freshly built engine under its rung and reports what the card now holds. The log line is the
/// operator's only window into the cache's occupancy — an eviction that happened silently would look
/// exactly like the rebuild-per-crossing behaviour this cache exists to remove.
pub(crate) fn remember_engine<T: Send + 'static>(
    cache: &mut RungCache<T>,
    what: &str,
    cap: usize,
    engine: T,
) {
    let capacity = cache.capacity;
    let Some((rung, evicted)) = cache.insert(cap, engine) else {
        tracing::info!(
            "{what}: built at cap {cap} — resident rung(s): {:?}",
            cache.caps()
        );
        return;
    };
    tracing::info!(
        "{what}: built at cap {cap}; rung {rung} evicted to stay within {capacity} — resident: {:?}",
        cache.caps()
    );
    teardown_off_the_lock(evicted, format!("{what} rung {rung}"));
}

/// Tears a dropped engine down on a thread of its own.
///
/// `RungCache::insert` hands the evicted engine back so the CALLER can choose where it dies — and this
/// caller holds the engine mutex every queued request is waiting on. An ort session teardown is not
/// instant: done inline it is paid by whoever is next in line, and it lands in their `queue_wait_ms`,
/// which is the field the README introduced to stop misattributing waiting. `/unload` has always dropped
/// outside its locks for exactly this reason (`drain_engines`); the cache eviction path did not, and at
/// the shipped `EMBED_ENGINE_CACHE_RUNGS=1` it fires on every cap change.
///
/// The completion line is not decoration. Off the lock, a slow teardown stops showing up as somebody
/// else's queue wait — so this becomes the only place that can say it was slow.
pub(crate) fn teardown_off_the_lock<T: Send + 'static>(engine: T, what: String) {
    let names_it = what.clone();
    let spawned = std::thread::Builder::new()
        .name("engine-teardown".to_string())
        .spawn(move || {
            let started = Instant::now();
            drop(engine);
            tracing::info!(
                "{names_it}: torn down in {:.1}s, off the engine lock",
                started.elapsed().as_secs_f32()
            );
        });
    // Spawn failure drops the closure — and the engine with it — right here. Slow beats leaked, but it
    // is the behaviour this function exists to avoid, so it is never silent.
    if let Err(e) = spawned {
        tracing::warn!(
            "{what}: no teardown thread could be spawned ({e}) — dropped inline instead, blocking whatever \
             is queued behind this engine"
        );
    }
}

/// The attention-score peak of ONE layer, in MB: `batch × 16 heads × seq² × 4 B`, doubled for the second
/// softmax buffer. Logged at startup so a misconfigured envelope is visible before the first embed.
pub(crate) fn attention_peak_mb(batch: usize, seq: usize) -> usize {
    batch * 16 * seq * seq * 4 * 2 / (1024 * 1024)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::app_state;
    use std::sync::Arc;

    // ---------- the dimension on /embed ----------
    /// The reported width is the width of the rows in the SAME response — the whole point is that a caller
    /// holding the response cannot get it wrong.
    #[test]
    fn the_reported_dimension_is_the_width_of_the_rows_beside_it() {
        let rows = vec![vec![0.0f32; 1024], vec![0.0f32; 1024]];

        assert_eq!(dense_dimension(&rows), Some(rows[0].len()));
        assert_eq!(
            dense_dimension(&[]),
            None,
            "an empty batch measured nothing — null, never 0"
        );
    }

    // ---------- poisoned bookkeeping heals instead of going quiet ----------

    /// A panic anywhere near these cells must cost one write, not every future one.
    ///
    /// They used to `let Ok(..) = lock() else { return }` with no log and no `clear_poison` — so a single
    /// poisoning made /health stop reporting that field **for the life of the process**, with nothing in
    /// the log saying why. The engine mutexes have healed poison since the day a panicked load made every
    /// later request answer "engine poisoned"; the bookkeeping beside them did not.
    #[test]
    fn a_poisoned_bookkeeping_cell_still_records_and_says_it_was_poisoned() {
        let state = app_state();
        let poisoner = Arc::clone(&state);
        std::thread::spawn(move || {
            let _held = poisoner.loaded_max_batch.lock().expect("fresh lock");
            panic!("a load panicked while holding the bookkeeping");
        })
        .join()
        .expect_err("the thread panicked, which is the point");
        assert!(
            state.loaded_max_batch.is_poisoned(),
            "the cell is poisoned before we record"
        );

        record_max_batch(&state, 64);

        assert_eq!(
            *state
                .loaded_max_batch
                .lock()
                .unwrap_or_else(|p| p.into_inner()),
            Some(64),
            "one panic costs one write, not every write from here on"
        );
        assert!(
            !state.loaded_max_batch.is_poisoned(),
            "and the cell was healed, as the engine locks are"
        );
    }
}
