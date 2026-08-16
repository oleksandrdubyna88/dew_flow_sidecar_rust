use std::time::Instant;
use crate::engine_cache::{RungCache};
use crate::state::{AppState};

// ---------- model loading ----------

/// Records the sequence cap this request runs at, so `cap_for` can keep a query on the rung a pass is
/// using and /health can report it.
///
/// It used to EVICT both embedding engines whenever the cap changed, because `max_length` is baked into
/// an ort session at build time. That was the right shape for one engine slot and the wrong price for a
/// two-rung ladder: a pass crosses the boundary twice and each crossing cost 156-173 s of rebuild (see
/// `RungCache`). The engines are now kept per rung, so a change is a lookup and this records, nothing more.
pub(crate) fn record_embed_max_length(state: &AppState, requested: usize) {
    let Ok(mut loaded) = state.loaded_embed_max_length.lock() else {
        return; // poisoned bookkeeping: keep serving at whatever is loaded rather than failing the embed
    };
    *loaded = Some(requested);
}

/// Same bookkeeping for the BATCH, so /health can report what ran rather than what was configured.
pub(crate) fn record_max_batch(state: &AppState, used: usize) {
    let Ok(mut loaded) = state.loaded_max_batch.lock() else {
        return;
    };
    *loaded = Some(used);
}

/// The width of the vectors that just came back, so `/models` can state it instead of a caller having
/// to already know it. Recorded from a real row — see `loaded_embed_dimension`.
pub(crate) fn record_embed_dimension(state: &AppState, dimension: Option<usize>) {
    let Some(dimension) = dimension else {
        return; // an empty batch measured nothing; leave whatever an earlier pass established
    };
    let Ok(mut loaded) = state.loaded_embed_dimension.lock() else {
        return;
    };
    *loaded = Some(dimension);
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
pub(crate) fn remember_engine<T: Send + 'static>(cache: &mut RungCache<T>, what: &str, cap: usize, engine: T) {
    let capacity = cache.capacity;
    let Some((rung, evicted)) = cache.insert(cap, engine) else {
        tracing::info!("{what}: built at cap {cap} — resident rung(s): {:?}", cache.caps());
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
    let spawned = std::thread::Builder::new().name("engine-teardown".to_string()).spawn(move || {
        let started = Instant::now();
        drop(engine);
        tracing::info!("{names_it}: torn down in {:.1}s, off the engine lock", started.elapsed().as_secs_f32());
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
    
    
    
    
    
    
    
    

    // ---------- the dimension on /embed ----------
    /// The reported width is the width of the rows in the SAME response — the whole point is that a caller
    /// holding the response cannot get it wrong.
    #[test]
    fn the_reported_dimension_is_the_width_of_the_rows_beside_it() {
        let rows = vec![vec![0.0f32; 1024], vec![0.0f32; 1024]];

        assert_eq!(dense_dimension(&rows), Some(rows[0].len()));
        assert_eq!(dense_dimension(&[]), None, "an empty batch measured nothing — null, never 0");
    }
}
