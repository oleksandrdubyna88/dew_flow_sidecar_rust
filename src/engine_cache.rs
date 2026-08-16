
/// Engines that have been built, keyed by the sequence cap baked into them.
///
/// `max_length` is compiled into an ort session AND into the EP's program, so it used to be a reason to
/// EVICT: a cap change dropped both embedding engines and the next request rebuilt them. Measured
/// 2026-07-30, that cost 156-173 s per crossing — of which only ~13 s was session building, the rest
/// being MIGraphX materialising its ~2.4 GB compiled program at the FIRST `session.run`, which no
/// amount of eager loading can move. And a Fast pass crosses the boundary TWICE (it walks the ladder
/// down, then the next pass starts at the ceiling again), so the toll was ~5.5 min per pass, forever.
/// Keeping one engine per rung turns a crossing into a lookup.
///
/// An insertion-ordered `Vec` rather than a `HashMap`: the ladder has at most two rungs
/// (`SidecarRungPlan`), and the order IS the eviction policy — least-recently-used first, so a map
/// would need a second structure to carry it.
pub(crate) struct RungCache<T> {
    pub(crate) capacity: usize,
    pub(crate) rungs: Vec<(usize, T)>,
}

impl<T> RungCache<T> {
    /// `capacity` is the operator's `EMBED_ENGINE_CACHE_RUNGS`; 1 reproduces the pre-cache behaviour
    /// exactly (every cap change evicts), which is the escape hatch if the VRAM budget ever demands it.
    pub(crate) fn new(capacity: usize) -> Self {
        Self { capacity: capacity.max(1), rungs: Vec::new() }
    }

    /// The engine built for this cap, if any — marking it most-recently-used, so eviction always takes
    /// the rung the pass is LEAST likely to come back to.
    pub(crate) fn get_mut(&mut self, cap: usize) -> Option<&mut T> {
        let at = self.rungs.iter().position(|(resident, _)| *resident == cap)?;
        let entry = self.rungs.remove(at);
        self.rungs.push(entry);
        self.rungs.last_mut().map(|(_, engine)| engine)
    }

    /// Stores a freshly built engine as most-recently-used and returns whatever had to make room for
    /// it. The caller drops the evicted engine — ort session teardown is not instant and the caller
    /// already knows whether it is on a blocking thread.
    pub(crate) fn insert(&mut self, cap: usize, engine: T) -> Option<(usize, T)> {
        self.rungs.retain(|(resident, _)| *resident != cap);
        self.rungs.push((cap, engine));
        (self.rungs.len() > self.capacity).then(|| self.rungs.remove(0))
    }

    /// Every resident engine, emptying the cache — what `/unload` hands to the GPU lease.
    pub(crate) fn drain(&mut self) -> Vec<(usize, T)> {
        std::mem::take(&mut self.rungs)
    }

    /// ONE rung's engine, if resident — the partial `/unload`'s per-rung eviction (the host's
    /// budget-aware planner drops the largest unnecessary rung and keeps the rest warm).
    pub(crate) fn remove(&mut self, cap: usize) -> Option<T> {
        let at = self.rungs.iter().position(|(resident, _)| *resident == cap)?;
        Some(self.rungs.remove(at).1)
    }

    /// The caps currently resident, least-recently-used first. Reported by `/health` and logged on
    /// every build so the occupancy is observable rather than inferred.
    pub(crate) fn caps(&self) -> Vec<usize> {
        self.rungs.iter().map(|(cap, _)| *cap).collect()
    }
}

/// What an engine slot can be asked, independently of whether it holds one engine (rerank — a single
/// fixed cap) or one per rung (dense/sparse). Exists so `loaded_now` and `lock_healing` keep serving
/// both shapes without either growing a branch.
pub(crate) trait EngineSlot {
    /// Whether anything is loaded at all — `/health`'s per-model boolean.
    fn is_loaded(&self) -> bool;

    /// Drop everything, because a panic left state we cannot vouch for. See `lock_healing`.
    fn discard_all(&mut self);
}

impl<T> EngineSlot for Option<T> {
    fn is_loaded(&self) -> bool {
        self.is_some()
    }

    fn discard_all(&mut self) {
        *self = None;
    }
}

impl<T> EngineSlot for RungCache<T> {
    fn is_loaded(&self) -> bool {
        !self.rungs.is_empty()
    }

    fn discard_all(&mut self) {
        self.rungs.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    use std::sync::mpsc;
    use std::time::Duration;
    
    
    
    use crate::testing::*;
    use crate::bookkeeping::{remember_engine};
    

    /// The whole point of the cache: a rung already built is HANDED BACK, not rebuilt. Rebuilding cost
    /// 156-173 s measured, because MIGraphX re-materialises its ~2.4 GB program on the first run.
    #[test]
    fn a_rung_already_built_is_returned_rather_than_rebuilt() {
        let mut cache = cache_of(2, &[(256, 7)]);

        assert_eq!(cache.get_mut(256).copied(), Some(7), "the built engine comes back");
        assert_eq!(cache.get_mut(1024), None, "a rung never built is a miss, not a wrong engine");
        assert_eq!(cache.caps(), vec![256], "a miss builds nothing on its own");
    }

    /// THE regression this cache exists for. A Fast pass walks the ladder DOWN (ceiling first), ends on
    /// the low rung, and the next pass starts at the ceiling again — so the boundary is crossed twice per
    /// pass. Before the cache each crossing evicted both engines: ~5.5 min per pass, forever. At a
    /// capacity of 1 this test goes red on the last assertion, which is exactly the old behaviour.
    #[test]
    fn walking_the_ladder_down_and_back_up_evicts_nothing() {
        let mut cache = cache_of(2, &[(1024, 10), (256, 20)]);

        assert_eq!(cache.get_mut(1024).copied(), Some(10), "the ceiling survived the step down");
        assert_eq!(cache.get_mut(256).copied(), Some(20), "and the low rung survived the step back up");
        assert_eq!(cache.caps().len(), 2, "a two-rung ladder never evicts at capacity 2");
    }

    /// The escape hatch: EMBED_ENGINE_CACHE_RUNGS=1 must reproduce the pre-cache behaviour exactly, so a
    /// VRAM budget that cannot hold two pairs has somewhere to go without a code change.
    #[test]
    fn capacity_one_reproduces_the_evicting_behaviour() {
        let mut cache = cache_of(1, &[(1024, 10), (256, 20)]);

        assert_eq!(cache.caps(), vec![256], "the newcomer displaced the previous rung");
        assert_eq!(cache.get_mut(1024), None, "stepping back up rebuilds, exactly as before the cache");
    }

    /// Eviction order decides whether the cache helps or hurts: dropping the OLDEST would throw away the
    /// rung the pass is actively using and keep one it has moved on from. Least-recently-USED is the rule.
    #[test]
    fn a_third_rung_evicts_the_least_recently_used_not_the_oldest() {
        let mut cache = cache_of(2, &[(1024, 10), (256, 20)]);
        cache.get_mut(1024).expect("1024 is resident and now the most recently used");

        cache.insert(512, 30);

        assert_eq!(cache.caps(), vec![1024, 512], "256 went — it was the least recently USED");
        assert_eq!(cache.get_mut(1024).copied(), Some(10), "the rung in active use survived");
    }

    /// Rebuilding a rung that is already resident must REPLACE it, never leave two entries for one cap —
    /// a duplicate would hold a second session's worth of VRAM that nothing can ever hand back.
    #[test]
    fn rebuilding_a_resident_rung_replaces_it_instead_of_duplicating() {
        let mut cache = cache_of(2, &[(256, 1)]);

        cache.insert(256, 2);

        assert_eq!(cache.caps(), vec![256], "one entry per cap");
        assert_eq!(cache.get_mut(256).copied(), Some(2), "the newest build wins");
    }

    /// The partial unload's per-rung eviction: only the NAMED rung goes, the rest keep their order,
    /// and a non-resident cap is a no-op — the host may name a rung another eviction already took.
    #[test]
    fn remove_drops_only_the_named_rung() {
        let mut cache: RungCache<u8> = RungCache::new(2);
        cache.insert(1024, 1);
        cache.insert(256, 2);

        assert_eq!(cache.remove(1024), Some(1));
        assert_eq!(cache.caps(), vec![256], "the other rung stays resident");
        assert_eq!(cache.remove(512), None, "a non-resident cap is a no-op");
        assert_eq!(cache.caps(), vec![256]);
    }

    // ---------- teardown must leave the critical path ----------
    /// The evicted engine must not be torn down on the thread holding the engine mutex.
    ///
    /// ort teardown is not instant, every queued request waits it out, and the wait then lands in the
    /// NEXT caller's `queue_wait_ms` — the field the README introduced precisely to stop misattributing
    /// waiting. `RungCache::insert` hands the engine back so the CALLER can choose where it dies, and
    /// `remember_engine` used to choose the worst available place. At the shipped default
    /// `EMBED_ENGINE_CACHE_RUNGS=1` this fires on every cap change.
    #[test]
    fn an_evicted_engine_is_dropped_outside_the_lock() {
        struct DropSpy(mpsc::Sender<std::thread::ThreadId>);
        impl Drop for DropSpy {
            fn drop(&mut self) {
                let _ = self.0.send(std::thread::current().id());
            }
        }

        let (tell, dropped) = mpsc::channel();
        let mut cache = RungCache::new(1);
        cache.insert(256, DropSpy(tell.clone()));

        remember_engine(&mut cache, "embed", 512, DropSpy(tell));

        let torn_down_on = dropped.recv_timeout(Duration::from_secs(5)).expect("the evicted engine is dropped");
        assert_ne!(
            torn_down_on,
            std::thread::current().id(),
            "the teardown ran on the thread that holds the engine mutex, so everything queued behind it waited"
        );
    }
}
