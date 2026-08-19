use crate::config::Config;
use crate::state::AppState;
use crate::wire::TokenUsage;
use std::path::{Path, PathBuf};

/// BGE-M3's tokenizer — the one `/embed`'s own truncation accounting counts with.
pub(crate) const BGE_TOKENIZER: &str = "bge";
/// Qwen's, for the SEMANTIC channel this sidecar never embeds. Counting only.
pub(crate) const QWEN_TOKENIZER: &str = "qwen";

/// One row of the registry AS DECLARED, before anything is read from disk.
///
/// `consequence` is why this is a struct and not a pair. A missing tokenizer costs different things for
/// different names — bge's absence turns `/embed`'s truncation accounting off, qwen's only makes one
/// `/tokenize` name unavailable — and one generic warning would have made the two read identically in
/// the log, which is the only place an operator ever meets them.
pub(crate) struct TokenizerSource {
    pub(crate) name: &'static str,
    /// `None` = discovery found no file to even try, which is a different fact from "a path that would
    /// not parse" and is logged as one.
    pub(crate) path: Option<PathBuf>,
    pub(crate) consequence: &'static str,
}

/// A tokenizer name this build answers for, resolved.
pub(crate) struct TokenizerEntry {
    pub(crate) name: &'static str,
    /// The file it was actually loaded from. Recorded rather than inferred: a count whose tokenizer
    /// nobody can name afterwards is a count nobody can reproduce.
    pub(crate) source: Option<PathBuf>,
    /// `None` = registered, but not loadable in THIS deployment. On the wire that is a 200 with
    /// `available: false`, never a 400 — the caller's name was right and the machine is what is missing.
    pub(crate) tokenizer: Option<tokenizers::Tokenizer>,
}

/// Every tokenizer this build can count with.
///
/// A table, not a `match`. The two names used to be two hand-written fields resolved by a two-arm string
/// match, so a third model was a code change in three places and a caller could not discover what this
/// build counts for. Here a model is a row and a path, and the refusal for an unknown name can NAME the
/// alternatives instead of repeating a sentence somebody has to remember to edit.
pub(crate) struct TokenizerRegistry {
    pub(crate) entries: Vec<TokenizerEntry>,
}

impl TokenizerRegistry {
    /// Declares this build's rows and loads every one of them NOW.
    ///
    /// Eager on purpose (`research/PLAN_reliability_tail.md` item 2): the counters used to be `OnceLock`s
    /// filled by `get_or_init` inside the FIRST `/tokenize` call, so that request paid a directory walk
    /// and a multi-MB parse on the async runtime — and a model cache swapped under a running sidecar
    /// silently decided the answer.
    pub(crate) fn load(config: &Config) -> Self {
        Self::from_sources(vec![
            TokenizerSource {
                name: BGE_TOKENIZER,
                // The snapshot folder is a content hash that changes when the model is re-pulled, so the
                // path is discovered rather than hardcoded.
                path: find_tokenizer_file(&config.cache_dir),
                consequence:
                    "/embed will report token_accounting: false, and the host cannot then prove \
                              that no input was silently truncated",
            },
            TokenizerSource {
                name: QWEN_TOKENIZER,
                path: Some(config.qwen_tokenizer_path.clone()),
                consequence:
                    "/tokenize will report that name unavailable rather than estimate a count",
            },
        ])
    }

    pub(crate) fn from_sources(sources: Vec<TokenizerSource>) -> Self {
        Self {
            entries: sources.into_iter().map(load_tokenizer_row).collect(),
        }
    }

    /// The row for a name — whether or not it could be loaded. `None` means the name is not registered
    /// here at all, which is the only case that is the CALLER's mistake.
    pub(crate) fn entry(&self, name: &str) -> Option<&TokenizerEntry> {
        self.entries.iter().find(|entry| entry.name == name)
    }

    /// Every registered name, in declaration order, quoted — what an unknown-name refusal reports so the
    /// caller can correct itself rather than guess.
    pub(crate) fn names(&self) -> String {
        self.entries
            .iter()
            .map(|entry| format!("'{}'", entry.name))
            .collect::<Vec<_>>()
            .join(", ")
    }

    /// Every row with its file and whether it can answer — the startup summary.
    ///
    /// Naming the SOURCE is the point rather than decoration: a count is reproducible only if the file
    /// behind it can be named afterwards, and a silently updated `tokenizer.json` changes every count
    /// without changing anything a consumer can see. `GET /models`
    /// (`todo/PLAN_tokenizer_registry.md` §3.2) will report these same facts on the wire.
    pub(crate) fn describe(&self) -> String {
        self.entries
            .iter()
            .map(|entry| match (&entry.source, entry.tokenizer.is_some()) {
                (Some(path), true) => format!("'{}' <- {}", entry.name, path.display()),
                (Some(path), false) => format!(
                    "'{}' UNAVAILABLE (would not parse: {})",
                    entry.name,
                    path.display()
                ),
                (None, _) => format!("'{}' UNAVAILABLE (no file found)", entry.name),
            })
            .collect::<Vec<_>>()
            .join(", ")
    }
}

/// Reads one declared row.
///
/// Best-effort throughout: a name whose file is absent or unparseable is registered-but-unavailable,
/// never a startup failure. A sidecar that cannot COUNT must still EMBED — the missing tokenizer has to
/// stop us CLAIMING nothing was truncated, not stop us serving.
pub(crate) fn load_tokenizer_row(source: TokenizerSource) -> TokenizerEntry {
    let TokenizerSource {
        name,
        path,
        consequence,
    } = source;
    let Some(path) = path.filter(|p| p.is_file()) else {
        tracing::warn!("no {name} tokenizer file found — {consequence}");
        return TokenizerEntry {
            name,
            source: None,
            tokenizer: None,
        };
    };
    match tokenizers::Tokenizer::from_file(&path) {
        Ok(tokenizer) => {
            tracing::info!("{name} token counting enabled from `{}`", path.display());
            TokenizerEntry {
                name,
                source: Some(path),
                tokenizer: Some(tokenizer),
            }
        }
        Err(e) => {
            tracing::warn!(
                "{name} tokenizer at `{}` would not parse ({e}) — {consequence}",
                path.display()
            );
            TokenizerEntry {
                name,
                source: Some(path),
                tokenizer: None,
            }
        }
    }
}

/// Finds `tokenizer.json` for BGE-M3 inside the HuggingFace-layout model cache. The snapshot folder is
/// a content hash that changes when the model is re-pulled, so it is discovered rather than hardcoded.
pub(crate) fn find_tokenizer_file(cache_dir: &Path) -> Option<PathBuf> {
    let snapshots = cache_dir.join("models--BAAI--bge-m3").join("snapshots");
    let entries = std::fs::read_dir(snapshots).ok()?;
    entries
        .filter_map(|e| e.ok())
        .map(|e| e.path().join("tokenizer.json"))
        .find(|p| p.is_file())
}

/// Counts what each text really costs and flags the ones whose tail the cap will discard.
///
/// The count is taken with truncation OFF on purpose: a truncating tokenizer reports `max_length` for
/// every over-long text, which is precisely the information that hides the problem. Special tokens are
/// included because they occupy the same window the content competes for.
pub(crate) fn token_usage(state: &AppState, texts: &[String], max_length: usize) -> TokenUsage {
    let Some(tokenizer) = state.token_counter() else {
        return TokenUsage {
            max_length,
            ..TokenUsage::default()
        };
    };
    usage_from_counts(count_tokens(tokenizer, texts, "embed"), max_length)
}

/// Counts each text, keeping a REFUSAL as `None` instead of folding it to `0` — and saying so in the
/// log.
///
/// The fold used to be `.map(|e| e.len()).unwrap_or(0)`, silently, in both places that count. A text
/// the tokenizer refused was then reported as 0 tokens and `truncated: false` — "measured, and
/// definitely not truncated", which is the exact inversion of the only guarantee this accounting
/// exists to give. The host owns no tokenizer, so it has no second opinion to check that against.
pub(crate) fn count_tokens(
    tokenizer: &tokenizers::Tokenizer,
    texts: &[String],
    what: &str,
) -> Vec<Option<usize>> {
    texts
        .iter()
        .map(|text| match tokenizer.encode(text.as_str(), true) {
            Ok(encoded) => Some(encoded.len()),
            Err(e) => {
                tracing::warn!(
                    "{what}: the tokenizer refused a text of {} char(s) ({e}) — reporting the count as UNKNOWN \
                     rather than as zero, which a caller reads as proof that nothing was truncated",
                    text.len()
                );
                None
            }
        })
        .collect()
}

/// Folds per-text counts into the wire shape, keeping UNKNOWN distinct from CLEAN.
///
/// One refusal turns accounting off for the WHOLE response, because that is the distinction this file
/// already models one level up (`token_accounting: false` = NOT MEASURED, both arrays empty) and the
/// only one the host understands. A per-text hole would have to be invented on the wire, and every
/// caller that did not learn about it would read the hole as a zero — the defect again, with an extra
/// field. Conservative in the right direction: the batch reads as unmeasured, never as proven clean.
pub(crate) fn usage_from_counts(counts: Vec<Option<usize>>, max_length: usize) -> TokenUsage {
    let Some(measured) = counts.into_iter().collect::<Option<Vec<usize>>>() else {
        return TokenUsage {
            max_length,
            ..TokenUsage::default()
        };
    };
    TokenUsage {
        truncated: measured.iter().map(|&n| n > max_length).collect(),
        token_count: measured,
        max_length,
        token_accounting: true,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::testing::*;

    // ---------- token accounting ----------
    // The snapshot folder is a content hash that changes whenever the model is re-pulled, so the path
    // has to be discovered. A hardcoded one would silently stop resolving after an update — and a
    // tokenizer that cannot be found turns the guard off without anyone noticing.
    #[test]
    fn the_tokenizer_is_found_by_scanning_the_snapshot_folder() {
        let root = std::env::temp_dir().join(format!("bge-tok-{}", std::process::id()));
        let snapshot = root
            .join("models--BAAI--bge-m3")
            .join("snapshots")
            .join("deadbeef");
        std::fs::create_dir_all(&snapshot).unwrap();
        std::fs::write(snapshot.join("tokenizer.json"), b"{}").unwrap();

        let found = find_tokenizer_file(&root).expect("the snapshot holds a tokenizer.json");

        assert_eq!(found, snapshot.join("tokenizer.json"));
        std::fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn a_cache_without_the_model_yields_no_tokenizer() {
        let empty = std::env::temp_dir().join(format!("bge-tok-empty-{}", std::process::id()));
        std::fs::create_dir_all(&empty).unwrap();

        assert!(find_tokenizer_file(&empty).is_none());
        // ...and the registry degrades to "off" rather than failing the sidecar: a missing tokenizer must
        // stop us CLAIMING nothing was truncated, never stop us embedding. The row still EXISTS — the
        // name stays registered so /tokenize answers `available: false` instead of "no such tokenizer".
        let mut config = config("");
        config.cache_dir = empty.clone();
        let registry = TokenizerRegistry::load(&config);
        let bge = registry
            .entry(BGE_TOKENIZER)
            .expect("the row is declared even with no file behind it");
        assert!(bge.tokenizer.is_none(), "nothing loaded");
        assert!(bge.source.is_none(), "and there was no file to name");
        std::fs::remove_dir_all(&empty).ok();
    }

    /// The whole point of token accounting is a TRUSTWORTHY truncation signal, so an encode failure
    /// folded to `0` tokens is the exact inversion of the contract: it reads as "measured, and
    /// definitely not truncated". Unknown must stay unknown.
    #[test]
    fn an_unencodable_text_is_reported_as_unknown_rather_than_zero_tokens() {
        let usage = usage_from_counts(vec![Some(10), None, Some(300)], 256);

        assert!(
            !usage.token_accounting,
            "one refusal makes the whole answer UNMEASURED"
        );
        assert!(
            usage.token_count.is_empty() && usage.truncated.is_empty(),
            "and empties the arrays with it"
        );
        assert_eq!(
            usage.max_length, 256,
            "the cap they would have been judged against still travels"
        );

        // The refuted approach, reproduced so the defect stays visible in the suite: the shipped fold was
        // `.map(|e| e.len()).unwrap_or(0)`, which reported the refused text as 0 tokens and NOT truncated —
        // "measured, and definitely nothing lost", from the one signal the host cannot compute itself.
        let folded: Vec<usize> = [Ok(10usize), Err("refused"), Ok(300usize)]
            .into_iter()
            .map(|encoded| encoded.unwrap_or(0))
            .collect();
        assert_eq!(
            (folded[1], folded[1] > 256),
            (0, false),
            "which is exactly what it claimed"
        );

        // A clean batch still measures, truncation flags and all.
        let clean = usage_from_counts(vec![Some(10), Some(300)], 256);
        assert!(clean.token_accounting);
        assert_eq!(clean.truncated, vec![false, true]);
    }

    /// The pre-warm guarantee, stated as something observable: a tokenizer that was on disk when the
    /// process started keeps counting even after the file goes away.
    ///
    /// It is the only way to assert "loaded at startup" without watching syscalls, and the symptom it
    /// pins is real — the counter used to be an `OnceLock` filled by `get_or_init` inside the FIRST
    /// /tokenize call, so the first request of a run paid a directory walk and a file parse on the async
    /// runtime (PLAN_reliability_tail item 2), and a cache swapped underneath a running sidecar decided
    /// the answer.
    #[test]
    fn a_tokenizer_present_at_startup_still_counts_after_its_file_is_gone() {
        let cache = model_cache_with_a_tokenizer("startup");
        let mut config = config("");
        config.cache_dir = cache.clone();

        let state = app_state_with(config);
        // The file is what a lazy loader would need at request time. It is gone before the first ask.
        std::fs::remove_dir_all(&cache).expect("take the file away");

        assert!(
            state.token_counter().is_some(),
            "the tokenizer was on disk at startup, so this run counts — a loader that reads on first use \
             instead answers None here, and the sidecar silently stops accounting for truncation"
        );
    }
}
