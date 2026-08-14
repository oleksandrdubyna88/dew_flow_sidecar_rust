//! VENDORED ADDITION (clauderag): both BGE-M3 heads from ONE session and ONE forward pass.
//!
//! The official `BAAI/bge-m3` ONNX export returns two outputs per run: `sentence_embedding`
//! `[batch, hidden]` (CLS-pooled in-graph — what [`crate::TextEmbedding`] reads) and
//! `token_embeddings` `[batch, seq, hidden]` (what [`crate::SparseTextEmbedding`] feeds its lexical
//! head). Running the model through two separate sessions therefore computes every tensor twice and
//! keeps two copies of the 2.27 GB weights resident — see
//! research/PLAN_bge_sidecar_unified_session.md for the measured cost on the MIGraphX flavor.
//!
//! This type is deliberately assembled out of the SAME pieces the two single-head types use — the
//! tokenizer loader, `normalize`, `output_covers_input`, `post_process_bgem3` — so its results are
//! bit-identical to theirs by construction, not by tolerance. Both outputs are selected BY NAME:
//! MIGraphX orders session outputs differently than CPU/DirectML, and positional selection once fed
//! the pooled 2-D tensor to the sparse head, which panicked (`assertion failed: index < dim`) and
//! poisoned the engine mutex.

use anyhow::{Context, Result};
use ndarray::Array;
use ort::{session::Session, value::Value};
use tokenizers::Tokenizer;

use crate::common::{init_session_builder, load_tokenizer_hf_hub, normalize, pull_from_hf, Embedding};
use crate::models::text_embedding::EmbeddingModel;
use crate::sparse_text_embedding::output_covers_input;
use crate::{ExecutionProviderDispatch, SparseEmbedding, TextEmbedding};

/// Default batch size, matching the single-head types.
const DEFAULT_BATCH_SIZE: usize = 256;

/// Init options mirroring the subset of [`crate::TextInitOptions`] the sidecar actually sets.
pub struct Bgem3DualInitOptions {
    pub max_length: usize,
    pub cache_dir: std::path::PathBuf,
    pub execution_providers: Vec<ExecutionProviderDispatch>,
    pub intra_threads: Option<usize>,
    pub show_download_progress: bool,
}

impl Default for Bgem3DualInitOptions {
    fn default() -> Self {
        Self {
            max_length: 512,
            cache_dir: crate::common::get_cache_dir().into(),
            execution_providers: Vec::new(),
            intra_threads: None,
            show_download_progress: false,
        }
    }
}

impl Bgem3DualInitOptions {
    pub fn with_max_length(mut self, max_length: usize) -> Self {
        self.max_length = max_length;
        self
    }

    pub fn with_cache_dir(mut self, cache_dir: std::path::PathBuf) -> Self {
        self.cache_dir = cache_dir;
        self
    }

    pub fn with_execution_providers(
        mut self,
        execution_providers: Vec<ExecutionProviderDispatch>,
    ) -> Self {
        self.execution_providers = execution_providers;
        self
    }

    pub fn with_intra_threads(mut self, intra_threads: usize) -> Self {
        self.intra_threads = Some(intra_threads);
        self
    }
}

/// One BGE-M3 session serving both heads. See the module docs.
pub struct Bgem3DualEmbedding {
    tokenizer: Tokenizer,
    session: Session,
    need_token_type_ids: bool,
}

impl Bgem3DualEmbedding {
    /// Loads the SAME model files [`crate::TextEmbedding`] uses for [`EmbeddingModel::BGEM3`]
    /// (`onnx/model.onnx` + `model.onnx_data`) into a single session.
    pub fn try_new(options: Bgem3DualInitOptions) -> Result<Self> {
        let model_info = TextEmbedding::get_model_info(&EmbeddingModel::BGEM3)?;
        let model_repo = pull_from_hf(
            model_info.model_code.clone(),
            options.cache_dir,
            options.show_download_progress,
        )?;

        let model_file = model_repo
            .get(&model_info.model_file)
            .context(format!("Failed to retrieve {}", model_info.model_file))?;
        for file in &model_info.additional_files {
            model_repo.get(file).context(format!("Failed to retrieve {file}"))?;
        }

        let session = init_session_builder(options.execution_providers, options.intra_threads)?
            .commit_from_file(model_file)?;
        let tokenizer = load_tokenizer_hf_hub(model_repo, options.max_length)?;
        let need_token_type_ids =
            session.inputs().iter().any(|input| input.name() == "token_type_ids");
        Ok(Self { tokenizer, session, need_token_type_ids })
    }

    /// Embeds `texts` once and returns `(dense, sparse)` PER TEXT, zipped. Zipped on purpose: the
    /// sidecar's settling retry and unpin guard treat a row as a unit, and a short batch (the
    /// MIGraphX first-run quirk) must shorten both heads together, never one of them.
    pub fn embed<S: AsRef<str> + Send + Sync>(
        &mut self,
        texts: impl AsRef<[S]>,
        batch_size: Option<usize>,
    ) -> Result<Vec<(Embedding, SparseEmbedding)>> {
        let texts = texts.as_ref();
        let batch_size = batch_size.unwrap_or(DEFAULT_BATCH_SIZE);
        anyhow::ensure!(batch_size > 0, "batch_size must be greater than 0");

        let mut rows = Vec::with_capacity(texts.len());
        for batch in texts.chunks(batch_size) {
            rows.extend(self.embed_batch(batch)?);
        }
        Ok(rows)
    }

    /// One tokenize + one `session.run` + both post-processings. The tokenization block is the same
    /// code both single-head types run (they are identical today); the post-processing is CALLED,
    /// not copied — parity with the two-session path is the whole point.
    fn embed_batch<S: AsRef<str> + Send + Sync>(
        &mut self,
        batch: &[S],
    ) -> Result<Vec<(Embedding, SparseEmbedding)>> {
        let inputs = batch.iter().map(|text| text.as_ref()).collect();
        let encodings = self
            .tokenizer
            .encode_batch(inputs, true)
            .map_err(|e| anyhow::Error::msg(e.to_string()).context("Failed to encode the batch."))?;

        let encoding_length = encodings
            .first()
            .ok_or_else(|| anyhow::anyhow!("Tokenizer returned empty encodings"))?
            .len();
        let batch_size = batch.len();
        let max_size = encoding_length * batch_size;

        let mut ids_array = Vec::with_capacity(max_size);
        let mut mask_array = Vec::with_capacity(max_size);
        let mut type_ids_array = Vec::with_capacity(max_size);
        for encoding in &encodings {
            ids_array.extend(encoding.get_ids().iter().map(|x| *x as i64));
            mask_array.extend(encoding.get_attention_mask().iter().map(|x| *x as i64));
            type_ids_array.extend(encoding.get_type_ids().iter().map(|x| *x as i64));
        }

        let input_ids = Array::from_shape_vec((batch_size, encoding_length), ids_array)?;
        let attention_mask = Array::from_shape_vec((batch_size, encoding_length), mask_array)?;
        let token_type_ids = Array::from_shape_vec((batch_size, encoding_length), type_ids_array)?;

        let mut session_inputs = ort::inputs![
            "input_ids" => Value::from_array(input_ids.clone())?,
            "attention_mask" => Value::from_array(attention_mask.clone())?,
        ];
        if self.need_token_type_ids {
            session_inputs.push(("token_type_ids".into(), Value::from_array(token_type_ids)?.into()));
        }

        let outputs = self.session.run(session_inputs)?;
        // (name, rank) of every output, materialised once — selection needs several passes over it.
        let named_ranks: Vec<(String, Option<usize>)> = outputs
            .keys()
            .map(|k| (k.to_string(), tensor_rank(&outputs[k])))
            .collect();

        // Dense head: `sentence_embedding` by name (the tensor TextEmbedding's precedence list lands
        // on), fallback to the only 2-D f32 output. Then the same normalize the dense head applies.
        let dense_key = select_output_key(&named_ranks, "sentence_embedding", 2)?;
        let (dense_shape, dense_data) = outputs[dense_key.as_str()].try_extract_tensor::<f32>()?;
        anyhow::ensure!(
            dense_shape.len() == 2 && dense_shape[0] as usize >= batch_size,
            "bge-m3 dense output `{dense_key}` came back shaped {dense_shape:?}, need [>= {batch_size}, hidden]"
        );
        let hidden = dense_shape[1] as usize;
        let dense: Vec<Embedding> = (0..batch_size)
            .map(|row| normalize(&dense_data[row * hidden..(row + 1) * hidden]))
            .collect();

        // Sparse head: the token-level tensor by name (the vendored patch's rule), the same shape
        // guard, the same lexical head.
        let sparse_key = select_output_key(&named_ranks, "token_embedding", 3)?;
        let (shape, data) = outputs[sparse_key.as_str()].try_extract_tensor::<f32>()?;
        let shape: Vec<usize> = shape.iter().map(|&d| d as usize).collect();
        anyhow::ensure!(
            output_covers_input(&shape, batch_size, encoding_length),
            "bge-m3 sparse output `{sparse_key}` came back shaped {shape:?}, which does not cover the \
             input batch x seq of {batch_size}x{encoding_length} — the runtime returned a mis-shaped \
             tensor, so no sparse weights can be read from it",
        );
        let hidden_states = ndarray::ArrayViewD::from_shape(shape.as_slice(), data)?;
        let sparse = crate::SparseTextEmbedding::post_process_bgem3(
            &hidden_states,
            &input_ids,
            &attention_mask,
        );

        anyhow::ensure!(
            sparse.len() == dense.len(),
            "bge-m3 heads disagree on row count (dense {}, sparse {}) — refusing to zip misaligned rows",
            dense.len(),
            sparse.len()
        );
        Ok(dense.into_iter().zip(sparse).collect())
    }
}

/// The rank of an ort output tensor, `None` when it is not an f32 tensor at all.
fn tensor_rank(value: &Value) -> Option<usize> {
    value.try_extract_tensor::<f32>().ok().map(|(shape, _)| shape.len())
}

/// Picks a session output BY NAME first, by unique rank second. Positional (`keys().next()`)
/// selection is the bug this replaces: MIGraphX lists outputs in a different order than CPU/DML.
/// Pure over `(name, rank)` pairs so the rule is testable without an ONNX session.
fn select_output_key(
    outputs: &[(String, Option<usize>)],
    name_fragment: &str,
    want_rank: usize,
) -> Result<String> {
    if let Some((key, _)) = outputs.iter().find(|(k, _)| k.contains(name_fragment)) {
        return Ok(key.clone());
    }
    let mut by_rank = outputs.iter().filter(|(_, rank)| *rank == Some(want_rank));
    match (by_rank.next(), by_rank.next()) {
        (Some((key, _)), None) => Ok(key.clone()),
        (Some(_), Some(_)) => anyhow::bail!(
            "no output named *{name_fragment}* and several rank-{want_rank} outputs — cannot pick one safely: {:?}",
            outputs.iter().map(|(k, _)| k.as_str()).collect::<Vec<_>>()
        ),
        _ => anyhow::bail!(
            "no output named *{name_fragment}* and no rank-{want_rank} tensor among the outputs: {:?}",
            outputs.iter().map(|(k, _)| k.as_str()).collect::<Vec<_>>()
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::select_output_key;

    fn named(pairs: &[(&str, Option<usize>)]) -> Vec<(String, Option<usize>)> {
        pairs.iter().map(|(k, r)| (k.to_string(), *r)).collect()
    }

    /// The MIGraphX ordering incident, distilled: the pooled 2-D output listed FIRST must never be
    /// picked for the sparse head. Name wins; order never decides.
    #[test]
    fn outputs_are_selected_by_name_never_by_position() {
        let outputs = named(&[("sentence_embedding", Some(2)), ("token_embeddings", Some(3))]);
        assert_eq!(select_output_key(&outputs, "token_embedding", 3).unwrap(), "token_embeddings");
        assert_eq!(select_output_key(&outputs, "sentence_embedding", 2).unwrap(), "sentence_embedding");

        // Same outputs, MIGraphX's order — the answers must not change.
        let reordered = named(&[("token_embeddings", Some(3)), ("sentence_embedding", Some(2))]);
        assert_eq!(select_output_key(&reordered, "sentence_embedding", 2).unwrap(), "sentence_embedding");
    }

    /// An export with renamed outputs still resolves when exactly one tensor has the right rank —
    /// and refuses to guess when the rank is ambiguous or absent.
    #[test]
    fn rank_fallback_applies_only_when_unambiguous() {
        let renamed = named(&[("pooled", Some(2)), ("hidden", Some(3))]);
        assert_eq!(select_output_key(&renamed, "token_embedding", 3).unwrap(), "hidden");

        let ambiguous = named(&[("a", Some(3)), ("b", Some(3))]);
        assert!(select_output_key(&ambiguous, "token_embedding", 3).is_err());

        let none = named(&[("pooled", Some(2))]);
        assert!(select_output_key(&none, "token_embedding", 3).is_err());
    }
}
