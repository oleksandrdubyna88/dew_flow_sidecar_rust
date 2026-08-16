# PLAN — tokenizers by name, and a model that can describe itself

> Status: **IMPLEMENTED, 2026-08-16.** All three steps shipped: the tokenizer registry replaces the two
> `OnceLock` fields and the two-arm match, `GET /models` states kind · dimension · max sequence ·
> tokenizer per row with unknown as its own state, and `/embed` reports the width of the vectors it just
> returned. Verified on the wire, not only in tests. The two questions in §9 stay open as questions —
> they are decisions nobody has needed yet, not unbuilt work.
>
> **Three deviations, all deliberate.**
>
> 1. **The `/tokenize` batch cap is not `/embed`'s.** [PLAN_reliability_tail.md](../todo/PLAN_reliability_tail.md)
>    item 2 drafted it as "matching the one `/embed` already enforces". `/embed` does not *refuse* at
>    `max_batch`, it re-batches internally, and the host assembles `/tokenize` calls of up to 512 rows by
>    design (`SidecarClient.RequestRowBudget`, `dew_flow_rag_qln`) — so a cap at 64 would have refused
>    batches its only caller builds on purpose. Shipped as `TOKENIZE_MAX_TEXTS`, default **4096**: a
>    backstop against a pathological caller, never a wall a normal pass walks into. The test asserts the
>    headroom, not the constant.
> 2. **`/models` carries a `rerank` kind, which §3.2 did not list.** An endpoint named `/models` that
>    hides a model this build serves is a half-truth, and the same argument that made `kind` exist from
>    the first version applies to the reranker. Its `dimension` and `tokenizer` are both `null` —
>    a cross-encoder has no width ever, and this process registers no counter for it. Claiming `bge`
>    because the two models look related would have been exactly the confident guess this read exists to
>    remove.
> 3. **`available` was split into `available` + `tokenizer_available`.** Neither was in the plan; the
>    first was added for self-sufficiency and immediately proved ambiguous. **Found by reading the real
>    response, not by a test**: the startup log said `bge token counting enabled from …tokenizer.json`
>    while `/models` reported that row `available: false` — true of the *engine* and silent about the
>    tokenizer. "Engine cold, tokenizer ready" is precisely the state a consumer is in while validating a
>    recipe before a pass, so one flag could not carry both.
>
> **Cross-repo:** the consumers' halves are unchanged and still open —
> `dew_flow_rag_qln · todo/PLAN_tokenizer_contract_and_chunk_coverage.md` (its §5.1 port work, and its
> open question 2 about a tokenizer hash, which this read now makes answerable) and
> `dew_flow_benchmark · todo/PLAN_corpus_axis_integrity.md`. Both still name this plan under its old
> `todo/` path; they are separate repositories and were left as found.
>
> Related: [../todo/PLAN_sidecar_product.md](../todo/PLAN_sidecar_product.md) (the distribution story this
> does not touch), [../README.md](../README.md) (the shipped contract),
> [module_http_surface.md](module_http_surface.md) (the wire shapes as they now are).

## 1. The goal, before any solution

The consumer is about to compare embedding models — dense against dense, and later sparse against sparse.
Each must be chunked with **its own** tokenizer, because a chunk sized by the wrong tokenizer is a chunk
fitted to a window the serving model does not have.

This side already does the hard part and does it well: `/tokenize` is a real endpoint, it counts with the
model's own HuggingFace `tokenizer.json`, and it counts with **truncation off** so a caller learns the
true length rather than the clipped one. That is exactly what a chunker needs, and it is why the .NET
side has no chars-per-token fallback anywhere.

What is missing is addressability. `/tokenize` resolves its `model` argument through a two-arm string
match — `"bge"` and `"qwen"` — and anything else is a `400`. The two tokenizers are two hand-written
`OnceLock` fields. A third model is not configuration; it is a code change in three places, and the
caller cannot discover what this build can count for.

Two smaller gaps travel with it:

- **A model cannot describe itself.** Embedding dimension appears on no response — not `/health`, not
  `/embed`. A caller creating a vector collection must already know that BGE-M3 is 1024-dimensional,
  which means the fact lives in two repositories and can disagree in one of them. Max sequence length is
  reachable only as the effective cap on an embed response.
- **`SparseModel::SPLADEPPV1` exists in the vendored library and is wired to nothing.** Not a gap this
  plan closes, but the reason the metadata below must carry a `kind` from the first version rather than
  gaining one later.

## 2. What exists today, verified

| Fact | Where |
|---|---|
| `POST /tokenize` — a real route, no embedding performed, counts with **truncation off** so the true length is reported | `src/main.rs`, the `tokenize` handler |
| Its `model` argument is a two-value match (`"bge"`, `"qwen"`), anything else a `400` naming the failure | same handler |
| Two tokenizers, two hand-written lazy fields; a missing tokenizer file degrades a feature rather than killing the process | `src/main.rs`, the token-counter loaders |
| The Qwen tokenizer is loaded **for counting only** — no Qwen model is ever loaded | `src/main.rs`, its loader comment |
| Tokenizers are HuggingFace `tokenizer.json`, loaded per model instance by the vendored library — the library's own design is already per-model | `vendor-fastembed/src/common.rs`, `vendor-fastembed/src/bgem3_dual/mod.rs` |
| `/embed` carries **no model field**; the model is fixed for the life of the process | `src/main.rs`, `EmbedRequest` |
| Truncation happens silently inside the `tokenizers` crate; the sidecar compensates by reporting `token_count[]`, `truncated[]` and `token_accounting` | `src/main.rs`, the token-usage path |
| `token_accounting: false` means **not measured** — it must never read as "nothing was truncated" | same |
| Embedding dimension is on no response | `/health`, `/embed` shapes |
| `SparseModel::SPLADEPPV1` is defined and never constructed | `vendor-fastembed/src/models/sparse.rs` |

## 3. The shape — decisions

### 3.1 A tokenizer registry, not a match arm

One table built at startup, mapping a **name** to a loaded tokenizer and the file it came from. The two
present entries (`bge`, `qwen`) become rows rather than arms; a third is a row and a path, not a code
change in three places.

- **Names stay stable and are the contract.** `bge` and `qwen` keep meaning what they mean today; the
  registry is how they are stored, not a renaming.
- **Lazy and best-effort, unchanged.** A tokenizer whose file is absent leaves its row unavailable and
  degrades that one name — it never stops the process, exactly as today.
- **An unknown name is still a `400` that names what IS available.** Today's refusal names the failure;
  the registry lets it name the alternatives, which is the difference between a caller guessing and a
  caller correcting itself.
- Where the file for a name comes from stays as it is — discovery under the model cache for `bge`, an
  env-pointed path for `qwen` — with the registry recording, per row, which path was used. A count whose
  tokenizer nobody can name afterwards is a count nobody can reproduce.

### 3.2 `GET /models` — what this build can do

One read, answering per entry: **id**, **kind** (`dense` | `sparse` | `dense+sparse` | `tokenizer-only`),
**embedding dimension** (absent for a tokenizer-only row, and absent is a distinct state from zero),
**max sequence length**, and **tokenizer id**.

- `tokenizer-only` is a real kind, not a hack: it is exactly what `qwen` is here, and a consumer that
  cannot see the difference between "a model you can embed with" and "a tokenizer you can count with"
  will eventually ask this process to embed with the second one.
- The dimension is reported from what is actually loaded, never from a constant — the same
  read-it-from-the-model discipline the consumer applies to the sequence cap. A model not yet loaded
  reports its dimension as unknown rather than guessing; unknown is a value, not an error.
- This read is what lets the consumer validate a corpus recipe **before** starting a pass, instead of
  discovering a mismatch in the middle of one.

### 3.3 Dimension on the embed response

`/embed` gains the dimension of the vectors it just returned. It is free — the length of a row it already
computed — and it removes the last place where a caller has to know a model constant that this process
could simply state. A vector store creating a collection from the response it already holds cannot then
create it at the wrong width.

## 4. Cross-repository contract

What `dew_flow_rag_qln` needs from this plan, named identically in both:

1. `/tokenize` accepts any registered tokenizer name and refuses an unknown one **naming the registered
   set**.
2. `GET /models` reports id, kind, dimension, max sequence length and tokenizer id per entry, with
   unknown as a distinct state from zero or absent.
3. Names already in use (`bge`, `qwen`) keep their meaning — this is additive, and no consumer has to
   change to keep working.

What this plan needs from nobody: it is standalone and can ship before or after its consumer.

## 5. What this plan deliberately does not do

- **`/embed` does not gain a model selector.** The consumer's own plan chooses an **Ollama-backed**
  second dense embedder precisely so this process does not have to grow multi-model engine management,
  and that decision is theirs to keep. When a second model does need to be served *here*, it needs engine
  slots keyed by id, its own reference-vector canary, and a VRAM budget across two loaded engines — a
  larger piece of work that deserves its own plan rather than a field on a request.
- **SPLADE is not wired.** It is named in §1 only to justify `kind` existing from the start.
- **No authentication, no versioning, no metrics.** Those remain [PLAN_sidecar_product.md](../todo/PLAN_sidecar_product.md)'s.

## 6. Build order

1. **The registry** — the two present tokenizers become rows, **loaded at startup**; `/tokenize` resolves
   through it; the refusal names the registered set. Behaviour for `bge`/`qwen` is byte-identical.
   Coordinate with [PLAN_reliability_tail.md](../todo/PLAN_reliability_tail.md) item 2, which touches the same
   handler: startup loading removes its first-call disk I/O, and its `spawn_blocking` + batch cap sit on
   top of this unchanged.
2. **`GET /models`** — the metadata read, with unknown as its own state.
3. **Dimension on `/embed`** — one field, from the vectors already returned.

## 7. Test plan

Inline `#[cfg(test)]` tests, as the repository does today.

- `/tokenize` with `bge` and with `qwen` returns exactly what it returns today — the regression that
  proves the registry changed storage and not behaviour.
- An unknown name refuses with `400` and the message contains every registered name.
- A registered name whose tokenizer file is missing reports unavailable for that name only; other names
  keep answering.
- Counting stays truncation-off: a text past the cap reports its true length, not the clipped one.
- `/models` reports `tokenizer-only` for a counting tokenizer and a dimension for a loaded embedding
  model; an unloaded model's dimension is unknown rather than `0`.
- `/embed`'s dimension equals the length of the dense rows in the same response.

## 8. Definition of Done

- [x] A new tokenizer is a registry row and a file path — no new match arm, no new field.
- [x] An unknown tokenizer name is refused naming the registered set.
      (`an_unknown_tokenizer_is_refused_naming_every_registered_name`, three rows — two can be hardcoded,
      three cannot.)
- [x] `bge` and `qwen` answer byte-identically to before.
      (`a_missing_tokenizer_file_degrades_one_name_and_leaves_the_others_answering` — green before the
      change and after it, which is the whole of what a regression guard is for.)
- [x] `GET /models` states kind, dimension, max sequence length and tokenizer per entry, with unknown
      distinct from zero. (Plus `rerank` as a kind and the `available` split — see the deviations.)
- [x] `/embed` reports its own dimension — measured from the rows in that same response, `null` for an
      empty batch, never `0`.
- [x] `README.md`, `research/module_http_surface.md` and `research/architecture.md` updated;
      `todo/README.md` table updated.

### The test record

| Guarantee | Test | Observed |
|---|---|---|
| Tokenizers are read at STARTUP, not on the first request | `a_tokenizer_present_at_startup_still_counts_after_its_file_is_gone` | **RED** first: *"a loader that reads on first use instead answers None here"* → GREEN |
| A batch past the cap is refused, not encoded | `tokenize_refuses_a_batch_beyond_the_cap` | **RED** first: returned `Ok` having counted all **1025** texts inline → GREEN |
| The refusal names every registered row | `an_unknown_tokenizer_is_refused_naming_every_registered_name` | ships with the registry — it cannot be expressed before a third row can exist |
| The default cap clears the host's own row budget | `the_default_batch_cap_leaves_room_above_the_hosts_own_row_budget` | ships with the config field |
| bge/qwen behaviour unchanged | `a_missing_tokenizer_file_degrades_one_name_and_leaves_the_others_answering` | green on both sides |

Steps 2 and 3 are new surface rather than defect repair, so their tests ship *with* the feature instead
of failing before it — stated plainly rather than implying a RED run that did not happen:

| Guarantee | Test |
|---|---|
| A counting tokenizer is its own kind, never an embedder | `models_reports_a_counting_tokenizer_as_its_own_kind_never_as_an_embedder` |
| A tokenizer a model claims is named on that model, once | `a_tokenizer_claimed_by_a_model_is_not_also_listed_on_its_own` |
| An unmeasured dimension is unknown, not `0` and not a constant | `an_unmeasured_dimension_is_unknown_rather_than_zero_or_a_constant` |
| A cross-encoder never reports a width | `the_reranker_never_reports_a_dimension` |
| A registered name with no file is listed, marked unavailable | `a_tokenizer_with_no_file_is_listed_and_marked_unavailable` |
| A loaded tokenizer is visible behind a cold engine | `a_loaded_tokenizer_is_visible_even_while_its_engine_is_cold` |
| `/models` answers while an engine is held | `models_answers_while_an_engine_is_held` |
| The reported width is the width of the rows beside it | `the_reported_dimension_is_the_width_of_the_rows_beside_it` |
| Unpinning removes rows, never columns | `unpinning_leaves_the_width_untouched` |

Suite: 62 → **76 passed, 0 failed**, no compiler warnings.

**Verified on the wire**, which is where deviation 3 was found — `GET /models` and `POST /tokenize` were
called against a running binary on port 5399 and their real JSON read. Two guarantees the unit tests do
not reach: the route is actually mounted, and the refusal a caller receives really does name the
registered set (`unknown tokenizer 'llama' — this sidecar counts for 'bge', 'qwen'`). `/embed`'s
`dimension` could not be exercised end-to-end here: it needs a loaded engine and therefore the GPU, so
what is proven for it is the pure function, the struct-update path and the unpinning invariant.

## 9. Open questions

1. **Should a tokenizer row carry the hash of its `tokenizer.json`?** A name is stable; a file behind it
   is not, and a silently updated tokenizer changes every token count without changing anything a
   consumer can see. Cheap to add here, and the consumer's recipe identity is the thing that would use it.
2. **Whether `qwen` should stay** once the consumer's second dense model is chosen. If that model is
   Ollama-served, its tokenizer may need to live here anyway — counting is this process's job precisely
   because .NET cannot read a HuggingFace `tokenizer.json` at all.
