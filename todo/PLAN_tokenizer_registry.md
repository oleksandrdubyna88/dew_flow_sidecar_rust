# PLAN — tokenizers by name, and a model that can describe itself

> Status: **step 1 implemented 2026-08-16 — the registry; steps 2 (`GET /models`) and 3 (dimension on
> `/embed`) remain open.** Scope: the `/tokenize` handler and its tokenizer loading, one new `/models`
> read, and the embedding dimension on the wire. **Not in scope:** a second embedding model behind
> `/embed` — see §5.
>
> **What step 1 shipped, and its one deviation.** `TokenizerRegistry` / `TokenizerEntry` /
> `TokenizerSource` replace the two `OnceLock` fields and the two-arm match; rows load at startup;
> `/tokenize` resolves through the table and an unknown name is refused naming the registered set; the
> startup log names every row with the file behind it. It landed together with
> [PLAN_reliability_tail.md](PLAN_reliability_tail.md) item 2, as §6 asked — the encode moved to
> `spawn_blocking` and gained a `TOKENIZE_MAX_TEXTS` cap in the same change.
>
> *Deviation:* item 2's cap was drafted as "matching the one `/embed` already enforces". It is **not**
> that number. `/embed` does not refuse at `max_batch`, it re-batches internally, and the host assembles
> `/tokenize` calls of up to 512 rows by design (`SidecarClient.RequestRowBudget`, `dew_flow_rag_qln`) —
> so a cap at `max_batch` (64) would have refused batches its only caller builds on purpose. Shipped at
> **4096**, eight times the host's ceiling: a backstop against a pathological caller, never a wall a
> normal pass walks into. A test asserts the headroom rather than the constant.
>
> Sibling half: `dew_flow_rag_qln · todo/PLAN_tokenizer_contract_and_chunk_coverage.md`, whose step 1
> makes the .NET tokenizer port name its model. This plan makes that name mean something on this side.
>
> Related: [PLAN_sidecar_product.md](PLAN_sidecar_product.md) (the distribution story this does not
> touch), [../README.md](../README.md) (the current contract).
>
> **Overlap, named rather than discovered later:** [PLAN_reliability_tail.md](PLAN_reliability_tail.md)
> item 2 also touches `/tokenize` — it encodes on the async runtime, loads a tokenizer from disk inside
> the first call's `OnceLock::get_or_init`, and has no batch cap. Its proposed fixes are `spawn_blocking`
> for the encode, a batch cap matching `/embed`'s, and **pre-warming the tokenizers at startup**. This
> plan's registry is built at startup, which *is* that pre-warm — so §3.1 subsumes the third fix and
> leaves the other two where they are. **Whichever lands second must not re-introduce lazy first-call
> loading**, and the two plans should ideally land together: they are the same twenty lines.

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
- **No authentication, no versioning, no metrics.** Those remain [PLAN_sidecar_product.md](PLAN_sidecar_product.md)'s.

## 6. Build order

1. **The registry** — the two present tokenizers become rows, **loaded at startup**; `/tokenize` resolves
   through it; the refusal names the registered set. Behaviour for `bge`/`qwen` is byte-identical.
   Coordinate with [PLAN_reliability_tail.md](PLAN_reliability_tail.md) item 2, which touches the same
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
- [ ] `GET /models` states kind, dimension, max sequence length and tokenizer per entry, with unknown
      distinct from zero.
- [ ] `/embed` reports its own dimension.
- [x] `README.md` and `research/module_http_surface.md` updated; `todo/README.md` table updated.
      (Step 1's half; both gain the `/models` shape when step 2 lands.)

### Step 1's test record

| Guarantee | Test | Observed |
|---|---|---|
| Tokenizers are read at STARTUP, not on the first request | `a_tokenizer_present_at_startup_still_counts_after_its_file_is_gone` | **RED** first: *"a loader that reads on first use instead answers None here"* → GREEN |
| A batch past the cap is refused, not encoded | `tokenize_refuses_a_batch_beyond_the_cap` | **RED** first: returned `Ok` having counted all **1025** texts inline → GREEN |
| The refusal names every registered row | `an_unknown_tokenizer_is_refused_naming_every_registered_name` | ships with the registry — it cannot be expressed before a third row can exist |
| The default cap clears the host's own row budget | `the_default_batch_cap_leaves_room_above_the_hosts_own_row_budget` | ships with the config field |
| bge/qwen behaviour unchanged | `a_missing_tokenizer_file_degrades_one_name_and_leaves_the_others_answering` | green on both sides |

Suite: 62 → **67 passed, 0 failed**, no compiler warnings.

## 9. Open questions

1. **Should a tokenizer row carry the hash of its `tokenizer.json`?** A name is stable; a file behind it
   is not, and a silently updated tokenizer changes every token count without changing anything a
   consumer can see. Cheap to add here, and the consumer's recipe identity is the thing that would use it.
2. **Whether `qwen` should stay** once the consumer's second dense model is chosen. If that model is
   Ollama-served, its tokenizer may need to live here anyway — counting is this process's job precisely
   because .NET cannot read a HuggingFace `tokenizer.json` at all.
