# Third-party notices

This product is **sold**, and this component is **built on the customer's machine** rather than shipped to
it — which changes what the licence position has to cover, not whether it has to exist. A licence position is
a shipping fact, not a footnote.

The rule that governs additions and bumps, shared with the rest of the family: resolve the licence of the
**exact version** from the artefact itself, never from memory. Metadata lies, and a licence can change
between versions of one package.

Everything below was resolved on **2026-08-19** from `cargo metadata` and `cargo tree` over this repository's
own `Cargo.lock`, for the `x86_64-pc-windows-msvc` target — the graph that actually becomes the default
(DirectML) binary.

## The position, in one paragraph

Of the **348 crates in the Windows build graph**, every one is permissive — MIT, Apache-2.0, BSD, ISC, Zlib,
Unicode-3.0 or a dual licence including one of those — **except one**: `option-ext` is MPL-2.0. There is no
GPL or LGPL component in any build. Three obligations survive distribution: Apache-2.0 attribution (§4(d)),
MIT copyright retention, and MPL-2.0 source availability (§3.2). [NOTICE](NOTICE) satisfies all three and
must travel with any build.

| Licence | Crates | Note |
|---|---|---|
| MIT, or MIT OR Apache-2.0 | 252 | the bulk of the Rust ecosystem |
| Apache-2.0 (alone) | 7 | includes `fastembed`, `tokenizers`, `hf-hub` |
| Unicode-3.0 | 25 | Unicode data tables |
| BSD-3-Clause / BSD-2-Clause / ISC / Zlib / Unlicense / CDLA / NCSA / BSL / CC0 | 24 | all permissive, all dual or standalone |
| **MPL-2.0** | **1** | **`option-ext` — see below** |

Counts are of distinct crate versions as `cargo tree` resolves them; a crate appearing at two versions counts
twice, which is why they sum above the number of unique names.

## `option-ext` 0.2.0 — the one copyleft component

**Licence:** MPL-2.0 · **Source:** https://github.com/soc/option-ext · **Shipped:** yes.

How it arrives, traced rather than guessed (`cargo tree -i option-ext`):

```
option-ext v0.2.0
└── dirs-sys v0.5.0
    └── dirs v6.0.0
        └── hf-hub v0.5.0
            └── fastembed v5.17.3 (vendor-fastembed)
                └── bge-sidecar v0.1.0
```

`hf-hub` uses `dirs` to find the model cache directory, and `dirs-sys` uses `option-ext` for two `Option`
combinators. It is three lines of convenience, four levels down.

**Why it is not a problem, and what the obligation actually is.** MPL-2.0 is *file-level* copyleft: the
obligations attach to the MPL-licensed files, not to a work that merely links them. Distributing a binary
containing them is permitted provided the source of **those files** stays available under MPL-2.0 (§3.2),
and provided any modification of them is released under MPL. We modify nothing in `option-ext`; the source
is at the URL above. Naming it here, with its URL, is the whole of what is owed.

**What would change the answer:** vendoring `option-ext` and patching it, or removing the ability of a
recipient to obtain its source. Neither is planned. If `hf-hub` ever drops `dirs`, this section goes with it.

## The vendored fork

`vendor-fastembed/` is a **full copy** of `fastembed` 5.17.3 (Apache-2.0), carried in-tree rather than taken
from crates.io, with **one** patch marked in place in
`vendor-fastembed/src/sparse_text_embedding/impl.rs` (search for `VENDORED PATCH`).

The patch: the BGEM3 sparse path selected its session output with `keys().next()`, which under the MIGraphX
execution provider picks the 2-D `sentence_embedding` instead of the 3-D `token_embeddings`; the
post-process then panicked (`assertion failed: index < dim`) on every sparse embed and poisoned the engine.
The patch selects the output by name and rank.

Apache-2.0 permits modification and requires that modified files carry prominent notices of change (§4(b))
and that the licence and any NOTICE travel with the copy (§4(a), §4(d)). The fork keeps upstream's own
licence files unaltered, the patch is marked at the site of the change, and this section is the record of
what was changed and why. **The copy is otherwise kept byte-identical to upstream**, which is what makes the
patch reviewable — and it is dropped as soon as upstream selects the BGEM3 output deterministically.

## Never redistributed

Three vendor components are **used and never shipped**. This is not caution; it is the reason the product's
whole distribution story is "the customer's machine builds it".

| Component | Vendor | Why it never ships |
|---|---|---|
| DirectML | Microsoft | Vendor-licensed redistributable with its own terms. One cached copy in our registry, an image layer or an offline bundle makes us a redistributor of it |
| CUDA runtime | NVIDIA | Same, under the CUDA EULA |
| cuDNN | NVIDIA | Same, under the cuDNN SLA — and the most commonly mis-shipped of the three, because it is large and arrives as a plain folder of DLLs |

The execution provider is a **compile-time feature** here (`default = ["dml"]`, plus `cuda` and `migraphx`),
so there is no fat binary that could carry one by accident. The CPU flavour
(`--no-default-features`) carries no vendor component at all and is the one that may be published as an
artefact.

`ort` resolves the provider libraries from the **executable's own directory** at run time, which is why
`preflight_provider` (`src/provider.rs`) refuses a flavour whose libraries are absent, naming them: a missing
vendor library must fail at startup with a sentence, never at the first user search with `Error 126`.

## ONNX Runtime itself

MIT, Copyright (c) Microsoft Corporation, https://github.com/microsoft/onnxruntime. The `ort` crate that
binds it is MIT OR Apache-2.0. On the DirectML and CUDA flavours a prebuilt ONNX Runtime travels beside the
executable; on the MIGraphX flavour the operator builds their own from source with `--use_migraphx` and
points `ORT_DYLIB_PATH` at it, so nothing is shipped at all. MIT attribution is carried in
[NOTICE](NOTICE).

## Models

The model weights are **not** in this repository and are not distributed with it. They are pulled from
Hugging Face into `MODEL_CACHE_DIR` on first use:

| Model | Source | Licence |
|---|---|---|
| BAAI/bge-m3 | https://huggingface.co/BAAI/bge-m3 | MIT |
| bge-reranker-v2-m3 | https://huggingface.co/BAAI/bge-reranker-v2-m3 | Apache-2.0 |

Named here because a reader who sees an embedding service reasonably asks what it embeds with, and because
a future decision to ship a model cache would change this file rather than being a detail of packaging.

## How to re-resolve this

```powershell
cargo metadata --format-version 1 --all-features | ConvertFrom-Json    # every crate + its licence
cargo tree --target x86_64-pc-windows-msvc --edges normal --format "{p}|{l}"   # what actually ships
cargo tree -i <crate>                                                  # who pulls something in
```

Re-run after any `Cargo.lock` change that adds a crate, and check the result against the one-paragraph
position above: the claim that everything is permissive bar one MPL component is a claim with a date on it.
