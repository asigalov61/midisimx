# midisimx

**Transformer-based MIDI similarity embeddings** — convert MIDI files into comparable vectors, search large pre-computed corpora, and reduce embedding dimensionality at scale.

Part of **[Project Los Angeles](https://github.com/Tegridy-Code/Project-Los-Angeles)** · Tegridy Code

| | |
|---|---|
| **Version** | 1.0.0 |
| **License** | Apache License 2.0 |
| **Python** | 3.9+ |
| **Hardware** | CUDA recommended (CPU fully supported) |

## Overview

`midisimx` maps MIDI files to fixed-size embedding vectors via a pre-trained Transformer encoder. Embeddings can be compared directly (cosine similarity) for duplicate detection, near-neighbor search, and dataset clustering. Pre-trained checkpoints and large pre-computed embedding corpora (Lakh MIDI Dataset) are hosted on Hugging Face.

| Module | Purpose |
|---|---|
| `midisimx` | Core: model loading, MIDI tokenization, pooling, embedding computation, similarity search & retrieval, HF downloads |
| `midisimx.helpers` | Bundled resources, MIDI normalization/hashing, apt package utilities |
| `midisimx.pca_reduce` | Streaming, GPU-accelerated PCA for embedding reduction |

## Function index

| Function | Module | Purpose |
|---|---|---|
| `download_model()` | core | Download a pre-trained model checkpoint from Hugging Face |
| `download_embeddings()` | core | Download a pre-computed embeddings file from Hugging Face |
| `download_all_embeddings()` | core | Download the entire embeddings dataset from Hugging Face |
| `load_model()` | core | Build the Transformer encoder and load checkpoint weights |
| `load_embeddings()` | core | Load a structured embeddings file → `(names, vectors)` |
| `save_embeddings()` | core | Save names + vectors as a structured `.npy` file |
| `midi_to_tokens()` | core | MIDI file → token sequences (with transpositions) |
| `random_ngram_replace()` | core | Random single-token / n-gram masking of a token sequence |
| `pad_and_mask()` | core | Pad token sequences and build validity masks |
| `masked_mean_pool()` | core | Masked mean pooling over token embeddings |
| `masked_weighted_mean_pool()` | core | Weighted mean pooling by token-id ranges |
| `masked_weighted_mean_aggregated_pool()` | core | Per-range pooled embeddings, concatenated or stacked |
| `get_embeddings_bf16()` | core | End-to-end embedding computation for token sequences |
| `cosine_similarity_topk()` | core | Chunked top-k cosine-similarity search against a corpus |
| `idxs_sims_to_sorted_list()` | core | Top-k arrays → sorted (index, transpose, similarity) list |
| `print_sorted_idxs_sims_list()` | core | Print search results, or convert to records |
| `get_corpus_midis()` | core | LRU-cached corpus directory scan → name/path dict |
| `copy_corpus_files()` | core | Copy matched corpus MIDIs and the query MIDI to disk |
| `get_package_models()` | helpers | List model checkpoints bundled with the package |
| `get_package_embeddings()` | helpers | List embedding files bundled with the package |
| `get_normalized_midi_md5_hash()` | helpers | Original + normalization-invariant MD5 hashes of a MIDI file |
| `normalize_midi_file()` | helpers | Write a normalized copy of a MIDI file to disk |
| `is_installed()` | helpers | Check whether a dpkg package is installed |
| `install_apt_package()` | helpers | Idempotent apt package installation (Debian/Ubuntu) |
| `pca_reduce_embeddings()` | pca_reduce | Streaming two-pass GPU PCA reduction |
| `PCAReductionResult` | pca_reduce | Result dataclass of `pca_reduce_embeddings()` |

## Installation

```bash
pip install huggingface_hub ipywidgets tqdm scikit-learn torch einops einx torch-summary matplotlib numpy==1.26.4
```

- `TMIDIX` and the `x-transformer` implementation ship **inside** the `midisimx` package — no extra installs needed.
- Optional system packages (e.g. **FluidSynth** on Debian/Ubuntu) can be installed programmatically via [`install_apt_package()`](#install_apt_package).

## Quick start

```python
import torch
from midisimx import (download_model, load_model, midi_to_tokens, get_embeddings_bf16,
                      download_embeddings, load_embeddings, cosine_similarity_topk,
                      idxs_sims_to_sorted_list, print_sorted_idxs_sims_list,
                      copy_corpus_files)

# 1) Pre-trained checkpoint from Hugging Face
model, ctx, dtype = load_model(download_model(), device='cuda')

# 2) MIDI -> token sequences (one per transpose variant; 12 by default)
seqs = midi_to_tokens('my_song.mid')

# 3) Embed every variant — each row of the query batch is one transpose
embs = get_embeddings_bf16(model, seqs, normalize=True)      # (12, 768), CPU

# 4) Corpus + chunked top-k search
names, corpus = load_embeddings(download_embeddings())
idxs, sims = cosine_similarity_topk(embs, corpus, topk=16)   # (12, 16) NumPy arrays

# 5) Rank matches across all transpose variants (best variant per corpus MIDI kept)
results = idxs_sims_to_sorted_list(idxs, sims)
records = print_sorted_idxs_sims_list(results, names, return_as_list=True)

# 6) Copy the best matches — and the query MIDI — to disk
out_dir = copy_corpus_files(records,
                            corpus_midis_dirs=['./Corpus MIDIs Dir/'],
                            original_midi_path='my_song.mid')
```

Save your own embeddings (compatible with `load_embeddings()`):

```python
from midisimx import save_embeddings
save_embeddings(['my_song'], embs[:1], output_file_name='my_embeddings.npy')
```

---

## Core module (`midisimx`)

### Hugging Face downloads

Convenience wrappers around `huggingface_hub`. All three accept `verbose` and `**kwargs` (forwarded to the underlying hub call).

#### `download_model()`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `repo_id` | `str` | `'projectlosangeles/midisimx'` | Hugging Face model repo. |
| `filename` | `str` | `'midisimx_trained_model_14391_steps_0.255_loss_0.9036_acc.pth'` | Checkpoint file to fetch. |
| `local_dir` | `str` | `'./midisimx-models/'` | Destination directory. |
| `verbose` | `bool` | `True` | Print progress. |
| `**kwargs` | — | — | Forwarded to `hf_hub_download()`. |

**Returns:** `str` — path to the downloaded checkpoint.

#### `download_embeddings()`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `repo_id` | `str` | `'projectlosangeles/midisimx-embeddings'` | Hugging Face dataset repo. |
| `filename` | `str` | `'lakh_midi_dataset_17203_clean_midis_embeddings_1_2_1_2_weighted_cc_by_nc_sa.npy'` | Embeddings file to fetch. |
| `local_dir` | `str` | `'./midisimx-embeddings/'` | Destination directory. |
| `verbose` | `bool` | `True` | Print progress. |
| `**kwargs` | — | — | Forwarded to `hf_hub_download()`. |

**Returns:** `str` — path to the downloaded file.

#### `download_all_embeddings()`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `repo_id` | `str` | `'projectlosangeles/midisimx-embeddings'` | Hugging Face dataset repo. |
| `revision` | `str` | `'main'` | Repo revision/branch. |
| `local_dir` | `str` | `'./midisimx-embeddings/'` | Destination directory. |
| `verbose` | `bool` | `True` | Print progress. |
| `**kwargs` | — | — | Forwarded to `snapshot_download()`. |

**Returns:** `str` — local directory containing all downloaded embeddings.

### Model and embedding I/O

#### `load_model()`

```python
load_model(
    model_path: str = './midisimx-models/midisimx_trained_model_14391_steps_0.255_loss_0.9036_acc.pth',
    dim: int = 768,
    depth: int = 16,
    heads: int = 12,
    max_seq_len: int = 3072,
    pad_idx: int = 719,
    dtype: torch.dtype = torch.bfloat16,
    device: str = 'cuda',
    compile_model: bool = False,
    dynamic_compile: bool = True,
    verbose: bool = True,
) -> Tuple[TransformerWrapper, torch.amp.autocast, torch.dtype]
```

Builds a `TransformerWrapper` + `Encoder` stack (rotary positional embeddings, FlashAttention) with vocabulary size `pad_idx + 1`, loads the checkpoint state dict, moves the model to `device`, and sets it to evaluation mode.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `model_path` | `str` | *(see signature)* | Path to a `.pth` state dict compatible with the architecture. |
| `dim` | `int` | `768` | Hidden dimension of the encoder layers. |
| `depth` | `int` | `16` | Number of encoder layers. |
| `heads` | `int` | `12` | Attention heads per layer. |
| `max_seq_len` | `int` | `3072` | Maximum supported sequence length. |
| `pad_idx` | `int` | `719` | Padding token id; vocabulary size = `pad_idx + 1` (720). |
| `dtype` | `torch.dtype` | `torch.bfloat16` | Autocast dtype (`bfloat16` / `float16` / `float32`). |
| `device` | `str` | `'cuda'` | `'cuda'`, `'cpu'`, or a `torch.device`. |
| `compile_model` | `bool` | `False` | Compile with `torch.compile()`. |
| `dynamic_compile` | `bool` | `True` | Dynamic compile mode — handles variable-length sequences. |
| `verbose` | `bool` | `True` | Print progress and a model summary. |

**Returns:** `(model, ctx, dtype)` —
- `model` — `TransformerWrapper` with loaded weights, on `device`, in `eval()` mode;
- `ctx` — a `torch.amp.autocast` context for `device`/`dtype` (wrap manual inference calls with it);
- `dtype` — the dtype passed in (echoed for convenience).

```python
model, ctx, dtype = load_model(device='cuda')
with ctx:
    ...  # manual inference
```

**Raises:** `FileNotFoundError` / `OSError` (unreadable checkpoint), `RuntimeError` (state-dict / architecture mismatch), or device errors from `model.to(device)`.

> **Note:** architecture parameters must match those used to train the checkpoint. Defaults match the released midisimx model.

#### `load_embeddings()`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `embeddings_path` | `str` | *(see signature)* | Path to a structured `.npy` embeddings file. |
| `midi_names_key` | `str` | `'midi_names'` | Field name for the names column. |
| `midi_embeddings_key` | `str` | `'midi_embeddings'` | Field name for the embedding vectors. |
| `verbose` | `bool` | `True` | Print progress. |

**Returns:** `Tuple[np.ndarray, np.ndarray]` — `(midi_names, midi_embeddings)`, loaded with `np.load(..., allow_pickle=True)`.

#### `save_embeddings()`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `embeddings_name_strings` | `list[str]` | — | One identifier (e.g. filename) per embedding. |
| `embeddings` | `Tensor` / `ndarray` | — | Vectors of shape `(n, D)`. Tensors are moved to CPU and converted; lists are converted via `np.array`. |
| `name_strings_key` | `str` | `'midi_names'` | Structured-array field name for names. |
| `embeddings_key` | `str` | `'midi_embeddings'` | Structured-array field name for vectors. |
| `output_file_name` | `str` | `'saved_midi_embeddings.npy'` | Destination when saving to disk. |
| `return_merged_array` | `bool` | `False` | `True`: return the structured array instead of writing to disk. |
| `verbose` | `bool` | `True` | Print diagnostics. |

Writes (or returns) a NumPy structured array with dtype `[(names, object), (embeddings, <input dtype>, (D,))]` — the same format consumed by `load_embeddings()` and used by the published corpora.

**Returns:** `np.ndarray` if `return_merged_array=True`, else `None`.

**Raises:** `ValueError` — embeddings not 2-D, or row count ≠ `len(embeddings_name_strings)`.

> **Tip:** detach gradient-tracking tensors first (`embeddings.detach().cpu()`).

### MIDI tokenization

#### `midi_to_tokens()`

```python
midi_to_tokens(
    midi_file_path: str,
    max_seq_len: int = 3072,
    transpose_factor: int = 6,
    clean_midi: bool = True,
    verbose: bool = True,
) -> list[list[int]]
```

Converts a MIDI file into compact integer token sequences ready for the model.

**Processing pipeline:**

1. Single-track millisecond score (`midi2single_track_ms_score`)
2. Enhanced score notes with sustain applied (`advanced_score_processor`)
3. Timing augmentation (`augment_enhanced_score_notes`, timings ÷ 32)
4. Optional instrument cleaning (lead/bass only — `TMIDIX.CLEAN_INSTRUMENTS`)
5. Solo-piano extraction
6. Duplicate-pitch removal, duration fixing, timing recalculation
7. Per transpose variant: transpose → chordify → tones-chord correction → re-chordify → encode

| Parameter | Type | Default | Description |
|---|---|---|---|
| `midi_file_path` | `str` | — | Path to the MIDI file. |
| `max_seq_len` | `int` | `3072` | Truncate each output sequence to this length. |
| `transpose_factor` | `int` | `6` | Semitone transpose range, **clamped to 0–6**. `> 0` produces `2 × transpose_factor` variants covering `[-tf, tf)`; `0` produces a single un-transposed sequence. |
| `clean_midi` | `bool` | `True` | Keep only lead/bass instruments; discard everything else. |
| `verbose` | `bool` | `True` | Progress messages and tqdm bars. |

**Returns:** `list[list[int]]` — one token sequence per transpose variant (**12 by default**). Exceptions are caught and printed (when `verbose`); an empty list is returned on failure or when no notes survive preprocessing.

**Token layout** (vocabulary = 720, padding id = 719):

| Token id(s) | Meaning |
|---|---|
| `0` | Sequence start (initial zero delta-time) |
| `1–127` | Delta-time since the previous chord (clipped to 0–127) |
| `129–255` | Note **pitch** (clipped 1–127, `+128`) |
| `257–383` | Note **duration** (clipped 1–127, `+256`) |
| `384+` | Chord token (`TMIDIX.ALL_CHORDS_SORTED` index `+12`, or pitch class `0–11` for single notes, then `+384`) |
| `718` | Masking value (see [`random_ngram_replace()`](#random_ngram_replace)) |
| `719` | Padding |

Each chord event is encoded as: `delta_time`, `chord_token`, then one `(pitch_token, duration_token)` pair per note in the chord.

### Token sequence augmentation

#### `random_ngram_replace()`

```python
random_ngram_replace(
    seq: List[int],
    prob_single: float = 0.10,
    prob_ngram: float = 0.10,
    max_ngram: int = 5,
    replace_value: int = 718,
    rng: Optional[np.random.Generator] = None,
) -> List[int]
```

Returns a copy of a token sequence with random tokens replaced by a masking value — useful for augmenting/corrupting sequences (robustness tests, contrastive-style training data). The original list is never modified.

Single-token replacements are applied first (each element independently with probability `prob_single`); then, at each index, an n-gram replacement of random length 2–`max_ngram` starts with probability `prob_ngram` (and may overwrite earlier single replacements).

| Parameter | Type | Default | Description |
|---|---|---|---|
| `seq` | `List[int]` | — | Input token sequence. |
| `prob_single` | `float` | `0.10` | Per-element probability of a single-token replacement. |
| `prob_ngram` | `float` | `0.10` | Per-index probability of starting an n-gram replacement. |
| `max_ngram` | `int` | `5` | Maximum n-gram length (actual lengths drawn uniformly from 2…`max_ngram`). |
| `replace_value` | `int` | `718` | Masking value — the token adjacent to the padding id `719`. |
| `rng` | `Optional[np.random.Generator]` | `None` | NumPy RNG; a fresh `default_rng()` is created when omitted. |

**Returns:** `List[int]` — new sequence with replacements applied.

### Batch preparation and pooling

Utilities used internally by `get_embeddings_bf16()` — also directly usable.

#### `pad_and_mask()`

Pads a batch of variable-length sequences and builds a boolean validity mask. Sequences longer than `seq_len` are truncated; if `seq_len` is `None` (or exceeds the batch maximum), the batch maximum is used.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `sequences` | `List[List[int]]` | — | Token-id sequences. |
| `pad_idx` | `int` | `719` | Padding token id. |
| `seq_len` | `Optional[int]` | `None` | Optional cap on sequence length. |
| `device` | `Optional[torch.device]` | `None` | Device for the returned tensors. |
| `verbose` | `bool` | `False` | Progress bar and summary. |

**Returns:** `(x, mask)` — `LongTensor (B, T)` of padded ids and `BoolTensor (B, T)` with `True` at real-token positions. Empty input → two empty `(0, 0)` tensors.

#### Pooling strategies

All pooling functions operate on per-token embeddings `(B, L, D)` with a validity mask; padding positions are always excluded.

| Function | Key arguments | Output |
|---|---|---|
| `masked_mean_pool(token_embeddings, mask, dim=1, eps=1e-9, verbose=True)` | — | `(B, D)` — plain masked mean. |
| `masked_weighted_mean_pool(token_embs, valid_mask, token_ids=None, token_type_weights=None, dim=1, verbose=False)` | weights per token | `(B, D)` — weighted mean pooling. |
| `masked_weighted_mean_aggregated_pool(token_embs, valid_mask, token_ids=None, token_type_weights=None, dim=1, concat=True, verbose=False)` | weights per range | `(B, D·R)` if `concat=True`, else `(B, R, D)` — one weighted mean per token range, scaled by the range weight. |

**Shared behavior:**
- `token_ids=None` or empty/`None` `token_type_weights` → fallback to `masked_mean_pool()`.
- In the aggregated variant, ranges are processed in ascending order of start token (deterministic concatenation order); a range containing no tokens yields a zero vector.

**`token_type_weights` accepts:**

| Form | Example | Meaning |
|---|---|---|
| `dict` | `{(0, 128): 1.0, (128, 256): 2.0}` | Weight per `[start, end)` token-id range (inclusive start, exclusive end). |
| `tuple` | `(1.0, 2.0, 1.5)` | Legacy shorthand for ranges `[0,128)`, `[128,256)`, `[256,384)` (onset / duration / pitch). |
| `None` | — | Uniform weights (equivalent to plain mean). |

Tokens outside all specified ranges get weight `1.0`; padding positions always get weight `0`.

> **Tip:** the explicit `dict` form is recommended. Check the [token layout](#midi-tokenization) to confirm which token ids land in which range (pitches occupy 129–255, durations 257–383 with the default tokenizer).

### Embedding computation

#### `get_embeddings_bf16()`

Computes embeddings end-to-end: batching → padding → forward pass (`torch.inference_mode()` + optional bfloat16 autocast) → pooling → optional L2 normalization.

> **Model contract:** the model's forward is invoked as `model(x, return_embeddings=True, mask=mask)` — it must accept a `LongTensor (B, T)` and `BoolTensor (B, T)` and return either `(B, D)` (already pooled) or `(B, L, D)` per-token embeddings (pooled according to `pooling`). The function itself moves the model to the resolved device and calls `model.eval()`.

```python
get_embeddings_bf16(
    model,
    sequences: List[List[int]],
    seq_len: Optional[int] = 3072,
    seq_pad_idx: int = 719,
    batch_size: int = 16,
    save_every_num_batches: int = -1,
    save_file_path: str = "saved_embeddings.npy",
    device: Optional[torch.device] = None,
    normalize: bool = False,
    pooling: str = "auto",
    token_type_weights=None,
    concat_aggregated_embeddings: bool = True,
    use_bfloat16: bool = True,
    return_dtype: str = "float32",
    return_numpy: bool = False,
    verbose: bool = True,
    show_progress_bar: bool = True,
) -> Union[Tensor, np.ndarray]
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `model` | — | — | PyTorch model (see contract above). |
| `sequences` | `List[List[int]]` | — | Token-id sequences; empty list → empty `(0, 0)` result. |
| `seq_len` | `Optional[int]` | `3072` | Truncation/padding length; `None` → per-batch maximum. |
| `seq_pad_idx` | `int` | `719` | Padding token id. |
| `batch_size` | `int` | `16` | Sequences per forward pass. |
| `save_every_num_batches` | `int` | `-1` | If `> 0`, checkpoint accumulated embeddings every N batches. |
| `save_file_path` | `str` | `'saved_embeddings.npy'` | Checkpoint file path. |
| `device` | `Optional[torch.device]` | `None` | Defaults to the model's parameter device. |
| `normalize` | `bool` | `False` | L2-normalize each embedding (always computed in float32). |
| `pooling` | `str` | `'auto'` | `"auto"`/`"mean"`, `"weighted_mean"`, or `"weighted_mean_aggregated"` — see [pooling strategies](#pooling-strategies). |
| `token_type_weights` | — | `None` | See [pooling strategies](#pooling-strategies). |
| `concat_aggregated_embeddings` | `bool` | `True` | Aggregated pooling: concatenate (`(N, D·R)`) vs. stack (`(N, R, D)`). |
| `use_bfloat16` | `bool` | `True` | bfloat16 autocast with graceful fallback if unsupported. |
| `return_dtype` | `str` | `'float32'` | `'float32'` or `'float16'`. |
| `return_numpy` | `bool` | `False` | Return `np.ndarray` instead of a tensor. |
| `verbose` | `bool` | `True` | Print progress and diagnostics. |
| `show_progress_bar` | `bool` | `True` | Show the tqdm batch bar. |

**Returns:** CPU `torch.Tensor` or `np.ndarray` (depending on `return_numpy`):
- `(N, D)` for standard pooling;
- `(N, D·R)` for `"weighted_mean_aggregated"` with `concat_aggregated_embeddings=True`;
- `(N, R, D)` for `"weighted_mean_aggregated"` with `concat_aggregated_embeddings=False`;
- `(0, 0)` when `sequences` is empty.

**Raises:**
- `AssertionError` — `return_dtype` is not `'float32'` or `'float16'`.
- `RuntimeError` — the model returned `None` for embeddings.
- `ValueError` — unexpected output dimensionality, or unsupported `pooling` value.

> Internal helpers (`_normalize_token_type_weights`, `_get_autocast_ctx`, `_pool_embeddings`) are private implementation details, not part of the stable API.

### Similarity search and retrieval

#### `cosine_similarity_topk()`

Chunked top-k cosine-similarity search between query embeddings and a corpus that may be far larger than device memory. The corpus is processed in chunks and per-chunk top-k results are merged into a running buffer, so only `chunk_size` corpus rows are ever resident on the compute device. NumPy corpora are converted chunk-by-chunk — the full corpus is never copied to the GPU.

```python
cosine_similarity_topk(
    query_embs,               # (Q, D) Tensor or ndarray
    corpus_embs,              # (N, D) Tensor or ndarray
    topk: int = 16,
    chunk_size: int = 10000,
    device: Optional[torch.device] = None,
    use_gpu_if_available: bool = True,
    normalize_inputs: bool = True,
    return_dtype: torch.dtype = torch.float32,
    use_fp32_accumulation: bool = True,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray]
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `query_embs` | `Tensor` / `ndarray` | — | Query embeddings, shape `(Q, D)`. |
| `corpus_embs` | `Tensor` / `ndarray` | — | Corpus embeddings, shape `(N, D)`; same `D` as queries. |
| `topk` | `int` | `16` | Matches to return per query. |
| `chunk_size` | `int` | `10000` | Corpus rows per chunk — lower to reduce peak device memory. |
| `device` | `Optional[torch.device]` | `None` | Compute device; `None` → CUDA if available and `use_gpu_if_available`, else CPU. |
| `use_gpu_if_available` | `bool` | `True` | Prefer GPU when `device` is not explicitly set. |
| `normalize_inputs` | `bool` | `True` | L2-normalize queries and corpus chunks before computing similarities. |
| `return_dtype` | `torch.dtype` | `torch.float32` | Output dtype for similarity values; `float16` is cast before returning. |
| `use_fp32_accumulation` | `bool` | `True` | Run matrix multiplications in float32 for numerical stability. |
| `verbose` | `bool` | `True` | Status messages and a per-chunk tqdm bar. |

**Returns:** `(best_idx, best_vals)` — NumPy arrays of shape `(Q, topk)` on CPU: global corpus indices and corresponding top-k similarity values, sorted descending per query.

**Raises:** `ValueError` — inputs not 2-D, or query/corpus embedding dimensions differ.

> **Notes:**
> - The signature annotation declares `Tuple[Tensor, Tensor]`, but **NumPy arrays** are what is actually returned.
> - If the corpus has fewer than `topk` rows, the remaining slots contain index `-1` and similarity `-inf`.
> - Peak device memory scales with `chunk_size` (plus a `(Q, chunk_size)` similarity block per chunk).

#### `idxs_sims_to_sorted_list()`

```python
idxs_sims_to_sorted_list(
    idxs: np.ndarray,
    sims: np.ndarray,
    sims_mult: int = 100,
    remove_dupes: bool = True,
) -> List
```

Flattens the `(num_variants, topk)` arrays returned by `cosine_similarity_topk()` into a single list sorted by similarity (descending). Each input row corresponds to one query — i.e. one transpose variant from `midi_to_tokens()` — and each output entry records which variant (semitone transpose) produced the match:

| `idxs.shape[0]` (rows) | Transpose values assigned |
|---|---|
| `1` | `0` |
| even `n` | `range(-n/2, n/2)` |
| odd `n > 1` | `range(-6, 6)` (default 12-variant scheme assumed) |

| Parameter | Type | Default | Description |
|---|---|---|---|
| `idxs` | `np.ndarray` | — | Corpus indices, shape `(num_variants, topk)`. |
| `sims` | `np.ndarray` | — | Similarity values, same shape as `idxs`. |
| `sims_mult` | `int` | `100` | Multiplier applied to similarities (e.g. ×100 for percentage-style scores). |
| `remove_dupes` | `bool` | `True` | Keep only the best (highest-similarity) entry per corpus index — i.e. the best transpose variant per matched MIDI. |

**Returns:** sorted list — `(corpus_index, transpose_value, similarity)` tuples when `remove_dupes=False`; `[corpus_index, transpose_value, similarity]` lists (one per unique corpus item) when `remove_dupes=True`.

**Raises:** `AssertionError` — `idxs.shape != sims.shape`.

#### `print_sorted_idxs_sims_list()`

```python
print_sorted_idxs_sims_list(
    sorted_idxs_sims_list: list,
    corpus_midi_names: Union[list, np.ndarray],
    return_as_list: bool = False,
) -> Union[List, None]
```

Pretty-prints a results list from `idxs_sims_to_sorted_list()` (format: `#rank  name --- transpose --- similarity`), or converts it into rank-annotated records for downstream use.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `sorted_idxs_sims_list` | `list` | — | Output of `idxs_sims_to_sorted_list()`. |
| `corpus_midi_names` | `list` / `np.ndarray` | — | Corpus names (e.g. the names array from `load_embeddings()`). |
| `return_as_list` | `bool` | `False` | `True`: skip printing and return records instead. |

**Returns:** with `return_as_list=True`, a list of `[rank, name, transpose, similarity]` records — **exactly the input format expected by `copy_corpus_files()`**. Otherwise `None`.

#### `get_corpus_midis()`

```python
get_corpus_midis(corpus_midis_dirs_tuple: Tuple, verbose: bool = True) -> Dict[str, str]
```

Scans one or more directories for MIDI files (via `TMIDIX.create_files_list`) and returns a mapping from file basename (without extension) to full path. Decorated with `@lru_cache(maxsize=1)`:

- The argument **must be a tuple** (hashable) — pass `tuple(dirs)`, not a list.
- Only the **most recent** call's result is cached; alternating between different directory sets re-scans each time.

**Returns:** `Dict[str, str]` — basename → full path.

#### `copy_corpus_files()`

```python
copy_corpus_files(
    sorted_idxs_sims_list: list[list],
    corpus_midis_dirs: list = ['./Corpus MIDIs Dir/'],
    main_output_dir: str = './Corpus Matches Dir/',
    sub_output_dir: str = '',
    copy_original_midi: bool = True,
    original_midi_path: str = '',
    verbose: bool = True,
) -> str
```

Copies matched corpus MIDI files into an output directory, optionally alongside the original query MIDI. Corpus files are located via `get_corpus_midis()` by basename, so names in the results list must match corpus filenames.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `sorted_idxs_sims_list` | `list[list]` | — | **4-element records** `[rank, name, transpose, similarity]` — as returned by `print_sorted_idxs_sims_list(..., return_as_list=True)`. Raw 3-element entries from `idxs_sims_to_sorted_list()` are **not** accepted and will raise an uncaught `ValueError`. |
| `corpus_midis_dirs` | `list` | `['./Corpus MIDIs Dir/']` | Directories to scan for corpus MIDI files. |
| `main_output_dir` | `str` | `'./Corpus Matches Dir/'` | Root output directory (created if needed). |
| `sub_output_dir` | `str` | `''` | Subdirectory under the root; if empty and `original_midi_path` is set, the query MIDI's basename (no extension) is used. |
| `copy_original_midi` | `bool` | `True` | Copy the query MIDI into the output directory once, as `{sub_output_dir}.mid`. |
| `original_midi_path` | `str` | `''` | Path to the original query MIDI. |
| `verbose` | `bool` | `True` | Print progress and per-file errors. |

Each match is copied as `{similarity}_{transpose}_{name}.mid`, where `similarity` is the (scaled) value from the results list rounded to 8 decimals. Per-file copy failures are caught, reported when `verbose`, and skipped.

**Returns:** `str` — output directory where files were copied (`''` if the results list was empty — in that case nothing, including the original MIDI, is copied).

#### Full search workflow

```python
import torch
from midisimx import (download_model, load_model, midi_to_tokens, get_embeddings_bf16,
                      download_embeddings, load_embeddings, cosine_similarity_topk,
                      idxs_sims_to_sorted_list, print_sorted_idxs_sims_list,
                      copy_corpus_files)

# 1) Model + query MIDI (12 transpose variants by default)
model, ctx, dtype = load_model(download_model(), device='cuda')
seqs = midi_to_tokens('my_song.mid')

# 2) Embed every variant — each row of the query batch is one transpose
embs = get_embeddings_bf16(model, seqs, normalize=True)      # (12, D), CPU

# 3) Corpus + chunked top-k search
names, corpus = load_embeddings(download_embeddings())
idxs, sims = cosine_similarity_topk(embs, corpus, topk=16)   # (12, 16) NumPy arrays

# 4) Rank matches across all transpose variants (best variant per corpus MIDI kept)
results = idxs_sims_to_sorted_list(idxs, sims)               # sorted, deduplicated
records = print_sorted_idxs_sims_list(results, names, return_as_list=True)

# 5) Copy the best matches — and the query MIDI — to disk
out_dir = copy_corpus_files(records,
                            corpus_midis_dirs=['./Corpus MIDIs Dir/'],
                            original_midi_path='my_song.mid')
```

---

## Helper utilities (`midisimx.helpers`)

Resource discovery, MIDI normalization/hashing, and system-package helpers.

#### `get_package_models()`

Lists model checkpoints (`.pth`) bundled with the package.

**Returns:** `List[Dict]` — `[{'model': <file name>, 'path': <full path>}, …]`, sorted by file name.

#### `get_package_embeddings()`

Lists pre-computed embedding files (`.npy`) bundled with the package.

**Returns:** `List[Dict]` — `[{'embeddings': <file name>, 'path': <full path>}, …]`, sorted by file name.

#### `get_normalized_midi_md5_hash()`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `midi_file` | `str` | — | Path to a MIDI file. |

Computes the file's raw MD5 plus a **normalized** MD5 obtained by round-tripping the MIDI through the TMIDIX score representation (`midi2score` → `score2midi`). Files with identical musical content but different byte-level encodings produce identical normalized hashes — ideal for deduplication and corpus alignment.

**Returns:** `Dict` — `{'midi_name', 'original_md5', 'normalized_md5'}`.

#### `normalize_midi_file()`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `midi_file` | `str` | — | Source MIDI file. |
| `output_dir` | `str` | `''` | Destination directory (created if needed); defaults to the current working directory. |
| `output_file_name` | `str` | `''` | Output filename; defaults to the source basename. If the target exists, `<name>_normalized.mid` is written instead. |

**Returns:** `str` — path of the written normalized MIDI file.

#### `is_installed()`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `pkg` | `str` | — | Debian/Ubuntu package name. |

Checks installation state via `dpkg-query`.

**Returns:** `bool`.

#### `install_apt_package()`

Idempotently installs an apt package (e.g. `'fluidsynth'`).

| Parameter | Type | Default | Description |
|---|---|---|---|
| `pkg` | `str` | — | Package name. |
| `update` | `bool` | `True` | Run `apt-get update` first (5 attempts, exponential back-off). |
| `timeout` | `int` | `600` | Per-operation timeout in seconds. |
| `require_root` | `bool` | `True` | Prefix commands with `sudo` when not running as root. |
| `use_python_apt` | `bool` | `False` | Try the `python-apt` API first, falling back to subprocess `apt-get`. |

**Returns:** `Dict` — `{'status': …, 'package': …}` where status is `'already_installed'`, `'installed'`, or `'installed_via_python_apt'`.

**Raises:** `PermissionError` (root required but `sudo` unavailable); `RuntimeError` (installation fails after 6 retry attempts with back-off on dpkg locks).

---

## PCA reduction (`midisimx.pca_reduce`)

Streaming, GPU-accelerated dimensionality reduction of large embedding matrices. Only batches are ever copied to the compute device — the full dataset never has to fit in device memory.

**Algorithm:**

1. **Pass 1 (float64)** — stream over batches, accumulating the global mean and full covariance matrix online (exact pairwise update of Chan et al., 1982 when `exact_covariance=True`).
2. **Eigen-decomposition** — `torch.linalg.eigh` on the covariance matrix.
3. **Pass 2 (float32)** — center each batch with the global mean and project onto the top `target_dim` eigenvectors.

#### `pca_reduce_embeddings()`

```python
pca_reduce_embeddings(
    embeddings,                # (n_samples, n_features) array-like or Tensor
    target_dim: int = 128,
    batch_size: int = 512_000,
    *,
    device=None,               # None -> 'cuda' if available, else 'cpu'
    use_tqdm: bool = True,
    verbose: bool = True,
    debug: bool = False,
    save_dir=None,
    exact_covariance: bool = True,
    return_torch: bool = False,
) -> PCAReductionResult
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `embeddings` | array-like / `Tensor` | — | `(n, d)` input; numpy arrays and torch tensors accepted directly. |
| `target_dim` | `int` | `128` | Components to keep (`1 ≤ target_dim ≤ n_features`). |
| `batch_size` | `int` | `512_000` | Rows per batch. Pass-1 device memory ≈ `batch_size × n_features × 8` bytes (float64) — lower for smaller GPUs. |
| `device` | `str` / `torch.device` | `None` | Compute device; `None` → CUDA if available, else CPU. |
| `use_tqdm` | `bool` | `True` | Show a progress bar per pass. |
| `verbose` | `bool` | `True` | Print timings, throughput, and the explained-variance table. |
| `debug` | `bool` | `False` | Extra diagnostics: finiteness checks, eigenvalue spectrum, covariance symmetry, peak GPU memory. |
| `save_dir` | `str` / `Path` | `None` | If set, save all artifacts as `.npy` files to this directory. |
| `exact_covariance` | `bool` | `True` | Exact online covariance update; `False` uses the cheaper batch-mean-centered approximation (underestimates covariance on sorted data). |
| `return_torch` | `bool` | `False` | Return `reduced` as a float32 `torch.Tensor` (sharing memory with the numpy buffer). |

**Returns:** [`PCAReductionResult`](#pcareductionresult)

**Raises:**
- `ValueError` — input not 2-D; fewer than 2 samples; `target_dim` out of range; CUDA requested but unavailable; zero or non-finite total variance (constant input or NaNs).
- `RuntimeWarning` — `target_dim` exceeds the achievable rank `min(n−1, d)`, or (debug mode) the first batch contains non-finite values.

**Artifacts written when `save_dir` is set:**

| File | Contents |
|---|---|
| `pca_mean.npy` | Global mean, `(d,)` float64 |
| `pca_cov.npy` | Covariance matrix, `(d, d)` float64 |
| `pca_eigvals.npy` | Eigenvalues (descending), `(d,)` float64 |
| `pca_eigvecs.npy` | Eigenvectors (columns), `(d, d)` float64 |
| `pca_projection_matrix.npy` | Top-`target_dim` eigenvectors, `(d, target_dim)` float32 |
| `embeddings_reduced.npy` | Reduced embeddings, `(n, target_dim)` float32 |

**Notes:**
- Statistics are computed in float64; the final projection runs in float32 (matching scikit-learn's numerics).
- Eigenvalues are clamped at zero (PSD covariance + floating-point noise).
- All computation runs under `torch.no_grad()`.
- The reduced result lives in host RAM — ensure `n × target_dim × 4` bytes fit in memory.

```python
from midisimx.pca_reduce import pca_reduce_embeddings

result = pca_reduce_embeddings(corpus, target_dim=128, save_dir='pca_out')
print(result.reduced.shape)                      # (N, 128)
print(result.cumulative_explained_variance[127]) # variance kept by 128 components
```

#### `PCAReductionResult`

Dataclass returned by `pca_reduce_embeddings()`:

| Attribute | Shape / Type | Description |
|---|---|---|
| `reduced` | `(n, target_dim)` float32 | Projected embeddings (`torch.Tensor` if `return_torch=True`). |
| `mean` | `(d,)` float64 | Global mean of the input. |
| `covariance` | `(d, d)` float64 | Full covariance (unbiased, `n−1` denominator). |
| `eigenvalues` | `(d,)` float64 | Descending, clamped at zero. |
| `eigenvectors` | `(d, d)` float64 | As columns; `cov ≈ V · diag(λ) · Vᵀ`. |
| `projection_matrix` | `(d, target_dim)` float32 | Top eigenvectors `W`; projection is `(x − mean) @ W`. |
| `explained_variance_ratio` | `(d,)` | Fraction of total variance per component (descending). |
| `cumulative_explained_variance` | `(d,)` | Cumulative sum of the above. |
| `n_samples`, `input_dim`, `target_dim` | `int` | Dataset dimensions. |
| `device` | `str` | Device used for computation. |
| `timings` | `Dict[str, float]` | Wall-clock seconds: `pass1_mean_cov`, `eigendecomposition`, `pass2_projection`, `total`. |

---

## License

Copyright 2026 Project Los Angeles / Tegridy Code. Licensed under the **Apache License, Version 2.0** — see [LICENSE](http://www.apache.org/licenses/LICENSE-2.0).

**Project Los Angeles** · Tegridy Code · <https://github.com/Tegridy-Code/Project-Los-Angeles>