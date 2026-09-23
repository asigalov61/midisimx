# midisimx — API Reference

**Version:** 26.9.26+ · **Python:** ≥ 3.8 · **License:** Apache-2.0
**Package:** `pip install -U midisimx`

Modules: `midisimx` (core), `midisimx.helpers`, `midisimx.instrumentation_similarity`, `midisimx.memmap`, `midisimx.pca_reduce`. The bundled `TMIDIX` is re-exported as `midisimx.TMIDIX` (e.g. `TMIDIX.create_files_list(dirs)`).

---

## Quick Start

```python
import torch
import midisimx

# Corpus embeddings (bundled, tiny) or downloaded:
# emb_path = midisimx.download_embeddings()
names, corpus_emb = midisimx.load_tiny_embeddings(verbose=False)

# Model (bundled tiny checkpoint, or download_model() + load_model())
model, ctx, dtype = midisimx.load_tiny_model(device='cuda')

# Tokenize query MIDI -> list of transposed variants
seqs = midisimx.midi_to_tokens('song.mid', transpose_factor=6)

# Embeddings (weighted: emphasize pitches & chords)
q = midisimx.get_embeddings_bf16(
    model, seqs, device=torch.device('cuda'),
    pooling='weighted_mean',
    token_type_weights={(128, 256): 2, (384, 718): 2})

# Search
idxs, sims = midisimx.cosine_similarity_topk(q, corpus_emb, verbose=False)
matches = midisimx.print_sorted_idxs_sims_list(
    midisimx.idxs_sims_to_sorted_list(idxs, sims),
    names, return_as_list=True)

midisimx.copy_corpus_files(matches,
                           corpus_midis_dirs=['./Corpus MIDIs Dir/'])
```

---

## Token Format (core vocabulary, 720 tokens)

| Range | Meaning |
|---|---|
| `0–127` | Delta start-time since previous event |
| `128–255` | Pitch (MIDI note 0–127) |
| `256–383` | Duration |
| `384–717` | Note/chord class (384–395 → 12 semitones; 396–716 → 321 chords) |
| `718` | Mask token · `719` | Pad token |

**Note event:** `[delta, note_tok, pitch, duration]` (4 tokens).
**Chord event:** `[delta, chord_tok, pitch, dur, pitch, dur, ...]`.

---

## 1. `midisimx` — Core Module

### Downloading

```python
download_all_embeddings(repo_id='projectlosangeles/midisimx-embeddings',
                        revision='main', local_dir='./midisimx-embeddings/',
                        verbose=True, **kwargs) -> str
```
Snapshot-downloads an entire embeddings dataset repo. Returns output dir path.

```python
download_embeddings(repo_id='projectlosangeles/midisimx-embeddings',
                    filename='lakh_midi_dataset_17203_clean_midis_embeddings_1_2_1_2_weighted_cc_by_nc_sa.npy',
                    local_dir='./midisimx-embeddings/', verbose=True, **kwargs) -> str
```
Downloads a single embeddings `.npy`. Returns file path.

```python
download_model(repo_id='projectlosangeles/midisimx',
               filename='midisimx_trained_model_14391_steps_0.255_loss_0.9036_acc.pth',
               local_dir='./midisimx-models/', verbose=True, **kwargs) -> str
```
Downloads a model checkpoint. Returns file path. `**kwargs` are forwarded to `huggingface_hub`.

### Model & Embeddings I/O

```python
load_model(model_path='./midisimx-models/midisimx_trained_model_14391_steps_0.255_loss_0.9036_acc.pth',
           dim=768, depth=16, heads=12, max_seq_len=3072, pad_idx=719,
           dtype=torch.bfloat16, device='cuda',
           compile_model=False, dynamic_compile=True, verbose=True)
    -> (model, ctx, dtype)
```
Builds a `TransformerWrapper`/`Encoder` (rotary pos-emb, flash attention), loads the state dict, moves to `device`, sets `eval()`. Returns `(model, torch.amp.autocast ctx, dtype)`. Use `with ctx: out = model(x)`. `compile_model=True` applies `torch.compile(dynamic=dynamic_compile)`.

```python
load_tiny_model(dim=128, depth=24, heads=4, max_seq_len=3072, pad_idx=719,
                dtype=torch.bfloat16, device='cuda',
                compile_model=False, dynamic_compile=True, verbose=True)
    -> (model, ctx, dtype)
```
Same as `load_model` but loads the tiny model **bundled with the package** (6.49M params, 128-dim).

```python
load_embeddings(embeddings_path, midi_names_key='midi_names',
                midi_embeddings_key='midi_embeddings', verbose=True)
    -> (names: np.ndarray, embeddings: np.ndarray)
```
Loads a structured `.npy` file saved by `save_embeddings`.

```python
load_tiny_embeddings(midi_names_key='midi_names',
                     midi_embeddings_key='midi_embeddings', verbose=True)
    -> (names, embeddings)
```
Loads the tiny 128-dim embeddings bundled with the package.

```python
save_embeddings(embeddings_name_strings, embeddings,
                name_strings_key='midi_names', embeddings_key='midi_embeddings',
                output_file_name='saved_midi_embeddings.npy',
                return_merged_array=False, verbose=True) -> np.ndarray | None
```
Builds a structured array (dtype `[(names, object), (embs, float32, (D,))]`), casts embeddings to float32, and saves with `np.save` — or returns it if `return_merged_array=True`. Accepts `torch.Tensor` or `np.ndarray` (tensor inputs are converted via `.cpu().numpy()`).

### MIDI → Tokens

```python
midi_to_tokens(midi_file_path, max_seq_len=3072, transpose_factor=6,
               clean_midi=True, return_drum_track=False,
               drum_track_style='pop', drum_track_bpm=120, verbose=True)
    -> list[list[int]] | (list[list[int]], list[int])
```
Converts a MIDI file into compact token sequences:
- `transpose_factor` (clamped 0–6): generates one sequence per transpose in `[-tf, tf-1]`; `0` → original only.
- `clean_midi=True`: keeps only lead/bass instruments (`TMIDIX.CLEAN_INSTRUMENTS`), then solo-piano extraction.
- Sequences truncated to `max_seq_len`; returns `[]` on failure / no notes.
- `return_drum_track=True`: also returns a drum token list `[style_id, bpm_id, 0, ...]` using style/bpm ids from `label_to_id`/`bpm_to_id`. Drum encoding: pitch+256, duration+384, velocity+640, time as-is.

```python
midi_to_instruments_list(midi_file_path, return_instruments_counts=False)
    -> list[int] | list[tuple[int, int]]
```
Returns GM program numbers ordered by note count (most used first), or `(program, count)` tuples if `return_instruments_counts=True`. Empty list if no notes.

### Tokens → MIDI

```python
tokens_to_midi(tokens, add_chords_labels=True, custom_labels=None,
               output_signature='midisimx', track_name='Project Los Angeles',
               output_fname='midisimx_composition', input_is_drum_track=False,
               return_score=False, verbose=False) -> list | None
```
Decodes tokens into a TMIDIX score and **writes a MIDI file** via `TMIDIX.Tegridy_ms_SONG_to_MIDI_Converter`. Melodic interpretation: `0–127` time (×32 ms), `128–255` pitch, `256–383` duration (×32 ms), `384–716` chord text labels (if `add_chords_labels`). With `input_is_drum_track=True`: time `0–255` (×16), pitch `256–383`, duration `384–639` (×16), velocity `640–767`, notes on channel 10. `custom_labels`: dict `{abs_time_ms: str}` appended as text events. Returns the score (`song_f`) if `return_score=True`, else `None`.

### Embedding Computation & Pooling

```python
get_embeddings_bf16(model, sequences, seq_len=3072, seq_pad_idx=719,
                    batch_size=16, save_every_num_batches=-1,
                    save_file_path='saved_embeddings.npy', device=None,
                    normalize=False, pooling='auto', token_type_weights=None,
                    concat_aggregated_embeddings=True, use_bfloat16=True,
                    return_dtype='float32', return_numpy=False,
                    verbose=True, show_progress_bar=True) -> Tensor | np.ndarray
```
Batched embedding extraction under `torch.inference_mode()` with optional bfloat16 autocast. The model must support `model(x, return_embeddings=True, mask=mask)` returning `(B, D)` or `(B, L, D)`.
- `pooling`: `'auto'`/`'mean'` (masked mean), `'weighted_mean'`, `'weighted_mean_aggregated'`.
- `token_type_weights`: dict `{(start, end): weight}` (end exclusive) or legacy tuple `(onset_w, duration_w, pitch_w)` → ranges `[0,128)`, `[128,256)`, `[256,384)`. Recommended for music emphasis: `{(128, 256): 2, (384, 718): 2}`.
- `normalize`: L2-normalize (float32). `return_dtype`: `'float32'|'float16'`.
- `save_every_num_batches > 0`: periodic `np.save` checkpoint.
- Returns CPU tensor (or numpy array). Shapes: `(N, D)`, `(N, D·R)` (aggregated+concat), or `(N, R, D)` (aggregated, no concat).

**Pooling primitives** (also public):
- `masked_mean_pool(token_embeddings, mask, dim=1, eps=1e-9, verbose=True) -> (B, D)` — mean over valid tokens.
- `masked_weighted_mean_pool(token_embs, valid_mask, token_ids=None, token_type_weights=None, dim=1, verbose=False) -> (B, D)` — per-token weights by id range; falls back to plain mean if no `token_ids`/weights.
- `masked_weighted_mean_aggregated_pool(token_embs, valid_mask, token_ids=None, token_type_weights=None, dim=1, concat=True, verbose=False)` — one pooled vector per range; `concat=True` → `(B, D·R)`, else `(B, R, D)`.
- `pad_and_mask(sequences, pad_idx=719, seq_len=None, device=None, verbose=False) -> (x: LongTensor (B,T), mask: BoolTensor (B,T))` — pads/truncates; `True` marks real tokens.

### Similarity Search

```python
cosine_similarity_topk(query_embs, corpus_embs, topk=16, chunk_size=10000,
                       device=None, use_gpu_if_available=True,
                       normalize_inputs=True, return_dtype=torch.float32,
                       use_fp32_accumulation=True, verbose=True)
    -> (idxs: np.ndarray (Q, k), vals: np.ndarray (Q, k))
```
Chunked, GPU-accelerated top-k cosine similarity between `(Q, D)` queries and `(N, D)` corpus (numpy or torch; numpy corpus is chunk-converted to limit GPU memory). Results on CPU. Raises `ValueError` on non-2-D input or dimension mismatch.

```python
idxs_sims_to_sorted_list(idxs, sims, sims_mult=100, remove_dupes=True)
    -> list[[corpus_index, transpose_value, similarity]]
```
Flattens per-transpose-variant results into one similarity-descending list. Transpose values map to the query's token variants (`[-tf, tf-1]` ordering). `sims_mult` scales scores (default ×100). `remove_dupes=True` keeps only the best transpose per corpus MIDI.

```python
print_sorted_idxs_sims_list(sorted_idxs_sims_list, corpus_midi_names,
                            return_as_list=False) -> list | None
```
Pretty-prints `#rank name --- transpose --- sim`, or returns `[[rank, name, transpose, sim], ...]` for `copy_corpus_files`.

```python
get_corpus_midis(corpus_midis_dirs_tuple, verbose=True) -> dict  # LRU-cached
```
Takes a **tuple** of dirs (hashable for caching). Returns `{basename_no_ext: full_path}`.

```python
copy_corpus_files(sorted_idxs_sims_list, corpus_midis_dirs=['./Corpus MIDIs Dir/'],
                  main_output_dir='./Corpus Matches Dir/', sub_output_dir='',
                  copy_original_midi=True, original_midi_path='',
                  verbose=True) -> str
```
Copies matched corpus MIDIs as `{sim}_{transpose}_{name}.mid` into `main_output_dir/sub_output_dir/`; optionally copies the query MIDI alongside (once). Input list = output of `print_sorted_idxs_sims_list(..., return_as_list=True)`. Returns output dir path.

### Utilities

```python
random_ngram_replace(seq, prob_single=0.10, prob_ngram=0.10, max_ngram=5,
                     replace_value=718, rng=None) -> list[int]
```
Randomly replaces single tokens and n-grams (length 2–`max_ngram`) with `replace_value` (mask token). Returns a new list; original untouched. For MLM-style augmentation.

**Constants:** `label_to_id` / `id_to_label` (20 styles: afrobeat…unknown), `bpm_to_id` / `id_to_bpm` (30 entries, 50–290 + none/unknown).

---

## 2. `midisimx.helpers`

```python
get_package_models() -> [{'model': name, 'path': full_path}, ...]  # sorted
get_package_embeddings() -> [{'embeddings': name, 'path': full_path}, ...]  # sorted
```
Enumerate `.pth` checkpoints / `.npy` embeddings bundled with the package.

```python
sort_aligned_lists(midi_names, midi_sequences, reverse=False) -> None
```
**In-place** stable sort of two aligned lists by sequence length (shortest first; longest first if `reverse=True`). Sort your corpus by length to speed up embedding.

```python
get_normalized_midi_md5_hash(midi_file) -> {'midi_name', 'original_md5', 'normalized_md5'}
```
MD5 of raw bytes + MD5 after a TMIDIX score round-trip (normalization-invariant hash for deduplication).

```python
normalize_midi_file(midi_file, output_dir='', output_file_name='') -> str
```
Normalizes a MIDI via TMIDIX and writes it to disk (default: cwd, original name; writes `<name>_normalized.mid` if the target exists). Returns output path.

```python
get_md5_hash(path, chunk_size=8*1024*1024, verbose=True) -> str
get_sha256_hash(path, chunk_size=8*1024*1024, verbose=True) -> str
```
Streaming hashes for very large files with tqdm progress; never loads more than `chunk_size` bytes.

```python
is_installed(pkg) -> bool                       # dpkg-query check
install_apt_package(pkg, update=True, timeout=600,
                    require_root=True, use_python_apt=False) -> {'status', 'package'}
```
Idempotent apt install with retries, sudo escalation, and optional python-apt path. Status: `already_installed` / `installed` / `installed_via_python_apt`. Linux/Debian only.

---

## 3. `midisimx.instrumentation_similarity`

```python
instrumentation_similarity(src, trg, *, strict=False, detail=False)
    -> float | (float, dict)
```
Deterministic, dependency-free similarity in `[0.0, 1.0]` between two instrumentations (GM programs: 0–127 melodic, 128 = drums). Duplicates/order ignored; junk elements dropped (or `ValueError` if `strict=True`); `None`/non-iterable → `TypeError`.

**Scoring:** bidirectional soft timbre affinity (exact = 1.0, ranked gentle substitutes = 0.95/0.85/0.75, same GM family = 0.30, unrelated = 0.0) + fading Jaccard exact-overlap bonus (×0.25); final = `0.9 × melodic + 0.1 × drum` when drums apply. Shared drums alone cap the score at 0.1.

With `detail=True`, returns `(score, dict)` with keys: `score, melodic_score, soft_affinity, jaccard, drum_score, src_melodic, trg_melodic, src_has_drums, trg_has_drums`.

```python
>>> instrumentation_similarity([0, 40, 128], [0, 40, 128])   # identical
1.0
>>> instrumentation_similarity([40], [41])                   # Violin vs Viola
0.95
>>> instrumentation_similarity([0, 40], [0, 40, 128])        # drum mismatch
0.9
>>> instrumentation_similarity([0, 40, 128], [65, 80, 128])  # incompatible + shared drums
0.1
```

---

## 4. `midisimx.memmap`

Custom single-file binary format (names + float32 embeddings) natively compatible with `numpy.memmap` — instant loading of massive datasets with zero RAM overhead. Fixed 64 KB header (`HEADER_SIZE = 65536`, Windows-safe alignment).

```python
save_paired_memmap(filepath, names, embeds, verbose=False)
```
Writes names (cast to fixed-width Unicode) + 2-D float32 embeds. `len(names)` must equal `len(embeds)`.

```python
load_paired_memmap(filepath, verbose=False) -> (names_memmap, embeds_memmap)
```
Memory-maps both arrays (read-only). No parsing, no RAM copy.

```python
merge_paired_memmaps(filepaths, output_filepath, verbose=False)
```
Merges multiple `.bin` files into one loadable file. All inputs must share embedding dimension `D` (else `ValueError`); differing name widths are re-cast (chunked to protect RAM). Empty input list → `ValueError`.

```python
save_paired_memmap('a.bin', ['x', 'y'], np.random.randn(2, 128).astype(np.float32))
names, emb = load_paired_memmap('a.bin')   # emb[0] -> vector, names[0] -> 'x'
```

---

## 5. `midisimx.pca_reduce`

Streaming, GPU-accelerated two-pass PCA: **pass 1** (float64) accumulates exact online mean/covariance (Chan et al. 1982); eigen-decomposition via `torch.linalg.eigh`; **pass 2** (float32) projects in batches. The full dataset never materializes on device.

```python
pca_reduce_embeddings(embeddings, target_dim=128, batch_size=512_000, *,
                      device=None, use_tqdm=True, verbose=True, debug=False,
                      save_dir=None, exact_covariance=True, return_torch=False)
    -> PCAReductionResult
```
Reduces `(n, d)` embeddings (numpy or torch) to `(n, target_dim)`. `device=None` → CUDA if available. `save_dir` (optional) writes all artifacts (`pca_mean.npy`, `pca_cov.npy`, `pca_eigvals.npy`, `pca_eigvecs.npy`, `pca_projection_matrix.npy`, `embeddings_reduced.npy`, `pca_reductor.npz`). Raises `ValueError` if input is not 2-D, has < 2 rows, `target_dim` ∉ `[1, d]`, CUDA requested but unavailable, or zero/non-finite variance.

**`PCAReductionResult`** (dataclass): `reduced`, `mean`, `covariance`, `eigenvalues`, `eigenvectors`, `projection_matrix`, `explained_variance_ratio`, `cumulative_explained_variance`, `n_samples`, `input_dim`, `target_dim`, `device`, `timings`, `reductor`.

### `PCAReductor` — fit once, reuse everywhere

```python
PCAReductor(target_dim=128)
```

| Method | Signature | Notes |
|---|---|---|
| `fit` | `(embeddings, batch_size=512_000, *, device=None, exact_covariance=True, use_tqdm=True, verbose=True, debug=False, save_dir=None) -> self` | Streams stats + eigen-decomposition; stores fitted state |
| `transform` | `(embeddings, batch_size=512_000, *, device=None, use_tqdm=True, verbose=False, debug=False, return_torch=False) -> ndarray \| Tensor` | Projects unseen data; accepts 1-D (single vector) or 2-D; `RuntimeError` if unfitted, `ValueError` on dim mismatch |
| `fit_transform` | `(embeddings, ...) -> ndarray \| Tensor` | `fit` + `transform` on the same data |
| `inverse_transform` | `(reduced, ...) -> ndarray \| Tensor` | Approximate reconstruction `(x @ Wᵀ) + mean` |
| `save` | `(filepath, *, include_covariance=True, compress=True, verbose=True) -> str` | Pickle-free `.npz` (arrays + JSON metadata) |
| `load` *(classmethod)* | `(filepath, *, validate=True) -> PCAReductor` | Restores fitted state; `FileNotFoundError`/`ValueError` on bad files |
| `__call__` / `is_fitted` | | Shorthand for `transform`; fitted-state property |

```python
load_pca_reductor(filepath) -> PCAReductor   # convenience wrapper for load()
```

```python
reductor = PCAReductor(target_dim=64).fit(train_emb)
reductor.save('pca_reductor.npz')
reductor2 = load_pca_reductor('pca_reductor.npz')
reduced_new = reductor2.transform(unseen_emb)  # no refit
```

---

### Notes

- Similarity covers start-times, durations, pitches, and chords. Channels, instruments, velocities, and drum similarity are **not** part of the model-based score (use `instrumentation_similarity` for instrumentation comparison).
- Model context: 3 072 tokens (~1 000 notes); longer MIDIs are truncated.
- Solo drum-track MIDIs cannot be embedded; drum tracks require the dedicated token path (`midi_to_tokens(..., return_drum_track=True)` / `tokens_to_midi(..., input_is_drum_track=True)`).
- `get_corpus_midis` is LRU-cached (`maxsize=1`); pass dirs as a tuple.

---

## 6. `midisimx.ldmb`

Memory-mappable binary storage for **large lists of dicts** whose contents are arbitrary Python objects (nested dicts/lists/tuples, `str`, `int`, `float`, `bool`, `None`). Each record is pickled independently with its own offset/length in a footer index — so individual records can be read lazily via `mmap` with zero parsing of the rest of the file, and files can be **merged at the byte level** (payloads block-copied, only index offsets rewritten).

**File format (little-endian, v1):**

| Section | Size | Contents |
|---|---|---|
| Header | 8 B | magic `b"LDMB"` · u16 version · u8 flags (bit 0 = zlib) · u8 zlib level |
| Records | var | N records, each pickled (`pickle.HIGHEST_PROTOCOL`), optionally zlib-compressed |
| Index | 16·N B | N × (u64 absolute file offset, u64 byte length) |
| Trailer | 20 B | u64 index offset · u64 count · magic `b"LDMB"` |

Minimum valid file size: 28 B. I/O granularity: 4 MiB. Optional `numpy` accelerates merge index math (pure-Python fallback otherwise).

> **Naming caveat:** the module's `__all__` lists short aliases (`save`, `load`, `open_ldd`, `merge`) that are **not defined** in the current version. Use the `*_ldmb` names below (or `import midisimx.ldmb as ldmb`); `from midisimx.ldmb import *` will raise `AttributeError`.

### Save / Load / Open

```python
save_ldmb(records, path, *, compress=None) -> int
```
Stream-serializes an iterable of dicts (generators accepted — never materialized in RAM). `compress`:
- `None` / `0` / `False` → raw pickle: fastest, best for memmap-style access.
- `1..9` → per-record zlib: smaller file, still random-access and mergeable.
- `True` → zlib level 6.

Returns the number of records written. Records must be picklable.

```python
load_ldmb(path) -> list
```
Materializes the whole file into a plain Python list (fast path). `path` may be a file path or an existing `MemmapList` (closed afterward only if it was opened by this call).

```python
open_ldmb(path) -> MemmapList
```
Opens an LDMB file as a lazy, mmap-backed random-access sequence. Raises `ValueError` if the file is too small, not an LDMB file, corrupt (bad magic / size mismatch), or an unsupported version.

### Merge

```python
merge_ldmb(sources, out_path, *, compress=None) -> int
```
Merges into one LDMB file. Each source may be:
- a file path (`str` / `os.PathLike`),
- an already-open `MemmapList` (its underlying path is re-read),
- an in-memory `list`/`tuple` of dicts (must be sized — generators raise `TypeError`; use `save_ldmb` for those).

Behavior:
- Sources stored with the same compression as the output are merged **byte-for-byte** (4 MiB block copies + index offset rewrite, numpy-vectorized when available); mixed-compression sources are re-encoded per record on the fly.
- Output compression: explicit `compress` arg (`None`/`0` → raw, `1..9` → zlib, `True` → 6); if `None`, it is **inherited from the first file source**, so merging same-format files is a pure byte copy.
- Raises `ValueError` if any source path resolves to `out_path` itself; `TypeError` for unsupported source types.

Returns the total record count written. **Append pattern:** `merge_ldmb(['old.ldmb', new_records], 'new.ldmb')` (output must differ from input).

### `MemmapList`

```python
MemmapList(path)  # or open_ldmb(path)
```
A lazy, read-only `Sequence` view backed by an `mmap`. Reading `ml[i]` touches only that record's 16-byte index entry and its own bytes. **Safe for concurrent readers.** Index is kept in RAM (16 B/record).

| Member | Description |
|---|---|
| `len(ml)` | Number of records |
| `ml[i]` | Random access; negative indices supported; `IndexError` when out of range |
| `ml[a:b]` | Slice → plain `list` of records |
| `iter(ml)` | Lazily yields all records in order |
| `ml.to_list()` | Materialize everything into a `list` |
| `ml.close()` | Release the mmap and file handle (idempotent; swallowed `BufferError`) |
| `with open_ldmb(p) as ml:` | Context-manager support (auto-`close`) |
| `ml.path` | Source file path |

### Example

```python
from midisimx import ldmb

records = [{'midi_md5': 'a' * 32, 'score_notes': 324, 'counts': {k: k for k in range(200)}},
           {'midi_md5': 'b' * 32, 'score_notes': 512}]

ldmb.save_ldmb(records, 'corpus.ldmb')                 # raw pickle (fastest)
ldmb.save_ldmb(records, 'corpus.z.ldmb', compress=6)   # smaller, still random-access

with ldmb.open_ldmb('corpus.ldmb') as ml:              # zero-RAM lazy view
    n = len(ml)
    rec = ml[123]                                      # only this record is unpickled
    chunk = ml[10:20]                                  # slice -> list
    everything = ml.to_list()

# Merge files, mixed formats, and in-memory lists; then append:
total = ldmb.merge_ldmb(['a.ldmb', 'b.ldmb', more_records], 'all.ldmb')
assert ldmb.load_ldmb('all.ldmb') == all_records
ldmb.merge_ldmb(['all.ldmb', new_records], 'all_v2.ldmb')
```

---

### Notes

- Compressed and uncompressed files are equally random-access and mergeable; compression trades load speed for file size.
- Errors: `ValueError` (invalid/corrupt/unsupported-version file; merging a file into itself), `IndexError` (bad index), `TypeError` (bad merge source).
- Unlike `midisimx.memmap` (fixed-schema paired names + embeddings), `ldmb` stores **arbitrary picklable dicts** — e.g. per-MIDI feature/statistics records — and trades numpy-native access for lazy per-record object loading.

*(Suggested edit: remove `ldmb` from the "not covered" list in Notes & Limitations.)*

---

## 7. `midisimx.crossmodal_mapper`

A **non-ML/DL, bi-directional cross-modal embedding mapper** built purely on closed-form linear algebra (NumPy / LAPACK: `eigh`, `svd`, `solve`) — no gradient descent, no epochs, no iterative training. `fit()` is a one-shot solve.

**In the midisimx pipeline**, it bridges the 768-d midisimx MIDI embedding space to any other modality (text, audio, etc.) given **row-aligned pairs** — e.g., LAKH-corpus MIDI embeddings ↔ text embeddings of `"artist - title"` strings, or MIDI ↔ audio embeddings of rendered MIDIs — enabling text→MIDI and MIDI→text cross-modal retrieval with Recall@k / MRR evaluation.

**Inputs:** `SRC (n, d_src)` and `TRG (n, d_trg)`, where row `i` of SRC corresponds to row `i` of TRG. Rectangular dims (`d_src ≠ d_trg`) are fully supported (semi-orthogonal maps, fitted independently per direction).

### Methods

| `method` | Solver | Notes |
|---|---|---|
| `"procrustes"` *(default)* | Whitened orthogonal Procrustes | Best generalisation, exact bidirectionality, scale-adaptive. With whitening on, the SVD singular values equal the canonical correlations of the pair |
| `"ridge"` | Closed-form Tikhonov: `W = (XᵀX + λI)⁻¹ XᵀY` | `λ` is *relative* to the mean eigenvalue (scale-invariant). Useful for rectangular dims / very noisy pairs |
| `"cca"` | Canonical correlation analysis | Maps both sides into a shared latent k-dim space (`P_src`, `P_trg`); `src_to_trg`/`trg_to_src` return shared-space projections; retrieval functions handle this transparently |

### Solver pipeline (procrustes, condensed)

```
1. (optional) L2 row length-normalisation            (scale invariance)
2. (optional) per-side mean-centering
3. (optional) per-side ZCA whitening:
     S = XcᵀXc = E diag(λ) Eᵀ
     W = E diag(max(λ, c·λ_max)^-1/2) Eᵀ   =>   XwᵀXw = I
4. SVD of cross-scatter:  M = XwᵀYw = U Σ Vᵀ
5. cores, reweight power ρ (default 0 = orthogonal Procrustes):
     W_fwd = U_r diag(σ^ρ) V_rᵀ      (d_src × d_trg)
     W_bwd = V_r diag(σ^ρ) U_rᵀ      (d_trg × d_src)
     square dims & ρ=0:  W_bwd = W_fwdᵀ  exactly
6. composed maps:
     M_fwd = W_src · W_fwd · W_trg⁻¹
     M_bwd = W_trg · W_bwd · W_src⁻¹
```

**Inference** (applied internally by `src_to_trg` / `trg_to_src`):
`y_hat = (pre(x) − μ_src) @ M_fwd + μ_trg` and `x_hat = (pre(y) − μ_trg) @ M_bwd + μ_src`. For square, orthogonal configs the model is an **exact bijective linear operator**: `M_bwd = M_fwd⁻¹` (verified to machine precision).

### `CrossModalMapper`

```python
CrossModalMapper(method="procrustes", verbose=1)   # verbose: 0 quiet, 1 normal, 2+ debug
```

| Method | Signature | Description |
|---|---|---|
| `fit` | `(src, trg) -> self` | One-shot solve on row-aligned matrices |
| `src_to_trg` | `(x) -> y_hat` | Map src-space vectors/matrices into trg space |
| `trg_to_src` | `(y) -> x_hat` | Map trg-space vectors/matrices into src space |
| `search_trg` | `(q_src, TRG_BANK, k=10) -> (idx, sims)` | Cross-modal retrieval: src-space queries vs. trg-space bank |
| `search_src` | `(q_trg, SRC_BANK, k=10) -> (idx, sims)` | trg-space queries vs. src-space bank |
| `evaluate` | `(src_test, trg_test) -> report` | Recall@k, MRR, cosine statistics |
| `save` | `("mapper.npz")` | Persist the fitted mapper |
| `load` *(classmethod)* | `("mapper.npz") -> CrossModalMapper` | Restore a saved mapper |

### Example (midisimx context)

```python
from midisimx import crossmodal_mapper

# Row-aligned pairs: one midisimx MIDI embedding per row (pool your choice),
# e.g. from get_embeddings_bf16 / load_embeddings — aligned with text
# embeddings of each MIDI's "artist - title" string from any text encoder.
midi_emb = ...   # (n, 768)
text_emb = ...   # (n, d_txt)

mapper = crossmodal_mapper.CrossModalMapper(method="procrustes",
                                            verbose=1).fit(midi_emb, text_emb)

y_hat = mapper.src_to_trg(midi_emb[:8])    # MIDI  -> text space
x_hat = mapper.trg_to_src(text_emb[:8])    # text  -> MIDI space

# Text query -> MIDI retrieval (query in trg space, search src/MIDI bank)
idx, sims = mapper.search_src(text_query_emb, corpus_midi_emb, k=10)

# MIDI query -> text retrieval
idx, sims = mapper.search_trg(midi_query_emb, text_bank, k=10)

report = mapper.evaluate(midi_test, text_test)   # Recall@k, MRR, cosines

mapper.save('midi_text_mapper.npz')
mapper2 = crossmodal_mapper.CrossModalMapper.load('midi_text_mapper.npz')
```

### Command line (synthetic-data demo with known ground truth)

```
python crossmodal_mapper.py                          # 768 <-> 768 demo
python crossmodal_mapper.py --dim-src 512 --dim-trg 768 --method ridge
python crossmodal_mapper.py --method cca --cca-dim 256
python crossmodal_mapper.py -v -v                    # + debug diagnostics
python crossmodal_mapper.py --self-test              # consistency tests
```

### Notes

- Requirements: Python ≥ 3.8, NumPy ≥ 1.17; `tqdm` optional (simple fallback progress meter). `float64` (default) recommended for maximum precision; `float32` supported.
- Deterministic: no random initialisation, no epochs; results depend only on the input pair.
- v1.1.0 fixes: retrieval metrics honour the **true ground-truth row indices** when queries are a sampled subset of the bank (previously assumed `query i ↔ bank row i`, silently producing chance-level numbers); stage diagnostics no longer crash when `d_src ≠ d_trg`; well-conditioned synthetic generator; stronger self-tests (machine-precision recovery on the exact-linear path, retrieval assertions on the default path).
- Math references: Schönemann (1966) Procrustes; Mikolov et al. (2013); Xing et al. (2015) whitening; Artetaxe et al. (2018) robust pipeline; Hotelling (1936) CCA; Golub & Van Loan.
- ⚠️ **Documentation caveat:** this section is based on the module header only. Constructor/`fit` keyword options (whitening / centering / length-norm switches, reweight power `ρ`, whitening floor `c`, ridge `λ`, `cca_dim`) exist in the implementation but are not documented here — consult the source for the full signatures.

---
