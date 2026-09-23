## midisimx API Functions Index

### Core module — `midisimx`

- `midisimx.copy_corpus_files` — *Copy matched corpus MIDIs to an output directory as `{sim}_{transpose}_{name}.mid`, optionally alongside the original query MIDI.*
- `midisimx.cosine_similarity_topk` — *Compute chunked, GPU-accelerated top-k cosine similarities between query and corpus embeddings, returning per-query top-k indices and values as NumPy arrays.*
- `midisimx.download_all_embeddings` — *Snapshot-download an entire embeddings dataset from a Hugging Face dataset repo to a local directory.*
- `midisimx.download_embeddings` — *Download a single pre-computed embeddings `.npy` file from a Hugging Face dataset repo.*
- `midisimx.download_model` — *Download a pre-trained model checkpoint from a Hugging Face model repo.*
- `midisimx.get_corpus_midis` — *Scan corpus MIDI directories (LRU-cached; dirs passed as a tuple) and return a `{basename: full_path}` dict.*
- `midisimx.get_embeddings_bf16` — *Compute batched embeddings for token sequences with optional bfloat16 autocast, configurable pooling, L2 normalization, and periodic checkpoint saves.*
- `midisimx.idxs_sims_to_sorted_list` — *Convert top-k index/similarity arrays into a similarity-descending list of `(corpus_index, transpose, similarity)` records, deduplicated per corpus MIDI.*
- `midisimx.load_embeddings` — *Load a structured NumPy embeddings file into `(midi_names, midi_embeddings)` arrays.*
- `midisimx.load_model` — *Build the 768-dim Transformer encoder, load checkpoint weights, and return `(model, AMP autocast context, dtype)`.*
- `midisimx.load_tiny_embeddings` — *Load the tiny 128-dim pre-computed embeddings file bundled with the package.*
- `midisimx.load_tiny_model` — *Load the tiny 6.49M-parameter Transformer model bundled with the package.*
- `midisimx.masked_mean_pool` — *Compute masked mean pooling over token embeddings, ignoring padded positions.*
- `midisimx.masked_weighted_mean_aggregated_pool` — *Compute one weighted mean-pooled embedding per token-id range, concatenated into a 2-D tensor or stacked into a 3-D tensor.*
- `midisimx.masked_weighted_mean_pool` — *Compute weighted mean pooling over token embeddings, with per-token weights determined by token-id range weights.*
- `midisimx.midi_to_instruments_list` — *Extract GM instrument programs from a MIDI file, ordered by note count (optionally with counts).*
- `midisimx.midi_to_tokens` — *Convert a MIDI file into compact integer token sequences (one per transpose variant) suitable for model input.*
- `midisimx.pad_and_mask` — *Pad variable-length token sequences to a common length and produce a boolean valid-token mask.*
- `midisimx.print_sorted_idxs_sims_list` — *Pretty-print sorted search results with corpus names, or return them as `[rank, name, transpose, similarity]` records.*
- `midisimx.random_ngram_replace` — *Randomly replace single tokens and n-grams (length 2–`max_ngram`) with a mask value, returning a new sequence and leaving the original intact.*
- `midisimx.save_embeddings` — *Save name strings and embedding vectors into a structured NumPy array, written to disk or returned in memory.*
- `midisimx.tokens_to_midi` — *Decode a token sequence into a TMIDIX score and write it to a MIDI file (melodic or drum-track modes).*

**Constants:** `label_to_id` / `id_to_label` (20 music styles), `bpm_to_id` / `id_to_bpm` (30 tempo entries) — lookup dicts for drum-track tokens.

### Helper functions — `midisimx.helpers`

- `midisimx.helpers.get_md5_hash` — *Stream-compute the MD5 hash of a very large file in chunks, with an optional progress bar.*
- `midisimx.helpers.get_normalized_midi_md5_hash` — *Compute the raw and normalization-invariant MD5 hashes of a MIDI file for deduplication and corpus alignment.*
- `midisimx.helpers.get_package_embeddings` — *Return a sorted list of pre-computed embeddings files bundled with the package (`{'embeddings', 'path'}` dicts).*
- `midisimx.helpers.get_package_models` — *Return a sorted list of model checkpoints bundled with the package (`{'model', 'path'}` dicts).*
- `midisimx.helpers.get_sha256_hash` — *Stream-compute the SHA-256 hash of a large file with optional progress display.*
- `midisimx.helpers.install_apt_package` — *Idempotently install an apt package with retries, optional `apt-get update`, sudo escalation, and optional python-apt fallback.*
- `midisimx.helpers.is_installed` — *Return True if a Debian/Ubuntu (dpkg) package is already installed.*
- `midisimx.helpers.normalize_midi_file` — *Normalize a MIDI file via a TMIDIX score round-trip, write it to disk, and return the output path.*
- `midisimx.helpers.sort_aligned_lists` — *In-place stable sort of two aligned (names, sequences) lists by sequence length.*

### Instrumentation similarity — `midisimx.instrumentation_similarity`

- `midisimx.instrumentation_similarity.instrumentation_similarity` — *Deterministic timbre-aware similarity score in [0, 1] between two GM instrumentation lists (0–127 melodic, 128 = drums), with optional strict and diagnostic (detail) modes.*

### Memmap storage — `midisimx.memmap`

- `midisimx.memmap.load_paired_memmap` — *Memory-map a paired names + float32 embeddings `.bin` file with zero parsing and zero RAM overhead.*
- `midisimx.memmap.merge_paired_memmaps` — *Merge multiple paired memmap files into one loadable file (requires matching embedding dimensions).*
- `midisimx.memmap.save_paired_memmap` — *Save string names and a 2-D float32 embeddings array into a single memmap-compatible `.bin` file.*

### PCA reduction — `midisimx.pca_reduce`

- `midisimx.pca_reduce.load_pca_reductor` — *Load a fitted `PCAReductor` from a `.npz` model file (convenience wrapper for `PCAReductor.load`).*
- `midisimx.pca_reduce.pca_reduce_embeddings` — *Reduce an `(n, d)` embeddings matrix to `(n, target_dim)` via streaming, two-pass, GPU-accelerated PCA that never materializes the full dataset on the compute device.*
- `midisimx.pca_reduce.PCAReductionResult` *(dataclass)* — *Container for PCA outputs: reduced embeddings, mean, covariance, eigenvalues/eigenvectors, projection matrix, explained-variance stats, dims, device, timings, and the fitted reductor.*
- `midisimx.pca_reduce.PCAReductor` *(class)* — *Fitted, serializable streaming-PCA reductor: fit once on a reference corpus, then project unseen embeddings without refitting.*
  - `PCAReductor.fit` — *Stream mean/covariance accumulation and eigen-decomposition over a reference corpus; returns `self`.*
  - `PCAReductor.fit_transform` — *Fit on a corpus and return the projection of the same data.*
  - `PCAReductor.transform` — *Batch-streamed projection of unseen embeddings using the fitted statistics.*
  - `PCAReductor.inverse_transform` — *Approximate mapping of reduced embeddings back to the original space.*
  - `PCAReductor.save` — *Persist the fitted state to a single pickle-free `.npz` model file.*
  - `PCAReductor.load` *(classmethod)* — *Restore a fitted reductor from a saved `.npz` file.*
  - `PCAReductor.is_fitted` *(property)* — *True if the reductor holds fitted statistics.*

### LDMB storage — `midisimx.ldmb`

- `midisimx.ldmb.load_ldmb` — *Materialize an entire LDMB file into a plain Python list of dicts (fast path).*
- `midisimx.ldmb.merge_ldmb` — *Merge LDMB files, in-memory dict lists, and/or `MemmapList` views into one file (byte-level copy for same-compression sources).*
- `midisimx.ldmb.open_ldmb` — *Open an LDMB file as a lazy, mmap-backed random-access sequence.*
- `midisimx.ldmb.save_ldmb` — *Stream-serialize an iterable of dicts to an LDMB file with optional per-record zlib compression.*
- `midisimx.ldmb.MemmapList` *(class)* — *Lazy, read-only `Sequence` view of an LDMB file: random access, negative indices, slicing, iteration, `to_list`, and context-manager support.*

### Cross-modal mapping — `midisimx.crossmodal_mapper`

- `midisimx.crossmodal_mapper.CrossModalMapper` *(class)* — *Non-ML, closed-form, bi-directional linear mapper between two row-aligned embedding spaces (methods: `procrustes` / `ridge` / `cca`), enabling e.g. text↔MIDI cross-modal retrieval.*
  - `CrossModalMapper.fit` — *One-shot solve on row-aligned `SRC (n, d_src)` / `TRG (n, d_trg)` matrices; returns `self`.*
  - `CrossModalMapper.src_to_trg` — *Map src-space vectors/matrices into the trg space.*
  - `CrossModalMapper.trg_to_src` — *Map trg-space vectors/matrices into the src space.*
  - `CrossModalMapper.search_trg` — *Cross-modal retrieval: src-space queries vs. a trg-space bank → `(indices, similarities)`.*
  - `CrossModalMapper.search_src` — *Cross-modal retrieval: trg-space queries vs. an src-space bank → `(indices, similarities)`.*
  - `CrossModalMapper.evaluate` — *Recall@k, MRR, and cosine statistics on held-out aligned pairs.*
  - `CrossModalMapper.save` — *Persist the fitted mapper to a `.npz` file.*
  - `CrossModalMapper.load` *(classmethod)* — *Restore a saved mapper.*

---