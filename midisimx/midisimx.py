# midisim Python module

r'''###############################################################################
###################################################################################
#
#
#	midisimx Python module
#	Version 1.0
#
#	Project Los Angeles
#
#	Tegridy Code 2026
#
#   https://github.com/Tegridy-Code/Project-Los-Angeles
#
#
###################################################################################
###################################################################################
#
#   Copyright 2026 Project Los Angeles / Tegridy Code
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
#
###################################################################################
###################################################################################
#
#   Critical dependencies
#
#   !pip install huggingface_hub
#   !pip install ipywidgets
#   !pip install tqdm
#   !pip install scikit-learn
#   !pip install torch
#   !pip install einops
@   !pip install einx
#   !pip install torch-summary
#   !pip install matplotlib
#   !pip install numpy==1.26.4
#
###################################################################################
'''

###################################################################################
###################################################################################

print('=' * 70)
print('Loading midisimx Python module...')
print('Please wait...')
print('=' * 70)

__version__ = '1.0.0'

print('midisimx module version', __version__)
print('=' * 70)

###################################################################################
###################################################################################

import os, copy, math, shutil

os.environ["HF_XET_HIGH_PERFORMANCE"] = "1"

from typing import List, Optional, Union, Tuple, Dict, Any

from functools import lru_cache

import tqdm

import numpy as np

import torch
import torch.nn.functional as F
from torch import Tensor

from .x_transformer_2_3_1 import TransformerWrapper, Encoder

from torchsummary import summary

from . import TMIDIX

from huggingface_hub import hf_hub_download, snapshot_download

###################################################################################

print('=' * 70)
print('PyTorch version:', torch.__version__)
print('=' * 70)

###################################################################################

def download_all_embeddings(repo_id: str = 'projectlosangeles/midisimx-embeddings',
                            revision: str = 'main',
                            local_dir: str = './midisimx-embeddings/',
                            verbose: bool = True,
                            **kwargs: dict[str, Any]
                           ) -> str:

    """
    Helper function that downloads all pre-computed midisimx embeddings from Hugging Face
    
    Returns
    -------
    Output directory path string where all embeddings were downloaded to
    """

    if verbose:
        print('=' * 70)
        print('Downloading all embeddings...')
        print('=' * 70)

    result = snapshot_download(repo_id=repo_id,
                               repo_type='dataset',
                               revision=revision,
                               local_dir=local_dir,
                               **kwargs
                              )

    if verbose:
        print('=' * 70)
        print('Done!')
        print('=' * 70)
    
    return result

###################################################################################

def download_embeddings(repo_id: str = 'projectlosangeles/midisimx-embeddings',
                        filename: str = 'lakh_midi_dataset_17203_clean_midis_embeddings_1_2_1_2_weighted_cc_by_nc_sa.npy',
                        local_dir: str = './midisimx-embeddings/',
                        verbose: bool = True,
                        **kwargs: dict[str, Any]
                       ) -> str:
    
    """
    Helper function that downloads pre-computed midisimx embeddings files from Hugging Face
    
    Returns
    -------
    Downloaded embeddings file path string
    """
    
    if verbose:
        print('=' * 70)
        print('Downloading embeddings...')
        print('=' * 70)

    result = hf_hub_download(repo_id=repo_id,
                             repo_type='dataset',
                             filename=filename,
                             local_dir=local_dir,
                             **kwargs
                            )
    if verbose:    
        print('=' * 70)
        print('Done!')
        print('=' * 70)
    
    return result

###################################################################################

def download_model(repo_id: str = 'projectlosangeles/midisimx',
                   filename: str = 'midisimx_trained_model_14391_steps_0.255_loss_0.9036_acc.pth',
                   local_dir: str = './midisimx-models/',
                   verbose: bool = True,
                   **kwargs: dict[str, Any]
                  ) -> str:
    
    """
    Helper function that downloads pre-trained midisim models from Hugging Face
    
    Returns
    -------
    Downloaded model checkpoint file path string
    """
    
    if verbose:
        print('=' * 70)
        print('Downloading model...')
        print('=' * 70)

    result = hf_hub_download(repo_id=repo_id,
                             repo_type='model',
                             filename=filename,
                             local_dir=local_dir,
                             **kwargs
                            )
    if verbose:    
        print('=' * 70)
        print('Done!')
        print('=' * 70)
    
    return result

###################################################################################

def load_model(model_path: str = './midisimx-models/midisimx_trained_model_14391_steps_0.255_loss_0.9036_acc.pth',
               dim: int = 768,
               depth: int = 16,
               heads: int = 12,
               max_seq_len: int = 3072,
               pad_idx: int = 719,
               dtype: torch.dtype = torch.bfloat16,
               device: str = 'cuda',
               compile_model: bool = False,
               dynamic_compile: bool = True,
               verbose: bool = True
              ) -> str:

    """Load and initialize a preconfigured midisim Transformer model from a checkpoint.
    
    One-line summary
    ----------------
    Create a `TransformerWrapper` with an `Encoder` attention stack, load weights
    from a checkpoint file, move the model to the requested device, set it to
    evaluation mode, and return the model together with an automatic mixed-precision
    (AMP) autocast context and the chosen dtype.
    
    Detailed description
    --------------------
    This helper constructs a Transformer-based model using the provided
    architecture hyperparameters, loads a saved state dictionary from `model_path`
    (using `torch.load`), transfers the model to `device`, and switches it to
    evaluation mode (`model.eval()`). It also creates and returns a `torch.amp.autocast`
    context manager configured for the requested `device` and `dtype`. When
    `verbose` is True, the function prints progress messages and a model summary.
    
    Parameters
    ----------
    model_path : str, optional
        Filesystem path to the saved PyTorch checkpoint (state dict). Default is
        `'midisimx_trained_model_14391_steps_0.255_loss_0.9036_acc.pth'`.
    dim : int, optional
        Hidden dimension size for the encoder attention layers. Default: 512.
    depth : int, optional
        Number of encoder layers (depth of the Transformer encoder). Default: 8.
    heads : int, optional
        Number of attention heads per multi-head attention layer. Default: 8.
    max_seq_len : int, optional
        Maximum sequence length the model supports (positional embedding length).
        Default: 3072.
    pad_idx : int, optional
        Index reserved for padding tokens. The model's vocabulary size is set to
        `pad_idx + 1`. Default: 719.
    dtype : torch.dtype, optional
        Floating-point dtype used for AMP autocasting (e.g., `torch.bfloat16`,
        `torch.float16`, `torch.float32`). Default: `torch.bfloat16`.
    device : str or torch.device, optional
        Target device for the model (e.g., `'cuda'`, `'cpu'`, or a `torch.device`).
        Default: `'cuda'`
    compile_model : bool
        If True, the model will be compiled using default (eager) torch.compile() mode.
        Default: `'False'`
    dynamic_compile : bool
        If True when compile_model arg is True, the model will be compiled in
        dynamic mode so that it works properly with sequences of different lengths
        Default: `'True'`
    verbose : bool, optional
        If True, print initialization/loading progress and a model summary.
        Default: True.
    
    Returns
    -------
    tuple
        A 3-tuple `(model, ctx, dtype)` where:
        - **model**: the `TransformerWrapper` instance with loaded weights,
          moved to `device` and set to evaluation mode.
        - **ctx**: a `torch.amp.autocast` context manager configured with
          `device_type=device` and `dtype=dtype`. Use this context when running
          inference to enable mixed-precision casting consistent with the model.
        - **dtype**: the `torch.dtype` passed into the function (returned for
          convenience so callers can reuse it when preparing inputs or contexts).
    
    Side effects and notes
    ----------------------
    - The function calls `torch.load(model_path)` and `model.load_state_dict(...)`.
      The checkpoint must be a compatible state dictionary for the constructed
      model architecture; otherwise `model.load_state_dict` may raise a
      `RuntimeError`.
    - The model is moved to `device` via `model.to(device)` and set to evaluation
      mode with `model.eval()`.
    - `num_tokens` is derived from `pad_idx + 1`. Ensure `pad_idx` matches the
      tokenizer/vocabulary used when the checkpoint was created.
    - The `summary(model)` call used when `verbose` is True requires an available
      `summary` function in scope (for example from `torchinfo` or a custom helper).
    - The returned `ctx` is a context manager; to use it:
      ```py
      with ctx:
          outputs = model(inputs)
      ```
    - If `device` is `'cuda'` but CUDA is unavailable, `model.to(device)` will raise
      an error; pass `'cpu'` to run on CPU.
    
    Exceptions
    ----------
    - `FileNotFoundError` or `OSError` if `model_path` does not exist or cannot be read.
    - `RuntimeError` if the checkpoint is incompatible with the model architecture
      (e.g., missing or unexpected keys in the state dict).
    - Any exceptions raised by `model.to(device)` if the device is invalid or
      resources are insufficient.
    
    Example
    -------
    model, amp_ctx, dtype = load_model(
        model_path='checkpoints/midisim.pth',
        dim=768,
        depth=16,
        heads=12,
        max_seq_len=3072,
        pad_idx=719,
        dtype=torch.bfloat16,
        device='cuda',
        verbose=True,
    )
    
    # Inference example
    model_input = ...  # prepare input tensor on the same device
    with amp_ctx:
        logits = model(model_input)
    """

    if verbose:
        print('=' * 70)
        print('midisim model loader')
        print('=' * 70)
        print('Initializing model...')

    ctx = torch.amp.autocast(device_type=device, dtype=dtype)

    model = TransformerWrapper(
                num_tokens=pad_idx+1,
                max_seq_len=max_seq_len,
                attn_layers=Encoder(
                    dim=dim,
                    depth=depth,
                    heads=heads,
                    rotary_pos_emb=True,
                    attn_flash=True,
                ),
    )

    if verbose:
        print('=' * 70)
        print('Loading model checkpoint...')
    
    model.load_state_dict(torch.load(model_path, map_location=device))

    if verbose:
        print('=' * 70)
    
    model.to(device)
    
    if compile_model:
        if verbose:
            print('Compiling model...')
        model = torch.compile(model, dynamic=dynamic_compile)
        
    model.eval()

    if verbose:
        print('Done!')

        print('=' * 70)
        print('Model Summary')
        summary(model)

    return model, ctx, dtype

###################################################################################

def load_embeddings(embeddings_path: str = './midisimx-embeddings/lakh_midi_dataset_17203_clean_midis_embeddings_1_2_1_2_weighted_cc_by_nc_sa.npy',
                    midi_names_key: str = 'midi_names',
                    midi_embeddings_key: str = 'midi_embeddings',
                    verbose: bool = True
                   ) -> Tuple[np.ndarray, np.ndarray]:

    """
    Helper function that loads pre-computed embeddings file

    Returns
    -------
    Tuple of nd.arrays (midi_names_arr, midi_embeddings_arr)
    """

    if verbose:
        print('=' * 70)
        print('Loading embeddings...')
        
    embeddings_data = np.load(embeddings_path, allow_pickle=True)
    
    if verbose:
        print('=' * 70)
        print('Done!')
        print('=' * 70)
        
    return embeddings_data[midi_names_key], embeddings_data[midi_embeddings_key]

###################################################################################

def save_embeddings(embeddings_name_strings: list[str],
                    embeddings: Union[torch.Tensor, np.ndarray],
                    name_strings_key: str = 'midi_names',
                    embeddings_key: str = 'midi_embeddings',
                    output_file_name: str = 'saved_midi_embeddings.npy',
                    return_merged_array: bool = False,
                    verbose=True
                   ) -> Union[np.ndarray, None]:

    """Save a list of name strings and their corresponding embedding vectors into a NumPy structured array
    and optionally persist it to disk.
    
    This function builds a NumPy structured array with two fields (one for the name strings and one for
    the embedding vectors), populates it from the provided inputs, casts embeddings to `np.float32`,
    and either returns the merged structured array or saves it to disk using `np.save`.
    
    Parameters
    ----------
    embeddings_name_strings : list[str]
        Sequence of Python strings that identify each embedding (e.g., filenames, IDs, labels).
        The length of this list determines the number of rows in the resulting structured array.
    embeddings : Union[torch.Tensor, np.ndarray]
        2D array-like of shape `(n, D)` containing the embedding vectors, where `n` must equal
        `len(embeddings_name_strings)` and `D` is the embedding dimensionality. If a `torch.Tensor`
        is provided it will be converted to a NumPy array with `.numpy()` (no automatic `.cpu()`
        or `.detach()` is performed by this function).
    name_strings_key : str, optional
        Field name to use for the name strings in the structured dtype (default: `'midi_names'`).
    embeddings_key : str, optional
        Field name to use for the embedding vectors in the structured dtype (default:
        `'midi_embeddings'`).
    output_file_name : str, optional
        Path (filename) where the structured array will be saved with `np.save` if
        `return_merged_array` is `False` (default: `'saved_midi_embeddings.npy'`).
    return_merged_array : bool, optional
        If `True`, the function returns the constructed structured NumPy array and does not write
        anything to disk. If `False`, the array is saved to `output_file_name` and the function
        returns `None` (default: `False`).
    verbose : bool, optional
        If `True`, print progress and diagnostic messages to stdout (default: `True`).
    
    Returns
    -------
    Union[np.ndarray, None]
        - If `return_merged_array` is `True`: the NumPy structured array of length `n` with dtype
          `[(name_strings_key, object), (embeddings_key, np.float32, (D,))]`.
        - If `return_merged_array` is `False`: `None` (the array is saved to `output_file_name`).
    
    Raises
    ------
    ValueError
        - If `embeddings` does not have a second dimension (i.e., is not 2D) so the embedding
          dimensionality `D` cannot be determined.
        - If the number of rows in `embeddings` does not match `len(embeddings_name_strings)`.
    TypeError
        - If `embeddings_name_strings` is not a sequence with a definable length.
    Exception
        - Any unexpected exceptions raised while reading attributes (e.g., `.shape`, `.dtype`) or
          during `np.save` will propagate to the caller.
    
    Notes
    -----
    - The function constructs a structured dtype where the name field uses Python `object` to allow
      variable-length strings and the embedding field is a fixed-size `np.float32` vector of length `D`.
    - Embeddings are explicitly cast to `np.float32` before assignment; this may change precision.
    - When a `torch.Tensor` is passed, the function calls `.numpy()` directly. If the tensor is on a
      GPU or requires gradient, callers should ensure it is detached and moved to CPU first (e.g.,
      `embeddings.detach().cpu()`), otherwise `.numpy()` may raise an error.
    - The file is written using `np.save`, producing a `.npy` file that can be loaded with `np.load`.
    - Verbose logging is intended for debugging and progress visibility; set `verbose=False` to silence.
    
    Example
    -------
    >>> # embeddings as numpy array
    >>> names = ['song_a.mid', 'song_b.mid']
    >>> embs = np.random.randn(2, 512)
    >>> save_embeddings(names, embs, output_file_name='embs.npy', verbose=False)
    >>> # embeddings as torch tensor, return array instead of saving
    >>> import torch
    >>> t = torch.randn(2, 512)
    >>> arr = save_embeddings(names, t, return_merged_array=True, verbose=False)
    >>> assert arr.dtype == np.dtype([('midi_names', object), ('midi_embeddings', np.float32, (512,))])
    
    """

    if verbose:
        print('=' * 70)
        print('Saving embeddings...')
        print('=' * 70)
        print("[save_embeddings]: called with parameters:")
        print(f"  number of name strings provided: {len(embeddings_name_strings)}")
        print(f"  output_file_name: {output_file_name}")
        print(f"  name_strings_key: {name_strings_key}")
        print(f"  embeddings_key: {embeddings_key}")
        print(f"  return_merged_array: {return_merged_array}")
        print(f"  verbose: {verbose}")
        print('=' * 70)

    # Convert torch tensor to numpy if needed
    if type(embeddings) == torch.Tensor:
        if verbose:
            print("[save_embeddings]: embeddings is a torch.Tensor, converting to numpy array with .numpy()")
        embeddings = embeddings.cpu().numpy()
    elif type(embeddings) == list:
        if verbose:
                print("[save_embeddings]: embeddings is a list, converting to numpy array")
        embeddings = np.array(embeddings)
    else:
        if verbose:
            print(f"[save_embeddings]: embeddings is of type {type(embeddings)}; no conversion performed")

    # Basic shape and length checks
    try:
        n = len(embeddings_name_strings)
    except Exception as e:
        if verbose:
            print("[save_embeddings]: ERROR computing length of embeddings_name_strings:", e)
        raise

    try:
        D = embeddings.shape[1]
    except Exception as e:
        if verbose:
            print("[save_embeddings]: ERROR reading embeddings.shape[1]:", e)
            print("  embeddings.shape is:", getattr(embeddings, "shape", None))
        raise

    if verbose:
        print(f"[save_embeddings]: determined n = {n} (number of entries)")
        print(f"[save_embeddings]: determined D = {D} (embedding dimensionality)")
        print("[save_embeddings]: preparing numpy structured dtype for storage")

    dtype = np.dtype([
        (name_strings_key, object),              # variable-length Python strings
        (embeddings_key, embeddings.dtype, (D,))       # fixed-size embedding vector
    ])

    if verbose:
        print("[save_embeddings]: dtype constructed as:")
        print(f"  {dtype}")

    # Create empty structured array
    if verbose:
        print(f"[save_embeddings]: allocating empty numpy array of length {n} with dtype above")
    arr = np.empty(n, dtype=dtype)

    # Fill name strings
    if verbose:
        print("[save_embeddings]: assigning name strings to structured array")
        print(f"  first 5 name strings (or fewer): {embeddings_name_strings[:5]}")
    arr[name_strings_key] = embeddings_name_strings

    # Cast embeddings to float32 and assign
    if verbose:
        print("[save_embeddings]: assigning embeddings to structured array")
        print(f"  embeddings original dtype: {getattr(embeddings, 'dtype', 'unknown')}")
        print(f"  embeddings shape: {getattr(embeddings, 'shape', 'unknown')}")
    arr[embeddings_key] = embeddings

    if return_merged_array:
        if verbose:
            print('=' * 70)
            print("[save_embeddings]: return_merged_array is True; returning the merged structured array without saving to disk")
            print(f"  returning array with length {len(arr)} and dtype {arr.dtype}")
            print('=' * 70)
            print('Done!')
            print('=' * 70)
        return arr

    # Save to disk
    if verbose:
        print('=' * 70)
        print(f"[save_embeddings]: return_merged_array is False; saving structured array to '{output_file_name}' using np.save")
    np.save(output_file_name, arr)
    if verbose:
        print(f"[save_embeddings]: save complete. File written: {output_file_name}")
        print(f"  saved array length: {len(arr)}; dtype: {arr.dtype}")
        print('=' * 70)
        print('Done!')
        print('=' * 70)
        
###################################################################################

def midi_to_tokens(midi_file_path: str,
                   max_seq_len: int = 3072,
                   transpose_factor: int = 6,
                   clean_midi: bool = True,
                   verbose: bool = True
                  )-> list[list[int]]:
    
    """
    Convert a single-track MIDI file into one or more compact token sequences suitable for model input.

    This function performs a sequence of TMIDIX preprocessing steps to extract an
    "enhanced score" from a MIDI file, normalizes and clips timing/pitch values,
    optionally generates transposed variants, and encodes events into a compact
    integer token stream.

    Key processing stages
    - Load MIDI and convert to a single-track millisecond score.
    - Produce an enhanced-score with sustain applied.
    - Extract solo-piano notes and recalculate/augment timings.
    - Remove duplicate pitches and fix note durations.
    - Convert to a delta-style event list and clip timing values to 0..127.
    - For each transpose variant, build a token sequence where:
        * nonzero delta times are appended as-is (0..127),
        * chords are encoded as (note/chord + 384),
        * note events are encoded as two tokens: (pitch + 128) and (duration + 256).
      The initial token of each sequence is 0 (start token).

    Parameters
    ----------
    midi_file_path : str
        Path to the MIDI file to process. The file is read by TMIDIX.midi2single_track_ms_score.
    max_seq_len : int
        Maximum output tokens sequence length (truncated to this value). Default is 3072
    transpose_factor : int, optional
        Maximum semitone transpose range. The value is clamped to the inclusive range 0..6.
        If > 0, the function generates variants for transpositions in the integer range
        [-transpose_factor, transpose_factor - 1]. If 0, only the original (no transpose)
        variant is produced. Default is 6.
    clean_midi : bool, optional
        If True, only lead and base instruments will be processed, the rest will be discarted.
        Default is True.
    verbose : bool, optional
        When True, prints concise progress messages and enables tqdm progress bars.
        Progress bars use `tqdm(disable=not verbose)` so they are suppressed when verbose is False.

    Returns
    -------
    list[list[int]]
        A list of token sequences. Each token sequence is a list of integers where:
        - The first element is 0 (start token).
        - Delta times (when nonzero) are appended as integers in 0..127.
        - Note events are encoded as two integers: duration_token and pitch_token,
          where duration_token = duration_clipped + 128 and pitch_token = pitch_clipped + 256.
        The function returns an empty list if processing fails or no notes are found.

    Notes and assumptions
    ---------------------
    - The function expects TMIDIX to provide the following functions used internally:
      midi2single_track_ms_score, advanced_score_processor, solo_piano_escore_notes,
      recalculate_score_timings, augment_enhanced_score_notes, remove_duplicate_pitches_from_escore_notes,
      fix_escore_notes_durations, delta_score_notes.
    - Delta events `d` are assumed to be indexable sequences where:
      d[1] is delta time, d[2] is duration, and d[4] is pitch (consistent with the original code).
    - Timing values are clipped to 0..127; durations are clipped to 1..127; pitches are clipped to 1..127
      after applying the transpose offset.
    - The function intentionally uses small integer ranges to match downstream token vocabularies
      that reserve offsets (e.g., +128, +256) for event encoding.

    Exceptions
    ----------
    - Any exception raised during processing is caught; the function prints a short error message
      (only when verbose) and returns the token sequences collected so far (often an empty list).

    Example
    -------
    >>> sequences = midi_to_tokens("example.mid", transpose_factor=2, verbose=True)
    >>> len(sequences)
    4  # variants for tv in [-2, -1, 0, 1] when transpose_factor == 2

    """
    
    if not verbose:
        TMIDIX.set_no_warning(True)
    
    all_toks_sequences = []

    try:
        if verbose:
            print('=' * 70)
            print(f"Loading MIDI file: {midi_file_path}")
            print('=' * 70)

        raw_score = TMIDIX.midi2single_track_ms_score(
            midi_file_path, do_not_check_MIDI_signature=True
        )

        if verbose:
            print("Running advanced score processor (enhanced notes, sustain applied)...")

        escore = TMIDIX.advanced_score_processor(
            raw_score, return_enhanced_score_notes=True, apply_sustain=True
        )

        if not escore or not escore[0]:
            if verbose:
                print("No enhanced score notes found after advanced processing. Returning empty list.")
                
            return all_toks_sequences
        
        if verbose:
            print("Augmenting enhanced-score notes...")
            
        escore_notes = TMIDIX.augment_enhanced_score_notes(escore[0], timings_divider=32)
        
        if clean_midi:
            if verbose:
                print('Cleaning MIDI...')
                
            escore_notes = [e for e in escore_notes if e[6] in TMIDIX.CLEAN_INSTRUMENTS]

        if not escore_notes:
            if verbose:
                print("No enhanced score notes found after augmentation and cleaning. Returning empty list.")
                
            return all_toks_sequences

        if verbose:
            print("Extracting solo piano enhanced-score notes...")

        escore_notes = TMIDIX.solo_piano_escore_notes(escore_notes)

        if not escore_notes:
            if verbose:
                print("Solo piano extraction returned no notes. Returning empty list.")
                
            return all_toks_sequences

        if verbose:
            print("Recalculating timings, augmenting timings, removing duplicates, and fixing durations...")

        escore_notes = TMIDIX.remove_duplicate_pitches_from_escore_notes(escore_notes)

        escore_notes = TMIDIX.fix_escore_notes_durations(escore_notes, min_notes_gap=0)
        
        escore_notes = TMIDIX.recalculate_score_timings(escore_notes)
        
        # Clamp transpose_factor to allowed range
        transpose_factor = max(0, min(6, transpose_factor))
            
        if verbose:
            print(f"Using transpose_factor={transpose_factor} (clamped to 0..6).")

        if transpose_factor > 0:
            sidx = -transpose_factor
            eidx = transpose_factor
        else:
            sidx = 0
            eidx = 1

        if verbose:
            print(f"Generating token sequences for transpose variants in range({sidx}, {eidx})...")

        # Outer loop: transpose variants with progress bar
        for tv in tqdm.tqdm(range(sidx, eidx), disable=not verbose, desc="Transpose variants"):
            if verbose:
                print(f"Processing transpose variant tv={tv}...")
                
            tscore = TMIDIX.transpose_escore_notes(escore_notes, tv)
                
            cscore = TMIDIX.chordify_score([1000, tscore])

            fixed_score = []

            for c in cscore:
                c.sort(key=lambda x: -x[4])

                tones_chord = sorted(set([p[4] % 12 for p in c]))

                if tones_chord not in TMIDIX.ALL_CHORDS_SORTED:
                    tones_chord = TMIDIX.check_and_fix_tones_chord(tones_chord, use_full_chords=False)

                for e in c:
                    if e[4] % 12 in tones_chord:
                        fixed_score.append(e)

            cscore = TMIDIX.chordify_score([1000, fixed_score])

            score = []

            pc = cscore[0]

            for c in cscore:
                c.sort(key=lambda x: -x[4])

                tones_chord = sorted(set([p[4] % 12 for p in c]))

                if len(c) > 1:
                    chord_tok = TMIDIX.ALL_CHORDS_SORTED.index(tones_chord)+12

                else:
                    chord_tok = tones_chord[0]

                dtime = max(0, min(127, c[0][1]-pc[0][1]))
                score.append(dtime)

                score.append(chord_tok+384)

                for e in c:
                    score.extend([max(1, min(127, e[4]))+128, max(1, min(127, e[2]))+256])

                pc = c

            if verbose:
                print(f"Variant tv={tv} produced sequence length {len(score[:max_seq_len])}.")
                
            all_toks_sequences.append(score[:max_seq_len])

        if verbose:
            print('=' * 70)
            print(f"Finished processing. Produced {len(all_toks_sequences)} token sequence(s).")
            print('=' * 70)

        return all_toks_sequences

    except Exception as ex:
        print("Exception while converting MIDI to token sequences!")
        print(f"File: {midi_file_path}")
        print(f"Error: {ex}")
            
        return all_toks_sequences

###################################################################################

def _normalize_token_type_weights(
    token_type_weights: Optional[Union[Dict[Tuple[int, int], float], Tuple[float, float, float]]],
) -> Dict[Tuple[int, int], float]:
    """
    Normalise ``token_type_weights`` into a dict mapping ``(start, end)``
    token-id ranges (inclusive start, exclusive end) to scalar weights.

    Accepts:
      - A **dict** ``{(start, end): weight, ...}`` — copied and cast to float.
      - A **tuple/list** ``(onset_w, duration_w, pitch_w)`` for backward
        compatibility, mapped to ``[0,128)``, ``[128,256)``, ``[256,384)``.
      - ``None`` → empty dict (uniform weights).

    Returns:
        Dict mapping ``(start, end)`` tuples to float weights.
    """
    if token_type_weights is None:
        return {}
    if isinstance(token_type_weights, dict):
        return {tuple(k): float(v) for k, v in token_type_weights.items()}
    if isinstance(token_type_weights, (tuple, list)):
        onset_w, duration_w, pitch_w = token_type_weights
        return {
            (0, 128): float(onset_w),
            (128, 256): float(duration_w),
            (256, 384): float(pitch_w),
        }
    raise TypeError(
        f"Unsupported token_type_weights type: {type(token_type_weights)}")

###################################################################################

def _get_autocast_ctx(device_type: str, use_bfloat16: bool):
    """
    Return a ``torch.amp.autocast`` context manager or ``None``.

    When ``use_bfloat16`` is ``True``, tries ``bfloat16`` autocast first and
    falls back to the device's default autocast.  When ``False``, tries only
    the default autocast.  Returns ``None`` if autocast is unavailable.
    """
    if use_bfloat16:
        try:
            return torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16)
        except Exception:
            pass
    try:
        return torch.amp.autocast(device_type=device_type)
    except Exception:
        return None

###################################################################################

def _pool_embeddings(
    out: Tensor,
    mask: Tensor,
    token_ids: Tensor,
    pooling: str,
    token_type_weights,
    concat_aggregated_embeddings: bool,
    verbose: bool,
) -> Tensor:
    """
    Pool model output based on tensor dimensionality and ``pooling`` strategy.

    - 2-D ``(B, D)``: returned as-is (already pooled by the model).
    - 3-D ``(B, L, D)``: pooled via the specified strategy:
        - ``"auto"`` / ``"mean"``: Simple masked mean pooling.
        - ``"weighted_mean"``: Weighted mean pooling across all tokens.
        - ``"weighted_mean_aggregated"``: Compute separate weighted mean pools
          per token range, then concatenate or stack them.

    Raises:
        ValueError: For unsupported ``pooling`` values or unexpected tensor
            dimensionality.
    """
    if out.dim() == 2:
        return out
    if out.dim() == 3:
        if pooling in ("mean", "auto"):
            return masked_mean_pool(out, mask, dim=1, verbose=verbose)
        if pooling == "weighted_mean":
            return masked_weighted_mean_pool(
                out, mask, token_ids=token_ids,
                token_type_weights=token_type_weights, dim=1, verbose=verbose)
        if pooling == "weighted_mean_aggregated":
            return masked_weighted_mean_aggregated_pool(
                out, mask, token_ids=token_ids,
                token_type_weights=token_type_weights, dim=1,
                concat=concat_aggregated_embeddings, verbose=verbose)
        raise ValueError(f"unsupported pooling: {pooling}")
    raise ValueError(f"unexpected embedding tensor shape: {out.shape}")


###################################################################################

def masked_mean_pool(
    token_embeddings: Tensor,
    mask: Tensor,
    dim: int = 1,
    eps: float = 1e-9,
    verbose: bool = True,
) -> Tensor:
    """
    Compute a masked mean pooling over a specified dimension.

    Positions where ``mask`` is ``False`` are ignored.  A small ``eps`` prevents
    division by zero for sequences that are entirely masked out.

    Args:
        token_embeddings: Tensor of shape ``(B, L, D)`` or similar where ``dim``
            indexes the sequence length.  Dtype may be float16/float32/bfloat16.
        mask: Boolean tensor broadcastable to the sequence dimension, e.g.
            ``(B, L)``.  ``True`` = valid token, ``False`` = padding.
        dim: Dimension along which to pool (default: ``1``).
        eps: Small value to avoid division by zero (default: ``1e-9``).
        verbose: If ``True``, prints a short summary via ``tqdm.write``.

    Returns:
        Pooled tensor with ``dim`` removed, typically ``(B, D)``.  Dtype matches
        ``token_embeddings.dtype``.
    """
    mask_f = mask.to(token_embeddings.dtype)
    summed = (token_embeddings * mask_f.unsqueeze(-1)).sum(dim=dim)
    counts = mask_f.sum(dim=dim).clamp_min(eps).unsqueeze(-1)
    pooled = summed / counts

    if verbose:
        valid_counts = counts.squeeze(-1)
        tqdm.tqdm.write(
            f"[masked_mean_pool] pooled shape={pooled.shape}, "
            f"counts min={valid_counts.min().item():.3f}, "
            f"max={valid_counts.max().item():.3f}")

    return pooled


###################################################################################

def masked_weighted_mean_pool(
    token_embs: Tensor,
    valid_mask: Tensor,
    token_ids: Optional[Tensor] = None,
    token_type_weights: Optional[Union[Dict[Tuple[int, int], float], Tuple[float, float, float]]] = None,
    dim: int = 1,
    verbose: bool = False,
) -> Tensor:
    """
    Weighted mean pooling across tokens.

    Per-token weights are determined by ``token_type_weights``, which may be:

    - A **dict** mapping ``(start, end)`` token-id ranges (inclusive start,
      exclusive end) to scalar weights, e.g.
      ``{(0, 128): 1.0, (128, 256): 2.0, (256, 384): 1.5}``.
    - A **tuple** ``(onset_w, duration_w, pitch_w)`` for backward compatibility,
      mapped to ``[0,128)``, ``[128,256)``, ``[256,384)``.

    Tokens outside any specified range receive a weight of ``1.0``.  Padding
    positions (``valid_mask`` is ``False``) always receive weight ``0``.

    If ``token_ids`` is ``None`` or ``token_type_weights`` is empty/``None``,
    the function falls back to :func:`masked_mean_pool`.

    Args:
        token_embs: Per-token embeddings of shape ``(B, L, D)``.
        valid_mask: BoolTensor of shape ``(B, L)`` — ``True`` for valid tokens.
        token_ids: Optional LongTensor ``(B, L)`` with token ids used to
            determine per-token weights.  If ``None``, falls back to
            :func:`masked_mean_pool`.
        token_type_weights: Dict ``{(start, end): weight}`` mapping token-id
            ranges to float weights, **or** a backward-compatible tuple
            ``(onset_w, duration_w, pitch_w)``.  ``None`` ⇒ uniform weights.
        dim: Dimension along which to pool (default: ``1``).
        verbose: If ``True``, prints diagnostic info via ``tqdm.write``.

    Returns:
        Pooled embeddings of shape ``(B, D)``; dtype matches ``token_embs``.
    """
    if token_ids is None:
        if verbose:
            tqdm.tqdm.write(
                "[masked_weighted_mean_pool] token_ids is None; "
                "falling back to masked_mean_pool")
        return masked_mean_pool(token_embs, valid_mask, dim=dim, verbose=verbose)

    B, L, _ = token_embs.shape
    device, dtype = token_embs.device, token_embs.dtype

    weights = _normalize_token_type_weights(token_type_weights)

    w = torch.ones((B, L), device=device, dtype=dtype)
    for (start, end), weight in weights.items():
        if weight == 1.0:
            continue
        in_range = (token_ids >= start) & (token_ids < end)
        w = torch.where(in_range, torch.tensor(weight, device=device, dtype=dtype), w)

    w = w * valid_mask.to(dtype)

    denom = w.sum(dim=dim, keepdim=True).clamp_min(1e-6)
    pooled = (token_embs * w.unsqueeze(-1)).sum(dim=dim) / denom
    return pooled


###################################################################################

def masked_weighted_mean_aggregated_pool(
    token_embs: Tensor,
    valid_mask: Tensor,
    token_ids: Optional[Tensor] = None,
    token_type_weights: Optional[Union[Dict[Tuple[int, int], float], Tuple[float, float, float]]] = None,
    dim: int = 1,
    concat: bool = True,
    verbose: bool = False,
) -> Tensor:
    """
    Aggregated weighted mean pooling across specific token ranges.

    Computes a separate mean-pooled embedding for each token-id range specified
    in ``token_type_weights``, scales that embedding by the corresponding weight,
    and then either concatenates or stacks the results.

    - If ``concat=True``: Returns a 2-D tensor of shape ``(B, D * R)`` where
      ``R`` is the number of specified ranges.
    - If ``concat=False``: Returns a 3-D tensor of shape ``(B, R, D)``.

    Tokens not present in a sequence's range result in a zero-vector for that
    range's embedding. Padding positions (``valid_mask`` is ``False``) are
    ignored.

    If ``token_ids`` is ``None`` or ``token_type_weights`` is empty/``None``,
    the function falls back to :func:`masked_mean_pool`.

    Args:
        token_embs: Per-token embeddings of shape ``(B, L, D)``.
        valid_mask: BoolTensor of shape ``(B, L)`` — ``True`` for valid tokens.
        token_ids: Optional LongTensor ``(B, L)`` with token ids. If ``None``,
            falls back to :func:`masked_mean_pool`.
        token_type_weights: Dict ``{(start, end): weight}`` mapping token-id
            ranges to float weights, **or** a backward-compatible tuple. Ranges
            are processed in sorted order of their start token.
        dim: Dimension along which to pool (default: ``1``).
        concat: If ``True``, concatenate pooled range embeddings along the
            feature dimension. If ``False``, stack them along a new dimension.
        verbose: If ``True``, prints diagnostic info via ``tqdm.write``.

    Returns:
        Aggregated pooled embeddings; shape depends on ``concat``.
    """
    if token_ids is None:
        if verbose:
            tqdm.tqdm.write(
                "[masked_weighted_mean_aggregated_pool] token_ids is None; "
                "falling back to masked_mean_pool")
        return masked_mean_pool(token_embs, valid_mask, dim=dim, verbose=verbose)

    weights = _normalize_token_type_weights(token_type_weights)
    if not weights:
        if verbose:
            tqdm.tqdm.write(
                "[masked_weighted_mean_aggregated_pool] token_type_weights empty; "
                "falling back to masked_mean_pool")
        return masked_mean_pool(token_embs, valid_mask, dim=dim, verbose=verbose)

    # Sort ranges by start token to ensure deterministic concatenation order
    sorted_ranges = sorted(weights.keys(), key=lambda x: x[0])
    device, dtype = token_embs.device, token_embs.dtype

    pooled_list = []
    for start, end in sorted_ranges:
        weight = weights[(start, end)]
        range_mask = (token_ids >= start) & (token_ids < end) & valid_mask
        range_mask_f = range_mask.to(dtype)

        # Sum embeddings in this range, apply the range-specific weight
        summed = (token_embs * weight * range_mask_f.unsqueeze(-1)).sum(dim=dim)
        counts = range_mask_f.sum(dim=dim).clamp_min(1e-6).unsqueeze(-1)
        
        # If count is 0 (no tokens in range), summed is 0, and 0 / 1e-6 = 0.
        # We cast back to float to avoid division anomalies, then back to dtype.
        pooled = summed / counts
        pooled_list.append(pooled.to(dtype))

    if concat:
        # (B, D * R)
        return torch.cat(pooled_list, dim=-1)
    else:
        # (B, R, D)
        return torch.stack(pooled_list, dim=1)


###################################################################################

def pad_and_mask(
    sequences: List[List[int]],
    pad_idx: int = 719,
    seq_len: Optional[int] = None,
    device: Optional[torch.device] = None,
    verbose: bool = False,
) -> Tuple[Tensor, Tensor]:
    """
    Pad a batch of variable-length token sequences and create a boolean mask.

    Sequences longer than ``seq_len`` are truncated.  If ``seq_len`` is ``None``
    or larger than the batch maximum, the batch maximum is used to avoid
    unnecessary padding.

    Args:
        sequences: List of token-id sequences (each a list of ints).
        pad_idx: Token id used for padding (default: ``719``).
        seq_len: Optional cap on sequence length.  If ``None``, the batch
            maximum is used.
        device: Device for the returned tensors.  ``None`` uses the default.
        verbose: If ``True``, shows a ``tqdm`` progress bar and prints a summary.

    Returns:
        ``(x, mask)`` where ``x`` is a ``LongTensor (B, T)`` of padded token ids
        and ``mask`` is a ``BoolTensor (B, T)`` with ``True`` at real-token
        positions.
    """
    if not sequences:
        empty = torch.empty((0, 0), dtype=torch.long, device=device)
        return empty, torch.empty((0, 0), dtype=torch.bool, device=device)

    lengths = [len(s) for s in sequences]
    batch_max = max(lengths)
    target_len = min(seq_len, batch_max) if seq_len is not None else batch_max
    b = len(sequences)

    x = torch.full((b, target_len), pad_idx, dtype=torch.long, device=device)
    mask = torch.zeros((b, target_len), dtype=torch.bool, device=device)

    iterator = (tqdm.tqdm(sequences, desc="Pad & mask", disable=not verbose)
                if verbose else sequences)
    for i, seq in enumerate(iterator):
        if seq:
            L = min(len(seq), target_len)
            x[i, :L] = torch.tensor(seq[:L], dtype=torch.long, device=device)
            mask[i, :L] = True

    if verbose:
        tqdm.tqdm.write(
            f"[pad_and_mask] batch_size={b}, target_len={target_len}, "
            f"min_len={min(lengths)}, max_len={max(lengths)}")

    return x, mask


###################################################################################

def get_embeddings_bf16(
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
    token_type_weights: Optional[Union[Dict[Tuple[int, int], float], Tuple[float, float, float]]] = None,
    concat_aggregated_embeddings: bool = True,
    use_bfloat16: bool = True,
    return_dtype: str = "float32",
    return_numpy: bool = False,
    verbose: bool = True,
    show_progress_bar: bool = True,
) -> Union[Tensor, np.ndarray]:
    """
    Compute embeddings for a list of token sequences.

    Batches input token-id sequences, pads/truncates them, runs the model under
    ``torch.inference_mode()`` with optional ``bfloat16`` autocast, pools
    per-token outputs, optionally L2-normalises, and returns a single tensor or
    NumPy array.

    The model ``forward`` must accept ``x`` (LongTensor), ``mask`` (BoolTensor)
    and ``return_embeddings=True``, returning either:

    - a 2-D tensor ``(B, D)`` of already-pooled embeddings, or
    - a 3-D tensor ``(B, L, D)`` of per-token embeddings (pooled according to
      ``pooling``).

    Pooling modes:
      - ``"auto"`` / ``"mean"``: Masked mean pooling.
      - ``"weighted_mean"``: Weighted mean pooling where per-token weights are
        determined by ``token_type_weights``.
      - ``"weighted_mean_aggregated"``: Computes a separate weighted mean pool
        for each token range specified in ``token_type_weights``. If
        ``concat_aggregated_embeddings`` is True, concatenates them into a 2D
        tensor ``(B, D * num_ranges)``; otherwise stacks them into a 3D tensor
        ``(B, num_ranges, D)``.

    Args:
        model: PyTorch model; moved to ``device`` and set to ``eval()``.
        sequences: List of token-id sequences (each a list of ints).  An empty
            list returns an empty ``(0, 0)`` result.
        seq_len: Target sequence length for truncation/padding (default:
            ``3072``).  If ``None``, the per-batch maximum is used.
        seq_pad_idx: Token id used for padding (default: ``719``).
        batch_size: Number of sequences per forward pass (default: ``64``).
        save_every_num_batches: If ``> 0``, save accumulated embeddings to
            ``save_file_path`` every this many batches (default: ``-1``,
            disabled).
        save_file_path: Path for periodic ``np.save`` (default:
            ``"saved_embeddings.npy"``).
        device: Device for model and tensors.  ``None`` uses the model's
            parameter device.
        normalize: If ``True``, L2-normalise each embedding in float32
            (default: ``False``).
        pooling: ``"auto"``, ``"mean"``, ``"weighted_mean"``, or
            ``"weighted_mean_aggregated"`` (default: ``"auto"``).
        token_type_weights: Dict ``{(start, end): weight}`` mapping token-id
            ranges (inclusive start, exclusive end) to float weights, **or** a
            backward-compatible tuple ``(onset_w, duration_w, pitch_w)`` mapped
            to ``[0,128)``, ``[128,256)``, ``[256,384)``.  ``None`` ⇒ uniform
            weights (equivalent to plain mean pooling).
        concat_aggregated_embeddings: If ``True`` and pooling is
            ``"weighted_mean_aggregated"``, embeddings for each token range are
            concatenated along the feature dimension (default: ``True``). If
            ``False``, they are stacked along a new dimension, returning a 3D
            tensor.
        use_bfloat16: If ``True``, attempt ``bfloat16`` autocast, falling back
            gracefully if unsupported (default: ``True``).
        return_dtype: ``"float32"`` or ``"float16"`` for returned embeddings
            (default: ``"float32"``).
        return_numpy: If ``True``, return a NumPy array instead of a tensor
            (default: ``False``).
        verbose: If ``True``, print progress and diagnostics via ``tqdm``
            (default: ``True``).
        show_progress_bar: If ``True``, display a ``tqdm`` progress bar
            (default: ``True``).

    Returns:
        CPU ``torch.Tensor`` or NumPy array. Shape depends on pooling:
        - ``(N, D)`` for standard pooling.
        - ``(N, D * R)`` for aggregated pooling with ``concat=True``.
        - ``(N, R, D)`` for aggregated pooling with ``concat=False``.

    Raises:
        AssertionError: If ``return_dtype`` is not ``"float32"`` or
            ``"float16"``.
        RuntimeError: If the model returns ``None`` for embeddings.
        ValueError: If the output tensor has unexpected dimensionality or
            ``pooling`` is unsupported.

    Example:
        >>> embs = get_embeddings_bf16(
        ...     model, sequences, seq_len=1024, batch_size=32,
        ...     pooling="weighted_mean_aggregated",
        ...     token_type_weights={(0, 128): 2.0, (128, 256): 3.0},
        ...     concat_aggregated_embeddings=True,
        ...     normalize=True, return_dtype="float32")
    """
    assert return_dtype in ("float32", "float16"), \
        "return_dtype must be 'float32' or 'float16'"

    model_device = next(model.parameters()).device if device is None else device
    model.to(model_device)
    model.eval()

    autocast_ctx = _get_autocast_ctx(model_device.type, use_bfloat16)
    total_batches = math.ceil(len(sequences) / batch_size) if batch_size > 0 else 0

    if verbose:
        tqdm.tqdm.write(
            f"[get_embeddings_bf16] sequences={len(sequences)}, "
            f"batch_size={batch_size}, batches={total_batches}, "
            f"device={model_device}, seq_len={seq_len}, pooling={pooling}")

    all_embs: List[Tensor] = []

    with torch.inference_mode():
        pbar = tqdm.tqdm(
            range(0, len(sequences), batch_size),
            total=total_batches, desc="Embedding batches",
            disable=not show_progress_bar)

        for batch_idx, i in enumerate(pbar):
            # --- Pad & mask ---
            x, mask = pad_and_mask(
                sequences[i:i + batch_size],
                pad_idx=seq_pad_idx, seq_len=seq_len,
                device=model_device, verbose=verbose)

            # --- Forward pass (under autocast if available) ---
            if autocast_ctx is not None:
                with autocast_ctx:
                    out = model(x, return_embeddings=True, mask=mask)
            else:
                out = model(x, return_embeddings=True, mask=mask)

            if out is None:
                raise RuntimeError(
                    "model returned None for embeddings. Check forward flags.")

            # --- Pool ---
            emb = _pool_embeddings(
                out, mask, x, pooling, token_type_weights,
                concat_aggregated_embeddings, verbose)

            # --- Dtype, normalize, cast ---
            # Convert to float32 for stable normalization (works for 2D and 3D)
            emb = emb.float()
            if normalize:
                emb = F.normalize(emb, p=2, dim=-1)
            if return_dtype == "float16":
                emb = emb.half()

            all_embs.append(emb.cpu())

            if verbose:
                pbar.set_postfix(
                    batch=batch_idx + 1,
                    shape=str(emb.shape),
                    dtype=str(emb.dtype))

            # --- Periodic save ---
            if (save_every_num_batches > 0
                    and (batch_idx + 1) % save_every_num_batches == 0):
                try:
                    arr = torch.cat(all_embs, dim=0).numpy()
                    np.save(save_file_path, arr)
                    if verbose:
                        tqdm.tqdm.write(
                            f"[get_embeddings_bf16] saved {arr.shape[0]} "
                            f"embeddings to {save_file_path}")
                except Exception as e:
                    if verbose:
                        tqdm.tqdm.write(
                            f"[get_embeddings_bf16] warning: failed to save "
                            f"embeddings: {e}")

    # --- Assemble result ---
    if not all_embs:
        dtype = torch.float16 if return_dtype == "float16" else torch.float32
        empty = torch.empty((0, 0), dtype=dtype)
        if verbose:
            tqdm.tqdm.write(
                "[get_embeddings_bf16] no embeddings produced; "
                "returning empty tensor")
        return empty.numpy() if return_numpy else empty

    # torch.cat works natively for both 2D (N, D) and 3D (N, R, D) tensors
    result = torch.cat(all_embs, dim=0)

    if verbose:
        tqdm.tqdm.write(
            f"[get_embeddings_bf16] finished: "
            f"total_embeddings={result.shape[0]}, "
            f"shape={result.shape}, dtype={result.dtype}")

    return result.numpy() if return_numpy else result

###################################################################################

def random_ngram_replace(
    seq: List[int],
    prob_single: float = 0.10,
    prob_ngram: float = 0.10,
    max_ngram: int = 5,
    replace_value: int = 718,
    rng: Optional[np.random.Generator] = None,
) -> List[int]:
    """
    Randomly replaces values in a list with a specified value.
    Also randomly replaces consecutive n-grams (up to max_ngram length).
    Original list remains intact.

    Args:
        seq: List[int] — original sequence.
        prob_single: Probability of replacing a single element.
        prob_ngram: Probability of replacing an n-gram starting at any index.
        max_ngram: Maximum n-gram length to replace.
        replace_value: Value to insert.
        rng: Optional NumPy random generator.

    Returns:
        A new list with random replacements applied.
    """

    if rng is None:
        rng = np.random.default_rng()

    arr = np.array(seq, dtype=np.int64)
    out = arr.copy()
    n = len(arr)

    # 1. Single-value replacements
    single_mask = rng.random(n) < prob_single
    out[single_mask] = replace_value

    # 2. N-gram replacements
    # For each index, decide whether to start an n-gram replacement
    start_mask = rng.random(n) < prob_ngram
    starts = np.where(start_mask)[0]

    for s in starts:
        # Random n-gram length between 2 and max_ngram
        L = rng.integers(2, max_ngram + 1)
        end = min(s + L, n)
        out[s:end] = replace_value

    return out.tolist()

###################################################################################

TensorOrArray = Union[torch.Tensor, np.ndarray]

###################################################################################

def cosine_similarity_topk(
    query_embs: TensorOrArray,
    corpus_embs: TensorOrArray,
    topk: int = 16,
    chunk_size: int = 10000,
    device: torch.device | None = None,
    use_gpu_if_available: bool = True,
    normalize_inputs: bool = True,
    return_dtype: torch.dtype = torch.float32,
    use_fp32_accumulation: bool = True,
    verbose: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
    
    """
    Compute top-k cosine similarities between query embeddings and a (potentially large)
    corpus of embeddings, returning the top-k similarity values and corresponding corpus
    indices for each query.

    This function accepts either `torch.Tensor` or `numpy.ndarray` for `query_embs`
    and `corpus_embs`. When numpy arrays are provided, slices of the corpus are
    converted to torch tensors on-the-fly to avoid copying the entire corpus to GPU.

    Parameters
    ----------
    query_embs : torch.Tensor | numpy.ndarray
        2-D array of shape (Q, D) containing Q query embeddings of dimension D.
    corpus_embs : torch.Tensor | numpy.ndarray
        2-D array of shape (N, D) containing N corpus embeddings of dimension D.
    topk : int, default 16
        Number of top matches to return per query.
    chunk_size : int, default 10000
        Number of corpus rows to process per chunk. Lower this to reduce peak memory.
    device : torch.device | None, default None
        Device to use. If None, will use CUDA if available and `use_gpu_if_available` is True,
        otherwise CPU.
    use_gpu_if_available : bool, default True
        If True and CUDA is available, prefer GPU when `device` is not explicitly set.
    normalize_inputs : bool, default True
        If True, L2-normalize both query and corpus embeddings before computing cosine sims.
    return_dtype : torch.dtype, default torch.float32
        Output dtype for similarity values. If `torch.float16` is requested, values are cast
        before returning.
    use_fp32_accumulation : bool, default True
        If True, perform matrix multiplications in float32 for numerical stability.
    verbose : bool, default True
        If True, print brief status messages and show a tqdm progress bar for corpus chunks.

    Returns
    -------
    best_idx : nd.array
        Numpy array of shape (Q, topk) with the global corpus indices corresponding to
        `best_vals`.
    best_vals : nd.array
        Numpy array of shape (Q, topk) with the top-k similarity values for each query.

    Notes
    -----
    - The function keeps only the top-k matches across all chunks by merging per-chunk
      top-k results into a running buffer.
    - Returned tensors are moved to CPU.
    - If `corpus_embs` is a numpy array, only the current chunk is converted to a tensor,
      minimizing memory usage on the target device.
    """

    # --- basic shape checks that work for both numpy and torch ---
    if isinstance(query_embs, np.ndarray):
        if query_embs.ndim != 2:
            raise ValueError("query_embs must be 2-D")
        Q, D = query_embs.shape
    else:
        if query_embs.dim() != 2:
            raise ValueError("query_embs must be 2-D")
        Q, D = query_embs.shape

    if isinstance(corpus_embs, np.ndarray):
        if corpus_embs.ndim != 2:
            raise ValueError("corpus_embs must be 2-D")
        N, D2 = corpus_embs.shape
    else:
        if corpus_embs.dim() != 2:
            raise ValueError("corpus_embs must be 2-D")
        N, D2 = corpus_embs.shape

    if D != D2:
        raise ValueError("query and corpus must have the same embedding dimension")

    # pick device
    if device is None:
        if use_gpu_if_available and torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

    if verbose:
        print(f"[cosine_similarity_topk] device: {device}; queries: {Q} x {D}; corpus: {N} x {D2}; topk: {topk}; chunk_size: {chunk_size}")

    # Convert queries to torch and move to device
    if isinstance(query_embs, np.ndarray):
        query = torch.from_numpy(query_embs)
    else:
        query = query_embs

    query = query.to(device)

    # normalize queries
    if normalize_inputs:
        query = torch.nn.functional.normalize(query.float(), p=2, dim=-1).to(query.dtype)

    # initialize topk buffers
    best_vals = torch.full((Q, topk), -float("inf"), dtype=torch.float32, device=device)
    best_idx  = torch.full((Q, topk), -1, dtype=torch.long, device=device)

    # iterate corpus in chunks
    corpus_is_numpy = isinstance(corpus_embs, np.ndarray)
    iterator = range(0, N, chunk_size)
    pbar = tqdm.tqdm(iterator, disable=not verbose, desc="Processing corpus", unit="chunk")

    for start in pbar:
        end = min(start + chunk_size, N)
        C = end - start

        if corpus_is_numpy:
            # convert slice to tensor (shares memory with numpy if possible)
            chunk = torch.from_numpy(corpus_embs[start:end])
            chunk = chunk.to(device)
        else:
            # corpus is a torch tensor; move slice to device
            chunk = corpus_embs[start:end].to(device)  # (C, D)

        # normalize chunk
        if normalize_inputs:
            chunk = torch.nn.functional.normalize(chunk.float(), p=2, dim=-1).to(chunk.dtype)

        # choose accumulation dtype
        if use_fp32_accumulation:
            q_mat = query.float()
            c_mat = chunk.float()
        else:
            q_mat = query
            c_mat = chunk

        # compute similarities: (Q, C)
        sims_block = q_mat @ c_mat.t()

        k_block = min(topk, C)

        # topk inside this block
        vals_block, idxs_block = torch.topk(sims_block, k=k_block, dim=1)

        # convert local indices → global corpus indices
        idxs_block = idxs_block + start  # (Q, k_block)

        # ---- MERGE LOGIC ----
        # concat along dim=1 → always (Q, topk + k_block)
        merged_vals = torch.cat([best_vals, vals_block.float()], dim=1)
        merged_idxs = torch.cat([best_idx, idxs_block.long()], dim=1)

        # select new topk
        new_vals, new_pos = torch.topk(merged_vals, k=topk, dim=1)

        # gather indices using 2‑D indexing
        row_ids = torch.arange(Q, device=device).unsqueeze(1)
        new_idxs = merged_idxs[row_ids, new_pos]

        best_vals = new_vals
        best_idx  = new_idxs

        # update progress bar postfix with a brief summary
        if verbose:
            pbar.set_postfix({"processed": f"{end}/{N}"})

        # cleanup
        del chunk, sims_block, vals_block, idxs_block, merged_vals, merged_idxs
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # cast output dtype
    if return_dtype == torch.float16:
        if verbose:
            print("[cosine_similarity_topk] Casting to float16")
        best_vals = best_vals.half()

    if verbose:
        print("[cosine_similarity_topk] done; moving results to CPU and converting to NumPy arrays")

    return best_idx.cpu().numpy(), best_vals.cpu().numpy()

###################################################################################

def idxs_sims_to_sorted_list(idxs: np.ndarray,
                             sims: np.ndarray,
                             sims_mult: int = 100,
                             remove_dupes=True,
                             ) -> List[Tuple]:
    
    """
    Helper function to convert indexes and similarities arrays into
    a sorted list with corresponding transpose values.
    
    Rwturns
    -------
    List of tuples: (corpus_index, transpose_value, cosine_similarity)
    """
    
    idxs = np.array(idxs)
    sims = np.array(sims)

    assert idxs.shape == sims.shape, f'Shape mismatch between indexes array and similarities array: {idxs.shape} != {sims.shape}'

    flat_idxs = [x for row in idxs.tolist() for x in row]
    flat_sims = [x * sims_mult for row in sims.tolist() for x in row]

    tv = idxs.shape[0]

    if tv == 1:
        sr = 0
        er = 1

    elif tv > 1 and tv % 2 == 0:
        sr = -(tv // 2)
        er = tv // 2

    else:
        sr = -6
        er = 6
    
    tkv = idxs.shape[1]
    
    flat_tvs = [v for v in range(sr, er) for _ in range(tkv)]

    sorted_list = sorted(zip(flat_idxs, flat_tvs, flat_sims), key=lambda x: -x[2])
    
    if remove_dupes:
        deduped_sorted_list = []
        seen = set()
        
        for idx, tv, sim in sorted_list:
            if idx not in seen:
                deduped_sorted_list.append([idx, tv, sim])
                seen.add(idx)
            
        return deduped_sorted_list
    
    return sorted_list   

###################################################################################

def print_sorted_idxs_sims_list(sorted_idxs_sims_list: list,
                                corpus_midi_names: Union[list, np.ndarray],
                                return_as_list: bool = False,
                                ) -> Union[List[Tuple], None]:
    
    """
    Helper function that prints search results list generated by idxs_sims_to_sorted_list function
    
    Returns
    -------
    List of tuples if return_as_list is True
    None if return_as_list is False
    """
    
    if type(corpus_midi_names) == np.ndarray:
        corpus_midi_names = corpus_midi_names.tolist()    

    if not return_as_list:
        print('=' * 70)
        print('Search results:')
        print('=' * 70)
    
    return_list = []

    for i, (idx, tv, sim) in enumerate(sorted_idxs_sims_list):

        if not return_as_list:
            print(f'#{str(i).zfill(3)} {corpus_midi_names[idx]} --- {tv} --- {round(sim, 8)}')
        
        else:
            return_list.append([i, corpus_midi_names[idx], tv, sim])    

    if not return_as_list:
        print('=' * 70)
        print('Total number of records:', len(sorted_idxs_sims_list))
        print('=' * 70)
    
    else:
        return return_list

###################################################################################

@lru_cache(maxsize=1)
def get_corpus_midis(corpus_midis_dirs_tuple: Tuple,
                     verbose: bool = True
                     ) -> Dict:
    
    """
    Returns corpus_midis_dic with LRU caching.
    corpus_midis_dirs_tuple must be a tuple for hashing.
    """

    if verbose:
        print("Scanning corpus MIDI directories...")

    # Create list
    corpus_midis_list = TMIDIX.create_files_list(
        list(corpus_midis_dirs_tuple),
        verbose=verbose
    )

    # Create dict: basename → full path

    if verbose:
        print('Converting files list to dict...')
        
    corpus_midis_dic = {
        os.path.splitext(os.path.basename(f))[0]: f
        for f in corpus_midis_list
    }

    if verbose:
        print('Done!')
    
    return corpus_midis_dic

###################################################################################

def copy_corpus_files(sorted_idxs_sims_list: list[list],
                      corpus_midis_dirs: list = ['./Corpus MIDIs Dir/'],
                      main_output_dir: str = './Corpus Matches Dir/',
                      sub_output_dir: str = '',
                      copy_original_midi: bool = True,
                      original_midi_path: str = '',
                      verbose: bool = True
                     ) -> str:

    """
    Helper function that copies matched corpus MIDIs to a specified directory

    Returns
    -------
    Output directory where files were copied as a string
    """

    if verbose:
        print('=' * 70)
        print('Corpus MIDI files copier')
        print('=' * 70)

    if verbose:
        print('Creating corpus MIDIs files list dict...')

    corpus_midis_dic = get_corpus_midis(tuple(corpus_midis_dirs),
                                        verbose=verbose
                                       )
    
    if verbose:
        print('Done!')
        print('=' * 70)
        print('Copying files...')

    out_dir = ''
    original_copied = False

    for i, cfname, tv, sim in sorted_idxs_sims_list:
        
        try:
        
            sim = str(round(sim, 8))
            tv = str(tv)

            inp_fn = corpus_midis_dic[cfname]
            
            if not sub_output_dir and original_midi_path:
                sub_output_dir = os.path.splitext(os.path.basename(original_midi_path))[0]
        
            out_dir = os.path.join(main_output_dir, sub_output_dir)
            os.makedirs(out_dir, exist_ok=True)

            if copy_original_midi and original_midi_path and not original_copied:
                
                src_fn = sub_output_dir + '.mid'
                out_src_fn = os.path.join(out_dir, src_fn)
                
                try:
                    shutil.copy2(original_midi_path, out_src_fn)
                    original_copied = True
                    
                except Exception as ex:
                    if verbose:
                        print(ex)
                        print('Could not copy original MIDI:', os.path.basename(original_midi_path))
            
            out_fn = os.path.join(out_dir, sim + '_' + tv + '_' + cfname + '.mid')
    
            shutil.copy2(inp_fn, out_fn)

        except Exception as ex:
            if verbose:
                print(ex)
                print('Could not copy file #', i, ':', cfname)
                
            continue

    if verbose:
        print('=' * 70)
        print('Done!')
        print('=' * 70)

    return out_dir

###################################################################################

print('Module is loaded!')
print('Enjoy! :)')
print('=' * 70)

###################################################################################
# This is the end of the midisim Python module
###################################################################################