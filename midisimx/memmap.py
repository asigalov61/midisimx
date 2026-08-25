#! /usr/bin/python3

r'''###############################################################################
###################################################################################
#
#	midisimx memmap Python Module
#	Version 1.0
#
#	Project Los Angeles
#
#	Tegridy Code 2026
#
#   https://github.com/Tegridy-Code/Project-Los-Angeles
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
#
#   A high-performance utility module for saving, loading, and merging paired 
#   NumPy arrays (string names + float32 embeddings) into a single custom binary file 
#   that is natively compatible with `numpy.memmap`.
# 
#   This module bypasses the limitations of pickled arrays and `.npz` files by 
#   writing raw, page-aligned C-bytes to disk. This allows the Operating System 
#   to map massive datasets directly into virtual memory with zero RAM overhead.
# 
#   Usage:
#       import memmap
# 
#       # Save
#       memmap.save_paired_memmap('dataset.bin', my_names, my_embeds, verbose=True)
# 
#       # Load
#       names, embeds = memmap.load_paired_memmap('dataset.bin', verbose=True)
# 
#       # Merge
#       memmap.merge_paired_memmaps(['data1.bin', 'data2.bin'], 'merged.bin', verbose=True)
#     
###################################################################################
'''

print('=' * 70)
print('Loading midisimx memmap module...')
print('Please wait...')
print('=' * 70)

__version__ = '1.0.0'

print('midisimx memmap module version', __version__)
print('=' * 70)

###################################################################################

import os
import shutil
import numpy as np
from tqdm import tqdm

# Windows requires memmap offsets to be aligned to the OS allocation granularity.
# 65536 (64KB) is the safest cross-platform header size.
HEADER_SIZE = 65536 

__all__ = ["save_paired_memmap", "load_paired_memmap", "merge_paired_memmaps"]


def _stream_exact_bytes(in_f, out_f, num_bytes, chunk_size=1024*1024):
    """
    Helper to stream exact raw bytes between file objects without RAM bloat.
    
    Args:
        in_f (file object): Input file handle (opened in 'rb' mode).
        out_f (file object): Output file handle (opened in 'wb' mode).
        num_bytes (int): Total number of bytes to stream.
        chunk_size (int): Size of chunks to read/write at a time.
    """
    remaining = num_bytes
    while remaining > 0:
        read_size = min(remaining, chunk_size)
        data = in_f.read(read_size)
        if not data:
            break
        out_f.write(data)
        remaining -= len(data)


def save_paired_memmap(filepath, names, embeds, verbose=False):
    """
    Saves string names and float32 embeds into a SINGLE file 
    that can be memmaped instantly later.
    
    Args:
        filepath (str): Output file path (e.g., 'dataset.bin')
        names (list/np.array): Array of strings (must be convertible to equal length)
        embeds (np.array): 2D array of float32 embeddings
        verbose (bool): If True, prints detailed step-by-step information.
    """
    if verbose: print(f"[1/4] Processing names...")
    names_arr = np.asarray(names, dtype=str)
    max_len = max(len(n) for n in names_arr)
    
    # Cast to fixed-width Unicode
    names_fixed = names_arr.astype(f'<U{max_len}')
    
    if verbose: print(f"[2/4] Processing embeds...")
    embeds_arr = np.ascontiguousarray(embeds, dtype=np.float32)
    
    N, D = embeds_arr.shape
    if len(names_fixed) != N:
        raise ValueError("Length of names and embeds must match!")

    if verbose:
        print(f"  -> Metadata: N={N}, D={D}, max_len={max_len}")
        print(f"  -> Names memory size: {names_fixed.nbytes / (1024**2):.2f} MB")
        print(f"  -> Embeds memory size: {embeds_arr.nbytes / (1024**2):.2f} MB")

    header = np.array([N, max_len, D], dtype=np.uint64)
    
    if verbose: print(f"[3/4] Writing to '{filepath}'...")
    with open(filepath, 'wb') as f:
        f.write(header.tobytes())
        # Pad the rest of the 64KB header with zeros
        f.write(b'\0' * (HEADER_SIZE - header.nbytes))
        
        # Append the raw string bytes and embed bytes
        f.write(names_fixed.tobytes())
        f.write(embeds_arr.tobytes())
        
    if verbose:
        file_size = os.path.getsize(filepath) / (1024**2)
        print(f"[4/4] Save complete. Total file size: {file_size:.2f} MB")


def load_paired_memmap(filepath, verbose=False):
    """
    Loads the single file via memory mapping. Zero parsing overhead.
    
    Args:
        filepath (str): Path to the .bin file.
        verbose (bool): If True, prints detailed memmap offset information.
        
    Returns:
        tuple: (names_memmap, embeds_memmap)
    """
    if verbose: print(f"Reading header from '{filepath}'...")
    with open(filepath, 'rb') as f:
        header = np.frombuffer(f.read(24), dtype=np.uint64)
        N, max_len, D = int(header[0]), int(header[1]), int(header[2])

    if verbose:
        print(f"  -> Header decoded: N={N}, max_len={max_len}, D={D}")

    names_byte_count = N * max_len * 4  # 4 bytes per char in numpy unicode
    
    if verbose: print("Memmaping names array...")
    names_raw = np.memmap(filepath, dtype=np.uint32, mode='r', 
                          offset=HEADER_SIZE, shape=(N, max_len))
    names_memmap = names_raw.view(np.dtype(f'<U{max_len}')).reshape(N)
    
    embeds_offset = HEADER_SIZE + names_byte_count
    if verbose:
        print(f"Memmaping embeds array (byte offset: {embeds_offset})...")
        
    embeds_memmap = np.memmap(filepath, dtype=np.float32, mode='r', 
                              offset=embeds_offset, shape=(N, D))
                              
    if verbose:
        print("Ready. Arrays are mapped to virtual memory (zero RAM copied).")
        
    return names_memmap, embeds_memmap


def merge_paired_memmaps(filepaths, output_filepath, verbose=False):
    """
    Merges multiple custom .bin datasets into a single file 
    compatible with load_paired_memmap().
    
    Args:
        filepaths (list): List of input .bin file paths.
        output_filepath (str): Path for the merged output file.
        verbose (bool): If True, prints metadata and shows progress bars.
    """
    if not filepaths:
        raise ValueError("filepaths list cannot be empty.")

    # 1. Read all headers to gather metadata
    file_infos = []
    for fp in filepaths:
        with open(fp, 'rb') as f:
            header = np.frombuffer(f.read(24), dtype=np.uint64)
            N, max_len, D = int(header[0]), int(header[1]), int(header[2])
            file_infos.append({'path': fp, 'N': N, 'max_len': max_len, 'D': D})
    
    total_N = sum(info['N'] for info in file_infos)
    global_max_len = max(info['max_len'] for info in file_infos)
    global_D = file_infos[0]['D']
    
    if any(info['D'] != global_D for info in file_infos):
        raise ValueError("Embedding dimensions (D) do not match across files!")

    if verbose:
        print(f"Merging {len(filepaths)} files into '{output_filepath}'...")
        print(f"  -> Total samples (N): {total_N}")
        print(f"  -> Global max string length: {global_max_len}")
        print(f"  -> Embedding dim (D): {global_D}")

    # 2. Prepare unified header
    header = np.array([total_N, global_max_len, global_D], dtype=np.uint64)
    
    with open(output_filepath, 'wb') as out_f:
        # Write 64KB aligned header
        out_f.write(header.tobytes())
        out_f.write(b'\0' * (HEADER_SIZE - header.nbytes))
        
        # Phase 1: Merge Names
        if verbose: print("Phase 1: Merging Names...")
        for info in tqdm(file_infos, desc="Merging Names", disable=not verbose):
            with open(info['path'], 'rb') as in_f:
                in_f.seek(HEADER_SIZE) # Skip input file's header
                
                if info['max_len'] == global_max_len:
                    # FAST PATH: Raw byte streaming (zero RAM overhead)
                    names_bytes = info['N'] * global_max_len * 4
                    _stream_exact_bytes(in_f, out_f, names_bytes)
                else:
                    # SLOW PATH: Needs padding/casting because string lengths differ
                    # Process in chunks to protect RAM
                    chunk_size = 50000
                    for i in range(0, info['N'], chunk_size):
                        end_i = min(i + chunk_size, info['N'])
                        chunk_len = end_i - i
                        
                        # Read raw uint32 bytes for this chunk
                        raw_bytes = in_f.read(chunk_len * info['max_len'] * 4)
                        # View as numpy unicode array, cast to global length, write bytes
                        chunk_arr = np.frombuffer(raw_bytes, dtype=np.uint32).reshape(chunk_len, info['max_len'])
                        casted_arr = chunk_arr.view(np.dtype(f'<U{info["max_len"]}')).astype(f'<U{global_max_len}')
                        out_f.write(casted_arr.tobytes())

        # Phase 2: Merge Embeds
        if verbose: print("Phase 2: Merging Embeds...")
        for info in tqdm(file_infos, desc="Merging Embeds", disable=not verbose):
            with open(info['path'], 'rb') as in_f:
                # Skip header and names section to reach embeds
                embeds_offset = HEADER_SIZE + (info['N'] * info['max_len'] * 4)
                in_f.seek(embeds_offset)
                
                # FAST PATH: Stream raw float32 bytes directly into the output file
                embeds_bytes = info['N'] * global_D * 4
                _stream_exact_bytes(in_f, out_f, embeds_bytes)

    if verbose:
        file_size = os.path.getsize(output_filepath) / (1024**2)
        print(f"Merge complete! Final file size: {file_size:.2f} MB")


if __name__ == "__main__":
    # ==========================================
    # Example Usage / Self-Test
    # ==========================================
    print("Running module self-test...")
    
    # 1. Create some dummy data
    names1 = ["song_1", "song_2", "song_3", "song_4"]
    embeds1 = np.random.randn(4, 128).astype(np.float32)
    
    names2 = ["track_5", "track_6"]
    embeds2 = np.random.randn(2, 128).astype(np.float32)
    
    # 2. Save as individual SINGLE files
    save_paired_memmap("dataset1.bin", names1, embeds1, verbose=True)
    save_paired_memmap("dataset2.bin", names2, embeds2, verbose=True)
    
    print("-" * 40)
    
    # 3. Merge them
    merge_paired_memmaps(["dataset1.bin", "dataset2.bin"], "merged_dataset.bin", verbose=True)
    
    print("-" * 40)
    
    # 4. Load the merged file instantly via memmap
    names, embeds = load_paired_memmap("merged_dataset.bin", verbose=True)
    
    print(f"\nAccessing index 0:")
    print("Name:", names[0])
    print("Embed vector:", embeds[0])
    
    # Clean up test files
    os.remove("dataset1.bin")
    os.remove("dataset2.bin")
    os.remove("merged_dataset.bin")
    print("\nSelf-test complete. Cleaned up test files.")
    
###################################################################################

print('Module is loaded!')
print('Enjoy! :)')
print('=' * 70)

###################################################################################
# This is the end of the midisimx memmap Python Module
###################################################################################