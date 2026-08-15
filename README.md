# midisim
## Calculate, search, and analyze MIDI-to-MIDI similarity at scale

<img width="1536" height="1024" alt="midisim" src="https://github.com/user-attachments/assets/0b379b3a-ec9f-42c7-ba09-6b7cce87a338" />

***

## Main features

* Ultra-fast and flexible GPU/CPU MIDI-to-MIDI similarity calculation, search and analysis
* Quality pre-trained models and comprehensive pre-computed embeddings sets
* Stand-alone, versatile, and extensive codebase for general or custom MIDI-to-MIDI similarity tasks
* Full cross-platform compatibility and support

***

## [Pre-trained models](https://huggingface.co/projectlosangeles/midisim)

* ```midisim_small_pre_trained_model_2_epochs_43117_steps_0.3148_loss_0.9229_acc.pth``` - Very fast and accurate small model, suitable for all tasks. This model is included in PyPI package or it can be downloaded from Hugging Face
* ```midisim_large_pre_trained_model_2_epochs_86275_steps_0.2054_loss_0.9385_acc.pth``` - Fast large model for more nuanced embeddings generation. Download checkpoint from Hugging Face

#### Both pre-trained models were trained on full [Godzilla Piano](https://huggingface.co/datasets/asigalov61/Godzilla-Piano) dataset for 2 complete epochs

***

## [Pre-computed embeddings sets](https://huggingface.co/datasets/projectlosangeles/midisim-embeddings)

### For small pre-trained model

#### Mean Pool Embeddings
* These are standard general purpose embeddings which work best for comparing overall MIDI-to-MIDI similarity

```discover_midi_dataset_37292_genres_midis_embeddings_cc_by_nc_sa.npy``` - 37292 genre MIDIs embeddings for genre (artist and song) identification tasks

```discover_midi_dataset_202400_identified_midis_embeddings_cc_by_nc_sa.npy``` - 202400 identified MIDIs embeddings for MIDI identification tasks

```discover_midi_dataset_3480123_clean_midis_embeddings_cc_by_nc_sa.npy``` - 3480123 select clean MIDIs embeddings for large scale similarity search and analysis tasks

#### Weighted Mean Pool Embeddings
* These embeddings work best for comparing MIDI-tp=MIDI music structure (MIDI pitches) similarity.

```discover_midi_dataset_37302_genre_midis_embeddings_1_1_2_weighted_cc_by_nc_sa.npy``` - 37302 genre MIDIs weighted embeddings for genre (artist and song) identification tasks

```discover_midi_dataset_190032_identified_midis_embeddings_1_1_2_weighted_cc_by_nc_sa.npy``` - 190032 identified MIDIs weighted embeddings for MIDI identification tasks

```discover_midi_dataset_3480123_clean_midis_embeddings_1_1_2_weighted_cc_by_nc_sa.npy``` - 3480123 select clean MIDIs weighted embeddings for large scale similarity search and analysis tasks

### For large pre-trained model

#### Mean Pool Embeddings
* These are standard general purpose embeddings which work best for comparing overall MIDI-to-MIDI similarity

```discover_midi_dataset_37303_genres_midis_embeddings_large_cc_by_nc_sa.npy``` - 37303 genre MIDIs embeddings for genre (artist and song) identification tasks

```discover_midi_dataset_202400_identified_midis_embeddings_large_cc_by_nc_sa.npy``` - 202400 identified MIDIs embeddings for MIDI identification tasks

```discover_midi_dataset_3480123_clean_midis_embeddings_large_cc_by_nc_sa.npy``` - 3480123 select clean MIDIs embeddings for large scale similarity search and analysis tasks

#### Weighted Mean Pool Embeddings
* These embeddings work best for comparing MIDI-tp=MIDI music structure (MIDI pitches) similarity.

```discover_midi_dataset_37287_genres_midis_embeddings_1_1_2_weighted_large_cc_by_nc_sa.npy``` - 37287 genre MIDIs weighted embeddings for genre (artist and song) identification tasks

```discover_midi_dataset_190032_identified_midis_embeddings_1_1_2_weighted_large_cc_by_nc_sa.npy``` - 190032 identified MIDIs weighted embeddings for MIDI identification tasks

```discover_midi_dataset_3480123_clean_midis_embeddings_1_1_2_weighted_large_cc_by_nc_sa.npy``` - 3480123 select clean MIDIs weighted embeddings for large scale similarity search and analysis tasks

#### Source MIDI dataset: [Discover MIDI Dataset](https://huggingface.co/datasets/projectlosangeles/Discover-MIDI-Dataset)

***

### [Similarity search output samples](https://huggingface.co/datasets/projectlosangeles/midisim-samples)

```midisim-similarity-search-output-samples-CC-BY-NC-SA.zip``` - ~300000 MIDIs indentified with midisim music discovery pipeline with both pre-trained models

```midisim-similarity-search-output-samples-1-1-2-weighted-CC-BY-NC-SA.zip``` - ~366000 MIDIs indentified with weighted midisim music discovery pipeline with both pre-trained models

#### Source MIDI dataset: [Discover MIDI Dataset](https://huggingface.co/datasets/projectlosangeles/Discover-MIDI-Dataset)

***

## Installation

### midisim PyPI package (for general use)

```sh
!pip install -U midisim
```

### x-transformers 2.3.1 (for raw/custom tasks)

```sh
!pip install x-transformers==2.3.1
```

***

## Basic use guide

### General use example

```python
# ================================================================================================
# Initalize midisim
# ================================================================================================

# Import main midisim module
import midisim

# ================================================================================================
# Prepare midisim embeddings
# ================================================================================================

# Option 1: Download sample pre-computed embeddings corpus from Hugging Face
emb_path = midisim.download_embeddings()

# Option 2: use custom pre-computed embeddings corpus
# See custom embeddings generation section of this README for details
# emb_path = './custom_midis_embeddings_corpus.npy'

# Load downloaded embeddings corpus
corpus_midi_names, corpus_emb = midisim.load_embeddings(emb_path)

# ================================================================================================
# Prepare midisim model
# ================================================================================================

# Option 1: Download main pre-trained midisim model from Hugging Face
model_path = midisim.download_model()

# Option 2: Use main pre-trained midisim model included in midisim PyPI package
# model_path = midisim.get_package_models()[0]['path']

# Load midisim model
model, ctx, dtype = midisim.load_model(model_path)

# ================================================================================================
# Prepare source MIDI
# ================================================================================================

# Load source MIDI
input_toks_seqs = midisim.midi_to_tokens('Come To My Window.mid')

# ================================================================================================
# Calculate and analyze embeddings
# ================================================================================================

# Compute source/query embeddings
query_emb = midisim.get_embeddings_bf16(model, input_toks_seqs)

# Calculate cosine similarity between source/query MIDI embeddings and embeddings corpus
idxs, sims = midisim.cosine_similarity_topk(query_emb, corpus_emb)

# ================================================================================================
# Processs, print and save results
# ================================================================================================

# Convert the results to sorted list with transpose values
idxs_sims_tvs_list = midisim.idxs_sims_to_sorted_list(idxs, sims)

# Print corpus matches (and optionally) convert the final result to a handy list for further processing
corpus_matches_list = midisim.print_sorted_idxs_sims_list(idxs_sims_tvs_list, corpus_midi_names, return_as_list=True)

# ================================================================================================
# Copy matched MIDIs from the MIDI corpus for listening and further evaluation and analysis
# ================================================================================================

# Copy matched corpus MIDI to a desired directory for easy evaluation and analysis
out_dir_path = midisim.copy_corpus_files(corpus_matches_list)

# ================================================================================================
```

### Raw/custom use example

#### Small model (8 layers - 2 epochs)

```python
import torch
from x_transformers import TransformerWrapper, Encoder

# Original model hyperparameters
SEQ_LEN = 3072

MASK_IDX     = 384 # Use this value for masked modelling
PAD_IDX      = 385 # Model pad index
VOCAB_SIZE   = 386 # Total vocab size

MASK_PROB    = 0.15 # Original training mask probability value (use for masked modelling)

DEVICE = 'cuda' # You can use any compatible device or CPU
DTYPE  = torch.bfloat16 # Original training dtype

# Official main midisim model checkpoint name
MODEL_CKPT = 'midisim_small_pre_trained_model_2_epochs_43117_steps_0.3148_loss_0.9229_acc.pth'

# Model architecture using x-transformers
model = TransformerWrapper(
    num_tokens = VOCAB_SIZE,
    max_seq_len = SEQ_LEN,
    attn_layers = Encoder(
        dim   = 512,
        depth = 8,
        heads = 8,
        rotary_pos_emb = True,
        attn_flash = True,
    ),
)

model.load_state_dict(torch.load(MODEL_CKPT, map_location=DEVICE))

model.to(DEVICE)
model.eval()

# Original training autoxast setup
autocast_ctx = torch.amp.autocast(device_type=DEVICE, dtype=DTYPE)
```

#### Large model (16 layers - 2 epochs)

```python
import torch
from x_transformers import TransformerWrapper, Encoder

# Original model hyperparameters
SEQ_LEN = 3072

MASK_IDX     = 384 # Use this value for masked modelling
PAD_IDX      = 385 # Model pad index
VOCAB_SIZE   = 386 # Total vocab size

MASK_PROB    = 0.15 # Original training mask probability value (use for masked modelling)

DEVICE = 'cuda' # You can use any compatible device or CPU
DTYPE  = torch.bfloat16 # Original training dtype

# Official main midisim model checkpoint name
MODEL_CKPT = 'midisim_large_pre_trained_model_2_epochs_86275_steps_0.2054_loss_0.9385_acc.pth'

# Model architecture using x-transformers
model = TransformerWrapper(
    num_tokens = VOCAB_SIZE,
    max_seq_len = SEQ_LEN,
    attn_layers = Encoder(
        dim   = 512,
        depth = 16,
        heads = 8,
        rotary_pos_emb = True,
        attn_flash = True,
    ),
)

model.load_state_dict(torch.load(MODEL_CKPT, map_location=DEVICE))

model.to(DEVICE)
model.eval()

# Original training autoxast setup
autocast_ctx = torch.amp.autocast(device_type=DEVICE, dtype=DTYPE)
```

***

## Creating custom MIDI corpus embeddings

```python
# ================================================================================================

# Load main midisim module
import midisim

# Import helper modules
import os
import tqdm

# ================================================================================================

# Call included TMIDIX module through midisim to create MIDI files list
custom_midi_corpus_file_names = midisim.TMIDIX.create_files_list(['./custom_midi_corpus_dir/'])

# ================================================================================================

# Create two lists: one with MIDI corpus file names 
# and another with MIDI corpus tokens representations suitable for embeddings generation
midi_corpus_file_names = []
midi_corpus_tokens = []

for midi_file in tqdm.tqdm(custom_midi_corpus_file_names):
    midi_corpus_file_names.append(os.path.splitext(os.path.basename(midi_file))[0])
    
    midi_tokens = midisim.midi_to_tokens(midi_file, transpose_factor=0, verbose=False)[0]
    midi_corpus_tokens.append(midi_tokens)

# It is highly recommended to sort the resulting corpus by tokens sequence length
# This greatly speeds up embeddings calculations
sorted_midi_corpus = sorted(zip(midi_corpus_file_names, midi_corpus_tokens), key=lambda x: len(x[1]))
midi_corpus_file_names, midi_corpus_tokens = map(list, zip(*sorted_midi_corpus))

# ================================================================================================
# Now you are ready to generate embeddings as follows:
# ================================================================================================

# Load main midisim model
model, ctx, dtype = midisim.load_model(verbose=False)

# Generate MIDI corpus embeddings
midi_corpus_embeddings = midisim.get_embeddings_bf16(model, midi_corpus_tokens, verbose=False)

# ================================================================================================

# Save generated MIDI corpus embeddings and MIDI corpus file names in one handy NumPy file
midisim.save_embeddings(midi_corpus_file_names,
                        midi_corpus_embeddings,
                        verbose=False
                       )

# ================================================================================================

# You now can use this saved custom MIDI corpus NumPy file with midisim.load_embeddings()
# and the rest of the pipeline outlined in the general use section above
```

***

## Music discovery pipeline
Here is a complete MIDI music discovery pipeline example using midisim and [Discover MIDI Dataset](https://huggingface.co/datasets/projectlosangeles/Discover-MIDI-Dataset)

### Install midisim and discovermidi PyPI packages

```sh
!pip install -U midisim
```

```sh
!pip install -U discovermidi
```

### Download and unzip Discover MIDI Dataset

```python
import discovermidi
from discovermidi import fast_parallel_extract

discovermidi.download_dataset()

fast_parallel_extract.fast_parallel_extract()
```

### Choose and prepare one midisim model and corresponding embeddings set

#### Small model (8 layers)

```python
model_ckpt = 'midisim_small_pre_trained_model_2_epochs_43117_steps_0.3148_loss_0.9229_acc.pth'
model_depth = 8

embeddings_file = 'discover_midi_dataset_3480123_clean_midis_embeddings_cc_by_nc_sa.npy'
```

#### Large model (16 layers)

```python
model_ckpt = 'midisim_large_pre_trained_model_2_epochs_86275_steps_0.2054_loss_0.9385_acc.pth'
model_depth = 16

embeddings_file = 'discover_midi_dataset_3480123_clean_midis_embeddings_large_cc_by_nc_sa.npy'
```

### Create Master MIDI dataset directory and upload your source/master MIDIs in it

```python
import os

os.makedirs('./Master-MIDI-Dataset/', exist_ok=True)
```

### Initialize midisim, download and load chosen midisim model and embeddings set

```python
# Import main midisim module
import midisim

# Download embeddings from Hugging Face
emb_path = midisim.download_embeddings(filename=embeddings_file)

# Load downloaded embeddings corpus
corpus_midi_names, corpus_emb = midisim.load_embeddings(embeddings_path=emb_path)

# Download midisim model from Hugging Face
model_path = midisim.download_model(filename=model_ckpt)

# Load midisim model
model, ctx, dtype = midisim.load_model(model_path,
                                       depth=model_depth
                                      )
```

### Create Master MIDI dataset files list

```python
filez = midisim.TMIDIX.create_files_list(['./Master-MIDI-Dataset/'])
```

### Launch the search

```python
import os
import tqdm

for fa in tqdm.tqdm(filez):
    
    # Load source MIDI
    input_toks_seqs = midisim.midi_to_tokens(fa, verbose=False)

    if input_toks_seqs:
    
        # ================================================================================================
        # Calculate and analyze embeddings
        # ================================================================================================
        
        # Compute source/query embeddings
        query_emb = midisim.get_embeddings_bf16(model,
                                                input_toks_seqs,
                                                verbose=False,
                                                show_progress_bar=False
                                               )
    
        # Calculate cosine similarity between source/query MIDI embeddings and embeddings corpus
        idxs, sims = midisim.cosine_similarity_topk(query_emb,
													corpus_emb,
													verbose=False
												   )
       
        # ================================================================================================
        # Processs, print and save results
        # ================================================================================================
         
        # Convert the results to sorted list with transpose values
        idxs_sims_tvs_list = midisim.idxs_sims_to_sorted_list(idxs, sims)
       
        # Print corpus matches (and optionally) convert the final result to a handy list for further processing
        corpus_matches_list = midisim.print_sorted_idxs_sims_list(idxs_sims_tvs_list,
                                                                  corpus_midi_names,
                                                                  return_as_list=True
                                                                 )
         
        # ================================================================================================
        # Copy matched MIDIs from the MIDI corpus for listening and further evaluation and analysis
        # ================================================================================================
        
        # Copy matched corpus MIDI to a desired directory for easy evaluation and analysis
        out_dir_path = midisim.copy_corpus_files(corpus_matches_list,
                                                 corpus_midis_dirs=['./Discover-MIDI-Dataset/MIDIs/'],
                                                 main_output_dir='Output-MIDI-Dataset',
                                                 sub_output_dir=os.path.splitext(os.path.basename(fa))[0],
                                                 verbose=False
                                                )
        # ================================================================================================
```

***

## midisim functions reference lists

### Main functions

- ```midisim.copy_corpus_files``` — *Copy or synchronize MIDI corpus files from a source directory to a target corpus location.*  
- ```midisim.cosine_similarity_topk``` — *Compute cosine similarities between a query embedding and a set of embeddings and return the top‑K matches.*  
- ```midisim.download_all_embeddings``` — *Download an entire embeddings dataset snapshot from a Hugging Face dataset repository to a local directory.*  
- ```midisim.download_embeddings``` — *Download a single precomputed embeddings `.npy` file from a Hugging Face dataset repository.*  
- ```midisim.download_model``` — *Download a pre-trained model checkpoint file from a Hugging Face model repository to a local directory.*  
- ```midisim.get_embeddings_bf16``` — *Load or convert embeddings into bfloat16 format for memory-efficient inference on supported hardware.*  
- ```midisim.idxs_sims_to_sorted_list``` — *Convert parallel index and similarity arrays into a single sorted list of (index, similarity) pairs ordered by similarity.*  
- ```midisim.load_embeddings``` — *Load a saved NumPy embeddings file and return the arrays of MIDI names and corresponding embedding vectors.*  
- ```midisim.load_model``` — *Construct a Transformer model, load weights from a checkpoint, move it to the requested device, and return the model with an AMP autocast context and dtype.*  
- ```midisim.masked_mean_pool``` — *Compute a masked mean pooling over sequence embeddings, ignoring padded positions via a boolean or numeric mask.*  
- ```midisim.midi_to_tokens``` — *Convert a single-track MIDI file into one or more compact integer token sequences (with optional transpositions) suitable for model input.*  
- ```midisim.pad_and_mask``` — *Pad a batch of variable-length token sequences to a common length and produce an attention/mask tensor indicating real tokens vs padding.*  
- ```midisim.print_sorted_idxs_sims_list``` — *Pretty-print a sorted list of (index, similarity) pairs, optionally annotating entries with filenames or metadata.*  
- ```midisim.save_embeddings``` — *Save a list of name strings and their corresponding embedding vectors into a structured NumPy array and optionally persist it to disk.*

### Helper functions

- ```midisim.helpers.get_package_models``` — *Return a sorted list of packaged model files and their paths.*
- ```midisim.helpers.get_package_embeddings``` — *Return a sorted list of packaged embedding files and their paths.*
- ```midisim.helpers.get_normalized_midi_md5_hash``` — *Compute original and normalized MD5 hashes for a MIDI file.*
- ```midisim.helpers.normalize_midi_file``` — *Normalize a MIDI file and write the result to disk.*
- ```midisim.helpers.install_apt_package``` — *Idempotently install an apt package with retries and optional python‑apt.*

***

## Limitations

* Current code and models support only MIDI music elements similarity (start-times, durations and pitches)
* MIDI channels, instruments, velocities and drums similarites are not currently supported due to complexity and practicality considerations
* Current pre-trained models are limited by 3k sequence length (~1000 MIDI music notes) so long running MIDIs can only be analyzed in chunks
* Solo drum track MIDIs are not currently supported and can't be analyzed

***

## Citations

```bibtex
@misc{project_los_angeles_2025,
	author       = { Project Los Angeles },
	title        = { midisim (Revision 707e311) },
	year         = 2025,
	url          = { https://huggingface.co/projectlosangeles/midisim },
	doi          = { 10.57967/hf/7383 },
	publisher    = { Hugging Face }
}
```

```bibtex
@misc{project_los_angeles_2025,
	author       = { Project Los Angeles },
	title        = { midisim-embeddings (Revision 8ebb453) },
	year         = 2025,
	url          = { https://huggingface.co/datasets/projectlosangeles/midisim-embeddings },
	doi          = { 10.57967/hf/7382 },
	publisher    = { Hugging Face }
}
```

```bibtex
@misc{project_los_angeles_2025,
	author       = { Project Los Angeles },
	title        = { midisim-samples (Revision 79afcc1) },
	year         = 2025,
	url          = { https://huggingface.co/datasets/projectlosangeles/midisim-samples },
	doi          = { 10.57967/hf/7388 },
	publisher    = { Hugging Face }
}
```

```bibtex
@misc{project_los_angeles_2025,
	author       = { Project Los Angeles },
	title        = { Discover-MIDI-Dataset (Revision 0eaecb5) },
	year         = 2025,
	url          = { https://huggingface.co/datasets/projectlosangeles/Discover-MIDI-Dataset },
	doi          = { 10.57967/hf/7361 },
	publisher    = { Hugging Face }
}
```

***

### Project Los Angeles
### Tegridy Code 2026
