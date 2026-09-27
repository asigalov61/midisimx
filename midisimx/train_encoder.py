# -*- coding: utf-8 -*-
"""midisimx — masked-language-model (MLM) encoder training module.

Refactored (v2.0.0) stand-alone version of the original midisimx encoder
training script. The whole training pipeline lives in one primary function,
:func:`train_encoder`, which takes pre-tokenized sequences and returns the
trained model together with a complete dictionary of textual statistics.

Quick start (programmatic or Jupyter Lab)::

    from midisimx import train_encoder

    trained_model, stats = train_encoder.train_encoder(list_of_lists_of_seqs)

    # fully configurable, e.g.:
    trained_model, stats = train_encoder.train_encoder(
        sequences,
        num_epochs=5,
        batch_size=18,
        learning_rate=1e-4,
        val_split=0.02,
        test_split=0.02,
        output_dir="./encoder_training",
    )

Input data
----------
``train_data`` must contain pre-tokenized sequences (integer token ids):

* a list of lists / list of tuples,
* a 2-D numpy array,
* a 2-D torch tensor,
* or a list of numpy arrays / torch tensors (ragged lengths are fine).

Sequences longer than ``seq_len`` are truncated and shorter ones are
right-padded with ``pad_idx`` — exactly as in the original script.

Returns
-------
(model, stats):

* ``model`` — the trained, *unwrapped* ``TransformerWrapper`` (set to
  ``eval()`` mode; use ``.train()`` to continue training).
* ``stats`` — dict with the effective config, data/model info, per-step
  losses & running accuracies, per-validation metrics (loss, perplexity,
  token accuracy, top-k accuracy, F1 macro/weighted, precision/recall)
  and the final held-out test metrics.

Artifacts written to ``output_dir`` (created on demand)
-------------------------------------------------------
training.log                       full DEBUG log of the run
training_config.json               effective configuration
training_stats.json / .pkl         full statistics (also returned)
training_state.json                step/epoch counters (used for resume)
encoder_checkpoint_..._acc.pth     final model checkpoint (raw state_dict)
checkpoint_latest.pth              most recent periodic checkpoint
checkpoint_best.pth                best validation-loss checkpoint
optim.pth                          optimizer state

x-transformers compatibility (>= 2.28.0)
----------------------------------------
The original script depended on a locally modified x-transformers v2.3.1
(vendored as ``x_transformer_2_3_1``). This module uses the pip-installed
``x_transformers`` (>= 2.28.0 supported; 2.3.x also works): model constructor
kwargs are adapted automatically to the installed API via signature
introspection (e.g. ``attn_flash=True`` is passed directly on versions that
accept it, and as ``attn_kwargs={'flash': True}`` on versions that moved
attention flags into ``attn_kwargs``). Every decision is logged at startup —
check the ``Encoder kwargs:`` log line to confirm what was applied.

Resume
------
``resume_from='.../checkpoint_latest.pth'`` restores model + optimizer
weights, step/epoch counters and the previous training histories (companion
files in the same directory are picked up automatically). The partially
completed epoch is re-run from its beginning.

Jupyter Lab notes
-----------------
* Plain-text ``tqdm`` progress bars and ``logging`` output are used
  (no matplotlib, no torchsummary, no notebook widgets).
* The HF/flash environment flags are set *before* torch is imported. If
  torch was already imported in the running kernel, restart the kernel once
  for them to take full effect (same note as the original script).

Behavior-preserving fixes w.r.t. the original script
----------------------------------------------------
1. Vendored ``x_transformer_2_3_1`` replaced by the pip ``x_transformers``
   package with a cross-version adapter shim (see above).
2. Removed TMIDIX (native pickle/JSON writers instead), torchsummary
   (textual parameter counts instead) and matplotlib (textual stats).
3. Added train/val/test splits (defaults: 2% / 2%; pass ``val_split=0``,
   ``test_split=0`` to train on 100% of the data exactly like the original),
   periodic validation (``VALIDATE_EVERY`` — defined-but-unused in the
   original), end-of-epoch validation and a final full test evaluation.
4. Fixed gradient-accumulation scaling for the final *partial* accumulation
   window of each epoch (the original always divided by
   ``GRADIENT_ACCUMULATE_EVERY`` even when fewer micro-batches remained).
5. ``SAVE_EVERY`` (unused in the original, which hardcoded 1000) now drives
   periodic checkpointing; removed the hardcoded '/home/ubuntu/...' paths
   and the stray ``+3681`` step offset in the final checkpoint filename.
6. ``torch.load`` uses weights_only-safe loading (torch >= 2.6 compatible).
7. Input token ids are validated against ``vocab_size`` before training
   (out-of-range ids would otherwise crash deep inside the embedding), and
   masking fractions are validated.
8. Added: seeding, resume support, best-checkpoint tracking, config/state/
   stats dumps, textual per-epoch and per-N-step reports, tokens/s and
   peak-memory reporting, model parameter printout.
9. ``GENERATE_EVERY`` / ``GENERATE_LENGTH`` are kept as legacy constants
   only (an MLM encoder has no autoregressive generation step; both were
   unused in the original script).
"""

from __future__ import annotations

import sys

# ---------------------------------------------------------------------------
# Environment flags — must be set *before* torch is imported.
# NOTE (from the original script): if torch was already imported in the
# current session (e.g. a running Jupyter kernel), a kernel restart may be
# required for these to take full effect.
# ---------------------------------------------------------------------------
_TORCH_ALREADY_IMPORTED = "torch" in sys.modules

import os

os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")
os.environ.setdefault("USE_FLASH_ATTENTION", "1")

import gc
import json
import math
import pickle
import random
import socket
import time
import inspect
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm  # plain tqdm (no auto/notebook), as requested
from sklearn import metrics

# ---------------------------------------------------------------------------
# x-transformers (pip package; supports >= 2.28.0 as well as 2.3.x)
# ---------------------------------------------------------------------------
try:
    import x_transformers as _xt
    from x_transformers import TransformerWrapper, Encoder
except ImportError as _exc:  # pragma: no cover
    raise ImportError(
        "midisimx.train_encoder requires the 'x-transformers' package "
        "(pip install x-transformers>=2.28.0). Original import error: "
        f"{_exc!r}"
    ) from _exc

try:  # for API introspection (best-effort, optional)
    from x_transformers.x_transformers import AttentionLayers as _XTAttentionLayers  # type: ignore
except Exception:
    _XTAttentionLayers = None  # type: ignore

try:
    from x_transformers.x_transformers import Attention as _XTAttention  # type: ignore
except Exception:
    _XTAttention = None  # type: ignore


def _detect_x_transformers_version() -> str:
    v = getattr(_xt, "__version__", None)
    if v:
        return str(v)
    try:
        from importlib.metadata import version as _pkg_version

        return str(_pkg_version("x-transformers"))
    except Exception:
        return "unknown"


def _init_param_names(obj: Any) -> set:
    """Return the parameter names of ``obj.__init__`` (best-effort)."""
    try:
        fn = obj.__init__ if inspect.isclass(obj) else obj
        return set(inspect.signature(fn).parameters) - {"self"}
    except (TypeError, ValueError):
        return set()


XTRANSFORMERS_VERSION: str = _detect_x_transformers_version()
_ATTN_LAYERS_PARAMS: set = _init_param_names(_XTAttentionLayers) if _XTAttentionLayers is not None else set()
_ATTENTION_PARAMS: set = _init_param_names(_XTAttention) if _XTAttention is not None else set()

# ====================================================================
# Version 2.0.0 / Apache 2.0
#
# Project Los Angeles
# Tegridy Code 2026
# ====================================================================

__version__ = "2.0.0"
__author__ = "Project Los Angeles / Tegridy Code"
__license__ = "Apache-2.0"

__all__ = [
    "train_encoder",
    "mask_tokens",
    "TokenizedDataset",
    "SEQ_LEN",
    "MASK_PROB",
    "BATCH_SIZE",
    "MASK_IDX",
    "PAD_IDX",
    "VOCAB_SIZE",
    "VALIDATE_EVERY",
    "SAVE_EVERY",
    "PRINT_STATS_EVERY",
    "NUM_EPOCHS",
    "GRADIENT_ACCUMULATE_EVERY",
    "LEARNING_RATE",
    "GRAD_CLIP",
]

# ---------------------------------------------------------------------------
# Original script constants — kept EXACTLY as in the original working code.
# ---------------------------------------------------------------------------
SEQ_LEN: int = 3072   # sequence length (truncate/pad target)
MASK_PROB: float = 0.15   # fraction of eligible tokens selected for masking
BATCH_SIZE: int = 18

# Data vocabulary & special tokens
MASK_IDX: int = 718
PAD_IDX: int = MASK_IDX + 1
VOCAB_SIZE: int = PAD_IDX + 1

# Cadence constants (defined-but-unused / hardcoded in the original script;
# all of them are actually wired up in this refactor)
VALIDATE_EVERY: int = 100
SAVE_EVERY: int = 500
PRINT_STATS_EVERY: int = 10

# Legacy constants from the original script — unused for the masked encoder
# (an MLM encoder has no autoregressive "generation" step). Kept only for
# backwards compatibility with the original header.
GENERATE_EVERY: int = 250
GENERATE_LENGTH: int = 512

# Training constants
NUM_EPOCHS: int = 5
GRADIENT_ACCUMULATE_EVERY: int = 32
LEARNING_RATE: float = 1e-4
GRAD_CLIP: float = 1.0

# 80/10/10 BERT-style masking split (identical to the original logic)
MLM_REPLACE_FRACTION: float = 0.8  # 80% -> [MASK] token
MLM_RANDOM_FRACTION: float = 0.1   # 10% -> random token; last 10% -> unchanged

# Runtime defaults
DEVICE: str = "cuda"
DTYPE: torch.dtype = torch.bfloat16

# The original script hardcoded '/home/ubuntu/...' — replaced by a proper
# output directory (created on demand).
DEFAULT_OUTPUT_DIR: str = "encoder_training"

logger: logging.Logger = logging.getLogger("midisimx.train_encoder")

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
SequenceInput = Union[
    Sequence[Sequence[int]],
    Sequence[np.ndarray],
    Sequence[torch.Tensor],
    np.ndarray,
    torch.Tensor,
]


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------
def _unwrap(model: Any) -> Any:
    """Return the underlying module of a torch.compile'd model (if any)."""
    return getattr(model, "_orig_mod", model)


def _jsonable(obj: Any) -> Any:
    """Recursively convert an object into JSON-serializable primitives."""
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, torch.dtype):
        return str(obj)
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, datetime):
        return obj.isoformat()
    return str(obj)


def _torch_load_compat(path: Union[str, Path], map_location: Any = "cpu") -> Any:
    """``torch.load`` wrapper handling the ``weights_only`` default flip in torch >= 2.6."""
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)  # very old torch
    except Exception:
        # weights_only=True can reject non-trivial pickles — retry unrestricted
        return torch.load(path, map_location=map_location, weights_only=False)


def _setup_logging(out_dir: Path, verbose: bool, log_level: Union[int, str]) -> None:
    """(Re)configure the module logger: console (optional) + DEBUG log file."""
    logger.setLevel(logging.DEBUG)
    # Remove handlers attached by previous calls (Jupyter re-runs)
    for handler in list(logger.handlers):
        if getattr(handler, "_midisimx_train_encoder", False):
            logger.removeHandler(handler)
            try:
                handler.close()
            except Exception:
                pass

    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    if verbose:
        console = logging.StreamHandler(sys.stdout)
        console.setLevel(log_level)
        console.setFormatter(fmt)
        console._midisimx_train_encoder = True  # type: ignore[attr-defined]
        logger.addHandler(console)

    file_handler = logging.FileHandler(out_dir / "training.log", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(fmt)
    file_handler._midisimx_train_encoder = True  # type: ignore[attr-defined]
    logger.addHandler(file_handler)
    logger.propagate = False


def _configure_torch_backends() -> None:
    """Configure torch global backends exactly as in the original script."""
    torch.set_float32_matmul_precision("high")
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
        torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_cudnn_sdp(True)
    try:
        torch.backends.cudnn.benchmark = True
    except Exception:
        pass


def _count_parameters(model: Any) -> Tuple[int, int]:
    """Return (total, trainable) parameter counts."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return int(total), int(trainable)


# ---------------------------------------------------------------------------
# Data handling
# ---------------------------------------------------------------------------
def _coerce_sequences(train_data: SequenceInput) -> List[List[int]]:
    """Coerce the supported input containers into ``List[List[int]]``.

    Accepts: list of lists/tuples (of ints), list of numpy arrays / torch
    tensors, a 2-D numpy array, or a 2-D torch tensor of pre-tokenized
    sequences.
    """
    if isinstance(train_data, torch.Tensor):
        if train_data.dim() != 2:
            raise ValueError(
                f"torch.Tensor input must be 2-D (num_sequences, seq_length); got shape {tuple(train_data.shape)}"
            )
        return train_data.detach().cpu().long().tolist()

    if isinstance(train_data, np.ndarray):
        if train_data.dtype == object:
            return [[int(tok) for tok in seq] for seq in train_data]
        if train_data.ndim != 2:
            raise ValueError(f"numpy input must be 2-D; got {train_data.ndim} dimension(s)")
        return train_data.astype(np.int64).tolist()

    sequences: List[List[int]] = []
    for i, seq in enumerate(train_data):
        if isinstance(seq, torch.Tensor):
            seq = seq.detach().cpu().tolist()
        elif isinstance(seq, np.ndarray):
            seq = seq.tolist()
        if not isinstance(seq, (list, tuple)):
            raise TypeError(
                f"Sequence #{i} has unsupported type '{type(seq).__name__}'; "
                "expected list/tuple/np.ndarray/torch.Tensor of integer token ids"
            )
        sequences.append(list(seq))
    return sequences


def _describe_sequences(sequences: List[List[int]]) -> Dict[str, Any]:
    """Compute textual dataset statistics (lengths, token id range, ...)."""
    lengths = [len(s) for s in sequences]
    n = len(sequences)
    total_tokens = int(sum(lengths))
    return {
        "num_sequences": n,
        "total_tokens": total_tokens,
        "min_length": int(min(lengths)) if lengths else 0,
        "max_length": int(max(lengths)) if lengths else 0,
        "mean_length": (total_tokens / n) if n else 0.0,
        "num_empty_sequences": int(sum(1 for ln in lengths if ln == 0)),
        "min_token_id": int(min((min(s) for s in sequences if s), default=0)),
        "max_token_id": int(max((max(s) for s in sequences if s), default=-1)),
    }


def _split_sequences(
    sequences: List[List[int]],
    val_split: float,
    test_split: float,
    seed: Optional[int],
) -> Tuple[List[List[int]], List[List[int]], List[List[int]]]:
    """Deterministically (given ``seed``) split into train/val/test lists."""
    n = len(sequences)
    n_val = int(round(n * val_split)) if val_split and val_split > 0 else 0
    n_test = int(round(n * test_split)) if test_split and test_split > 0 else 0
    if n_val + n_test >= n:
        raise ValueError(
            f"val_split={val_split} / test_split={test_split} leave no training sequences (total: {n})"
        )
    rng = random.Random(seed)
    indices = list(range(n))
    rng.shuffle(indices)
    val = [sequences[i] for i in indices[:n_val]]
    test = [sequences[i] for i in indices[n_val : n_val + n_test]]
    train = [sequences[i] for i in indices[n_val + n_test :]]
    return train, val, test


class TokenizedDataset(Dataset):
    """Map-style dataset of pre-tokenized sequences (truncate + right-pad).

    Behavior is identical to the original script: each sequence is truncated
    to ``seq_len`` and, if shorter, right-padded with ``pad_idx``.
    """

    def __init__(
        self,
        sequences: Sequence[Sequence[int]],
        seq_len: int = SEQ_LEN,
        pad_idx: int = PAD_IDX,
    ) -> None:
        self.data: Sequence[Sequence[int]] = sequences
        self.seq_len: int = int(seq_len)
        self.pad_idx: int = int(pad_idx)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> torch.Tensor:
        seq = list(self.data[idx][: self.seq_len])
        if len(seq) < self.seq_len:
            seq += [self.pad_idx] * (self.seq_len - len(seq))
        return torch.LongTensor(seq)


# ---------------------------------------------------------------------------
# MLM masking (identical logic to the original script)
# ---------------------------------------------------------------------------
def mask_tokens(
    inputs: torch.Tensor,
    mask_prob: float = MASK_PROB,
    mask_idx: int = MASK_IDX,
    pad_idx: int = PAD_IDX,
    vocab_size: int = VOCAB_SIZE,
    replace_fraction: float = MLM_REPLACE_FRACTION,
    random_fraction: float = MLM_RANDOM_FRACTION,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """BERT-style 80/10/10 MLM masking (identical to the original script).

    Of all eligible (non-pad) positions selected with probability
    ``mask_prob``: ``replace_fraction`` are replaced by ``mask_idx``,
    ``random_fraction`` by a random token in ``[0, vocab_size)``, and the
    remainder keep their original token. Returns ``(masked_inputs, labels)``
    where ``labels`` is ``-100`` everywhere except at masked positions.
    """
    B, L = inputs.shape
    device = inputs.device

    labels = torch.full_like(inputs, -100)

    # positions eligible for masking (never mask padding)
    u = torch.rand((B, L), device=device)
    mask_pos = (u < mask_prob) & (inputs != pad_idx)
    labels[mask_pos] = inputs[mask_pos]

    # second random draw for the 80/10/10 split
    u2 = torch.rand((B, L), device=device)
    masked = inputs.clone()

    # 80% -> [MASK] token
    masked[mask_pos & (u2 < replace_fraction)] = mask_idx

    # 10% -> random token
    rand_region = mask_pos & (u2 >= replace_fraction) & (u2 < replace_fraction + random_fraction)
    num_rand = int(rand_region.sum())
    if num_rand > 0:
        masked[rand_region] = torch.randint(
            low=0,
            high=vocab_size,  # IMPORTANT (original fix): exclusive upper bound
            size=(num_rand,),
            device=device,
        )

    # remaining 10% -> keep original token (already correct)
    return masked, labels


# ---------------------------------------------------------------------------
# Model construction (x-transformers cross-version adapter)
# ---------------------------------------------------------------------------
def _build_model(
    *,
    vocab_size: int,
    seq_len: int,
    model_dim: int,
    depth: int,
    heads: int,
    rotary_pos_emb: bool,
    attn_flash: bool,
    extra_model_kwargs: Optional[Dict[str, Any]] = None,
) -> TransformerWrapper:
    """Instantiate the masked encoder: ``TransformerWrapper + Encoder``.

    Compatible with both older (<= 2.3.x) and newer (>= 2.28.0) versions of
    lucidrains' x-transformers: constructor kwargs are adapted to the actual
    installed API via signature introspection, and every adaptation is logged.
    """
    kwargs: Dict[str, Any] = dict(extra_model_kwargs or {})

    if _ATTN_LAYERS_PARAMS:
        # --- rotary positional embeddings ---
        rotary_flag = next((name for name in ("rotary_pos_emb", "rotary_emb", "rotary") if name in _ATTN_LAYERS_PARAMS), None)
        if rotary_flag is not None:
            kwargs[rotary_flag] = rotary_pos_emb
        else:
            logger.warning(
                "This x-transformers version does not accept a rotary embedding flag "
                "on its attention layers — parameter dropped (rotary disabled)."
            )

        # --- flash attention ---
        if "attn_flash" in _ATTN_LAYERS_PARAMS:
            kwargs["attn_flash"] = attn_flash
        elif attn_flash:
            flash_flag = "flash" if "flash" in _ATTENTION_PARAMS else ("attn_flash" if "attn_flash" in _ATTENTION_PARAMS else None)
            if flash_flag is not None and "attn_kwargs" in _ATTN_LAYERS_PARAMS:
                attn_kwargs = dict(kwargs.get("attn_kwargs") or {})
                attn_kwargs.setdefault(flash_flag, True)
                kwargs["attn_kwargs"] = attn_kwargs
                logger.info(
                    "x-transformers >= 2.28 API: flash attention enabled via attn_kwargs={'%s': True}.",
                    flash_flag,
                )
            else:
                logger.warning(
                    "Flash attention flag not found in this x-transformers version "
                    "— using the library's default attention path."
                )
    else:
        # Could not introspect the API — assume the historical (2.3.x) kwargs.
        kwargs.update(rotary_pos_emb=rotary_pos_emb, attn_flash=attn_flash)

    logger.info("Encoder kwargs: %s", kwargs)

    try:
        attn_layers = Encoder(dim=model_dim, depth=depth, heads=heads, **kwargs)
    except TypeError as exc:
        if extra_model_kwargs:
            logger.warning("Encoder rejected extra kwargs (%r); retrying without extra_model_kwargs.", exc)
            core = {k: v for k, v in kwargs.items() if k in ("rotary_pos_emb", "rotary_emb", "rotary", "attn_flash", "attn_kwargs")}
            attn_layers = Encoder(dim=model_dim, depth=depth, heads=heads, **core)
        else:
            raise

    return TransformerWrapper(num_tokens=vocab_size, max_seq_len=seq_len, attn_layers=attn_layers)


# ---------------------------------------------------------------------------
# Evaluation (loss / accuracy / top-k / F1 / precision / recall / perplexity)
# ---------------------------------------------------------------------------
def _evaluate(
    model: Any,
    loader: DataLoader,
    *,
    device: torch.device,
    dtype: torch.dtype,
    vocab_size: int,
    mask_prob: float,
    mask_idx: int,
    pad_idx: int,
    mlm_replace_fraction: float,
    mlm_random_fraction: float,
    max_batches: Optional[int] = None,
    desc: str = "Validation",
    show_progress: bool = False,
    eval_top_k: int = 5,
) -> Dict[str, Any]:
    """Evaluate MLM loss/accuracy/F1-style metrics over a dataloader.

    If ``max_batches`` is given, at most that many (shuffled) batches are
    evaluated — pass ``None`` for a full pass (used for the final test eval).
    """
    was_training = model.training
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_topk = 0
    total_masked = 0
    total_seqs = 0
    n_batches = 0
    all_preds: List[torch.Tensor] = []
    all_labels: List[torch.Tensor] = []

    k = max(1, min(int(eval_top_k), vocab_size))
    total_planned = min(max_batches, len(loader)) if max_batches is not None else len(loader)

    with torch.no_grad():
        for batch in tqdm(
            loader,
            total=total_planned,
            desc=desc,
            leave=False,
            disable=not show_progress,
            dynamic_ncols=True,
            unit="batch",
        ):
            inputs = batch.to(device, non_blocking=True)
            total_seqs += inputs.size(0)

            masked_inputs, labels = mask_tokens(
                inputs,
                mask_prob=mask_prob,
                mask_idx=mask_idx,
                pad_idx=pad_idx,
                vocab_size=vocab_size,
                replace_fraction=mlm_replace_fraction,
                random_fraction=mlm_random_fraction,
            )
            attn_mask = torch.ones_like(masked_inputs).bool()

            with torch.amp.autocast(device_type=device.type, dtype=dtype):
                logits = model(masked_inputs, mask=attn_mask)
                loss = F.cross_entropy(logits.view(-1, vocab_size), labels.view(-1), ignore_index=-100)

            total_loss += loss.item()
            n_batches += 1

            mask_pos = labels.ne(-100)
            if mask_pos.any():
                preds = logits.argmax(dim=-1)
                total_correct += preds[mask_pos].eq(labels[mask_pos]).sum().item()
                if k > 1:
                    topk = logits.topk(k, dim=-1).indices
                    total_topk += topk[mask_pos].eq(labels[mask_pos].unsqueeze(-1)).any(dim=-1).sum().item()
                total_masked += mask_pos.sum().item()
                all_preds.append(preds[mask_pos].detach().cpu())
                all_labels.append(labels[mask_pos].detach().cpu())

            if max_batches is not None and n_batches >= max_batches:
                break

    if was_training:
        model.train()

    avg_loss = total_loss / max(n_batches, 1)
    result: Dict[str, Any] = {
        "loss": avg_loss,
        "perplexity": math.exp(min(avg_loss, 20.0)),
        "accuracy": (total_correct / total_masked) if total_masked else 0.0,
        "top_k_accuracy": (total_topk / total_masked) if total_masked else 0.0,
        "num_batches": n_batches,
        "num_sequences": total_seqs,
        "num_masked_tokens": total_masked,
    }

    if all_preds:
        y_pred = torch.cat(all_preds).numpy()
        y_true = torch.cat(all_labels).numpy()
        result["f1_macro"] = float(metrics.f1_score(y_true, y_pred, average="macro", zero_division=0))
        result["f1_weighted"] = float(metrics.f1_score(y_true, y_pred, average="weighted", zero_division=0))
        result["precision_macro"] = float(metrics.precision_score(y_true, y_pred, average="macro", zero_division=0))
        result["recall_macro"] = float(metrics.recall_score(y_true, y_pred, average="macro", zero_division=0))

    return result


def _record_validation(stats: Dict[str, Any], step: int, m: Dict[str, Any]) -> None:
    """Append one validation result to the stats history."""
    v = stats["validation"]
    v["steps"].append(int(step))
    v["losses"].append(float(m.get("loss", float("nan"))))
    v["accs"].append(float(m.get("accuracy", float("nan"))))
    v["top_k_accs"].append(float(m.get("top_k_accuracy", float("nan"))))
    v["f1_macro"].append(float(m.get("f1_macro", float("nan"))))
    v["f1_weighted"].append(float(m.get("f1_weighted", float("nan"))))
    v["precision_macro"].append(float(m.get("precision_macro", float("nan"))))
    v["recall_macro"].append(float(m.get("recall_macro", float("nan"))))
    v["perplexities"].append(float(m.get("perplexity", float("nan"))))


def _log_eval_metrics(prefix: str, m: Dict[str, Any]) -> None:
    """One-line textual report of an evaluation result."""
    logger.info(
        "%s loss %.4f | ppl %.2f | token acc %.4f | top-k acc %.4f | "
        "F1 macro %.4f | F1 weighted %.4f | precision %.4f | recall %.4f | masked tokens %d",
        prefix,
        m.get("loss", float("nan")),
        m.get("perplexity", float("nan")),
        m.get("accuracy", float("nan")),
        m.get("top_k_accuracy", float("nan")),
        m.get("f1_macro", float("nan")),
        m.get("f1_weighted", float("nan")),
        m.get("precision_macro", float("nan")),
        m.get("recall_macro", float("nan")),
        int(m.get("num_masked_tokens", 0)),
    )


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------
def _save_stats(out_dir: Path, stats: Dict[str, Any]) -> None:
    """Write the full statistics dict as JSON and pickle."""
    try:
        (out_dir / "training_stats.json").write_text(
            json.dumps(_jsonable(stats), indent=2), encoding="utf-8"
        )
    except Exception as exc:
        logger.warning("Could not write training_stats.json (%r)", exc)
    try:
        with open(out_dir / "training_stats.pkl", "wb") as f:
            pickle.dump(stats, f)
    except Exception as exc:
        logger.warning("Could not write training_stats.pkl (%r)", exc)


def _write_training_state(
    out_dir: Path,
    *,
    global_step: int,
    epoch: int,
    last_loss: float,
    last_acc: float,
    best_val_loss: float,
) -> None:
    """Write lightweight resume state (step/epoch counters and best val loss)."""
    payload = {
        "global_step": int(global_step),
        "epoch": int(epoch),
        "last_step_loss": float(last_loss),
        "last_running_acc": float(last_acc),
        "best_val_loss": None if best_val_loss == float("inf") else float(best_val_loss),
        "saved_at": datetime.now(timezone.utc).isoformat(),
    }
    try:
        (out_dir / "training_state.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    except Exception as exc:
        logger.warning("Could not write training_state.json (%r)", exc)


# ===========================================================================
# Primary (main) training function
# ===========================================================================
def train_encoder(
    train_data: SequenceInput,
    *,
    # ---- data / masking ---------------------------------------------------
    seq_len: int = SEQ_LEN,
    mask_prob: float = MASK_PROB,
    mask_idx: int = MASK_IDX,
    pad_idx: Optional[int] = None,           # default: mask_idx + 1 (original)
    vocab_size: Optional[int] = None,        # default: pad_idx + 1  (original)
    mlm_replace_fraction: float = MLM_REPLACE_FRACTION,
    mlm_random_fraction: float = MLM_RANDOM_FRACTION,
    val_split: float = 0.02,
    test_split: float = 0.02,
    val_max_batches: Optional[int] = 100,
    num_workers: int = 0,
    # ---- model ------------------------------------------------------------
    model_dim: int = 768,
    depth: int = 16,
    heads: int = 12,
    rotary_pos_emb: bool = True,
    attn_flash: bool = True,
    extra_model_kwargs: Optional[Dict[str, Any]] = None,
    # ---- optimization -----------------------------------------------------
    num_epochs: int = NUM_EPOCHS,
    batch_size: int = BATCH_SIZE,
    gradient_accumulate_every: int = GRADIENT_ACCUMULATE_EVERY,
    learning_rate: float = LEARNING_RATE,
    grad_clip: float = GRAD_CLIP,
    adamw_weight_decay: float = 0.01,  # PyTorch AdamW default (as in original)
    # ---- cadence ----------------------------------------------------------
    validate_every: int = VALIDATE_EVERY,
    save_every: int = SAVE_EVERY,
    print_stats_every: int = PRINT_STATS_EVERY,
    eval_top_k: int = 5,
    # ---- runtime / I/O ----------------------------------------------------
    device: Union[str, torch.device] = DEVICE,
    dtype: torch.dtype = DTYPE,
    torch_compile: bool = True,
    output_dir: Union[str, os.PathLike] = DEFAULT_OUTPUT_DIR,
    save_checkpoints: bool = True,
    save_optimizer: bool = True,
    save_stats: bool = True,
    save_best: bool = True,
    resume_from: Optional[Union[str, os.PathLike]] = None,
    resume_optimizer: bool = True,
    seed: Optional[int] = None,
    verbose: bool = True,
    log_level: Union[int, str] = logging.INFO,
) -> Tuple[TransformerWrapper, Dict[str, Any]]:
    """Train the midisimx masked encoder (MLM) end-to-end and return it.

    This is the single primary function of the module: it runs the complete
    pipeline (data coercion/validation/splitting, dataloaders, model
    construction, bf16 AMP training with gradient accumulation, periodic
    validation/checkpointing, final test evaluation and artifact saving).

    Args:
        train_data: Pre-tokenized sequences — list of lists/tuples of ints,
            2-D numpy array, 2-D torch tensor, or a list of numpy arrays /
            tensors. Longer sequences are truncated to ``seq_len``; shorter
            ones are right-padded with ``pad_idx``.
        seq_len: Training sequence length (truncate/pad target). Default 3072.
        mask_prob: Fraction of eligible (non-pad) tokens selected for MLM
            masking. Default 0.15.
        mask_idx: Vocabulary id of the ``[MASK]`` token. Default 718.
        pad_idx: Vocabulary id of the ``[PAD]`` token. Default ``mask_idx + 1``.
        vocab_size: Total vocabulary size. Default ``pad_idx + 1``.
        mlm_replace_fraction: Of the masked positions, the fraction replaced
            by ``mask_idx`` (default 0.8 — BERT-style 80/10/10).
        mlm_random_fraction: Of the masked positions, the fraction replaced by
            a random token (default 0.1; the remaining fraction is unchanged).
        val_split: Fraction of sequences held out for periodic validation
            (0 disables; use 0 to train on 100% of the data like the original).
        test_split: Fraction of sequences held out for the final test
            evaluation (0 disables).
        val_max_batches: Cap on validation batches per validation run (the
            val loader is shuffled, so each run sees a random subset). Pass
            ``None``/``0`` for a full pass each time.
        num_workers: DataLoader workers (default 0 — Jupyter-safe).
        model_dim / depth / heads: Encoder dimensions (default 768 / 16 / 12).
        rotary_pos_emb: Use rotary positional embeddings (default True).
        attn_flash: Use flash attention where supported (default True).
        extra_model_kwargs: Optional extra kwargs forwarded to the
            x-transformers ``Encoder`` constructor.
        num_epochs: Number of training epochs. Default 5.
        batch_size: Micro-batch size. Default 18.
        gradient_accumulate_every: Micro-batches per optimizer step. Default 32.
        learning_rate: AdamW learning rate. Default 1e-4.
        grad_clip: Gradient norm clip. Default 1.0.
        adamw_weight_decay: AdamW weight decay (PyTorch default 0.01, matching
            the original script's implicit setting).
        validate_every: Run validation every N optimizer steps (0 disables).
        save_every: Save periodic checkpoint/optimizer/stats every N optimizer
            steps (0 disables).
        print_stats_every: Log running loss/acc every N optimizer steps.
        eval_top_k: k for the top-k accuracy metric. Default 5.
        device: ``'cuda'`` (default), ``'cuda:0'``, ``'cpu'``, etc.
        dtype: AMP dtype — bfloat16 by default (as the original; a GradScaler
            is not needed for bf16).
        torch_compile: Wrap the model in ``torch.compile`` (default True, as
            the original; falls back to eager with a warning on failure).
        output_dir: Directory for all artifacts (created on demand).
        save_checkpoints / save_optimizer / save_stats / save_best: Artifact
            saving switches.
        resume_from: Path to a model checkpoint to resume from; companion
            ``optim.pth`` / ``training_state.json`` / ``training_stats.pkl``
            in the same directory are picked up automatically.
        resume_optimizer: Whether to restore optimizer state on resume.
        seed: Optional seed for python/numpy/torch and the data split.
        verbose: Console logging + progress bars (file logging is always on).
        log_level: Console log level (default ``logging.INFO``).

    Returns:
        Tuple ``(model, stats)`` — the trained (unwrapped) model in ``eval()``
        mode and the full statistics dictionary.
    """
    # =======================================================================
    # 0) Setup: output dir, logging, environment sanity checks
    # =======================================================================
    started_at = time.time()
    started_iso = datetime.now(timezone.utc).isoformat()

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _setup_logging(out_dir, verbose=verbose, log_level=log_level)

    logger.info("=" * 68)
    logger.info("midisimx masked encoder training (train_encoder v%s)", __version__)
    logger.info("=" * 68)

    if _TORCH_ALREADY_IMPORTED:
        logger.warning(
            "torch was already imported before this module — the HF_XET_HIGH_PERFORMANCE / "
            "USE_FLASH_ATTENTION environment flags may not take full effect. A kernel restart "
            "may be required (see original script note)."
        )
    logger.info("Torch version: %s | x-transformers version: %s", torch.__version__, XTRANSFORMERS_VERSION)

    # ---- argument sanity checks ------------------------------------------
    if seq_len <= 0 or batch_size <= 0 or gradient_accumulate_every <= 0:
        raise ValueError("seq_len, batch_size and gradient_accumulate_every must be positive")
    if num_epochs < 1:
        raise ValueError("num_epochs must be >= 1")
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    if not (0.0 < mask_prob <= 1.0):
        raise ValueError(f"mask_prob must be in (0, 1]; got {mask_prob}")
    if not (
        0.0 <= mlm_replace_fraction
        and 0.0 <= mlm_random_fraction
        and mlm_replace_fraction + mlm_random_fraction <= 1.0
    ):
        raise ValueError("MLM fractions must satisfy 0 <= replace, 0 <= random, replace + random <= 1")

    # =======================================================================
    # 1) Derived vocabulary constants (original: MASK/PAD/VOCAB = 718/719/720)
    # =======================================================================
    pad_idx_val = (mask_idx + 1) if pad_idx is None else int(pad_idx)
    vocab_size_val = (pad_idx_val + 1) if vocab_size is None else int(vocab_size)
    if not (0 <= mask_idx < vocab_size_val):
        raise ValueError(f"mask_idx ({mask_idx}) must be within [0, vocab_size={vocab_size_val})")
    if not (0 <= pad_idx_val < vocab_size_val):
        raise ValueError(f"pad_idx ({pad_idx_val}) must be within [0, vocab_size={vocab_size_val})")

    # =======================================================================
    # 2) Device / torch backends / seeding
    # =======================================================================
    _configure_torch_backends()
    dev = device if isinstance(device, torch.device) else torch.device(device)
    if dev.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "device='cuda' was requested but CUDA is not available. "
            "Pass device='cpu' (slow, not recommended for this model size) or run on a CUDA machine."
        )
    if dev.type == "cuda":
        try:
            logger.info(
                "CUDA device: %s | bf16 supported: %s",
                torch.cuda.get_device_name(dev),
                torch.cuda.is_bf16_supported(),
            )
        except Exception:
            pass
    if dev.type != "cuda" and dtype == torch.bfloat16:
        logger.warning("bfloat16 autocast on %s may be slow or unsupported; consider dtype=torch.float32.", dev.type)
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats(dev)

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        logger.info("Seeded python/numpy/torch with seed=%d", seed)
    else:
        logger.info("No seed provided — run is non-deterministic (pass seed=... for reproducibility).")

    # =======================================================================
    # 3) Data: coerce, validate, describe
    # =======================================================================
    logger.info("Loading and validating input data...")
    sequences = _coerce_sequences(train_data)
    if not sequences:
        raise ValueError("train_data is empty — nothing to train on")
    data_stats = _describe_sequences(sequences)
    if data_stats["max_token_id"] >= vocab_size_val or data_stats["min_token_id"] < 0:
        raise ValueError(
            f"Input data token id range [{data_stats['min_token_id']}, {data_stats['max_token_id']}] "
            f"is outside the valid range [0, {vocab_size_val - 1}] for vocab_size={vocab_size_val}. "
            "Please pass the correct vocab_size/mask_idx/pad_idx for your data."
        )
    logger.info(
        "Data: %d sequences | %d tokens | len min/mean/max: %d/%.1f/%d | empty: %d | token ids: [%d, %d]",
        data_stats["num_sequences"],
        data_stats["total_tokens"],
        data_stats["min_length"],
        data_stats["mean_length"],
        data_stats["max_length"],
        data_stats["num_empty_sequences"],
        data_stats["min_token_id"],
        data_stats["max_token_id"],
    )

    # =======================================================================
    # 4) Effective configuration (logged + saved)
    # =======================================================================
    config: Dict[str, Any] = {
        "seq_len": seq_len,
        "mask_prob": mask_prob,
        "mask_idx": mask_idx,
        "pad_idx": pad_idx_val,
        "vocab_size": vocab_size_val,
        "mlm_replace_fraction": mlm_replace_fraction,
        "mlm_random_fraction": mlm_random_fraction,
        "val_split": val_split,
        "test_split": test_split,
        "val_max_batches": val_max_batches,
        "num_workers": num_workers,
        "model_dim": model_dim,
        "depth": depth,
        "heads": heads,
        "rotary_pos_emb": rotary_pos_emb,
        "attn_flash": attn_flash,
        "extra_model_kwargs": extra_model_kwargs,
        "num_epochs": num_epochs,
        "batch_size": batch_size,
        "gradient_accumulate_every": gradient_accumulate_every,
        "learning_rate": learning_rate,
        "grad_clip": grad_clip,
        "adamw_weight_decay": adamw_weight_decay,
        "validate_every": validate_every,
        "save_every": save_every,
        "print_stats_every": print_stats_every,
        "eval_top_k": eval_top_k,
        "device": str(dev),
        "dtype": str(dtype),
        "torch_compile": torch_compile,
        "output_dir": str(out_dir),
        "save_checkpoints": save_checkpoints,
        "save_optimizer": save_optimizer,
        "save_stats": save_stats,
        "save_best": save_best,
        "resume_from": str(resume_from) if resume_from else None,
        "resume_optimizer": resume_optimizer,
        "seed": seed,
        "torch_version": torch.__version__,
        "x_transformers_version": XTRANSFORMERS_VERSION,
        "train_encoder_version": __version__,
    }
    logger.info("Configuration:")
    for key, value in config.items():
        logger.info("  %s = %s", key, value)
    try:
        (out_dir / "training_config.json").write_text(json.dumps(_jsonable(config), indent=2), encoding="utf-8")
    except Exception as exc:
        logger.warning("Could not write training_config.json (%r)", exc)

    # =======================================================================
    # 5) Splits & dataloaders
    # =======================================================================
    train_seqs, val_seqs, test_seqs = _split_sequences(sequences, val_split, test_split, seed)
    logger.info(
        "Split: train=%d (%.1f%%) | val=%d (%.1f%%) | test=%d (%.1f%%)",
        len(train_seqs), 100.0 * len(train_seqs) / len(sequences),
        len(val_seqs), 100.0 * len(val_seqs) / len(sequences),
        len(test_seqs), 100.0 * len(test_seqs) / len(sequences),
    )

    shuffle_generator = torch.Generator().manual_seed(seed) if seed is not None else None
    train_loader = DataLoader(
        TokenizedDataset(train_seqs, seq_len=seq_len, pad_idx=pad_idx_val),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=(dev.type == "cuda"),
        generator=shuffle_generator,
    )
    if len(train_loader) == 0:
        raise ValueError(
            f"Not enough training sequences ({len(train_seqs)}) to fill a single batch of "
            f"{batch_size} — reduce batch_size or the val/test splits."
        )
    full_steps_per_epoch = math.ceil(len(train_loader) / gradient_accumulate_every)

    val_loader: Optional[DataLoader] = None
    if val_seqs:
        val_loader = DataLoader(
            TokenizedDataset(val_seqs, seq_len=seq_len, pad_idx=pad_idx_val),
            batch_size=batch_size,
            shuffle=True,
            drop_last=False,
            num_workers=num_workers,
            pin_memory=(dev.type == "cuda"),
        )
    test_loader: Optional[DataLoader] = None
    if test_seqs:
        test_loader = DataLoader(
            TokenizedDataset(test_seqs, seq_len=seq_len, pad_idx=pad_idx_val),
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=num_workers,
            pin_memory=(dev.type == "cuda"),
        )

    logger.info(
        "Train loader: %d batches/epoch -> %d optimizer steps/epoch "
        "(effective batch: %d sequences x %d accum = %d sequences per step) | val loader: %s | test loader: %s",
        len(train_loader),
        full_steps_per_epoch,
        batch_size,
        gradient_accumulate_every,
        batch_size * gradient_accumulate_every,
        f"{len(val_loader)} batches" if val_loader is not None else "none",
        f"{len(test_loader)} batches" if test_loader is not None else "none",
    )

    # =======================================================================
    # 6) Model: build -> resume weights -> compile -> parameter printout
    # =======================================================================
    logger.info(
        "Building model: TransformerWrapper(num_tokens=%d, max_seq_len=%d) + Encoder(dim=%d, depth=%d, heads=%d, rotary=%s, flash=%s)",
        vocab_size_val, seq_len, model_dim, depth, heads, rotary_pos_emb, attn_flash,
    )
    model = _build_model(
        vocab_size=vocab_size_val,
        seq_len=seq_len,
        model_dim=model_dim,
        depth=depth,
        heads=heads,
        rotary_pos_emb=rotary_pos_emb,
        attn_flash=attn_flash,
        extra_model_kwargs=extra_model_kwargs,
    )
    model = model.to(dev)

    # ---- resume: model weights (before compilation) -----------------------
    resume_dir: Optional[Path] = None
    if resume_from is not None:
        resume_path = Path(resume_from)
        if not resume_path.is_file():
            raise FileNotFoundError(f"resume_from not found: {resume_path}")
        resume_dir = resume_path.parent
        logger.info("Resuming model weights from %s", resume_path)
        state = _torch_load_compat(resume_path, map_location=dev)
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        _unwrap(model).load_state_dict(state)
        logger.info("Model weights restored.")

    if torch_compile:
        try:
            model = torch.compile(model)
            logger.info("Model compiled with torch.compile (checkpoints use the unwrapped module).")
        except Exception as exc:
            logger.warning("torch.compile failed (%r) — continuing without compilation.", exc)

    n_params, n_trainable = _count_parameters(model)
    logger.info(
        "Model parameters: %d total / %d trainable (~%.1f MB fp32 / ~%.1f MB bf16 weights)",
        n_params, n_trainable, n_params * 4 / 1024**2, n_params * 2 / 1024**2,
    )
    for name, child in _unwrap(model).named_children():
        logger.info("  %-16s %12d params", name, sum(p.numel() for p in child.parameters()))
    logger.debug("Full model repr:\n%s", _unwrap(model))

    # =======================================================================
    # 7) Optimizer, stats container, resume bookkeeping
    # =======================================================================
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=adamw_weight_decay)
    logger.info(
        "Optimizer: AdamW | lr=%g | weight_decay=%g | grad_clip=%g | accumulation=%d | amp dtype=%s",
        learning_rate, adamw_weight_decay, grad_clip, gradient_accumulate_every, dtype,
    )

    stats: Dict[str, Any] = {
        "run_info": {
            "started": started_iso,
            "finished": None,
            "duration_sec": None,
            "hostname": socket.gethostname(),
            "torch_version": torch.__version__,
            "x_transformers_version": XTRANSFORMERS_VERSION,
        },
        "config": config,
        "data": {
            "overall": data_stats,
            "train_sequences": len(train_seqs),
            "val_sequences": len(val_seqs),
            "test_sequences": len(test_seqs),
        },
        "model": {"total_params": n_params, "trainable_params": n_trainable, "compiled": False},
        "epochs": [],
        "train": {"step_indices": [], "step_losses": [], "step_accs": []},
        "validation": {
            "steps": [], "losses": [], "accs": [], "top_k_accs": [],
            "f1_macro": [], "f1_weighted": [], "precision_macro": [],
            "recall_macro": [], "perplexities": [],
        },
        "best_validation": None,
        "test": None,
        "interrupted": False,
    }
    if torch_compile and getattr(model, "_orig_mod", None) is not None:
        stats["model"]["compiled"] = True

    # ---- resume: optimizer state, counters, previous histories ------------
    global_step = 0
    start_epoch = 1
    best_val_loss = float("inf")
    if resume_dir is not None:
        if resume_optimizer:
            opt_path = resume_dir / "optim.pth"
            if opt_path.is_file():
                try:
                    optimizer.load_state_dict(_torch_load_compat(opt_path, map_location=dev))
                    logger.info("Optimizer state restored from %s", opt_path)
                except Exception as exc:
                    logger.warning("Could not restore optimizer state (%r) — starting optimizer fresh.", exc)
        state_path = resume_dir / "training_state.json"
        if state_path.is_file():
            try:
                ts = json.loads(state_path.read_text(encoding="utf-8"))
                global_step = int(ts.get("global_step", 0))
                start_epoch = max(1, int(ts.get("epoch", 1)))  # partially-completed epoch is re-run
                if ts.get("best_val_loss") is not None:
                    best_val_loss = float(ts["best_val_loss"])
                logger.info(
                    "Resumed training state: global_step=%d, start_epoch=%d, best_val_loss=%s",
                    global_step, start_epoch, "inf" if best_val_loss == float("inf") else f"{best_val_loss:.4f}",
                )
            except Exception as exc:
                logger.warning("Could not read training_state.json (%r).", exc)
        stats_path = resume_dir / "training_stats.pkl"
        if stats_path.is_file():
            try:
                with open(stats_path, "rb") as f:
                    prev = pickle.load(f)
                if isinstance(prev, dict):
                    for key in ("step_indices", "step_losses", "step_accs"):
                        stats["train"][key] = list(prev.get("train", {}).get(key, []))
                    for key in ("steps", "losses", "accs", "top_k_accs", "f1_macro", "f1_weighted", "precision_macro", "recall_macro", "perplexities"):
                        stats["validation"][key] = list(prev.get("validation", {}).get(key, []))
                    stats["epochs"] = list(prev.get("epochs", []))
                    logger.info("Restored previous training histories from %s", stats_path)
            except Exception as exc:
                logger.warning("Could not restore previous stats (%r).", exc)
    if start_epoch > num_epochs:
        logger.info(
            "start_epoch (%d) > num_epochs (%d) — training loop skipped (resume already complete); "
            "running final evaluation/artifacts only.",
            start_epoch, num_epochs,
        )

    # =======================================================================
    # 8) Baseline (pre-training) validation
    # =======================================================================
    if val_loader is not None:
        logger.info("Running baseline (pre-training) validation...")
        baseline = _evaluate(
            model, val_loader,
            device=dev, dtype=dtype, vocab_size=vocab_size_val,
            mask_prob=mask_prob, mask_idx=mask_idx, pad_idx=pad_idx_val,
            mlm_replace_fraction=mlm_replace_fraction, mlm_random_fraction=mlm_random_fraction,
            max_batches=(val_max_batches if val_max_batches else None),
            desc="Baseline validation", show_progress=verbose, eval_top_k=eval_top_k,
        )
        _record_validation(stats, 0, baseline)
        _log_eval_metrics("Baseline val:", baseline)

    # =======================================================================
    # 9) Training loop (identical logic to the original, with partial-window fix)
    # =======================================================================
    current_epoch = start_epoch
    interrupted = False

    try:
        for epoch in range(start_epoch, num_epochs + 1):
            current_epoch = epoch
            model.train()
            running_loss = 0.0
            running_correct = 0
            running_total = 0
            epoch_start = time.time()
            epoch_tokens = 0

            data_iter = iter(train_loader)
            batches_remaining = len(train_loader)

            pbar = tqdm(
                total=full_steps_per_epoch,
                desc=f"Epoch {epoch}/{num_epochs}",
                disable=not verbose,
                dynamic_ncols=True,
                unit="step",
            )

            for full_step in range(1, full_steps_per_epoch + 1):
                global_step += 1

                # FIX: size of the *actual* accumulation window — the final
                # window of an epoch may hold fewer micro-batches than
                # gradient_accumulate_every; scaling by the actual window
                # keeps gradient magnitudes identical to the original code
                # for all full windows and correct for the tail window.
                window = min(gradient_accumulate_every, batches_remaining)
                if window <= 0:
                    break

                optimizer.zero_grad(set_to_none=True)

                accumulated_loss_value = 0.0
                accumulated_correct = 0
                accumulated_total_mask = 0
                micro_steps_done = 0

                for micro in range(window):
                    try:
                        batch = next(data_iter)
                    except StopIteration:
                        break  # defensive; window math above prevents this

                    micro_steps_done += 1

                    inputs, labels = mask_tokens(
                        batch,
                        mask_prob=mask_prob,
                        mask_idx=mask_idx,
                        pad_idx=pad_idx_val,
                        vocab_size=vocab_size_val,
                        replace_fraction=mlm_replace_fraction,
                        random_fraction=mlm_random_fraction,
                    )
                    inputs = inputs.to(dev, non_blocking=True)
                    labels = labels.to(dev, non_blocking=True)
                    mask = torch.ones_like(inputs).bool()

                    with torch.amp.autocast(device_type=dev.type, dtype=dtype):
                        logits = model(inputs, mask=mask)
                        loss = F.cross_entropy(
                            logits.view(-1, vocab_size_val),
                            labels.view(-1),
                            ignore_index=-100,
                        )

                    accumulated_loss_value += loss.item()

                    # scale by the actual accumulation window size (see FIX above)
                    loss = loss / window
                    loss.backward()

                    with torch.no_grad():
                        mask_pos = labels.ne(-100)
                        if mask_pos.any():
                            preds = logits.argmax(dim=-1)
                            accumulated_correct += preds[mask_pos].eq(labels[mask_pos]).sum().item()
                            accumulated_total_mask += mask_pos.sum().item()

                # if no micro-steps ran, skip optimizer step entirely
                if micro_steps_done == 0:
                    break  # end of epoch

                batches_remaining -= window

                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

                epoch_tokens += window * batch_size * seq_len  # approximate (incl. padding)

                # average loss over the *micro-steps actually processed*
                avg_batch_loss = accumulated_loss_value / window
                running_loss += avg_batch_loss
                running_correct += accumulated_correct
                running_total += accumulated_total_mask

                avg_loss = running_loss / full_step
                avg_acc = running_correct / running_total if running_total else 0.0

                pbar.set_postfix(loss=f"{avg_loss:.4f}", acc=f"{avg_acc:.4f}")
                pbar.update(1)

                stats["train"]["step_indices"].append(global_step)
                stats["train"]["step_losses"].append(avg_batch_loss)
                stats["train"]["step_accs"].append(avg_acc)

                if verbose and print_stats_every > 0 and global_step % print_stats_every == 0:
                    logger.info(
                        "[step %6d] running loss %.4f | running MLM acc %.4f | last step loss %.4f",
                        global_step, avg_loss, avg_acc, avg_batch_loss,
                    )

                # ---- periodic validation ---------------------------------
                if val_loader is not None and validate_every > 0 and global_step % validate_every == 0:
                    val_metrics = _evaluate(
                        model, val_loader,
                        device=dev, dtype=dtype, vocab_size=vocab_size_val,
                        mask_prob=mask_prob, mask_idx=mask_idx, pad_idx=pad_idx_val,
                        mlm_replace_fraction=mlm_replace_fraction, mlm_random_fraction=mlm_random_fraction,
                        max_batches=(val_max_batches if val_max_batches else None),
                        desc=f"Validation @ step {global_step}", show_progress=verbose, eval_top_k=eval_top_k,
                    )
                    _record_validation(stats, global_step, val_metrics)
                    _log_eval_metrics(f"Validation @ step {global_step}:", val_metrics)
                    if val_metrics["loss"] < best_val_loss:
                        best_val_loss = val_metrics["loss"]
                        stats["best_validation"] = {"step": global_step, **val_metrics}
                        if save_checkpoints and save_best:
                            torch.save(_unwrap(model).state_dict(), out_dir / "checkpoint_best.pth")
                            logger.info("New best validation loss (%.4f) — checkpoint_best.pth saved.", best_val_loss)

                # ---- periodic checkpointing -------------------------------
                if save_checkpoints and save_every > 0 and global_step % save_every == 0:
                    torch.save(_unwrap(model).state_dict(), out_dir / "checkpoint_latest.pth")
                    if save_optimizer:
                        torch.save(optimizer.state_dict(), out_dir / "optim.pth")
                    _write_training_state(
                        out_dir, global_step=global_step, epoch=epoch,
                        last_loss=avg_batch_loss, last_acc=avg_acc, best_val_loss=best_val_loss,
                    )
                    if save_stats:
                        _save_stats(out_dir, stats)
                    logger.info("Periodic checkpoint saved @ step %d (checkpoint_latest.pth).", global_step)

            pbar.close()

            # ---- end-of-epoch summary (original format + extras) ----------
            train_loss = running_loss / full_steps_per_epoch if full_steps_per_epoch else 0.0
            train_acc = running_correct / running_total if running_total else 0.0
            epoch_dur = time.time() - epoch_start
            tokens_per_sec = epoch_tokens / epoch_dur if epoch_dur > 0 else 0.0

            epoch_val: Optional[Dict[str, Any]] = None
            if val_loader is not None:
                epoch_val = _evaluate(
                    model, val_loader,
                    device=dev, dtype=dtype, vocab_size=vocab_size_val,
                    mask_prob=mask_prob, mask_idx=mask_idx, pad_idx=pad_idx_val,
                    mlm_replace_fraction=mlm_replace_fraction, mlm_random_fraction=mlm_random_fraction,
                    max_batches=(val_max_batches if val_max_batches else None),
                    desc=f"Validation (epoch {epoch} end)", show_progress=verbose, eval_top_k=eval_top_k,
                )
                _record_validation(stats, global_step, epoch_val)
                _log_eval_metrics(f"Epoch {epoch} validation:", epoch_val)
                if epoch_val["loss"] < best_val_loss:
                    best_val_loss = epoch_val["loss"]
                    stats["best_validation"] = {"step": global_step, **epoch_val}
                    if save_checkpoints and save_best:
                        torch.save(_unwrap(model).state_dict(), out_dir / "checkpoint_best.pth")
                        logger.info("New best validation loss (%.4f) — checkpoint_best.pth saved.", best_val_loss)

            stats["epochs"].append({
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "duration_sec": epoch_dur,
                "tokens_per_sec": tokens_per_sec,
                "val": epoch_val,
            })

            logger.info(
                "Epoch %d/%d — Loss: %.4f — MLM Acc: %.4f — %.1fs — ~%.0f tokens/s",
                epoch, num_epochs, train_loss, train_acc, epoch_dur, tokens_per_sec,
            )

            gc.collect()
            if dev.type == "cuda":
                torch.cuda.empty_cache()

    except KeyboardInterrupt:
        interrupted = True
        logger.warning("KeyboardInterrupt received — saving emergency checkpoint and stats before returning...")
        torch.save(_unwrap(model).state_dict(), out_dir / "checkpoint_latest.pth")
        if save_optimizer:
            torch.save(optimizer.state_dict(), out_dir / "optim.pth")
        _last_loss = stats["train"]["step_losses"][-1] if stats["train"]["step_losses"] else 0.0
        _last_acc = stats["train"]["step_accs"][-1] if stats["train"]["step_accs"] else 0.0
        _write_training_state(
            out_dir, global_step=global_step, epoch=current_epoch,
            last_loss=_last_loss, last_acc=_last_acc, best_val_loss=best_val_loss,
        )
        if save_stats:
            _save_stats(out_dir, stats)

    # =======================================================================
    # 10) Finalization: test evaluation, final artifacts, summary
    # =======================================================================
    if not interrupted and test_loader is not None:
        logger.info(
            "Running final evaluation on the held-out test split (%d sequences, full pass)...",
            len(test_seqs),
        )
        test_metrics = _evaluate(
            model, test_loader,
            device=dev, dtype=dtype, vocab_size=vocab_size_val,
            mask_prob=mask_prob, mask_idx=mask_idx, pad_idx=pad_idx_val,
            mlm_replace_fraction=mlm_replace_fraction, mlm_random_fraction=mlm_random_fraction,
            max_batches=None,  # full pass
            desc="Final test evaluation", show_progress=verbose, eval_top_k=eval_top_k,
        )
        stats["test"] = test_metrics
        _log_eval_metrics("FINAL TEST:", test_metrics)
    elif test_loader is None:
        logger.info("No test split — skipping final test evaluation.")

    final_loss = stats["train"]["step_losses"][-1] if stats["train"]["step_losses"] else 0.0
    final_acc = stats["train"]["step_accs"][-1] if stats["train"]["step_accs"] else 0.0

    if save_checkpoints:
        fname = f"encoder_checkpoint_{global_step}_steps_{round(final_loss, 4)}_loss_{round(final_acc, 4)}_acc.pth"
        torch.save(_unwrap(model).state_dict(), out_dir / fname)
        torch.save(_unwrap(model).state_dict(), out_dir / "checkpoint_latest.pth")
        logger.info("Final model checkpoint saved: %s", out_dir / fname)
    if save_optimizer:
        torch.save(optimizer.state_dict(), out_dir / "optim.pth")
        logger.info("Optimizer state saved: %s", out_dir / "optim.pth")
    _write_training_state(
        out_dir, global_step=global_step, epoch=current_epoch,
        last_loss=final_loss, last_acc=final_acc, best_val_loss=best_val_loss,
    )
    if save_stats:
        _save_stats(out_dir, stats)

    if dev.type == "cuda":
        peak_gb = torch.cuda.max_memory_allocated(dev) / 1024**3
        stats["model"]["peak_cuda_memory_gb"] = peak_gb
        logger.info("Peak CUDA memory: %.2f GB", peak_gb)

    stats["interrupted"] = interrupted
    stats["run_info"]["finished"] = datetime.now(timezone.utc).isoformat()
    stats["run_info"]["duration_sec"] = time.time() - started_at

    trained_model = _unwrap(model)
    trained_model.eval()

    logger.info("=" * 68)
    logger.info(
        "TRAINING COMPLETE%s in %.1f min",
        " (interrupted)" if interrupted else "",
        stats["run_info"]["duration_sec"] / 60.0,
    )
    logger.info("Final training loss: %.4f | final running MLM acc: %.4f", final_loss, final_acc)
    if stats.get("best_validation"):
        logger.info(
            "Best validation loss: %.4f @ step %d",
            stats["best_validation"]["loss"], stats["best_validation"]["step"],
        )
    if stats.get("test"):
        t = stats["test"]
        logger.info(
            "Test: acc %.4f | F1 macro %.4f | F1 weighted %.4f",
            t["accuracy"], t.get("f1_macro", 0.0), t.get("f1_weighted", 0.0),
        )
    logger.info("All artifacts are in: %s", out_dir.resolve())
    logger.info("Returned model is the unwrapped module, set to eval() mode.")
    logger.info("=" * 68)

    return trained_model, stats


if __name__ == "__main__":  # pragma: no cover
    # No CLI execution intended — call train_encoder() programmatically
    # or from Jupyter Lab. See the module docstring for usage examples.
    print(__doc__)