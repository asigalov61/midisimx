"""
pca_reduce
==========

Streaming, GPU-accelerated dimensionality reduction of large embedding
matrices via Principal Component Analysis (PCA).

The main entry point is :func:`pca_reduce_embeddings`, which reduces an
``(n_samples, n_features)`` array of embeddings (e.g. 768-dimensional
encoder vectors) to ``(n_samples, target_dim)`` using a two-pass,
batch-streaming algorithm:

1. **Pass 1 (float64)** -- streams over the input in batches and
   accumulates the global mean and the full covariance matrix online, so
   the dataset never has to fit in device memory at once.
2. **Eigen-decomposition** -- the ``(n_features x n_features)`` covariance
   matrix is diagonalized with ``torch.linalg.eigh``.
3. **Pass 2 (float32)** -- every batch is centered with the global mean and
   projected onto the top ``target_dim`` eigenvectors.

Optionally, all intermediate artifacts (mean, covariance, eigenvalues,
eigenvectors, projection matrix) and the reduced embeddings can be saved
to disk as ``.npy`` files by passing ``save_dir``.

This module has no import-time side effects; all work happens inside
:func:`pca_reduce_embeddings`.
"""

from __future__ import annotations

import os
import time
import warnings
from dataclasses import dataclass, field
from typing import Dict, Optional, Union

import numpy as np
import torch
from tqdm import tqdm

__all__ = ["pca_reduce_embeddings", "PCAReductionResult"]

# ---------------------------------------------------------------------------
# Default artifact file names used when ``save_dir`` is passed to
# :func:`pca_reduce_embeddings`.
# ---------------------------------------------------------------------------
MEAN_FILENAME = "pca_mean.npy"
COV_FILENAME = "pca_cov.npy"
EIGVALS_FILENAME = "pca_eigvals.npy"
EIGVECS_FILENAME = "pca_eigvecs.npy"
PROJ_FILENAME = "pca_projection_matrix.npy"
REDUCED_FILENAME = "embeddings_reduced.npy"

ArrayLike = Union[np.ndarray, torch.Tensor]


# ---------------------------------------------------------------------------
# RESULT CONTAINER
# ---------------------------------------------------------------------------
@dataclass
class PCAReductionResult:
    """Container for the outputs of :func:`pca_reduce_embeddings`.

    Attributes
    ----------
    reduced : numpy.ndarray or torch.Tensor
        The projected embeddings, shape ``(n_samples, target_dim)``,
        dtype float32. A ``torch.Tensor`` sharing memory with the numpy
        buffer if ``return_torch=True`` was passed, else a
        ``numpy.ndarray``.
    mean : numpy.ndarray
        Global mean of the input embeddings, shape ``(input_dim,)``,
        float64.
    covariance : numpy.ndarray
        Full covariance matrix of the input, shape
        ``(input_dim, input_dim)``, float64, normalized by ``n - 1``.
    eigenvalues : numpy.ndarray
        Eigenvalues of the covariance matrix, sorted in descending order,
        shape ``(input_dim,)``, float64 (clamped at zero).
    eigenvectors : numpy.ndarray
        Corresponding eigenvectors as columns, shape
        ``(input_dim, input_dim)``, float64, such that
        ``covariance ~= eigenvectors @ diag(eigenvalues) @ eigenvectors.T``.
    projection_matrix : numpy.ndarray
        Top-``target_dim`` eigenvectors, shape ``(input_dim, target_dim)``,
        float32 -- the matrix ``W`` used for the projection
        ``(x - mean) @ W``.
    explained_variance_ratio : numpy.ndarray
        Fraction of total variance explained by each component
        (descending), shape ``(input_dim,)``.
    cumulative_explained_variance : numpy.ndarray
        Cumulative sum of ``explained_variance_ratio``, shape
        ``(input_dim,)``.
    n_samples : int
        Number of input embeddings.
    input_dim : int
        Dimensionality of the input embeddings.
    target_dim : int
        Requested reduced dimensionality.
    device : str
        String representation of the compute device that was used.
    timings : dict
        Wall-clock durations (seconds) of the phases:
        ``"pass1_mean_cov"``, ``"eigendecomposition"``,
        ``"pass2_projection"`` and ``"total"``.
    """

    reduced: Union[np.ndarray, torch.Tensor]
    mean: np.ndarray
    covariance: np.ndarray
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    projection_matrix: np.ndarray
    explained_variance_ratio: np.ndarray
    cumulative_explained_variance: np.ndarray
    n_samples: int
    input_dim: int
    target_dim: int
    device: str
    timings: Dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# INTERNAL HELPERS
# ---------------------------------------------------------------------------
def _to_torch(batch, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    """Convert a data batch to a ``torch.Tensor`` on ``device`` with ``dtype``.

    Accepts numpy arrays (any subclass), PyTorch tensors, or anything
    ``np.asarray`` can handle. A copy is made only when a dtype/device
    conversion is actually required. Unlike ``torch.from_numpy``, this also
    works for tensor inputs and read-only numpy arrays.
    """
    if isinstance(batch, torch.Tensor):
        return batch.to(device=device, dtype=dtype)
    return torch.as_tensor(np.asarray(batch), dtype=dtype, device=device)


def _format_bytes(num_bytes: float) -> str:
    """Format a byte count as a human-readable string (KiB/MiB/GiB/...)."""
    size = float(num_bytes)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if size < 1024.0 or unit == "TiB":
            return f"{size:,.1f} {unit}"
        size /= 1024.0
    return f"{size:,.1f} TiB"


def _variance_checkpoints(target_dim: int, input_dim: int):
    """Return sorted component counts for the explained-variance table.

    The list contains power-of-two style checkpoints below ``target_dim``
    plus ``target_dim`` itself, so the printed table always adapts to the
    requested reduction size. All values are clamped to ``[1, input_dim]``.
    """
    base = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512)
    ks = {k for k in base if 1 <= k < target_dim}
    ks.add(target_dim)
    return sorted(k for k in ks if 1 <= k <= input_dim)


def _save_array(directory, filename: str, array: np.ndarray, verbose: bool) -> None:
    """Save ``array`` as ``.npy`` inside ``directory`` (created if needed).

    Prints the destination path, shape and dtype when ``verbose`` is True.
    """
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, filename)
    np.save(path, array)
    if verbose:
        print(f"Saved {filename} (shape={array.shape}, dtype={array.dtype}) -> {path}")


def _sync(device: torch.device) -> None:
    """Synchronize ``device`` so CUDA timings are accurate (no-op on CPU)."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)


# ---------------------------------------------------------------------------
# MAIN API
# ---------------------------------------------------------------------------
def pca_reduce_embeddings(
    embeddings: ArrayLike,
    target_dim: int = 128,
    batch_size: int = 512_000,
    *,
    device: Optional[Union[str, torch.device]] = None,
    use_tqdm: bool = True,
    verbose: bool = True,
    debug: bool = False,
    save_dir: Optional[Union[str, os.PathLike]] = None,
    exact_covariance: bool = True,
    return_torch: bool = False,
) -> PCAReductionResult:
    """Reduce high-dimensional embeddings with streaming, two-pass PCA.

    The input is processed in batches so the full dataset never has to be
    materialized on the compute device:

    **Pass 1 (float64)** computes the global mean and the full covariance
    matrix online. With ``exact_covariance=True`` (default) the batch
    statistics are merged with the running statistics using the exact
    pairwise update of Chan et al. (1982), including a rank-1 correction
    term for the distance between the batch mean and the global mean, so
    the result is exact regardless of how the data is ordered. With
    ``exact_covariance=False`` the original, cheaper approximation is used,
    where each batch is centered by its *own* mean; this is accurate for
    large, randomly ordered batches but *underestimates* the covariance
    when the data is sorted/ordered across batches.

    **Pass 2 (float32)** projects every batch onto the top ``target_dim``
    eigenvectors and writes the result into a preallocated float32 array
    of shape ``(n_samples, target_dim)`` which is returned (and optionally
    saved).

    Parameters
    ----------
    embeddings : array_like or torch.Tensor, shape (n_samples, n_features)
        Input embeddings. Numpy arrays and PyTorch tensors are accepted
        directly (any dtype); other array-likes are converted with
        ``np.asarray``. Only batches are ever copied to the compute device,
        so the input may be larger than device memory.
    target_dim : int, default=128
        Number of principal components to keep. Must satisfy
        ``1 <= target_dim <= n_features``. A warning is emitted if it
        exceeds the achievable rank ``min(n_samples - 1, n_features)``.
    batch_size : int, default=512_000
        Number of rows processed per batch. The pass-1 device-memory
        footprint is roughly ``batch_size * n_features * 8`` bytes
        (float64). Lower this value for smaller hardware.
    device : str, torch.device or None, default=None
        Compute device for all tensor math. If None, uses ``"cuda"`` when
        available, otherwise ``"cpu"``.
    use_tqdm : bool, default=True
        Show tqdm progress bars (one per pass).
    verbose : bool, default=True
        Print progress statistics: input summary, per-phase timings and
        throughput, sample counts, mean norm, total variance, a dynamic
        explained-variance table, variance-coverage thresholds, output
        size, and saved-file paths.
    debug : bool, default=False
        Print extra diagnostics: torch/CUDA environment info, a finiteness
        check of the first batch, per-batch statistics for a sample of
        batches, covariance symmetry error and diagonal range, eigenvalue
        spectrum statistics (largest/smallest eigenvalue, number of
        negative eigenvalues before clamping, condition number of the top
        ``target_dim`` block), and per-pass peak GPU memory usage.
    save_dir : str, os.PathLike or None, default=None
        If given, all artifacts are saved to this directory (created if
        necessary) under the filenames ``pca_mean.npy``, ``pca_cov.npy``,
        ``pca_eigvals.npy``, ``pca_eigvecs.npy``, ``pca_projection_matrix.npy``
        and ``embeddings_reduced.npy``. If None (default), nothing is
        written to disk.
    exact_covariance : bool, default=True
        Use the exact online covariance update (Chan et al., 1982). Set to
        False to reproduce the batch-mean-centered approximation of the
        original script.
    return_torch : bool, default=False
        If True, ``PCAReductionResult.reduced`` is a float32
        ``torch.Tensor`` instead of a ``numpy.ndarray`` (sharing memory
        with the internal numpy buffer).

    Returns
    -------
    PCAReductionResult
        Dataclass containing the reduced embeddings, mean, covariance,
        eigenvalues/eigenvectors, projection matrix, explained-variance
        ratios, dataset dimensions, device string and phase timings. See
        :class:`PCAReductionResult` for details.

    Raises
    ------
    ValueError
        If ``embeddings`` is not 2-D, has fewer than two rows, if
        ``target_dim`` is outside ``[1, n_features]``, if CUDA is
        explicitly requested but unavailable, or if the covariance has
        zero or non-finite total variance (e.g. constant input or NaNs).
    RuntimeWarning
        If ``target_dim`` exceeds the achievable rank, or (in debug mode)
        the first batch contains non-finite values.

    Notes
    -----
    * Statistics (mean, covariance, eigen-decomposition) are computed in
      float64 for numerical stability; the final projection is performed
      in float32, matching the numerics of e.g. scikit-learn's PCA.
    * Eigenvalues are clamped at zero after decomposition: for a
      positive-semidefinite covariance, ``torch.linalg.eigh`` can return
      tiny negative values due to floating-point noise.
    * All computation runs under ``torch.no_grad()``, so inputs with
      ``requires_grad=True`` are also handled safely.
    * The reduced result is held in host RAM and returned; make sure
      ``n_samples * target_dim * 4`` bytes fit in memory.

    Examples
    --------
    >>> from pca_reduce import pca_reduce_embeddings
    >>> result = pca_reduce_embeddings(emb, target_dim=64, batch_size=500_000)
    >>> result.reduced.shape
    (12000000, 64)
    >>> result.cumulative_explained_variance[63]   # variance kept by 64 comps
    0.987...
    >>> result = pca_reduce_embeddings(emb, 64, save_dir="artifacts")
    """
    # ========================================================================
    # INPUT VALIDATION & SETUP
    # ========================================================================
    if not isinstance(embeddings, (torch.Tensor, np.ndarray)):
        embeddings = np.asarray(embeddings)
    if embeddings.ndim != 2:
        raise ValueError(
            f"`embeddings` must be 2-D (n_samples, n_features); "
            f"got shape {tuple(embeddings.shape)}."
        )
    n_samples, input_dim = embeddings.shape
    if n_samples < 2:
        raise ValueError(
            f"At least 2 samples are required to estimate a covariance; "
            f"got {n_samples}."
        )
    target_dim = int(target_dim)
    if not 1 <= target_dim <= input_dim:
        raise ValueError(
            f"`target_dim` must be in [1, {input_dim}] (input dim); got {target_dim}."
        )
    batch_size = max(1, int(batch_size))

    max_rank = min(n_samples - 1, input_dim)
    if target_dim > max_rank:
        warnings.warn(
            f"`target_dim` ({target_dim}) exceeds the maximum achievable rank "
            f"min(n_samples - 1, n_features) = {max_rank}; the surplus "
            f"components will capture (numerically) zero variance.",
            RuntimeWarning,
        )

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA was explicitly requested but is not available.")

    n_batches = len(range(0, n_samples, batch_size))
    timings: Dict[str, float] = {}

    if verbose:
        bar = "=" * 68
        print(bar)
        print("Streaming PCA - embedding reduction")
        print(bar)
        device_desc = str(device)
        if device.type == "cuda":
            device_desc += f" ({torch.cuda.get_device_name(device)})"
        print(f"Device:          {device_desc}")
        print(
            f"Input:           {n_samples:,} embeddings x {input_dim} dims "
            f"(dtype={embeddings.dtype}, type={type(embeddings).__name__})"
        )
        print(
            f"Target dim:      {target_dim} | batch size: {batch_size:,} "
            f"-> {n_batches:,} batch(es)"
        )
        print(f"Covariance:      exact online update = {exact_covariance}")
        if debug:
            print(
                f"[debug] torch {torch.__version__}, "
                f"CUDA available: {torch.cuda.is_available()}"
            )
            if device.type == "cuda":
                print(f"[debug] device capability: "
                      f"{torch.cuda.get_device_capability(device)}")

    # ========================================================================
    # FIRST PASS: STREAMING MEAN + COVARIANCE (float64)
    # ========================================================================
    global_mean = torch.zeros(input_dim, dtype=torch.float64, device=device)
    global_scatter = torch.zeros(
        (input_dim, input_dim), dtype=torch.float64, device=device
    )
    n_total = 0

    starts = range(0, n_samples, batch_size)
    pbar = (
        tqdm(starts, total=n_batches, desc="Computing mean & covariance",
             unit="batch")
        if use_tqdm
        else None
    )
    iterator = pbar if pbar is not None else starts
    debug_every = max(1, n_batches // 10)

    if debug and device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    if verbose:
        print("Pass 1: streaming mean & covariance computation (float64)...")

    _sync(device)
    t0 = time.perf_counter()

    with torch.no_grad():
        for batch_idx, start in enumerate(iterator):
            end = min(start + batch_size, n_samples)
            batch_np = embeddings[start:end]
            B = batch_np.shape[0]

            batch = _to_torch(batch_np, torch.float64, device)

            if debug and batch_idx == 0 and not bool(torch.isfinite(batch).all()):
                warnings.warn(
                    "The first batch contains non-finite (NaN/Inf) values; "
                    "the resulting PCA statistics will be NaN.",
                    RuntimeWarning,
                )

            # Batch mean
            batch_mean = batch.mean(dim=0)

            # Batch scatter (sum of squared deviations from the batch mean)
            batch_centered = batch - batch_mean
            batch_scatter = batch_centered.T @ batch_centered

            # Merge batch statistics into the running global statistics.
            delta = batch_mean - global_mean
            new_n_total = n_total + B

            global_scatter = global_scatter + batch_scatter
            if exact_covariance and n_total > 0:
                # Exact rank-1 correction (Chan et al., 1982) accounting for
                # the distance between the batch mean and the global mean.
                combine_weight = float(n_total * B) / float(new_n_total)
                global_scatter = global_scatter + torch.outer(delta, delta) * combine_weight

            # Online mean update
            global_mean = global_mean + delta * (B / new_n_total)
            n_total = new_n_total

            report = pbar is not None or (debug and batch_idx % debug_every == 0)
            if report:
                delta_norm = float(delta.norm())
                if pbar is not None:
                    pbar.set_postfix(
                        {"samples": f"{n_total:,}", "dmean": f"{delta_norm:.2e}"}
                    )
                if debug and batch_idx % debug_every == 0:
                    batch_var = float(batch_scatter.diagonal().sum()) / B
                    print(
                        f"  [debug] pass1 batch {batch_idx + 1}/{n_batches}: "
                        f"B={B:,}, |batch_mean|={float(batch_mean.norm()):.4f}, "
                        f"|dmean|={delta_norm:.3e}, batch_var={batch_var:.6f}"
                    )

    _sync(device)
    timings["pass1_mean_cov"] = time.perf_counter() - t0
    if pbar is not None:
        pbar.close()

    if verbose:
        elapsed = timings["pass1_mean_cov"]
        rate = n_samples / elapsed if elapsed > 0 else float("nan")
        print(f"Pass 1 done in {elapsed:.2f}s ({rate:,.0f} samples/s)")
        print(f"Total samples processed: {n_total:,}")
        print(f"Global mean L2 norm: {float(global_mean.norm()):.4f}")

    # Normalize covariance (unbiased estimator, denominator n - 1)
    global_cov = global_scatter / (n_total - 1)

    if debug:
        sym_err = float((global_cov - global_cov.T).abs().max())
        diag = torch.diagonal(global_cov)
        print(
            f"  [debug] covariance: symmetry max|C - C^T|={sym_err:.3e}, "
            f"diag range=[{float(diag.min()):.6e}, {float(diag.max()):.6e}], "
            f"trace={float(diag.sum()):.6e}"
        )
        if device.type == "cuda":
            print(
                f"  [debug] pass1 peak GPU memory: "
                f"{_format_bytes(torch.cuda.max_memory_allocated(device))}"
            )

    mean_np = global_mean.cpu().numpy()
    cov_np = global_cov.cpu().numpy()

    if save_dir is not None:
        _save_array(save_dir, MEAN_FILENAME, mean_np, verbose)
        _save_array(save_dir, COV_FILENAME, cov_np, verbose)

    # ========================================================================
    # PCA VIA EIGEN-DECOMPOSITION
    # ========================================================================
    if verbose:
        print("Performing eigen-decomposition...")

    _sync(device)
    t0 = time.perf_counter()
    n_negative = 0
    with torch.no_grad():
        eigvals, eigvecs = torch.linalg.eigh(global_cov)

        # Sort descending
        idx = torch.argsort(eigvals, descending=True)
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]

        if debug:
            n_negative = int((eigvals < 0).sum().item())

        # Clamp tiny negative eigenvalues (numerical noise; covariance is PSD)
        eigvals = torch.clamp_min(eigvals, 0.0)
    _sync(device)
    timings["eigendecomposition"] = time.perf_counter() - t0

    if debug:
        lam_min_top = torch.clamp_min(
            eigvals[target_dim - 1], torch.finfo(eigvals.dtype).eps
        )
        cond = float(eigvals[0] / lam_min_top)
        print(
            f"  [debug] eigen-decomposition done in "
            f"{timings['eigendecomposition']:.3f}s | "
            f"lambda_max={float(eigvals[0]):.6e}, "
            f"lambda_min={float(eigvals[-1]):.6e}, "
            f"negatives before clamp={n_negative}, "
            f"condition number (top {target_dim})={cond:.3e}"
        )

    total_variance = float(eigvals.sum())
    if not np.isfinite(total_variance) or total_variance <= 0.0:
        raise ValueError(
            "Total variance of the covariance matrix is zero or non-finite; "
            "cannot compute explained variance. Is the input constant, or "
            "does it contain NaN/Inf values?"
        )

    eigvals_np = eigvals.cpu().numpy()
    eigvecs_np = eigvecs.cpu().numpy()

    if save_dir is not None:
        _save_array(save_dir, EIGVALS_FILENAME, eigvals_np, verbose)
        _save_array(save_dir, EIGVECS_FILENAME, eigvecs_np, verbose)

    # ========================================================================
    # EXPLAINED VARIANCE (dynamic w.r.t. target_dim)
    # ========================================================================
    explained_ratio = eigvals / eigvals.sum()
    cumulative = torch.cumsum(explained_ratio, dim=0)

    if verbose:
        print("\nExplained variance for top components:")
        for k in _variance_checkpoints(target_dim, input_dim):
            print(f"  Top {k:>4d}: {cumulative[k - 1].item() * 100:6.2f}%")
        print(f"  Top {input_dim} (all): {cumulative[-1].item() * 100:6.2f}%")

        cum_cpu = cumulative.cpu()
        needed = []
        for pct in (0.90, 0.95, 0.99):
            k_needed = int(
                torch.searchsorted(
                    cum_cpu, torch.tensor(pct, dtype=cum_cpu.dtype)
                ).item()
            ) + 1
            needed.append(f"{min(k_needed, input_dim)} for {int(pct * 100)}%")
        print(
            f"  Components needed: {', '.join(needed)} "
            f"(requested: {target_dim})"
        )

    # Projection matrix
    W = eigvecs[:, :target_dim].to(torch.float32)
    W_np = W.cpu().numpy()
    if verbose:
        print(f"Projection matrix W: shape {tuple(W.shape)} (float32)")
    if save_dir is not None:
        _save_array(save_dir, PROJ_FILENAME, W_np, verbose)

    # ========================================================================
    # SECOND PASS: PROJECT EMBEDDINGS TO TARGET DIM (float32)
    # ========================================================================
    reduced = np.empty((n_samples, target_dim), dtype=np.float32)
    mean_fp32 = global_mean.to(torch.float32)

    starts = range(0, n_samples, batch_size)
    pbar = (
        tqdm(starts, total=n_batches, desc="Projecting embeddings", unit="batch")
        if use_tqdm
        else None
    )
    iterator = pbar if pbar is not None else starts

    if debug and device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    _sync(device)
    t0 = time.perf_counter()

    with torch.no_grad():
        for start in iterator:
            end = min(start + batch_size, n_samples)
            batch = _to_torch(embeddings[start:end], torch.float32, device)

            batch_centered = batch - mean_fp32
            batch_reduced = batch_centered @ W

            reduced[start:end] = batch_reduced.cpu().numpy()

    _sync(device)
    timings["pass2_projection"] = time.perf_counter() - t0
    if pbar is not None:
        pbar.close()

    if verbose:
        elapsed = timings["pass2_projection"]
        rate = n_samples / elapsed if elapsed > 0 else float("nan")
        print(f"Pass 2 done in {elapsed:.2f}s ({rate:,.0f} samples/s)")
        print(
            f"Reduced embeddings: shape={reduced.shape}, dtype=float32, "
            f"size={_format_bytes(reduced.nbytes)}"
        )
    if debug and device.type == "cuda":
        print(
            f"  [debug] pass2 peak GPU memory: "
            f"{_format_bytes(torch.cuda.max_memory_allocated(device))}"
        )

    if save_dir is not None:
        _save_array(save_dir, REDUCED_FILENAME, reduced, verbose)

    timings["total"] = (
        timings["pass1_mean_cov"]
        + timings["eigendecomposition"]
        + timings["pass2_projection"]
    )
    if verbose:
        print(f"Total PCA reduction time: {timings['total']:.2f}s")

    return PCAReductionResult(
        reduced=torch.from_numpy(reduced) if return_torch else reduced,
        mean=mean_np,
        covariance=cov_np,
        eigenvalues=eigvals_np,
        eigenvectors=eigvecs_np,
        projection_matrix=W_np,
        explained_variance_ratio=explained_ratio.cpu().numpy(),
        cumulative_explained_variance=cumulative.cpu().numpy(),
        n_samples=n_samples,
        input_dim=input_dim,
        target_dim=target_dim,
        device=str(device),
        timings=timings,
    )


# ---------------------------------------------------------------------------
# SMOKE TEST
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Synthetic data with an embedded 8-dim structure on top of noise.
    rng = np.random.default_rng(0)
    data = rng.normal(size=(50_000, 64)).astype(np.float32)
    data += (
        rng.normal(size=(50_000, 8)).astype(np.float32)
        @ rng.normal(size=(8, 64)).astype(np.float32)
    )
    res = pca_reduce_embeddings(data, target_dim=8, batch_size=10_000, debug=True)
    print("reduced:", res.reduced.shape, "| variance kept @8:",
          f"{res.cumulative_explained_variance[-1] * 100:.2f}%")