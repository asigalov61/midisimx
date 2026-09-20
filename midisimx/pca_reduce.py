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

**Reusable, saveable reductors.** The :class:`PCAReductor` class wraps the
pipeline: fit it once on a reference corpus, then project *unseen*
embeddings with :meth:`PCAReductor.transform` -- no refitting and no second
scan of the training data. A fitted reductor can be persisted to a single,
pickle-free ``.npz`` file via :meth:`PCAReductor.save` and restored with
:meth:`PCAReductor.load` or :func:`load_pca_reductor`::

    reductor = PCAReductor(target_dim=128).fit(train_embeddings)
    reductor.save("pca_reductor.npz")

    # ... later, possibly in another process:
    reductor = PCAReductor.load("pca_reductor.npz")
    reduced_unseen = reductor.transform(unseen_embeddings)

:func:`pca_reduce_embeddings` keeps its original behaviour; its result now
additionally carries the fitted reductor in ``PCAReductionResult.reductor``.

Optionally, all intermediate artifacts (mean, covariance, eigenvalues,
eigenvectors, projection matrix) and the reduced embeddings can be saved
to disk as ``.npy`` files by passing ``save_dir`` (the reloadable model
file ``pca_reductor.npz`` is written there as well).

This module has no import-time side effects; all work happens inside
:meth:`PCAReductor.fit`, :meth:`PCAReductor.transform` and
:func:`pca_reduce_embeddings`.
"""

from __future__ import annotations

import json
import os
import time
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Union

import numpy as np
import torch
from tqdm import tqdm

__all__ = [
    "pca_reduce_embeddings",
    "PCAReductionResult",
    "PCAReductor",
    "load_pca_reductor",
]

# ---------------------------------------------------------------------------
# Default artifact file names used when ``save_dir`` is passed to
# :func:`pca_reduce_embeddings` / :meth:`PCAReductor.fit`.
# ---------------------------------------------------------------------------
MEAN_FILENAME = "pca_mean.npy"
COV_FILENAME = "pca_cov.npy"
EIGVALS_FILENAME = "pca_eigvals.npy"
EIGVECS_FILENAME = "pca_eigvecs.npy"
PROJ_FILENAME = "pca_projection_matrix.npy"
REDUCED_FILENAME = "embeddings_reduced.npy"

# Combined, reloadable model file written by :meth:`PCAReductor.save`.
MODEL_FILENAME = "pca_reductor.npz"
MODEL_FORMAT_VERSION = 1

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
    reductor : PCAReductor, optional
        The fitted reductor that produced this result. Use it to project
        unseen embeddings without refitting (``reductor.transform``) and
        to persist/reload the model (``reductor.save`` /
        ``PCAReductor.load``). ``None`` only if the result was constructed
        manually.
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
    reductor: Optional["PCAReductor"] = None


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
# FITTED REDUCTOR: fit once, transform unseen data, save / load
# ---------------------------------------------------------------------------
class PCAReductor:
    """A fitted, serializable streaming-PCA dimensionality reductor.

    Fit once on a reference corpus, then reduce *unseen* embeddings with
    the very same principal components -- without refitting:

    - :meth:`fit` -- stream over a (possibly huge) embedding matrix and
      compute mean, covariance and eigen-decomposition; returns ``self``.
    - :meth:`transform` -- project unseen embeddings batch-streamed with
      the fitted statistics (numerically identical to pass 2 of
      :func:`pca_reduce_embeddings`).
    - :meth:`save` / :meth:`load` -- persist / restore the fitted state as
      a single ``.npz`` file (plain arrays + JSON metadata; no pickling).
    - :meth:`inverse_transform` -- approximate mapping back to the original
      space (valid because the projection matrix is orthonormal).
    - :meth:`fit_transform` -- convenience: fit and project the same data.

    Attributes
    ----------
    target_dim : int
        Number of principal components kept.
    input_dim : int or None
        Dimensionality of the fitted embeddings (None until fitted).
    n_samples : int or None
        Number of samples used for fitting (None until fitted).
    mean : numpy.ndarray or None
        Global mean, shape ``(input_dim,)``, float64.
    covariance : numpy.ndarray or None
        Full covariance matrix, shape ``(input_dim, input_dim)``, float64
        (may be None for models saved with ``include_covariance=False``).
    eigenvalues : numpy.ndarray or None
        Eigenvalues (descending, clamped at zero), shape ``(input_dim,)``.
    eigenvectors : numpy.ndarray or None
        Eigenvectors as columns, shape ``(input_dim, input_dim)``.
    projection_matrix : numpy.ndarray or None
        Top-``target_dim`` eigenvectors, shape
        ``(input_dim, target_dim)``, float32.
    explained_variance_ratio, cumulative_explained_variance : numpy.ndarray or None
        Per-component and cumulative explained-variance fractions.
    exact_covariance : bool or None
        Whether the exact online covariance update was used for fitting.
    fit_device : str or None
        Device string used during fitting.
    fit_timings, last_transform_timings : dict
        Wall-clock durations (seconds) of the fit phases and of the most
        recent :meth:`transform` call.
    metadata : dict
        Provenance information (populated when loading a saved model).

    Examples
    --------
    >>> reductor = PCAReductor(target_dim=64).fit(train_embeddings)
    >>> reduced_new = reductor.transform(unseen_embeddings)   # no refit
    >>> reductor.save("pca_reductor.npz")
    >>> reductor2 = PCAReductor.load("pca_reductor.npz")
    >>> np.allclose(reductor2.transform(unseen_embeddings), reduced_new)
    True
    """

    def __init__(self, target_dim: int = 128) -> None:
        target_dim = int(target_dim)
        if target_dim < 1:
            raise ValueError(f"`target_dim` must be >= 1; got {target_dim}.")
        self.target_dim = target_dim

        # ---- fitted state (None / empty until fit() or load()) ----
        self.mean: Optional[np.ndarray] = None
        self.covariance: Optional[np.ndarray] = None
        self.eigenvalues: Optional[np.ndarray] = None
        self.eigenvectors: Optional[np.ndarray] = None
        self.projection_matrix: Optional[np.ndarray] = None
        self.explained_variance_ratio: Optional[np.ndarray] = None
        self.cumulative_explained_variance: Optional[np.ndarray] = None
        self.input_dim: Optional[int] = None
        self.n_samples: Optional[int] = None
        self.exact_covariance: Optional[bool] = None
        self.fit_device: Optional[str] = None
        self.fit_timings: Dict[str, float] = {}
        self.last_transform_timings: Dict[str, float] = {}
        self.metadata: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # STATE HELPERS
    # ------------------------------------------------------------------
    @property
    def is_fitted(self) -> bool:
        """True if the reductor holds fitted statistics."""
        return self.mean is not None and self.projection_matrix is not None

    def _check_is_fitted(self) -> None:
        if not self.is_fitted:
            raise RuntimeError(
                "This PCAReductor is not fitted yet. Call `fit(...)` on a "
                "reference corpus, or `PCAReductor.load(path)` to restore a "
                "saved model."
            )

    @staticmethod
    def _resolve_device(device: Optional[Union[str, torch.device]]) -> torch.device:
        if device is None:
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        device = torch.device(device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA was explicitly requested but is not available.")
        return device

    def _validate_state(self) -> None:
        """Sanity-check internal array shapes (used after loading)."""
        input_dim, target_dim = self.input_dim, self.target_dim
        if self.mean is None or self.mean.shape != (input_dim,):
            raise ValueError(
                f"`mean` must have shape ({input_dim},); "
                f"got {None if self.mean is None else self.mean.shape}."
            )
        expected_w = (input_dim, target_dim)
        if self.projection_matrix is None or self.projection_matrix.shape != expected_w:
            raise ValueError(
                f"`projection_matrix` must have shape {expected_w}; got "
                f"{None if self.projection_matrix is None else self.projection_matrix.shape}."
            )
        if self.eigenvalues is None or self.eigenvalues.shape != (input_dim,):
            raise ValueError(
                f"`eigenvalues` must have shape ({input_dim},); got "
                f"{None if self.eigenvalues is None else self.eigenvalues.shape}."
            )
        if self.eigenvectors is None or self.eigenvectors.shape != (input_dim, input_dim):
            raise ValueError(
                f"`eigenvectors` must have shape {(input_dim, input_dim)}; got "
                f"{None if self.eigenvectors is None else self.eigenvectors.shape}."
            )
        if self.covariance is not None and self.covariance.shape != (input_dim, input_dim):
            raise ValueError(
                f"`covariance` must have shape {(input_dim, input_dim)}; "
                f"got {self.covariance.shape}."
            )

    def __repr__(self) -> str:
        if self.is_fitted:
            return (
                f"PCAReductor(fitted=True, input_dim={self.input_dim}, "
                f"target_dim={self.target_dim}, n_samples={self.n_samples:,})"
            )
        return f"PCAReductor(fitted=False, target_dim={self.target_dim})"

    def __call__(self, embeddings: ArrayLike, **kwargs):
        """Shorthand for :meth:`transform`."""
        return self.transform(embeddings, **kwargs)

    # ------------------------------------------------------------------
    # FIT
    # ------------------------------------------------------------------
    def fit(
        self,
        embeddings: ArrayLike,
        batch_size: int = 512_000,
        *,
        device: Optional[Union[str, torch.device]] = None,
        exact_covariance: bool = True,
        use_tqdm: bool = True,
        verbose: bool = True,
        debug: bool = False,
        save_dir: Optional[Union[str, os.PathLike]] = None,
    ) -> "PCAReductor":
        """Fit the reductor on a reference corpus (pass 1 + eigen-decomposition).

        Streams over ``embeddings`` in batches (float64) to accumulate the
        global mean and covariance, then diagonalizes the covariance and
        keeps the top ``self.target_dim`` eigenvectors. All statistics are
        stored on the instance so that :meth:`transform` can afterwards
        project unseen data without any refitting.

        Parameters
        ----------
        embeddings : array_like or torch.Tensor, shape (n_samples, n_features)
            Reference corpus. Only batches are copied to the compute device,
            so the input may be larger than device memory.
        batch_size : int, default=512_000
            Rows processed per batch (pass-1 footprint is roughly
            ``batch_size * n_features * 8`` bytes on the device).
        device : str, torch.device or None, default=None
            Compute device; None auto-selects CUDA when available.
        exact_covariance : bool, default=True
            Use the exact online covariance update (Chan et al., 1982).
            Set False to reproduce the batch-mean-centered approximation.
        use_tqdm, verbose, debug : bool
            Progress bar, progress statistics and extra diagnostics --
            identical in spirit to :func:`pca_reduce_embeddings`.
        save_dir : str, os.PathLike or None, default=None
            If given, the artifacts ``pca_mean.npy``, ``pca_cov.npy``,
            ``pca_eigvals.npy``, ``pca_eigvecs.npy``,
            ``pca_projection_matrix.npy`` and the reloadable model file
            ``pca_reductor.npz`` are written to this directory.

        Returns
        -------
        PCAReductor
            ``self``, ready for :meth:`transform`.

        Raises
        ------
        ValueError
            If ``embeddings`` is not 2-D, has fewer than two rows, if
            ``target_dim`` is outside ``[1, n_features]``, if CUDA is
            explicitly requested but unavailable, or if the covariance has
            zero or non-finite total variance.
        RuntimeWarning
            If ``target_dim`` exceeds the achievable rank, or (in debug
            mode) the first batch contains non-finite values.
        """
        # ====================================================================
        # INPUT VALIDATION & SETUP
        # ====================================================================
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
        target_dim = self.target_dim
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

        device = self._resolve_device(device)

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

        # ====================================================================
        # FIRST PASS: STREAMING MEAN + COVARIANCE (float64)
        # ====================================================================
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
                    global_scatter = (
                        global_scatter
                        + torch.outer(delta, delta) * combine_weight
                    )

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

        # ====================================================================
        # PCA VIA EIGEN-DECOMPOSITION
        # ====================================================================
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

        # ====================================================================
        # EXPLAINED VARIANCE (dynamic w.r.t. target_dim)
        # ====================================================================
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

        # ====================================================================
        # STORE FITTED STATE
        # ====================================================================
        self.mean = mean_np
        self.covariance = cov_np
        self.eigenvalues = eigvals_np
        self.eigenvectors = eigvecs_np
        self.projection_matrix = W_np
        self.explained_variance_ratio = explained_ratio.cpu().numpy()
        self.cumulative_explained_variance = cumulative.cpu().numpy()
        self.input_dim = int(input_dim)
        self.n_samples = int(n_samples)
        self.exact_covariance = bool(exact_covariance)
        self.fit_device = str(device)
        self.fit_timings = {
            "pass1_mean_cov": timings["pass1_mean_cov"],
            "eigendecomposition": timings["eigendecomposition"],
            "fit_total": timings["pass1_mean_cov"] + timings["eigendecomposition"],
        }
        self.metadata = {}  # provenance is (re)built on save()

        if save_dir is not None:
            _save_array(save_dir, MEAN_FILENAME, self.mean, verbose)
            _save_array(save_dir, COV_FILENAME, self.covariance, verbose)
            _save_array(save_dir, EIGVALS_FILENAME, self.eigenvalues, verbose)
            _save_array(save_dir, EIGVECS_FILENAME, self.eigenvectors, verbose)
            _save_array(save_dir, PROJ_FILENAME, self.projection_matrix, verbose)
            self.save(os.path.join(save_dir, MODEL_FILENAME), verbose=verbose)

        return self

    # ------------------------------------------------------------------
    # TRANSFORM (project unseen embeddings with the fitted statistics)
    # ------------------------------------------------------------------
    def transform(
        self,
        embeddings: ArrayLike,
        batch_size: int = 512_000,
        *,
        device: Optional[Union[str, torch.device]] = None,
        use_tqdm: bool = True,
        verbose: bool = False,
        debug: bool = False,
        return_torch: bool = False,
    ) -> Union[np.ndarray, torch.Tensor]:
        """Project unseen embeddings with the fitted PCA (no refitting).

        Every batch is centered with the fitted mean and multiplied with
        the fitted projection matrix in float32 -- numerically identical to
        pass 2 of :func:`pca_reduce_embeddings`, and streamed so unseen
        data larger than device memory is handled as well.

        Parameters
        ----------
        embeddings : array_like or torch.Tensor, shape (n_samples, input_dim)
            Unseen embeddings. Must have the same number of features as the
            data the reductor was fitted on. A 1-D array of length
            ``input_dim`` is treated as a single embedding and a 1-D reduced
            vector is returned.
        batch_size : int, default=512_000
            Rows processed per batch.
        device : str, torch.device or None, default=None
            Compute device; None auto-selects CUDA when available.
        use_tqdm : bool, default=True
            Show a tqdm progress bar.
        verbose : bool, default=False
            Print a short summary (device, shapes, timing, output size).
            Quiet by default because transform is typically called many
            times on unseen shards.
        debug : bool, default=False
            Print peak GPU memory usage (CUDA only).
        return_torch : bool, default=False
            If True, return a float32 ``torch.Tensor`` (CPU) instead of a
            ``numpy.ndarray``.

        Returns
        -------
        numpy.ndarray or torch.Tensor
            Projected embeddings, shape ``(n_samples, target_dim)``
            (or ``(target_dim,)`` for 1-D input), dtype float32.

        Raises
        ------
        RuntimeError
            If the reductor has not been fitted (or loaded) yet.
        ValueError
            If the embeddings are not 1-D/2-D or their feature dimension
            does not match the fitted ``input_dim``.
        """
        reduced = self._project_streaming(
            embeddings,
            batch_size=batch_size,
            device=device,
            use_tqdm=use_tqdm,
            verbose=verbose,
            debug=debug,
        )
        return torch.from_numpy(reduced) if return_torch else reduced

    def _project_streaming(
        self,
        embeddings: ArrayLike,
        batch_size: int,
        device: Optional[Union[str, torch.device]],
        use_tqdm: bool,
        verbose: bool,
        debug: bool,
    ) -> np.ndarray:
        """Batch-streamed projection; returns a float32 numpy array."""
        self._check_is_fitted()

        if not isinstance(embeddings, (torch.Tensor, np.ndarray)):
            embeddings = np.asarray(embeddings)
        single = False
        if embeddings.ndim == 1:
            single = True
            embeddings = embeddings.reshape(1, -1)
        if embeddings.ndim != 2:
            raise ValueError(
                f"`embeddings` must be 1-D or 2-D (n_samples, n_features); "
                f"got shape {tuple(embeddings.shape)}."
            )
        if embeddings.shape[1] != self.input_dim:
            raise ValueError(
                f"Embeddings have {embeddings.shape[1]} features, but the "
                f"reductor was fitted on {self.input_dim} features."
            )
        n_samples = embeddings.shape[0]
        if n_samples == 0:
            raise ValueError("`embeddings` contains no samples.")
        batch_size = max(1, int(batch_size))
        device = self._resolve_device(device)
        target_dim = self.target_dim
        n_batches = len(range(0, n_samples, batch_size))

        if verbose:
            print("\nStreaming projection - PCAReductor.transform")
            print(f"Device:     {device}")
            print(
                f"Input:      {n_samples:,} embeddings x {self.input_dim} dims "
                f"-> {target_dim} dims "
                f"(dtype={embeddings.dtype}, type={type(embeddings).__name__})"
            )
            print(f"Batches:    {n_batches:,} x <= {batch_size:,} samples")

        reduced = np.empty((n_samples, target_dim), dtype=np.float32)
        mean_fp32 = torch.as_tensor(self.mean, dtype=torch.float32, device=device)
        W = torch.as_tensor(self.projection_matrix, dtype=torch.float32, device=device)

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

                batch_reduced = (batch - mean_fp32) @ W
                reduced[start:end] = batch_reduced.cpu().numpy()

        _sync(device)
        elapsed = time.perf_counter() - t0
        if pbar is not None:
            pbar.close()

        self.last_transform_timings = {"transform": elapsed}

        if verbose:
            rate = n_samples / elapsed if elapsed > 0 else float("nan")
            print(f"Projection done in {elapsed:.2f}s ({rate:,.0f} samples/s)")
            print(
                f"Reduced embeddings: shape={reduced.shape}, dtype=float32, "
                f"size={_format_bytes(reduced.nbytes)}"
            )
        if debug and device.type == "cuda":
            print(
                f"  [debug] transform peak GPU memory: "
                f"{_format_bytes(torch.cuda.max_memory_allocated(device))}"
            )

        return reduced[0] if single else reduced

    # ------------------------------------------------------------------
    # CONVENIENCE
    # ------------------------------------------------------------------
    def fit_transform(
        self,
        embeddings: ArrayLike,
        batch_size: int = 512_000,
        *,
        device: Optional[Union[str, torch.device]] = None,
        exact_covariance: bool = True,
        use_tqdm: bool = True,
        verbose: bool = True,
        debug: bool = False,
        save_dir: Optional[Union[str, os.PathLike]] = None,
        return_torch: bool = False,
    ) -> Union[np.ndarray, torch.Tensor]:
        """Fit on ``embeddings`` and return the projected training data.

        Convenience combining :meth:`fit` and :meth:`transform` in one call
        (what :func:`pca_reduce_embeddings` does, minus the
        ``PCAReductionResult`` container). If ``save_dir`` is given, the
        model artifacts and the reloadable ``.npz`` model file are saved,
        but *not* the reduced training embeddings.
        """
        self.fit(
            embeddings,
            batch_size=batch_size,
            device=device,
            exact_covariance=exact_covariance,
            use_tqdm=use_tqdm,
            verbose=verbose,
            debug=debug,
            save_dir=save_dir,
        )
        return self.transform(
            embeddings,
            batch_size=batch_size,
            device=device,
            use_tqdm=use_tqdm,
            verbose=verbose,
            debug=debug,
            return_torch=return_torch,
        )

    def inverse_transform(
        self,
        reduced: ArrayLike,
        batch_size: int = 512_000,
        *,
        device: Optional[Union[str, torch.device]] = None,
        use_tqdm: bool = False,
        verbose: bool = False,
        return_torch: bool = False,
    ) -> Union[np.ndarray, torch.Tensor]:
        """Approximately map reduced embeddings back to the original space.

        Because the columns of ``projection_matrix`` are orthonormal, the
        least-squares optimal reconstruction is
        ``x_hat = (x_reduced @ W.T) + mean``. Variance along the discarded
        components is lost, so this is only an approximation of the original
        embeddings. Output dtype is float32.

        Parameters
        ----------
        reduced : array_like, shape (n_samples, target_dim) or (target_dim,)
            Reduced embeddings produced by :meth:`transform` (or
            :func:`pca_reduce_embeddings`).
        batch_size : int, default=512_000
            Rows processed per batch.
        device, use_tqdm, verbose, return_torch
            See :meth:`transform`.

        Returns
        -------
        numpy.ndarray or torch.Tensor
            Reconstructed embeddings, shape ``(n_samples, input_dim)``
            (or ``(input_dim,)`` for 1-D input), dtype float32.
        """
        self._check_is_fitted()

        if not isinstance(reduced, (torch.Tensor, np.ndarray)):
            reduced = np.asarray(reduced)
        single = False
        if reduced.ndim == 1:
            single = True
            reduced = reduced.reshape(1, -1)
        if reduced.ndim != 2:
            raise ValueError(
                f"`reduced` must be 1-D or 2-D; got shape {tuple(reduced.shape)}."
            )
        if reduced.shape[1] != self.target_dim:
            raise ValueError(
                f"`reduced` has {reduced.shape[1]} dims, but the reductor "
                f"projects to {self.target_dim} dims."
            )
        n_samples = reduced.shape[0]
        if n_samples == 0:
            raise ValueError("`reduced` contains no samples.")
        batch_size = max(1, int(batch_size))
        device = self._resolve_device(device)

        W_t = torch.as_tensor(
            np.ascontiguousarray(self.projection_matrix.T),
            dtype=torch.float32,
            device=device,
        )
        mean_fp32 = torch.as_tensor(self.mean, dtype=torch.float32, device=device)

        reconstructed = np.empty((n_samples, self.input_dim), dtype=np.float32)
        starts = range(0, n_samples, batch_size)
        pbar = (
            tqdm(starts, total=len(starts), desc="Reconstructing embeddings",
                 unit="batch")
            if use_tqdm
            else None
        )
        iterator = pbar if pbar is not None else starts

        with torch.no_grad():
            for start in iterator:
                end = min(start + batch_size, n_samples)
                batch = _to_torch(reduced[start:end], torch.float32, device)
                reconstructed[start:end] = (batch @ W_t + mean_fp32).cpu().numpy()
        if pbar is not None:
            pbar.close()

        if verbose:
            print(
                f"Reconstructed {n_samples:,} embeddings -> "
                f"shape={reconstructed.shape}, dtype=float32"
            )

        out = reconstructed[0] if single else reconstructed
        return torch.from_numpy(out) if return_torch else out

    # ------------------------------------------------------------------
    # SAVE / LOAD
    # ------------------------------------------------------------------
    def save(
        self,
        filepath: Union[str, os.PathLike],
        *,
        include_covariance: bool = True,
        compress: bool = True,
        verbose: bool = True,
    ) -> str:
        """Persist the fitted reductor to a single ``.npz`` model file.

        The file contains plain NumPy arrays plus a JSON metadata string --
        no pickling -- so it is safe to transport and can be loaded with
        :meth:`load` across machines, processes and Python sessions.

        Parameters
        ----------
        filepath : str or os.PathLike
            Destination path. If it does not end with ``.npz``, the
            extension is appended (NumPy convention). If an existing
            directory is given, ``pca_reductor.npz`` is written inside it.
        include_covariance : bool, default=True
            Also store the full ``(input_dim, input_dim)`` float64
            covariance matrix. It is not needed for :meth:`transform`; set
            to False for very high-dimensional models to keep the file
            small.
        compress : bool, default=True
            Use ``np.savez_compressed`` instead of ``np.savez``.
        verbose : bool, default=True
            Print the destination path and file size.

        Returns
        -------
        str
            The actual path the model was written to.
        """
        self._check_is_fitted()

        filepath = os.fspath(filepath)
        if os.path.isdir(filepath):
            filepath = os.path.join(filepath, MODEL_FILENAME)
        directory = os.path.dirname(filepath)
        if directory:
            os.makedirs(directory, exist_ok=True)

        metadata = {
            "format_version": MODEL_FORMAT_VERSION,
            "model_class": type(self).__name__,
            "created": time.strftime("%Y-%m-%d %H:%M:%S"),
            "n_samples": int(self.n_samples),
            "input_dim": int(self.input_dim),
            "target_dim": int(self.target_dim),
            "exact_covariance": (
                bool(self.exact_covariance)
                if self.exact_covariance is not None
                else None
            ),
            "fit_device": self.fit_device,
            "fit_timings": dict(self.fit_timings),
            "torch_version": torch.__version__,
            "numpy_version": np.__version__,
        }

        payload: Dict[str, Any] = {
            "format_version": np.int64(MODEL_FORMAT_VERSION),
            "target_dim": np.int64(self.target_dim),
            "input_dim": np.int64(self.input_dim),
            "n_samples": np.int64(self.n_samples),
            "mean": self.mean,
            "projection_matrix": self.projection_matrix,
            "eigenvalues": self.eigenvalues,
            "eigenvectors": self.eigenvectors,
            "explained_variance_ratio": self.explained_variance_ratio,
            "cumulative_explained_variance": self.cumulative_explained_variance,
            "exact_covariance": np.bool_(bool(self.exact_covariance)),
            "metadata": np.array(json.dumps(metadata, sort_keys=True, default=str)),
        }
        if include_covariance and self.covariance is not None:
            payload["covariance"] = self.covariance

        saver = np.savez_compressed if compress else np.savez
        saver(filepath, **payload)
        if not filepath.endswith(".npz"):
            filepath += ".npz"

        if verbose:
            size = os.path.getsize(filepath)
            print(
                f"Saved fitted PCAReductor "
                f"(input_dim={self.input_dim}, target_dim={self.target_dim}, "
                f"n_samples={self.n_samples:,}) -> {filepath} "
                f"({_format_bytes(size)})"
            )
        return filepath

    @classmethod
    def load(
        cls,
        filepath: Union[str, os.PathLike],
        *,
        validate: bool = True,
    ) -> "PCAReductor":
        """Load a reductor previously written by :meth:`save`.

        Parameters
        ----------
        filepath : str or os.PathLike
            Path to the ``.npz`` model file. If an existing directory is
            given, ``pca_reductor.npz`` inside it is loaded; if the exact
            path does not exist, ``.npz`` is appended (mirroring
            ``np.savez``).
        validate : bool, default=True
            Verify that the internal array shapes are mutually consistent.

        Returns
        -------
        PCAReductor
            A fitted reductor, ready for :meth:`transform`.

        Raises
        ------
        FileNotFoundError
            If no model file exists at ``filepath``.
        ValueError
            If the file is not a valid PCAReductor model file, was written
            by a newer format version, or has internally inconsistent shapes.
        """
        filepath = os.fspath(filepath)
        if os.path.isdir(filepath):
            filepath = os.path.join(filepath, MODEL_FILENAME)
        if not os.path.isfile(filepath) and not filepath.endswith(".npz"):
            candidate = filepath + ".npz"
            if os.path.isfile(candidate):
                filepath = candidate
        if not os.path.isfile(filepath):
            raise FileNotFoundError(
                f"No PCAReductor model file found at: {filepath}"
            )

        loaded = np.load(filepath, allow_pickle=False)
        if not isinstance(loaded, np.lib.npyio.NpzFile):
            raise ValueError(
                f"{filepath} is not a PCAReductor model file "
                f"(expected a .npz archive written by PCAReductor.save)."
            )

        with loaded as data:
            files = set(data.files)
            required = {
                "mean",
                "projection_matrix",
                "eigenvalues",
                "eigenvectors",
                "target_dim",
                "input_dim",
            }
            missing = sorted(required - files)
            if missing:
                raise ValueError(
                    f"{filepath} is not a valid PCAReductor file; "
                    f"missing arrays: {missing}"
                )

            version = (
                int(data["format_version"]) if "format_version" in files else 1
            )
            if version > MODEL_FORMAT_VERSION:
                raise ValueError(
                    f"Model file format version {version} is newer than the "
                    f"version supported by this module "
                    f"({MODEL_FORMAT_VERSION}); please upgrade `pca_reduce`."
                )

            obj = cls(target_dim=int(data["target_dim"]))
            obj.input_dim = int(data["input_dim"])
            obj.n_samples = (
                int(data["n_samples"]) if "n_samples" in files else None
            )
            obj.mean = np.ascontiguousarray(data["mean"], dtype=np.float64)
            obj.projection_matrix = np.ascontiguousarray(
                data["projection_matrix"], dtype=np.float32
            )
            obj.eigenvalues = np.ascontiguousarray(
                data["eigenvalues"], dtype=np.float64
            )
            obj.eigenvectors = np.ascontiguousarray(
                data["eigenvectors"], dtype=np.float64
            )
            if "covariance" in files:
                obj.covariance = np.ascontiguousarray(
                    data["covariance"], dtype=np.float64
                )
            if "explained_variance_ratio" in files:
                obj.explained_variance_ratio = np.ascontiguousarray(
                    data["explained_variance_ratio"], dtype=np.float64
                )
            if "cumulative_explained_variance" in files:
                obj.cumulative_explained_variance = np.ascontiguousarray(
                    data["cumulative_explained_variance"], dtype=np.float64
                )
            if "exact_covariance" in files:
                obj.exact_covariance = bool(data["exact_covariance"])
            if "metadata" in files:
                try:
                    parsed = json.loads(str(data["metadata"]))
                    obj.metadata = parsed if isinstance(parsed, dict) else {}
                except (json.JSONDecodeError, TypeError, ValueError):
                    obj.metadata = {"raw_metadata": str(data["metadata"])}

        # Restore optional provenance from the metadata blob.
        if isinstance(obj.metadata, dict):
            if obj.fit_device is None:
                obj.fit_device = obj.metadata.get("fit_device")
            ft = obj.metadata.get("fit_timings")
            if not obj.fit_timings and isinstance(ft, dict):
                obj.fit_timings = {
                    k: float(v)
                    for k, v in ft.items()
                    if isinstance(v, (int, float))
                }
            if obj.exact_covariance is None:
                ec = obj.metadata.get("exact_covariance")
                if ec is not None:
                    obj.exact_covariance = bool(ec)

        if validate:
            obj._validate_state()
        return obj


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

    The fitted reductor is attached to the returned result
    (``result.reductor``), so unseen embeddings can be reduced with the
    same PCA -- without refitting -- via
    ``result.reductor.transform(unseen)``, and persisted via
    ``result.reductor.save(path)`` / :meth:`PCAReductor.load`.

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
        ``pca_eigvals.npy``, ``pca_eigvecs.npy``,
        ``pca_projection_matrix.npy`` and ``embeddings_reduced.npy``, plus
        the reloadable fitted model ``pca_reductor.npz`` (load it later
        with :meth:`PCAReductor.load`). If None (default), nothing is
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
        ratios, dataset dimensions, device string, phase timings and the
        fitted :class:`PCAReductor` (``result.reductor``). See
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
    * Use ``result.reductor`` to project unseen embeddings
      (``reductor.transform``) or to persist/reload the fitted PCA
      (``reductor.save`` / ``PCAReductor.load``).

    Examples
    --------
    >>> from pca_reduce import pca_reduce_embeddings
    >>> result = pca_reduce_embeddings(train_emb, target_dim=64, batch_size=500_000)
    >>> result.reduced.shape
    (12000000, 64)
    >>> result.cumulative_explained_variance[63]   # variance kept by 64 comps
    0.987...
    >>> result = pca_reduce_embeddings(emb, 64, save_dir="artifacts")
    >>> new_reduced = result.reductor.transform(unseen_emb)   # no refit
    >>> result.reductor.save("pca_reductor.npz")
    """
    reductor = PCAReductor(target_dim=target_dim)
    reductor.fit(
        embeddings,
        batch_size=batch_size,
        device=device,
        exact_covariance=exact_covariance,
        use_tqdm=use_tqdm,
        verbose=verbose,
        debug=debug,
        save_dir=save_dir,
    )

    # Second pass: project (in this case, the training data itself).
    reduced = reductor.transform(
        embeddings,
        batch_size=batch_size,
        device=device,
        use_tqdm=use_tqdm,
        verbose=verbose,
        debug=debug,
    )

    if save_dir is not None:
        _save_array(save_dir, REDUCED_FILENAME, reduced, verbose)

    timings: Dict[str, float] = {
        "pass1_mean_cov": reductor.fit_timings["pass1_mean_cov"],
        "eigendecomposition": reductor.fit_timings["eigendecomposition"],
        "pass2_projection": reductor.last_transform_timings["transform"],
    }
    timings["total"] = (
        timings["pass1_mean_cov"]
        + timings["eigendecomposition"]
        + timings["pass2_projection"]
    )
    if verbose:
        print(f"Total PCA reduction time: {timings['total']:.2f}s")

    return PCAReductionResult(
        reduced=torch.from_numpy(reduced) if return_torch else reduced,
        mean=reductor.mean,
        covariance=reductor.covariance,
        eigenvalues=reductor.eigenvalues,
        eigenvectors=reductor.eigenvectors,
        projection_matrix=reductor.projection_matrix,
        explained_variance_ratio=reductor.explained_variance_ratio,
        cumulative_explained_variance=reductor.cumulative_explained_variance,
        n_samples=reductor.n_samples,
        input_dim=reductor.input_dim,
        target_dim=reductor.target_dim,
        device=reductor.fit_device,
        timings=timings,
        reductor=reductor,
    )


# ---------------------------------------------------------------------------
# CONVENIENCE LOADER
# ---------------------------------------------------------------------------
def load_pca_reductor(filepath: Union[str, os.PathLike]) -> PCAReductor:
    """Load a fitted :class:`PCAReductor` from a ``.npz`` model file.

    Convenience wrapper around :meth:`PCAReductor.load`.

    Parameters
    ----------
    filepath : str or os.PathLike
        Path to the model file, or to a directory containing
        ``pca_reductor.npz``.

    Returns
    -------
    PCAReductor
        The fitted reductor, ready for :meth:`PCAReductor.transform`.

    Examples
    --------
    >>> reductor = load_pca_reductor("artifacts/pca_reductor.npz")
    >>> reduced = reductor.transform(unseen_embeddings)
    """
    return PCAReductor.load(filepath)


# ---------------------------------------------------------------------------
# SMOKE TEST
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import tempfile

    # Synthetic data with an embedded 8-dim structure on top of noise.
    rng = np.random.default_rng(0)
    structure = rng.normal(size=(8, 64)).astype(np.float32)

    def make_embeddings(n: int) -> np.ndarray:
        data = rng.normal(size=(n, 64)).astype(np.float32)
        data += rng.normal(size=(n, 8)).astype(np.float32) @ structure
        return data

    train = make_embeddings(50_000)
    res = pca_reduce_embeddings(train, target_dim=8, batch_size=10_000, debug=True)
    print("reduced:", res.reduced.shape, "| variance kept @8:",
          f"{res.cumulative_explained_variance[-1] * 100:.2f}%")

    # ---- save / load round trip; project unseen embeddings without refit ----
    unseen = make_embeddings(2_000)
    with tempfile.TemporaryDirectory() as tmp:
        model_path = res.reductor.save(os.path.join(tmp, "demo_reductor.npz"))
        loaded = load_pca_reductor(model_path)
        print("loaded:", loaded)

        red_unseen = loaded.transform(unseen, batch_size=4_000)
        print("unseen reduced:", red_unseen.shape, red_unseen.dtype,
              "| finite:", bool(np.isfinite(red_unseen).all()))

        # Transforming the training data with the *loaded* reductor must
        # reproduce the original reduction.
        red_train = loaded.transform(train, batch_size=10_000,
                                     use_tqdm=False, verbose=False)
        assert np.allclose(red_train, res.reduced, rtol=1e-4, atol=1e-4), (
            "loaded reductor does not reproduce the fitted projection"
        )
        print("round-trip check OK: loaded reductor reproduces the "
              "training projection")