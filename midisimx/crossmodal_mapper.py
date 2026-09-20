#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
crossmodal_mapper.py
====================

A **non-ML/DL, bi-directional cross-modal embedding mapper** built purely on
closed-form linear algebra (NumPy / LAPACK: ``eigh``, ``svd``, ``solve`` —
no gradient descent, no epochs, no iterative "training").

Given two *row-aligned* embedding matrices from different domains/modalities

    SRC : (n, d_src)      e.g. text encoder outputs
    TRG : (n, d_trg)      e.g. image encoder outputs

where row ``i`` of SRC corresponds to row ``i`` of TRG, the default method
(``procrustes``) computes, in one shot:

    1. (optional) L2 length-normalisation of rows           (scale invariance)
    2. (optional) mean-centering per side
    3. (optional) ZCA whitening per side:
           S = Xc'Xc = E diag(lambda) E'                    (eigendecomposition)
           W_src  = E diag(max(lambda, c*lambda_max)^-1/2) E'
           Xw = Xc W_src      =>  Xw'Xw = I                 (unit scatter)
    4. SVD of the cross-scatter  M = Xw' Yw = U Sigma V'
    5. cores (rho = reweight power, default 0 = orthogonal Procrustes):
           W_fwd = U_r diag(sigma^rho) V_r'      (d_src x d_trg)
           W_bwd = V_r diag(sigma^rho) U_r'      (d_trg x d_src)
       (for square dims & rho=0:  W_bwd = W_fwd'  exactly)
    6. composed full maps:
           M_fwd = W_src  . W_fwd . W_trg^{-1}
           M_bwd = W_trg  . W_bwd . W_src^{-1}

    inference:   y_hat = (pre(x) - mu_src) @ M_fwd + mu_trg     (src -> trg)
                 x_hat = (pre(y) - mu_trg) @ M_bwd + mu_src     (trg -> src)

For square, orthogonal configs the model is an exact bijective linear
operator:  M_bwd = M_fwd^{-1}  (verified to machine precision in the report).
Rectangular dims (d_src != d_trg) are fully supported (semi-orthogonal maps,
fitted independently per direction).

Methods
-------
procrustes : (default) whitened orthogonal Procrustes. Best generalisation,
             exact bidirectionality, scale-adaptive. With whitening on, the
             SVD singular values equal the canonical correlations of the pair.
ridge      : closed-form Tikhonov regression  W = (X'X + lambda I)^{-1} X'Y
             (lambda is *relative* to the mean eigenvalue -> scale-invariant).
             Useful for rectangular dims / very noisy pairs.
cca        : canonical correlation analysis; maps both sides into a shared
             latent k-dim space (P_src, P_trg). ``src_to_trg``/``trg_to_src``
             return shared-space projections; retrieval functions handle it
             transparently.

Quick start (library)
---------------------
>>> from crossmodal_mapper import CrossModalMapper
>>> mapper = CrossModalMapper(method="procrustes", verbose=1).fit(src, trg)
>>> y_hat = mapper.src_to_trg(x)                       # src -> trg
>>> x_hat = mapper.trg_to_src(y)                       # trg -> src
>>> idx, sims = mapper.search_trg(q_src, TRG_BANK, k=10)   # cross-modal retrieval
>>> idx, sims = mapper.search_src(q_trg, SRC_BANK, k=10)
>>> report = mapper.evaluate(src_test, trg_test)       # Recall@k, MRR, cosines
>>> mapper.save("mapper.npz")
>>> mapper2 = CrossModalMapper.load("mapper.npz")

Command line (demo on synthetic data with known ground truth)
-------------------------------------------------------------
    python crossmodal_mapper.py                          # 768 <-> 768 demo
    python crossmodal_mapper.py --dim-src 512 --dim-trg 768 --method ridge
    python crossmodal_mapper.py --method cca --cca-dim 256
    python crossmodal_mapper.py -v -v                    # + debug diagnostics
    python crossmodal_mapper.py --self-test              # consistency tests

Requirements: Python >= 3.8, numpy >= 1.17.  ``tqdm`` is optional (a simple
fallback progress meter is used otherwise). float64 (default) is recommended
for maximum precision; float32 is supported.

Changelog (v1.1.0)
------------------
- FIX: retrieval metrics now honour the true ground-truth row indices when
  queries are a sampled subset of the bank (in-sample report in fit() and
  evaluate() with sampling). Previously "query i <-> bank row i" was assumed,
  which silently produced chance-level numbers for sampled queries.
- FIX: stage diagnostics no longer crash when d_src != d_trg (cross-space
  cosine stages are skipped with a debug note).
- FIX: synthetic generator is well-conditioned by construction (orthonormal
  latent maps, smaller offsets), so the whitening eigenvalue floor is no
  longer tripped by artefacts of random square Gaussian matrices.
- STRONGER self-tests: the exact-linear path (length_norm off, noise 0)
  asserts machine-precision recovery; the default path asserts retrieval.

References
----------
- Schönemann (1966): A generalized solution of the orthogonal Procrustes problem.
- Mikolov, Le, Sutskever (2013): Exploiting similarities among languages.
- Xing et al. (2015): Normalized word embeddings (whitening).
- Artetxe, Labaka, Agirre (2018): Robust self-learning method — best-practice
  normalisation/whitening/orthogonalisation/re-weighting pipeline.
- Hotelling (1936): Canonical correlation analysis.
- Golub & Van Loan: Matrix Computations (numerics of SVD/ridge/CCA).
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, fields
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

try:  # optional progress bars
    from tqdm import tqdm as _tqdm
    _HAS_TQDM = True
except ImportError:  # pragma: no cover
    _HAS_TQDM = False

__version__ = "1.1.0"
__all__ = ["CrossModalMapper", "MapperConfig", "fit_crossmodal_mapper",
           "SyntheticCrossModal", "main"]

EPS = 1e-12
_TRACE = 5  # custom log level below DEBUG (verbose >= 3)
logging.addLevelName(_TRACE, "TRACE")

_LOG = logging.getLogger("crossmodal_mapper")
if not _LOG.handlers:  # configure once
    _h = logging.StreamHandler(sys.stderr)
    _h.setFormatter(logging.Formatter("%(asctime)s.%(msecs)03d | %(levelname)-7s | %(message)s",
                                      datefmt="%H:%M:%S"))
    _LOG.addHandler(_h)
    _LOG.propagate = False

ArrayLike = Union[np.ndarray, Sequence[Sequence[float]], Sequence[float]]


# --------------------------------------------------------------------------- #
#  formatting helpers                                                          #
# --------------------------------------------------------------------------- #
def _fmt(x: Optional[float], sig: int = 4) -> str:
    """Compact float formatting for reports."""
    if x is None:
        return "n/a"
    try:
        x = float(x)
    except (TypeError, ValueError):
        return str(x)
    if x == 0:
        return "0"
    if not math.isfinite(x):
        return str(x)
    if abs(x) >= 1e4 or abs(x) < 1e-3:
        return f"{x:.3e}"
    return f"{x:.{sig}g}"


def _pct(x: Optional[float]) -> str:
    return "n/a" if x is None else f"{100.0 * float(x):.2f}%"


def _fmt_vec(v: Any, k: int = 8) -> str:
    arr = np.asarray(v, dtype=float).ravel()
    head = ", ".join(_fmt(float(x)) for x in arr[:k])
    return f"[{head}{', …' if arr.size > k else ''}]"


# --------------------------------------------------------------------------- #
#  progress bars (tqdm with graceful fallback)                                 #
# --------------------------------------------------------------------------- #
class _SimpleProgress:
    """Minimal fallback progress meter when tqdm is unavailable."""

    def __init__(self, iterable, total: int, desc: str, unit: str):
        self._it = iterable
        self._total = max(1, int(total))
        self._desc = desc
        self._unit = unit
        self._i = 0

    def __iter__(self):
        step = max(1, self._total // 20)
        for x in self._it:
            if self._i % step == 0:
                sys.stderr.write(f"  {self._desc}: {self._i}/{self._total} {self._unit}s\n")
                sys.stderr.flush()
            self._i += 1
            yield x
        sys.stderr.write(f"  {self._desc}: {self._total}/{self._total} done\n")


def _progress(iterable, *, total: Optional[int] = None, desc: str = "work",
              enabled: bool = True, unit: str = "batch"):
    """Wrap ``iterable`` in a progress bar; identity wrapper when disabled."""
    if total is None:
        try:
            total = len(iterable)  # type: ignore[arg-type]
        except Exception:
            total = None
    if not enabled:
        return iterable
    if _HAS_TQDM:
        return _tqdm(iterable=iterable, total=total, desc=desc, unit=unit,
                     dynamic_ncols=True)
    return _SimpleProgress(iterable, total or 1, desc, unit)


# --------------------------------------------------------------------------- #
#  numeric helpers                                                             #
# --------------------------------------------------------------------------- #
def _row_norm_stats(X: np.ndarray) -> Dict[str, float]:
    norms = np.linalg.norm(X, axis=1)
    return {"norm_mean": float(norms.mean()), "norm_std": float(norms.std()),
            "norm_min": float(norms.min()), "norm_max": float(norms.max()),
            "zero_rows": int((norms < EPS).sum())}


def _l2_normalize_rows(X: np.ndarray) -> Tuple[np.ndarray, Dict[str, float]]:
    """Row-wise unit normalisation (zero rows are kept as zeros, never NaN)."""
    norms = np.linalg.norm(X, axis=1)
    zero = norms < EPS
    safe = np.where(zero, 1.0, norms)
    Xn = X / safe[:, None]
    nz = norms[~zero]
    stats = {"norm_mean": float(nz.mean()) if nz.size else 0.0,
             "norm_std": float(nz.std()) if nz.size else 0.0,
             "norm_min": float(nz.min()) if nz.size else 0.0,
             "norm_max": float(nz.max()) if nz.size else 0.0,
             "zero_rows": int(zero.sum())}
    return Xn, stats


def _pairwise_cosine_stats(pred: np.ndarray, target: np.ndarray) -> Dict[str, float]:
    """Alignment statistics over aligned row pairs (cosine, RMSE, MAE)."""
    pn = np.linalg.norm(pred, axis=1)
    tn = np.linalg.norm(target, axis=1)
    dots = np.einsum("ij,ij->i", pred, target)
    denom = pn * tn
    cos = np.zeros_like(dots)
    ok = denom > EPS
    cos[ok] = dots[ok] / denom[ok]
    diff = pred - target
    rmse = float(np.sqrt(np.mean(np.einsum("ij,ij->i", diff, diff))))
    mae = float(np.mean(np.abs(diff)))
    return {"n": int(cos.size),
            "cos_mean": float(cos.mean()), "cos_median": float(np.median(cos)),
            "cos_p5": float(np.percentile(cos, 5)), "cos_p25": float(np.percentile(cos, 25)),
            "cos_p75": float(np.percentile(cos, 75)), "cos_p95": float(np.percentile(cos, 95)),
            "cos_min": float(cos.min()), "cos_max": float(cos.max()),
            "rmse": rmse, "mae": mae,
            "n_zero_norm_pred": int((~ok).sum())}


def _column_pearson(A: np.ndarray, B: np.ndarray) -> Tuple[float, int]:
    """Mean per-dimension Pearson correlation between two aligned matrices."""
    ac = A - A.mean(0)
    bc = B - B.mean(0)
    sa, sb = ac.std(0), bc.std(0)
    valid = (sa > EPS) & (sb > EPS)
    n_const = int((~valid).sum())
    if not valid.any():
        return 0.0, n_const
    num = (ac[:, valid] * bc[:, valid]).mean(0)
    r = num / (sa[valid] * sb[valid])
    return float(r.mean()), n_const


def _effective_rank(eig: np.ndarray) -> float:
    """Participation ratio  (sum lambda)^2 / sum lambda^2  of a spectrum."""
    e = np.asarray(eig, dtype=float)
    e = e[e > 0]
    if e.size == 0:
        return 0.0
    return float((e.sum() ** 2) / np.sum(e * e))


def _semi_orth_error(W: np.ndarray) -> float:
    """||W'W - I||_F (tall) or ||WW' - I||_F (wide); ~1e-15 if semi-orthogonal."""
    if W.shape[0] >= W.shape[1]:
        G = W.T @ W
        k = W.shape[1]
    else:
        G = W @ W.T
        k = W.shape[0]
    return float(np.linalg.norm(G - np.eye(k), ord="fro"))


def _topk_search(queries: np.ndarray, bank: np.ndarray, k: int, *,
                 chunk_elems: int = 2 ** 24, desc: str = "search",
                 enabled: bool = True, dtype: np.dtype = np.float64,
                 return_scores: bool = True):
    """Chunked top-k cosine-similarity retrieval. Returns (idx, sims)."""
    qn = _l2_normalize_rows(np.asarray(queries, dtype=dtype))[0]
    bn = _l2_normalize_rows(np.asarray(bank, dtype=dtype))[0]
    nq, nb = qn.shape[0], bn.shape[0]
    if nq == 0 or nb == 0:
        raise ValueError("empty queries or bank")
    k = int(min(max(1, k), nb))
    idx = np.empty((nq, k), dtype=np.int64)
    sims = np.empty((nq, k), dtype=np.float64) if return_scores else None
    rows = max(1, int(chunk_elems) // max(1, nb))
    nbatches = (nq + rows - 1) // rows
    show = bool(enabled and nbatches > 1)
    for s in _progress(range(0, nq, rows), total=nbatches, desc=desc,
                       enabled=show, unit="chunk"):
        e = min(nq, s + rows)
        S = qn[s:e] @ bn.T
        if k < nb:
            part = np.argpartition(S, -k, axis=1)[:, -k:]
            val = np.take_along_axis(S, part, axis=1)
            order = np.argsort(-val, axis=1)
            top = np.take_along_axis(part, order, axis=1)
            topv = np.take_along_axis(val, order, axis=1)
        else:
            order = np.argsort(-S, axis=1)[:, :k]
            top = order
            topv = np.take_along_axis(S, order, axis=1)
        idx[s:e] = top
        if return_scores:
            sims[s:e] = topv  # type: ignore[index]
    return (idx, sims) if return_scores else idx


def _retrieval_metrics(queries: np.ndarray, bank: np.ndarray,
                       ks: Sequence[int] = (1, 5, 10), *,
                       truth: Optional[np.ndarray] = None,
                       chunk_elems: int = 2 ** 24, rank_depth: int = 100,
                       desc: str = "retrieval", enabled: bool = True,
                       dtype: np.dtype = np.float64) -> Dict[str, float]:
    """Recall@k / MRR / rank stats.

    ``truth[i]`` is the bank-row index of the correct answer for query i
    (default: query i <-> bank row i, i.e. fully aligned query/bank sets).
    Pass ``truth`` explicitly whenever queries are a subset/sample of the bank.
    """
    nq = int(queries.shape[0])
    nb = int(bank.shape[0])
    if truth is None:
        truth = np.arange(nq)
    truth = np.asarray(truth).astype(np.int64).reshape(-1)
    if truth.shape[0] != nq:
        raise ValueError(f"truth has {truth.shape[0]} entries but there are {nq} queries")
    if truth.size and (truth.min() < 0 or truth.max() >= nb):
        raise ValueError("truth indices out of bank range")
    ks_u = sorted({max(1, min(int(x), nb)) for x in ks})
    K = min(nb, max(int(rank_depth), max(ks_u)))
    idx = _topk_search(queries, bank, K, chunk_elems=chunk_elems, desc=desc,
                       enabled=enabled, dtype=dtype, return_scores=False)
    match = idx == truth[:, None]
    found = match.any(axis=1)
    pos = np.argmax(match, axis=1)
    ranks = np.where(found, pos + 1, K + 1)
    out: Dict[str, float] = {f"recall@{k}": float(np.mean(ranks <= k)) for k in ks_u}
    out["mrr"] = float(np.mean(np.where(found, 1.0 / np.maximum(pos + 1, 1), 0.0)))
    out["median_rank"] = float(np.median(ranks))
    out["mean_rank"] = float(np.mean(ranks))
    out["rank_depth"] = int(K)
    out["n_queries"] = nq
    out["bank_size"] = nb
    return out


# --------------------------------------------------------------------------- #
#  configuration                                                               #
# --------------------------------------------------------------------------- #
@dataclass
class MapperConfig:
    """All persistent hyper-parameters of the mapper (serialisable)."""
    method: str = "procrustes"
    length_normalize: Optional[bool] = None   # None -> auto per method
    center: Optional[bool] = None             # None -> True
    whiten: Optional[bool] = None             # None -> auto per method
    reweight_power: float = 0.0               # 0=orthogonal, 1=least-squares
    whiten_clamp: float = 1e-4                # relative eigenvalue floor
    ridge_lambda: float = 1e-3                # relative to mean eigenvalue
    cca_dim: Optional[int] = None             # None -> full rank
    dtype: str = "float64"
    batch_size: int = 8192
    sim_chunk_elems: int = 2 ** 24            # ~128 MB of sims per chunk
    nan_policy: str = "raise"                 # "raise" | "drop"
    retrieval_sample: int = 4000              # in-sample retrieval queries
    rank_depth: int = 100                     # MRR / rank truncation depth


_METHOD_ALIASES = {"procrustes": "procrustes", "orthogonal": "procrustes", "svd": "procrustes",
                   "ridge": "ridge", "ridge_regression": "ridge", "lsqr": "ridge",
                   "cca": "cca", "canonical": "cca"}
_METHODS = ("procrustes", "ridge", "cca")


def _resolve_method(name: Any) -> str:
    key = str(name).strip().lower()
    if key in _METHOD_ALIASES:
        return _METHOD_ALIASES[key]
    raise ValueError(f"unknown method {name!r}; expected one of {list(_METHODS)} "
                     f"(aliases: orthogonal/svd, lsqr, canonical)")


def _jsonable(o: Any) -> Any:
    """Recursively convert numpy scalars/arrays so json.dumps can serialise."""
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return [_jsonable(v) for v in o.tolist()]
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    return o


# ========================================================================== #
#  THE MODEL                                                                  #
# ========================================================================== #
class CrossModalMapper:
    """Non-ML/DL bi-directional cross-modal embedding mapper (closed-form).

    Parameters
    ----------
    method : {"procrustes", "ridge", "cca"} (default "procrustes")
    length_normalize, center, whiten : bool | None
        None = method-appropriate default (procrustes: all True; ridge:
        no whitening; cca: whitening forced, no length-norm).
    reweight_power : float in [0,1] (procrustes only)
        0 = orthogonal Procrustes (recommended; exact bidirectional inverse
        for square dims), 1 = least-squares in whitened space, between =
        shrinkage.
    whiten_clamp : float in (0,1)
        Relative eigenvalue floor for whitening. Bounds the amplification of
        directions the training data barely spans (see fit report
        "whiten max gain").
    ridge_lambda : float >= 0 (ridge only), relative to the mean eigenvalue.
    cca_dim : int | None (cca only) — dimensionality of the shared space.
    dtype : "float64" (recommended) | "float32"
    verbose : 0=warnings, 1=info+reports+progress bars, 2=+debug diagnostics,
              3=+trace (full spectra, condition numbers).
    progress : enable progress bars.
    """

    # ---------------------------------------------------------------- init
    def __init__(self, method: Optional[str] = None, *, config: Optional[MapperConfig] = None,
                 length_normalize: Optional[bool] = None, center: Optional[bool] = None,
                 whiten: Optional[bool] = None, reweight_power: Optional[float] = None,
                 whiten_clamp: Optional[float] = None, ridge_lambda: Optional[float] = None,
                 cca_dim: Optional[int] = None, dtype: Optional[str] = None,
                 batch_size: Optional[int] = None, sim_chunk_elems: Optional[int] = None,
                 nan_policy: Optional[str] = None, retrieval_sample: Optional[int] = None,
                 rank_depth: Optional[int] = None, verbose: int = 1, progress: bool = True):
        base = config if config is not None else MapperConfig()
        if not isinstance(base, MapperConfig):
            raise TypeError("config must be a MapperConfig instance")

        self.verbose = int(verbose)
        self.progress = bool(progress)
        _LOG.setLevel(logging.WARNING if self.verbose == 0 else
                      logging.INFO if self.verbose == 1 else
                      logging.DEBUG if self.verbose == 2 else _TRACE)

        cfg = MapperConfig(
            method=_resolve_method(method if method is not None else base.method),
            length_normalize=length_normalize if length_normalize is not None else base.length_normalize,
            center=center if center is not None else base.center,
            whiten=whiten if whiten is not None else base.whiten,
            reweight_power=reweight_power if reweight_power is not None else base.reweight_power,
            whiten_clamp=whiten_clamp if whiten_clamp is not None else base.whiten_clamp,
            ridge_lambda=ridge_lambda if ridge_lambda is not None else base.ridge_lambda,
            cca_dim=cca_dim if cca_dim is not None else base.cca_dim,
            dtype=dtype if dtype is not None else base.dtype,
            batch_size=batch_size if batch_size is not None else base.batch_size,
            sim_chunk_elems=sim_chunk_elems if sim_chunk_elems is not None else base.sim_chunk_elems,
            nan_policy=nan_policy if nan_policy is not None else base.nan_policy,
            retrieval_sample=retrieval_sample if retrieval_sample is not None else base.retrieval_sample,
            rank_depth=rank_depth if rank_depth is not None else base.rank_depth,
        )

        # resolve "auto" flags
        if cfg.length_normalize is None:
            cfg.length_normalize = (cfg.method != "cca")
        if cfg.center is None:
            cfg.center = True
        if cfg.whiten is None:
            cfg.whiten = (cfg.method != "ridge")
        if cfg.method == "cca" and not cfg.whiten:
            self._warn("CCA requires whitening — forcing whiten=True.")
            cfg.whiten = True
        if cfg.method != "procrustes" and cfg.reweight_power != 0.0:
            self._warn("reweight_power applies only to method='procrustes'; ignoring.")
            cfg.reweight_power = 0.0

        # validate
        if not (0.0 <= cfg.reweight_power <= 1.0):
            raise ValueError("reweight_power must be within [0, 1]")
        if not (0.0 < cfg.whiten_clamp < 1.0):
            raise ValueError("whiten_clamp must be within (0, 1)")
        if cfg.ridge_lambda < 0.0:
            raise ValueError("ridge_lambda must be >= 0")
        if cfg.cca_dim is not None and cfg.cca_dim < 1:
            raise ValueError("cca_dim must be >= 1")
        if cfg.batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if cfg.rank_depth < 1:
            raise ValueError("rank_depth must be >= 1")
        if cfg.retrieval_sample < 0:
            raise ValueError("retrieval_sample must be >= 0")
        if cfg.nan_policy not in ("raise", "drop"):
            raise ValueError("nan_policy must be 'raise' or 'drop'")
        try:
            dt = np.dtype(cfg.dtype)
        except Exception as e:  # pragma: no cover
            raise ValueError(f"invalid dtype {cfg.dtype!r}: {e}")
        if dt.kind != "f":
            raise ValueError("dtype must be a floating type ('float64'/'float32')")

        self.config = cfg
        self.dtype = dt
        self.config.sim_chunk_elems = max(2 ** 16, int(cfg.sim_chunk_elems))
        self._reset_state()
        self._debug(f"initialised: {_jsonable(asdict(cfg))}")

    # ------------------------------------------------------------- logging
    def _info(self, msg: str) -> None:
        if self.verbose >= 1:
            _LOG.info(msg)

    def _debug(self, msg: str) -> None:
        if self.verbose >= 2:
            _LOG.debug(msg)

    def _trace(self, msg: str) -> None:
        if self.verbose >= 3:
            _LOG.log(_TRACE, msg)

    def _warn(self, msg: str) -> None:
        _LOG.warning(msg)

    def _auto_progress(self, n_batches: int) -> bool:
        return bool(self.progress and self.verbose >= 1 and n_batches > 1)

    # ------------------------------------------------------------ lifecycle
    def _reset_state(self) -> None:
        self.fitted_: bool = False
        self.M_fwd: Optional[np.ndarray] = None
        self.M_bwd: Optional[np.ndarray] = None
        self.mu_src: Optional[np.ndarray] = None
        self.mu_trg: Optional[np.ndarray] = None
        self.W_src: Optional[np.ndarray] = None
        self.W_src_inv: Optional[np.ndarray] = None
        self.W_trg: Optional[np.ndarray] = None
        self.W_trg_inv: Optional[np.ndarray] = None
        self.W_fwd_core: Optional[np.ndarray] = None
        self.W_bwd_core: Optional[np.ndarray] = None
        self.P_src: Optional[np.ndarray] = None
        self.P_trg: Optional[np.ndarray] = None
        self.singular_values: Optional[np.ndarray] = None
        self.canonical_correlations: Optional[np.ndarray] = None
        self.eig_src: Optional[np.ndarray] = None
        self.eig_trg: Optional[np.ndarray] = None
        self.n_pairs_ = 0
        self.d_src_ = 0
        self.d_trg_ = 0
        self.stats_: Dict[str, Any] = {}
        self._cca_notice = False

    def _require_fitted(self) -> None:
        if not self.fitted_:
            raise RuntimeError("model is not fitted — call fit(src, trg) first, "
                               "or CrossModalMapper.load(path).")

    # ------------------------------------------------------------ validation
    def _validate_pair(self, src: ArrayLike, trg: ArrayLike) -> Tuple[np.ndarray, np.ndarray]:
        X = np.asarray(src, dtype=self.dtype)
        Y = np.asarray(trg, dtype=self.dtype)
        if X.ndim != 2 or Y.ndim != 2:
            raise ValueError(f"fit/evaluate inputs must be 2-D (n, d); "
                             f"got shapes {X.shape} and {Y.shape}")
        if X.shape[1] < 1 or Y.shape[1] < 1:
            raise ValueError("embedding dimension must be >= 1")
        if X.shape[0] != Y.shape[0]:
            raise ValueError(f"src and trg must hold the same number of aligned "
                             f"pairs: {X.shape[0]} vs {Y.shape[0]}")
        if X.shape[0] < 2:
            raise ValueError("need at least 2 aligned pairs")
        bad = ~np.isfinite(X)
        bady = ~np.isfinite(Y)
        if bad.any() or bady.any():
            rows = bad.any(1) | bady.any(1)
            n_bad = int(rows.sum())
            if self.config.nan_policy == "raise":
                raise ValueError(f"inputs contain NaN/Inf in {n_bad} aligned pair(s); "
                                 f"clean the data or set nan_policy='drop'")
            self._warn(f"dropping {n_bad} aligned pair(s) with NaN/Inf (nan_policy='drop')")
            X, Y = X[~rows], Y[~rows]
            if X.shape[0] < 2:
                raise ValueError("too few finite pairs remain after dropping NaN/Inf rows")
        return X, Y

    def _coerce_query(self, x: ArrayLike, expected: int, name: str) -> Tuple[bool, np.ndarray]:
        A = np.asarray(x, dtype=self.dtype)
        single = False
        if A.ndim == 1:
            single = True
            A = A.reshape(1, -1)
        elif A.ndim != 2:
            raise ValueError(f"{name} input must be 1-D or 2-D, got ndim={A.ndim}")
        if A.shape[0] == 0 or A.shape[1] == 0:
            raise ValueError(f"{name} input is empty")
        if A.shape[1] != expected:
            raise ValueError(f"{name} input has dim {A.shape[1]}, model expects {expected}")
        if not np.isfinite(A).all():
            raise ValueError(f"{name} input contains NaN/Inf values")
        return single, A

    # ------------------------------------------------------- linear algebra
    def _spectral_decomp(self, Xc: np.ndarray, side: str) -> Tuple[np.ndarray, np.ndarray]:
        """Eigendecomposition of the (unnormalised) scatter S = Xc'Xc, descending."""
        S = Xc.T @ Xc
        self._debug(f"[{side}] scatter matrix {S.shape}, trace={_fmt(np.trace(S))}")
        w, V = np.linalg.eigh(S)          # ascending
        w = np.maximum(w[::-1], 0.0)
        V = V[:, ::-1]
        if self.verbose >= 3:
            self._trace(f"[{side}] eigenvalues top50: {_fmt_vec(w, 50)} | "
                        f"bottom10: {_fmt_vec(np.sort(w)[:10], 10)}")
        return w, V

    def _whiten_from(self, w: np.ndarray, V: np.ndarray, side: str):
        """ZCA whitening W = E diag(max(l, c*lmax)^-1/2) E' and its inverse."""
        emax = float(w[0]) if w.size else 0.0
        if emax <= 0.0:
            self._warn(f"[{side}] zero variance — whitening disabled for this side.")
            return None, None, 0
        floor = self.config.whiten_clamp * emax
        safe = np.maximum(w, floor)
        n_clamped = int((w < floor).sum())
        W = (V / np.sqrt(safe)) @ V.T           # whitening   (Xc W)'(Xc W) = I
        W_inv = (V * np.sqrt(safe)) @ V.T        # de-whitening (exact inverse)
        return W, W_inv, n_clamped

    @staticmethod
    def _compose(core: np.ndarray, W_in: Optional[np.ndarray],
                  W_out_inv: Optional[np.ndarray]) -> np.ndarray:
        A = core if W_in is None else W_in @ core
        return A if W_out_inv is None else A @ W_out_inv

    # ---------------------------------------------------------------- fit
    def fit(self, src: ArrayLike, trg: ArrayLike, *, evaluate: bool = True) -> "CrossModalMapper":
        """Fit the bi-directional maps on aligned pairs.

        Parameters
        ----------
        src, trg : (n, d_src) / (n, d_trg) array-likes, row-aligned.
        evaluate : also run in-sample retrieval diagnostics (Recall@k, MRR).

        Returns ``self``. Prints a full fit report when verbose >= 1.
        """
        self._reset_state()
        t_fit = time.perf_counter()
        cfg = self.config

        X, Y = self._validate_pair(src, trg)
        n, ds, dt = X.shape[0], X.shape[1], Y.shape[1]
        self.n_pairs_, self.d_src_, self.d_trg_ = n, ds, dt

        self._info(f"fit() | method={cfg.method} | n={n} | dims {ds}->{dt} | "
                   f"len_norm={cfg.length_normalize} center={cfg.center} "
                   f"whiten={cfg.whiten} rho={cfg.reweight_power} dtype={cfg.dtype}")
        if n <= max(ds, dt):
            self._warn(f"n ({n}) <= max(d) ({max(ds, dt)}): covariance is rank-deficient; "
                       "spectral flooring / regularisation compensates, but more pairs are advised.")

        T: Dict[str, float] = {}

        # ---------------- 1. preprocessing --------------------------------
        t0 = time.perf_counter()
        src_stats: Dict[str, Any] = {"dim": ds, "rows": n, **_row_norm_stats(X)}
        trg_stats: Dict[str, Any] = {"dim": dt, "rows": n, **_row_norm_stats(Y)}
        Xp = _l2_normalize_rows(X)[0] if cfg.length_normalize else X
        Yp = _l2_normalize_rows(Y)[0] if cfg.length_normalize else Y
        if cfg.length_normalize and (src_stats["zero_rows"] or trg_stats["zero_rows"]):
            self._warn("zero-norm rows found (kept as zero vectors)")
        mu_src = Xp.mean(0) if cfg.center else np.zeros(ds, dtype=self.dtype)
        mu_trg = Yp.mean(0) if cfg.center else np.zeros(dt, dtype=self.dtype)
        Xc, Yc = Xp - mu_src, Yp - mu_trg
        T["preprocess"] = time.perf_counter() - t0
        self._debug(f"preprocess {T['preprocess']:.3f}s | |mu_src|={_fmt(np.linalg.norm(mu_src))} "
                    f"|mu_trg|={_fmt(np.linalg.norm(mu_trg))}")

        # ---------------- 2. spectra + whitening --------------------------
        t0 = time.perf_counter()
        eig_src, V_src = self._spectral_decomp(Xc, "src")
        eig_trg, V_trg = self._spectral_decomp(Yc, "trg")
        T["spectra"] = time.perf_counter() - t0

        W_src = W_src_inv = W_trg = W_trg_inv = None
        n_clamp_src = n_clamp_trg = 0
        if cfg.whiten:
            t0 = time.perf_counter()
            W_src, W_src_inv, n_clamp_src = self._whiten_from(eig_src, V_src, "src")
            W_trg, W_trg_inv, n_clamp_trg = self._whiten_from(eig_trg, V_trg, "trg")
            T["whiten"] = time.perf_counter() - t0
            if n_clamp_src:
                self._warn(f"src whitening: {n_clamp_src}/{ds} eigenvalue(s) floored "
                           f"(rank-deficient directions; clamp={cfg.whiten_clamp:g})")
            if n_clamp_trg:
                self._warn(f"trg whitening: {n_clamp_trg}/{dt} eigenvalue(s) floored "
                           f"(rank-deficient directions; clamp={cfg.whiten_clamp:g})")
            if cfg.method == "cca" and (W_src is None or W_trg is None):
                raise ValueError("CCA failed: one side has zero variance (constant embeddings).")
            self._debug(f"whitening | clamped src={n_clamp_src} trg={n_clamp_trg} | "
                        f"||W_src||_F={_fmt(np.linalg.norm(W_src)) if W_src is not None else 'n/a'}")
        Xw = Xc if W_src is None else Xc @ W_src
        Yw = Yc if W_trg is None else Yc @ W_trg

        # ---------------- 3. solve ----------------------------------------
        t_solve = time.perf_counter()
        cross = Xw.T @ Yw                          # (d_src x d_trg) cross-scatter
        U, s, Vt = np.linalg.svd(cross, full_matrices=True)
        r = min(ds, dt)
        self._trace(f"cross-spectrum sigma (top 50): {_fmt_vec(s, 50)}")

        lam_src = lam_trg = 0.0
        if cfg.method == "procrustes":
            rho = float(cfg.reweight_power)
            sw = s[:r] ** rho
            U_r, Vt_r = U[:, :r], Vt[:r, :]
            # forward core (d_src x d_trg), backward core (d_trg x d_src)
            self.W_fwd_core = (U_r * sw) @ Vt_r
            self.W_bwd_core = (Vt_r.T * sw) @ U_r.T
            self.M_fwd = self._compose(self.W_fwd_core, W_src, W_trg_inv)
            self.M_bwd = self._compose(self.W_bwd_core, W_trg, W_src_inv)
            self._info(f"solved orthogonal Procrustes ({ds}x{dt} SVD) | sigma: "
                       f"max={_fmt(s[0])} mean={_fmt(s[:r].mean())} min={_fmt(s[r - 1])}")
        elif cfg.method == "ridge":
            Sx = Xw.T @ Xw
            lam_src = cfg.ridge_lambda * max(float(np.trace(Sx)), 0.0) / ds
            if lam_src <= 0.0:
                raise ValueError("src side has zero variance; cannot fit ridge map.")
            self.W_fwd_core = np.linalg.solve(Sx + lam_src * np.eye(ds, dtype=self.dtype), cross)
            Sy = Yw.T @ Yw
            lam_trg = cfg.ridge_lambda * max(float(np.trace(Sy)), 0.0) / dt
            if lam_trg <= 0.0:
                raise ValueError("trg side has zero variance; cannot fit ridge map.")
            self.W_bwd_core = np.linalg.solve(Sy + lam_trg * np.eye(dt, dtype=self.dtype), cross.T)
            self.M_fwd = self._compose(self.W_fwd_core, W_src, W_trg_inv)
            self.M_bwd = self._compose(self.W_bwd_core, W_trg, W_src_inv)
            self._info(f"solved ridge | lambda_src={_fmt(lam_src)} lambda_trg={_fmt(lam_trg)}")
        else:  # cca
            k_cca = r if cfg.cca_dim is None else int(cfg.cca_dim)
            if k_cca > r:
                self._warn(f"cca_dim={cfg.cca_dim} exceeds rank limit {r}; clamping.")
                k_cca = r
            if k_cca < 1:
                raise ValueError("cca_dim must be >= 1")
            self.P_src = W_src @ U[:, :k_cca]           # (d_src x k)
            self.P_trg = W_trg @ Vt[:k_cca, :].T        # (d_trg x k)
            self.canonical_correlations = s[:k_cca].copy()
            self.M_fwd = self.M_bwd = None
            self._info(f"solved CCA | k={k_cca} | canonical correlations: "
                       f"max={_fmt(float(self.canonical_correlations.max()))} "
                       f"mean={_fmt(float(self.canonical_correlations.mean()))}")
        T["solve"] = time.perf_counter() - t_solve

        # ---------------- 4. persist state, then measure via public API ----
        self.mu_src, self.mu_trg = mu_src, mu_trg
        self.W_src, self.W_src_inv = W_src, W_src_inv
        self.W_trg, self.W_trg_inv = W_trg, W_trg_inv
        self.eig_src, self.eig_trg = eig_src, eig_trg
        self.singular_values = s.copy()
        self.fitted_ = True

        t0 = time.perf_counter()
        pred_f = self.src_to_trg(X)      # src -> trg   (shared proj. for CCA)
        pred_b = self.trg_to_src(Y)      # trg -> src   (shared proj. for CCA)
        if cfg.method != "cca":
            res_f = _pairwise_cosine_stats(pred_f, Yp)
            pm, nc = _column_pearson(pred_f, Yp)
            res_f.update({"pearson_mean": pm, "n_const_dims": nc})
            res_b = _pairwise_cosine_stats(pred_b, Xp)
            pm, nc = _column_pearson(pred_b, Xp)
            res_b.update({"pearson_mean": pm, "n_const_dims": nc})
        else:
            res = _pairwise_cosine_stats(pred_f, pred_b)
            pm, nc = _column_pearson(pred_f, pred_b)
            res.update({"pearson_mean": pm, "n_const_dims": nc})
            res_f = dict(res)
            res_b = {**res, "note": "shared-space pair alignment (symmetric)"}
        T["residuals"] = time.perf_counter() - t0
        self._info(f"train residuals | fwd cos mean={_fmt(res_f['cos_mean'])} "
                   f"rmse={_fmt(res_f['rmse'])} | bwd cos mean={_fmt(res_b['cos_mean'])}")

        # ---------------- 5. stage diagnostics (verbose >= 2) --------------
        diag: Dict[str, Any] = {}
        if self.verbose >= 2:
            t0 = time.perf_counter()

            def _stage(name: str, A: np.ndarray, Bm: np.ndarray) -> None:
                if A.shape[1] != Bm.shape[1]:
                    self._debug(f"stage '{name}': skipped (cross-space cosine undefined, "
                                f"dims {A.shape[1]} != {Bm.shape[1]})")
                    return
                st = _pairwise_cosine_stats(A, Bm)
                diag[name] = {"cos_mean": st["cos_mean"], "cos_p5": st["cos_p5"],
                              "cos_p95": st["cos_p95"], "rmse": st["rmse"]}
                self._debug(f"stage '{name}': cos mean={_fmt(st['cos_mean'])} "
                            f"p5={_fmt(st['cos_p5'])} rmse={_fmt(st['rmse'])}")

            _stage("raw", X, Y)
            _stage("centered", Xc, Yc)
            if cfg.whiten and W_src is not None and W_trg is not None:
                _stage("whitened", Xw, Yw)
            if cfg.method == "procrustes":
                _stage("mapped_whitened", Xw @ self.W_fwd_core, Yw)
            if cfg.method == "cca":
                _stage("final_shared", pred_f, pred_b)
            else:
                _stage("final_fwd", pred_f, Yp)
            T["diagnostics"] = time.perf_counter() - t0
            if self.verbose >= 3 and cfg.method != "cca" and min(ds, dt) <= 2048:
                self._trace(f"cond(M_fwd)={_fmt(np.linalg.cond(self.M_fwd))} "
                            f"cond(M_bwd)={_fmt(np.linalg.cond(self.M_bwd))}")

        # ---------------- 6. in-sample retrieval --------------------------
        insample: Optional[Dict[str, Any]] = None
        if evaluate:
            t0 = time.perf_counter()
            m = cfg.retrieval_sample
            idx = np.arange(n)
            if m and n > m:
                idx = np.sort(np.random.default_rng(0).choice(n, size=m, replace=False))
            if cfg.method != "cca":
                Qf, Bf = pred_f[idx], Y
                Qb, Bb = pred_b[idx], X
            else:
                Qf, Bf = pred_f[idx], pred_b
                Qb, Bb = pred_b[idx], pred_f
            show = bool(self.progress and self.verbose >= 1)
            mf = _retrieval_metrics(Qf, Bf, ks=(1, 5, 10), truth=idx,
                                    chunk_elems=cfg.sim_chunk_elems,
                                    rank_depth=cfg.rank_depth, desc="fit: in-sample src->trg",
                                    enabled=show, dtype=self.dtype)
            mb = _retrieval_metrics(Qb, Bb, ks=(1, 5, 10), truth=idx,
                                    chunk_elems=cfg.sim_chunk_elems,
                                    rank_depth=cfg.rank_depth, desc="fit: in-sample trg->src",
                                    enabled=show, dtype=self.dtype)
            insample = {"forward": mf, "backward": mb,
                        "n_queries": int(len(idx)), "bank_size": int(n)}
            T["insample_retrieval"] = time.perf_counter() - t0
            self._info(f"in-sample retrieval | fwd R@1={_pct(mf['recall@1'])} "
                       f"MRR={_fmt(mf['mrr'])} | bwd R@1={_pct(mb['recall@1'])} MRR={_fmt(mb['mrr'])}")

        # ---------------- 7. matrix / spectrum statistics ------------------
        mat_stats: Dict[str, Any] = {}
        if cfg.method == "cca":
            cc = self.canonical_correlations
            mat_stats = {"P_src_shape": list(self.P_src.shape),
                         "P_trg_shape": list(self.P_trg.shape),
                         "canonical_corr_max": float(cc.max()),
                         "canonical_corr_mean": float(cc.mean()),
                         "canonical_corr_min": float(cc.min()),
                         "canonical_corr_top10": [float(v) for v in cc[:10]]}
        else:
            mat_stats = {"M_fwd_shape": list(self.M_fwd.shape),
                         "M_bwd_shape": list(self.M_bwd.shape),
                         "M_fwd_fro": float(np.linalg.norm(self.M_fwd)),
                         "M_bwd_fro": float(np.linalg.norm(self.M_bwd)),
                         "W_fwd_orth_err": _semi_orth_error(self.W_fwd_core),
                         "W_bwd_orth_err": _semi_orth_error(self.W_bwd_core)}
            if self.W_fwd_core.shape == self.W_bwd_core.shape:
                mat_stats["core_bwd_vs_fwdT_maxdiff"] = float(
                    np.max(np.abs(self.W_bwd_core - self.W_fwd_core.T)))
            if ds == dt and ds <= 2048:
                mat_stats["fwd_bwd_inverse_err"] = float(np.linalg.norm(
                    self.M_fwd @ self.M_bwd - np.eye(ds, dtype=self.dtype)))
            if cfg.method == "ridge":
                mat_stats["ridge_lambda_src"] = float(lam_src)
                mat_stats["ridge_lambda_trg"] = float(lam_trg)

        spec = {"n_components": int(r), "s_max": float(s[0]), "s_min": float(s[r - 1]),
                "s_mean": float(s[:r].mean()), "s_top10": [float(v) for v in s[:10]],
                "energy": float(np.sum(s[:r] ** 2)),
                "whitened": bool(cfg.whiten and W_src is not None and W_trg is not None)}

        def _side_stats(eig: np.ndarray, base: Dict[str, Any], n_clamped: int) -> Dict[str, Any]:
            cond = float(eig[0] / eig[-1]) if eig[-1] > 0 else float("inf")
            gain = (float(1.0 / math.sqrt(cfg.whiten_clamp * float(eig[0])))
                    if (cfg.whiten and eig[0] > 0) else None)
            return {**base, "eff_rank": _effective_rank(eig), "cov_cond": cond,
                    "n_whiten_clamped": int(n_clamped), "whiten_max_gain": gain,
                    "eig_top10": [float(v) for v in eig[:10]]}

        T["total"] = time.perf_counter() - t_fit
        self.stats_ = {
            "method": cfg.method,
            "model": {"n_pairs": int(n), "dim_src": int(ds), "dim_trg": int(dt),
                      "dtype": cfg.dtype,
                      "length_normalize": bool(cfg.length_normalize),
                      "center": bool(cfg.center), "whiten": bool(cfg.whiten),
                      "reweight_power": float(cfg.reweight_power),
                      "whiten_clamp": float(cfg.whiten_clamp)},
            "src": _side_stats(eig_src, src_stats, n_clamp_src),
            "trg": _side_stats(eig_trg, trg_stats, n_clamp_trg),
            "cross_spectrum": spec,
            "matrices": mat_stats,
            "train_residuals": {"forward": res_f, "backward": res_b},
            "diagnostics": diag,
            "insample_retrieval": insample,
            "timings": {k: float(v) for k, v in T.items()},
        }
        self._info(f"fit() complete in {T['total']:.2f}s")
        if self.verbose >= 1:
            print(self.summary(), flush=True)
        return self

    # ----------------------------------------------------------- inference
    def _pre_rows(self, Z: np.ndarray) -> np.ndarray:
        if self.config.length_normalize:
            return _l2_normalize_rows(Z)[0]
        return Z

    def _zero(self, k: int) -> np.ndarray:
        return np.zeros(int(k), dtype=self.dtype)

    def _apply_map(self, Z: np.ndarray, M: np.ndarray, mu_in: np.ndarray,
                   mu_out: np.ndarray, desc: str,
                   show_progress: Optional[bool] = None) -> np.ndarray:
        """Chunked application of  out = (Z - mu_in) @ M + mu_out  with progress bar."""
        bs = max(1, int(self.config.batch_size))
        n = Z.shape[0]
        out = np.empty((n, M.shape[1]), dtype=self.dtype)
        nb = (n + bs - 1) // bs
        show = self._auto_progress(nb) if show_progress is None else bool(show_progress and nb > 1)
        for s in _progress(range(0, n, bs), total=nb, desc=desc, enabled=show, unit="batch"):
            e = min(n, s + bs)
            out[s:e] = (Z[s:e] - mu_in) @ M
            out[s:e] += mu_out
        return out

    def src_to_trg(self, src: ArrayLike, *, show_progress: Optional[bool] = None) -> np.ndarray:
        """Inference: map src-space embedding(s) into the trg space.

        Accepts a single vector (1-D, returns 1-D) or a batch (n, d_src).
        For method='cca' this returns the *shared latent-space* projection
        (compare against ``trg_to_src`` outputs, or use the retrieval API).
        """
        self._require_fitted()
        single, X = self._coerce_query(src, self.d_src_, "src")
        Xp = self._pre_rows(X)
        if self.config.method == "cca":
            out = self._apply_map(Xp, self.P_src, self.mu_src, self._zero(self.P_src.shape[1]),
                                  "project src->shared", show_progress)
            if not self._cca_notice:
                self._info("note: CCA mode — src_to_trg() returns shared-space projections; "
                           "use search_trg()/cross_similarity() or compare with trg_to_src().")
                self._cca_notice = True
        else:
            out = self._apply_map(Xp, self.M_fwd, self.mu_src, self.mu_trg,
                                  "map src->trg", show_progress)
        return out[0] if single else out

    def trg_to_src(self, trg: ArrayLike, *, show_progress: Optional[bool] = None) -> np.ndarray:
        """Inference: map trg-space embedding(s) into the src space (1-D or (n, d_trg))."""
        self._require_fitted()
        single, Y = self._coerce_query(trg, self.d_trg_, "trg")
        Yp = self._pre_rows(Y)
        if self.config.method == "cca":
            out = self._apply_map(Yp, self.P_trg, self.mu_trg, self._zero(self.P_trg.shape[1]),
                                  "project trg->shared", show_progress)
        else:
            out = self._apply_map(Yp, self.M_bwd, self.mu_trg, self.mu_src,
                                  "map trg->src", show_progress)
        return out[0] if single else out

    # aliases
    forward = src_to_trg
    backward = trg_to_src

    # -------------------------------------------------------- retrieval API
    def _encode_bank(self, bank: ArrayLike, side: str) -> np.ndarray:
        B = np.asarray(bank, dtype=self.dtype)
        if B.ndim == 1:
            B = B.reshape(1, -1)
        if B.ndim != 2:
            raise ValueError(f"{side} bank must be 1-D or 2-D, got ndim={B.ndim}")
        expected = self.d_trg_ if side == "trg" else self.d_src_
        if B.shape[1] != expected:
            raise ValueError(f"{side} bank has dim {B.shape[1]}, model expects {expected}")
        if not np.isfinite(B).all():
            raise ValueError(f"{side} bank contains NaN/Inf values")
        if self.config.method == "cca":
            P = self.P_trg if side == "trg" else self.P_src
            mu = self.mu_trg if side == "trg" else self.mu_src
            return self._apply_map(self._pre_rows(B), P, mu, self._zero(P.shape[1]),
                                   f"project {side} bank")
        return B

    def search_trg(self, query_src: ArrayLike, trg_bank: ArrayLike, *, k: int = 10,
                   return_scores: bool = True, show_progress: Optional[bool] = None):
        """Cross-modal retrieval src -> trg: map src queries, kNN in the trg bank.

        Returns ``(indices, cosine_scores)`` of shape (n_queries, k), sorted
        descending; or just indices if ``return_scores=False``.
        """
        self._require_fitted()
        Q = self.src_to_trg(query_src, show_progress=False)
        B = self._encode_bank(trg_bank, "trg")
        enabled = (self._auto_progress(2) if show_progress is None else bool(show_progress))
        return _topk_search(Q, B, k, chunk_elems=self.config.sim_chunk_elems,
                            desc="search src->trg", enabled=enabled,
                            dtype=self.dtype, return_scores=return_scores)

    def search_src(self, query_trg: ArrayLike, src_bank: ArrayLike, *, k: int = 10,
                   return_scores: bool = True, show_progress: Optional[bool] = None):
        """Cross-modal retrieval trg -> src: map trg queries, kNN in the src bank."""
        self._require_fitted()
        Q = self.trg_to_src(query_trg, show_progress=False)
        B = self._encode_bank(src_bank, "src")
        enabled = (self._auto_progress(2) if show_progress is None else bool(show_progress))
        return _topk_search(Q, B, k, chunk_elems=self.config.sim_chunk_elems,
                            desc="search trg->src", enabled=enabled,
                            dtype=self.dtype, return_scores=return_scores)

    def cross_similarity(self, src: ArrayLike, trg: ArrayLike,
                         *, show_progress: Optional[bool] = None) -> np.ndarray:
        """Cosine similarity matrix (n_src x n_trg) between mapped src and trg items."""
        self._require_fitted()
        Q = self.src_to_trg(src, show_progress=False)
        B = self._encode_bank(trg, "trg")
        qn = _l2_normalize_rows(Q)[0]
        bn = _l2_normalize_rows(B)[0]
        out = np.empty((qn.shape[0], bn.shape[0]), dtype=np.float64)
        rows = max(1, self.config.sim_chunk_elems // max(1, bn.shape[0]))
        nb = (qn.shape[0] + rows - 1) // rows
        show = (self._auto_progress(nb) if show_progress is None else bool(show_progress))
        for s in _progress(range(0, qn.shape[0], rows), total=nb, desc="cross similarity",
                           enabled=show, unit="chunk"):
            e = min(qn.shape[0], s + rows)
            out[s:e] = qn[s:e] @ bn.T
        return out

    # ---------------------------------------------------------- evaluation
    def evaluate(self, src: ArrayLike, trg: ArrayLike, *, k: Sequence[int] = (1, 5, 10),
                 retrieval_sample: int = 20000, baseline: Optional[bool] = None,
                 name: str = "eval") -> Dict[str, Any]:
        """Evaluate on held-out aligned pairs: pair-alignment stats + retrieval.

        Reports (per direction): mean/median/percentile cosine, RMSE, MAE,
        mean per-dim Pearson, Recall@k, MRR, median rank; plus an unmapped
        raw-cosine baseline when dims match. Returns the full metrics dict.
        """
        self._require_fitted()
        cfg = self.config
        X, Y = self._validate_pair(src, trg)
        n = X.shape[0]
        self._info(f"evaluate('{name}') on {n} pairs ...")

        pred_f = self.src_to_trg(X)
        pred_b = self.trg_to_src(Y)
        if cfg.method != "cca":
            res_f = _pairwise_cosine_stats(pred_f, self._pre_rows(Y))
            pm, nc = _column_pearson(pred_f, self._pre_rows(Y))
            res_f.update({"pearson_mean": pm, "n_const_dims": nc})
            res_b = _pairwise_cosine_stats(pred_b, self._pre_rows(X))
            pm, nc = _column_pearson(pred_b, self._pre_rows(X))
            res_b.update({"pearson_mean": pm, "n_const_dims": nc})
        else:
            res_f = _pairwise_cosine_stats(pred_f, pred_b)
            pm, nc = _column_pearson(pred_f, pred_b)
            res_f.update({"pearson_mean": pm, "n_const_dims": nc})
            res_b = {**res_f, "note": "shared-space pair alignment (symmetric)"}

        idx = np.arange(n)
        if retrieval_sample and n > retrieval_sample:
            idx = np.sort(np.random.default_rng(0).choice(n, size=retrieval_sample, replace=False))
            self._info(f"retrieval: sampling {retrieval_sample}/{n} queries (deterministic)")
        show = bool(self.progress and self.verbose >= 1)

        if cfg.method != "cca":
            Qf, Bf = pred_f[idx], self._encode_bank(Y, "trg")
            Qb, Bb = pred_b[idx], self._encode_bank(X, "src")
        else:
            Qf, Bf = pred_f[idx], pred_b
            Qb, Bb = pred_b[idx], pred_f
        rf = _retrieval_metrics(Qf, Bf, ks=k, truth=idx, chunk_elems=cfg.sim_chunk_elems,
                                rank_depth=cfg.rank_depth, desc=f"{name}: retrieval src->trg",
                                enabled=show, dtype=self.dtype)
        rb = _retrieval_metrics(Qb, Bb, ks=k, truth=idx, chunk_elems=cfg.sim_chunk_elems,
                                rank_depth=cfg.rank_depth, desc=f"{name}: retrieval trg->src",
                                enabled=show, dtype=self.dtype)

        out: Dict[str, Any] = {"name": name, "n": int(n), "method": cfg.method,
                               "dim_src": self.d_src_, "dim_trg": self.d_trg_,
                               "forward": res_f, "backward": res_b,
                               "retrieval_forward": rf, "retrieval_backward": rb}
        if baseline is None:
            baseline = (self.d_src_ == self.d_trg_)
        if baseline:
            if self.d_src_ == self.d_trg_:
                out["retrieval_baseline"] = _retrieval_metrics(
                    X[idx], Y, ks=k, truth=idx, chunk_elems=cfg.sim_chunk_elems,
                    rank_depth=cfg.rank_depth, desc=f"{name}: baseline (no mapping)",
                    enabled=show, dtype=self.dtype)
            else:
                self._warn("baseline retrieval skipped: dims differ (raw cosine "
                           "undefined across spaces)")
        if self.verbose >= 1:
            print(_format_eval(out), flush=True)
        self._info(f"evaluate('{name}') done | fwd R@1={_pct(rf['recall@1'])} "
                   f"bwd R@1={_pct(rb['recall@1'])}")
        return out

    # -------------------------------------------------------- serialization
    def save(self, path: str) -> None:
        """Save the fitted model (matrices, config, stats) to a .npz file."""
        self._require_fitted()
        items = [("M_fwd", self.M_fwd), ("M_bwd", self.M_bwd),
                 ("mu_src", self.mu_src), ("mu_trg", self.mu_trg),
                 ("W_src", self.W_src), ("W_src_inv", self.W_src_inv),
                 ("W_trg", self.W_trg), ("W_trg_inv", self.W_trg_inv),
                 ("W_fwd_core", self.W_fwd_core), ("W_bwd_core", self.W_bwd_core),
                 ("P_src", self.P_src), ("P_trg", self.P_trg),
                 ("singular_values", self.singular_values),
                 ("canonical_correlations", self.canonical_correlations),
                 ("eig_src", self.eig_src), ("eig_trg", self.eig_trg)]
        arrays = {name: np.asarray(v) for name, v in items if v is not None}
        meta = {"version": __version__, "created": time.strftime("%Y-%m-%d %H:%M:%S"),
                "config": _jsonable(asdict(self.config)), "keys": sorted(arrays),
                "d_src": self.d_src_, "d_trg": self.d_trg_, "n_pairs": self.n_pairs_,
                "stats": _jsonable(self.stats_)}
        if not str(path).endswith(".npz"):
            path = str(path) + ".npz"
        with open(path, "wb") as f:
            np.savez_compressed(f, _meta=np.array(json.dumps(meta)), **arrays)
        self._info(f"saved model -> {path} ({os.path.getsize(path) / 1e3:.1f} kB)")

    @classmethod
    def load(cls, path: str, *, verbose: int = 1, progress: bool = True) -> "CrossModalMapper":
        """Load a model saved with :meth:`save`."""
        with np.load(path, allow_pickle=False) as z:
            meta = json.loads(str(z["_meta"].item()))
            known = {f.name for f in fields(MapperConfig)}
            cfg = MapperConfig(**{k: v for k, v in meta["config"].items() if k in known})
            obj = cls(config=cfg, verbose=verbose, progress=progress)

            def get(key: str) -> Optional[np.ndarray]:
                return z[key] if key in z.files else None

            obj.M_fwd, obj.M_bwd = get("M_fwd"), get("M_bwd")
            obj.mu_src, obj.mu_trg = get("mu_src"), get("mu_trg")
            obj.W_src, obj.W_src_inv = get("W_src"), get("W_src_inv")
            obj.W_trg, obj.W_trg_inv = get("W_trg"), get("W_trg_inv")
            obj.W_fwd_core, obj.W_bwd_core = get("W_fwd_core"), get("W_bwd_core")
            obj.P_src, obj.P_trg = get("P_src"), get("P_trg")
            obj.singular_values = get("singular_values")
            obj.canonical_correlations = get("canonical_correlations")
            obj.eig_src, obj.eig_trg = get("eig_src"), get("eig_trg")
            obj.d_src_, obj.d_trg_ = int(meta["d_src"]), int(meta["d_trg"])
            obj.n_pairs_ = int(meta.get("n_pairs", 0))
            obj.stats_ = meta.get("stats", {})
            obj.fitted_ = True
        obj._info(f"loaded model from {path} (method={obj.config.method}, dims "
                  f"{obj.d_src_}<->{obj.d_trg_}, trained on {obj.n_pairs_} pairs)")
        return obj

    # ------------------------------------------------------------- reports
    def summary(self) -> str:
        return _format_report(self.stats_)

    def print_report(self, file: Any = None) -> None:
        print(self.summary(), file=file if file is not None else sys.stdout)

    def __repr__(self) -> str:
        if not self.fitted_:
            return f"<CrossModalMapper(method={self.config.method}, not fitted)>"
        cos = self.stats_.get("train_residuals", {}).get("forward", {}).get("cos_mean")
        return (f"<CrossModalMapper(method={self.config.method}, fitted, n={self.n_pairs_}, "
                f"dims={self.d_src_}<->{self.d_trg_}, train fwd cos={_fmt(cos)})>")

    # ----------------------------------------------------------- properties
    @property
    def is_fitted(self) -> bool:
        return self.fitted_

    @property
    def dim_src(self) -> int:
        return self.d_src_

    @property
    def dim_trg(self) -> int:
        return self.d_trg_

    @property
    def method(self) -> str:
        return self.config.method


def fit_crossmodal_mapper(src: ArrayLike, trg: ArrayLike, **kwargs: Any) -> CrossModalMapper:
    """Convenience one-liner:  CrossModalMapper(**kwargs).fit(src, trg)."""
    evaluate = kwargs.pop("evaluate", True)
    return CrossModalMapper(**kwargs).fit(src, trg, evaluate=evaluate)


# --------------------------------------------------------------------------- #
#  report formatting                                                           #
# --------------------------------------------------------------------------- #
def _format_report(stats: Dict[str, Any]) -> str:
    if not stats:
        return "CrossModalMapper: no stats yet (not fitted)."
    W = 78
    L: List[str] = []
    L.append("=" * W)
    L.append("CrossModalMapper — FIT REPORT".center(W))
    L.append("=" * W)

    def kv(key: str, val: Any) -> None:
        L.append(f"  {key:<20}: {val}")

    def sec(title: str) -> None:
        L.append("-" * W)
        L.append(f"[{title}]")

    m = stats.get("model", {})
    kv("method", stats.get("method"))
    kv("pairs (train)", m.get("n_pairs"))
    kv("dims", f"{m.get('dim_src')} -> {m.get('dim_trg')}  (dtype {m.get('dtype')})")
    kv("preprocessing", f"length_norm={m.get('length_normalize')}  center={m.get('center')}  "
                        f"whiten={m.get('whiten')}  reweight rho={m.get('reweight_power')}")
    if m.get("whiten"):
        kv("whiten clamp", f"{m.get('whiten_clamp'):.0e} (relative eigenvalue floor)")

    for side in ("src", "trg"):
        d = stats.get(side)
        if not d:
            continue
        sec(f"{side} data")
        kv("rows x dim", f"{d.get('rows')} x {d.get('dim')}")
        kv("row norms", f"mean {_fmt(d.get('norm_mean'))}  std {_fmt(d.get('norm_std'))}  "
                        f"min {_fmt(d.get('norm_min'))}  max {_fmt(d.get('norm_max'))}")
        kv("zero rows", d.get("zero_rows"))
        kv("effective rank", f"{_fmt(d.get('eff_rank'))} / {d.get('dim')}")
        kv("cov cond", _fmt(d.get("cov_cond")))
        if d.get("n_whiten_clamped") is not None:
            kv("whiten floored dims", f"{d.get('n_whiten_clamped')}  "
                                      f"(max gain {_fmt(d.get('whiten_max_gain'))}x)")
        kv("eig top10", _fmt_vec(d.get("eig_top10", [])))

    spec = stats.get("cross_spectrum")
    if spec:
        tag = (" (whitened => canonical correlations)" if spec.get("whitened")
               else " (raw cross-scatter)")
        sec("cross spectrum" + tag)
        kv("components", spec.get("n_components"))
        kv("singular values", f"max {_fmt(spec.get('s_max'))}  mean {_fmt(spec.get('s_mean'))}  "
                              f"min {_fmt(spec.get('s_min'))}")
        kv("top 10", _fmt_vec(spec.get("s_top10", [])))
        kv("energy sum(sigma^2)", _fmt(spec.get("energy")))

    mat = stats.get("matrices")
    if mat:
        sec("transform matrices")
        if stats.get("method") == "cca":
            kv("P_src (src->shared)", f"{mat.get('P_src_shape')}")
            kv("P_trg (trg->shared)", f"{mat.get('P_trg_shape')}")
            kv("canonical corr", f"max {_fmt(mat.get('canonical_corr_max'))}  "
                                 f"mean {_fmt(mat.get('canonical_corr_mean'))}  "
                                 f"min {_fmt(mat.get('canonical_corr_min'))}")
            kv("corr top10", _fmt_vec(mat.get("canonical_corr_top10", [])))
        else:
            kv("M_fwd (src->trg)", f"{mat.get('M_fwd_shape')}, ||.||_F={_fmt(mat.get('M_fwd_fro'))}")
            kv("M_bwd (trg->src)", f"{mat.get('M_bwd_shape')}, ||.||_F={_fmt(mat.get('M_bwd_fro'))}")
            kv("core orthog. error", f"fwd {_fmt(mat.get('W_fwd_orth_err'))}  "
                                     f"bwd {_fmt(mat.get('W_bwd_orth_err'))}")
            if mat.get("core_bwd_vs_fwdT_maxdiff") is not None:
                kv("|W_bwd - W_fwd'|_max", _fmt(mat.get("core_bwd_vs_fwdT_maxdiff")))
            if mat.get("fwd_bwd_inverse_err") is not None:
                kv("||M_fwd M_bwd - I||_F", f"{_fmt(mat.get('fwd_bwd_inverse_err'))}   "
                                            f"(~0 => backward is the exact inverse)")
            if mat.get("ridge_lambda_src") is not None:
                kv("ridge lambda", f"src {_fmt(mat.get('ridge_lambda_src'))} / "
                                   f"trg {_fmt(mat.get('ridge_lambda_trg'))}")

    tr = stats.get("train_residuals") or {}
    for label, key in (("forward  src->trg", "forward"), ("backward trg->src", "backward")):
        res = tr.get(key)
        if res:
            sec(f"train residuals — {label}")
            kv("cosine", f"mean {_fmt(res.get('cos_mean'))}  median {_fmt(res.get('cos_median'))}  "
                         f"p5 {_fmt(res.get('cos_p5'))}  p95 {_fmt(res.get('cos_p95'))}")
            kv("cosine range", f"[{_fmt(res.get('cos_min'))}, {_fmt(res.get('cos_max'))}]")
            kv("rmse / mae", f"{_fmt(res.get('rmse'))} / {_fmt(res.get('mae'))}")
            kv("per-dim pearson", f"mean {_fmt(res.get('pearson_mean'))} "
                                  f"({res.get('n_const_dims')} constant dims)")
            if res.get("note"):
                kv("note", res["note"])

    diag = stats.get("diagnostics")
    if diag:
        sec("stage diagnostics (pipeline progression)")
        for name, d in diag.items():
            kv(name, f"cos mean {_fmt(d.get('cos_mean'))}  p5 {_fmt(d.get('cos_p5'))}  "
                     f"rmse {_fmt(d.get('rmse'))}")

    ir = stats.get("insample_retrieval")
    if ir:
        sec("in-sample retrieval (optimistic, train bank)")
        for name, key in (("src->trg", "forward"), ("trg->src", "backward")):
            d = ir.get(key)
            if d:
                kv(name, f"R@1 {_pct(d.get('recall@1'))}  R@5 {_pct(d.get('recall@5'))}  "
                         f"R@10 {_pct(d.get('recall@10'))}  MRR {_fmt(d.get('mrr'))}  "
                         f"med-rank {_fmt(d.get('median_rank'))}")
        kv("queries / bank", f"{ir.get('n_queries')} / {ir.get('bank_size')}")

    T = stats.get("timings")
    if T:
        sec("timings")
        kv("fit total", f"{_fmt(T.get('total'))} s")
        kv("breakdown", "  ".join(f"{k} {v:.2f}s" for k, v in T.items() if k != "total"))

    L.append("=" * W)
    return "\n".join(L)


def _format_eval(res: Dict[str, Any]) -> str:
    W = 78
    L: List[str] = []
    L.append("-" * W)
    L.append(f" EVALUATION — {res.get('name')}  (n={res.get('n')} pairs, "
             f"dims {res.get('dim_src')}<->{res.get('dim_trg')}, method={res.get('method')})")
    L.append("-" * W)
    for label, key in (("src->trg", "forward"), ("trg->src", "backward")):
        d = res.get(key)
        if d:
            L.append(f"  pair alignment {label}: cos mean {_fmt(d.get('cos_mean'))}  "
                     f"median {_fmt(d.get('cos_median'))}  p5 {_fmt(d.get('cos_p5'))}  |  "
                     f"rmse {_fmt(d.get('rmse'))}  pearson {_fmt(d.get('pearson_mean'))}")
    for label, key in (("src->trg", "retrieval_forward"), ("trg->src", "retrieval_backward")):
        d = res.get(key)
        if d:
            L.append(f"  retrieval {label}: R@1 {_pct(d['recall@1'])}  "
                     f"R@5 {_pct(d['recall@5'])}  R@10 {_pct(d['recall@10'])}  |  "
                     f"MRR {_fmt(d['mrr'])}  median rank {_fmt(d['median_rank'])}")
    d = res.get("retrieval_baseline")
    if d:
        L.append(f"  baseline (no mapping): R@1 {_pct(d['recall@1'])}  "
                 f"MRR {_fmt(d['mrr'])}   <- what you get without alignment")
    if res.get("method") == "cca":
        L.append("  (CCA mode: pair stats are computed in the shared latent space)")
    L.append("-" * W)
    return "\n".join(L)


# --------------------------------------------------------------------------- #
#  synthetic cross-modal generator (ground truth) for demo & self-test         #
# --------------------------------------------------------------------------- #
def _random_orthogonal(rng: np.random.Generator, d: int) -> np.ndarray:
    """Haar-distributed random orthogonal matrix via QR with sign correction."""
    Q, R = np.linalg.qr(rng.standard_normal((d, d)))
    diag = np.diagonal(R)
    signs = np.where(np.abs(diag) < EPS, 1.0, np.sign(diag))
    return Q * signs[None, :]


def _row_orthonormal(rng: np.random.Generator, k: int, d: int) -> np.ndarray:
    """Random (k, d) matrix with orthonormal rows (k <= d), Haar-distributed."""
    Q, R = np.linalg.qr(rng.standard_normal((d, k)))      # Q: (d, k), orthonormal columns
    diag = np.diagonal(R)
    signs = np.where(np.abs(diag) < EPS, 1.0, np.sign(diag))
    return (Q * signs[None, :]).T                          # (k, d), B @ B.T = I


class SyntheticCrossModal:
    """Generates aligned cross-modal pairs with a known exact linear relation.

    Clean pairs satisfy  trg_clean = (src_clean - b_src) @ W_true + b_trg,
    plus anisotropic covariance (to exercise whitening). ``sample(noise>0)``
    adds per-dimension Gaussian noise relative to the clean signal std.

    The latent maps B have orthonormal rows, so the data is well-conditioned
    by construction: the covariance spectrum is governed only by ``anisotropy``
    (no random near-degenerate directions that would trip the whitening floor).
    """

    def __init__(self, dim_src: int = 768, dim_trg: int = 768, seed: int = 0,
                 anisotropy: float = 0.2):
        rng = np.random.default_rng(seed)
        self._seed = int(seed)
        self.dim_src, self.dim_trg = int(dim_src), int(dim_trg)
        self.d_lat = min(self.dim_src, self.dim_trg)
        self.Q_src = _random_orthogonal(rng, self.dim_src)
        self.Q_trg = _random_orthogonal(rng, self.dim_trg)
        self._A_src = self.Q_src * np.geomspace(1.0, float(anisotropy), self.dim_src)[None, :]
        self._A_trg = self.Q_trg * np.geomspace(1.0, float(anisotropy), self.dim_trg)[None, :]
        self.B_src = _row_orthonormal(rng, self.d_lat, self.dim_src)
        self.B_trg = _row_orthonormal(rng, self.d_lat, self.dim_trg)
        self.b_src = 0.15 * rng.standard_normal(self.dim_src)
        self.b_trg = 0.15 * rng.standard_normal(self.dim_trg)
        self._rng = rng
        pilot_src, pilot_trg = self._clean(4096, np.random.default_rng(seed + 777))
        self.src_std = np.maximum(pilot_src.std(0), 1e-12)
        self.trg_std = np.maximum(pilot_trg.std(0), 1e-12)

    def _clean(self, n: int, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
        H = rng.standard_normal((n, self.d_lat))
        src = H @ self.B_src @ self._A_src + self.b_src
        trg = H @ self.B_trg @ self._A_trg + self.b_trg
        return src, trg

    def sample(self, n: int, noise: float = 0.0, seed: Optional[int] = None
               ) -> Tuple[np.ndarray, np.ndarray]:
        rng = self._rng if seed is None else np.random.default_rng(seed)
        src, trg = self._clean(int(n), rng)
        if noise > 0:
            src = src + noise * self.src_std[None, :] * rng.standard_normal(src.shape)
            trg = trg + noise * self.trg_std[None, :] * rng.standard_normal(trg.shape)
        return src, trg


# --------------------------------------------------------------------------- #
#  self-test                                                                   #
# --------------------------------------------------------------------------- #
def _self_test() -> int:
    print("running built-in self-tests ...")
    gen = SyntheticCrossModal(32, 32, seed=7)
    X, Y = gen.sample(3000, noise=0.0, seed=1)

    # 1) exact linear path: no length-norm, no noise => machine-precision recovery.
    #    (with length_norm off, centred data satisfies Yc = Xc @ W exactly, so the
    #    centre + whiten + Procrustes + compose pipeline is algebraically exact)
    m_lin = CrossModalMapper(length_normalize=False, verbose=0).fit(
        X[:2500], Y[:2500], evaluate=False)
    assert m_lin.M_fwd.shape == (32, 32) and m_lin.M_bwd.shape == (32, 32)
    inv_err = float(np.linalg.norm(m_lin.M_fwd @ m_lin.M_bwd - np.eye(32)))
    assert inv_err < 1e-8, inv_err
    print(f"  [ok] square procrustes: M_bwd == M_fwd^-1 (err={inv_err:.2e})")
    cos = _pairwise_cosine_stats(m_lin.src_to_trg(X[2500:]), Y[2500:])["cos_mean"]
    assert cos > 1.0 - 1e-6, cos
    print(f"  [ok] exact linear path recovers ground truth (cos={cos:.9f})")
    idx, _ = m_lin.search_trg(X[2500:2505], Y[2500:], k=1)
    assert (idx[:, 0] == np.arange(5)).all()
    idx, _ = m_lin.search_src(Y[2500:2505], X[2500:], k=1)
    assert (idx[:, 0] == np.arange(5)).all()
    print("  [ok] exact path retrieval (both directions) returns ground truth")

    # 2) default path (length_norm=True): the normalised-pair relation is only
    #    approximately linear, so retrieval is the meaningful metric.
    m_def = CrossModalMapper(verbose=0).fit(X[:2500], Y[:2500], evaluate=False)
    res = m_def.evaluate(X[2500:], Y[2500:], name="self-test/default")
    r1f = res["retrieval_forward"]["recall@1"]
    r1b = res["retrieval_backward"]["recall@1"]
    assert r1f > 0.99 and r1b > 0.99, (r1f, r1b)
    print(f"  [ok] default pipeline (length-norm) retrieval: R@1 fwd={r1f:.3f} bwd={r1b:.3f}")

    # 3) rectangular dims
    gen2 = SyntheticCrossModal(16, 40, seed=8)
    X2, Y2 = gen2.sample(3000, noise=0.05, seed=2)
    m_rect = CrossModalMapper(verbose=0).fit(X2[:2500], Y2[:2500], evaluate=False)
    assert m_rect.M_fwd.shape == (16, 40) and m_rect.M_bwd.shape == (40, 16)
    res2 = m_rect.evaluate(X2[2500:], Y2[2500:], name="self-test/rect")
    assert res2["retrieval_forward"]["recall@1"] > 0.9
    print("  [ok] rectangular dims (16 -> 40) supported, retrieval works")

    # 4) procrustes variants
    m_rho = CrossModalMapper(whiten=False, reweight_power=0.5, verbose=0).fit(
        X, Y, evaluate=False)
    idx, _ = m_rho.search_trg(X[2500:2505], Y[2500:], k=1)
    assert (idx[:, 0] == np.arange(5)).all()
    print("  [ok] procrustes without whitening, rho=0.5")

    # 5) ridge & cca
    CrossModalMapper(method="ridge", verbose=0).fit(X, Y, evaluate=False)
    print("  [ok] ridge method runs")
    m_cca = CrossModalMapper(method="cca", verbose=0).fit(X, Y, evaluate=False)
    assert m_cca.P_src.shape == (32, 32) and m_cca.P_trg.shape == (32, 32)
    idx, _ = m_cca.search_trg(X[2500:2505], Y[2500:], k=1)
    assert (idx[:, 0] == np.arange(5)).all()
    print("  [ok] cca method runs & retrieves correctly in shared space")

    # 6) NaN policy
    Xn = X.copy()
    Xn[3, 0] = np.nan
    m_nan = CrossModalMapper(nan_policy="drop", verbose=0).fit(Xn, Y, evaluate=False)
    assert m_nan.n_pairs_ == len(Xn) - 1
    print("  [ok] nan_policy='drop' removes bad pairs")

    # 7) save/load roundtrip
    fd, path = tempfile.mkstemp(suffix=".npz")
    os.close(fd)
    try:
        m_def.save(path)
        m_ld = CrossModalMapper.load(path, verbose=0)
        assert np.allclose(m_ld.src_to_trg(X[2500:]), m_def.src_to_trg(X[2500:]), atol=1e-12)
    finally:
        os.unlink(path)
    print("  [ok] save/load roundtrip reproduces inference exactly")

    print("all self-tests passed.")
    return 0


# --------------------------------------------------------------------------- #
#  command-line demo                                                           #
# --------------------------------------------------------------------------- #
class _HelpFmt(argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter):
    pass


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="crossmodal_mapper",
        description=("Non-ML/DL bi-directional cross-modal embedding mapper "
                     "(closed-form linear algebra). Runs a full demo on synthetic "
                     "data with known ground truth unless --self-test is given."),
        formatter_class=_HelpFmt,
        epilog=("examples:\n"
                "  python crossmodal_mapper.py\n"
                "  python crossmodal_mapper.py --dim-src 512 --dim-trg 768 --method ridge\n"
                "  python crossmodal_mapper.py --method cca --cca-dim 256\n"
                "  python crossmodal_mapper.py -v -v            # debug diagnostics\n"
                "  python crossmodal_mapper.py --self-test\n"
                "\n"
                "tip: BLAS threading speeds up large runs (e.g. OMP_NUM_THREADS=8)."))
    p.add_argument("--method", default="procrustes", choices=list(_METHODS),
                   help="alignment method (aliases in the library: orthogonal/svd, lsqr, canonical)")
    p.add_argument("--n-pairs", type=int, default=20000, help="synthetic pairs to generate")
    p.add_argument("--dim-src", type=int, default=768, help="source embedding dimension")
    p.add_argument("--dim-trg", type=int, default=768, help="target embedding dimension")
    p.add_argument("--noise", type=float, default=0.15,
                   help="relative per-dim Gaussian noise on both modalities")
    p.add_argument("--train-frac", type=float, default=0.8, help="train fraction of pairs")
    p.add_argument("--oracle-n", type=int, default=2000,
                   help="noiseless held-out pairs to measure ground-truth recovery")
    p.add_argument("--demo-queries", type=int, default=4, help="live retrieval queries to show")
    p.add_argument("--reweight-power", type=float, default=0.0,
                   help="procrustes shrinkage rho: 0=orthogonal (exact bijection), 1=least squares")
    p.add_argument("--ridge-lambda", type=float, default=1e-3, help="ridge lambda (relative)")
    p.add_argument("--cca-dim", type=int, default=None, help="shared-space dims for CCA")
    p.add_argument("--retrieval-sample", type=int, default=4000,
                   help="in-sample retrieval queries during fit")
    p.add_argument("--no-whiten", action="store_true", help="disable ZCA whitening")
    p.add_argument("--no-length-norm", action="store_true", help="disable row length normalisation")
    p.add_argument("--no-center", action="store_true", help="disable mean centering")
    p.add_argument("--dtype", default="float64", choices=["float64", "float32"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save", metavar="PATH", default=None, help="save fitted model to .npz")
    p.add_argument("-v", "--verbosity", action="count", default=1,
                   help="-v info+report, -vv +debug, -vvv +trace")
    p.add_argument("-q", "--quiet", action="store_true", help="suppress reports/bars")
    p.add_argument("--self-test", action="store_true", help="run consistency tests and exit")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    if args.self_test:
        return _self_test()
    verbose = 0 if args.quiet else max(1, min(3, args.verbosity))

    t0 = time.perf_counter()
    print("=" * 70)
    print(" crossmodal_mapper — non-ML/DL bi-directional cross-modal alignment demo")
    print(f" numpy {np.__version__} | tqdm: {'yes' if _HAS_TQDM else 'no (fallback meter)'} | "
          f"seed {args.seed} | method {args.method}")
    print("=" * 70)

    gen = SyntheticCrossModal(dim_src=args.dim_src, dim_trg=args.dim_trg, seed=args.seed)
    X, Y = gen.sample(args.n_pairs, noise=args.noise)
    if X.shape[1] == Y.shape[1]:
        b = _pairwise_cosine_stats(X, Y)
        print(f"[data] generated {X.shape} and {Y.shape} | latent dim {gen.d_lat} | "
              f"noise {args.noise}")
        print(f"[data] aligned raw cosine BEFORE mapping: mean={b['cos_mean']:.4f} "
              f"p5={b['cos_p5']:.4f}   <- this is what unaligned spaces give you")
    else:
        print(f"[data] generated {X.shape} and {Y.shape} | latent dim {gen.d_lat} | "
              f"noise {args.noise} | dims differ — raw cosine undefined across spaces")

    n_tr = int(round(args.n_pairs * args.train_frac))
    Xtr, Ytr = X[:n_tr], Y[:n_tr]
    Xev, Yev = X[n_tr:], Y[n_tr:]
    print(f"[data] split: train {len(Xtr)} pairs | held-out {len(Xev)} pairs")

    mapper = CrossModalMapper(
        method=args.method, verbose=verbose, progress=not args.quiet,
        whiten=(False if args.no_whiten else None),
        length_normalize=(False if args.no_length_norm else None),
        center=(False if args.no_center else None),
        reweight_power=args.reweight_power, ridge_lambda=args.ridge_lambda,
        cca_dim=args.cca_dim, dtype=args.dtype,
        retrieval_sample=args.retrieval_sample)
    mapper.fit(Xtr, Ytr)
    if verbose == 0:
        st = mapper.stats_.get("train_residuals", {}).get("forward", {})
        print(f"[fit] done (quiet) — train fwd cos mean: {_fmt(st.get('cos_mean'))}")

    print("\nHeld-out evaluation (noisy pairs the model never saw):")
    mapper.evaluate(Xev, Yev, name="held-out")

    print("\nOracle evaluation (noiseless pairs => measures ground-truth recovery):")
    Xo, Yo = gen.sample(args.oracle_n, noise=0.0, seed=args.seed + 999)
    mapper.evaluate(Xo, Yo, name="oracle (noiseless)")

    # ---------------------- live inference demo ---------------------------
    print("\n" + "-" * 70)
    print("INFERENCE DEMO")
    n_q = min(args.demo_queries, len(Xev))
    k_demo = min(3, len(Yev))
    print(f"src->trg: {n_q} held-out src queries vs held-out trg bank ({len(Yev)} items), top-{k_demo}:")
    idx, sims = mapper.search_trg(Xev[:n_q], Yev, k=k_demo)
    for j in range(n_q):
        items = "  ".join(f"#{idx[j, t]:>5}({sims[j, t]:+.4f})" for t in range(idx.shape[1]))
        pos = np.flatnonzero(idx[j] == j)
        verdict = ("HIT @1" if pos.size and pos[0] == 0
                   else (f"rank {pos[0] + 1}" if pos.size else f"miss (>top-{k_demo})"))
        print(f"        query {j:>3} (true trg #{j:>3}) -> {items}   [{verdict}]")

    print(f"trg->src: {n_q} held-out trg queries vs held-out src bank ({len(Xev)} items), top-{k_demo}:")
    idx2, sims2 = mapper.search_src(Yev[:n_q], Xev, k=k_demo)
    for j in range(n_q):
        items = "  ".join(f"#{idx2[j, t]:>5}({sims2[j, t]:+.4f})" for t in range(idx2.shape[1]))
        pos = np.flatnonzero(idx2[j] == j)
        verdict = ("HIT @1" if pos.size and pos[0] == 0
                   else (f"rank {pos[0] + 1}" if pos.size else f"miss (>top-{k_demo})"))
        print(f"        query {j:>3} (true src #{j:>3}) -> {items}   [{verdict}]")

    v = mapper.src_to_trg(Xev[0])
    print(f"        example: src_to_trg(Xev[0]) -> dim {v.shape[0]}, "
          f"first 5 components {np.round(v[:5], 4).tolist()}, ||v||={np.linalg.norm(v):.3f}")

    if args.save:
        mapper.save(args.save)

    print("\nInference API recap:")
    print("  y_hat   = mapper.src_to_trg(x)                      # src -> trg (1-D or batch)")
    print("  x_hat   = mapper.trg_to_src(y)                      # trg -> src")
    print("  idx, s  = mapper.search_trg(x_q, TRG_BANK, k=10)    # cross-modal retrieval src->trg")
    print("  idx, s  = mapper.search_src(y_q, SRC_BANK, k=10)    # cross-modal retrieval trg->src")
    print("  S       = mapper.cross_similarity(x_q, y_bank)      # cosine score matrix")
    print(f"\n[done] total wall time {time.perf_counter() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())