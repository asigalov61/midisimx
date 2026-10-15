"""streaming_memory.py

Loss-minimizing streaming aggregation of fixed-dim (768) encoder embeddings
into a single 768-d memory vector, updated iteratively.

Core math (training-free path, MemoryAggregator):
  * Inputs are unit-normalized: direction = semantics, norm = confidence.
  * Weighted streaming mean (Welford, weight-aware):
        mu_{t+1} = mu_t + (w / (W + w)) * (x_hat - mu_t)
    which is EXACTLY the weighted mean  mu = sum(w_i x_i) / sum(w_i),
    i.e. the point maximizing average cosine similarity to all members.
  * Streaming weighted variance (diagonal, or full covariance optionally)
    tracks dispersion -> enables uncertainty-aware output shaping and
    split/consolidate decisions in a memory system.
  * All statistics are EXACTLY mergeable (Chan et al.), enabling lossless
    hierarchical memory trees.

  add() is O(d) (O(d^2) with full covariance). Internals run in float64
  for long-stream numerical stability; outputs are float32.

Optional learned path (LearnedSetCompressor): a GRU-cell folder whose
hidden state is strictly 768-d at every step, trained so the final vector
reconstructs / covers / discriminates its member set better than the mean.

License: Apache 2.0

Project Los Angeles
Tegridy Code 2026
"""

from __future__ import annotations

import warnings
from collections.abc import Iterable
from dataclasses import asdict, dataclass, replace
from typing import Any, Literal, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

Tensor = torch.Tensor
ArrayLike = Union[Tensor, np.ndarray]

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------


@dataclass
class AggregatorConfig:
    """Settings for :class:`MemoryAggregator` (all defaults are lossless)."""

    dim: int = 768
    internal_dtype: torch.dtype = torch.float64   # stable accumulators
    output_dtype: torch.dtype = torch.float32     # emitted embedding dtype
    device: str = "cuda"

    # --- weighting (defaults = maximally lossless) ---
    weight_by_norm: bool = False    # encoder norms carry confidence/informativeness
    norm_power: float = 1.0         # w_i = ||x_i|| ** norm_power
    min_weight: float = 1e-8
    max_weight: float = 1e6

    # --- optional robustness shaping (all OFF by default = lossless) ---
    novelty_boost: float = 0.0      # >0: damp redundant, boost novel inputs
    novelty_floor: float = 0.1      # min relative weight for fully-redundant input
    recency_lambda: float = 1.0     # 1.0 = no forgetting (lossless)

    # --- statistics ---
    track_covariance: bool = False  # True: full 768x768 covariance (O(d^2)/step)

    # --- output shaping ---
    variance_sharpening: float = 0.0  # >0: variance-whitened centroid (use
                                      #     .similarity() for consistent scoring)
    eps: float = 1e-12
    warn_low_consensus: bool = True   # warn when members disagree (mu-norm tiny)
    zero_norm_policy: Literal["skip", "eps"] = "skip"


_DTYPE_BY_NAME: dict[str, torch.dtype] = {
    "torch.float16": torch.float16,
    "torch.float32": torch.float32,
    "torch.float64": torch.float64,
}


def _cfg_to_dict(cfg: AggregatorConfig) -> dict[str, Any]:
    """Config -> JSON-safe dict (dtypes rendered as strings)."""
    return {k: (str(v) if isinstance(v, torch.dtype) else v)
            for k, v in asdict(cfg).items()}


def _cfg_from_dict(d: dict[str, Any]) -> AggregatorConfig:
    """Inverse of :func:`_cfg_to_dict`."""
    kw = dict(d)
    for k in ("internal_dtype", "output_dtype"):
        if isinstance(kw.get(k), str):
            kw[k] = _DTYPE_BY_NAME.get(kw[k], kw[k])
    return AggregatorConfig(**kw)


# ----------------------------------------------------------------------------
# Training-free streaming aggregator (recommended default)
# ----------------------------------------------------------------------------
class MemoryAggregator:
    """
    Iteratively folds one-or-many 768-d embeddings into one 768-d memory
    embedding with minimal information loss for cosine-based retrieval.

    Usage:
        mem = MemoryAggregator()          # dim=768 by default
        mem.add(x1)                       # one embedding, or
        mem.add(torch.stack([x2, x3]))    # several at once (streamed in order)
        v768 = mem.embedding()            # current consolidated memory
        score = mem.similarity(query)     # consistent query scoring
    """

    def __init__(self, cfg: AggregatorConfig | None = None, **overrides: Any) -> None:
        if cfg is None:
            cfg = AggregatorConfig(**overrides)
        elif overrides:
            cfg = replace(cfg, **overrides)
        self.cfg = cfg
        self.dim = cfg.dim
        self.device = torch.device(cfg.device)
        self.reset()

    # ------------------------------------------------------------------ state
    def reset(self) -> None:
        """Initialize (or reset) all streaming statistics to empty."""
        dt, d = self.cfg.internal_dtype, self.dim
        self._mu = torch.zeros(d, dtype=dt, device=self.device)
        self._M2 = torch.zeros(d, dtype=dt, device=self.device)          # diag
        self._C = (torch.zeros(d, d, dtype=dt, device=self.device)
                   if self.cfg.track_covariance else None)               # full
        self._W = 0.0        # total weight
        self._W2 = 0.0       # total squared weight (for ESS)
        self._n = 0          # accepted count
        self._sum_norm = 0.0
        self._min_norm: float | None = None
        self._max_norm: float | None = None
        self._warned_low_consensus = False

    @property
    def count(self) -> int:
        """Number of accepted embeddings."""
        return self._n

    def __len__(self) -> int:
        return self._n

    def __repr__(self) -> str:
        return (f"MemoryAggregator(dim={self.dim}, n={self._n}, "
                f"W={self._W:.3g}, consensus={self.consensus():.4f})")

    def clone(self) -> MemoryAggregator:
        """Return an independent copy of this aggregator (same config)."""
        c = MemoryAggregator(self.cfg)
        c._mu, c._M2 = self._mu.clone(), self._M2.clone()
        c._C = None if self._C is None else self._C.clone()
        c._W, c._W2, c._n = self._W, self._W2, self._n
        c._sum_norm, c._min_norm, c._max_norm = (
            self._sum_norm, self._min_norm, self._max_norm)
        return c

    # ------------------------------------------------------------------ input
    def _prepare(self, x: ArrayLike) -> Tensor:
        """Coerce/validate input to an (N, d) internal-dtype tensor on device."""
        if not torch.is_tensor(x):
            x = torch.as_tensor(x)
        x = x.to(device=self.device, dtype=self.cfg.internal_dtype)
        if x.ndim == 1:
            x = x.unsqueeze(0)
        x = x.reshape(-1, self.dim)
        if x.shape[1] != self.dim:
            raise ValueError(f"expected trailing dim {self.dim}, got {x.shape}")
        if not torch.isfinite(x).all():
            raise ValueError("non-finite values in input embedding(s)")
        return x

    # ------------------------------------------------------------- insertion
    def add(self, x: ArrayLike, weight: float = 1.0,
            show_progress: bool = False) -> bool:
        """Add one embedding (d,) or a batch (N, d), streamed in order.

        Args:
            x: single embedding or stacked batch.
            weight: scalar weight applied to every row.
            show_progress: show a tqdm bar over batch rows.

        Returns:
            True iff every embedding was accepted.
        """
        X = self._prepare(x)
        ok = True
        for row in tqdm(X, disable=not show_progress, desc="add"):
            ok = self._add_one(row, float(weight)) and ok
        return ok

    def extend(self, Xs: Iterable[ArrayLike], weight: float = 1.0,
               show_progress: bool = False) -> None:
        """Stream an iterable of embeddings/batches through :meth:`add`."""
        for x in tqdm(Xs, disable=not show_progress, desc="extend"):
            self.add(x, weight)

    def _add_one(self, x: Tensor, user_w: float) -> bool:
        """Fold one prepared (d,) row into the running statistics."""
        cfg = self.cfg
        nrm = x.norm()

        # unit direction; handle degenerate zero vectors
        if float(nrm) < cfg.eps:
            if cfg.zero_norm_policy == "skip":
                return False
            x_hat = x / (nrm + cfg.eps)
        else:
            x_hat = x / nrm

        # ---- importance weight: encoder norm as confidence, plus user weight
        w = user_w
        if cfg.weight_by_norm:
            w *= float(nrm) ** cfg.norm_power

        # ---- optional novelty shaping: damp redundancy, boost novelty
        if cfg.novelty_boost > 0.0 and self._W > 0.0:
            mu_hat = F.normalize(self._mu, dim=0, eps=cfg.eps)
            novelty = 1.0 - float(x_hat @ mu_hat)          # 1 = orthogonal/new
            g = cfg.novelty_floor + (1.0 - cfg.novelty_floor) * novelty
            w *= g ** cfg.novelty_boost

        w = min(max(w, cfg.min_weight), cfg.max_weight)
        if w <= 0.0:
            return False

        # ---- optional recency decay of accumulated mass
        if cfg.recency_lambda != 1.0:
            lam = cfg.recency_lambda
            self._W *= lam
            self._W2 *= lam
            if self._C is not None:
                self._C *= lam
            else:
                self._M2 *= lam

        # ---- exact weighted streaming mean (Welford, weight-aware)
        W_new = self._W + w
        delta = x_hat - self._mu
        self._mu = self._mu + (w / W_new) * delta
        resid = x_hat - self._mu

        # ---- streaming weighted second moments
        if self._C is not None:
            self._C += w * torch.outer(delta, resid)
        else:
            self._M2 += w * delta * resid

        self._W = W_new
        self._W2 += w * w
        self._n += 1

        n = float(nrm)
        self._sum_norm += n
        self._min_norm = n if self._min_norm is None else min(self._min_norm, n)
        self._max_norm = n if self._max_norm is None else max(self._max_norm, n)
        return True

    # ----------------------------------------------------------------- output
    def _variance(self) -> Tensor:
        """Current weighted diagonal variance of the member directions."""
        W = max(self._W, self.cfg.eps)
        return (self._C.diagonal() if self._C is not None else self._M2) / W

    def embedding(self) -> Tensor:
        """Current consolidated memory: one 768-d unit vector.
        Default = normalized weighted mean (max-average-cosine centroid).
        With variance_sharpening > 0, it is variance-whitened before normalizing
        (score queries with .similarity() so the transform is applied there too)."""
        cfg = self.cfg
        if self._n == 0:
            return torch.zeros(self.dim, dtype=cfg.output_dtype, device=self.device)

        z = self._mu
        if cfg.variance_sharpening > 0.0:
            z = z * (self._variance() + cfg.eps) ** (-0.5 * cfg.variance_sharpening)

        nrm = z.norm()
        if float(nrm) < cfg.eps:   # members nearly cancel; no meaningful direction
            return torch.zeros(self.dim, dtype=cfg.output_dtype, device=self.device)
        if cfg.warn_low_consensus and not self._warned_low_consensus \
                and self._n > 1 and float(nrm) < 0.05:
            warnings.warn(
                "Low consensus (||mu||<0.05): members are highly dissimilar; a "
                "single 768-d vector is a poor summary. Consider splitting this "
                "memory into children (aggregates merge exactly).")
            self._warned_low_consensus = True
        return (z / nrm).to(cfg.output_dtype)

    def similarity(self, queries: ArrayLike) -> Tensor:
        """Cosine similarity of query embedding(s) (m, d) to this memory.
        Applies the same variance-sharpening transform to queries as to the
        memory vector, so scoring stays consistent with .embedding()."""
        cfg = self.cfg
        if self._n == 0:
            m = self._prepare(queries).shape[0]
            return torch.zeros(m, dtype=cfg.output_dtype, device=self.device)
        q = F.normalize(self._prepare(queries), dim=1, eps=cfg.eps)
        z = self._mu
        if cfg.variance_sharpening > 0.0:
            s = (self._variance() + cfg.eps) ** (-0.5 * cfg.variance_sharpening)
            q, z = q * s, z * s
            q = F.normalize(q, dim=1, eps=cfg.eps)
        z = F.normalize(z, dim=0, eps=cfg.eps)
        return (q @ z).to(cfg.output_dtype)

    # ------------------------------------------------------------ diagnostics
    def consensus(self) -> float:
        """||mu|| = exact weighted mean cosine of members to the centroid.
        (1 - it) x 2 = expected squared distance to centroid. Watch this:
        dropping consensus => the memory is covering heterogeneous content."""
        return 0.0 if self._n == 0 else float(self._mu.norm())

    def stats(self) -> dict[str, Any]:
        """Summary diagnostics (counts, weights, Kish ESS, consensus, norms)."""
        if self._n == 0:
            return {"count": 0, "total_weight": 0.0,
                    "effective_sample_size": 0.0, "consensus_cosine": 0.0}
        ess = (self._W ** 2 / self._W2) if self._W2 > 0 else 0.0
        return {
            "count": self._n,
            "total_weight": self._W,
            "effective_sample_size": ess,          # Kish ESS (1 if equal weights)
            "consensus_cosine": self.consensus(),  # E[cos(member, centroid)]
            "dispersion_l2": 2.0 * (1.0 - self.consensus()),
            "mean_input_norm": self._sum_norm / self._n,
            "min_input_norm": self._min_norm,
            "max_input_norm": self._max_norm,
        }

    # ------------------------------------------------- exact merging (lossless)
    def merge_inplace(self, other: MemoryAggregator) -> MemoryAggregator:
        """Exactly merge `other` into self (Chan et al. parallel formulas).
        Exact for mean/variance/count when recency_lambda == 1 (default)."""
        if self.dim != other.dim:
            raise ValueError("dim mismatch")
        if self.cfg.track_covariance != other.cfg.track_covariance:
            raise ValueError("track_covariance mismatch")
        if self.cfg.recency_lambda != 1.0 or other.cfg.recency_lambda != 1.0:
            warnings.warn("merging decayed memories treats each total weight "
                          "as its effective mass (approximate under decay)")
        if other._n == 0:
            return self
        if self._n == 0:
            c = other.clone()
            self.__dict__.update(c.__dict__)
            return self

        Wa, Wb = self._W, other._W
        W = Wa + Wb
        delta = other._mu - self._mu
        coef = Wa * Wb / W
        self._mu = self._mu + (Wb / W) * delta
        if self._C is not None:
            self._C = self._C + other._C + coef * torch.outer(delta, delta)
        else:
            self._M2 = self._M2 + other._M2 + coef * delta * delta
        self._W, self._W2 = W, self._W2 + other._W2
        self._n += other._n
        self._sum_norm += other._sum_norm
        if other._min_norm is not None:
            self._min_norm = (other._min_norm if self._min_norm is None
                              else min(self._min_norm, other._min_norm))
            self._max_norm = (other._max_norm if self._max_norm is None
                              else max(self._max_norm, other._max_norm))
        return self

    @staticmethod
    def merge(a: MemoryAggregator, b: MemoryAggregator) -> MemoryAggregator:
        """Lossless merge of two aggregators (returns a new object)."""
        return a.clone().merge_inplace(b)

    @classmethod
    def merge_all(cls, aggs: Iterable[MemoryAggregator]) -> MemoryAggregator:
        """Losslessly fold an iterable of aggregators into the first one."""
        it = iter(aggs)
        out = next(it)
        for a in it:
            out.merge_inplace(a)
        return out

    # ------------------------------------------------------------ persistence
    def to_dict(self) -> dict[str, Any]:
        """Serialize config + state to a plain (JSON-safe) dict."""
        st = {"n": self._n, "W": self._W, "W2": self._W2,
              "sum_norm": self._sum_norm, "min_norm": self._min_norm,
              "max_norm": self._max_norm,
              "mu": self._mu.detach().cpu().numpy() if self._n else None,
              "M2": self._M2.detach().cpu().numpy()
                    if (self._n and self._C is None) else None,
              "C": self._C.detach().cpu().numpy()
                   if (self._n and self._C is not None) else None}
        return {"format": "streaming-memory-v1",
                "config": _cfg_to_dict(self.cfg), "state": st}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> MemoryAggregator:
        """Rebuild an aggregator from :meth:`to_dict` output."""
        m = cls(_cfg_from_dict(d["config"]))
        st = d["state"]
        dev, dt = m.device, m.cfg.internal_dtype
        m._n, m._W, m._W2 = int(st["n"]), float(st["W"]), float(st["W2"])
        m._sum_norm = float(st["sum_norm"])
        m._min_norm = st["min_norm"]
        m._max_norm = st["max_norm"]
        if st["mu"] is not None:
            m._mu = torch.as_tensor(st["mu"], device=dev, dtype=dt)
            if st["M2"] is not None:
                m._M2 = torch.as_tensor(st["M2"], device=dev, dtype=dt)
            if st["C"] is not None:
                m._C = torch.as_tensor(st["C"], device=dev, dtype=dt)
        return m


# ----------------------------------------------------------------------------
# Optional: trainable single-vector folder (state is STRICTLY 768-d per step)
# ----------------------------------------------------------------------------
class LearnedSetCompressor(nn.Module):
    """
    Folds a set of embeddings into one 768-d vector with a GRU cell, trained
    self-supervised so the final vector is maximally informative about members:
      * reconstruction: a decoder must recover a prototype similar to each member
      * coverage: cosine to members should be high AND uniform (no forgotten items)
      * contrast: different sets must map to distinguishable vectors (InfoNCE)
      * anchor: stay near the spherical mean (stability regularizer)
    Use this when you can train on your own embedding distribution and want
    pure single-vector iterative folding with no side state.

    Usage example:
    --------------

    device = 'cuda'

    corpus = torch.from_numpy(corpus_emb[:-1024])
    query = torch.from_numpy(corpus_emb[-1024:])
    
    model = LearnedSetCompressor(dim=768)
    
    train_set_compressor(model, corpus_emb[:-1024])
    
    with torch.inference_mode():
        Xe = query.unsqueeze(0).to(device)
        Xe_h = F.normalize(Xe, dim=-1)
        z = model(Xe)
        ours = torch.einsum('bnd,bd->bn', Xe_h, z).mean()
        base = F.normalize(Xe_h.mean(1), dim=1)
        bl = torch.einsum('bnd,bd->bn', Xe_h, base).mean()
        
    print(f"  held-out mean member-cosine: learned={ours:.4f} vs mean-baseline={bl:.4f}")
    """

    def __init__(self, dim: int = 768, contrast_temp: float = 0.07) -> None:
        super().__init__()
        self.dim = dim
        self.cell = nn.GRUCell(dim, dim)
        self.decoder = nn.Sequential(nn.Linear(dim, 2 * dim), nn.GELU(),
                                     nn.Linear(2 * dim, dim))
        self.tau = contrast_temp

    def fold(self, X: ArrayLike, device: str = "cuda") -> Tensor:
        """X: (B, N, d) or (N, d) -> z: (B, d) unit-norm memory vector."""
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X)
        if X.ndim == 2:
            X = X.unsqueeze(0)
        Xh = F.normalize(X, dim=-1, eps=1e-6)
        Xh = Xh.to(device)
        h = Xh.new_zeros(Xh.size(0), self.dim)
        for t in range(Xh.size(1)):                 # iterative, order-aware
            h = self.cell(Xh[:, t, :], h)
        return F.normalize(h, dim=-1, eps=1e-6)

    def forward(self, X: ArrayLike, device: str = "cuda") -> Tensor:
        """Alias for :meth:`fold`."""
        return self.fold(X, device=device)

    def losses(self, X: Tensor, w_recon: float = 1.0, w_spread: float = 0.5,
               w_anchor: float = 0.25, w_contrast: float = 0.5
               ) -> tuple[Tensor, dict[str, float]]:
        """Self-supervised objective; returns (total loss, per-term values).

        Terms: reconstruction, spread (coverage uniformity), anchor, contrast.
        """
        if X.ndim == 2:
            X = X.unsqueeze(0)
        z, Xh = self.fold(X), F.normalize(X, dim=-1, eps=1e-6)
        dec = F.normalize(self.decoder(z), dim=-1, eps=1e-6)
        recon = (1 - torch.einsum('bnd,bd->bn', Xh, dec)).mean()
        spread = torch.einsum('bnd,bd->bn', Xh, z).std(dim=1).mean()
        set_mean = F.normalize(Xh.mean(dim=1), dim=-1, eps=1e-6)
        anchor = (1 - (z * set_mean).sum(-1)).mean()
        if z.size(0) > 1:
            logits = (z @ set_mean.t()) / self.tau
            labels = torch.arange(z.size(0), device=z.device)
            contrast = F.cross_entropy(logits, labels)
        else:
            contrast = torch.zeros((), device=z.device)
        total = w_recon * recon + w_spread * spread + w_anchor * anchor \
            + w_contrast * contrast
        return total, {"recon": float(recon), "spread": float(spread),
                       "anchor": float(anchor), "contrast": float(contrast)}


class StreamingLearnedCompressor:
    """MemoryAggregator-style streaming API around a trained LearnedSetCompressor.
    The GRU hidden state h (768-d) IS the memory; we just keep it across calls.
    Equivalent to model.fold() run one step at a time."""

    def __init__(self, model: LearnedSetCompressor, device: str = "cpu") -> None:
        self.model = model.to(device).eval()
        self.device = device
        self.dim = model.dim
        self._h: torch.Tensor | None = None   # (1, d) running memory state
        self._n = 0

    def reset(self) -> None:
        """Clear the running memory state and count."""
        self._h, self._n = None, 0

    @torch.no_grad()
    def add(self, x: ArrayLike, show_progress: bool = False) -> None:
        """x: (d,) single embedding or (N, d) batch — streamed in order."""
        X = torch.as_tensor(x, device=self.device, dtype=torch.float32)
        if X.ndim == 1:
            X = X.unsqueeze(0)
        X = F.normalize(X, dim=-1, eps=1e-6)          # same preprocessing as training
        for t in tqdm(range(X.size(0)), disable=not show_progress, desc="fold"):
            h0 = (torch.zeros(1, self.dim, device=self.device)
                  if self._h is None else self._h)
            self._h = self.model.cell(X[t].unsqueeze(0), h0)
            self._n += 1

    @torch.no_grad()
    def embedding(self) -> torch.Tensor:
        """Current consolidated 768-d memory (unit norm), after every step."""
        if self._h is None:
            return torch.zeros(self.dim)
        return F.normalize(self._h, dim=-1, eps=1e-6).squeeze(0)

    @torch.no_grad()
    def similarity(self, q: ArrayLike) -> torch.Tensor:
        """Cosine similarity of query (d,) or (m, d) against the memory."""
        q = torch.as_tensor(q, device=self.device, dtype=torch.float32)
        if q.ndim == 1:
            q = q.unsqueeze(0)
        return F.normalize(q, dim=-1, eps=1e-6) @ self.embedding()

    @property
    def count(self) -> int:
        """Number of embeddings folded so far."""
        return self._n


def train_set_compressor(model: LearnedSetCompressor, sets: ArrayLike,
                         epochs: int = 5, batch_size: int = 32, lr: float = 1e-3,
                         permute: bool = False, device: str = "cuda",
                         log: bool = True, show_progress: bool = False
                         ) -> LearnedSetCompressor:
    """Train a :class:`LearnedSetCompressor` on a corpus of embedding-sets.

    Args:
        model: model to train in place.
        sets: (S, N, d) tensor of S embedding-sets with N members each.
        epochs, batch_size, lr: optimization knobs (AdamW).
        permute: shuffle member order within each batch (the fold is
            order-aware, so this is a cheap augmentation).
        device: target device.
        log: print per-epoch mean loss.
        show_progress: show a tqdm bar over batches with live loss.

    Returns:
        The model, in eval mode.
    """
    model.to(device).train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    if isinstance(sets, np.ndarray):
        sets = torch.from_numpy(sets)

    n_sets = sets.shape[0]
    for ep in range(1, epochs + 1):
        set_order, tot, nb = torch.randperm(n_sets), 0.0, 0
        pbar = tqdm(range(0, n_sets, batch_size),
                    disable=not show_progress, desc=f"epoch {ep}/{epochs}")
        for i in pbar:
            xb = sets[set_order[i:i + batch_size]]
            if permute:
                member_order = torch.randperm(xb.shape[1])
                xb = xb[:, member_order]              # shuffle members, not sets
            xb = xb.to(device)
            loss, _ = model.losses(xb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss)
            nb += 1
            pbar.set_postfix(loss=f"{float(loss):.4f}", avg=f"{tot / nb:.4f}")
        if log:
            print(f"  epoch {ep}: loss={tot / nb:.4f}")
    return model.eval()

# ----------------------------------------------------------------------------
# Deduping functions
# ----------------------------------------------------------------------------

def dedupe_exact(X: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Collapse exact duplicate rows, losslessly, preserving stream order.

    Returns:
      Xu        (U, d) unique rows in first-occurrence order
      w         (U,)   multiplicity weights — pass to add(weight=...)
      inv       (N,)   original row -> index into Xu
      kept      (U,)   original row index of each representative
    Identity: sum over original rows == sum over Xu rows weighted by w,
    so any aggregate built with weights is bit-identical (in exact math)
    to the one built from raw X."""
    Xu_lex, inv, counts = torch.unique(X, dim=0, return_inverse=True,
                                       return_counts=True)
    U, N = Xu_lex.size(0), X.size(0)
    # torch.unique sorts lexicographically -> restore first-occurrence order
    pos = torch.arange(N, device=X.device)
    first = torch.full((U,), N, dtype=torch.long, device=X.device)
    first.scatter_reduce_(0, inv, pos, reduce="amin")     # earliest row per unique id
    order = torch.argsort(first)                          # unique ids in stream order
    kept = first[order]
    Xu = X[kept]
    remap = torch.empty(U, dtype=torch.long, device=X.device)
    remap[order] = torch.arange(U, device=X.device)
    inv = remap[inv]                                      # remap to stream order
    w = counts[order].float()
    assert int(w.sum()) == N
    return Xu, w, inv, kept


@torch.no_grad()
def find_near_dups(X: Tensor, threshold: float = 0.999, block: int = 2048,
                   show_progress: bool = True) -> list[tuple[int, int]]:
    """Report-only: pairs of rows with cos >= threshold (exact dups removed
    above, so these are the 0.999-but-not-identical cases). Collapsing them
    is a modeling choice — it slightly moves the mean."""
    Xh = F.normalize(X, dim=1)
    N, pairs = X.size(0), set()
    for s in tqdm(range(0, N, block), desc="near-dup scan",
                  disable=not show_progress):
        S = Xh[s:s + block] @ Xh.T
        js, ks = (S >= threshold).nonzero(as_tuple=True)
        for j, k in zip(js.tolist(), ks.tolist()):
            a, b = s + j, k
            if a < b:
                pairs.add((a, b))
    return sorted(pairs)


def collapse_near_dups(Xu: Tensor, w: Tensor, threshold: float = 0.999,
                       show_progress: bool = True) -> tuple[Tensor, Tensor]:
    """Transitively union rows with cos >= threshold; keep earliest row per
    component, weight = summed multiplicities. LOSSY — re-check downstream."""
    n = Xu.size(0)
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for a, b in find_near_dups(Xu, threshold=threshold,
                               show_progress=show_progress):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)
    roots = torch.tensor([find(i) for i in range(n)], device=Xu.device)
    uniq_r, inv = torch.unique(roots, return_inverse=True)
    first = torch.full((uniq_r.numel(),), n, dtype=torch.long, device=Xu.device)
    first.scatter_reduce_(0, inv, torch.arange(n, device=Xu.device), reduce="amin")
    wr = torch.zeros(uniq_r.numel(), device=Xu.device)
    wr.index_add_(0, inv, w)
    return Xu[first], wr

# ----------------------------------------------------------------------------
# Leaves functions
# ----------------------------------------------------------------------------

def _kmeans2w(Xh: Tensor, w: Tensor, iters: int = 15, restarts: int = 5,
              gen: torch.Generator | None = None) -> Tensor:
    """Weighted spherical 2-means: dup mass votes with its multiplicity."""
    n, best, best_score = Xh.size(0), None, -float("inf")
    for _ in range(restarts):
        i0 = int(torch.randint(n, (1,), generator=gen))      # CPU draw → int: device-safe
        i1 = int((Xh @ Xh[i0]).argmin())
        C = torch.stack([Xh[i0], Xh[i1]])
        for _ in range(iters):
            assign = (Xh @ C.T).argmax(1)
            for j in (0, 1):
                m = assign == j
                if m.any():
                    C[j] = F.normalize((Xh[m] * w[m, None]).sum(0), dim=0)
        score = float((w * (Xh @ C.T).max(1).values).sum())
        if score > best_score:
            best, best_score = assign.clone(), score
    return best


def build_leaves(X: Tensor, w: Tensor | None = None, idx: Tensor | None = None,
                 tau: float = 0.92, min_size: int = 64, min_balance: float = 0.2,
                 max_depth: int = 16, depth: int = 0,
                 temporal_fallback: bool = True, seed: int | None = None,
                 gen: torch.Generator | None = None, verbose: bool = False
                 ) -> list[tuple[MemoryAggregator, Tensor]]:
    """Recursively split the deduped corpus into consensus-bounded leaves.

    X is the DEDUPED corpus, w its multiplicity weights (or None).
    Leaf consensus/centroids are weight-aware -> identical to building on
    the raw corpus, while k-means and scan cost see each direction once.
    Floors count UNIQUE members, so dup mass can't create sub-floor leaves.

    Args:
        verbose: print size/consensus/stop-reason for each finished leaf.

    Returns:
        List of (leaf aggregator, member index tensor) pairs.
    """
    # CHANGED 1: temporal fallback now fires on ANY failed k-means split
    #            (sliver OR sub-floor) -> no more mega-leaves; continuum
    #            regions become temporal subtrees.
    # CHANGED 2: weight_by_norm=False (your corr(norm, centrality) = -0.30).
    # CHANGED 3: seed param — k-means restarts use the global RNG.
    if seed is not None and depth == 0:
        gen = torch.Generator().manual_seed(seed) if seed is not None else None
    if idx is None:
        idx = torch.arange(X.size(0), device=X.device)
    if w is None:
        w = torch.ones(X.size(0), device=X.device)

    leaf = MemoryAggregator(dim=X.size(1), weight_by_norm=False)     # CHANGED 2
    wi = w[idx]
    if bool((wi == 1).all()):
        leaf.add(X[idx])
    else:
        for i, wi_ in zip(idx.tolist(), wi.tolist()):
            leaf.add(X[i], weight=float(wi_))

    if depth >= max_depth or leaf.consensus() >= tau or idx.numel() < 2 * min_size:
        leaf.stop_reason = ("tau" if leaf.consensus() >= tau else
                            "min_size" if idx.numel() < 2 * min_size else "max_depth")
        if verbose:
            print(f"[leaf] n={idx.numel()} consensus={leaf.consensus():.4f} "
                  f"stop={leaf.stop_reason}")
        return [(leaf, idx)]

    Xh = F.normalize(X[idx], dim=1)
    assign = _kmeans2w(Xh, wi, gen=gen)
    parts = [idx[assign == j] for j in range(2)]
    small = min(p.numel() for p in parts)

    if small >= min_size and small >= min_balance * idx.numel():
        pass                                                  # k-means split is usable
    elif temporal_fallback:                                   # CHANGED 1
        order = idx[idx.argsort()]                            # balanced temporal cut
        parts = [order[:order.numel() // 2], order[order.numel() // 2:]]
    else:
        leaf.stop_reason = "floor" if small < min_size else "unbalanced"
        if verbose:
            print(f"[leaf] n={idx.numel()} consensus={leaf.consensus():.4f} "
                  f"stop={leaf.stop_reason}")
        return [(leaf, idx)]

    out = []
    for p in parts:
        out += build_leaves(X, w, p, tau, min_size, min_balance,
                            max_depth, depth + 1, temporal_fallback,
                            seed, gen, verbose)
    return out

# ----------------------------------------------------------------------------
# Centroids functions
# ----------------------------------------------------------------------------

@torch.no_grad()
def train_routing_centroids(Xu: Tensor, K: int, iters: int = 30, seed: int = 0,
                            verbose: bool = True, show_progress: bool = True) -> Tensor:
    """Global spherical k-means. Returns C (K, d), rows unit-norm."""
    g = torch.Generator().manual_seed(seed)
    Xh = F.normalize(Xu, dim=1)
    C = Xh[torch.randperm(Xu.size(0), generator=g)[:K].to(Xu.device)]
    for it in tqdm(range(iters), desc="spherical k-means",
                   disable=not show_progress):
        sims = Xh @ C.T
        assign = sims.argmax(1)
        sums = torch.zeros_like(C).index_add_(0, assign, Xh)
        empty = torch.bincount(assign, minlength=K) == 0
        if empty.any():                                   # reseed dead cells
            worst = sims.max(1).values.argsort()[:int(empty.sum())]
            sums[empty] = Xh[worst]
        C = F.normalize(sums, dim=1)                      # FIX: was dim=0
        if verbose and (it + 1) % 10 == 0:                # convergence check
            print(f"    it{it+1}: mean cos to assigned centroid = "
                  f"{sims.max(1).values.mean():.4f}")
    return C

# ----------------------------------------------------------------------------
# IVF functions
# ----------------------------------------------------------------------------

@torch.no_grad()
def ivf_report(Xu: Tensor, C: Tensor, R: int = 1, m: int = 10, k: int = 10,
               n_query: int = 2000, seed: int = 0, label: str = "",
               show_progress: bool = True) -> tuple[float, float]:
    """PURE vector-level IVF (no leaf indirection). This is the frontier that
    any leaf-based scheme built on top cannot beat."""
    dev = Xu.device
    Xh = F.normalize(Xu, dim=1)
    g = torch.Generator().manual_seed(seed)
    q = torch.randperm(Xu.size(0), generator=g)[:n_query].to(dev)
    votes = (Xh @ C.T).topk(R, 1).indices                 # point -> top-R cells
    lists = [[] for _ in range(C.size(0))]
    for i, row in enumerate(votes.tolist()):
        for c in row:
            lists[c].append(i)
    pick = (Xh[q] @ C.T).topk(m, 1).indices
    full = Xh[q] @ Xh.T
    full[torch.arange(n_query, device=dev), q] = -float("inf")
    truth = full.topk(k, 1).indices
    hit = tot = 0
    for r in tqdm(range(n_query), desc="ivf scan", disable=not show_progress):
        cand = set()
        for c in pick[r].tolist():
            cand.update(lists[c])
        cand.discard(int(q[r]))
        if not cand:
            continue
        cand = torch.tensor(sorted(cand), device=dev)
        tot += cand.numel()
        kk = min(k, cand.numel())
        got = cand[(Xh[q[r]] @ Xh[cand].T).topk(kk).indices]
        hit += len(set(truth[r].tolist()) & set(got.tolist()))
    rec, scan = hit / (k * n_query), tot / n_query / Xu.size(0)
    print(f"{label:<14} m={m:<3} scan={scan:6.1%} recall@{k}={rec:.3f}")
    return rec, scan


@torch.no_grad()
def ivf_query(q: Tensor, C: Tensor, lists: list[list[int]], Xu: Tensor,
              m: int = 20, k: int = 10, kept: Tensor | None = None) -> Tensor:
    """Single-query IVF search.

    Probes the top-m cells for `q`, gathers all candidate rows, reranks them
    by cosine similarity, and returns the top-k row indices.

    Args:
        q: query embedding, (d,) or (1, d).
        C: routing centroids (K, d), unit-norm rows.
        lists: inverted lists, centroid id -> row indices into `Xu`.
        Xu: corpus the lists index into (the deduped matrix).
        m: number of cells to probe.
        k: number of neighbours to return.
        kept: optional (U,) original-row mapping from `dedupe_exact()`. When
            given, returned indices are mapped back to original corpus rows;
            otherwise indices into `Xu` are returned.

    Returns:
        (k',) long tensor of row indices, k' = min(k, |candidates|); empty if
        the probed cells contain no candidates.
    """
    dev = Xu.device
    qh = F.normalize(q.flatten().unsqueeze(0).to(dev), dim=1)
    pick = (qh @ C.T).topk(m, 1).indices[0]
    cand = torch.tensor(sorted({i for c in pick.tolist() for i in lists[c]}),
                        dtype=torch.long, device=dev)
    if cand.numel() == 0:
        return torch.empty(0, dtype=torch.long, device=dev)
    sims = qh @ F.normalize(Xu[cand], dim=1).T
    top = sims.topk(min(k, cand.numel())).indices[0]
    rows = cand[top]
    if kept is not None:                              # map back to original rows
        rows = kept.to(dev)[rows]
    return rows

# ----------------------------------------------------------------------------
# Routing functions
# ----------------------------------------------------------------------------

@torch.no_grad()
def build_routing_index(Xu: Tensor, leaves: list[tuple[MemoryAggregator, Tensor]],
                        C: Tensor, T: int = 3, R: int = 1) -> list[list[int]]:
    """Register each leaf under its top-T centroids by MAJORITY VOTE of
    members (was: every centroid any member touched -> scan=100%)."""
    Xh = F.normalize(Xu, dim=1)
    votes = (Xh @ C.T).topk(R, 1).indices
    K = C.size(0)
    cent_leaves = [[] for _ in range(K)]
    for li, (_, idx) in enumerate(leaves):
        cnt = torch.bincount(votes[idx].reshape(-1), minlength=K)
        for c in cnt.topk(min(T, K)).indices.tolist():
            if cnt[c] > 0:
                cent_leaves[c].append(li)
    return cent_leaves


@torch.no_grad()
def routing_report_indexed(Xu: Tensor, leaves: list[tuple[MemoryAggregator, Tensor]],
                           C: Tensor, cent_leaves: list[list[int]], m: int = 10,
                           k: int = 10, n_query: int = 2000, seed: int = 0,
                           label: str = "", show_progress: bool = True
                           ) -> tuple[float, float]:
    """Recall/scan of centroid->leaf routing vs exact brute-force truth."""
    dev = Xu.device
    Xh = F.normalize(Xu, dim=1)
    g = torch.Generator().manual_seed(seed)
    q = torch.randperm(Xu.size(0), generator=g)[:n_query].to(dev)
    pick = (Xh[q] @ C.T).topk(m, 1).indices        # (nq, m) centroid ids
    full = Xh[q] @ Xh.T
    full[torch.arange(n_query, device=dev), q] = -float("inf")
    truth = full.topk(k, 1).indices
    hit = tot = 0
    for r in tqdm(range(n_query), desc="routing scan", disable=not show_progress):
        cand = set()
        for c in pick[r].tolist():
            cand.update(cent_leaves[c])
        if not cand:
            continue
        members = torch.cat([leaves[li][1] for li in cand]).to(dev)
        members = members[members != q[r]]
        tot += members.numel()
        kk = min(k, members.numel())
        got = members[(Xh[q[r]] @ Xh[members].T).topk(kk).indices]
        hit += len(set(truth[r].tolist()) & set(got.tolist()))
    rec, scan = hit / (k * n_query), tot / n_query / Xu.size(0)
    if label:
        print(f"{label:<16} m={m:<3} scan={scan:6.1%} recall@{k}={rec:.3f}")
    return rec, scan

# ----------------------------------------------------------------------------
# Demo / self-tests
# ----------------------------------------------------------------------------
def _demo() -> None:
    """Run end-to-end smoke tests / usage demos."""
    torch.manual_seed(0)
    d = 768
    device = 'cuda'
    print("=" * 72)
    print("A) Streaming a 200-member episode; comparing to baselines")
    print("=" * 72)
    topic = F.normalize(torch.randn(d), dim=0)
    X = torch.stack([
        F.normalize(topic + 0.30 * torch.randn(d), dim=0)
        * (0.7 + 0.6 * torch.rand(1)).item()          # varying norms
        for _ in range(200)
    ])
    X = X.to(device)
    Xu = F.normalize(X, dim=1)

    mem = MemoryAggregator(dim=d)
    for x in X:                                        # iterative, step by step
        mem.add(x)
        # v = mem.embedding()                           # available at every step
    e = mem.embedding()

    def report(name: str, v: Tensor) -> None:
        s = Xu @ F.normalize(v, dim=0)
        print(f"  {name:<28} mean_cos={s.mean():.4f}  min_cos={s.min():.4f}")

    report("MemoryAggregator (ours)", e)
    report("plain mean of unit vecs", Xu.mean(0))
    report("naive running sum (norm)", X.sum(0))
    s_ema = torch.zeros(d).to(device)
    for x in X:
        s_ema = 0.05 * x + 0.95 * s_ema                # EMA forgets early items
    report("EMA alpha=0.05", s_ema)
    print("  stats:", {k: (round(v, 4) if isinstance(v, float) else v)
                       for k, v in mem.stats().items()})

    print()
    print("=" * 72)
    print("B) Redundancy flooding: 100 near-duplicates + 5 unique memories")
    print("=" * 72)
    dup = F.normalize(topic + 0.05 * torch.randn(d), dim=0)
    uniques = [F.normalize(torch.randn(d) + 0.2 * torch.randn(d), dim=0)
               for _ in range(5)]
    stream = [dup] * 100 + uniques
    Xu_b = F.normalize(torch.stack(stream), dim=1)
    Xu_b = Xu_b.to(device)

    for boost in (0.0, 1.5):
        m = MemoryAggregator(dim=d, novelty_boost=boost)
        for x in stream:
            m.add(x)
        s = Xu_b[100:] @ m.embedding()                 # similarity to the 5 uniques
        print(f"  novelty_boost={boost:<4} cos-to-uniques: "
              f"mean={s.mean():.4f}  min={s.min():.4f}")
    print("  -> with novelty shaping, unique memories survive duplicate floods.")

    print()
    print("=" * 72)
    print("C) Exact-merge test (hierarchical memory building block)")
    print("=" * 72)
    full = MemoryAggregator(dim=d)
    for x in X:
        full.add(x)
    a, b = MemoryAggregator(dim=d), MemoryAggregator(dim=d)
    for x in X[:100]:
        a.add(x)
    for x in X[100:]:
        b.add(x)
    merged = MemoryAggregator.merge(a, b)
    diff = (merged.embedding() - full.embedding()).abs().max().item()
    print(f"  max |merged - single-pass| = {diff:.3e}   (float64: ~machine eps)")

    print()
    print("=" * 72)
    print("D) Long-stream numerical stability (10k random adds)")
    print("=" * 72)
    m = MemoryAggregator(dim=d, warn_low_consensus=False)
    for _ in tqdm(range(10_000), desc="random adds"):
        m.add(torch.randn(d))
    v = m.embedding()
    print(f"  finite={torch.isfinite(v).all().item()}  ||v||={v.norm().item():.6f}")

    print()
    print("=" * 72)
    print("E) Optional learned single-vector folder (strict 768-d state)")
    print("=" * 72)
    K, S, N = 16, 96, 8
    centers = F.normalize(torch.randn(K, d), dim=1)
    sets = torch.stack([
        F.normalize(centers[s % K] + 0.25 * torch.randn(N, d), dim=1)
        for s in range(S)
    ])
    model = LearnedSetCompressor(d)
    train_set_compressor(model, sets[:-8], epochs=4, batch_size=24)
    with torch.no_grad():
        Xe = sets[-8:]
        Xe = Xe.to(device)
        Xe_h = F.normalize(Xe, dim=-1)
        z = model(Xe)
        ours = torch.einsum('bnd,bd->bn', Xe_h, z).mean()
        base = F.normalize(Xe_h.mean(1), dim=1)
        bl = torch.einsum('bnd,bd->bn', Xe_h, base).mean()
    print(f"  held-out mean member-cosine: learned={ours:.4f} vs mean-baseline={bl:.4f}")


if __name__ == "__main__":
    _demo()