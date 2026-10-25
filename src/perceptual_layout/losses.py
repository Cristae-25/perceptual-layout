"""Recovery loss, layout regularizers, and link-prediction metrics."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def pair_loss(logits, A, pair_mask, pos_weight: float | None = None):
    """Class-weighted BCE over real node pairs i<j. Sparse graphs => mostly negatives."""
    l, y = logits[pair_mask], A[pair_mask]
    if pos_weight is None:
        npos = y.sum().clamp_min(1.0)
        pos_weight = ((y.numel() - npos) / npos).detach()
    return F.binary_cross_entropy_with_logits(l, y, pos_weight=torch.as_tensor(pos_weight, dtype=l.dtype, device=l.device))


def stress(pos, D):
    """Scale-free stress: keeps optimized layouts sensible. D = graph distances (N,N)."""
    N = pos.shape[0]
    i, j = torch.triu_indices(N, N, 1)
    d = torch.sqrt(((pos[i] - pos[j]) ** 2).sum(-1) + 1e-12)
    Dij = D[i, j]
    w = 1.0 / Dij**2
    s = (w * d * Dij).sum() / (w * Dij**2).sum()   # optimal scale, closed form
    return (w * (d - s * Dij) ** 2).sum() / w.sum()


@torch.no_grad()
def link_metrics(logits_flat, y_flat, thr: float = 0.5) -> dict:
    p = torch.sigmoid(logits_flat)
    pred = p >= thr
    tp = (pred & (y_flat > 0.5)).sum().item()
    fp = (pred & (y_flat < 0.5)).sum().item()
    fn = (~pred & (y_flat > 0.5)).sum().item()
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    # rank-based ROC AUC (ties ignored)
    order = p.argsort()
    ranks = torch.empty_like(p)
    ranks[order] = torch.arange(1, len(p) + 1, dtype=p.dtype, device=p.device)
    npos = (y_flat > 0.5).sum().item()
    nneg = len(y_flat) - npos
    auc = float("nan")
    if npos and nneg:
        auc = ((ranks[y_flat > 0.5].sum().item() - npos * (npos + 1) / 2) / (npos * nneg))
    return dict(precision=prec, recall=rec, f1=2 * prec * rec / max(prec + rec, 1e-12), auc=auc)
