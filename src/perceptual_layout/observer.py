"""The surrogate human observer: blurred dots in, adjacency logits out."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def mlp(*dims: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    for i in range(len(dims) - 1):
        layers += [nn.Linear(dims[i], dims[i + 1]), nn.ReLU()]
    return nn.Sequential(*layers[:-1])  # no ReLU on the last layer


class PointNetObserver(nn.Module):
    """Plain PointNet with a segmentation-style node branch and a pair head.

    Baseline only: it has no notion of neighbourhoods, so expect it to work on
    small graphs and degrade as they grow (swap in DGCNN / PointNet++ later;
    keep this forward() signature so the rest of the repo does not change).
    """

    def __init__(self, d: int = 256, d_local: int = 64, sigma_feature: bool = True):
        super().__init__()
        # False: zero the sigma column so the observer must infer blur from the jitter alone.
        # Sigma still acts (and still gets gradients) through perceive().
        self.sigma_feature = sigma_feature
        self.local = mlp(4, 64, d_local)          # shared per-point
        self.deep = mlp(d_local, 128, d)          # shared per-point, then max-pool
        self.node = mlp(d_local + d, d, d)        # node-dot local + global context
        self.pair = mlp(2 * d + 1, d, 1)          # symmetric pair features + distance

    def forward(self, node_pts, node_mask, edge_pts, edge_mask):
        """node_pts (B,N,4), edge_pts (B,M,4), masks True = real. Returns logits (B,N,N)."""
        if not self.sigma_feature:
            node_pts = torch.cat([node_pts[..., :2], torch.zeros_like(node_pts[..., 2:3]), node_pts[..., 3:]], dim=-1)
            edge_pts = torch.cat([edge_pts[..., :2], torch.zeros_like(edge_pts[..., 2:3]), edge_pts[..., 3:]], dim=-1)
        pts = torch.cat([node_pts, edge_pts], dim=1)
        mask = torch.cat([node_mask, edge_mask], dim=1)

        loc = F.relu(self.local(pts))                         # (B,P,dl)
        f = self.deep(loc)                                    # (B,P,d)
        f = f.masked_fill(~mask[..., None], -1e9)             # padded slots never win the max
        g = f.max(dim=1).values                               # (B,d)

        N = node_pts.shape[1]
        h = self.node(torch.cat([loc[:, :N], g[:, None].expand(-1, N, -1)], dim=-1))  # (B,N,d)
        hi, hj = h[:, :, None], h[:, None, :]
        xy = node_pts[..., :2]
        sq = ((xy[:, :, None] - xy[:, None, :]) ** 2).sum(-1, keepdim=True)
        dist = torch.sqrt(sq + 1e-8)                          # eps: safe gradient on the diagonal
        z = torch.cat([hi + hj, (hi - hj).abs(), dist], dim=-1)
        return self.pair(z).squeeze(-1)                       # symmetric by construction


def load_observer(path: str, device="cpu"):
    ckpt = torch.load(path, map_location=device)
    cfg = ckpt["config"]
    model = PointNetObserver(d=cfg["d"], sigma_feature=not cfg.get("no_sigma_feature", False)).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, cfg
