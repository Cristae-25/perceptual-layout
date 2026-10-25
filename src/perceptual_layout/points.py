"""Layout -> point set -> perceived (blurred) point set.

Everything here is differentiable with respect to node positions AND sigma,
so the same code serves training, layout optimization, and sigma calibration.

Row format for every dot: [x, y, sigma, type], type 0 = node-dot, 1 = edge-dot.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

NODE, EDGE = 0.0, 1.0


def normalize_layout(pos: torch.Tensor) -> torch.Tensor:
    """Centre at the origin and scale so the largest |coordinate| is 1.

    Sigma values are expressed in these normalized units, so a layout's
    absolute scale never changes what the observer can see.
    """
    c = pos - pos.mean(dim=0, keepdim=True)
    return c / c.abs().max().clamp_min(1e-8)


def random_rotation(pos: torch.Tensor, generator: torch.Generator | None = None) -> torch.Tensor:
    """Data augmentation: readability is (roughly) rotation invariant."""
    theta = torch.rand((), generator=generator).item() * 2 * math.pi
    c, s = math.cos(theta), math.sin(theta)
    R = pos.new_tensor([[c, -s], [s, c]])
    return pos @ R.T


def edge_dot_counts(pos, edges, k: int = 16, spacing: float | None = None) -> torch.Tensor:
    """Interior dots per edge, (E,) long.

    spacing None: k on every edge, so density is k / length (long edges look sparser).
    spacing s: about one dot every s units (normalized), so every edge has the same
    density, like ink on a screen. Counts are integers and carry no gradient; the dot
    positions still do.
    """
    if spacing is None:
        return torch.full((edges.shape[0],), k, dtype=torch.long, device=pos.device)
    with torch.no_grad():
        length = (pos[edges[:, 1]] - pos[edges[:, 0]]).norm(dim=1)
        return (torch.round(length / spacing).long() - 1).clamp_min(1)


def layout_to_points(pos, edges, k: int = 16, sigma_node=0.03, sigma_edge=0.01, shuffle: bool = True,
                     spacing: float | None = None):
    """pos: (N,2), may require grad. edges: (E,2) long.
    sigma_node / sigma_edge: float or 0-d tensor (a tensor lets you fit sigma).
    spacing: None = k dots per edge; a float = constant spacing (see edge_dot_counts).

    Returns (node_pts (N,4), edge_pts (E*k,4)). Node-dots are in node-index
    order (so pair (i,j) can be read out); edge-dots are shuffled and carry
    no edge identity.
    """
    N = pos.shape[0]
    ones_n = pos.new_ones(N, 1)
    node_pts = torch.cat([pos, ones_n * sigma_node, ones_n * NODE], dim=1)

    if edges.numel() == 0:
        return node_pts, pos.new_zeros(0, 4)

    # n_e interior samples per edge, evenly spaced, endpoints excluded (endpoints are node-dots)
    counts = edge_dot_counts(pos, edges, k, spacing)
    eid = torch.repeat_interleave(torch.arange(edges.shape[0], device=pos.device), counts)
    start = torch.cumsum(counts, 0) - counts
    step = torch.arange(eid.shape[0], device=pos.device) - start[eid] + 1   # 1..n_e within each edge
    t = (step.to(pos.dtype) / (counts[eid] + 1).to(pos.dtype))[:, None]
    a, b = pos[edges[eid, 0]], pos[edges[eid, 1]]
    xy = (1 - t) * a + t * b
    ones_m = pos.new_ones(xy.shape[0], 1)
    edge_pts = torch.cat([xy, ones_m * sigma_edge, ones_m * EDGE], dim=1)
    if shuffle:
        edge_pts = edge_pts[torch.randperm(edge_pts.shape[0], device=pos.device)]
    return node_pts, edge_pts


def perceive(pts: torch.Tensor, generator: torch.Generator | None = None) -> torch.Tensor:
    """Apply the eye model: x' = x + sigma * eps, eps ~ N(0, I).

    This is the reparameterization trick, so gradients flow to both x and
    sigma. Padded rows have sigma = 0 and are left untouched.

    Why not just pass sigma as a feature on clean positions? Because then the
    observer can read exact positions and learn to ignore sigma entirely:
    the "blur" would be a label, not a limitation. Jitter makes the
    information loss real. Average the loss over a few samples per step.
    """
    if pts.numel() == 0:
        return pts
    eps = torch.randn(pts.shape[:-1] + (2,), generator=generator, dtype=pts.dtype).to(pts.device)
    return torch.cat([pts[..., :2] + pts[..., 2:3] * eps, pts[..., 2:]], dim=-1)


def _stack_padded(rows: list[torch.Tensor]):
    L = max(1, max(r.shape[0] for r in rows))
    out = torch.stack([F.pad(r, (0, 0, 0, L - r.shape[0])) for r in rows])
    mask = torch.zeros(len(rows), L, dtype=torch.bool, device=out.device)
    for b, r in enumerate(rows):
        mask[b, : r.shape[0]] = True
    return out, mask


def points_batch(pos_list, edges_list, k=16, sigma_node=0.03, sigma_edge=0.01,
                 normalize=True, A_list=None, spacing=None):
    """Build a padded batch from several layouts.

    sigma_node / sigma_edge: a single value, or a list with one value per graph.
    Returns a dict with node_pts, node_mask, edge_pts, edge_mask and, if
    A_list is given, A and pair_mask (upper triangle, real nodes only).
    """
    B = len(pos_list)
    sn = sigma_node if isinstance(sigma_node, (list, tuple)) else [sigma_node] * B
    se = sigma_edge if isinstance(sigma_edge, (list, tuple)) else [sigma_edge] * B
    nodes, edges_ = [], []
    for b in range(B):
        p = normalize_layout(pos_list[b]) if normalize else pos_list[b]
        n, e = layout_to_points(p, edges_list[b], k, sn[b], se[b], spacing=spacing)
        nodes.append(n)
        edges_.append(e)
    node_pts, node_mask = _stack_padded(nodes)
    edge_pts, edge_mask = _stack_padded(edges_)
    batch = dict(node_pts=node_pts, node_mask=node_mask, edge_pts=edge_pts, edge_mask=edge_mask)

    if A_list is not None:
        Nmax = node_pts.shape[1]
        A = torch.stack([F.pad(a, (0, Nmax - a.shape[0], 0, Nmax - a.shape[0])) for a in A_list])
        triu = torch.ones(Nmax, Nmax, dtype=torch.bool, device=A.device).triu(1)
        batch["A"] = A.to(node_pts)
        batch["pair_mask"] = node_mask[:, :, None] & node_mask[:, None, :] & triu
    return batch


def observe(batch: dict, generator: torch.Generator | None = None) -> dict:
    """The observer's input: blurred dots plus masks. Never the adjacency."""
    return dict(
        node_pts=perceive(batch["node_pts"], generator),
        node_mask=batch["node_mask"],
        edge_pts=perceive(batch["edge_pts"], generator),
        edge_mask=batch["edge_mask"],
    )
