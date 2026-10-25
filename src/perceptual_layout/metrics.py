"""Classic, model-free readability metrics. Use them to catch the optimizer
gaming the frozen observer (high recovery score, bad drawing)."""
from __future__ import annotations

import math

import numpy as np


def _orient(a, b, c):
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def crossings(pos: np.ndarray, edges: np.ndarray):
    """Returns (crossing count, list of crossing angles in degrees, 0-90)."""
    count, angles = 0, []
    E = len(edges)
    for a in range(E):
        i, j = edges[a]
        for b in range(a + 1, E):
            k, l = edges[b]
            if len({i, j, k, l}) < 4:
                continue
            P, Q, R, S = pos[i], pos[j], pos[k], pos[l]
            if _orient(P, Q, R) * _orient(P, Q, S) < 0 and _orient(R, S, P) * _orient(R, S, Q) < 0:
                count += 1
                v1, v2 = Q - P, S - R
                c = abs(v1 @ v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-12)
                angles.append(math.degrees(math.acos(min(1.0, c))))
    return count, angles


def node_edge_occlusions(pos: np.ndarray, edges: np.ndarray, tol: float = 0.02):
    """Count (node, edge) pairs where a node lies within tol of an edge it is not on.
    tol is in the same units as pos (use normalized layouts)."""
    count = 0
    for i, j in edges:
        a, b = pos[i], pos[j]
        ab = b - a
        L2 = ab @ ab + 1e-12
        t = np.clip(((pos - a) @ ab) / L2, 0.0, 1.0)
        d = np.linalg.norm(pos - (a + t[:, None] * ab), axis=1)
        d[[i, j]] = np.inf
        count += int((d < tol).sum())
    return count


def summary(pos: np.ndarray, edges: np.ndarray, shallow_deg: float = 20.0) -> dict:
    p = pos - pos.mean(0)
    p = p / (np.abs(p).max() + 1e-12)
    c, ang = crossings(p, edges)
    return dict(
        crossings=c,
        shallow_crossings=sum(a < shallow_deg for a in ang),
        min_crossing_angle=min(ang) if ang else None,
        node_edge_occlusions=node_edge_occlusions(p, edges),
    )
