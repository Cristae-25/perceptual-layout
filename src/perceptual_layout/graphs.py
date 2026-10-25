"""Synthetic graphs with known adjacency, drawn in good and deliberately bad layouts."""
from __future__ import annotations

import math
import random
from dataclasses import dataclass

import networkx as nx
import numpy as np
import torch

from .points import points_batch, random_rotation

FAMILIES = ["er", "ba", "tree", "grid", "sbm"]
LAYOUTS = ["spring", "spring_unconverged", "kamada_kawai", "circular", "random", "perturbed"]


@dataclass
class GraphSample:
    pos: torch.Tensor    # (N,2) float32
    edges: torch.Tensor  # (E,2) long, i < j
    A: torch.Tensor      # (N,N) float32, symmetric
    family: str
    layout: str


def random_graph(n: int, rng: random.Random):
    """Returns (G with integer labels 0..N-1, family). Keeps the largest component."""
    for _ in range(20):
        family = rng.choice(FAMILIES)
        seed = rng.randrange(2**31)
        if family == "er":
            G = nx.gnp_random_graph(n, min(1.0, rng.uniform(3.0, 6.0) / max(n - 1, 1)), seed=seed)
        elif family == "ba":
            G = nx.barabasi_albert_graph(n, rng.choice([1, 2]), seed=seed)
        elif family == "tree":
            G = nx.random_labeled_tree(n, seed=seed)
        elif family == "grid":
            w = max(2, round(n ** 0.5))
            G = nx.grid_2d_graph(w, max(2, n // w))
        else:
            c = rng.choice([2, 3, 4])
            sizes = [n // c] * c
            sizes[0] += n - sum(sizes)
            p = [[0.4 if i == j else 0.03 for j in range(c)] for i in range(c)]
            G = nx.stochastic_block_model(sizes, p, seed=seed)
        G = G.subgraph(max(nx.connected_components(G), key=len)).copy()
        if G.number_of_nodes() >= 4:
            return nx.convert_node_labels_to_integers(nx.Graph(G)), family
    raise RuntimeError("could not generate a connected graph")


def layout(G: nx.Graph, name: str, seed: int) -> np.ndarray:
    if name == "spring":
        p = nx.spring_layout(G, seed=seed)
    elif name == "spring_unconverged":  # the "poorly converged" case of Figure 1
        p = nx.spring_layout(G, iterations=random.Random(seed).randint(1, 8), seed=seed)
    elif name == "kamada_kawai":
        p = nx.kamada_kawai_layout(G)
    elif name == "circular":
        p = nx.circular_layout(G)
    elif name == "random":
        p = nx.random_layout(G, seed=seed)
    elif name == "perturbed":
        p = nx.spring_layout(G, seed=seed)
    else:
        raise ValueError(name)
    pos = np.array([p[i] for i in range(G.number_of_nodes())], dtype=np.float32)
    if name == "perturbed":
        pos += np.random.default_rng(seed).normal(0, 0.15 * pos.std(), pos.shape).astype(np.float32)
    return pos


def to_sample(G: nx.Graph, pos: np.ndarray, family="custom", layout_name="custom") -> GraphSample:
    e = sorted((min(u, v), max(u, v)) for u, v in G.edges() if u != v)
    edges = torch.tensor(e, dtype=torch.long).view(-1, 2)
    N = G.number_of_nodes()
    A = torch.zeros(N, N)
    if len(e):
        A[edges[:, 0], edges[:, 1]] = 1
        A[edges[:, 1], edges[:, 0]] = 1
    return GraphSample(torch.as_tensor(pos, dtype=torch.float32), edges, A, family, layout_name)


def make_sample(n: int, rng: random.Random) -> GraphSample:
    G, fam = random_graph(n, rng)
    name = rng.choice(LAYOUTS)
    return to_sample(G, layout(G, name, rng.randrange(2**31)), fam, name)


def make_dataset(num: int, n_min: int, n_max: int, seed: int = 0) -> list[GraphSample]:
    rng = random.Random(seed)
    return [make_sample(rng.randint(n_min, n_max), rng) for _ in range(num)]


def load_graph_file(path: str) -> nx.Graph:
    """GraphML (e.g. Rome graphs) or a whitespace edge list."""
    G = nx.read_graphml(path) if path.endswith((".graphml", ".xml")) else nx.read_edgelist(path)
    G = nx.Graph(G)
    G = G.subgraph(max(nx.connected_components(G), key=len)).copy()
    return nx.convert_node_labels_to_integers(G)


def _pick_sigma(spec, rng: random.Random):
    """Float -> fixed. (lo, hi) -> log-uniform, so the observer learns every blur level."""
    if isinstance(spec, (tuple, list)):
        lo, hi = spec
        return math.exp(rng.uniform(math.log(lo), math.log(hi)))
    return spec


def collate(samples, k, sigma_node, sigma_edge, rng: random.Random, augment=False):
    pos = [random_rotation(s.pos) if augment else s.pos for s in samples]
    sn = [_pick_sigma(sigma_node, rng) for _ in samples]
    se = [_pick_sigma(sigma_edge, rng) for _ in samples]
    return points_batch(pos, [s.edges for s in samples], k, sn, se,
                        normalize=True, A_list=[s.A for s in samples])
