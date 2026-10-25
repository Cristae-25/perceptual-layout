"""Stage 3: freeze the observer and move NODES to make the graph recoverable.

    python -m perceptual_layout.optimize --ckpt runs/baseline/observer.pt --n 30 --plot out.png
    python -m perceptual_layout.optimize --ckpt runs/baseline/observer.pt --graph data/rome/grafo10000.38.graphml
"""
from __future__ import annotations

import argparse
import json
import random

import networkx as nx
import numpy as np
import torch

from .graphs import layout, load_graph_file, random_graph, to_sample
from .losses import pair_loss, stress
from .metrics import summary
from .observer import load_observer
from .points import normalize_layout, observe, points_batch


def recovery_loss(model, pos, sample, k, sn, se, samples=4):
    b = points_batch([pos], [sample.edges], k, sn, se, normalize=False, A_list=[sample.A])
    return sum(pair_loss(model(**observe(b)), b["A"], b["pair_mask"]) for _ in range(samples)) / samples


def optimize_layout(model, sample, k, sn, se, steps=500, lr=5e-3, w_stress=0.1, samples=4, log_every=50):
    G = nx.Graph(sample.edges.tolist())
    N = sample.pos.shape[0]
    G.add_nodes_from(range(N))
    D = torch.tensor(nx.floyd_warshall_numpy(G, nodelist=range(N)), dtype=torch.float32)
    pos = normalize_layout(sample.pos.clone()).requires_grad_(True)
    opt = torch.optim.Adam([pos], lr=lr)
    history = []
    for step in range(steps):
        p = normalize_layout(pos)  # fixed box every pass, as in training
        rec = recovery_loss(model, p, sample, k, sn, se, samples)
        st = stress(p, D)
        loss = rec + w_stress * st
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % log_every == 0 or step == steps - 1:
            history.append(dict(step=step, recovery=rec.item(), stress=st.item()))
            print(json.dumps(history[-1]))
    return normalize_layout(pos.detach()), history


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--graph", help="GraphML (e.g. Rome) or edge list; default: random graph")
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--init", default="spring_unconverged")
    ap.add_argument("--sigma-node", type=float, help="default: calibrated/mid value from training config")
    ap.add_argument("--sigma-edge", type=float)
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--lr", type=float, default=5e-3)
    ap.add_argument("--w-stress", type=float, default=0.1)
    ap.add_argument("--samples", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--plot")
    ap.add_argument("--save", default="optimized_layout.json")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    model, cfg = load_observer(args.ckpt)
    sn = args.sigma_node or (cfg["sigma_node"][0] * cfg["sigma_node"][1]) ** 0.5
    se = args.sigma_edge or (cfg["sigma_edge"][0] * cfg["sigma_edge"][1]) ** 0.5

    G = load_graph_file(args.graph) if args.graph else random_graph(args.n, random.Random(args.seed))[0]
    sample = to_sample(G, layout(G, args.init, args.seed))
    edges_np = sample.edges.numpy()

    before = summary(sample.pos.numpy(), edges_np)
    pos, hist = optimize_layout(model, sample, cfg["k"], sn, se, args.steps, args.lr, args.w_stress, args.samples)
    after = summary(pos.numpy(), edges_np)
    print("before:", before)
    print("after: ", after)
    if after["crossings"] > before["crossings"]:
        print("WARNING: crossings went UP. The optimizer may be exploiting the observer.")

    with open(args.save, "w") as f:
        json.dump(dict(pos=pos.tolist(), edges=edges_np.tolist(), before=before, after=after,
                       history=hist, sigma_node=sn, sigma_edge=se), f)

    if args.plot:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        for ax, P, title in [(axes[0], normalize_layout(sample.pos).numpy(), f"initial ({before['crossings']} crossings)"),
                             (axes[1], pos.numpy(), f"optimized ({after['crossings']} crossings)")]:
            for i, j in edges_np:
                ax.plot(*P[[i, j]].T, color="0.4", lw=0.8, zorder=1)
            ax.scatter(*P.T, s=25, color="k", zorder=2)
            ax.set_title(title)
            ax.set_aspect("equal")
            ax.axis("off")
        fig.savefig(args.plot, dpi=150, bbox_inches="tight")


if __name__ == "__main__":
    main()
