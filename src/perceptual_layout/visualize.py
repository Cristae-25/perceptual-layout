"""Figures that show the data pipeline and the blur, for checking by eye.

    python -m perceptual_layout.visualize --out runs/figures --log runs/smoke_cpu/log.json

1_gallery.png     every family x layout, dots drawn over the true drawing
2_sigma_grid.png  one graph under sigma_node x sigma_edge (what the observer sees)
3_sigma_sweep.png same graph, both sigmas scaled together across the training range
4_rome.png        a Rome graph through the same pipeline
5_auc_vs_sigma.png  psychometric curve from a train.py log (if --log is given)
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random
from collections import Counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import torch

from .graphs import FAMILIES, LAYOUTS, layout, load_graph_file, make_dataset, random_graph, to_sample
from .points import layout_to_points, normalize_layout, perceive

NODE_C, EDGE_C, LINE_C = "#d1495b", "#00798c", "#bbbbbb"


def draw(ax, s, k, sigma_node, sigma_edge, seed=0, title=None, show_drawing=True):
    """True drawing (grey lines) under the perceived dots, in normalized units."""
    pos = normalize_layout(s.pos)
    n, e = layout_to_points(pos, s.edges, k, sigma_node, sigma_edge, shuffle=False)
    g = torch.Generator().manual_seed(seed)
    n, e = perceive(n, g), perceive(e, g)
    if show_drawing:
        for i, j in s.edges.tolist():
            ax.plot(*pos[[i, j]].T, color=LINE_C, lw=0.6, zorder=0)
    ax.scatter(*e[:, :2].T, s=2, color=EDGE_C, lw=0, zorder=1)
    ax.scatter(*n[:, :2].T, s=12, color=NODE_C, lw=0, zorder=2)
    ax.set_xlim(-1.15, 1.15)
    ax.set_ylim(-1.15, 1.15)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=8)


def gallery(out, k, n_nodes, seed):
    rng = random.Random(seed)
    fig, axes = plt.subplots(len(FAMILIES), len(LAYOUTS), figsize=(2 * len(LAYOUTS), 2 * len(FAMILIES)))
    for r, fam in enumerate(FAMILIES):
        while True:  # random_graph picks the family itself; draw until we get the one we want
            G, f = random_graph(n_nodes, rng)
            if f == fam:
                break
        for c, name in enumerate(LAYOUTS):
            s = to_sample(G, layout(G, name, seed), fam, name)
            draw(axes[r, c], s, k, 0.0, 0.0)
            if r == 0:
                axes[r, c].set_title(name, fontsize=9)
        axes[r, 0].set_ylabel(f"{fam}\nN={G.number_of_nodes()} E={G.number_of_edges()}", fontsize=9)
    fig.suptitle(f"Data pipeline: node-dots (red) and k={k} edge-dots (teal) over the true drawing, sigma = 0")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "1_gallery.png"), dpi=160)
    plt.close(fig)


def sigma_grid(out, s, k, node_vals, edge_vals):
    fig, axes = plt.subplots(len(edge_vals), len(node_vals), figsize=(2.2 * len(node_vals), 2.2 * len(edge_vals)))
    for r, se in enumerate(edge_vals):
        for c, sn in enumerate(node_vals):
            draw(axes[r, c], s, k, sn, se, show_drawing=False)
            if r == 0:
                axes[r, c].set_title(f"sigma_node = {sn:.3g}", fontsize=9)
        axes[r, 0].set_ylabel(f"sigma_edge = {se:.3g}", fontsize=9)
    fig.suptitle(f"What the observer sees: {s.family} graph, {s.layout} layout, N={s.pos.shape[0]}")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "2_sigma_grid.png"), dpi=160)
    plt.close(fig)


def sigma_sweep(out, s, k, sn_mid, se_mid, scales, name="3_sigma_sweep.png", label=""):
    fig, axes = plt.subplots(2, len(scales), figsize=(2.4 * len(scales), 5))
    for c, m in enumerate(scales):
        draw(axes[0, c], s, k, sn_mid * m, se_mid * m, title=f"{m:g}x  (node {sn_mid * m:.3f}, edge {se_mid * m:.3f})")
        draw(axes[1, c], s, k, sn_mid * m, se_mid * m, show_drawing=False)
    axes[0, 0].set_ylabel("dots over drawing", fontsize=9)
    axes[1, 0].set_ylabel("observer input", fontsize=9)
    fig.suptitle(f"Blur sweep{label}: multiples of the training-range midpoint")
    fig.tight_layout()
    fig.savefig(os.path.join(out, name), dpi=160)
    plt.close(fig)


def auc_curve(out, log_path):
    log = json.load(open(log_path))
    fig, ax = plt.subplots(figsize=(4.5, 3.2))
    for row in log:
        xs = [float(x) for x in row["auc_vs_sigma_scale"]]
        ax.plot(xs, list(row["auc_vs_sigma_scale"].values()), marker="o", ms=3, label=f"epoch {row['epoch']}")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("sigma (multiple of training midpoint)")
    ax.set_ylabel("val AUC")
    ax.set_title("Psychometric check: AUC must fall as blur grows", fontsize=9)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "5_auc_vs_sigma.png"), dpi=160)
    plt.close(fig)


def dataset_stats(n, n_min, n_max, k, seed):
    """Family/layout mix and edge-dot density (dots per unit length), short vs long edges."""
    data = make_dataset(n, n_min, n_max, seed=seed)
    print("family:", dict(Counter(s.family for s in data)))
    print("layout:", dict(Counter(s.layout for s in data)))
    lengths = torch.cat([
        (normalize_layout(s.pos)[s.edges[:, 0]] - normalize_layout(s.pos)[s.edges[:, 1]]).norm(dim=1)
        for s in data if s.edges.numel()
    ])
    q = torch.quantile(lengths, torch.tensor([0.1, 0.5, 0.9]))
    print(f"edge length (normalized) p10/p50/p90: {q[0]:.3f} / {q[1]:.3f} / {q[2]:.3f}")
    print(f"edge-dot spacing at k={k}: p10 {q[0] / (k + 1):.4f}, p50 {q[1] / (k + 1):.4f}, p90 {q[2] / (k + 1):.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/figures")
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--n", type=int, default=25, help="nodes in the example graphs")
    ap.add_argument("--sigma-node", type=float, nargs=2, default=[0.005, 0.08], help="match train.py")
    ap.add_argument("--sigma-edge", type=float, nargs=2, default=[0.002, 0.04], help="match train.py")
    ap.add_argument("--rome", default="data/rome")
    ap.add_argument("--log", help="train.py log.json for the AUC-vs-sigma plot")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    dataset_stats(1000, 10, 50, args.k, args.seed)
    gallery(args.out, args.k, args.n, args.seed)

    sn_lo, sn_hi = args.sigma_node
    se_lo, se_hi = args.sigma_edge
    sn_mid, se_mid = (sn_lo * sn_hi) ** 0.5, (se_lo * se_hi) ** 0.5
    G, fam = random_graph(args.n, random.Random(args.seed + 1))
    s = to_sample(G, layout(G, "kamada_kawai", args.seed), fam, "kamada_kawai")
    sigma_grid(args.out, s, args.k, [0.0, sn_lo, sn_mid, sn_hi], [0.0, se_lo, se_mid, se_hi])
    sigma_sweep(args.out, s, args.k, sn_mid, se_mid, [0, 0.25, 1, 4])

    files = sorted(glob.glob(os.path.join(args.rome, "*.graphml")))
    if files:
        # Rome filenames are grafo<id>.<n_nodes>.graphml; pick a mid-sized one
        mid = [f for f in files if 25 <= int(os.path.basename(f).split(".")[1]) <= 40]
        G = load_graph_file(random.Random(args.seed).choice(mid or files))
        r = to_sample(G, layout(G, "kamada_kawai", args.seed), "rome", "kamada_kawai")
        sigma_sweep(args.out, r, args.k, sn_mid, se_mid, [0, 0.25, 1, 4], "4_rome.png",
                    f" on a Rome graph (N={G.number_of_nodes()}, E={G.number_of_edges()})")
    else:
        print(f"no GraphML files in {args.rome}; skipping Rome figure")

    if args.log:
        auc_curve(args.out, args.log)
    print(f"figures written to {args.out}")


if __name__ == "__main__":
    main()
