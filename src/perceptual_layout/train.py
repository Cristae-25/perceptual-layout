"""Stage 1: train the observer to recover TRUE adjacency from blurred dots,
across a whole range of sigma (so it becomes sigma-conditional and can be
calibrated to humans later without retraining).

    python -m perceptual_layout.train --n-train 20000 --epochs 20 --out runs/baseline
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time

import torch

from .graphs import collate, make_dataset
from .losses import link_metrics, pair_loss
from .observer import PointNetObserver
from .points import observe

# multipliers on the geometric mean of the training sigma range, for the psychometric check
SIGMA_SCALES = [0.25, 0.5, 1.0, 2.0, 4.0]


def to_dev(b, dev):
    return {k: v.to(dev) for k, v in b.items()}


@torch.no_grad()
def evaluate(model, data, k, sn, se, dev, batch=32, seed=0, spacing=None):
    model.eval()
    rng, g = random.Random(seed), torch.Generator().manual_seed(seed)
    L, Y = [], []
    for i in range(0, len(data), batch):
        b = to_dev(collate(data[i:i + batch], k, sn, se, rng, spacing=spacing), dev)
        logits = model(**observe(b, g))
        L.append(logits[b["pair_mask"]].cpu())
        Y.append(b["A"][b["pair_mask"]].cpu())
    return link_metrics(torch.cat(L), torch.cat(Y))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=20000)
    ap.add_argument("--n-val", type=int, default=1000)
    ap.add_argument("--n-min", type=int, default=10)
    ap.add_argument("--n-max", type=int, default=50)
    ap.add_argument("--k", type=int, default=16, help="edge-dots per edge; keep FIXED across train/calibrate/optimize")
    ap.add_argument("--spacing", type=float, default=None,
                    help="constant edge-dot spacing in normalized units (overrides --k); ~0.027 matches k=16 "
                         "on a median edge. Saved in the checkpoint and reused by calibrate/optimize")
    ap.add_argument("--sigma-node", type=float, nargs=2, default=[0.005, 0.08])
    ap.add_argument("--sigma-edge", type=float, nargs=2, default=[0.002, 0.04])
    ap.add_argument("--samples", type=int, default=2, help="blur samples per step")
    ap.add_argument("--no-sigma-feature", action="store_true",
                    help="ablation: blur the dots but hide sigma from the observer")
    ap.add_argument("--d", type=int, default=256)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs/baseline")
    ap.add_argument("--device", default="auto", help="auto | cuda | mps | cpu")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    dev = args.device
    if dev == "auto":
        dev = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"device: {dev}")

    t0 = time.time()
    train = make_dataset(args.n_train, args.n_min, args.n_max, seed=args.seed)
    val = make_dataset(args.n_val, args.n_min, args.n_max, seed=args.seed + 1)
    print(f"data: {len(train)} train / {len(val)} val in {time.time() - t0:.0f}s")

    model = PointNetObserver(d=args.d, sigma_feature=not args.no_sigma_feature).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sn_range, se_range = tuple(args.sigma_node), tuple(args.sigma_edge)
    sn_mid = (sn_range[0] * sn_range[1]) ** 0.5
    se_mid = (se_range[0] * se_range[1]) ** 0.5
    log = []

    for epoch in range(args.epochs):
        model.train()
        rng.shuffle(train)
        total, nb = 0.0, 0
        for i in range(0, len(train), args.batch):
            b = to_dev(collate(train[i:i + args.batch], args.k, sn_range, se_range, rng, augment=True,
                               spacing=args.spacing), dev)
            loss = sum(pair_loss(model(**observe(b)), b["A"], b["pair_mask"])
                       for _ in range(args.samples)) / args.samples
            opt.zero_grad()
            loss.backward()
            opt.step()
            total, nb = total + loss.item(), nb + 1

        # Psychometric sanity check: AUC MUST fall as blur grows.
        # If it is flat, the observer is ignoring the noise and is not a model of perception.
        curve = {s: evaluate(model, val, args.k, sn_mid * s, se_mid * s, dev, spacing=args.spacing)["auc"]
                 for s in SIGMA_SCALES}
        row = dict(epoch=epoch, train_loss=total / max(nb, 1), auc_vs_sigma_scale=curve)
        log.append(row)
        print(json.dumps(row))
        torch.save(dict(state_dict=model.state_dict(), config=vars(args)), os.path.join(args.out, "observer.pt"))

    with open(os.path.join(args.out, "log.json"), "w") as f:
        json.dump(log, f, indent=2)


if __name__ == "__main__":
    main()
