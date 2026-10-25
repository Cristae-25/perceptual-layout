"""Stage 2: make the observer's adjacency judgements match a HUMAN's.

The observer is frozen and sigma-conditional. We fit only the eye model:
    P(human says "i and j are connected") = sigmoid(observer_logit_ij(sigma_node, sigma_edge) + beta)
beta is a response-bias term (the criterion in signal detection theory):
people, and a class-weighted observer, both have a lean toward "yes" or "no".

Trials file (JSONL), one per line:
    {"pos": [[x,y],...], "edges": [[i,j],...], "query": [i,j], "answer": 0 or 1, "condition": "..."}

    # 1) test the fitting code on synthetic "humans" with a known sigma (do this first)
    python -m perceptual_layout.calibrate --ckpt runs/baseline/observer.pt --simulate
    # 2) fit to real responses
    python -m perceptual_layout.calibrate --ckpt runs/baseline/observer.pt --trials data/human_trials.jsonl
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import random

import numpy as np
import torch
import torch.nn.functional as F

from .graphs import make_sample
from .observer import load_observer
from .points import observe, points_batch


def load_trials(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def query_logits(model, trials, k, sn, se, batch=64, spacing=None):
    """Logit for each trial's query pair under one blur sample. sn/se may be tensors with grad."""
    out = []
    for i in range(0, len(trials), batch):
        tr = trials[i:i + batch]
        pos = [torch.tensor(t["pos"], dtype=torch.float32) for t in tr]
        edges = [torch.tensor(t["edges"], dtype=torch.long).view(-1, 2) for t in tr]
        b = points_batch(pos, edges, k, sn, se, normalize=True, spacing=spacing)
        logits = model(**observe(b))
        q = torch.tensor([t["query"] for t in tr])
        out.append(logits[torch.arange(len(tr)), q[:, 0], q[:, 1]])
    return torch.cat(out)


def nll(model, trials, k, sn, se, beta, samples, spacing=None):
    y = torch.tensor([float(t["answer"]) for t in trials])
    return sum(F.binary_cross_entropy_with_logits(query_logits(model, trials, k, sn, se, spacing=spacing) + beta, y)
               for _ in range(samples)) / samples


def fit(model, trials, k, sn_range, se_range, grid=6, steps=150, samples=4, spacing=None):
    # Coarse grid (robust to the noisy objective) ...
    sns = np.geomspace(*sn_range, grid)
    ses = np.geomspace(*se_range, grid)
    best = None
    with torch.no_grad():
        for sn, se in itertools.product(sns, ses):
            for beta in (-2.0, -1.0, 0.0, 1.0, 2.0):
                v = nll(model, trials, k, float(sn), float(se), beta, samples, spacing).item()
                if best is None or v < best[0]:
                    best = (v, sn, se, beta)
    print(f"grid best: nll={best[0]:.4f} sigma_node={best[1]:.4g} sigma_edge={best[2]:.4g} beta={best[3]}")

    # ... then gradient refinement through the reparameterized blur.
    log_sn = torch.tensor(math.log(best[1]), requires_grad=True)
    log_se = torch.tensor(math.log(best[2]), requires_grad=True)
    beta = torch.tensor(float(best[3]), requires_grad=True)
    opt = torch.optim.Adam([log_sn, log_se, beta], lr=0.03)
    for step in range(steps):
        loss = nll(model, trials, k, log_sn.exp(), log_se.exp(), beta, samples, spacing)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 25 == 0:
            print(f"step {step}: nll={loss.item():.4f} sigma_node={log_sn.exp().item():.4g} "
                  f"sigma_edge={log_se.exp().item():.4g} beta={beta.item():.3f}")
    return dict(sigma_node=log_sn.exp().item(), sigma_edge=log_se.exp().item(), beta=beta.item(), nll=loss.item())


def simulate_trials(model, k, sn_true, se_true, beta_true, num=600, seed=0, spacing=None):
    """Synthetic 'humans' = the observer itself at a known sigma. Balanced yes/no queries."""
    rng = random.Random(seed)
    trials = []
    for t in range(num):
        s = make_sample(rng.randint(8, 25), rng)
        e = s.edges.tolist()
        if t % 2 == 0:
            q = rng.choice(e)
        else:
            N = s.pos.shape[0]
            non = [(i, j) for i in range(N) for j in range(i + 1, N) if s.A[i, j] == 0]
            q = rng.choice(non)
        trials.append(dict(pos=s.pos.tolist(), edges=e, query=list(q), answer=None))
    with torch.no_grad():
        p = torch.sigmoid(query_logits(model, trials, k, sn_true, se_true, spacing=spacing) + beta_true)
    for t, pi in zip(trials, p.tolist()):
        t["answer"] = int(rng.random() < pi)
    return trials


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--trials")
    ap.add_argument("--simulate", action="store_true")
    ap.add_argument("--samples", type=int, default=4)
    ap.add_argument("--out", default="calibration.json")
    args = ap.parse_args()

    torch.manual_seed(0)
    model, cfg = load_observer(args.ckpt)
    sn_range, se_range = cfg["sigma_node"], cfg["sigma_edge"]
    spacing = cfg.get("spacing")  # older checkpoints: fixed k

    if args.simulate:
        truth = dict(sigma_node=0.03, sigma_edge=0.012, beta=-0.5)
        trials = simulate_trials(model, cfg["k"], truth["sigma_node"], truth["sigma_edge"], truth["beta"],
                                 spacing=spacing)
        print("true:", truth)
    else:
        trials = load_trials(args.trials)

    result = fit(model, trials, cfg["k"], sn_range, se_range, samples=args.samples, spacing=spacing)
    print("fitted:", result)
    # NOTE: sigma is only identifiable where it changes behaviour. If nll is flat
    # along one axis, your stimuli do not constrain that sigma: add trials there.
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
