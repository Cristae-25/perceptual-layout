"""Stimuli for the sigma-calibration study. Writes trials.jsonl (answer = null,
to be filled by jsPsych/PsychoPy) and one SVG per trial, all edges the same colour.

    python -m perceptual_layout.stimuli --out data/stimuli

Crossing task: two edges a-b and c-d cross at angle theta. Query a-b (truth yes)
or a-d (truth no). Shallow angles make "cross" and "bounce" look alike.
Occlusion task: does the edge end at node m, or pass behind it? Node m sits at
distance dist from the line; truth depends on which edges exist.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random

ANGLES = [5, 10, 15, 20, 30, 45, 60, 90]
DISTANCES = [0.0, 0.01, 0.02, 0.04, 0.08, 0.16]   # normalized units (layout fits [-1, 1])
SIZE_PX = 600   # rendered width. Unit 1 = 0.85 * SIZE_PX / 2 px, so sigma_px = sigma * 255


def crossing_trial(theta_deg, query_true, rng):
    h = math.radians(theta_deg) / 2
    rot = rng.uniform(0, math.pi)
    def pt(angle):
        a = angle + rot
        return [math.cos(a), math.sin(a)]
    pos = [pt(math.pi + h), pt(h), pt(math.pi - h), pt(-h)]   # a, b, c, d
    edges = [[0, 1], [2, 3]]
    return dict(task="crossing", theta=theta_deg, pos=pos, edges=edges,
                query=[0, 1] if query_true else [0, 3], truth=int(query_true), answer=None)


def occlusion_trial(dist, ends_at_m, rng):
    pos = [[-1.0, 0.0], [1.0, 0.0], [rng.uniform(-0.3, 0.3), dist]]   # i, j, m
    edges = [[0, 2], [1, 2]] if ends_at_m else [[0, 1]]
    return dict(task="occlusion", dist=dist, pos=pos, edges=edges,
                query=[0, 2], truth=int(ends_at_m), answer=None)


def to_svg(trial, size=SIZE_PX, node_r=6, stroke=2):
    def px(p):
        return (size / 2 * (1 + 0.85 * p[0]), size / 2 * (1 - 0.85 * p[1]))
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}">',
             f'<rect width="{size}" height="{size}" fill="white"/>']
    for i, j in trial["edges"]:
        (x1, y1), (x2, y2) = px(trial["pos"][i]), px(trial["pos"][j])
        parts.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="#333" stroke-width="{stroke}"/>')
    for n, p in enumerate(trial["pos"]):
        x, y = px(p)
        hl = n in trial["query"]
        parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{node_r}" fill="{"#d33" if hl else "#333"}"/>')
    parts.append("</svg>")
    return "\n".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/stimuli")
    ap.add_argument("--reps", type=int, default=6, help="repeats per condition")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    os.makedirs(args.out, exist_ok=True)

    trials = []
    for _ in range(args.reps):
        trials += [crossing_trial(a, q, rng) for a in ANGLES for q in (True, False)]
        trials += [occlusion_trial(d, e, rng) for d in DISTANCES for e in (True, False)]
    rng.shuffle(trials)
    with open(os.path.join(args.out, "trials.jsonl"), "w") as f:
        for n, t in enumerate(trials):
            t["id"] = n
            t["svg"] = f"trial_{n:04d}.svg"
            with open(os.path.join(args.out, t["svg"]), "w") as g:
                g.write(to_svg(t))
            f.write(json.dumps(t) + "\n")
    print(f"wrote {len(trials)} trials to {args.out}")


if __name__ == "__main__":
    main()
