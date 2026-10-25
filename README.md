# perceptual-layout

A cheap, differentiable stand-in for a human reader of graph drawings.

A layout is turned into a set of blurred dots (one per node, a trail per edge). A neural
observer tries to rebuild the adjacency matrix from those dots alone. Its success is a
readability score, and because every step is plain tensor math, gradients flow from that
score back to the node positions, so we can optimize layouts for human readability.

The project's central claim: **readability = recoverability under human-like blur.**
The Gaussian blur width σ is the model of the eye. Calibrating σ against a small human
study is what makes the observer's adjacency judgements match a person's.

## How "match a human" works

We never try to learn human judgements end to end from a few hundred labels. Instead:

| Stage | What is learned | From what | Script |
|---|---|---|---|
| 1. Train | Observer weights | Synthetic graphs with known adjacency, blurred at *many* σ values | `train.py` |
| 2. Calibrate | σ_node, σ_edge, response bias β | Human answers to "are i and j connected?" | `calibrate.py` |
| 3. Optimize | Node positions | Frozen observer at the calibrated σ | `optimize.py` |

Stage 1 makes the observer σ-conditional: one network that knows how well it *can* see at
any blur level. Stage 2 freezes it and finds the σ at which its mistakes line up with human
mistakes (shallow crossings, nodes sitting on edges). β plays the role of the criterion in
signal detection theory. Optional stage 2b, once there is enough human data: fine-tune the
observer on human answers as soft labels.

### One deliberate change from the design document

The design doc suggests passing σ as an input feature on clean positions. We do *not* do
that alone: with exact coordinates available, the network can learn to ignore σ and see
perfectly, and then calibration has nothing to act on. Here σ is both a feature and a real
limitation: `perceive()` jitters every dot by `σ·ε` (reparameterization trick), and the loss
is averaged over a few blur samples. This stays differentiable in both positions and σ.
`train.py` checks this every epoch by printing AUC against blur level, which must fall as
σ grows.

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e . pytest
pytest -q                                                     # shapes, gradients, invariances

python -m perceptual_layout.train --n-train 2000 --epochs 3 --out runs/smoke     # smoke test
python -m perceptual_layout.visualize --out runs/figures --log runs/smoke/log.json # pipeline + blur figures
python -m perceptual_layout.train --n-train 20000 --epochs 20 --out runs/baseline

python -m perceptual_layout.optimize --ckpt runs/baseline/observer.pt --n 30 --plot before_after.png
python -m perceptual_layout.calibrate --ckpt runs/baseline/observer.pt --simulate  # parameter recovery
python -m perceptual_layout.stimuli --out data/stimuli                            # human-study stimuli
```

## Layout

```
src/perceptual_layout/
  points.py     layout -> dots, blur (perceive), padded batching
  graphs.py     random graphs (ER, BA, tree, grid, SBM) x good/bad layouts; GraphML loader (Rome)
  observer.py   PointNet baseline: shared MLP -> masked max-pool -> node branch -> symmetric pair head
  losses.py     class-weighted pair BCE, stress, precision/recall/AUC
  metrics.py    model-free checks: crossings, crossing angles, node-edge occlusions
  train.py      stage 1, with the AUC-vs-sigma sanity curve
  calibrate.py  stage 2, grid search + gradient refinement of sigma and beta
  optimize.py   stage 3, recovery loss + stress, before/after metrics and plot
  visualize.py  dots-over-drawing gallery, sigma grid/sweep (synthetic + Rome), AUC-vs-sigma plot
  stimuli.py    crossing-angle and node-occlusion trials (JSONL + SVG) for jsPsych/PsychoPy
tests/          build-order checks as unit tests
docs/PLAN.md    timeline, milestones, risks
```

## Conventions

- Positions are normalized (centred, max |coordinate| = 1) before entering the observer; all σ are in those units.
- Edge-dots carry no edge identity. Node-dots come first, in node-index order.
- `k` (dots per edge) must stay fixed across training, calibration and optimization.
- Any model swap (DGCNN, PointNet++) keeps `forward(node_pts, node_mask, edge_pts, edge_mask) -> (B,N,N) logits`.

## Status

Early scaffold. The code compiles and the torch-free parts (graph generation, metrics,
stimuli) have been run; the PyTorch pieces have not been run end to end yet. Run `pytest`
first. See [docs/PLAN.md](docs/PLAN.md).

## Key references

Qi et al. 2017 (PointNet, PointNet++); Zaheer et al. 2017 (Deep Sets); Wang et al. 2019 (DGCNN);
Ahmed et al. 2020 ((GD)²); Gansner et al. 2004 (stress majorization); Purchase 1997;
Green & Swets 1966 (signal detection theory). Full list in the design document.
