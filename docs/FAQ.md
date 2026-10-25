# FAQ and vocabulary

A running record of terms and questions that come up while working on this project.
Newest questions go at the bottom of their section.

---

## Vocabulary

| Term | Meaning in this project |
|---|---|
| **Layout** | Node positions for a graph, an `(N, 2)` array. Different layout algorithms draw the same graph differently. |
| **Normalized units** | Every layout is centred and rescaled so the farthest node is at ±1, so all drawings fit in a box from −1 to 1 (2 units wide). σ is measured in these units. |
| **Node-dot** | One point per node, at the node's position. Row format `[x, y, σ, type=0]`. Node-dots come first, in node-index order. |
| **Edge-dot** | One of `k` points sampled evenly along an edge (endpoints excluded). Row format `[x, y, σ, type=1]`. Edge-dots are shuffled and carry no edge identity, so the observer has to infer which nodes they connect. |
| **k** | Edge-dots per edge (default 16). It must stay the same across training, calibration and optimization. |
| **σ (sigma)** | The blur level: the standard deviation of the Gaussian jitter added to each dot. Larger σ means blurrier vision. |
| **σ_node / σ_edge** | Separate blur levels for node-dots and edge-dots: roughly, how precisely a person locates a node and how well they can follow a thin line. |
| **Jitter / `perceive()`** | Applying the blur: `x' = x + σ·ε`, with ε ~ N(0, 1). This is the model's version of the eye. |
| **Reparameterization trick** | Writing the noise as `σ·ε` with ε drawn independently, so gradients can flow through the noise to both the positions and σ. |
| **Observer** | The neural network that takes blurred dots and outputs, for every node pair (i, j), a logit that they are connected. It's a stand-in for a human reader. |
| **PointNet** | A network that reads an unordered set of points. A shared MLP is applied to each point, then max-pooling gives a global feature that doesn't depend on point order. It's the baseline observer here. |
| **Deep Sets** | The general theory behind PointNet: functions on sets can be written as "per-element transform, symmetric pooling, then transform". |
| **DGCNN** | Dynamic Graph CNN. It adds local neighbourhoods (k-nearest-neighbour edges between points) to PointNet. Planned for Week 12 to handle larger graphs. |
| **Pair head** | The part of the observer that combines the features of node i and node j (plus their distance) into one connection logit. It's symmetric by construction. |
| **Adjacency matrix (A)** | An `N × N` 0/1 matrix where `A[i, j] = 1` if i and j share an edge. The observer is trained to rebuild it. |
| **Recovery loss** | How badly the observer rebuilds A from the blurred dots (class-weighted binary cross-entropy). Low recovery loss means a readable layout. |
| **AUC** | Area under the ROC curve: the probability that the observer ranks a true edge above a non-edge. 0.5 is chance, 1.0 is perfect. |
| **Psychometric check** | Plotting AUC against σ. AUC must fall as σ grows; a flat curve means the observer is ignoring the blur. |
| **Calibration** | Freezing the observer and fitting σ_node, σ_edge and a bias β so its mistakes match human mistakes. |
| **β (bias)** | A response-bias term from signal detection theory: how willing someone is to say "connected" when unsure. |
| **Stress** | A classic layout-quality measure: how well drawn distances match graph distances. Used as a regularizer during optimization. |
| **Rome graphs** | A standard benchmark of about 11.5k sparse graphs (10–110 nodes), stored as GraphML with no coordinates. Used here for evaluation, not training. |
| **Smoke test** | A tiny, fast run that only checks the pipeline works end to end, not that results are good. |
| **MPS** | Metal Performance Shaders, PyTorch's backend for Apple-silicon GPUs. |

---

## Project and pipeline

**Q: What is the first step of the project?**
Week 1 of `docs/PLAN.md`: get the environment running, run `pytest`, and run a smoke training. The data loader, the blur, and a PointNet observer were already scaffolded. What had never been done was running the PyTorch parts end to end.
Status (2026-10-24): done. 6/6 tests pass, and the smoke run trains and gives a falling AUC-vs-σ curve.

**Q: Do we blur pixels (an image) or jitter dots?**
We jitter dots. A layout becomes a point cloud (node-dots plus edge-dots), and blur means Gaussian jitter on those points. PointNet reads point sets, not images. This keeps everything differentiable with respect to node positions and σ.

**Q: Where do the Rome graphs come in?**
In evaluation (Week 8 onward), comparing optimized layouts against spring, KK, stress and the other baselines. Training uses synthetic graphs (ER, BA, tree, grid, SBM), where the true adjacency is known and data is unlimited. Rome files have topology only, so we lay them out ourselves (e.g. Kamada–Kawai). The local copy is `data/rome`, a link to `../RL_Graphs/rome`.

**Q: Why not just pass σ as an input feature on clean positions?**
With exact coordinates available, the network could learn to ignore σ and see perfectly, and calibration would then have nothing to act on. So σ is both a feature and a real limit on what the observer can see: the dots really are jittered.

**Q: How do I make the pipeline and blur figures?**
`python -m perceptual_layout.visualize --out runs/figures --log runs/<run>/log.json` writes:
1. `1_gallery.png`: every graph type × layout method, dots over the true drawing (σ = 0)
2. `2_sigma_grid.png`: one graph under a grid of σ_node × σ_edge values
3. `3_sigma_sweep.png`: both σ scaled together (0×, 0.25×, 1×, 4× the middle of the training range)
4. `4_rome.png`: the same sweep on a Rome graph
5. `5_auc_vs_sigma.png`: the psychometric curve from a training log

---

## Blur and σ

**Q: What do 0.005 and 0.002 mean? Is that 0.2% noise?**
No. σ isn't a percentage of each coordinate. It is the **standard deviation of an additive Gaussian shift**, measured in normalized units. Each dot is moved independently in x and y:

```
x' = x + σ · ε_x,   y' = y + σ · ε_y,   ε ~ N(0, 1)
```

- The shift is the same size everywhere: a node near the centre moves as much as one near the edge of the box.
- About 68% of dots move less than σ along each axis, and 95% move less than 2σ.
- The typical total distance a dot moves is about 1.25σ (the mean of a 2-D Gaussian's radius, σ·√(π/2)).

0.005 and 0.002 are the *lower* (sharpest) ends of the training ranges:

| | σ range | as a share of drawing width (2 units) | on the 600 px study screen |
|---|---|---|---|
| node | 0.005 – 0.08 | 0.25% – 4% | ~1 px – 20 px |
| edge | 0.002 – 0.04 | 0.1% – 2% | ~0.5 px – 10 px |

**The pixel mapping:** `stimuli.py` renders 600 px wide and maps 1 normalized unit to 0.85 × 600 / 2 = 255 px. So **σ in pixels = σ × 255**. For example, σ_node = 0.02 is about 5 px.

During training, each graph in a batch gets its own σ, drawn **log-uniformly** from these ranges (so each factor-of-2 band of blur is equally likely), and fresh noise on every step. One network therefore learns every blur level. Calibration later picks the σ that matches people.

**Q: Why does edge blur hurt readability more than node blur?**
An edge is only visible as a line of dots, and with k = 16 the median gap between neighbouring edge-dots is about 0.027 units (about 0.011 on short edges, about 0.082 on long ones). Once σ_edge gets close to that gap, the dots stop lining up and the line breaks up. Node blur moves only N dots. Edge blur scrambles E × k dots, and those are the only evidence that an edge exists. Long edges are the most fragile because their dots are already the most spread out.

**Q: Why is it harder to recover the adjacency matrix with more noise?**
Blurred edge-dots drift off the segment between their two endpoints, and blurred node-dots drift away from where their edges end. The observer can no longer tell which trail connects which pair of nodes. This happens especially where edges cross at shallow angles or pass close to unrelated nodes. Falling AUC as σ grows is the intended behaviour: it mirrors how people misread cluttered drawings.

---

## Setup and running

**Q: How do I set up the environment?**
```bash
python -m venv .venv && source .venv/bin/activate
pip install -e . pytest
pytest -q
```

**Q: Which device does training use?**
`train.py --device auto` (the default) picks CUDA, then Apple GPU (MPS), then CPU. On this Mac, MPS gives the same results as CPU and is about 1.5× faster (68 s vs 104 s for the 2k-graph, 3-epoch smoke run).

**Q: What did the first smoke run show?**
2k graphs, 3 epochs: validation AUC rose each epoch (0.725 → 0.758) and fell as σ grew. The direction is right, but it's still far from the Week 4 target of AUC > 0.9 at the lowest σ, which needs the full 20k-graph run.
