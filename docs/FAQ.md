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
| **PointNet++** | A hierarchical PointNet: *set abstraction* (farthest-point sampling, ball-query grouping, mini-PointNet per group) builds local features; *feature propagation* interpolates them back to every point. |
| **Segmentation branch** | The part of a PointNet that outputs a feature for *each* point (local feature plus global vector), as opposed to one label for the whole cloud. We use it to get node embeddings. |
| **Pair head** | The part of the observer that combines the features of node i and node j (plus their distance) into one connection logit. It's symmetric by construction. |
| **Broadcasting** | Tensor arithmetic that expands size-1 dimensions, e.g. `(B,N,1,d)` with `(B,1,N,d)` gives `(B,N,N,d)`. It's how one MLP scores all N² pairs at once. |
| **Mask / pair_mask** | Boolean tensors marking real (non-padded) rows. `pair_mask` = real nodes, upper triangle i < j; only those pairs enter the loss and metrics. |
| **Adjacency matrix (A)** | An `N × N` 0/1 matrix where `A[i, j] = 1` if i and j share an edge. The observer is trained to rebuild it. |
| **Recovery loss** | How badly the observer rebuilds A from the blurred dots (class-weighted binary cross-entropy). Low recovery loss means a readable layout. |
| **AUC** | Area under the ROC curve: the probability that the observer ranks a true edge above a non-edge. 0.5 is chance, 1.0 is perfect. |
| **Psychometric check** | Plotting AUC against σ. AUC must fall as σ grows; a flat curve means the observer is ignoring the blur. |
| **Calibration** | Freezing the observer and fitting σ_node, σ_edge and a bias β so its mistakes match human mistakes. |
| **β (bias)** | A response-bias term from signal detection theory: how willing someone is to say "connected" when unsure. |
| **Stress** | A classic layout-quality measure: how well drawn distances match graph distances. Used as a regularizer during optimization. |
| **Rome graphs** | A standard benchmark of about 11.5k sparse graphs (10–110 nodes), stored as GraphML with no coordinates. Used here for evaluation, not training. |
| **Shortcut / leakage** | When a model scores well by using information a person wouldn't have, or a cue that's only true of the training data (e.g. "close nodes are connected"), rather than solving the real task. |
| **Ablation** | Removing or changing one input or component to see how much it matters (e.g. the pair head without the distance input). |
| **Monte Carlo sample** | A random draw used to approximate an expectation. Jittered dots are a Monte Carlo sample of the Gaussian-blurred drawing. |
| **Differentiable rasterizer** | A renderer (e.g. diffvg) that turns shapes into pixels while still giving gradients to the shape coordinates. It's the pixel-based alternative to dots. |
| **Edge-dot density** | Dots per unit length along an edge. With a fixed `k` it's k / length, so long edges look sparser than short ones. |
| **Dropout (augmentation)** | Randomly removing inputs (here, edge-dots) during training so the model doesn't over-rely on any of them. Turned off at evaluation. |
| **Arc-length sampling** | Placing dots at a fixed distance apart along each edge, so the number of dots grows with edge length. |
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

**Q: How does a PointNet-style model output an adjacency matrix? Is it a final layer padded into a 2-D matrix?**
No. There's no output layer of size N×N; it couldn't work, because N changes from graph to graph. The model works in three steps (`observer.py`):

1. **Per-point features.** Every dot (node and edge) goes through a shared MLP, then max-pooling over all dots gives one global vector `g` that summarizes the whole picture.
2. **Per-node embeddings** (as in PointNet's segmentation branch). For each node-dot, combine its local feature with `g`, giving an embedding `h_i` of shape `(B, N, d)`. Node-dots come first in node-index order, so row i is node i.
3. **Pair head.** Apply one small MLP to *every* pair by broadcasting: `[h_i + h_j, |h_i − h_j|, dist(i, j)] → logit_ij`. Broadcasting `(B, N, 1, d)` against `(B, 1, N, d)` produces the `(B, N, N)` matrix directly. Using `+` and `|−|` makes it symmetric by construction.

Padding is used only for **batching**. Graphs in a batch are padded to the largest N, and masks mark the real rows. The padded slots get −∞ before the max-pool so they're never picked. The loss only looks at `pair_mask` (real nodes, upper triangle i < j), so padded entries never count.
This is the same idea as link prediction in graph neural networks: learn node embeddings, then score pairs.

**Q: How would PointNet++ (or DGCNN) change this?**
Only step 1 changes. The node embeddings and the pair head stay the same, and so does the `forward(node_pts, node_mask, edge_pts, edge_mask) → (B, N, N)` signature.
- **PointNet++:** *set abstraction* layers repeatedly sample centre points (farthest-point sampling), group nearby dots (ball query), and run a mini-PointNet on each group, building local-to-global features. *Feature propagation* layers then interpolate those features back to every original dot. We read out the first N rows (the node-dots) as `h_i`. What it adds is that a node's feature can describe the trail of edge-dots leaving it in a given direction, which plain PointNet's single global max-pool mostly loses.
- **DGCNN** (Week 12 in the plan): instead of fixed groups, each layer links every dot to its k nearest neighbours in feature space and convolves over those links (`EdgeConv`). It gets a similar local-structure benefit and is easy to add with `torch_geometric`.
Why it matters: plain PointNet is expected to degrade as graphs grow, because one global vector can't hold "which trail goes from i to j" for hundreds of dots.

**Q: The shared MLP gets `[x, y, σ, type]` for each dot. Isn't passing raw x, y cheating?**
The x, y are the **blurred** coordinates, not the clean ones. Every call to the model (`train.py`, `calibrate.py`, `optimize.py`) goes through `observe()`, which runs `perceive()` first. The node-to-node distance in the pair head is also computed from the blurred node-dots.
Positions themselves aren't cheating: a drawing *is* positions, and pixels encode positions too. Cheating means a shortcut to the answer that a person wouldn't have. The ones to watch:

| Possible leak | Status |
|---|---|
| Clean coordinates, with σ only as a label | Prevented: the jitter is real, so the information is actually lost. |
| Edge identity or order (which dots belong to which edge) | Prevented: edge-dots are shuffled and carry no edge ID, and the shared MLP sees each dot on its own. |
| Node index | Prevented: the index is never an input. Row order only lets us read out pair (i, j), and the model is permutation-equivariant. |
| **Distance shortcut** ("close nodes are connected") | **Real risk.** In spring/KK layouts, neighbours really are close. Mitigations: random/circular layouts in training, plus the planned ablation of the pair head without `dist`. |
| Edge count (the number of real edge-dots is E·k) | Minor: it gives a density hint. A person also sees roughly how many lines there are. |

**Q: Should the Gaussian noise be an input factor, or should we train on (blurred) pixels?**
Right now it's both a feature and real: σ is in each dot's row *and* the dots are actually jittered. The jitter is what matters. σ as a feature only tells the model how much to trust what it sees.
- **Jittered dots ≈ a sampled blurred image.** The expected density of Gaussian-jittered dots is the drawing convolved with a Gaussian, which is exactly a blurred image. So the point version is a Monte Carlo sample of the pixel version.
- **Why points for now:** cheap, exact gradients to node positions and σ, no resolution or aliasing problems with thin lines, and it scales to large graphs. Reading an N×N matrix out of an image would still need node locations (sampling features at each node's pixel).
- **When pixels would be worth it:** as a comparison or check that the results aren't an artifact of the point representation. A differentiable rasterizer (e.g. diffvg) plus Gaussian blur plus a CNN.
- **Ablations to run** (σ-input arms): (a) σ feature only, clean dots: expected to ignore σ; (b) feature + jitter: the current default; (c) **jitter only, no σ feature**: the model has to infer the blur from the dots, the way a person never "reads" their own σ. Calibration still works in (c), because σ acts through the jitter.

**Q: How do I run the "jitter only" ablation?**
`python -m perceptual_layout.train ... --no-sigma-feature`. The dots are still blurred, but the σ column is zeroed inside the model, so it has to judge the blur from the dots themselves. The setting is saved in the checkpoint, and `load_observer()` restores it, so `calibrate.py` and `optimize.py` need no extra flag. σ still gets gradients through the jitter, so calibration still works (covered by a test in `tests/test_core.py`).

**Q: Should we randomly drop some edge-dots, e.g. keep only half? Doesn't a high density of dots make an edge easier to trace?**
Yes, density makes tracing easier, and right now it varies for a reason that has nothing to do with perception. Every edge gets the same `k` dots, so **dot density = k / edge length**: short edges are dense solid trails, and long edges are sparse dotted lines (median spacing about 0.011 on short edges vs 0.082 on long ones). On a screen, every line has the same ink per unit length. So the observer may find long edges harder than a person would, just because of how we sample.
Two different fixes, for two different problems:
- **Constant spacing (fixes the density artifact):** sample dots by arc length, e.g. one dot every 0.02 units, so `k` varies per edge. This makes the dots look like real ink. Side effects: the number of dots grows with total edge length, so memory varies per graph; and node movements change the number of dots, which is a discrete jump the gradients can't see. Using a fixed `k` per graph but distributing it by length avoids that.
- **Random dropout (augmentation, robustness):** keep each edge-dot with probability p (e.g. p drawn from [0.5, 1] per graph) during *training only*. This stops the observer relying on exact counts or spacing and acts as a rough model of partial occlusion or attention lapses. It doesn't fix the density artifact, because it thins short and long edges equally. Keep all dots at evaluation, calibration and optimization, or the observer sees something different from what humans see.
Recommendation: first measure whether it matters, by checking AUC on short vs long edges after the full training run. If long edges lag, switch to constant spacing. Add dropout later if the observer looks fragile.

**Q: How can I eye-check the sampling options before training?**
Run `visualize` and look at `6_sampling_kk.png`, `6_sampling_circular.png` and `6_sampling_random.png`. Rows are the sampling options: fixed k (current), fixed k + 50% dropout, constant spacing, and constant spacing + 50% dropout. Constant spacing is tuned to give the same total number of dots as fixed k. Columns are blur levels (0×, 0.5×, 1×, 2× and 4× the middle of the training range). Two node pairs are circled: **A** is the longest true edge (the sparsest trail) and **B** is the closest pair that is *not* connected (it looks connected). The answer key is at the bottom.
The rule behind it: if a case is hard for a person to read, the observer should find it hard too. If the observer finds something easy that a person can't see, it's probably using a shortcut.
What the figures showed (2026-10-24):
- **Circular layout** shows the difference most. With fixed k, the long chords across the circle are faint and the short arcs along the rim are solid, so a person mostly sees the rim. With constant spacing, every chord is a visible line, which matches how a real drawing looks.
- **Dropout at 0.5** makes every edge look dashed. At 2× blur and above, trails mostly disappear in all four rows; dropout reaches that point about one blur step earlier.
- **B pairs** (close but not connected) look connected at almost every setting, which is the distance shortcut in action. The model needs layouts where neighbours are far apart to learn not to rely on distance.
Constant spacing is now a training option (`train.py --spacing 0.027`); dropout is still preview-only (`edge_dots()` in `visualize.py`).

**Q: How do I train with constant spacing?**
`python -m perceptual_layout.train ... --spacing 0.027`. Each edge gets `round(length / spacing) − 1` interior dots (at least 1), placed evenly, so every edge has the same dot density. 0.027 gives about the same number of dots as k = 16 on a median-length edge; long edges get more dots and short edges fewer. Leaving out `--spacing` keeps the old fixed k. The value is saved in the checkpoint, and `calibrate.py` and `optimize.py` read it from there. Dot *counts* carry no gradient, but dot *positions* still do. During layout optimization the count changes in jumps as an edge stretches, and the gradient can't see those jumps.

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
