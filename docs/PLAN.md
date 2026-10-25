# Project plan

**Goal.** Train a point-based observer whose adjacency judgements match a human's, by
calibrating the Gaussian blur (σ), then use it as a differentiable readability loss for
layout optimization.

**Research questions**

1. Does recovery loss under blur track known readability factors (crossings, crossing angle, node-edge occlusion) *without* being told about them?
2. Can two σ values plus a bias term reproduce human error patterns on simple stimuli?
3. Do layouts optimized against the calibrated observer score better with humans than stress / force-directed baselines?

Timeline: 16 working weeks plus a two-week winter buffer, starting **Mon 26 Oct 2026**.
Adjust dates to your term; keep the order.

---

## Timeline

| Wk | Dates | Phase | Work | Exit criterion |
|---|---|---|---|---|
| 1 | Oct 26 – 30 | 0 Setup | Repo, env, CI running `pytest`. Read Deep Sets, PointNet, PointNet++. Agree scope with advisor. **Start IRB paperwork** for the human study. | Tests pass on your machine/GPU |
| 2 | Nov 2 – 6 | 1 Data | Plot dots over drawings for every layout type (build step 1). Check edge-dot density on short vs long edges. | Dots visibly match drawings |
| 3 | Nov 9 – 13 | 1 Data | Generate 20k train / 1k val graphs (10–50 nodes). Log family and layout mix. Submit IRB. | Dataset cached to disk |
| 4 | Nov 16 – 20 | 2 Observer | Train PointNet baseline. Tune pos-weighting. | Val AUC > 0.9 at the lowest σ |
| 5 | Nov 23 – 27 | 2 Observer | *Short week (Thanksgiving).* Psychometric check: AUC vs σ, AUC vs layout type. | AUC falls monotonically with σ; random/circular layouts score lower than KK/spring |
| 6 | Nov 30 – Dec 4 | 2 Observer | Gradient checks (build step 4): move one node by hand, plot loss; confirm smoothness. Run `calibrate --simulate`. | σ recovered within ~20% on synthetic humans |
| 7 | Dec 7 – 11 | 3 Optimize | `optimize.py` on random graphs from bad initial layouts. Sweep `w_stress`. | Recovery loss drops, crossings do not rise |
| 8 | Dec 14 – 18 | 3 Optimize | Rome graphs: compare against spring, KK, stress, (GD)², and the existing GD+SLS crossing pipeline on crossings, min angle, occlusions. **Mid-project check-in.** | Results table + figure |
| – | Dec 21 – Jan 1 | Buffer | Winter break. Catch-up only. | |
| 9 | Jan 4 – 8 | 4 Human | Build jsPsych task from `stimuli.py` output. Pilot with 3–5 people (if IRB allows). Fix timing, instructions, display size. | Pilot data loads into `calibrate.py` |
| 10 | Jan 11 – 15 | 4 Human | Main calibration study (Prolific or in-person). Target ≈ 20–30 participants × 168 trials. | Clean trials file |
| 11 | Jan 18 – 22 | 4 Human | Fit σ_node, σ_edge, β. Compare observer vs human accuracy curves per angle and per distance. | Curves overlap within CI on held-out trials |
| 12 | Jan 25 – 29 | 5 Scale | Swap in DGCNN (`torch_geometric.nn.DynamicEdgeConv`), same `forward` signature. Retrain across σ range. | Beats PointNet on 50–100 node graphs |
| 13 | Feb 1 – 5 | 5 Scale | Concatenated batching (`batch` vector) for 100–500 nodes. Re-run Rome comparison. | Runs on Rome without OOM |
| 14 | Feb 8 – 12 | 5 Scale | Adversarial round: add optimized layouts to training data, retrain, re-optimize. | Gap between observer score and classic metrics shrinks |
| 15 | Feb 15 – 19 | 6 Validate | Final human evaluation: optimized vs baseline layouts, task = "are i and j connected?", accuracy + time. | Data collected |
| 16 | Feb 22 – 26 | 6 Validate | Analysis, write-up, figures, clean repo, tag release. | Report / paper draft |

**Critical path:** IRB approval gates weeks 9–11 and 15. Submit by week 3. If it slips,
swap phases 4 and 5 (scaling does not need humans).

---

## Milestones

- **M1 (end wk 3):** pipeline from graph to dots, verified visually.
- **M2 (end wk 6):** trained σ-conditional observer; gradients and σ-recovery verified.
- **M3 (end wk 8):** first optimized layouts and a baseline comparison on Rome graphs.
- **M4 (end wk 11):** observer calibrated to human data.
- **M5 (end wk 16):** human-validated results and write-up.

## Experiments to log (one row each in W&B / TensorBoard)

- AUC vs σ curve, per graph family and per layout type
- Ablation: σ as feature only vs feature + jitter vs jitter only (`train.py --no-sigma-feature`); expect feature-only to ignore σ
- Edge-dot density: fixed `k` per edge (long edges sparser) vs constant spacing by length; edge-dot dropout as augmentation
- `k` (dots per edge): 8 / 16 / 32
- Pair-head input: with vs without node distance (does it shortcut to "close nodes are connected"?)
- Optimization: `w_stress` sweep; crossings, shallow crossings, occlusions before/after
- Calibration: NLL surface over (σ_node, σ_edge) to check both are identifiable

## Risks and mitigations

| Risk | Sign | Mitigation |
|---|---|---|
| Observer ignores blur | Flat AUC-vs-σ curve | Jitter is on by default; raise `--samples`; check ablation |
| Distance shortcut | High AUC from distance alone | Ablate distance input; add layouts where neighbours are far apart (random, circular) |
| Optimizer exploits observer | Recovery improves, crossings rise | Warning in `optimize.py`; adversarial retraining (wk 14); report classic metrics always |
| Plain PointNet too weak on larger graphs | AUC drops with N | Expected; DGCNN in wk 12. Idea to test: a "trail probe" feature sampling edge-dot density along segment i→j |
| σ not identifiable | Flat NLL along one axis | Add stimuli in the ambiguous range (shallow angles, small distances) |
| Human data noisy / small | Wide CIs | Fit per-participant and pooled; more reps of the ambiguous conditions |
| IRB delay | No approval by wk 9 | Reorder phases; pilot only with what the protocol allows |

## Decisions already made

- Positions normalized to [-1, 1]; σ in those units. Display mapping in `stimuli.py`.
- Blur is real (reparameterized jitter), not only a feature.
- Observer is trained on *true* adjacency; humans enter only through σ and β (and later optional fine-tuning).
- Padding with masks first; concatenated batching later.
- No T-Nets; centre/scale + random rotation augmentation instead.

## Open questions for the advisor

- Which baselines matter most for the comparison (stress, (GD)², the crossing-minimization pipeline)?
- Target venue and therefore final evaluation design?
- Is a per-participant σ (individual differences) in scope, or only a population σ?
