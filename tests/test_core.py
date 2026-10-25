"""Build-order steps 1 and 4 as tests. Run: pytest -q"""
import random

import torch

from perceptual_layout.graphs import collate, make_dataset
from perceptual_layout.losses import pair_loss
from perceptual_layout.observer import PointNetObserver
from perceptual_layout.points import layout_to_points, observe, perceive, points_batch


def small():
    return make_dataset(4, 8, 12, seed=0)


def test_point_shapes():
    s = small()[0]
    n, e = layout_to_points(s.pos, s.edges, k=5)
    assert n.shape == (s.pos.shape[0], 4)
    assert e.shape == (s.edges.shape[0] * 5, 4)
    assert (n[:, 3] == 0).all() and (e[:, 3] == 1).all()


def test_gradient_reaches_positions_and_sigma():
    s = small()[0]
    model = PointNetObserver(d=32)
    pos = s.pos.clone().requires_grad_(True)
    sig = torch.tensor(0.02, requires_grad=True)
    b = points_batch([pos], [s.edges], 8, sig, sig, A_list=[s.A])
    loss = pair_loss(model(**observe(b)), b["A"], b["pair_mask"])
    loss.backward()
    assert pos.grad is not None and torch.isfinite(pos.grad).all() and pos.grad.abs().sum() > 0
    assert sig.grad is not None and torch.isfinite(sig.grad)


def test_edge_dot_order_does_not_matter():
    s = small()[0]
    model = PointNetObserver(d=32).eval()
    b = points_batch([s.pos], [s.edges], 8, 0.0, 0.0)  # sigma 0: deterministic
    out1 = model(**observe(b))
    perm = torch.randperm(b["edge_pts"].shape[1])
    out2 = model(b["node_pts"], b["node_mask"], b["edge_pts"][:, perm], b["edge_mask"][:, perm])
    assert torch.allclose(out1, out2, atol=1e-5)


def test_padding_does_not_change_output():
    data = small()
    model = PointNetObserver(d=32).eval()
    b1 = points_batch([data[0].pos], [data[0].edges], 8, 0.0, 0.0)
    b2 = points_batch([data[0].pos, data[1].pos], [data[0].edges, data[1].edges], 8, 0.0, 0.0)
    N = data[0].pos.shape[0]
    o1 = model(**observe(b1))[0]
    o2 = model(**observe(b2))[0, :N, :N]
    assert torch.allclose(o1, o2, atol=1e-5)


def test_output_symmetric():
    b = collate(small(), 8, 0.0, 0.0, random.Random(0))
    out = PointNetObserver(d=32).eval()(**observe(b))
    assert torch.allclose(out, out.transpose(1, 2), atol=1e-5)


def test_blur_is_real():
    pts = torch.zeros(1, 100, 4)
    pts[..., 2] = 0.1
    assert perceive(pts)[..., :2].std() > 0.05


def test_no_sigma_feature_hides_sigma_but_keeps_its_gradient():
    s = small()[0]
    model = PointNetObserver(d=32, sigma_feature=False).eval()
    b_lo = points_batch([s.pos], [s.edges], 8, 0.01, 0.01)
    b_hi = points_batch([s.pos], [s.edges], 8, 0.05, 0.05)
    # same dot positions, different sigma column: output must not change
    assert torch.allclose(model(**b_lo), model(**b_hi), atol=1e-6)

    sig = torch.tensor(0.02, requires_grad=True)
    b = points_batch([s.pos], [s.edges], 8, sig, sig, A_list=[s.A])
    pair_loss(model(**observe(b)), b["A"], b["pair_mask"]).backward()
    assert sig.grad is not None and torch.isfinite(sig.grad) and sig.grad != 0


def test_fixed_k_matches_linspace():
    s = small()[0]
    _, e = layout_to_points(s.pos, s.edges, k=5, sigma_node=0.0, sigma_edge=0.0, shuffle=False)
    t = torch.linspace(0, 1, 7)[1:-1].view(1, 5, 1)
    a, b = s.pos[s.edges[:, 0]], s.pos[s.edges[:, 1]]
    expected = ((1 - t) * a[:, None] + t * b[:, None]).reshape(-1, 2)
    assert torch.allclose(e[:, :2], expected, atol=1e-6)


def test_constant_spacing_density_and_gradient():
    pos = torch.tensor([[0.0, 0.0], [0.2, 0.0], [0.0, 1.0]], requires_grad=True)
    edges = torch.tensor([[0, 1], [0, 2]])  # lengths 0.2 and 1.0
    _, e = layout_to_points(pos, edges, k=16, sigma_node=0.0, sigma_edge=0.0, shuffle=False, spacing=0.05)
    on_short = e[:, 1].abs() < 1e-6
    assert on_short.sum() == 3 and (~on_short).sum() == 19  # round(L/0.05) - 1 interior dots
    gaps = e[~on_short, 1].diff()
    assert torch.allclose(gaps, torch.full_like(gaps, 0.05), atol=1e-5)
    e[:, :2].sum().backward()
    assert pos.grad is not None and pos.grad.abs().sum() > 0


def test_spacing_threads_through_batch():
    data = small()
    b = collate(data, 8, 0.0, 0.0, random.Random(0), spacing=0.05)
    out = PointNetObserver(d=32).eval()(**observe(b))
    assert out.shape == b["A"].shape and torch.isfinite(out).all()
