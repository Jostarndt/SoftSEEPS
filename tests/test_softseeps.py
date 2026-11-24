"""Run with ``pytest`` or ``python tests/test_softseeps.py``."""
import torch

from softseeps import soft_seeps


def _data(seed):
    g = torch.Generator().manual_seed(seed)
    rain = lambda: torch.rand(32, 64, generator=g).pow(3) * 20 * (torch.rand(32, 64, generator=g) > 0.5)
    return rain(), rain(), 0.1 + 0.75 * torch.rand(32, 64, generator=g), 1 + 5 * torch.rand(32, 64, generator=g)


def _discrete_seeps(fc, ob, p1, t2, dry=0.25):
    """Reference: explicit Rodwell lookup, point by point."""
    cat = lambda x, t: 0 if x <= dry else (1 if x <= t else 2)
    total = 0.0
    for f, o, p, t in zip(fc.flatten().tolist(), ob.flatten().tolist(), p1.flatten().tolist(), t2.flatten().tolist()):
        q = (1 - p) / 3
        S = [[0, 1 / (1 - p), 1 / q + 1 / (1 - p)], [1 / p, 0, 1 / q], [1 / p + 1 / (1 - q), 1 / (1 - q), 0]]
        total += 0.5 * S[cat(f, t)][cat(o, t)]
    return total / fc.numel()


def test_small_tau_gives_discrete_seeps():
    fc, ob, p1, t2 = _data(0)
    assert abs(soft_seeps(fc, ob, p1, t2, tau=1e-5).item() - _discrete_seeps(fc, ob, p1, t2)) < 1e-4


def test_perfect_forecast_is_zero():
    _, ob, p1, t2 = _data(1)
    assert soft_seeps(ob, ob, p1, t2, tau=1e-5).item() < 1e-4


def test_invalid_points_ignored_and_gradient_finite():
    fc, ob, p1, t2 = _data(2)
    ob[0, :5] = float("nan")
    p1[1, :5] = 0.95
    t2[2, :5] = float("nan")
    fc.requires_grad_(True)
    loss = soft_seeps(fc, ob, p1, t2, tau=1.0)
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(fc.grad).all()
    assert (fc.grad[:3, :5] == 0).all()


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
