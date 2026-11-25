"""SoftSEEPS: a differentiable relaxation of the SEEPS precipitation score (Rodwell et al., 2010)."""
import torch


def soft_seeps(forecast, observation, p1, t2, dry_threshold=0.25, tau=1.0, weights=None,
               soft_observation=False):
    """Mean SoftSEEPS of ``forecast`` against ``observation``.

    forecast, observation: precipitation, any shape ``[..., H, W]``.
    p1: climatological probability of a dry day, shape ``[H, W]`` or ``[..., H, W]``.
    t2: climatological light/heavy threshold, shape ``[H, W]`` or ``[..., H, W]``.
    dry_threshold: dry/light threshold (e.g. 0.25 mm for 24 h accumulations).
    tau: smoothing parameter, width of the sigmoids; tau -> 0 gives discrete SEEPS.
    weights: optional point weights, shape ``[H, W]``, or ``[H, 1]`` for cos(latitude).
    soft_observation: relax the observed category with the same sigmoids (no gradient
        through the observation); by default the observation keeps its hard category.

    Forecast, observation, thresholds and tau must use the same unit. Points with
    ``p1`` outside [0.1, 0.85] or non-finite inputs are ignored.
    """
    p1 = torch.broadcast_to(torch.as_tensor(p1, dtype=forecast.dtype, device=forecast.device), forecast.shape)
    t2 = torch.broadcast_to(torch.as_tensor(t2, dtype=forecast.dtype, device=forecast.device), forecast.shape)
    valid = (torch.isfinite(forecast) & torch.isfinite(observation) & torch.isfinite(t2)
             & (p1 >= 0.1) & (p1 <= 0.85))
    # Fill invalid points before computing, so that NaNs cannot reach the gradient.
    forecast = torch.where(valid, forecast, 0.0)
    observation = torch.where(valid, observation, 0.0)
    p1, t2 = torch.where(valid, p1, 0.5), torch.where(valid, t2, 1.0)

    # Rodwell penalty matrix (rows: forecast category, columns: observed category).
    p3 = (1 - p1) / 3
    a, b, c, d = 1 / p1, 1 / (1 - p1), 1 / p3, 1 / (1 - p3)
    z = torch.zeros_like(p1)
    S = 0.5 * torch.stack([torch.stack([z, b, b + c], -1),
                           torch.stack([a, z, c], -1),
                           torch.stack([a + d, d, z], -1)], -2)

    # Soft categories (dry, light, heavy) of the forecast; hard or soft ones of the observation.
    def soft_category(x):
        dry, heavy = torch.sigmoid((dry_threshold - x) / tau), torch.sigmoid((x - t2) / tau)
        return torch.stack([dry, (1 - dry - heavy).clamp(min=0), heavy], -1)

    f = soft_category(forecast)
    if soft_observation:
        o = soft_category(observation).detach()
    else:
        o_cat = (observation > dry_threshold).long() + (observation > t2).long()
        o = torch.nn.functional.one_hot(o_cat, 3).to(f.dtype)

    score = torch.einsum("...i,...ij,...j->...", f, S, o)
    w = valid.to(score.dtype) if weights is None else valid * torch.as_tensor(weights, device=score.device)
    return (score * w).sum() / w.sum()
