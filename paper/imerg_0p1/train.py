"""0.1 deg IMERG decoder on a frozen ArchesWeather-M: MSLE + lambda * SoftSEEPS, both area-weighted.
SoftSEEPS relaxes the observed category as well (soft_observation=True), as in the paper runs.

    python paper/imerg_0p1/train.py --imerg_dir IMERG/ --seeps_dir imerg_seeps/ \
        --geoarches geoarches/ --backbone archesweather-m-seed1_checkpoint.ckpt --lam 2
    python paper/imerg_0p1/train.py ... --lam 0                               # MSLE-only
    python paper/imerg_0p1/train.py ... --static static_0p1.npy --lam 4       # decoder + static

See data.py (data, SEEPS climatology), static.py (static fields) and model.py (model, checkpoints).
"""
import argparse
import math
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from data import DRY, ERA5IMERG, climatology, collate  # noqa: E402
from model import build  # noqa: E402
from softseeps import soft_seeps  # noqa: E402


class TauSchedule:
    """Multiply tau by `factor` when the validation SEEPS has not improved for more than `patience` checks."""

    def __init__(self, tau, factor, patience, min_tau=1e-3, threshold=1e-4):
        self.tau, self.factor, self.patience, self.min_tau, self.threshold = tau, factor, patience, min_tau, threshold
        self.best, self.bad = float("inf"), 0

    def step(self, seeps):
        if seeps < self.best - self.threshold:
            self.best, self.bad = seeps, 0
        else:
            self.bad += 1
            if self.bad > self.patience:
                self.tau, self.bad = max(self.tau * self.factor, self.min_tau), 0
        return self.tau


def area_weights(device, h=1800):
    """cos(latitude) of the cell centres, mean 1, [1800, 1]."""
    lat = 90.0 - (torch.arange(h, dtype=torch.float64) + 0.5) * 180.0 / h
    w = torch.cos(torch.deg2rad(lat))
    return (w / w.mean()).float()[:, None].to(device)


def msle(pred, obs, w):
    valid = torch.isfinite(obs) & (obs >= 0)
    err = (torch.log1p(pred) - torch.log1p(torch.where(valid, obs, 0.0))).pow(2)
    w = w.expand_as(obs) * valid
    return (err * w).sum() / w.sum()


def to_device(b, device):
    """ERA5 states in bfloat16 (the backbone's precision), the IMERG target in float32."""
    dtype = {"timestamp": torch.int64, "target": torch.float32}
    return {k: v.to(device).to(dtype.get(k, torch.bfloat16)) for k, v in b.items()}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--imerg_dir", required=True)
    p.add_argument("--seeps_dir", required=True, help="output of data.py")
    p.add_argument("--geoarches", required=True, help="clone of github.com/INRIA/geoarches")
    p.add_argument("--backbone", required=True, help="archesweather-m-seed1_checkpoint.ckpt")
    p.add_argument("--static", default=None, help="static_0p1.npy from static.py: adds the 0.1 deg head")
    p.add_argument("--lam", type=float, default=2.0, help="SoftSEEPS weight; 0 = MSLE-only")
    p.add_argument("--tau", type=float, default=5.0, help="initial SoftSEEPS smoothing, mm/day")
    p.add_argument("--epochs", type=int, default=None, help="default 20, 10 with --static")
    p.add_argument("--out", default="runs/imerg")
    args = p.parse_args()
    device = torch.device("cuda")
    os.makedirs(args.out, exist_ok=True)
    static = torch.from_numpy(np.load(args.static)) if args.static else None
    epochs = args.epochs or (10 if static is not None else 20)

    train = ERA5IMERG(args.imerg_dir, "train")
    val = ERA5IMERG(args.imerg_dir, "val")
    val = torch.utils.data.Subset(val, np.linspace(0, len(val) - 1, 100).round().astype(int).tolist())
    loader = torch.utils.data.DataLoader(train, batch_size=2, shuffle=True, num_workers=16, collate_fn=collate)
    val_loader = torch.utils.data.DataLoader(val, batch_size=2, num_workers=16, collate_fn=collate)

    # The decoder with the static head trains in float32 (it diverged in bfloat16); the backbone is bfloat16.
    model = build(args.geoarches, args.backbone, static, device,
                  torch.float32 if static is not None else torch.bfloat16)
    params = [q for q in model.parameters() if q.requires_grad]
    opt = torch.optim.AdamW(params, lr=1e-4, betas=(0.9, 0.98), weight_decay=1e-4)
    total = epochs * len(loader)
    lr = torch.optim.lr_scheduler.SequentialLR(opt, [
        torch.optim.lr_scheduler.LinearLR(opt, start_factor=0.01, total_iters=500),
        torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total - 500)], milestones=[500])
    sched = TauSchedule(args.tau, factor=0.7, patience=20)
    w = area_weights(device)

    def losses(b, tau, soft_obs=True):
        pred = F.softplus(model(b).float())[:, 0]                       # mm/day, [B, 1800, 3600]
        obs = b["target"][:, 0]
        p1, t2 = climatology(args.seeps_dir, b["timestamp"], device)
        return pred, obs, msle(pred, obs, w), soft_seeps(pred, obs, p1, t2, DRY, tau, weights=w,
                                                          soft_observation=soft_obs)

    for epoch in range(1, epochs + 1):
        for step, b in enumerate(loader, 1):
            b = to_device(b, device)
            _, _, l_msle, l_seeps = losses(b, sched.tau)
            loss = l_msle + args.lam * l_seeps if args.lam else l_msle
            opt.zero_grad()
            loss.backward()
            if torch.isfinite(torch.nn.utils.clip_grad_norm_(params, max_norm=0.5)):
                opt.step()
            lr.step()

            if step % 50 == 0:                    # discrete validation SEEPS drives tau
                with torch.no_grad():
                    seeps = [losses(to_device(v, device), 1e-6, soft_obs=False)[3].item() for v in val_loader]
                seeps = float(np.nanmean(seeps))
                if args.lam and math.isfinite(seeps):
                    sched.step(seeps)
                print(f"epoch {epoch} step {step} loss {loss.item():.4f} val SEEPS {seeps:.4f} "
                      f"tau {sched.tau:.4g}", flush=True)

        decoder = {k: v.cpu() for k, v in model.imerg_decoder.state_dict().items()}
        torch.save(decoder, os.path.join(args.out, f"epoch{epoch}.pt"))


if __name__ == "__main__":
    main()
