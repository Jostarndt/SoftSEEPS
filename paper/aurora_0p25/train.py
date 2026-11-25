"""Aurora 0.25 deg, 6 h precipitation: LoRA fine-tune with MSLE + lambda * SoftSEEPS.

Pretrained Aurora (0.25 deg, frozen except LoRA adapters and a new precipitation head):
    https://huggingface.co/microsoft/aurora/resolve/main/aurora-0.25-pretrained.ckpt

    python paper/aurora_0p25/train.py --checkpoint aurora-0.25-pretrained.ckpt --lam 0.08
    python paper/aurora_0p25/train.py ... --lam 0          # MSLE-only

--era5 and --climatology default to the public WeatherBench 2 stores; a local copy of the
ERA5 zarr is much faster.
"""
import argparse
import dataclasses
import datetime
import functools
import os
import sys

import numpy as np
import torch
import xarray as xr
from aurora import Aurora, Batch, Metadata
from aurora.model import decoder as aurora_decoder
from aurora.normalisation import normalise_atmos_var, normalise_surf_var

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from softseeps import soft_seeps  # noqa: E402

ERA5_ZARR = "gs://weatherbench2/datasets/era5/1959-2023_01_10-wb13-6h-1440x721_with_derived_variables.zarr"
CLIM_ZARR = "gs://weatherbench2/datasets/era5-hourly-climatology/1990-2019_6h_1440x721.zarr"
SURF = {"2t": "2m_temperature", "10u": "10m_u_component_of_wind",
        "10v": "10m_v_component_of_wind", "msl": "mean_sea_level_pressure"}
ATMOS = {"t": "temperature", "u": "u_component_of_wind", "v": "v_component_of_wind",
         "q": "specific_humidity", "z": "geopotential"}
STATIC = {"lsm": "land_sea_mask", "slt": "soil_type", "z": "geopotential_at_surface"}
LEVELS = (50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000)
DRY = 0.1      # mm, the WeatherBench 2 dry threshold for 6 h accumulations
H = 720        # Aurora drops the southernmost of the 721 rows


def open_zarr(path):
    return xr.open_zarr(path, storage_options={"token": "anon"} if path.startswith("gs://") else None)


# ---------------------------------------------------------------- model
class Perceiver3DDecoder(aurora_decoder.Perceiver3DDecoder):
    """Aurora's decoder with one more surface head, 'tp', which is not an input variable.
    Same class name as Aurora's, so that configure_activation_checkpointing finds it."""

    def forward(self, x, batch, patch_res, lead_time):
        # The decoder only reads the variable names from batch.surf_vars.
        surf = {**batch.surf_vars, "tp": batch.surf_vars["2t"]}
        return super().forward(x, dataclasses.replace(batch, surf_vars=surf), patch_res, lead_time)


class PrecipAurora(Aurora):
    def __init__(self, **kwargs):
        super().__init__(surf_stats={"tp": (0.0, 1.0)}, **kwargs)   # tp is not (un)normalised
        self.decoder = Perceiver3DDecoder(surf_vars=tuple(SURF) + ("tp",), atmos_vars=("z", "u", "v", "t", "q"),
                                          patch_size=4, embed_dim=1024, head_dim=64, num_heads=16, depth=1,
                                          mlp_ratio=2.0, perceiver_ln_eps=1e-5)

    def forward(self, batch):
        pred = super().forward(batch)
        tp = torch.nn.functional.softplus(pred.surf_vars.pop("tp")[:, 0])   # [B, 720, 1440], m
        return pred, tp


# ---------------------------------------------------------------- data
class ERA5(torch.utils.data.Dataset):
    """State at t - 6 h and t (inputs), state and total_precipitation_6hr at t + 6 h (targets).
    WeatherBench 2 0.25 deg ERA5 is north-first and Greenwich-first, like Aurora."""

    def __init__(self, path, start, end):
        self.ds = open_zarr(path)
        t = self.ds.time.sel(time=slice(start, end)).values
        self.times = t[1:-1]
        self.static = {k: torch.from_numpy(self.ds[v].values.astype(np.float32)) for k, v in STATIC.items()}

    def __len__(self):
        return len(self.times)

    def _state(self, t):
        x = self.ds.sel(time=t)
        surf = np.stack([x[v].values for v in SURF.values()])
        atmos = np.stack([x[v].sel(level=list(LEVELS)).values for v in ATMOS.values()])
        return torch.from_numpy(surf.astype(np.float32)), torch.from_numpy(atmos.astype(np.float32))

    def __getitem__(self, i):
        t = self.times[i]
        dt = np.timedelta64(6, "h")
        (s0, a0), (s1, a1), (s2, a2) = (self._state(t - dt), self._state(t), self._state(t + dt))
        tp = self.ds["total_precipitation_6hr"].sel(time=t + dt).values.astype(np.float32)
        return {"time": int(t.astype("datetime64[s]").astype(np.int64)),
                "surf": torch.stack([s0, s1]), "atmos": torch.stack([a0, a1]),
                "next_surf": s2, "next_atmos": a2, "tp": torch.from_numpy(tp)}


def to_aurora(b, static, device):
    times = [datetime.datetime.fromtimestamp(int(t), tz=datetime.timezone.utc) for t in b["time"]]
    return Batch(surf_vars={k: b["surf"][:, :, i] for i, k in enumerate(SURF)},
                 static_vars=static,
                 atmos_vars={k: b["atmos"][:, :, i] for i, k in enumerate(ATMOS)},
                 metadata=Metadata(lat=torch.linspace(90, -90, 721), lon=torch.linspace(0, 360, 1441)[:-1],
                                   time=tuple(times), atmos_levels=LEVELS)).to(device)


@functools.lru_cache(maxsize=64)
def _climatology(path, doy, hour):
    c = open_zarr(path).sel(dayofyear=doy, hour=hour)
    p1 = c["total_precipitation_6hr_seeps_dry_fraction"].values[:H]
    t2 = c["total_precipitation_6hr_seeps_threshold"].values[:H] * 1000.0    # m -> mm
    return torch.from_numpy(p1.astype(np.float32)), torch.from_numpy(t2.astype(np.float32))


def climatology(path, times, device):
    """p1, t2 of each sample's valid time (t + 6 h): day of year and hour, [B, 720, 1440]."""
    valid = [datetime.datetime.fromtimestamp(int(t) + 6 * 3600, tz=datetime.timezone.utc) for t in times]
    pairs = [_climatology(path, v.timetuple().tm_yday, v.hour) for v in valid]
    return (torch.stack([p for p, _ in pairs]).to(device), torch.stack([t for _, t in pairs]).to(device))


# ---------------------------------------------------------------- losses
def lat_weights():
    """Grid-cell area weights of the 721 rows (mean 1), cropped to Aurora's 720 rows, [720, 1]."""
    lat = torch.deg2rad(torch.linspace(-90, 90, 721))
    bounds = torch.cat([torch.tensor([-torch.pi / 2]), (lat[:-1] + lat[1:]) / 2, torch.tensor([torch.pi / 2])])
    w = torch.sin(bounds[1:]) - torch.sin(bounds[:-1])
    return (w / w.mean())[:H, None]


def msle(pred, obs, w):
    """Area-weighted MSE of log1p(precipitation in mm)."""
    return torch.nanmean((torch.log1p(1000 * pred) - torch.log1p(1000 * obs)).pow(2) * w)


def anti_forgetting(pred, b, device):
    """MSE of all Aurora output variables in Aurora's normalised units (keeps the backbone's skill)."""
    losses = [(normalise_surf_var(pred.surf_vars[k][:, 0], k)
               - normalise_surf_var(b["next_surf"][:, i, :H].to(device), k)).pow(2).mean()
              for i, k in enumerate(SURF)]
    losses += [(normalise_atmos_var(pred.atmos_vars[k][:, 0], k, LEVELS)
                - normalise_atmos_var(b["next_atmos"][:, i, :, :H].to(device), k, LEVELS)).pow(2).mean()
               for i, k in enumerate(ATMOS)]
    return torch.stack(losses).mean()


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


# ---------------------------------------------------------------- training
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True, help="aurora-0.25-pretrained.ckpt")
    p.add_argument("--era5", default=ERA5_ZARR)
    p.add_argument("--climatology", default=CLIM_ZARR)
    p.add_argument("--train_start", default=None, help="first training date; default: start of the ERA5 store")
    p.add_argument("--train_end", default="2014-06-30")
    p.add_argument("--val", nargs=2, default=("2014-07-01", "2015-12-31"))
    p.add_argument("--lam", type=float, default=0.08, help="SoftSEEPS weight; 0 = MSLE-only")
    p.add_argument("--anti_forgetting", type=float, default=None, help="default 0.02, 0.01 for MSLE-only")
    p.add_argument("--tau", type=float, default=2.0, help="initial SoftSEEPS smoothing, mm")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--out", default="runs/aurora")
    args = p.parse_args()
    af = args.anti_forgetting if args.anti_forgetting is not None else (0.02 if args.lam else 0.01)
    device = torch.device("cuda")
    os.makedirs(args.out, exist_ok=True)

    train = ERA5(args.era5, args.train_start, args.train_end)
    val = ERA5(args.era5, *args.val)
    val = torch.utils.data.Subset(val, np.linspace(0, len(val) - 1, 20).round().astype(int).tolist())
    loader = torch.utils.data.DataLoader(train, batch_size=1, shuffle=True, num_workers=8)
    static = {k: v.to(device) for k, v in train.static.items()}

    model = PrecipAurora(autocast=True)
    model.load_checkpoint_local(args.checkpoint, strict=False)   # the tp head stays random
    model.configure_activation_checkpointing()
    for n, q in model.named_parameters():
        q.requires_grad = "lora" in n or "surf_heads.tp" in n
    model.to(device)
    opt = [torch.optim.AdamW([q for n, q in model.named_parameters() if "lora" in n],
                             lr=1e-4, betas=(0.9, 0.98), weight_decay=0.05),
           torch.optim.AdamW([q for n, q in model.named_parameters() if "surf_heads.tp" in n],
                             lr=1e-3, betas=(0.9, 0.98), weight_decay=0.05)]
    w = lat_weights().to(device)
    sched = TauSchedule(args.tau, factor=0.5, patience=10)

    for epoch in range(1, args.epochs + 1):
        for step, b in enumerate(loader, 1):
            pred, tp = model(to_aurora(b, static, device))
            obs = b["tp"][:, :H].to(device)
            loss = msle(tp, obs, w) + af * anti_forgetting(pred, b, device)
            if args.lam:
                p1, t2 = climatology(args.climatology, b["time"], device)
                loss = loss + args.lam * soft_seeps(1000 * tp, 1000 * obs, p1, t2, DRY, sched.tau)
            for o in opt:
                o.zero_grad()
            if torch.isfinite(loss):
                loss.backward()
                for o in opt:
                    o.step()

            if step % 100 == 0:                       # discrete validation SEEPS drives tau
                model.eval()
                with torch.no_grad():
                    scores = []
                    for i in range(len(val)):
                        v = torch.utils.data.default_collate([val[i]])
                        _, vtp = model(to_aurora(v, static, device))
                        p1, t2 = climatology(args.climatology, v["time"], device)
                        scores.append(soft_seeps(1000 * vtp, 1000 * v["tp"][:, :H].to(device),
                                                 p1, t2, DRY, tau=1e-6).item())
                model.train()
                if args.lam:
                    sched.step(float(np.nanmean(scores)))
                print(f"epoch {epoch} step {step} loss {loss.item():.4f} "
                      f"val SEEPS {np.nanmean(scores):.4f} tau {sched.tau:.4g}", flush=True)

        trainable = {k: v.cpu() for k, v in model.state_dict().items() if "lora" in k or "surf_heads.tp" in k}
        torch.save(trainable, os.path.join(args.out, f"epoch{epoch}.pt"))


if __name__ == "__main__":
    main()
