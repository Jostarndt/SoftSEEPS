"""ERA5 1.5 deg inputs and IMERG 0.1 deg targets, and the IMERG SEEPS climatology.

ERA5: WeatherBench 2, 1.5 deg (240 x 121), public:
    gs://weatherbench2/datasets/era5/1959-2023_01_10-6h-240x121_equiangular_with_poles_conservative.zarr
IMERG Final Run V07B daily (GPM_3IMERGDF.07), free NASA Earthdata account:
    https://gpm1.gesdisc.eosdis.nasa.gov/data/GPM_L3/GPM_3IMERGDF.07/
    files 3B-DAY.MS.MRG.3IMERG.<YYYYMMDD>-S000000-E235959.V07B.nc4, all in one folder.

Everything is on the IMERG grid convention: north up, dateline first. A sample is the
ERA5 state at 00 UTC of day D (and of day D - 1); the target is IMERG of day D.

Build the SEEPS climatology once (per day of year from 1998-2017):
    python paper/imerg_0p1/data.py --imerg_dir IMERG/ --out_dir imerg_seeps/
"""
import argparse
import datetime
import functools
import os

import numpy as np
import torch
import xarray as xr
from tensordict import TensorDict

ERA5 = "gs://weatherbench2/datasets/era5/1959-2023_01_10-6h-240x121_equiangular_with_poles_conservative.zarr"
SURFACE = ["10m_u_component_of_wind", "10m_v_component_of_wind", "2m_temperature", "mean_sea_level_pressure"]
LEVEL = ["geopotential", "u_component_of_wind", "v_component_of_wind", "temperature",
         "specific_humidity", "vertical_velocity"]
SPLITS = {"train": (1998, 2018), "val": (2019, 2019), "test": (2020, 2020)}
DRY = 0.25   # mm/day


def imerg_file(imerg_dir, day):
    return os.path.join(imerg_dir, f"3B-DAY.MS.MRG.3IMERG.{day:%Y%m%d}-S000000-E235959.V07B.nc4")


def read_imerg(path):
    """Daily precipitation in mm/day, [1800, 3600], north up, dateline first; NaN = missing."""
    with xr.open_dataset(path) as ds:
        return ds["precipitation"].sortby("lat", ascending=False).values[0].T.astype(np.float32)


class ERA5IMERG(torch.utils.data.Dataset):
    def __init__(self, imerg_dir, split, era5=ERA5):
        first, last = SPLITS[split]
        self.imerg_dir = imerg_dir
        self.ds = xr.open_zarr(era5, storage_options={"token": "anon"} if era5.startswith("gs://") else None)
        days = np.arange(np.datetime64(f"{first}-01-01"), np.datetime64(f"{last + 1}-01-01"))
        self.days = [datetime.datetime.fromisoformat(str(d)) for d in days]

    def __len__(self):
        return len(self.days)

    def _state(self, day):
        x = self.ds.sel(time=np.datetime64(day, "ns"))
        surf = np.stack([x[v].transpose("latitude", "longitude").values for v in SURFACE])[:, None]
        level = np.stack([x[v].transpose("level", "latitude", "longitude").values for v in LEVEL])  # [6, 13, 121, 240]
        # WeatherBench 2 is south-first and Greenwich-first; ArchesWeather is north-first, dateline-first.
        flip = lambda a: np.roll(a[..., ::-1, :], a.shape[-1] // 2, axis=-1).copy()
        return TensorDict({"surface": torch.from_numpy(flip(surf).astype(np.float32)),
                           "level": torch.from_numpy(flip(level).astype(np.float32))})

    def __getitem__(self, i):
        day = self.days[i]
        target = read_imerg(imerg_file(self.imerg_dir, day))
        return {"timestamp": torch.tensor(int(day.replace(tzinfo=datetime.timezone.utc).timestamp())),
                "prev_state": self._state(day - datetime.timedelta(days=1)),
                "state": self._state(day),
                "target": torch.from_numpy(target)[None]}                        # [1, 1800, 3600]


def collate(samples):
    return {k: torch.stack([s[k] for s in samples]) for k in samples[0]}


@functools.lru_cache(maxsize=32)
def _climatology(seeps_dir, doy):
    c = np.load(os.path.join(seeps_dir, f"seeps_doy_{doy:03d}.npz"))
    return torch.from_numpy(c["p1"]), torch.from_numpy(c["t2"])


def climatology(seeps_dir, timestamps, device):
    """p1 and t2 (mm/day) for the day of year of each sample, [B, 1800, 3600]."""
    doys = [datetime.datetime.fromtimestamp(int(t), tz=datetime.timezone.utc).timetuple().tm_yday
            for t in timestamps]
    pairs = [_climatology(seeps_dir, d) for d in doys]
    return torch.stack([p for p, _ in pairs]).to(device), torch.stack([t for _, t in pairs]).to(device)


def build_climatology(imerg_dir, out_dir, years=(1998, 2017)):
    """Per day of year: p1 = fraction of days <= 0.25 mm, t2 = 2/3 quantile of the wetter days.
    Missing values count as dry; p1 and t2 are NaN where p1 is outside [0.1, 0.85]."""
    os.makedirs(out_dir, exist_ok=True)
    for doy in range(1, 367):
        days = [datetime.date(y, 1, 1) + datetime.timedelta(days=doy - 1) for y in range(years[0], years[1] + 1)]
        days = [d for d in days if d.timetuple().tm_yday == doy]          # day 366 only in leap years
        x = np.nan_to_num(np.stack([read_imerg(imerg_file(imerg_dir, d)) for d in days]), nan=0.0)
        p1 = (x <= DRY).mean(axis=0)
        t2 = np.nanpercentile(np.where(x > DRY, x, np.nan), 66.67, axis=0)
        t2 = np.where(np.isfinite(t2), t2, DRY)
        bad = (p1 < 0.1) | (p1 > 0.85)
        p1[bad], t2[bad] = np.nan, np.nan
        np.savez(os.path.join(out_dir, f"seeps_doy_{doy:03d}.npz"),
                 p1=p1.astype(np.float32), t2=t2.astype(np.float32))
        print(f"day of year {doy}: {len(days)} years", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--imerg_dir", required=True)
    p.add_argument("--out_dir", required=True)
    a = p.parse_args()
    build_climatology(a.imerg_dir, a.out_dir)
