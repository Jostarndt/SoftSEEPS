"""Static 0.1 deg inputs of the decoder head, on the IMERG grid (north up, dateline first).

Sources (public, no login):
  ERA5-Land surface geopotential, 0.1 deg (ECMWF):
    https://confluence.ecmwf.int/download/attachments/140385202/geo_1279l4_0.1x0.1.grib2_v4_unpack.nc
  ETOPO 2022 v1, 60 arc-second surface elevation (NOAA NCEI, doi:10.25921/fd45-gt74):
    https://www.ngdc.noaa.gov/thredds/fileServer/global/ETOPO2022/60s/60s_surface_elev_netcdf/ETOPO_2022_v1_60s_N90W180_surface.nc

Channels: orography (m), sub-grid roughness (log1p of the std of max(elevation, 0) over the
6 x 6 ETOPO cells of each 0.1 deg cell), land fraction, sin/cos of latitude and longitude;
each standardised over the grid.

    python paper/imerg_0p1/static.py --era5land geo_1279l4_0.1x0.1.grib2_v4_unpack.nc \
        --etopo ETOPO_2022_v1_60s_N90W180_surface.nc --out static_0p1.npy
"""
import argparse

import numpy as np
import xarray as xr

NLAT, NLON = 1800, 3600


def orography(path):
    z = xr.open_dataset(path).z.values.squeeze().astype(np.float64) / 9.80665   # [1801, 3600], 90..-90, 0..359.9
    z = 0.5 * (z[:-1] + z[1:])                    # cell centres 89.95 .. -89.95
    z = 0.5 * (z + np.roll(z, -1, axis=1))        # 0.05 .. 359.95
    return np.roll(z, -NLON // 2, axis=1)         # dateline first: -179.95 .. 179.95


def etopo(path):
    e = xr.open_dataset(path).z.values[::-1].astype(np.float64)    # cell-registered, now north up
    b = np.maximum(e.reshape(NLAT, 6, NLON, 6), 0.0)
    return np.log1p(b.std(axis=(1, 3))), (e.reshape(NLAT, 6, NLON, 6) > 0).mean(axis=(1, 3))


def static_fields(era5land, etopo_path):
    roughness, land = etopo(etopo_path)
    lat = np.deg2rad(89.95 - 0.1 * np.arange(NLAT))[:, None] * np.ones((1, NLON))
    lon = np.deg2rad(-179.95 + 0.1 * np.arange(NLON))[None, :] * np.ones((NLAT, 1))
    fields = [orography(era5land), roughness, land, np.sin(lat), np.cos(lat), np.sin(lon), np.cos(lon)]
    return np.stack([(f - f.mean()) / f.std() for f in fields]).astype(np.float32)   # [7, 1800, 3600]


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--era5land", required=True)
    p.add_argument("--etopo", required=True)
    p.add_argument("--out", default="static_0p1.npy")
    a = p.parse_args()
    np.save(a.out, static_fields(a.era5land, a.etopo))
