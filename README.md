# SoftSEEPS

A differentiable relaxation of **SEEPS** (Stable Equitable Error in Probability Space;
Rodwell et al., 2010), the WMO headline score for precipitation, for use as a training loss.

This is the repository to a paper currently under review.


SEEPS sorts forecast and observation into three categories (dry, light, heavy) using
climatological thresholds and penalises category errors with a matrix that depends on the
climatological probability of a dry day, `p1`. SoftSEEPS replaces the hard forecast category
by sigmoids of width τ, so the score has a gradient; τ → 0 gives the discrete SEEPS.

## Installation

```bash
pip install git+https://github.com/Jostarndt/SoftSEEPS
```

## Usage

```python
from softseeps import soft_seeps

loss = soft_seeps(forecast, observation, p1, t2, dry_threshold=0.25, tau=1.0, weights=None)
```

| Argument | Meaning |
|---|---|
| `forecast`, `observation` | precipitation tensors, any shape `[..., H, W]` |
| `p1` | climatological probability of a dry day, shape `[H, W]` or `[..., H, W]` |
| `t2` | climatological light/heavy threshold (splits the wet days 2:1), shape `[H, W]` or `[..., H, W]` |
| `dry_threshold` | dry/light threshold, e.g. 0.25 mm for 24 h and 0.1 mm for 6 h accumulations |
| `tau` | smoothing parameter (width of the sigmoids), in the unit of the precipitation |
| `weights` | optional point weights, shape `[H, W]`, or `[H, 1]` for cos(latitude) |

Forecast, observation, `t2`, `dry_threshold` and `tau` must use the same unit. Points with `p1`
outside [0.1, 0.85] (where SEEPS is not defined) or with non-finite inputs are ignored. The
observation keeps its hard category. Combined with an MSLE loss:

```python
loss = msle(pred, obs) + lam * soft_seeps(pred, obs, p1, t2, dry_threshold, tau)
```

## Climatologies

`p1` and `t2` are needed for the day of year (and hour) of each valid time, on the forecast grid.
WeatherBench 2 provides them for ERA5, publicly readable without an account:

```
gs://weatherbench2/datasets/era5-hourly-climatology/1990-2019_6h_1440x721.zarr
    total_precipitation_24hr_seeps_dry_fraction   # p1
    total_precipitation_24hr_seeps_threshold      # t2, in metres
    total_precipitation_6hr_seeps_dry_fraction
    total_precipitation_6hr_seeps_threshold
```

The same folder has other resolutions (e.g. `240x121`, `64x32`), see the
[WeatherBench 2 data guide](https://weatherbench2.readthedocs.io/en/latest/data-guide.html).
For other data, compute per grid point and day of year `p1` = fraction of days with
precipitation ≤ `dry_threshold`, and `t2` = 2/3 quantile of the wet days.

## Example

`examples/weatherbench2_example.ipynb` loads the WeatherBench 2 climatology, an IFS HRES
forecast and ERA5, and computes SoftSEEPS and its gradient globally and for a European domain
(`pip install ".[example]"` from the repository root). Forecasts in the same bucket include
`gs://weatherbench2/datasets/hres/2016-2022-0012-1440x721.zarr` (IFS HRES) and
`gs://weatherbench2/datasets/era5/1959-2023_01_10-wb13-6h-1440x721_with_derived_variables.zarr`
(ERA5 truth).

## Citation

If you use SoftSEEPS, please cite our paper (in preparation), and

Rodwell, M. J., Richardson, D. S., Hewson, T. D., & Haiden, T. (2010). A new equitable score
suitable for verifying precipitation in numerical weather prediction. *Quarterly Journal of
the Royal Meteorological Society*, 136(650), 1344–1363.

## License

MIT, see `LICENSE`.
