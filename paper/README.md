# Paper experiments

Training code for the two settings of the paper. All data is public, and no trained weights are
included. The pretrained backbones are linked in the code.

| | `aurora_0p25/` | `imerg_0p1/` |
|---|---|---|
| Model | Aurora 0.25°, LoRA adapters + new precipitation head | frozen ArchesWeather-M (1.5°) + decoder to 0.1°, optionally with static fields |
| Target | ERA5 6 h precipitation | IMERG V07B daily precipitation |
| Loss | MSLE + λ · SoftSEEPS + 0.02 · anti-forgetting MSE | MSLE + λ · SoftSEEPS |
| Area weighting | MSLE | MSLE and SoftSEEPS |
| Observed category in SoftSEEPS | hard | soft (`soft_observation=True`) |
| SEEPS climatology | WeatherBench 2 ERA5 1990–2019, day of year and hour | IMERG 1998–2017, day of year (`imerg_0p1/data.py`) |
| τ schedule | 2 mm, × 0.5 after 10 checks without improvement | 5 mm/day, × 0.7 after 20 checks |
| Splits | train from the start of the store – 2014-06, val 2014-07 – 2015, test 2020 | train 1998–2018, val 2019, test 2020 |

In both settings τ is lowered on plateaus of the discrete validation SEEPS, down to 0.001, and
`--lam 0` trains the MSLE-only model.

## Requirements

```bash
pip install -e ".[example]" microsoft-aurora==1.8.0 tensordict netCDF4 pandas timm
```

Tested with Python 3.10, PyTorch 2.12. A single GPU per run. The 0.1° decoder fits on
40 GB, while the Aurora fine-tune needs a larger GPU (80 GB).

## Data

| Data | Source |
|---|---|
| ERA5 0.25°, 1.5° and the SEEPS climatology | WeatherBench 2, `gs://weatherbench2/datasets/...`, read directly (paths in the code); a local copy of the years used is faster |
| IMERG Final V07B, daily | `https://gpm1.gesdisc.eosdis.nasa.gov/data/GPM_L3/GPM_3IMERGDF.07/` (free NASA Earthdata account), all files in one folder |
| ERA5-Land geopotential, ETOPO 2022 | URLs in `imerg_0p1/static.py` |
| Aurora 0.25° pretrained | URL in `aurora_0p25/train.py` |
| ArchesWeather-M, geoarches | URL in `imerg_0p1/model.py` |

## Aurora 0.25°

```bash
python paper/aurora_0p25/train.py --checkpoint aurora-0.25-pretrained.ckpt --lam 0.08
```

By default this trains on all of ERA5 from 1959 up to 2014-06, i.e. far more steps per epoch than
the paper runs, whose checkpoints are from epoch 3 (MSLE-only) and 4 (λ = 0.08) of a shorter period.

## IMERG 0.1°

```bash
python paper/imerg_0p1/data.py --imerg_dir IMERG/ --out_dir imerg_seeps/      # SEEPS climatology, once
python paper/imerg_0p1/static.py --era5land geo_1279l4_0.1x0.1.grib2_v4_unpack.nc \
    --etopo ETOPO_2022_v1_60s_N90W180_surface.nc --out static_0p1.npy       # static fields, once

python paper/imerg_0p1/train.py --imerg_dir IMERG/ --seeps_dir imerg_seeps/ \
    --geoarches geoarches/ --backbone archesweather-m-seed1_checkpoint.ckpt --lam 2
python paper/imerg_0p1/train.py ... --static static_0p1.npy --lam 4         # decoder + static
```

Each epoch writes the trained weights (LoRA + head, or the decoder) to `--out`.
