"""Frozen ArchesWeather-M backbone (1.5 deg) + trainable decoder to 0.1 deg precipitation.

Backbone code: https://github.com/INRIA/geoarches (used at commit 65642be), cloned to --geoarches.
Backbone weights (ArchesWeather-M, seed 1):
    https://huggingface.co/gcouairon/ArchesWeather/resolve/main/archesweather-m-seed1_checkpoint.ckpt

The decoder maps the backbone's surface latent [B, 384, 60, 120] to precipitation logits
[B, 1, 1800, 3600] (mm/day after softplus). With static fields, a 0.1 deg head adds a learned
correction to the logits from the decoder features and the fields of static.py.
"""
import os
import sys

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.conv1 = nn.Conv2d(c, c, 3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(c, c, 3, padding=1, bias=False)
        nn.init.kaiming_normal_(self.conv1.weight, nonlinearity="linear")
        nn.init.kaiming_normal_(self.conv2.weight, nonlinearity="linear")

    def forward(self, x):
        return x + self.conv2(F.gelu(self.conv1(x)))


class StaticHead(nn.Module):
    """Logit correction at 0.1 deg from x5-upsampled decoder features and the static fields.
    Its last layer starts at zero, so it does not change the decoder output at initialisation."""

    def __init__(self, in_dim, n_static, channels=64):
        super().__init__()
        n_proj = -(-(channels + n_static) // 8) * 8 - n_static     # concatenation: multiple of 8 channels
        self.proj = nn.Conv2d(in_dim, n_proj, 1, bias=False)
        self.conv1 = nn.Conv2d(n_proj + n_static, channels, 3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.conv_out = nn.Conv2d(channels, 1, 3, padding=1)
        nn.init.zeros_(self.conv_out.weight)
        nn.init.zeros_(self.conv_out.bias)

    def forward(self, x, static):
        x = F.interpolate(self.proj(x).float(), scale_factor=5, mode="bilinear").to(x.dtype)
        x = torch.cat([x, static.to(x.dtype).expand(x.shape[0], -1, -1, -1)], dim=1)
        return self.conv_out(F.gelu(self.conv2(F.gelu(self.conv1(x)))))


class Decoder(nn.Module):
    """Residual conv stages at 60x120, 120x240 and 360x720, then x5 bilinear to 1800x3600."""

    def __init__(self, in_dim=384, h1=512, h2=384, n_res=4, static=None):
        super().__init__()
        self.input_norm = nn.LayerNorm(in_dim)
        self.conv_in = nn.Conv2d(in_dim, h1, 3, padding=1, bias=False)
        self.res1 = nn.Sequential(*[ResBlock(h1) for _ in range(n_res)])
        self.conv12 = nn.Conv2d(h1, h1, 3, padding=1, bias=False)
        self.res2 = nn.Sequential(*[ResBlock(h1) for _ in range(n_res)])
        self.conv23 = nn.Conv2d(h1, h2, 3, padding=1, bias=False)
        self.res3 = nn.Sequential(*[ResBlock(h2) for _ in range(n_res)])
        self.conv_out = nn.Conv2d(h2, 1, 1)
        for m in (self.conv_in, self.conv12, self.conv23):
            nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
        nn.init.zeros_(self.conv_out.weight)
        nn.init.zeros_(self.conv_out.bias)
        self.head = None
        if static is not None:                     # fixed input, not trained, not saved
            self.register_buffer("static", static[None], persistent=False)
            self.head = StaticHead(h2, static.shape[0])

    def forward(self, z):
        x = F.layer_norm(z.permute(0, 2, 3, 1).float(), self.input_norm.normalized_shape,
                         self.input_norm.weight.float(), self.input_norm.bias.float(), self.input_norm.eps)
        x = F.gelu(self.conv_in(x.permute(0, 3, 1, 2).to(self.conv_in.weight.dtype)))
        x = F.interpolate(F.gelu(self.conv12(self.res1(x))), scale_factor=2, mode="bilinear")
        x = F.interpolate(F.gelu(self.conv23(self.res2(x))), scale_factor=3, mode="bilinear")
        x = self.res3(x)
        logits = F.interpolate(self.conv_out(x), scale_factor=5, mode="bilinear")
        if self.head is not None:
            logits = logits + self.head(x, self.static)
        return logits


class ArchesWeatherIMERG(nn.Module):
    def __init__(self, geoarches, static=None):
        super().__init__()
        sys.path.insert(0, geoarches)
        from geoarches.backbones.archesweather import ArchesWeatherCondBackbone, WeatherEncodeDecodeLayer
        from geoarches.backbones.dit import TimestepEmbedder
        # ArchesWeather-M configuration
        self.backbone = ArchesWeatherCondBackbone(
            tensor_size=(8, 60, 120), emb_dim=192, cond_dim=256, window_size=(1, 6, 10),
            droppath_coeff=0.2, depth_multiplier=2, use_skip=True, first_interaction_layer="linear",
            axis_attn=True, mlp_layer="swiglu", mlp_ratio=4.0)
        self.embedder = WeatherEncodeDecodeLayer(
            img_size=(13, 121, 240), emb_dim=192, out_emb_dim=384, patch_size=(2, 2, 2),
            surface_ch=4, level_ch=6, n_concatenated_states=1)
        self.month_embedder = TimestepEmbedder(256)
        self.hour_embedder = TimestepEmbedder(256)
        self.imerg_decoder = Decoder(static=static)
        stats = torch.load(os.path.join(geoarches, "geoarches", "stats", "pangu_norm_stats2_with_w.pt"),
                           weights_only=True)
        for k in ("surface_mean", "surface_std", "level_mean", "level_std"):
            self.register_buffer(k, stats[k].float(), persistent=False)

    def load_backbone(self, path):
        sd = torch.load(path, map_location="cpu", weights_only=False)
        missing = self.load_state_dict(sd.get("state_dict", sd), strict=False).missing_keys
        missing = [k for k in missing if not k.startswith("imerg_decoder.")]
        if missing:
            raise RuntimeError(f"backbone checkpoint lacks {missing[:5]}")
        for n, p in self.named_parameters():
            p.requires_grad = n.startswith("imerg_decoder.")

    def _normalise(self, s, dtype):
        out = s.clone()
        out["surface"] = ((s["surface"].float() - self.surface_mean) / self.surface_std).to(dtype)
        out["level"] = ((s["level"].float() - self.level_mean) / self.level_std).to(dtype)
        return out

    def forward(self, batch):
        dtype = next(self.backbone.parameters()).dtype
        with torch.no_grad():                      # the backbone is frozen
            t = pd.to_datetime(batch["timestamp"].cpu().numpy(), unit="s")
            dev = batch["timestamp"].device
            cond = (self.month_embedder(torch.tensor(t.month.to_numpy(), device=dev))
                    + self.hour_embedder(torch.tensor(t.hour.to_numpy(), device=dev))).to(dtype)
            x = self.embedder.encode(self._normalise(batch["state"], dtype),
                                     self._normalise(batch["prev_state"], dtype))
            z = self.backbone(x, cond)[:, :, 0]    # surface latent [B, 384, 60, 120]
        return self.imerg_decoder(z)


def build(geoarches, backbone_ckpt, static, device, decoder_dtype):
    model = ArchesWeatherIMERG(geoarches, static).to(torch.bfloat16)
    model.month_embedder.float()                   # TimestepEmbedder builds float32 sinusoids
    model.hour_embedder.float()
    model.imerg_decoder.to(decoder_dtype)
    model.load_backbone(backbone_ckpt)
    model.to(device)
    model.embedder.constant_masks = model.embedder.constant_masks.to(torch.bfloat16).to(device)
    return model.eval()                            # no drop-path in the frozen backbone
