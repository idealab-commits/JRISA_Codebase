# -*- coding: utf-8 -*-
"""
model.py -- JRISA network definition.

Implements, following the manuscript's Proposed Method section exactly:
  - Shared 3D encoder f (Section III-D)
  - Shared decoder g (Section III-D), applied to both branches
  - Pseudo-isotropic module h (Eq. 2, Section III-E)
  - Dual heads per branch: semantic (C classes) + instance BCD (3 channels)
    (Section III-F)
  - Adaptive Frequency Gate (Eqs. 11-14, Section III-G) -- ablation only,
    enabled via AFG_ENABLED

Channel widths (16, 32, 64, 128, 256) match the backbone used by ReIsoSeg
[26] and SynReEM [18] for direct comparability (Section III-D).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

CHANNELS = (16, 32, 64, 128, 256)


class ConvBlock3D(nn.Module):
    """Two 3x3x3 conv-instancenorm-leakyrelu layers."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.InstanceNorm3d(out_ch, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.InstanceNorm3d(out_ch, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class Encoder3D(nn.Module):
    """5-stage 3D encoder, channel widths (16,32,64,128,256) -- Section III-D."""
    def __init__(self, in_channels=1, channels=CHANNELS):
        super().__init__()
        self.stages = nn.ModuleList()
        self.pools = nn.ModuleList()
        prev = in_channels
        for i, ch in enumerate(channels):
            self.stages.append(ConvBlock3D(prev, ch))
            if i < len(channels) - 1:
                self.pools.append(nn.Conv3d(ch, ch, kernel_size=2, stride=2))
            prev = ch

    def forward(self, x):
        """Returns list [F_1, ..., F_5] (finest -> coarsest), Section III-D."""
        feats = []
        h = x
        for i, stage in enumerate(self.stages):
            h = stage(h)
            feats.append(h)
            if i < len(self.pools):
                h = self.pools[i](h)
        return feats


class Decoder3D(nn.Module):
    """Mirrors the encoder with skip connections. Shared across both
    branches (Section III-D: 'g's weights are shared, not duplicated')."""
    def __init__(self, channels=CHANNELS):
        super().__init__()
        rev = list(reversed(channels))
        self.ups = nn.ModuleList()
        self.blocks = nn.ModuleList()
        for i in range(len(rev) - 1):
            self.ups.append(nn.ConvTranspose3d(rev[i], rev[i + 1], kernel_size=2, stride=2))
            self.blocks.append(ConvBlock3D(rev[i + 1] * 2, rev[i + 1]))

    def forward(self, feats):
        """feats = [F_1,...,F_5] from the encoder (or PI-module-deformed features)."""
        x = feats[-1]
        skips = list(reversed(feats[:-1]))
        for up, block, skip in zip(self.ups, self.blocks, skips):
            x = up(x)
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:], mode="trilinear", align_corners=False)
            x = torch.cat([x, skip], dim=1)
            x = block(x)
        return x  # channel width = channels[0], e.g. 16


class PseudoIsotropicModule(nn.Module):
    """h(F) = D_z(Conv_7^3(U_z(F; s)))  -- Eq. (2), Section III-E.

    Applied to the first three encoder stages, following [26]: up-samples
    along z by the dataset-specific factor `s`, applies a 7x7x7 conv, and
    the caller's decoder performs the corresponding z-downsampling implicitly
    via its own transposed convs (matched spatial dims are enforced there).
    """
    def __init__(self, channels=CHANNELS, n_stages=3, s=(1, 2, 2)):
        super().__init__()
        self.s = s
        self.n_stages = n_stages
        self.convs = nn.ModuleList([
            nn.Conv3d(ch, ch, kernel_size=7, padding=3)
            for ch in channels[:n_stages]
        ])

    def forward(self, feats):
        """feats: list of encoder features [F_1,...,F_5]. Returns a new list
        with the first `n_stages` entries passed through the PI deformation."""
        out = list(feats)
        sz, sy, sx = self.s
        for i in range(self.n_stages):
            f = out[i]
            # up-sample along z by factor sz (the anisotropic axis), following [26]
            if sz > 1:
                f = F.interpolate(f, scale_factor=(sz, 1, 1), mode="trilinear", align_corners=False)
            f = self.convs[i](f)
            # restore original z-extent so the shared decoder's skip connections align
            if sz > 1:
                f = F.interpolate(f, size=out[i].shape[2:], mode="trilinear", align_corners=False)
            out[i] = f
        return out


class WaveletHF(nn.Module):
    """High-frequency extraction, H = HL + LH + HH (Eq. 11, Section III-G).

    Practical stand-in for a full 3D Haar decomposition: a fixed, non-learned
    depthwise box-blur is used to estimate the low-frequency component L, and
    H = input - L approximates the summed high-frequency sub-bands. This is
    documented here explicitly as an approximation -- swap in a proper 3D DWT
    (e.g. via a custom separable Haar filter bank) if exact fidelity to [13]'s
    formulation is required.
    """
    def __init__(self, channels, kernel_size=3):
        super().__init__()
        self.channels = channels
        pad = kernel_size // 2
        weight = torch.ones(channels, 1, kernel_size, kernel_size, kernel_size)
        weight = weight / weight[0].numel()
        self.register_buffer("blur_weight", weight)
        self.pad = pad

    def forward(self, x):
        low = F.conv3d(x, self.blur_weight, padding=self.pad, groups=self.channels)
        return x - low


class AdaptiveFrequencyGate(nn.Module):
    """AFG -- Eqs. (11)-(14), Section III-G. Ablation-only (Objective O5)."""
    def __init__(self, channels, reduction=4):
        super().__init__()
        self.hf = WaveletHF(channels)
        hidden = max(channels // reduction, 4)
        self.gap = nn.AdaptiveAvgPool3d(1)
        self.fc1 = nn.Conv3d(channels, hidden, kernel_size=1)
        self.fc2 = nn.Conv3d(hidden, channels, kernel_size=1)

    def forward(self, feat):
        H = self.hf(feat)                              # Eq. (11)
        pooled = self.gap(H)                            # Eq. (12), GAP(H)
        alpha = torch.sigmoid(self.fc2(F.relu(self.fc1(pooled))))  # Eq. (13)
        return feat + alpha * H                          # Eq. (14)


class JRISA(nn.Module):
    """Full model: shared encoder/decoder, PI module, AFG (optional), and
    dual semantic + instance heads on both the anisotropic and isotropic
    branches (Sections III-D through III-G)."""

    def __init__(self, in_channels=1, num_sem_classes=2, z_sampling=(1, 2, 2),
                 afg_enabled=False, channels=CHANNELS):
        super().__init__()
        self.encoder = Encoder3D(in_channels, channels)
        self.decoder = Decoder3D(channels)  # SHARED across both branches
        self.pi_module = PseudoIsotropicModule(channels, n_stages=3, s=z_sampling)

        self.afg_enabled = afg_enabled
        if afg_enabled:
            self.afg = nn.ModuleList([AdaptiveFrequencyGate(ch) for ch in channels])

        base_ch = channels[0]
        # Dual heads, shared weights across ani/iso branches (only the input
        # features differ), matching Algorithm 1 lines 14 and 17.
        self.sem_head = nn.Conv3d(base_ch, num_sem_classes, kernel_size=1)
        self.inst_head = nn.Conv3d(base_ch, 3, kernel_size=1)  # B, C, D channels

    def _apply_afg(self, feats):
        return [gate(f) for gate, f in zip(self.afg, feats)]

    def _decode_branch(self, feats):
        dec = self.decoder(feats)
        sem = self.sem_head(dec)
        inst = self.inst_head(dec)  # raw logits for B,C and raw value for D
        inst_b = inst[:, 0:1]
        inst_c = inst[:, 1:2]
        inst_d = inst[:, 2:3]
        return sem, inst_b, inst_c, inst_d

    def forward(self, x, training_mode=True):
        """
        training_mode=True  -> returns both anisotropic and isotropic branch
                                outputs (Algorithm 1).
        training_mode=False -> inference: only the anisotropic branch is
                                computed (h and AFG discarded), per Algorithm 2.
        """
        feats = self.encoder(x)
        if self.afg_enabled:
            feats = self._apply_afg(feats)

        ani_sem, ani_b, ani_c, ani_d = self._decode_branch(feats)

        if not training_mode:
            return {
                "sem": ani_sem, "inst_b": ani_b, "inst_c": ani_c, "inst_d": ani_d,
            }

        iso_feats = self.pi_module(feats)
        iso_sem, iso_b, iso_c, iso_d = self._decode_branch(iso_feats)

        return {
            "ani_sem": ani_sem, "ani_b": ani_b, "ani_c": ani_c, "ani_d": ani_d,
            "iso_sem": iso_sem, "iso_b": iso_b, "iso_c": iso_c, "iso_d": iso_d,
        }
