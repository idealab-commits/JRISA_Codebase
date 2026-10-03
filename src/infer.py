# -*- coding: utf-8 -*-
"""
infer.py -- Algorithm 2 (Inference and Instance Decoding).

Runs the trained model on a full volume via overlapping sliding-window
tiling (since a full volume is far larger than one training patch), then
decodes instances via marker-controlled watershed (Eq. 10). Only the
anisotropic branch is computed -- h and AFG are discarded, per Algorithm 2.

Usage:
    python infer.py --checkpoint ./runs/ac3ac4_full_seed0/best.pt \\
        --volume ./data/ac3_ac4/ac4_raw.npy --out-dir ./predictions/ac4_full_seed0
"""
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from model import JRISA
from decode import decode_instances


def sliding_window_infer(model, volume, patch_size, device, overlap=0.25):
    """Overlapping tiled inference with averaged logits in overlap regions."""
    d, h, w = volume.shape
    pd, ph, pw = patch_size
    stride = tuple(max(1, int(p * (1 - overlap))) for p in patch_size)

    sem_accum = torch.zeros((2, d, h, w))
    inst_accum = torch.zeros((3, d, h, w))
    count = torch.zeros((1, d, h, w))

    z_starts = list(range(0, max(d - pd, 0) + 1, stride[0])) or [0]
    y_starts = list(range(0, max(h - ph, 0) + 1, stride[1])) or [0]
    x_starts = list(range(0, max(w - pw, 0) + 1, stride[2])) or [0]
    if z_starts[-1] + pd < d:
        z_starts.append(d - pd)
    if y_starts[-1] + ph < h:
        y_starts.append(h - ph)
    if x_starts[-1] + pw < w:
        x_starts.append(w - pw)

    model.eval()
    with torch.no_grad():
        for z0 in z_starts:
            for y0 in y_starts:
                for x0 in x_starts:
                    patch = volume[z0:z0 + pd, y0:y0 + ph, x0:x0 + pw]
                    patch = (patch - patch.min()) / (patch.max() - patch.min() + 1e-8)
                    tensor = torch.from_numpy(patch[None, None]).float().to(device)
                    out = model(tensor, training_mode=False)
                    sem_accum[:, z0:z0 + pd, y0:y0 + ph, x0:x0 + pw] += out["sem"][0].cpu()
                    inst_b = out["inst_b"][0].cpu()
                    inst_c = out["inst_c"][0].cpu()
                    inst_d = out["inst_d"][0].cpu()
                    inst_accum[0:1, z0:z0 + pd, y0:y0 + ph, x0:x0 + pw] += inst_b
                    inst_accum[1:2, z0:z0 + pd, y0:y0 + ph, x0:x0 + pw] += inst_c
                    inst_accum[2:3, z0:z0 + pd, y0:y0 + ph, x0:x0 + pw] += inst_d
                    count[:, z0:z0 + pd, y0:y0 + ph, x0:x0 + pw] += 1

    count = torch.clamp(count, min=1)
    sem_avg = sem_accum / count
    inst_avg = inst_accum / count
    return sem_avg.numpy(), inst_avg.numpy()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--volume", required=True, help="Path to a <name>_raw.npy volume")
    p.add_argument("--dataset", choices=["ac3ac4", "cremi", "mitoem"], default="ac3ac4")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--tau-b", type=float, default=0.5)
    p.add_argument("--tau-c", type=float, default=0.5)
    args = p.parse_args()

    patch_size = (16, 384, 320) if args.dataset == "cremi" else (16, 320, 320)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.checkpoint, map_location=device)
    # A full_afg checkpoint contains the gate's weights, so the model must be built with the
    # gate to load it. The pseudo-isotropic module h is never used here (inference runs the
    # anisotropic branch only, training_mode=False).
    afg = ckpt.get("args", {}).get("config_name") == "full_afg"
    model = JRISA(in_channels=1, num_sem_classes=2, afg_enabled=afg).to(device)
    model.load_state_dict(ckpt["model"])

    volume = np.load(args.volume).astype(np.float32)
    print(f"Running sliding-window inference on volume shape {volume.shape} ...")
    sem_logits, inst_out = sliding_window_infer(model, volume, patch_size, device)

    sem_pred = np.argmax(sem_logits, axis=0).astype(np.uint8)
    b_prob = 1 / (1 + np.exp(-inst_out[0]))  # sigmoid
    c_prob = 1 / (1 + np.exp(-inst_out[1]))
    d_pred = inst_out[2]

    instances = decode_instances(b_prob, c_prob, d_pred, tau_b=args.tau_b, tau_c=args.tau_c)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "sem_pred.npy", sem_pred)
    np.save(out_dir / "instances.npy", instances)
    print(f"Saved semantic map and {instances.max()} instances to {out_dir}")


if __name__ == "__main__":
    main()
