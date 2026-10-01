# -*- coding: utf-8 -*-
"""
data.py -- dataset loading, BCD label generation, patch extraction, augmentation.

Generates the three instance-representation channels (foreground B, contour C,
signed distance transform D) from an integer-labeled instance volume, per
Section III-F. Extracts random 16x320x320 (or 16x384x320 for CREMI) patches
from full volumes with simple flip/rotation augmentation, per Algorithm 1
line 4 and Section III-I.

IMPORTANT (see the "annotations problem" discussion): `generate_bcd_labels`
asserts that the input looks like an INSTANCE-labeled volume (multiple
distinct positive integer IDs), not a binary semantic mask. If your
downloaded ground truth turns out to be semantic-only, this will raise a
clear error rather than silently producing wrong/merged instances -- fix the
annotation source before proceeding, don't suppress this check.
"""
import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.ndimage import distance_transform_edt, binary_erosion


class AnnotationFormatError(RuntimeError):
    """Raised when the provided ground truth doesn't look like a valid
    instance-labeled volume (see module docstring)."""
    pass


def generate_bcd_labels(instance_labels: np.ndarray):
    """
    instance_labels: (D,H,W) int array, 0 = background, 1..K = distinct
                      object instances.

    Returns: (B, C, D_map) each of shape (D,H,W), float32:
        B      -- binary foreground (any instance)
        C      -- binary contour (instance boundary voxels)
        D_map  -- signed distance transform (positive inside instances,
                   magnitude = distance to nearest boundary)
    """
    unique_ids = np.unique(instance_labels)
    n_instances = (unique_ids != 0).sum()
    if n_instances == 0:
        raise AnnotationFormatError(
            "generate_bcd_labels: no positive instance IDs found. This "
            "usually means the ground truth is empty for this patch, or "
            "(more seriously) the annotation source is semantic-only rather "
            "than instance-labeled -- verify the annotation channel before "
            "proceeding (see the annotation-risk notes for this dataset)."
        )
    if n_instances == 1 and instance_labels.max() == 1 and set(unique_ids) == {0, 1}:
        # A single binary mask with no distinct IDs is a common symptom of
        # accidentally loading a semantic mask as if it were instance labels.
        import warnings
        warnings.warn(
            "generate_bcd_labels: only a single instance ID (1) was found in "
            "this patch. If this happens across MANY patches (not just one "
            "sparse crop), your annotation source is likely semantic-only, "
            "not instance-labeled -- double check before trusting BCD labels "
            "derived from it.", RuntimeWarning
        )

    foreground = (instance_labels > 0).astype(np.float32)

    contour = np.zeros_like(foreground)
    dist_map = np.zeros_like(foreground)
    for inst_id in unique_ids:
        if inst_id == 0:
            continue
        mask = instance_labels == inst_id
        eroded = binary_erosion(mask)
        contour = np.logical_or(contour, np.logical_and(mask, ~eroded))
        dist_map += distance_transform_edt(mask)

    return foreground, contour.astype(np.float32), dist_map.astype(np.float32)


class EMVolumeDataset(Dataset):
    """
    Random-patch 3D dataset over one or more (raw, instance_label) volume
    pairs. BCD labels are generated once per volume at construction time
    (not per-patch) for efficiency.

    volumes: list of dicts, each {"raw": np.ndarray (D,H,W), "labels": np.ndarray (D,H,W)}
    patch_size: (d, h, w) -- e.g. (16, 320, 320) per Section III-I
    patches_per_epoch: fixed iteration count per epoch (a training-loop
                       design choice, NOT copied from [26]/[18] -- see the
                       discussion on reducing "iterations/epoch" for compute
                       budgeting; default follows common nnU-Net-style practice)
    """
    def __init__(self, volumes, patch_size=(16, 320, 320), patches_per_epoch=250, augment=True):
        self.volumes = volumes
        self.patch_size = patch_size
        self.patches_per_epoch = patches_per_epoch
        self.augment = augment

        self.bcd_cache = []
        for v in self.volumes:
            b, c, d = generate_bcd_labels(v["labels"])
            self.bcd_cache.append((b, c, d))

        for v in self.volumes:
            for axis, (dim_size, patch_size_axis) in enumerate(zip(v["raw"].shape, patch_size)):
                if dim_size < patch_size_axis:
                    raise ValueError(
                        f"Volume shape {v['raw'].shape} is smaller than patch "
                        f"size {patch_size} along axis {axis}; cannot extract patches."
                    )

    def __len__(self):
        return self.patches_per_epoch

    def _random_crop_coords(self, vol_shape):
        pd, ph, pw = self.patch_size
        d, h, w = vol_shape
        z0 = np.random.randint(0, d - pd + 1)
        y0 = np.random.randint(0, h - ph + 1)
        x0 = np.random.randint(0, w - pw + 1)
        return z0, y0, x0

    def __getitem__(self, idx):
        v_idx = np.random.randint(0, len(self.volumes))
        raw = self.volumes[v_idx]["raw"]
        b_full, c_full, d_full = self.bcd_cache[v_idx]
        sem_full = self.volumes[v_idx].get("semantic", (self.volumes[v_idx]["labels"] > 0).astype(np.int64))

        pd, ph, pw = self.patch_size
        z0, y0, x0 = self._random_crop_coords(raw.shape)
        sl = (slice(z0, z0 + pd), slice(y0, y0 + ph), slice(x0, x0 + pw))

        raw_patch = raw[sl].astype(np.float32)
        sem_patch = sem_full[sl].astype(np.int64)
        b_patch = b_full[sl]
        c_patch = c_full[sl]
        d_patch = d_full[sl]

        if self.augment:
            raw_patch, sem_patch, b_patch, c_patch, d_patch = self._augment(
                raw_patch, sem_patch, b_patch, c_patch, d_patch
            )

        # normalize raw intensities to [0,1]
        raw_patch = (raw_patch - raw_patch.min()) / (raw_patch.max() - raw_patch.min() + 1e-8)

        return {
            "image": torch.from_numpy(raw_patch[None]).float(),       # (1,D,H,W)
            "sem": torch.from_numpy(sem_patch).long(),                 # (D,H,W)
            "inst_b": torch.from_numpy(b_patch[None]).float(),
            "inst_c": torch.from_numpy(c_patch[None]).float(),
            "inst_d": torch.from_numpy(d_patch[None]).float(),
        }

    @staticmethod
    def _augment(raw, sem, b, c, d):
        """Simple flip + 90-degree-rotation augmentation (lateral axes only,
        the depth axis is left un-rotated to respect anisotropy)."""
        if np.random.rand() < 0.5:
            raw, sem, b, c, d = (np.flip(a, axis=1).copy() for a in (raw, sem, b, c, d))
        if np.random.rand() < 0.5:
            raw, sem, b, c, d = (np.flip(a, axis=2).copy() for a in (raw, sem, b, c, d))
        k = np.random.randint(0, 4)
        if k > 0:
            raw, sem, b, c, d = (np.rot90(a, k, axes=(1, 2)).copy() for a in (raw, sem, b, c, d))
        return raw, sem, b, c, d
