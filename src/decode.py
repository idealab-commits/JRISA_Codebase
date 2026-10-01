# -*- coding: utf-8 -*-
"""
decode.py -- marker-controlled watershed instance decoding (Eq. 10, Section III-F).

    {p_1,...,p_K} = Watershed(-D_hat, markers=S, mask={B_hat > tau_B})
    S = ConnectedComponents({B_hat > tau_B} AND {C_hat <= tau_C})
"""
import numpy as np
from scipy import ndimage as ndi
from skimage.segmentation import watershed


def decode_instances(b_prob, c_prob, d_pred, tau_b=0.5, tau_c=0.5):
    """
    b_prob, c_prob: (D,H,W) foreground / contour probabilities in [0,1]
                    (apply sigmoid to raw logits before calling this)
    d_pred:         (D,H,W) predicted signed distance transform

    Returns: (D,H,W) int32 label volume, 0 = background, 1..K = instances.
    """
    fg_mask = b_prob > tau_b
    contour_mask = c_prob > tau_c

    # S = ConnectedComponents(foreground AND NOT contour) -- Section III-F / Eq. (10)
    seed_mask = np.logical_and(fg_mask, np.logical_not(contour_mask))
    markers, num_markers = ndi.label(seed_mask)

    if num_markers == 0:
        # No confident seeds found -- fall back to treating the whole
        # foreground mask as unlabeled background rather than crashing.
        return np.zeros_like(fg_mask, dtype=np.int32)

    # Watershed floods from the seeds using -D_hat as the topographic
    # surface (basins = instance interiors), restricted to the foreground mask.
    labels = watershed(-d_pred, markers=markers, mask=fg_mask)
    return labels.astype(np.int32)
