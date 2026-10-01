# -*- coding: utf-8 -*-
"""
losses.py -- loss functions, matching the manuscript's equations exactly.

    Eq. (3)  L_Dice
    Eq. (4)  L_CE          (implemented via nn.CrossEntropyLoss)
    Eq. (5)/(6) L_ani,sem / L_iso,sem
    Eq. (7)  L_reiso
    Eq. (8)  L_BCE         (implemented via nn.BCEWithLogitsLoss)
    Eq. (9)  L_inst
    Eq. (15) L_total
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def dice_loss(pred_logits, target, eps=1e-5):
    """Eq. (3): L_Dice(y_hat, y) = 1 - (2*sum(y_hat*y)+eps)/(sum(y_hat)+sum(y)+eps)

    pred_logits: (N, C, D, H, W) raw logits
    target: (N, D, H, W) integer class labels
    """
    num_classes = pred_logits.shape[1]
    probs = F.softmax(pred_logits, dim=1)
    target_onehot = F.one_hot(target.long(), num_classes).permute(0, 4, 1, 2, 3).float()

    dims = (0, 2, 3, 4)
    intersection = torch.sum(probs * target_onehot, dims)
    denom = torch.sum(probs, dims) + torch.sum(target_onehot, dims)
    dice_per_class = (2.0 * intersection + eps) / (denom + eps)
    return 1.0 - dice_per_class.mean()


def semantic_loss(pred_logits, target):
    """L_ani,sem / L_iso,sem = L_Dice + L_CE  (Eqs. 5-6, using Eqs. 3-4)."""
    ce = F.cross_entropy(pred_logits, target.long())
    dice = dice_loss(pred_logits, target)
    return dice + ce


def instance_loss(pred_b, pred_c, pred_d, y_b, y_c, y_d):
    """Eq. (9): L_inst = L_BCE(B) + L_BCE(C) + ||D_hat - D||_2^2

    pred_b, pred_c: raw logits (N,1,D,H,W); y_b, y_c: binary targets
    pred_d, y_d: signed distance transform (regression target)
    """
    bce_b = F.binary_cross_entropy_with_logits(pred_b, y_b)
    bce_c = F.binary_cross_entropy_with_logits(pred_c, y_c)
    mse_d = F.mse_loss(pred_d, y_d)
    return bce_b + bce_c + mse_d


def reiso_loss(ani_sem, iso_sem, y_sem, lam=0.5):
    """Eq. (7): L_reiso = (1-lambda)*L_ani,sem + lambda*L_iso,sem"""
    l_ani = semantic_loss(ani_sem, y_sem)
    l_iso = semantic_loss(iso_sem, y_sem)
    return (1 - lam) * l_ani + lam * l_iso, l_ani, l_iso


def total_loss(outputs, targets, lam=0.5, gamma=1.0):
    """Eq. (15): L_total = (1-lambda)(L_ani,sem + gamma*L_ani,inst)
                          + lambda*(L_iso,sem + gamma*L_iso,inst)

    outputs: dict from JRISA.forward(training_mode=True)
    targets: dict with keys 'sem', 'inst_b', 'inst_c', 'inst_d'
    """
    y_sem, y_b, y_c, y_d = targets["sem"], targets["inst_b"], targets["inst_c"], targets["inst_d"]

    l_ani_sem = semantic_loss(outputs["ani_sem"], y_sem)
    l_iso_sem = semantic_loss(outputs["iso_sem"], y_sem)
    l_ani_inst = instance_loss(outputs["ani_b"], outputs["ani_c"], outputs["ani_d"], y_b, y_c, y_d)
    l_iso_inst = instance_loss(outputs["iso_b"], outputs["iso_c"], outputs["iso_d"], y_b, y_c, y_d)

    total = (1 - lam) * (l_ani_sem + gamma * l_ani_inst) + lam * (l_iso_sem + gamma * l_iso_inst)

    return total, {
        "L_ani_sem": l_ani_sem.item(), "L_iso_sem": l_iso_sem.item(),
        "L_ani_inst": l_ani_inst.item(), "L_iso_inst": l_iso_inst.item(),
        "L_total": total.item(),
    }
