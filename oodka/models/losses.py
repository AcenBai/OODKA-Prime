"""Loss functions for OODKA training."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def ortho_corr_loss(
    Zp: torch.Tensor,
    Zs: torch.Tensor,
    eps: float = 1e-6,
    valid_z: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Orthogonality loss between private and shared features."""
    X = Zp.flatten(2)
    Y = Zs.flatten(2)
    if valid_z is None or bool(valid_z.all()):
        X = X - X.mean(dim=2, keepdim=True)
        Y = Y - Y.mean(dim=2, keepdim=True)
        X = X / (X.std(dim=2, keepdim=True) + eps)
        Y = Y / (Y.std(dim=2, keepdim=True) + eps)
        corr = torch.matmul(X, Y.transpose(1, 2)) / X.shape[-1]
        return corr.abs().mean()

    B, _, D, H, W = Zp.shape
    if valid_z.shape != (B, D):
        raise ValueError(f"valid_z must be [B,D]={B,D}, got {valid_z.shape}")
    mask = valid_z[:, None, :, None, None].expand(B, 1, D, H, W).flatten(2).to(X)
    count = mask.sum(dim=2, keepdim=True).clamp_min(1.0)
    X = (X - (X * mask).sum(dim=2, keepdim=True) / count) * mask
    Y = (Y - (Y * mask).sum(dim=2, keepdim=True) / count) * mask
    X = X / (torch.sqrt((X.square() * mask).sum(dim=2, keepdim=True) / count) + eps)
    Y = Y / (torch.sqrt((Y.square() * mask).sum(dim=2, keepdim=True) / count) + eps)
    corr = torch.matmul(X, Y.transpose(1, 2)) / count
    return corr.abs().mean()


def dice_loss_with_logits(
    logits: torch.Tensor, targets: torch.Tensor,
    valid: torch.Tensor, eps: float = 1e-6,
) -> torch.Tensor:
    probs = torch.sigmoid(logits) * valid.float()
    targets = targets.float() * valid.float()
    inter = (probs.flatten(1) * targets.flatten(1)).sum(1)
    union = probs.flatten(1).sum(1) + targets.flatten(1).sum(1)
    return (1.0 - (2.0 * inter + eps) / (union + eps)).mean()


def bce_loss_with_logits(
    logits: torch.Tensor, targets: torch.Tensor,
    valid: torch.Tensor,
) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(logits, targets.float(), reduction="none")
    bce = bce * valid.float()
    denom = valid.float().sum(dim=tuple(range(1, bce.ndim))).clamp_min(1.0)
    return (bce.sum(dim=tuple(range(1, bce.ndim))) / denom).mean()


def dice_score_from_logits_3d(
    logits: torch.Tensor, targets: torch.Tensor,
    valid: torch.Tensor, threshold: float = 0.5, eps: float = 1e-6,
) -> torch.Tensor:
    preds = (torch.sigmoid(logits) > threshold).float() * valid.float()
    targets = targets.float() * valid.float()
    inter = (preds.flatten(1) * targets.flatten(1)).sum(1)
    union = preds.flatten(1).sum(1) + targets.flatten(1).sum(1)
    return ((2.0 * inter + eps) / (union + eps)).mean()
