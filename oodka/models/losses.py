"""Loss functions for OODKA training."""

from __future__ import annotations

import math
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


def spatial_cka_loss(
    X: torch.Tensor, Y: torch.Tensor,
    max_samples: int = 8192, eps: float = 1e-6,
    zscore: bool = True,
    valid_z: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Spatial linear CKA loss (1 - CKA)."""
    assert X.shape == Y.shape and X.ndim == 5
    B, C, D, H, W = X.shape
    S = D * H * W

    if valid_z is None or bool(valid_z.all()):
        Xs = X.flatten(2)
        Ys = Y.flatten(2)
        m = min(int(math.ceil(max_samples / max(B, 1))), S)
        idx = torch.randint(0, S, (m,), device=X.device)
        Xs = Xs.index_select(2, idx)
        Ys = Ys.index_select(2, idx)
        Xf = Xs.permute(0, 2, 1).reshape(-1, C)
        Yf = Ys.permute(0, 2, 1).reshape(-1, C)
    else:
        if valid_z.shape != (B, D):
            raise ValueError(f"valid_z must be [B,D]={B,D}, got {valid_z.shape}")
        spatial_mask = (
            valid_z[:, :, None, None]
            .expand(B, D, H, W)
            .reshape(-1)
        )
        Xf = X.permute(0, 2, 3, 4, 1).reshape(-1, C)[spatial_mask]
        Yf = Y.permute(0, 2, 3, 4, 1).reshape(-1, C)[spatial_mask]
        if Xf.shape[0] > max_samples:
            idx = torch.randperm(Xf.shape[0], device=X.device)[:max_samples]
            Xf = Xf.index_select(0, idx)
            Yf = Yf.index_select(0, idx)
    Xf = Xf - Xf.mean(0, keepdim=True)
    Yf = Yf - Yf.mean(0, keepdim=True)
    if zscore:
        Xf = Xf / (Xf.std(0, keepdim=True) + eps)
        Yf = Yf / (Yf.std(0, keepdim=True) + eps)

    XtY = Xf.t() @ Yf
    XtX = Xf.t() @ Xf
    YtY = Yf.t() @ Yf
    cka = (XtY * XtY).sum() / (torch.sqrt((XtX * XtX).sum() * (YtY * YtY).sum()) + eps)
    return 1.0 - cka.clamp(0.0, 1.0)


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


def entropy_loss(p: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    ent = -(p * torch.log(p + eps) + (1 - p) * torch.log(1 - p + eps))
    return -ent.mean()
