"""Token pooling and pairwise OT cost construction."""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def pool_feature_map(
    feature: torch.Tensor, target_size: Tuple[int, int]
) -> torch.Tensor:
    """Pool ``[M,C,H,W]`` to an OT grid while preserving raw magnitudes."""
    if feature.ndim != 4:
        raise ValueError(f"feature must be [M,C,H,W], got {feature.shape}")
    target = (int(target_size[0]), int(target_size[1]))
    if min(target) <= 0:
        raise ValueError(f"target_size must be positive, got {target}")
    if feature.shape[-2:] == target:
        return feature
    return F.adaptive_avg_pool2d(feature, target)


def _tokens(feature: torch.Tensor) -> torch.Tensor:
    return feature.flatten(2).transpose(1, 2).contiguous()


def _coordinates(
    height: int, width: int, *, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    y = torch.linspace(-1.0, 1.0, height, device=device, dtype=dtype)
    x = torch.linspace(-1.0, 1.0, width, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    return torch.stack((yy, xx), dim=-1).reshape(-1, 2)


class OTCostBuilder(nn.Module):
    """Construct feature/coordinate/optional semantic costs in float32."""

    def __init__(
        self,
        feature_weight: float = 1.0,
        coordinate_weight: float = 0.1,
        semantic_weight: float = 0.0,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.feature_weight = float(feature_weight)
        self.coordinate_weight = float(coordinate_weight)
        self.semantic_weight = float(semantic_weight)
        self.eps = float(eps)

    def forward(
        self,
        base_feature: torch.Tensor,
        expert_feature: torch.Tensor,
        *,
        target_size: Tuple[int, int],
        base_semantic: Optional[torch.Tensor] = None,
        expert_semantic: Optional[torch.Tensor] = None,
    ) -> dict:
        """Return cost and pooled raw tokens.

        Base/expert inputs are adapter-aligned ``[M,C,H,W]`` features. Raw
        pooled tokens are retained for mass and teacher construction; only
        normalized copies are used for feature cost.
        """
        if base_feature.ndim != 4 or expert_feature.ndim != 4:
            raise ValueError("base_feature and expert_feature must be 4D")
        if base_feature.shape[:2] != expert_feature.shape[:2]:
            raise ValueError(
                "Adapter-aligned base/expert batch and channels must match, got "
                f"{base_feature.shape[:2]} and {expert_feature.shape[:2]}"
            )

        with torch.autocast(device_type=base_feature.device.type, enabled=False):
            base_pooled = pool_feature_map(base_feature.float(), target_size)
            expert_pooled = pool_feature_map(expert_feature.float(), target_size)
            base_tokens = _tokens(base_pooled)
            expert_tokens = _tokens(expert_pooled)
            base_norm = F.normalize(base_tokens.detach(), dim=-1, eps=self.eps)
            expert_norm = F.normalize(
                expert_tokens.detach(), dim=-1, eps=self.eps
            )
            feature_cost = 1.0 - torch.bmm(
                base_norm, expert_norm.transpose(1, 2)
            )
            cost = self.feature_weight * feature_cost.clamp_min(0.0)

            hb, wb = base_pooled.shape[-2:]
            he, we = expert_pooled.shape[-2:]
            if self.coordinate_weight:
                coord_b = _coordinates(
                    hb, wb, device=cost.device, dtype=cost.dtype
                )
                coord_e = _coordinates(
                    he, we, device=cost.device, dtype=cost.dtype
                )
                coordinate_cost = torch.cdist(coord_b, coord_e).square()
                cost = cost + self.coordinate_weight * coordinate_cost.unsqueeze(0)

            if self.semantic_weight:
                if base_semantic is None or expert_semantic is None:
                    raise ValueError(
                        "semantic tensors are required when semantic_weight is nonzero"
                    )
                base_semantic = F.adaptive_avg_pool2d(
                    base_semantic.float(), (hb, wb)
                ).flatten(2).transpose(1, 2)
                expert_semantic = F.adaptive_avg_pool2d(
                    expert_semantic.float(), (he, we)
                ).flatten(2).transpose(1, 2)
                base_semantic = F.normalize(
                    base_semantic, p=1, dim=-1, eps=self.eps
                )
                expert_semantic = F.normalize(
                    expert_semantic, p=1, dim=-1, eps=self.eps
                )
                semantic_cost = 1.0 - torch.bmm(
                    base_semantic, expert_semantic.transpose(1, 2)
                )
                cost = cost + self.semantic_weight * semantic_cost.clamp_min(0.0)

        if not torch.isfinite(cost).all():
            raise FloatingPointError("OT cost contains NaN or Inf")
        return {
            "cost": cost,
            "base_tokens": base_tokens,
            "expert_tokens": expert_tokens,
            "base_grid": (hb, wb),
            "expert_grid": (he, we),
        }
