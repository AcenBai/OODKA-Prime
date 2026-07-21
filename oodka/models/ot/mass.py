"""Detached P-structure and S-residual transport mass construction."""

from __future__ import annotations

from typing import Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .cost import pool_feature_map


def normalize_mass(value: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Normalize ``[M,N]`` with a uniform per-sample zero-mass fallback."""
    if value.ndim != 2:
        raise ValueError(f"mass source must be [M,N], got {value.shape}")
    value = torch.nan_to_num(value.float(), nan=0.0, posinf=0.0, neginf=0.0)
    value = value.clamp_min(0.0)
    total = value.sum(dim=-1, keepdim=True)
    uniform = torch.full_like(value, 1.0 / max(value.shape[-1], 1))
    return torch.where(total > eps, value / total.clamp_min(eps), uniform)


def _normalized_by_mean(value: torch.Tensor, eps: float) -> torch.Tensor:
    mean = value.mean(dim=-1, keepdim=True)
    return value / mean.clamp_min(eps)


def _feature_energy(
    feature: torch.Tensor, target_size: Tuple[int, int], eps: float
) -> torch.Tensor:
    pooled = pool_feature_map(feature.float(), target_size)
    energy = pooled.square().sum(dim=1).add(eps).sqrt().flatten(1)
    return _normalized_by_mean(energy, eps)


class StructureMassBuilder(nn.Module):
    """Build GT-anchored balanced P marginals for aligned base/expert P."""

    def __init__(
        self,
        occupancy_weight: float = 1.0,
        boundary_weight: float = 1.5,
        rescue_weight: float = 0.2,
        energy_weight: float = 0.1,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        self.occupancy_weight = float(occupancy_weight)
        self.boundary_weight = float(boundary_weight)
        self.rescue_weight = float(rescue_weight)
        self.energy_weight = float(energy_weight)
        self.eps = float(eps)

    def forward(
        self,
        gt: torch.Tensor,
        p_base: torch.Tensor,
        p_expert: torch.Tensor,
        *,
        class_ids: Sequence[int],
        target_size: Tuple[int, int],
    ) -> dict:
        if gt.ndim != 3:
            raise ValueError(f"gt must be [M,H,W], got {gt.shape}")
        if p_base.shape != p_expert.shape or p_base.ndim != 4:
            raise ValueError("P base/expert features must share [M,C,H,W]")
        if p_base.shape[0] != gt.shape[0]:
            raise ValueError("GT and P feature batch sizes must match")
        if not class_ids:
            raise ValueError("class_ids cannot be empty")

        with torch.no_grad(), torch.autocast(
            device_type=p_base.device.type, enabled=False
        ):
            valid = gt >= 0
            one_hot = torch.stack(
                [(gt == int(class_id)) & valid for class_id in class_ids], dim=1
            ).float()
            occupancy = F.adaptive_avg_pool2d(one_hot, target_size)
            dilation = F.max_pool2d(one_hot, kernel_size=3, stride=1, padding=1)
            erosion = -F.max_pool2d(-one_hot, kernel_size=3, stride=1, padding=1)
            boundary = F.adaptive_avg_pool2d(
                (dilation - erosion).clamp(0.0, 1.0), target_size
            )
            rescue = F.adaptive_max_pool2d(one_hot, target_size)

            class_area = one_hot.flatten(2).sum(dim=-1)
            class_budget = torch.where(
                class_area > 0,
                class_area.clamp_min(1.0).rsqrt(),
                torch.zeros_like(class_area),
            )
            class_budget = normalize_mass(class_budget, self.eps)
            structure_by_class = (
                self.occupancy_weight * occupancy
                + self.boundary_weight * boundary
                + self.rescue_weight * rescue
            )
            structure = (
                class_budget[:, :, None, None] * structure_by_class
            ).sum(dim=1).flatten(1)

            energy_base = _feature_energy(p_base.detach(), target_size, self.eps)
            energy_expert = _feature_energy(
                p_expert.detach(), target_size, self.eps
            )
            q_base = structure * (1.0 + self.energy_weight * energy_base)
            q_expert = structure * (1.0 + self.energy_weight * energy_expert)
            no_structure = structure.sum(dim=-1, keepdim=True) <= self.eps
            q_base = torch.where(no_structure, energy_base, q_base)
            q_expert = torch.where(no_structure, energy_expert, q_expert)
            a = normalize_mass(q_base, self.eps)
            b = normalize_mass(q_expert, self.eps)

        return {
            "a": a,
            "b": b,
            "structure": structure,
            "occupancy": occupancy.sum(dim=1),
            "boundary": boundary.sum(dim=1),
            "rescue": rescue.sum(dim=1),
        }


class ResidualMassBuilder(nn.Module):
    """Build task-aware base demand and expert supply for S-UOT."""

    def __init__(self, lambda0: float = 0.1, clip_value: float = 3.0, eps: float = 1e-8):
        super().__init__()
        self.lambda0 = float(lambda0)
        self.clip_value = float(clip_value)
        self.eps = float(eps)

    def _energy_specificity(
        self,
        p_feature: torch.Tensor,
        s_feature: torch.Tensor,
        target_size: Tuple[int, int],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        p = pool_feature_map(p_feature.float(), target_size)
        s = pool_feature_map(s_feature.float(), target_size)
        p_norm = p.square().sum(dim=1).add(self.eps).sqrt().flatten(1)
        s_norm = s.square().sum(dim=1).add(self.eps).sqrt().flatten(1)
        energy = _normalized_by_mean(s_norm, self.eps)
        specificity = s_norm / (p_norm + s_norm + self.eps)
        return energy, specificity

    def forward(
        self,
        p_base: torch.Tensor,
        s_base: torch.Tensor,
        p_expert: torch.Tensor,
        s_expert: torch.Tensor,
        *,
        base_error: torch.Tensor,
        expert_error: torch.Tensor,
        target_size: Tuple[int, int],
    ) -> dict:
        if not (p_base.shape == s_base.shape == p_expert.shape == s_expert.shape):
            raise ValueError("All adapter-aligned P/S features must share shape")
        if base_error.ndim != 3 or expert_error.shape != base_error.shape:
            raise ValueError("base/expert error maps must share [M,H,W]")
        if p_base.shape[0] != base_error.shape[0]:
            raise ValueError("Feature and error-map batch sizes must match")

        with torch.no_grad(), torch.autocast(
            device_type=p_base.device.type, enabled=False
        ):
            e_base, r_base = self._energy_specificity(
                p_base.detach(), s_base.detach(), target_size
            )
            e_expert, r_expert = self._energy_specificity(
                p_expert.detach(), s_expert.detach(), target_size
            )
            difficulty = F.adaptive_avg_pool2d(
                base_error.detach().float().unsqueeze(1), target_size
            ).flatten(1)
            expert_error_pooled = F.adaptive_avg_pool2d(
                expert_error.detach().float().unsqueeze(1), target_size
            ).flatten(1)
            gain = (difficulty - expert_error_pooled).clamp_min(0.0)
            difficulty_hat = _normalized_by_mean(difficulty, self.eps).clamp(
                0.0, self.clip_value
            )
            gain_hat = _normalized_by_mean(gain, self.eps).clamp(
                0.0, self.clip_value
            )
            q_base = e_base * r_base * (self.lambda0 + difficulty_hat)
            q_expert = e_expert * r_expert * (self.lambda0 + gain_hat)
            a = normalize_mass(q_base, self.eps)
            b = normalize_mass(q_expert, self.eps)

        return {
            "a": a,
            "b": b,
            "base_energy": e_base,
            "expert_energy": e_expert,
            "base_specificity": r_base,
            "expert_specificity": r_expert,
            "difficulty": difficulty,
            "gain": gain,
        }
