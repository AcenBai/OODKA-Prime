"""Multiscale P-Balanced/S-Unbalanced OT training objective."""

from __future__ import annotations

from typing import Dict, Sequence, Tuple

import torch
import torch.nn as nn

from .cost import OTCostBuilder, _coordinates
from .losses import WeightedCosineDistillation
from .mass import ResidualMassBuilder, StructureMassBuilder
from .sinkhorn import BalancedSinkhorn, UnbalancedSinkhorn
from .transport import BarycentricProjector


class MultiScaleOTDistillation(nn.Module):
    """Build dynamic no-grad transports and differentiable student losses."""

    def __init__(
        self,
        *,
        levels: Sequence[int] = (2, 3, 4, 5),
        max_grid_size: int = 32,
        feature_weight: float = 1.0,
        coordinate_weight: float = 0.25,
        coordinate_radius: float = 0.25,
        p_semantic_weight: float = 0.25,
        s_gain_mode: str = "smooth_advantage",
        s_gain_temperature: float = 0.5,
        p_epsilon: float = 0.1,
        s_epsilon: float = 0.1,
        rho_base: float = 1.0,
        rho_expert: float = 0.2,
        sinkhorn_iterations: int = 30,
        min_received_mass: float = 1e-6,
        relative_kd: bool = False,
        relative_kd_expert_weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.levels = tuple(int(level) for level in levels)
        if not self.levels:
            raise ValueError("levels cannot be empty")
        if len(set(self.levels)) != len(self.levels):
            raise ValueError(f"levels must be unique, got {self.levels}")
        self.max_grid_size = int(max_grid_size)
        if self.max_grid_size <= 0:
            raise ValueError("max_grid_size must be positive")
        self.coordinate_radius = float(coordinate_radius)
        self.structure_mass = StructureMassBuilder()
        self.residual_mass = ResidualMassBuilder(
            gain_mode=s_gain_mode,
            gain_temperature=s_gain_temperature,
        )
        self.p_cost = OTCostBuilder(
            feature_weight=feature_weight,
            coordinate_weight=coordinate_weight,
            coordinate_radius=coordinate_radius,
            semantic_weight=p_semantic_weight,
        )
        self.s_cost = OTCostBuilder(
            feature_weight=feature_weight,
            coordinate_weight=coordinate_weight,
            coordinate_radius=coordinate_radius,
            semantic_weight=0.0,
        )
        self.balanced = BalancedSinkhorn(
            epsilon=p_epsilon, iterations=sinkhorn_iterations
        )
        self.unbalanced = UnbalancedSinkhorn(
            epsilon=s_epsilon,
            rho_base=rho_base,
            rho_expert=rho_expert,
            iterations=sinkhorn_iterations,
        )
        self.projector = BarycentricProjector()
        self.distillation = WeightedCosineDistillation()
        self.min_received_mass = float(min_received_mass)
        self.relative_kd = bool(relative_kd)
        self.relative_kd_expert_weight = float(relative_kd_expert_weight)
        if self.relative_kd_expert_weight < 0.0:
            raise ValueError("relative_kd_expert_weight must be non-negative")

    @staticmethod
    def _valid_feature_slices(
        feature: torch.Tensor, valid_flat: torch.Tensor
    ) -> torch.Tensor:
        if feature.ndim != 5:
            raise ValueError(f"feature must be [B,C,Z,H,W], got {feature.shape}")
        flat = (
            feature.permute(0, 2, 1, 3, 4)
            .reshape(
                feature.shape[0] * feature.shape[2],
                feature.shape[1],
                feature.shape[3],
                feature.shape[4],
            )
            .contiguous()
        )
        return flat[valid_flat]

    def _target_size(self, feature: torch.Tensor) -> Tuple[int, int]:
        """Cap each native spatial dimension without ever upsampling it."""
        return (
            min(int(feature.shape[-2]), self.max_grid_size),
            min(int(feature.shape[-1]), self.max_grid_size),
        )

    def _transport_geometry(
        self,
        transport: torch.Tensor,
        base_grid: Tuple[int, int],
        expert_grid: Tuple[int, int],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return mass-weighted distance and mass outside the free radius."""
        with torch.no_grad():
            base_coords = _coordinates(
                *base_grid,
                device=transport.device,
                dtype=transport.dtype,
            )
            expert_coords = _coordinates(
                *expert_grid,
                device=transport.device,
                dtype=transport.dtype,
            )
            distance = torch.cdist(base_coords, expert_coords).unsqueeze(0)
            total = transport.sum(dim=(-2, -1)).clamp_min(1e-8)
            mean_distance = (
                (transport * distance).sum(dim=(-2, -1)) / total
            ).mean()
            outside_ratio = (
                (
                    transport
                    * (distance > self.coordinate_radius).to(transport)
                ).sum(dim=(-2, -1))
                / total
            ).mean()
        return mean_distance, outside_ratio

    def forward(
        self,
        features: Dict[str, torch.Tensor],
        *,
        gt: torch.Tensor,
        base_error: torch.Tensor,
        expert_error: torch.Tensor,
        valid_z: torch.Tensor,
        class_ids: Sequence[int],
        enable_p: bool = True,
        enable_s: bool = True,
        expert_perturbation: str | None = None,
    ) -> dict:
        """Compute multiscale losses from adapter-aligned P/S features.

        ``gt``, ``base_error`` and ``expert_error`` are ``[B,Z,H,W]``.
        Invalid repeated tail slices are filtered before any OT computation.
        """
        if gt.ndim != 4 or base_error.shape != gt.shape or expert_error.shape != gt.shape:
            raise ValueError("GT and error maps must share [B,Z,H,W]")
        if valid_z.shape != gt.shape[:2]:
            raise ValueError(
                f"valid_z must be {gt.shape[:2]}, got {valid_z.shape}"
            )
        valid_flat = valid_z.reshape(-1).bool()
        zero = next(iter(features.values())).sum() * 0.0
        if not valid_flat.any() or not (enable_p or enable_s):
            return {"loss_p": zero, "loss_s": zero, "levels": {}}

        gt_valid = gt.reshape(-1, gt.shape[-2], gt.shape[-1])[valid_flat]
        base_error_valid = base_error.reshape(
            -1, base_error.shape[-2], base_error.shape[-1]
        )[valid_flat]
        expert_error_valid = expert_error.reshape(
            -1, expert_error.shape[-2], expert_error.shape[-1]
        )[valid_flat]
        semantic = torch.stack(
            [(gt_valid == int(class_id)).float() for class_id in class_ids], dim=1
        )

        p_losses = []
        s_losses = []
        p_reverse_losses = []
        s_reverse_losses = []
        level_logs = {}
        for level in sorted(self.levels):
            controlled_s_cost_offsets = {
                "s_cost_offset_0p25": 0.25,
                "s_cost_offset_0p5": 0.5,
                "s_cost_offset_1p0": 1.0,
                "s_cost_offset_2p0": 2.0,
            }
            s_cost_offset = controlled_s_cost_offsets.get(
                expert_perturbation, 0.0
            )
            p_base = self._valid_feature_slices(
                features[f"Zb{level}_p"], valid_flat
            )
            s_base = self._valid_feature_slices(
                features[f"Zb{level}_s"], valid_flat
            )
            p_expert = self._valid_feature_slices(
                features[f"Zn{level}_p"], valid_flat
            )
            s_expert = self._valid_feature_slices(
                features[f"Zn{level}_s"], valid_flat
            )
            target_size = self._target_size(p_base)
            if expert_perturbation == "spatial_shift":
                shift = (
                    max(1, p_expert.shape[-2] // 4),
                    max(1, p_expert.shape[-1] // 4),
                )
                p_expert = torch.roll(p_expert, shifts=shift, dims=(-2, -1))
                s_expert = torch.roll(s_expert, shifts=shift, dims=(-2, -1))
            elif expert_perturbation == "channel_reverse":
                p_expert = p_expert.flip(1)
                s_expert = s_expert.flip(1)
            elif (
                expert_perturbation is not None
                and expert_perturbation not in controlled_s_cost_offsets
            ):
                raise ValueError(
                    f"Unknown expert_perturbation={expert_perturbation!r}"
                )
            logs = {
                "grid_h": torch.as_tensor(target_size[0], device=p_base.device),
                "grid_w": torch.as_tensor(target_size[1], device=p_base.device),
            }

            if enable_p:
                p_mass = self.structure_mass(
                    gt_valid,
                    p_base,
                    p_expert,
                    class_ids=class_ids,
                    target_size=target_size,
                )
                p_cost = self.p_cost(
                    p_base,
                    p_expert,
                    target_size=target_size,
                    base_semantic=semantic,
                    expert_semantic=semantic,
                )
                p_transport = self.balanced(
                    p_mass["a"], p_mass["b"], p_cost["cost"]
                )
                p_teacher = self.projector(
                    p_transport["transport"], p_cost["expert_tokens"]
                )
                p_mean_distance, p_outside_radius = self._transport_geometry(
                    p_transport["transport"],
                    p_cost["base_grid"],
                    p_cost["expert_grid"],
                )
                p_loss = self.distillation(
                    p_cost["base_tokens"], p_teacher["teacher"], p_mass["a"]
                )
                p_losses.append(p_loss)
                if self.relative_kd:
                    # The same fixed correspondence is reused in reverse:
                    # base/student values teach expert tokens, while neither
                    # the transport nor the teacher values receive gradients.
                    p_reverse_teacher = self.projector(
                        p_transport["transport"].transpose(1, 2),
                        p_cost["base_tokens"],
                    )
                    p_reverse_loss = self.distillation(
                        p_cost["expert_tokens"],
                        p_reverse_teacher["teacher"],
                        p_reverse_teacher["received"],
                    )
                    p_reverse_losses.append(p_reverse_loss)
                else:
                    p_reverse_loss = zero
                logs.update(
                    p_loss=p_loss.detach(),
                    p_reverse_loss=p_reverse_loss.detach(),
                    p_cost=p_transport["cost"].mean(),
                    p_row_error=p_transport["row_error"].mean(),
                    p_col_error=p_transport["col_error"].mean(),
                    p_entropy=p_transport["entropy"].mean(),
                    p_mean_distance=p_mean_distance,
                    p_outside_radius=p_outside_radius,
                )

            if enable_s:
                s_mass = self.residual_mass(
                    p_base,
                    s_base,
                    p_expert,
                    s_expert,
                    base_error=base_error_valid,
                    expert_error=expert_error_valid,
                    target_size=target_size,
                )
                s_cost = self.s_cost(
                    s_base, s_expert, target_size=target_size
                )
                s_cost_value = s_cost["cost"] + s_cost_offset
                s_transport = self.unbalanced(
                    s_mass["a"], s_mass["b"], s_cost_value
                )
                s_teacher = self.projector(
                    s_transport["transport"], s_cost["expert_tokens"]
                )
                s_mean_distance, s_outside_radius = self._transport_geometry(
                    s_transport["transport"],
                    s_cost["base_grid"],
                    s_cost["expert_grid"],
                )
                received_total = s_transport["received"].sum()
                if received_total.detach().item() > self.min_received_mass:
                    s_loss = self.distillation(
                        s_cost["base_tokens"],
                        s_teacher["teacher"],
                        s_transport["received"],
                    )
                else:
                    s_loss = s_cost["base_tokens"].sum() * 0.0
                s_losses.append(s_loss)
                if self.relative_kd and received_total.detach().item() > self.min_received_mass:
                    s_reverse_teacher = self.projector(
                        s_transport["transport"].transpose(1, 2),
                        s_cost["base_tokens"],
                    )
                    s_reverse_loss = self.distillation(
                        s_cost["expert_tokens"],
                        s_reverse_teacher["teacher"],
                        s_reverse_teacher["received"],
                    )
                    s_reverse_losses.append(s_reverse_loss)
                else:
                    s_reverse_loss = zero
                logs.update(
                    s_loss=s_loss.detach(),
                    s_reverse_loss=s_reverse_loss.detach(),
                    s_cost=s_transport["cost"].mean(),
                    s_received=s_transport["received"].sum(dim=-1).mean(),
                    s_transported=s_transport["transported"].sum(dim=-1).mean(),
                    s_rejected=s_transport["rejected"].sum(dim=-1).mean(),
                    s_accept_ratio=s_transport["accept_ratio"].mean(),
                    s_entropy=s_transport["entropy"].mean(),
                    s_gain=s_mass["gain"].mean(),
                    s_expert_better_ratio=(
                        s_mass["advantage"] > 0.0
                    ).float().mean(),
                    s_mean_distance=s_mean_distance,
                    s_outside_radius=s_outside_radius,
                    s_cost_offset=torch.as_tensor(
                        s_cost_offset, device=s_cost_value.device
                    ),
                )
            level_logs[level] = logs

        loss_p_forward = torch.stack(p_losses).mean() if p_losses else zero
        loss_s_forward = torch.stack(s_losses).mean() if s_losses else zero
        loss_p_reverse = (
            torch.stack(p_reverse_losses).mean() if p_reverse_losses else zero
        )
        loss_s_reverse = (
            torch.stack(s_reverse_losses).mean() if s_reverse_losses else zero
        )
        reverse_scale = self.relative_kd_expert_weight if self.relative_kd else 0.0
        loss_p = loss_p_forward + reverse_scale * loss_p_reverse
        loss_s = loss_s_forward + reverse_scale * loss_s_reverse
        return {
            "loss_p": loss_p,
            "loss_s": loss_s,
            "loss_p_forward": loss_p_forward,
            "loss_s_forward": loss_s_forward,
            "loss_p_reverse": loss_p_reverse,
            "loss_s_reverse": loss_s_reverse,
            "levels": level_logs,
        }
