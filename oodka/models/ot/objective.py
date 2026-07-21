"""Multiscale P-Balanced/S-Unbalanced OT training objective."""

from __future__ import annotations

from typing import Dict, Mapping, Sequence, Tuple

import torch
import torch.nn as nn

from .cost import OTCostBuilder
from .losses import WeightedCosineDistillation
from .mass import ResidualMassBuilder, StructureMassBuilder
from .sinkhorn import BalancedSinkhorn, UnbalancedSinkhorn
from .transport import BarycentricProjector


class MultiScaleOTDistillation(nn.Module):
    """Build dynamic no-grad transports and differentiable student losses."""

    def __init__(
        self,
        grids: Mapping[int, Tuple[int, int]] | None = None,
        *,
        feature_weight: float = 1.0,
        coordinate_weight: float = 0.1,
        p_semantic_weight: float = 0.25,
        p_epsilon: float = 0.1,
        s_epsilon: float = 0.1,
        rho_base: float = 1.0,
        rho_expert: float = 0.2,
        sinkhorn_iterations: int = 30,
        min_received_mass: float = 1e-6,
    ) -> None:
        super().__init__()
        self.grids = dict(
            grids
            or {
                2: (16, 16),
                3: (16, 16),
                4: (32, 32),
                5: (16, 16),
            }
        )
        self.structure_mass = StructureMassBuilder()
        self.residual_mass = ResidualMassBuilder()
        self.p_cost = OTCostBuilder(
            feature_weight=feature_weight,
            coordinate_weight=coordinate_weight,
            semantic_weight=p_semantic_weight,
        )
        self.s_cost = OTCostBuilder(
            feature_weight=feature_weight,
            coordinate_weight=coordinate_weight,
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
        level_logs = {}
        for level in sorted(self.grids):
            target_size = self.grids[level]
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
            logs = {}

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
                p_loss = self.distillation(
                    p_cost["base_tokens"], p_teacher["teacher"], p_mass["a"]
                )
                p_losses.append(p_loss)
                logs.update(
                    p_loss=p_loss.detach(),
                    p_cost=p_transport["cost"].mean(),
                    p_row_error=p_transport["row_error"].mean(),
                    p_col_error=p_transport["col_error"].mean(),
                    p_entropy=p_transport["entropy"].mean(),
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
                logs.update(
                    s_loss=s_loss.detach(),
                    s_cost=s_transport["cost"].mean(),
                    s_received=s_transport["received"].sum(dim=-1).mean(),
                    s_transported=s_transport["transported"].sum(dim=-1).mean(),
                    s_rejected=s_transport["rejected"].sum(dim=-1).mean(),
                    s_accept_ratio=s_transport["accept_ratio"].mean(),
                    s_entropy=s_transport["entropy"].mean(),
                    s_gain=s_mass["gain"].mean(),
                    s_cost_offset=torch.as_tensor(
                        s_cost_offset, device=s_cost_value.device
                    ),
                )
            level_logs[level] = logs

        loss_p = torch.stack(p_losses).mean() if p_losses else zero
        loss_s = torch.stack(s_losses).mean() if s_losses else zero
        return {"loss_p": loss_p, "loss_s": loss_s, "levels": level_logs}
