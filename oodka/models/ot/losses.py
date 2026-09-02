"""OT teacher distillation objectives."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightedCosineDistillation(nn.Module):
    def __init__(self, eps: float = 1e-8) -> None:
        super().__init__()
        self.eps = float(eps)

    def forward(
        self,
        student_tokens: torch.Tensor,
        teacher_tokens: torch.Tensor,
        weight: torch.Tensor,
    ) -> torch.Tensor:
        if student_tokens.shape != teacher_tokens.shape:
            raise ValueError(
                f"Student/teacher shapes differ: {student_tokens.shape}, "
                f"{teacher_tokens.shape}"
            )
        if weight.shape != student_tokens.shape[:2]:
            raise ValueError(
                f"weight must be {student_tokens.shape[:2]}, got {weight.shape}"
            )
        student = F.normalize(student_tokens.float(), dim=-1, eps=self.eps)
        teacher = F.normalize(
            teacher_tokens.detach().float(), dim=-1, eps=self.eps
        )
        distance = 1.0 - (student * teacher).sum(dim=-1)
        weight = weight.detach().float().clamp_min(0.0)
        return (weight * distance).sum() / weight.sum().clamp_min(self.eps)


class WeightedLogRMSAlignment(nn.Module):
    """Match token energy while remaining stable across absolute scales."""

    def __init__(self, eps: float = 1e-8) -> None:
        super().__init__()
        self.eps = float(eps)

    def forward(
        self,
        student_tokens: torch.Tensor,
        teacher_tokens: torch.Tensor,
        weight: torch.Tensor,
    ) -> torch.Tensor:
        if student_tokens.shape != teacher_tokens.shape:
            raise ValueError(
                f"Student/teacher shapes differ: {student_tokens.shape}, "
                f"{teacher_tokens.shape}"
            )
        if weight.shape != student_tokens.shape[:2]:
            raise ValueError(
                f"weight must be {student_tokens.shape[:2]}, got {weight.shape}"
            )
        student_rms = student_tokens.float().square().mean(dim=-1).add(
            self.eps
        ).sqrt()
        teacher_rms = teacher_tokens.detach().float().square().mean(
            dim=-1
        ).add(self.eps).sqrt()
        distance = F.smooth_l1_loss(
            student_rms.log(),
            teacher_rms.log(),
            reduction="none",
        )
        weight = weight.detach().float().clamp_min(0.0)
        return (weight * distance).sum() / weight.sum().clamp_min(self.eps)
