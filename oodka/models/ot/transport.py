"""Barycentric feature transport."""

from __future__ import annotations

import torch
import torch.nn as nn


class BarycentricProjector(nn.Module):
    def __init__(self, eps: float = 1e-8) -> None:
        super().__init__()
        self.eps = float(eps)

    def forward(
        self, transport: torch.Tensor, expert_tokens: torch.Tensor
    ) -> dict:
        """Transport expert values to base tokens.

        Args:
            transport: ``[M,N_base,N_expert]``.
            expert_tokens: Unnormalized adapter-space values
                ``[M,N_expert,C]``.
        """
        if transport.ndim != 3 or expert_tokens.ndim != 3:
            raise ValueError("transport and expert_tokens must be 3D")
        if transport.shape[0] != expert_tokens.shape[0] or transport.shape[2] != expert_tokens.shape[1]:
            raise ValueError(
                f"Incompatible transport {transport.shape} and values {expert_tokens.shape}"
            )
        with torch.no_grad():
            transport = transport.float()
            expert_tokens = expert_tokens.detach().float()
            received = transport.sum(dim=-1)
            teacher = torch.bmm(transport, expert_tokens) / received.unsqueeze(-1).clamp_min(
                self.eps
            )
        return {"teacher": teacher, "received": received}
