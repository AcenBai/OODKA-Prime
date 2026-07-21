"""Numerically stable balanced and KL-relaxed unbalanced Sinkhorn solvers."""

from __future__ import annotations

import torch
import torch.nn as nn


def _validate(a: torch.Tensor, b: torch.Tensor, cost: torch.Tensor) -> None:
    if a.ndim != 2 or b.ndim != 2 or cost.ndim != 3:
        raise ValueError("Expected a[M,N], b[M,K], cost[M,N,K]")
    if cost.shape != (a.shape[0], a.shape[1], b.shape[1]):
        raise ValueError(
            f"cost shape {cost.shape} is incompatible with {a.shape} and {b.shape}"
        )
    if not (torch.isfinite(a).all() and torch.isfinite(b).all() and torch.isfinite(cost).all()):
        raise FloatingPointError("Sinkhorn inputs contain NaN or Inf")
    if (a < 0).any() or (b < 0).any() or (cost < 0).any():
        raise ValueError("Sinkhorn masses and cost must be non-negative")


class BalancedSinkhorn(nn.Module):
    def __init__(self, epsilon: float = 0.1, iterations: int = 50, eps: float = 1e-8):
        super().__init__()
        self.epsilon = float(epsilon)
        self.iterations = int(iterations)
        self.eps = float(eps)

    def forward(self, a: torch.Tensor, b: torch.Tensor, cost: torch.Tensor) -> dict:
        with torch.no_grad(), torch.autocast(device_type=cost.device.type, enabled=False):
            a, b, cost = a.float(), b.float(), cost.float()
            _validate(a, b, cost)
            log_a = a.clamp_min(self.eps).log()
            log_b = b.clamp_min(self.eps).log()
            log_kernel = -cost / self.epsilon
            log_u = torch.zeros_like(log_a)
            log_v = torch.zeros_like(log_b)
            for _ in range(self.iterations):
                log_u = log_a - torch.logsumexp(
                    log_kernel + log_v[:, None, :], dim=-1
                )
                log_v = log_b - torch.logsumexp(
                    log_kernel + log_u[:, :, None], dim=-2
                )
            transport = torch.exp(
                log_u[:, :, None] + log_kernel + log_v[:, None, :]
            )
            row = transport.sum(dim=-1)
            col = transport.sum(dim=-2)
            row_error = (row - a).abs().sum(dim=-1)
            col_error = (col - b).abs().sum(dim=-1)
            entropy = -(transport.clamp_min(self.eps) * transport.clamp_min(self.eps).log()).sum(
                dim=(-2, -1)
            )
            transport_cost = (transport * cost).sum(dim=(-2, -1))
        if not torch.isfinite(transport).all():
            raise FloatingPointError("Balanced transport contains NaN or Inf")
        return {
            "transport": transport,
            "row_error": row_error,
            "col_error": col_error,
            "entropy": entropy,
            "cost": transport_cost,
        }


class UnbalancedSinkhorn(nn.Module):
    def __init__(
        self,
        epsilon: float = 0.1,
        rho_base: float = 1.0,
        rho_expert: float = 0.2,
        iterations: int = 50,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        self.epsilon = float(epsilon)
        self.rho_base = float(rho_base)
        self.rho_expert = float(rho_expert)
        self.iterations = int(iterations)
        self.eps = float(eps)

    def forward(self, a: torch.Tensor, b: torch.Tensor, cost: torch.Tensor) -> dict:
        with torch.no_grad(), torch.autocast(device_type=cost.device.type, enabled=False):
            a, b, cost = a.float(), b.float(), cost.float()
            _validate(a, b, cost)
            log_a = a.clamp_min(self.eps).log()
            log_b = b.clamp_min(self.eps).log()
            log_kernel = -cost / self.epsilon
            tau_base = self.rho_base / (self.rho_base + self.epsilon)
            tau_expert = self.rho_expert / (self.rho_expert + self.epsilon)
            log_u = torch.zeros_like(log_a)
            log_v = torch.zeros_like(log_b)
            for _ in range(self.iterations):
                log_u = tau_base * (
                    log_a
                    - torch.logsumexp(log_kernel + log_v[:, None, :], dim=-1)
                )
                log_v = tau_expert * (
                    log_b
                    - torch.logsumexp(log_kernel + log_u[:, :, None], dim=-2)
                )
            transport = torch.exp(
                log_u[:, :, None] + log_kernel + log_v[:, None, :]
            )
            received = transport.sum(dim=-1)
            transported = transport.sum(dim=-2)
            rejected = (b - transported).clamp_min(0.0)
            accepted = torch.minimum(transported, b)
            entropy = -(transport.clamp_min(self.eps) * transport.clamp_min(self.eps).log()).sum(
                dim=(-2, -1)
            )
            transport_cost = (transport * cost).sum(dim=(-2, -1))
            planned_total = b.sum(dim=-1).clamp_min(self.eps)
            accept_ratio = accepted.sum(dim=-1) / planned_total
        if not torch.isfinite(transport).all():
            raise FloatingPointError("Unbalanced transport contains NaN or Inf")
        return {
            "transport": transport,
            "received": received,
            "transported": transported,
            "accepted": accepted,
            "rejected": rejected,
            "accept_ratio": accept_ratio,
            "entropy": entropy,
            "cost": transport_cost,
        }
