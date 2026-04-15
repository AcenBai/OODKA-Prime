"""Class-query pooling and gating modules for per-class tau generation."""

from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn as nn


class ClassQueryPooler(nn.Module):
    """Cross-attention from learnable class queries to encoder features."""

    def __init__(self, P: int, C_e: int, d_q: int = 256, n_heads: int = 8):
        super().__init__()
        self.P = P
        self.d_q = d_q
        self.Q = nn.Parameter(torch.randn(P, d_q) * 0.02)
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=d_q, num_heads=n_heads,
            kdim=C_e, vdim=C_e, batch_first=True,
        )

    def forward(self, F_enc: torch.Tensor):
        """
        Args:
            F_enc: [B, C_e, D, H, W]
        Returns:
            mu: [B, P, d_q], attn_weights: [B, P, S]
        """
        B = F_enc.shape[0]
        X = F_enc.flatten(2).transpose(1, 2)  # [B, S, C_e]
        Q = self.Q.unsqueeze(0).expand(B, self.P, self.d_q)
        mu, attn = self.multihead_attn(query=Q, key=X, value=X,
                                        need_weights=True, average_attn_weights=True)
        return mu, attn


class GateNet(nn.Module):
    """Generate per-class channel-wise tau values for feature mixing."""

    def __init__(self, d_q: int = 256, out_ch_mask: int = 512,
                 out_ch_ms: List[int] = None):
        super().__init__()
        if out_ch_ms is None:
            out_ch_ms = []
        self.mlp_mask = nn.Sequential(nn.Linear(d_q, out_ch_mask), nn.Sigmoid())
        self.mlp_ms = nn.ModuleList(
            [nn.Sequential(nn.Linear(d_q, c), nn.Sigmoid()) for c in out_ch_ms]
        )

    def forward(self, mu: torch.Tensor) -> Dict[str, object]:
        """
        Args:
            mu: [B, P, d_q]
        Returns:
            {"mask": [B, P, C_mask], "ms": list of [B, P, C_i]}
        """
        return {
            "mask": self.mlp_mask(mu),
            "ms": [mlp(mu) for mlp in self.mlp_ms],
        }
