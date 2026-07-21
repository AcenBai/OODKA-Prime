"""Prompt-conditioned stochastic routing between P/S visual branches."""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Beta, kl_divergence


class PromptBetaRouter(nn.Module):
    """Map frozen text embeddings to a continuous scalar P/S gate.

    The distribution parameters depend only on prompt semantics. During
    training, independent reparameterized samples are drawn for each block;
    all Z slices inside a block share the same sampled prompt gate. Evaluation
    uses the deterministic distribution mean.
    """

    def __init__(
        self,
        text_dim: int,
        hidden_dim: int = 256,
        prior_alpha: float = 2.0,
        prior_beta: float = 2.0,
    ) -> None:
        super().__init__()
        if text_dim <= 0 or hidden_dim <= 0:
            raise ValueError("text_dim and hidden_dim must be positive")
        if prior_alpha <= 0 or prior_beta <= 0:
            raise ValueError("Beta prior parameters must be positive")

        self.norm = nn.LayerNorm(text_dim)
        self.trunk = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.GELU(),
        )
        self.alpha_head = nn.Linear(hidden_dim, 1)
        self.beta_head = nn.Linear(hidden_dim, 1)
        self.register_buffer(
            "prior_alpha", torch.tensor(float(prior_alpha)), persistent=True
        )
        self.register_buffer(
            "prior_beta", torch.tensor(float(prior_beta)), persistent=True
        )

    def forward(
        self,
        text_embedding: torch.Tensor,
        *,
        batch_size: int,
        sample: bool | None = None,
    ) -> Dict[str, torch.Tensor]:
        """Return Beta parameters, scalar gates, and prior KL.

        Args:
            text_embedding: Frozen prompt embeddings shaped ``[P,D]``.
            batch_size: Number of independent Z blocks, ``B``.
            sample: Override stochastic routing. Defaults to ``self.training``.

        Returns:
            ``alpha`` and ``beta`` are ``[P,1]``; ``gate`` is ``[B,P,1]``;
            ``kl`` is a scalar.
        """
        if text_embedding.ndim != 2:
            raise ValueError(
                f"text_embedding must be [P,D], got {text_embedding.shape}"
            )
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")

        hidden = self.trunk(self.norm(text_embedding.float()))
        alpha = 1.0 + F.softplus(self.alpha_head(hidden))
        beta = 1.0 + F.softplus(self.beta_head(hidden))
        distribution = Beta(alpha, beta)

        if sample is None:
            sample = self.training
        if sample:
            gate = distribution.rsample((batch_size,))
        else:
            gate = distribution.mean.unsqueeze(0).expand(batch_size, -1, -1)

        prior = Beta(
            self.prior_alpha.to(alpha).expand_as(alpha),
            self.prior_beta.to(beta).expand_as(beta),
        )
        route_kl = kl_divergence(distribution, prior).mean()
        return {
            "alpha": alpha,
            "beta": beta,
            "gate": gate,
            "mean": distribution.mean,
            "concentration": alpha + beta,
            "kl": route_kl,
        }
