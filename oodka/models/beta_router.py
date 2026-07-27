"""Prompt-conditioned stochastic routing between P/S visual branches."""

from __future__ import annotations

from typing import Dict, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Beta, kl_divergence


class PromptBetaRouter(nn.Module):
    """Map frozen text embeddings to scale-specific continuous P/S gates.

    The distribution parameters depend only on prompt semantics. During
    training, independent reparameterized samples are drawn for each block;
    all Z slices inside a block share the same sampled prompt gate. Evaluation
    uses the deterministic distribution mean.
    """

    def __init__(
        self,
        text_dim: int,
        hidden_dim: int = 256,
        prior_p_means: Sequence[float] = (0.5, 0.6, 0.7, 0.8),
        prior_concentration: float = 10.0,
    ) -> None:
        super().__init__()
        if text_dim <= 0 or hidden_dim <= 0:
            raise ValueError("text_dim and hidden_dim must be positive")
        if len(prior_p_means) != 4:
            raise ValueError("prior_p_means must contain res2,res3,res4,res5")
        prior_mean = torch.tensor(tuple(float(v) for v in prior_p_means))
        if torch.any((prior_mean <= 0) | (prior_mean >= 1)):
            raise ValueError("Every P prior mean must be strictly between 0 and 1")
        if prior_concentration <= 0:
            raise ValueError("prior_concentration must be positive")
        prior_alpha = prior_mean * float(prior_concentration)
        prior_beta = (1.0 - prior_mean) * float(prior_concentration)
        if torch.any(prior_alpha <= 1) or torch.any(prior_beta <= 1):
            raise ValueError(
                "All prior alpha/beta values must exceed 1 for this router"
            )

        self.norm = nn.LayerNorm(text_dim)
        self.trunk = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.GELU(),
        )
        self.alpha_head = nn.Linear(hidden_dim, 4)
        self.beta_head = nn.Linear(hidden_dim, 4)
        self.register_buffer("prior_alpha", prior_alpha, persistent=True)
        self.register_buffer("prior_beta", prior_beta, persistent=True)

        # Begin exactly at the requested scale prior while retaining prompt-wise
        # learnability as soon as gradients update the zero-initialized weights.
        nn.init.zeros_(self.alpha_head.weight)
        nn.init.zeros_(self.beta_head.weight)
        with torch.no_grad():
            self.alpha_head.bias.copy_(torch.log(torch.expm1(prior_alpha - 1.0)))
            self.beta_head.bias.copy_(torch.log(torch.expm1(prior_beta - 1.0)))

    def forward(
        self,
        text_embedding: torch.Tensor,
        *,
        batch_size: int,
        sample: bool | None = None,
    ) -> Dict[str, torch.Tensor]:
        """Return per-scale Beta parameters, gates, and prior KL.

        Args:
            text_embedding: Frozen prompt embeddings shaped ``[P,D]``.
            batch_size: Number of independent Z blocks, ``B``.
            sample: Override stochastic routing. Defaults to ``self.training``.

        Returns:
            ``alpha`` and ``beta`` are ``[P,4]``; ``gate`` is ``[B,P,4]``;
            the last axis is ordered ``[res2,res3,res4,res5]`` and ``kl`` is
            a scalar.
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
