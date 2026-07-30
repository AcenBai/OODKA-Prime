"""Prompt-conditioned stochastic spatial routing between P/S visual branches."""

from __future__ import annotations

import math
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Beta, kl_divergence


def _inverse_softplus(value: torch.Tensor) -> torch.Tensor:
    """Return x such that softplus(x) == value for strictly positive value."""
    return torch.log(torch.expm1(value))


class PromptBetaRouter(nn.Module):
    """Map frozen text embeddings to one finest-resolution spatial P gate.

    Prompt semantics predict coefficients over smooth radial basis functions
    defined on normalized ``[-1, 1]`` coordinates. This produces a continuous
    prompt-specific Beta field at any requested spatial resolution without
    conditioning on image features. One sampled field is shared by all Z
    slices in a block; downstream feature levels are resized from this field.
    Evaluation uses the deterministic distribution mean.
    """

    def __init__(
        self,
        text_dim: int,
        hidden_dim: int = 256,
        prior_p_mean: float = 0.7,
        prior_concentration: float = 10.0,
        basis_grid_size: int = 8,
        basis_sigma: float = 0.0,
    ) -> None:
        super().__init__()
        if text_dim <= 0 or hidden_dim <= 0:
            raise ValueError("text_dim and hidden_dim must be positive")
        if not 0.0 < prior_p_mean < 1.0:
            raise ValueError("prior_p_mean must be strictly between 0 and 1")
        if prior_concentration <= 0.0:
            raise ValueError("prior_concentration must be positive")
        if basis_grid_size < 2:
            raise ValueError("basis_grid_size must be at least 2")

        prior_alpha = float(prior_p_mean) * float(prior_concentration)
        prior_beta = (1.0 - float(prior_p_mean)) * float(prior_concentration)
        if prior_alpha <= 1.0 or prior_beta <= 1.0:
            raise ValueError(
                "Prior alpha and beta must exceed 1 for a unimodal gate"
            )

        grid = torch.linspace(-1.0, 1.0, int(basis_grid_size))
        yy, xx = torch.meshgrid(grid, grid, indexing="ij")
        centers = torch.stack((yy, xx), dim=-1).reshape(-1, 2)
        spacing = 2.0 / float(basis_grid_size - 1)
        sigma = float(basis_sigma) if basis_sigma > 0.0 else 1.5 * spacing

        self.norm = nn.LayerNorm(text_dim)
        self.trunk = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.GELU(),
        )
        basis_count = int(centers.shape[0])
        self.alpha_coeff = nn.Linear(hidden_dim, basis_count)
        self.beta_coeff = nn.Linear(hidden_dim, basis_count)
        self.alpha_bias = nn.Parameter(
            _inverse_softplus(torch.tensor(prior_alpha - 1.0))
        )
        self.beta_bias = nn.Parameter(
            _inverse_softplus(torch.tensor(prior_beta - 1.0))
        )

        self.register_buffer("basis_centers", centers, persistent=True)
        self.register_buffer(
            "basis_sigma", torch.tensor(sigma), persistent=True
        )
        self.register_buffer(
            "prior_alpha", torch.tensor(prior_alpha), persistent=True
        )
        self.register_buffer(
            "prior_beta", torch.tensor(prior_beta), persistent=True
        )

        # Begin at one homogeneous Beta(7, 3)-style prior. Prompt-specific
        # spatial structure appears as soon as the coefficient heads update.
        nn.init.zeros_(self.alpha_coeff.weight)
        nn.init.zeros_(self.alpha_coeff.bias)
        nn.init.zeros_(self.beta_coeff.weight)
        nn.init.zeros_(self.beta_coeff.bias)

    def _spatial_basis(
        self,
        spatial_size: Tuple[int, int],
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        height, width = (int(value) for value in spatial_size)
        if height <= 0 or width <= 0:
            raise ValueError(
                f"spatial_size must be positive, got {spatial_size}"
            )
        y = torch.linspace(-1.0, 1.0, height, device=device, dtype=dtype)
        x = torch.linspace(-1.0, 1.0, width, device=device, dtype=dtype)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        coordinates = torch.stack((yy, xx), dim=-1).reshape(-1, 2)
        centers = self.basis_centers.to(device=device, dtype=dtype)
        sigma = self.basis_sigma.to(device=device, dtype=dtype)
        squared_distance = (
            coordinates[:, None, :] - centers[None, :, :]
        ).square().sum(dim=-1)
        return torch.exp(-0.5 * squared_distance / sigma.square())

    def forward(
        self,
        text_embedding: torch.Tensor,
        *,
        spatial_size: Tuple[int, int],
        batch_size: int,
        sample: bool | None = None,
    ) -> Dict[str, torch.Tensor]:
        """Return prompt-specific spatial Beta fields and prior KL.

        ``alpha``, ``beta`` and ``mean`` are ``[P,H,W]``; ``gate`` is
        ``[B,P,H,W]``. The same finest field is later resized to every
        Predictor feature level.
        """
        if text_embedding.ndim != 2:
            raise ValueError(
                f"text_embedding must be [P,D], got {text_embedding.shape}"
            )
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")

        hidden = self.trunk(self.norm(text_embedding.float()))
        basis = self._spatial_basis(
            spatial_size,
            device=hidden.device,
            dtype=hidden.dtype,
        )
        scale = math.sqrt(float(basis.shape[-1]))
        alpha_raw = (
            torch.matmul(self.alpha_coeff(hidden), basis.transpose(0, 1))
            / scale
            + self.alpha_bias
        )
        beta_raw = (
            torch.matmul(self.beta_coeff(hidden), basis.transpose(0, 1))
            / scale
            + self.beta_bias
        )
        prompt_count = int(text_embedding.shape[0])
        height, width = (int(value) for value in spatial_size)
        alpha = (
            1.0 + F.softplus(alpha_raw)
        ).reshape(prompt_count, height, width)
        beta = (
            1.0 + F.softplus(beta_raw)
        ).reshape(prompt_count, height, width)
        distribution = Beta(alpha, beta)

        if sample is None:
            sample = self.training
        if sample:
            gate = distribution.rsample((batch_size,))
        else:
            gate = distribution.mean.unsqueeze(0).expand(
                batch_size, -1, -1, -1
            )

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
