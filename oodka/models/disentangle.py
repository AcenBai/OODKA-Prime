"""Disentanglement modules: split features into private/shared components."""

from __future__ import annotations

import torch
import torch.nn as nn


class TwoBranchDisentangle(nn.Module):
    """Split a feature tensor into private (p) and shared (s) via 1x1 conv."""

    def __init__(self, C: int = 512):
        super().__init__()
        self.proj_p = nn.Conv3d(C, C, 1, bias=False)
        self.proj_s = nn.Conv3d(C, C, 1, bias=False)

    def forward(self, Z: torch.Tensor):
        return self.proj_p(Z), self.proj_s(Z)


class DualBranchAutoEncoder(nn.Module):
    """
    Channel-align + disentangle encoder with reconstruction decoder.

    Maps from c_in channels (nnUNet encoder) to c_out channels (BiomedParse res),
    splitting into private and shared branches, each with a reconstruction path.
    """

    def __init__(self, c_in: int = 32, c_mid: int = 128, c_out: int = 512):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv3d(c_in, c_mid, 1, bias=False),
            nn.InstanceNorm3d(c_mid, affine=True),
            nn.GELU(),
        )
        self.enc_p = nn.Sequential(
            nn.Conv3d(c_mid, c_out, 1, bias=False),
            nn.InstanceNorm3d(c_out, affine=True),
        )
        self.enc_s = nn.Sequential(
            nn.Conv3d(c_mid, c_out, 1, bias=False),
            nn.InstanceNorm3d(c_out, affine=True),
        )
        self.dec_p = nn.Sequential(
            nn.Conv3d(c_out, c_mid, 1, bias=False),
            nn.InstanceNorm3d(c_mid, affine=True),
            nn.GELU(),
            nn.Conv3d(c_mid, c_in, 1, bias=False),
        )
        self.dec_s = nn.Sequential(
            nn.Conv3d(c_out, c_mid, 1, bias=False),
            nn.InstanceNorm3d(c_mid, affine=True),
            nn.GELU(),
            nn.Conv3d(c_mid, c_in, 1, bias=False),
        )

    def forward(self, Z: torch.Tensor):
        """
        Args:
            Z: [B, c_in, D, H, W]
        Returns:
            (Zp, Zs, Zp_rec, Zs_rec) each same spatial dims as input.
        """
        shared = self.enc(Z)
        Zp = self.enc_p(shared)
        Zs = self.enc_s(shared)
        Zp_rec = self.dec_p(Zp)
        Zs_rec = self.dec_s(Zs)
        return Zp, Zs, Zp_rec, Zs_rec
