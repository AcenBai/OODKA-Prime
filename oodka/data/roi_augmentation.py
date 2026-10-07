"""Synchronized ROI-training augmentation shared by all modalities."""

from __future__ import annotations

import torch
import torch.nn.functional as F

def augment_lge_batch(
    batch_data: dict,
    *,
    rotation_degrees: float,
    scale_min: float,
    scale_max: float,
    translation_fraction: float,
    horizontal_flip_probability: float,
    vertical_flip_probability: float,
    intensity_probability: float,
) -> dict:
    """Apply synchronized affine and mild modality-aware intensity jitter."""
    output = dict(batch_data)
    nn_image = batch_data["nnunet_image"]
    bp_image = batch_data["biomedparse_image"]
    gt = batch_data["gt"]
    if nn_image.ndim != 5 or bp_image.ndim != 5 or gt.ndim != 4:
        raise ValueError("Unexpected batch tensor rank for augmentation")
    batch_size, block_z = nn_image.shape[:2]
    angles = torch.empty(batch_size).uniform_(-rotation_degrees, rotation_degrees)
    angles = angles * torch.pi / 180.0
    scales = torch.empty(batch_size).uniform_(scale_min, scale_max)
    flip_x = torch.where(
        torch.rand(batch_size) < horizontal_flip_probability, -1.0, 1.0
    )
    flip_y = torch.where(
        torch.rand(batch_size) < vertical_flip_probability, -1.0, 1.0
    )
    tx = torch.empty(batch_size).uniform_(-translation_fraction, translation_fraction) * 2.0
    ty = torch.empty(batch_size).uniform_(-translation_fraction, translation_fraction) * 2.0
    theta = torch.zeros((batch_size, 2, 3), dtype=nn_image.dtype)
    theta[:, 0, 0] = torch.cos(angles) * flip_x / scales
    theta[:, 0, 1] = -torch.sin(angles) * flip_y / scales
    theta[:, 1, 0] = torch.sin(angles) * flip_x / scales
    theta[:, 1, 1] = torch.cos(angles) * flip_y / scales
    theta[:, 0, 2] = tx
    theta[:, 1, 2] = ty

    # Preserve 3-D block coherence by sharing one spatial transform across all
    # Z slices of a block. Each slice's pseudo-RGB channels remain synchronized.
    theta_bz = theta.repeat_interleave(block_z, dim=0)

    def spatial(images: torch.Tensor, mode: str) -> torch.Tensor:
        flat = images.flatten(0, 1)
        grid = F.affine_grid(theta_bz, flat.shape, align_corners=False)
        transformed = F.grid_sample(
            flat, grid, mode=mode, padding_mode="zeros", align_corners=False
        )
        return transformed.unflatten(0, (batch_size, block_z))

    nn_aug = spatial(nn_image, "bilinear")
    bp_aug = spatial(bp_image, "bilinear")
    gt_aug = spatial(gt[:, :, None].float(), "nearest")[:, :, 0].to(gt.dtype)

    for index in range(batch_size * block_z):
        if float(torch.rand(())) >= intensity_probability:
            continue
        contrast = float(torch.empty(()).uniform_(0.8, 1.2))
        shift = float(torch.empty(()).uniform_(-0.1, 0.1))
        noise = float(torch.empty(()).uniform_(0.0, 0.04))
        batch_index, z_index = divmod(index, block_z)
        for images, clamp in ((nn_aug, False), (bp_aug, True)):
            value = images[batch_index, z_index]
            mean = value.mean()
            std = value.std().clamp_min(1e-6)
            value = mean + contrast * (value - mean) + shift * std
            value = value + torch.randn_like(value) * (noise * std)
            if clamp:
                gamma = float(torch.empty(()).uniform_(0.75, 1.35))
                value = (value.clamp(0.0, 255.0) / 255.0).pow(gamma) * 255.0
            images[batch_index, z_index] = value
    output["nnunet_image"] = nn_aug
    output["biomedparse_image"] = bp_aug
    output["gt"] = gt_aug
    return output
