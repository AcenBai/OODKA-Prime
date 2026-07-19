"""Feature extraction from frozen nnUNet and BiomedParse backbones."""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn


def extract_nnunet_features(
    model: nn.Module,
    blocks: torch.Tensor,
    device: torch.device,
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
    """Extract nnUNet stages 2-5 and restore flattened 2D slices to 5D.

    ``blocks`` is ``[B,C,Z,H,W]``. A 2D nnUNet sees ``B*Z`` slices and each
    captured feature level is returned as ``[B,Cf,Z,Hf,Wf]``.
    """
    model.eval()
    feature_dict: Dict[str, torch.Tensor] = {}

    def make_hook(name):
        def hook(_module, _inputs, output):
            feature_dict[name] = output.detach()

        return hook

    handles = [
        model.encoder.stages[-1].register_forward_hook(make_hook("enc_deepest"))
    ]
    for stage_index in [2, 3, 4, 5]:
        handles.append(
            model.encoder.stages[stage_index].register_forward_hook(
                make_hook(f"enc{stage_index}")
            )
        )

    blocks = blocks.to(device)
    is_2d = any(isinstance(module, nn.Conv2d) for module in model.modules()) and not any(
        isinstance(module, nn.Conv3d) for module in model.modules()
    )
    reshape_back = None
    with torch.no_grad():
        if is_2d and blocks.ndim == 5:
            B, C, Z, H, W = blocks.shape
            slices = (
                blocks.permute(0, 2, 1, 3, 4)
                .reshape(B * Z, C, H, W)
                .contiguous()
            )
            reshape_back = (B, Z)
            model(slices)
        else:
            model(blocks)

    for handle in handles:
        handle.remove()

    deepest = feature_dict["enc_deepest"]
    result = {
        f"enc{stage_index}": feature_dict[f"enc{stage_index}"]
        for stage_index in [2, 3, 4, 5]
    }
    if reshape_back is not None:
        B, Z = reshape_back

        def unflatten(feature):
            return (
                feature.reshape(
                    B, Z, feature.shape[1], feature.shape[2], feature.shape[3]
                )
                .permute(0, 2, 1, 3, 4)
                .contiguous()
            )

        deepest = unflatten(deepest)
        result = {name: unflatten(feature) for name, feature in result.items()}
    return result, deepest


def extract_biomedparse_backbone_features_2p5d(
    model: nn.Module,
    images: torch.Tensor,
    device: torch.device,
    res_names: Tuple[str, ...] = ("res2", "res3", "res4", "res5"),
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """Extract BiomedParse features from ``[B,Z,3,H,W]`` inputs.

    The frozen 2D backbone sees ``B*Z`` pseudo-RGB images. Selected levels are
    restored as ``[B,C,Z,Hf,Wf]`` for the Conv3d fusion adapters.
    """
    if images.ndim != 5 or images.shape[2] != 3:
        raise ValueError(f"Expected BiomedParse input [B,Z,3,H,W], got {images.shape}")
    model.eval()
    B, Z, C, H, W = images.shape
    flat = images.reshape(B * Z, C, H, W).to(device=device, dtype=torch.float32)
    pixel_mean = model.pixel_mean.view(1, 3, 1, 1).to(device)
    pixel_std = model.pixel_std.view(1, 3, 1, 1).to(device)
    flat = (flat - pixel_mean) / pixel_std
    with torch.no_grad():
        embeds = model.backbone(flat)

    block_features = {}
    for name in res_names:
        feature = embeds[name]
        block_features[name] = (
            feature.reshape(
                B, Z, feature.shape[1], feature.shape[2], feature.shape[3]
            )
            .permute(0, 2, 1, 3, 4)
            .contiguous()
        )
    return embeds, block_features
