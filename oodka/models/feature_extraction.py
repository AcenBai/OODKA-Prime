"""Feature extraction from frozen nnUNet and BiomedParse backbones."""

from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn as nn


def volume_to_rgb_slices(imgs_3d: torch.Tensor) -> torch.Tensor:
    """
    Convert 3D volume to pseudo-RGB slices for BiomedParse backbone.

    Supports:
    - [Z, H, W] or [1, Z, H, W] -> adjacent-slice pseudo-RGB -> [Z, 3, H, W]
    - [3, Z, H, W] -> direct multimodal RGB -> [Z, 3, H, W]
    """
    if imgs_3d.ndim == 4:
        C = imgs_3d.shape[0]
        if C == 3:
            return imgs_3d.float().permute(1, 0, 2, 3).contiguous()
        elif C == 1:
            imgs_3d = imgs_3d.squeeze(0)
        else:
            raise ValueError(f"Unsupported 4D shape: {imgs_3d.shape}")

    assert imgs_3d.ndim == 3, f"Expected [Z,H,W], got {imgs_3d.shape}"
    Z, H, W = imgs_3d.shape
    imgs_3d = imgs_3d.float()
    rgb = []
    for z in range(Z):
        triplet = torch.stack([
            imgs_3d[max(z - 1, 0)],
            imgs_3d[z],
            imgs_3d[min(z + 1, Z - 1)],
        ], dim=0)
        rgb.append(triplet)
    return torch.stack(rgb, dim=0)


def extract_nnunet_features(
    model: nn.Module, patches: torch.Tensor, device: torch.device,
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
    """
    Extract nnUNet encoder features at stages 2-5 and the deepest stage.

    Supports 2D nnUNet (auto-detects Conv2d) with 3D input via slice flattening.

    Returns:
        enc_feats: dict with keys "enc2"..."enc5"
        F_enc: deepest encoder features
    """
    model.eval()
    feat_dict: Dict[str, torch.Tensor] = {}

    def make_hook(name):
        def _hook(_m, _i, o):
            feat_dict[name] = o.detach()
        return _hook

    handles = [model.encoder.stages[-1].register_forward_hook(make_hook("enc_deepest"))]
    for si in [2, 3, 4, 5]:
        handles.append(model.encoder.stages[si].register_forward_hook(make_hook(f"enc{si}")))

    patches = patches.to(device)
    is_2d = any(isinstance(m, nn.Conv2d) for m in model.modules()) and \
            not any(isinstance(m, nn.Conv3d) for m in model.modules())
    reshape_back = None

    with torch.no_grad():
        if is_2d and patches.ndim == 5:
            B, C, D, H, W = patches.shape
            patches_in = patches.permute(0, 2, 1, 3, 4).reshape(B * D, C, H, W).contiguous()
            reshape_back = (B, D)
            _ = model(patches_in)
        else:
            _ = model(patches)

    for h in handles:
        h.remove()

    F_enc = feat_dict["enc_deepest"]
    result = {f"enc{i}": feat_dict[f"enc{i}"] for i in [2, 3, 4, 5]}

    if reshape_back is not None:
        B, D = reshape_back
        def _unflatten(x4):
            return x4.reshape(B, D, x4.shape[1], x4.shape[2], x4.shape[3]).permute(0, 2, 1, 3, 4).contiguous()
        F_enc = _unflatten(F_enc)
        result = {k: _unflatten(v) for k, v in result.items()}

    return result, F_enc


def extract_biomedparse_backbone_embeds_and_res_levels_3d(
    model: nn.Module,
    patches_3d: torch.Tensor,
    device: torch.device,
    res_names: Tuple[str, ...] = ("res3", "res4"),
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """
    Extract BiomedParse backbone features as 2D-batched dict + selected res levels as 3D blocks.

    Returns:
        img_embeds_base: {name: [B*Z, C, Hf, Wf]}
        res_3d: {name: [B, C, Z, Hf, Wf]}
    """
    model.eval()

    if patches_3d.ndim == 5 and patches_3d.shape[1] == 3:
        B, _, Z, H, W = patches_3d.shape
        is_multi = True
    elif patches_3d.ndim == 5 and patches_3d.shape[1] == 1:
        patches_3d = patches_3d.squeeze(1)
        B, Z, H, W = patches_3d.shape
        is_multi = False
    elif patches_3d.ndim == 4:
        B, Z, H, W = patches_3d.shape
        is_multi = False
    else:
        raise ValueError(f"Unsupported shape: {patches_3d.shape}")

    pixel_mean = model.pixel_mean.view(1, 3, 1, 1).to(device)
    pixel_std = model.pixel_std.view(1, 3, 1, 1).to(device)

    embeds_accum = None
    res_accum: Dict[str, List[torch.Tensor]] = {k: [] for k in res_names}

    for b in range(B):
        patch = patches_3d[b]
        rgb = volume_to_rgb_slices(patch).to(device).float()
        rgb = (rgb - pixel_mean) / pixel_std

        with torch.no_grad():
            img_embeds = model.backbone(rgb)

        if embeds_accum is None:
            embeds_accum = {k: [] for k in img_embeds.keys()}
        for k, v in img_embeds.items():
            embeds_accum[k].append(v.detach())
        for rn in res_names:
            res_accum[rn].append(
                img_embeds[rn].detach().permute(1, 0, 2, 3).unsqueeze(0).contiguous()
            )

    img_embeds_base = {k: torch.cat(vs, dim=0) for k, vs in embeds_accum.items()}
    res_3d = {k: torch.cat(vs, dim=0) for k, vs in res_accum.items()}
    return img_embeds_base, res_3d


def extract_biomedparse_pixeldecoder_outputs_3d(
    model: nn.Module, patches_3d: torch.Tensor, device: torch.device,
) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    """
    Run BiomedParse backbone + pixel_decoder, return 3D mask_features and multi-scale features.
    """
    model.eval()
    B, Z, H, W = patches_3d.shape
    pixel_mean = model.pixel_mean.view(1, 3, 1, 1).to(device)
    pixel_std = model.pixel_std.view(1, 3, 1, 1).to(device)
    all_mask, all_ms = [], None

    for b in range(B):
        rgb = volume_to_rgb_slices(patches_3d[b].unsqueeze(0)).to(device).float()
        rgb = (rgb - pixel_mean) / pixel_std
        with torch.no_grad():
            img_embeds = model.backbone(rgb)
            out = model.sem_seg_head.pixel_decoder.forward_features(img_embeds)

        from .biomedparse_helpers import parse_pixel_decoder_out
        mask_feat, ms = parse_pixel_decoder_out(out)
        all_mask.append(mask_feat.permute(1, 0, 2, 3).unsqueeze(0).contiguous())
        ms_3d = [m.permute(1, 0, 2, 3).unsqueeze(0).contiguous() for m in ms]
        if all_ms is None:
            all_ms = [[] for _ in ms_3d]
        for i, m in enumerate(ms_3d):
            all_ms[i].append(m)

    return torch.cat(all_mask, dim=0), [torch.cat(x, dim=0) for x in all_ms]
