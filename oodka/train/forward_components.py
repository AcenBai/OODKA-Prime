"""Core forward passes for contiguous 2.5D OODKA slice blocks."""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..models.feature_extraction import (
    extract_nnunet_features,
    extract_biomedparse_backbone_features_2p5d,
)
from ..models.biomedparse_helpers import (
    expand_prompt_features_for_blocks,
    parse_pixel_decoder_out,
    gates_for_biomedparse_predictor,
    select_best_mask_from_queries,
    run_biomedparse_predictor_override,
)
from ..models.losses import (
    ortho_corr_loss,
)


def _disentangle_and_inject(
    enc_feats: Dict[str, torch.Tensor],
    res3d: Dict[str, torch.Tensor],
    img_embeds_base: Dict[str, torch.Tensor],
    ae_modules: Dict[str, nn.Module],
    dis_modules: Dict[str, nn.Module],
    device: torch.device,
) -> Dict:
    """
    Run channel-alignment + disentanglement on encoder/backbone features at res2-5.
    Returns a dict with all intermediates needed for loss computation and pixel decoder injection.
    """
    out = {}
    levels = [2, 3, 4, 5]

    for i in levels:
        Z_enc = enc_feats[f"enc{i}"]
        Zb_res = res3d[f"res{i}"].to(device)

        if tuple(Z_enc.shape[-3:]) != tuple(Zb_res.shape[-3:]):
            Z_n = F.interpolate(Z_enc, size=Zb_res.shape[-3:], mode="trilinear", align_corners=False)
        else:
            Z_n = Z_enc

        ae = ae_modules[f"ae_enc{i}_to_res{i}"]
        dis = dis_modules[f"dis_b_res{i}"]

        expert_outputs = ae(Z_n)
        if len(expert_outputs) == 4:
            Zn_p, Zn_s, Zn_p_rec, Zn_s_rec = expert_outputs
            Zn_rec = Zn_p_rec + Zn_s_rec
        elif len(expert_outputs) == 3:
            Zn_p, Zn_s, Zn_rec = expert_outputs
        else:
            raise RuntimeError(
                "Expert adapter must return (P,S,reconstruction) or "
                "(P,S,P_reconstruction,S_reconstruction)"
            )
        Zb_p, Zb_s = dis(Zb_res)

        out[f"Z_n{i}"] = Z_n
        out[f"Zn{i}_p"] = Zn_p
        out[f"Zn{i}_s"] = Zn_s
        out[f"Zn{i}_rec"] = Zn_rec
        out[f"Zb_res{i}"] = Zb_res
        out[f"Zb{i}_p"] = Zb_p
        out[f"Zb{i}_s"] = Zb_s

    return out


def _normalized_mse_5d(
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid_z: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Energy-normalized MSE excluding repeated tail slices."""
    with torch.autocast(device_type=target.device.type, enabled=False):
        prediction = prediction.float()
        target = target.float()
        B, C, D, H, W = target.shape
        if valid_z.shape != (B, D):
            raise ValueError(f"valid_z must be [B,D]={B,D}, got {valid_z.shape}")
        mask = valid_z[:, None, :, None, None].to(target)
        denom = mask.sum().clamp_min(1.0) * C * H * W
        error = ((prediction - target).square() * mask).sum() / denom
        energy = (target.square() * mask).sum() / denom
        return error / (energy + eps)


def _normalized_mse_flat_z(
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid_z: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Energy-normalized MSE for pixel-decoder tensors shaped [B*Z,C,H,W]."""
    with torch.autocast(device_type=target.device.type, enabled=False):
        prediction = prediction.float()
        target = target.float()
        B, D = valid_z.shape
        if target.ndim != 4 or target.shape[0] != B * D:
            raise ValueError(
                f"Expected pixel-decoder tensor [B*Z,C,H,W] with B*Z={B*D}, "
                f"got {target.shape}"
            )
        mask = valid_z.reshape(B * D, 1, 1, 1).to(target)
        elements_per_slice = target.shape[1] * target.shape[2] * target.shape[3]
        denom = mask.sum().clamp_min(1.0) * elements_per_slice
        error = ((prediction - target).square() * mask).sum() / denom
        energy = (target.square() * mask).sum() / denom
        return error / (energy + eps)


def _compute_reconstruction_separation_losses(
    feats: Dict,
    valid_z: torch.Tensor,
    expert_ortho_weight: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute reconstruction and P/S separation across all levels."""
    eps = 1e-8
    levels = [2, 3, 4, 5]

    ae_losses = []
    student_ortho_losses = []
    expert_ortho_losses = []

    for i in levels:
        Z_n = feats[f"Z_n{i}"]
        Zn_rec = feats[f"Zn{i}_rec"]
        Zb_res = feats[f"Zb_res{i}"]
        Zn_p = feats[f"Zn{i}_p"]
        Zn_s = feats[f"Zn{i}_s"]
        Zb_p = feats[f"Zb{i}_p"]
        Zb_s = feats[f"Zb{i}_s"]

        ae_n = _normalized_mse_5d(Zn_rec, Z_n, valid_z, eps)
        ae_b = _normalized_mse_5d(Zb_p + Zb_s, Zb_res, valid_z, eps)
        ae_losses.extend([ae_n, ae_b])

        with torch.autocast(device_type=Zb_p.device.type, enabled=False):
            student_ortho_losses.append(
                ortho_corr_loss(Zb_p.float(), Zb_s.float(), valid_z=valid_z)
            )
            expert_ortho_losses.append(
                ortho_corr_loss(Zn_p.float(), Zn_s.float(), valid_z=valid_z)
            )

    student_ortho = sum(student_ortho_losses)
    expert_ortho = sum(expert_ortho_losses)
    combined = student_ortho + float(expert_ortho_weight) * expert_ortho
    return sum(ae_losses), combined, student_ortho, expert_ortho


def _run_pixel_decoder(
    model_biomedparse: nn.Module,
    img_embeds_base: Dict[str, torch.Tensor],
    feats: Dict,
    branch: str,
    B: int, Dm: int,
) -> Tuple[torch.Tensor, list]:
    """Run pixel decoder with injected res2-5 features for a given branch (p or s)."""
    levels = [2, 3, 4, 5]
    N = B * Dm

    injected = {k: v for k, v in img_embeds_base.items()}
    for i in levels:
        feat_5d = feats[f"Zb{i}_{branch}"]
        Ci = feat_5d.shape[1]
        Hi, Wi = feat_5d.shape[3], feat_5d.shape[4]
        injected[f"res{i}"] = feat_5d.permute(0, 2, 1, 3, 4).reshape(N, Ci, Hi, Wi).contiguous()

    pd_out = model_biomedparse.sem_seg_head.pixel_decoder.forward_features(injected)
    return parse_pixel_decoder_out(pd_out)


def _fuse_all_prompt_features(
    p_feature: torch.Tensor,
    s_feature: torch.Tensor,
    gate: torch.Tensor,
    *,
    B: int,
    Z: int,
) -> torch.Tensor:
    """Spatially fuse P/S and flatten prompt pairs in ``[B,Z,P]`` order."""
    if p_feature.shape != s_feature.shape or p_feature.ndim != 4:
        raise ValueError(
            f"P/S features must have equal [B*Z,C,H,W] shapes, "
            f"got {p_feature.shape} and {s_feature.shape}"
        )
    N, C, H, W = p_feature.shape
    if N != B * Z:
        raise ValueError(f"Visual batch={N} != B*Z={B*Z}")
    if (
        gate.ndim != 4
        or gate.shape[0] != B
        or gate.shape[-2:] != (H, W)
    ):
        raise ValueError(
            f"spatial gate must be [B,P,{H},{W}], got {gate.shape}"
        )
    P = gate.shape[1]
    p_bz = p_feature.reshape(B, Z, C, H, W)[:, :, None]
    s_bz = s_feature.reshape(B, Z, C, H, W)[:, :, None]
    gate_bzp = gate[:, None, :, None]
    fused = gate_bzp * p_bz + (1.0 - gate_bzp) * s_bz
    return fused.reshape(B * Z * P, C, H, W).contiguous()


def _predict_all_prompt_logits(
    *,
    sem_seg_head: nn.Module,
    mask_features_p: torch.Tensor,
    mask_features_s: torch.Tensor,
    ms_p: List[torch.Tensor],
    ms_s: List[torch.Tensor],
    gate: torch.Tensor,
    prompt_features: dict,
    B: int,
    Z: int,
    P: int,
    output_shape: Tuple[int, int, int],
) -> torch.Tensor:
    """Run one predictor call for all aligned visual-prompt pairs."""
    mask_gate, multi_scale_gates = gates_for_biomedparse_predictor(
        gate,
        B=B,
        P=P,
        mask_size=mask_features_p.shape[-2:],
        multi_scale_sizes=[feature.shape[-2:] for feature in ms_p],
    )
    if len(ms_p) != len(ms_s):
        raise ValueError(
            "P/S multi-scale feature counts must match, got "
            f"{len(ms_p)} and {len(ms_s)}"
        )

    if len(ms_p) != 3:
        raise ValueError(
            "BiomedParse predictor must expose three multi-scale features "
            f"(res5,res4,res3), got {len(ms_p)}"
        )

    # One finest prompt gate is shared by mask features and all coarse-to-fine
    # Predictor inputs through area downsampling.
    fused_mask = _fuse_all_prompt_features(
        mask_features_p, mask_features_s, mask_gate, B=B, Z=Z
    )
    fused_multi_scale = [
        _fuse_all_prompt_features(mp, ms, scale_gate, B=B, Z=Z)
        for mp, ms, scale_gate in zip(ms_p, ms_s, multi_scale_gates)
    ]
    expanded_prompts = expand_prompt_features_for_blocks(
        prompt_features,
        B=B,
        Z=Z,
        P=P,
    )
    pred_out = run_biomedparse_predictor_override(
        sem_seg_head,
        fused_multi_scale,
        fused_mask,
        expanded_prompts,
    )
    mask_logits = select_best_mask_from_queries(
        pred_out["pred_gmasks"], pred_out.get("object_existence")
    )
    expected_pairs = B * Z * P
    if mask_logits.shape[0] != expected_pairs:
        raise RuntimeError(
            f"Predictor output batch={mask_logits.shape[0]} != B*Z*P={expected_pairs}"
        )

    height, width = mask_logits.shape[-2:]
    logits_bpzhw = (
        mask_logits.reshape(B, Z, P, height, width)
        .permute(0, 2, 1, 3, 4)
        .contiguous()
    )
    out_z, out_h, out_w = (int(value) for value in output_shape)
    resized = F.interpolate(
        logits_bpzhw.reshape(B * P, 1, Z, height, width),
        size=(out_z, out_h, out_w),
        mode="trilinear",
        align_corners=False,
    )
    return resized.reshape(B, P, out_z, out_h, out_w)


def _compute_segmentation_loss_and_metrics(
    logits: torch.Tensor,
    gt: torch.Tensor,
    valid_z: torch.Tensor,
    class_ids: torch.Tensor,
    prompt_reduction: str = "mean",
    prompt_valid: torch.Tensor | None = None,
) -> Tuple[torch.Tensor, float, Dict[int, float | None]]:
    """Vectorized BCE/Dice objective over all ``B*P`` class-volume pairs."""
    B, P, Z, H, W = logits.shape
    if gt.shape != (B, Z, H, W):
        raise ValueError(f"GT must be [B,Z,H,W]={B,Z,H,W}, got {gt.shape}")
    if valid_z.shape != (B, Z):
        raise ValueError(f"valid_z must be [B,Z]={B,Z}, got {valid_z.shape}")
    if class_ids.shape != (P,):
        raise ValueError(f"class_ids must be [P]={P}, got {class_ids.shape}")
    if prompt_valid is None:
        prompt_valid = torch.ones((B, P), dtype=torch.bool, device=logits.device)
    else:
        prompt_valid = prompt_valid.to(device=logits.device, dtype=torch.bool)
        if prompt_valid.shape != (B, P):
            raise ValueError(f"prompt_valid must be [B,P]={B,P}, got {prompt_valid.shape}")

    valid = (gt != -1) & valid_z[:, :, None, None]
    valid_bp = valid[:, None]
    valid_float = valid_bp.float()
    gt_all = gt[:, None] == class_ids[None, :, None, None, None]
    gt_float = gt_all.float()

    valid_voxels = valid_bp.flatten(2).sum(dim=2).float()
    threshold = valid_voxels * 0.0005
    gt_foreground = (gt_all & valid_bp).flatten(2).sum(dim=2).float()
    gt_empty = gt_foreground < threshold

    bce = F.binary_cross_entropy_with_logits(logits, gt_float, reduction="none")
    denominator = valid_float.sum(dim=(2, 3, 4)).clamp_min(1.0)
    bce_per_pair = (bce * valid_float).sum(dim=(2, 3, 4)) / denominator

    probabilities = torch.sigmoid(logits) * valid_float
    targets = gt_float * valid_float
    intersection = (probabilities * targets).sum(dim=(2, 3, 4))
    union = probabilities.sum(dim=(2, 3, 4)) + targets.sum(dim=(2, 3, 4))
    dice_soft = (2.0 * intersection + 1e-6) / (union + 1e-6)
    dice_loss = 1.0 - dice_soft

    loss_per_pair = torch.where(
        gt_empty,
        bce_per_pair,
        bce_per_pair + dice_loss,
    )
    pair_weights = torch.where(
        gt_empty,
        torch.full_like(loss_per_pair, 0.5),
        torch.ones_like(loss_per_pair),
    ) * prompt_valid.float()
    if prompt_reduction == "mean":
        loss_seg = (
            (pair_weights * loss_per_pair).sum()
            / pair_weights.sum().clamp_min(1e-6)
        )
    elif prompt_reduction in {"sum", "prompt_mean"}:
        prompt_weight = pair_weights.sum(dim=0)
        loss_per_prompt = (
            (pair_weights * loss_per_pair).sum(dim=0)
            / prompt_weight.clamp_min(1e-6)
        )
        if prompt_reduction == "sum":
            loss_seg = loss_per_prompt.sum()
        else:
            prompt_is_valid = prompt_weight > 0
            loss_seg = (
                (loss_per_prompt * prompt_is_valid).sum()
                / prompt_is_valid.sum().clamp_min(1)
            )
    else:
        raise ValueError(f"Unknown prompt_reduction={prompt_reduction!r}")

    dice_values = []
    dice_per_class: Dict[int, float | None] = {}
    with torch.no_grad():
        predicted = torch.sigmoid(logits) > 0.5
        hard_intersection = (predicted & gt_all & valid_bp).flatten(2).sum(2).float()
        hard_union = (
            (predicted & valid_bp).flatten(2).sum(2).float() + gt_foreground
        )
        hard_dice = (2.0 * hard_intersection + 1e-6) / (hard_union + 1e-6)
        for prompt_index in range(P):
            nonempty = (~gt_empty[:, prompt_index]) & prompt_valid[:, prompt_index]
            if nonempty.any():
                value = float(hard_dice[nonempty, prompt_index].mean().item())
                dice_per_class[prompt_index] = value
                dice_values.append(value)
            else:
                dice_per_class[prompt_index] = None
    dice_mean = float(np.mean(dice_values)) if dice_values else 0.0
    return loss_seg, dice_mean, dice_per_class


def _compute_detached_pixel_error_maps(
    base_logits: torch.Tensor,
    expert_logits: torch.Tensor,
    gt: torch.Tensor,
    valid_z: torch.Tensor,
    class_ids: torch.Tensor,
    expert_class_groups: Sequence[Sequence[int]] | None = None,
    prompt_valid: torch.Tensor | None = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return comparable prompt-wise BCE maps for student and expert.

    Base logits are ``[B,P,Z,H,W]``. nnUNet logits are
    ``[B,C_nn,Z,H_nn,W_nn]`` and are explicitly indexed by semantic class id.
    Both returned maps are detached ``[B,Z,H,W]`` tensors.
    """
    B, P, Z, H, W = base_logits.shape
    if gt.shape != (B, Z, H, W) or valid_z.shape != (B, Z):
        raise ValueError("GT/valid_z do not match base logits")
    if class_ids.shape != (P,):
        raise ValueError(f"class_ids must be [P]={P}, got {class_ids.shape}")
    if expert_logits.ndim != 5 or expert_logits.shape[0] != B:
        raise ValueError(
            f"expert_logits must be [B,C,Z,H,W], got {expert_logits.shape}"
        )
    max_class = int(class_ids.max().item())
    if expert_logits.shape[1] <= max_class:
        raise ValueError(
            f"nnUNet has {expert_logits.shape[1]} channels but class id "
            f"{max_class} was requested"
        )

    with torch.no_grad():
        if expert_class_groups is None:
            selected_expert = expert_logits[:, class_ids.long()]
        else:
            if len(expert_class_groups) != P:
                raise ValueError(
                    "expert_class_groups must contain one group per prompt"
                )
            expert_probabilities = expert_logits.float().softmax(dim=1)
            grouped_probabilities = []
            for group in expert_class_groups:
                indices = torch.as_tensor(
                    list(group), device=expert_logits.device, dtype=torch.long
                )
                if indices.numel() == 0 or int(indices.max()) >= expert_logits.shape[1]:
                    raise ValueError(f"Invalid expert class group: {tuple(group)}")
                grouped_probabilities.append(
                    expert_probabilities.index_select(1, indices).sum(dim=1)
                )
            grouped = torch.stack(grouped_probabilities, dim=1).clamp(
                1e-6, 1.0 - 1e-6
            )
            selected_expert = torch.logit(grouped)
        if selected_expert.shape[-3:] != (Z, H, W):
            selected_expert = F.interpolate(
                selected_expert.float(),
                size=(Z, H, W),
                mode="trilinear",
                align_corners=False,
            )
        target = (
            gt[:, None] == class_ids[None, :, None, None, None]
        ).float()
        base_error_by_prompt = F.binary_cross_entropy_with_logits(
            base_logits.detach().float(), target, reduction="none"
        )
        expert_error_by_prompt = F.binary_cross_entropy_with_logits(
            selected_expert.detach().float(), target, reduction="none"
        )
        if prompt_valid is None:
            prompt_weights = torch.ones(
                (B, P, 1, 1, 1), device=base_logits.device
            )
        else:
            prompt_weights = prompt_valid.to(
                device=base_logits.device, dtype=torch.float32
            )[:, :, None, None, None]
        prompt_denominator = prompt_weights.sum(dim=1).clamp_min(1.0)
        base_error = (
            base_error_by_prompt * prompt_weights
        ).sum(dim=1) / prompt_denominator
        expert_error = (
            expert_error_by_prompt * prompt_weights
        ).sum(dim=1) / prompt_denominator
        valid = (gt != -1) & valid_z[:, :, None, None]
        base_error = torch.where(valid, base_error, torch.zeros_like(base_error))
        expert_error = torch.where(
            valid, expert_error, torch.zeros_like(expert_error)
        )
    return base_error, expert_error
