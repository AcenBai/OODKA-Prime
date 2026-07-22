"""Core forward passes for contiguous 2.5D OODKA slice blocks."""

from __future__ import annotations

from typing import Dict, List, Tuple

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

        Zn_p, Zn_s, Zn_p_rec, Zn_s_rec = ae(Z_n)
        Zb_p, Zb_s = dis(Zb_res)

        out[f"Z_n{i}"] = Z_n
        out[f"Zn{i}_p"] = Zn_p
        out[f"Zn{i}_s"] = Zn_s
        out[f"Zn{i}_p_rec"] = Zn_p_rec
        out[f"Zn{i}_s_rec"] = Zn_s_rec
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
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute reconstruction and P/S separation across all levels."""
    eps = 1e-8
    levels = [2, 3, 4, 5]

    ae_losses = []
    ortho_losses = []

    for i in levels:
        Z_n = feats[f"Z_n{i}"]
        Zn_p_rec = feats[f"Zn{i}_p_rec"]
        Zn_s_rec = feats[f"Zn{i}_s_rec"]
        Zb_res = feats[f"Zb_res{i}"]
        Zn_p = feats[f"Zn{i}_p"]
        Zn_s = feats[f"Zn{i}_s"]
        Zb_p = feats[f"Zb{i}_p"]
        Zb_s = feats[f"Zb{i}_s"]

        ae_n = _normalized_mse_5d(Zn_p_rec + Zn_s_rec, Z_n, valid_z, eps)
        ae_b = _normalized_mse_5d(Zb_p + Zb_s, Zb_res, valid_z, eps)
        ae_losses.extend([ae_n, ae_b])

        with torch.autocast(device_type=Zb_p.device.type, enabled=False):
            ortho_losses.append(
                ortho_corr_loss(Zb_p.float(), Zb_s.float(), valid_z=valid_z)
            )
            ortho_losses.append(
                ortho_corr_loss(Zn_p.float(), Zn_s.float(), valid_z=valid_z)
            )

    return sum(ae_losses), sum(ortho_losses)


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
    """Create prompt-specific visual features and flatten in ``[B,Z,P]`` order."""
    if p_feature.shape != s_feature.shape or p_feature.ndim != 4:
        raise ValueError(
            f"P/S features must have equal [B*Z,C,H,W] shapes, "
            f"got {p_feature.shape} and {s_feature.shape}"
        )
    N, C, H, W = p_feature.shape
    if N != B * Z:
        raise ValueError(f"Visual batch={N} != B*Z={B*Z}")
    if gate.ndim != 2 or gate.shape[0] != B:
        raise ValueError(f"scale gate must be [B,P], got {gate.shape}")
    P = gate.shape[1]
    p_bz = p_feature.reshape(B, Z, C, H, W)[:, :, None]
    s_bz = s_feature.reshape(B, Z, C, H, W)[:, :, None]
    gate_bzp = gate[:, None, :, None, None, None]
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
        gate, B=B, P=P
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

    # Router order is [res2,res3,res4,res5]. BiomedParse predictor order is
    # coarse-to-fine [res5,res4,res3], while mask_features represents res2.
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
) -> Tuple[torch.Tensor, float, Dict[int, float | None]]:
    """Vectorized BCE/Dice objective over all ``B*P`` class-volume pairs."""
    B, P, Z, H, W = logits.shape
    if gt.shape != (B, Z, H, W):
        raise ValueError(f"GT must be [B,Z,H,W]={B,Z,H,W}, got {gt.shape}")
    if valid_z.shape != (B, Z):
        raise ValueError(f"valid_z must be [B,Z]={B,Z}, got {valid_z.shape}")
    if class_ids.shape != (P,):
        raise ValueError(f"class_ids must be [P]={P}, got {class_ids.shape}")

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
    )
    loss_seg = (pair_weights * loss_per_pair).sum() / pair_weights.sum().clamp_min(1e-6)

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
            nonempty = ~gt_empty[:, prompt_index]
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
        selected_expert = expert_logits[:, class_ids.long()]
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
        base_error = F.binary_cross_entropy_with_logits(
            base_logits.detach().float(), target, reduction="none"
        ).mean(dim=1)
        expert_error = F.binary_cross_entropy_with_logits(
            selected_expert.detach().float(), target, reduction="none"
        ).mean(dim=1)
        valid = (gt != -1) & valid_z[:, :, None, None]
        base_error = torch.where(valid, base_error, torch.zeros_like(base_error))
        expert_error = torch.where(
            valid, expert_error, torch.zeros_like(expert_error)
        )
    return base_error, expert_error


def forward_one_batch(
    batch_data: Dict,
    block_shape: List[int],
    prompt_features: dict,
    P: int,
    prompt_to_class_id: Dict[int, int],
    w_seg: float,
    w_ae: float,
    w_ort: float,
    model_nnunet: nn.Module,
    model_biomedparse: nn.Module,
    fusion_modules: Dict[str, nn.Module],
    device: torch.device,
    w_route: float = 0.0,
    w_p_ot: float = 0.0,
    w_s_ot: float = 0.0,
    route_sample: bool | None = None,
    ot_expert_perturbation: str | None = None,
) -> Tuple[torch.Tensor, Dict]:
    """
    Single training/validation forward pass on a batch.

    Args:
        fusion_modules: dict from build_fusion_modules()
        All other args same as original forward_one_batch.

    Returns:
        (total_loss, logs_dict)
    """
    pd, ph, pw = block_shape
    nnunet_images = batch_data["nnunet_image"].to(device)
    biomedparse_images = batch_data["biomedparse_image"].to(device)
    gt_patches = batch_data["gt"].to(device)
    valid_z = batch_data["valid_z"].to(device)
    if nnunet_images.ndim != 5 or biomedparse_images.ndim != 5:
        raise ValueError(
            "Expected nnUNet [B,Z,C,H,W] and BiomedParse [B,Z,3,H,W], "
            f"got {nnunet_images.shape} and {biomedparse_images.shape}"
        )
    B, batch_z = nnunet_images.shape[:2]
    if batch_z != pd:
        raise ValueError(f"Batch Z={batch_z} does not match configured block_z={pd}")
    if biomedparse_images.shape[:2] != (B, batch_z) or biomedparse_images.shape[2] != 3:
        raise ValueError(
            f"BiomedParse batch must be [B,Z,3,H,W], got {biomedparse_images.shape}"
        )
    if gt_patches.shape != (B, batch_z, ph, pw):
        raise ValueError(
            f"GT must be [B,Z,H,W]=[{B},{batch_z},{ph},{pw}], got {gt_patches.shape}"
        )
    if valid_z.shape != (B, batch_z):
        raise ValueError(f"valid_z must be [B,Z], got {valid_z.shape}")

    ae_mods = {k: fusion_modules[k] for k in fusion_modules if k.startswith("ae_")}
    dis_mods = {k: fusion_modules[k] for k in fusion_modules if k.startswith("dis_")}
    beta_router = fusion_modules["beta_router"]

    # Feature extraction
    nnunet_blocks = nnunet_images.permute(0, 2, 1, 3, 4).contiguous()
    enc_feats, _F_enc, expert_logits = extract_nnunet_features(
        model_nnunet, nnunet_blocks, device, return_logits=True
    )
    img_embeds_base, res3d = extract_biomedparse_backbone_features_2p5d(
        model_biomedparse,
        biomedparse_images,
        device,
        res_names=("res2", "res3", "res4", "res5"),
    )
    for rn in ["res2", "res3", "res4", "res5"]:
        img_embeds_base.pop(rn, None)

    # Disentangle
    feats = _disentangle_and_inject(enc_feats, res3d, img_embeds_base, ae_mods, dis_mods, device)

    # Reconstruction and P/S separation losses.
    loss_ae_z, loss_ortho = _compute_reconstruction_separation_losses(
        feats, valid_z
    )

    # Pixel decoder for p and s branches
    Dm = res3d["res3"].shape[2]
    if Dm != batch_z:
        raise RuntimeError(f"BiomedParse feature Z={Dm} != input Z={batch_z}")
    N = B * Dm

    mask_features_p, ms_p = _run_pixel_decoder(model_biomedparse, img_embeds_base, feats, "p", B, Dm)
    mask_features_s, ms_s = _run_pixel_decoder(model_biomedparse, img_embeds_base, feats, "s", B, Dm)

    # Pixel decoder reconstruction loss (full = p + s)
    injected_full = {k: v for k, v in img_embeds_base.items()}
    for i in [2, 3, 4, 5]:
        p5d = feats[f"Zb{i}_p"]
        s5d = feats[f"Zb{i}_s"]
        full = p5d + s5d
        Ci, Hi, Wi = full.shape[1], full.shape[3], full.shape[4]
        injected_full[f"res{i}"] = full.permute(0, 2, 1, 3, 4).reshape(N, Ci, Hi, Wi).contiguous()
    pd_out_full = model_biomedparse.sem_seg_head.pixel_decoder.forward_features(injected_full)
    mask_features_full, ms_full = parse_pixel_decoder_out(pd_out_full)

    loss_ae_pd_mask = _normalized_mse_flat_z(
        mask_features_p + mask_features_s,
        mask_features_full,
        valid_z,
    )
    loss_ae_pd_ms = []
    for mf, mp, ms_ in zip(ms_full, ms_p, ms_s):
        loss_ae_pd_ms.append(_normalized_mse_flat_z(mp + ms_, mf, valid_z))
    loss_ae_pd = (loss_ae_pd_mask + sum(loss_ae_pd_ms)) / (1 + len(loss_ae_pd_ms))
    loss_ae = (loss_ae_z + loss_ae_pd) / 9.0  # 8 Z-layer terms + 1 PD term in loss_ae_z

    # Prompt-only, scale-specific P/S routing. Expert features never enter this path.
    text_embedding = prompt_features.get("class_emb")
    if not torch.is_tensor(text_embedding):
        raise ValueError("prompt_features['class_emb'] is required by PromptBetaRouter")
    route = beta_router(
        text_embedding.detach(), batch_size=B, sample=route_sample
    )
    gate = route["gate"]

    # All prompts run through one predictor batch in [B,Z,P] pair order.
    all_prompt_logits = _predict_all_prompt_logits(
        sem_seg_head=model_biomedparse.sem_seg_head,
        mask_features_p=mask_features_p,
        mask_features_s=mask_features_s,
        ms_p=ms_p,
        ms_s=ms_s,
        gate=gate,
        prompt_features=prompt_features,
        B=B,
        Z=Dm,
        P=P,
        output_shape=(pd, ph, pw),
    )
    class_ids = torch.tensor(
        [prompt_to_class_id[prompt_index] for prompt_index in range(P)],
        device=device,
        dtype=gt_patches.dtype,
    )
    loss_seg, dice_mean, dice_per_class = _compute_segmentation_loss_and_metrics(
        all_prompt_logits,
        gt_patches,
        valid_z,
        class_ids,
    )

    # Dynamic expert transports are a detached training-only supervision path.
    if w_p_ot > 0 or w_s_ot > 0:
        base_error, expert_error = _compute_detached_pixel_error_maps(
            all_prompt_logits,
            expert_logits,
            gt_patches,
            valid_z,
            class_ids,
        )
        ot_output = fusion_modules["ot_distillation"](
            feats,
            gt=gt_patches,
            base_error=base_error,
            expert_error=expert_error,
            valid_z=valid_z,
            class_ids=[int(value) for value in class_ids.tolist()],
            enable_p=w_p_ot > 0,
            enable_s=w_s_ot > 0,
            expert_perturbation=ot_expert_perturbation,
        )
        loss_p_ot = ot_output["loss_p"]
        loss_s_ot = ot_output["loss_s"]
    else:
        loss_p_ot = all_prompt_logits.sum() * 0.0
        loss_s_ot = all_prompt_logits.sum() * 0.0
        ot_output = {"levels": {}}

    loss_route = route["kl"]
    total_loss = (
        w_seg * loss_seg
        + w_ae * loss_ae
        + w_ort * loss_ortho
        + w_route * loss_route
        + w_p_ot * loss_p_ot
        + w_s_ot * loss_s_ot
    )

    level_names = ("res2", "res3", "res4", "res5")
    gate_per_class_mean = {
        i: {
            level: float(route["mean"][i, level_index].detach().item())
            for level_index, level in enumerate(level_names)
        }
        for i in range(P)
    }

    logs = {
        "loss_total": float(total_loss.detach().item()),
        "loss_seg": float(loss_seg.detach().item()),
        "loss_ae": float(loss_ae.detach().item()),
        "loss_ortho": float(loss_ortho.detach().item()),
        "loss_route": float(loss_route.detach().item()),
        "loss_p_ot": float(loss_p_ot.detach().item()),
        "loss_s_ot": float(loss_s_ot.detach().item()),
        "dice_mean": dice_mean,
        "dice_per_class": dice_per_class,
        "gate_mean": float(gate.detach().mean().item()),
        "gate_std": float(gate.detach().std().item()),
        "gate_per_class_mean": gate_per_class_mean,
        "gate_per_level_mean": {
            level: float(gate[:, :, level_index].detach().mean().item())
            for level_index, level in enumerate(level_names)
        },
        "alpha_mean": float(route["alpha"].detach().mean().item()),
        "beta_mean": float(route["beta"].detach().mean().item()),
        "concentration_mean": float(
            route["concentration"].detach().mean().item()
        ),
        "ot_levels": {
            int(level): {
                name: float(value.detach().item())
                for name, value in values.items()
            }
            for level, values in ot_output["levels"].items()
        },
    }
    for level, values in logs["ot_levels"].items():
        for name, value in values.items():
            logs[f"ot_res{level}_{name}"] = value
    return total_loss, logs


@torch.no_grad()
def predict_block_logits_per_class(
    biomedparse_images: torch.Tensor,
    valid_z: torch.Tensor,
    output_size: Tuple[int, int],
    prompt_features: dict,
    P: int,
    model_biomedparse: nn.Module,
    fusion_modules: Dict[str, nn.Module],
    device: torch.device,
) -> torch.Tensor:
    """Pure-student inference on independent contiguous blocks.

    BiomedParse input is ``[B,Z,3,H,W]``. Returns ``[B,P,Z,H,W]`` logits.
    No nnUNet input, module, preprocessing, or cached expert statistic is used.
    """
    if biomedparse_images.ndim != 5:
        raise ValueError("Expected a 5D BiomedParse block tensor")
    B, Dm = biomedparse_images.shape[:2]
    if biomedparse_images.shape[:3] != (B, Dm, 3):
        raise ValueError(
            f"BiomedParse input must be [B,Z,3,H,W], got {biomedparse_images.shape}"
        )
    if valid_z.shape != (B, Dm):
        raise ValueError(f"valid_z must be [B,Z]={B,Dm}, got {valid_z.shape}")
    ph, pw = (int(output_size[0]), int(output_size[1]))

    dis_mods = {k: fusion_modules[k] for k in fusion_modules if k.startswith("dis_")}
    beta_router = fusion_modules["beta_router"]

    biomedparse_images = biomedparse_images.to(device)
    valid_z = valid_z.to(device)

    img_embeds_base, res3d = extract_biomedparse_backbone_features_2p5d(
        model_biomedparse,
        biomedparse_images,
        device,
        res_names=("res2", "res3", "res4", "res5"),
    )
    for rn in ["res2", "res3", "res4", "res5"]:
        img_embeds_base.pop(rn, None)

    # Disentangle BiomedParse features
    disentangled = {}
    for i in [2, 3, 4, 5]:
        Zb = res3d[f"res{i}"].to(device)
        Zb_p, Zb_s = dis_mods[f"dis_b_res{i}"](Zb)
        disentangled[f"Zb{i}_p"] = Zb_p
        disentangled[f"Zb{i}_s"] = Zb_s
        disentangled[f"Zb_res{i}"] = Zb

    text_embedding = prompt_features.get("class_emb")
    if not torch.is_tensor(text_embedding):
        raise ValueError("prompt_features['class_emb'] is required by PromptBetaRouter")
    route = beta_router(text_embedding.detach(), batch_size=B, sample=False)

    if res3d["res3"].shape[2] != Dm:
        raise RuntimeError(
            f"BiomedParse feature Z={res3d['res3'].shape[2]} != input Z={Dm}"
        )
    N = B * Dm

    # Pixel decoder for p and s branches
    def _inject_and_decode(branch):
        injected = {k: v for k, v in img_embeds_base.items()}
        for i in [2, 3, 4, 5]:
            f5d = disentangled[f"Zb{i}_{branch}"]
            Ci, Hi, Wi = f5d.shape[1], f5d.shape[3], f5d.shape[4]
            injected[f"res{i}"] = f5d.permute(0, 2, 1, 3, 4).reshape(N, Ci, Hi, Wi).contiguous()
        pd_out = model_biomedparse.sem_seg_head.pixel_decoder.forward_features(injected)
        return parse_pixel_decoder_out(pd_out)

    mask_features_p, ms_p = _inject_and_decode("p")
    mask_features_s, ms_s = _inject_and_decode("s")

    return _predict_all_prompt_logits(
        sem_seg_head=model_biomedparse.sem_seg_head,
        mask_features_p=mask_features_p,
        mask_features_s=mask_features_s,
        ms_p=ms_p,
        ms_s=ms_s,
        gate=route["gate"],
        prompt_features=prompt_features,
        B=B,
        Z=Dm,
        P=P,
        output_shape=(Dm, ph, pw),
    )
