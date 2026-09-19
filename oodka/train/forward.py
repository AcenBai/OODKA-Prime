"""Batch-level training and student-inference orchestration."""

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
    gates_for_biomedparse_predictor,
    parse_pixel_decoder_out,
)
from .forward_components import (
    _compute_detached_pixel_error_maps,
    _compute_reconstruction_separation_losses,
    _compute_segmentation_loss_and_metrics,
    _disentangle_and_inject,
    _fuse_all_prompt_features,
    _normalized_mse_5d,
    _normalized_mse_flat_z,
    _predict_all_prompt_logits,
    _run_pixel_decoder,
)

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
    expert_ortho_weight: float = 1.0,
    w_route: float = 0.0,
    w_p_ot: float = 0.0,
    w_s_ot: float = 0.0,
    route_sample: bool | None = None,
    ot_expert_perturbation: str | None = None,
    expert_class_groups: Sequence[Sequence[int]] | None = None,
    prompt_loss_reduction: str = "mean",
    return_logits: bool = False,
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
    loss_ae_z, loss_ortho, loss_ortho_student, loss_ortho_expert = (
        _compute_reconstruction_separation_losses(
            feats,
            valid_z,
            expert_ortho_weight=expert_ortho_weight,
        )
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

    # Prompt-only spatial P/S routing. One finest gate is shared across
    # Predictor levels through downsampling; expert features never enter.
    text_embedding = prompt_features.get("class_emb")
    if not torch.is_tensor(text_embedding):
        raise ValueError("prompt_features['class_emb'] is required by PromptBetaRouter")
    route = beta_router(
        text_embedding.detach(),
        spatial_size=mask_features_p.shape[-2:],
        batch_size=B,
        sample=route_sample,
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
        prompt_reduction=prompt_loss_reduction,
        prompt_valid=batch_data.get("prompt_valid"),
    )

    # Dynamic expert transports are a detached training-only supervision path.
    if w_p_ot > 0 or w_s_ot > 0:
        base_error, expert_error = _compute_detached_pixel_error_maps(
            all_prompt_logits,
            expert_logits,
            gt_patches,
            valid_z,
            class_ids,
            expert_class_groups=expert_class_groups,
            prompt_valid=batch_data.get("prompt_valid"),
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

    mean_mask_gate, mean_multi_scale_gates = gates_for_biomedparse_predictor(
        route["mean"].unsqueeze(0),
        B=1,
        P=P,
        mask_size=mask_features_p.shape[-2:],
        multi_scale_sizes=[feature.shape[-2:] for feature in ms_p],
    )
    mean_gates_by_level = {
        "res2": mean_mask_gate[0],
        "res5": mean_multi_scale_gates[0][0],
        "res4": mean_multi_scale_gates[1][0],
        "res3": mean_multi_scale_gates[2][0],
    }
    sampled_mask_gate, sampled_multi_scale_gates = (
        gates_for_biomedparse_predictor(
            gate,
            B=B,
            P=P,
            mask_size=mask_features_p.shape[-2:],
            multi_scale_sizes=[feature.shape[-2:] for feature in ms_p],
        )
    )
    sampled_gates_by_level = {
        "res2": sampled_mask_gate,
        "res5": sampled_multi_scale_gates[0],
        "res4": sampled_multi_scale_gates[1],
        "res3": sampled_multi_scale_gates[2],
    }
    level_names = ("res2", "res3", "res4", "res5")
    gate_per_class_mean = {
        i: {
            level: float(
                mean_gates_by_level[level][i].detach().mean().item()
            )
            for level in level_names
        }
        for i in range(P)
    }

    logs = {
        "loss_total": float(total_loss.detach().item()),
        "loss_seg": float(loss_seg.detach().item()),
        "loss_ae": float(loss_ae.detach().item()),
        "loss_ortho": float(loss_ortho.detach().item()),
        "loss_ortho_student": float(loss_ortho_student.detach().item()),
        "loss_ortho_expert": float(loss_ortho_expert.detach().item()),
        "loss_route": float(loss_route.detach().item()),
        "loss_p_ot": float(loss_p_ot.detach().item()),
        "loss_s_ot": float(loss_s_ot.detach().item()),
        "loss_p_ot_forward": float(
            ot_output.get("loss_p_forward", loss_p_ot).detach().item()
        ),
        "loss_s_ot_forward": float(
            ot_output.get("loss_s_forward", loss_s_ot).detach().item()
        ),
        "loss_p_ot_reverse": float(
            ot_output.get("loss_p_reverse", loss_p_ot * 0.0).detach().item()
        ),
        "loss_s_ot_reverse": float(
            ot_output.get("loss_s_reverse", loss_s_ot * 0.0).detach().item()
        ),
        "loss_p_ot_reverse_cosine": float(
            ot_output.get(
                "loss_p_reverse_cosine", loss_p_ot * 0.0
            ).detach().item()
        ),
        "loss_s_ot_reverse_cosine": float(
            ot_output.get(
                "loss_s_reverse_cosine", loss_s_ot * 0.0
            ).detach().item()
        ),
        "loss_p_ot_reverse_rms": float(
            ot_output.get("loss_p_reverse_rms", loss_p_ot * 0.0).detach().item()
        ),
        "loss_s_ot_reverse_rms": float(
            ot_output.get("loss_s_reverse_rms", loss_s_ot * 0.0).detach().item()
        ),
        "dice_mean": dice_mean,
        "dice_per_class": dice_per_class,
        "gate_mean": float(gate.detach().mean().item()),
        "gate_std": float(gate.detach().std().item()),
        "gate_per_class_mean": gate_per_class_mean,
        "gate_per_level_mean": {
            level: float(
                sampled_gates_by_level[level].detach().mean().item()
            )
            for level in level_names
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
    if return_logits:
        logs["_logits"] = all_prompt_logits
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
    text_embedding = prompt_features.get("class_emb")
    if not torch.is_tensor(text_embedding):
        raise ValueError(
            "prompt_features['class_emb'] is required by PromptBetaRouter"
        )
    route = beta_router(
        text_embedding.detach(),
        spatial_size=mask_features_p.shape[-2:],
        batch_size=B,
        sample=False,
    )

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
