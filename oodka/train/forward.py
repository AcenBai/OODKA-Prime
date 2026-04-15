"""Core forward pass: training one batch and inference for one patch."""

from __future__ import annotations

from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..models.feature_extraction import (
    extract_nnunet_features,
    extract_biomedparse_backbone_embeds_and_res_levels_3d,
)
from ..models.biomedparse_helpers import (
    parse_pixel_decoder_out,
    slice_prompt_features,
    select_best_mask_from_queries,
    run_biomedparse_predictor_override,
)
from ..models.losses import (
    mse_loss,
    ortho_corr_loss,
    spatial_cka_loss,
    entropy_loss,
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


def _compute_ae_ortho_ka_losses(feats: Dict, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute AE reconstruction, orthogonality, and CKA losses across all levels."""
    eps = 1e-8
    levels = [2, 3, 4, 5]

    ae_losses = []
    ortho_losses = []
    ka_losses = []

    for i in levels:
        Z_n = feats[f"Z_n{i}"]
        Zn_p_rec = feats[f"Zn{i}_p_rec"]
        Zn_s_rec = feats[f"Zn{i}_s_rec"]
        Zb_res = feats[f"Zb_res{i}"]
        Zn_p = feats[f"Zn{i}_p"]
        Zn_s = feats[f"Zn{i}_s"]
        Zb_p = feats[f"Zb{i}_p"]
        Zb_s = feats[f"Zb{i}_s"]

        ae_n = mse_loss(Z_n, Zn_p_rec + Zn_s_rec) / (Z_n.pow(2).mean() + eps)
        ae_b = mse_loss(Zb_res, Zb_p + Zb_s) / (Zb_res.pow(2).mean() + eps)
        ae_losses.extend([ae_n, ae_b])

        ortho_losses.append(ortho_corr_loss(Zb_p, Zb_s))
        ortho_losses.append(ortho_corr_loss(Zn_p, Zn_s))

        ka_losses.append(spatial_cka_loss(Zb_p, Zn_p))
        ka_losses.append(spatial_cka_loss(Zb_s, Zn_s))

    return sum(ae_losses), sum(ortho_losses), sum(ka_losses)


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


def forward_one_batch(
    batch_data: Dict,
    patch_size: List[int],
    prompt_features: dict,
    class_emb: torch.Tensor,
    P: int,
    prompt_to_class_id: Dict[int, int],
    w_seg: float,
    w_ae: float,
    w_ort: float,
    w_ka: float,
    model_nnunet: nn.Module,
    model_biomedparse: nn.Module,
    fusion_modules: Dict[str, nn.Module],
    device: torch.device,
    w_p_reg: float = 0.0,
    train: bool = True,
) -> Tuple[torch.Tensor, Dict]:
    """
    Single training/validation forward pass on a batch.

    Args:
        fusion_modules: dict from build_fusion_modules()
        All other args same as original forward_one_batch.

    Returns:
        (total_loss, logs_dict)
    """
    pd, ph, pw = patch_size
    nnunet_patches = batch_data["nnunet_patch"].to(device)
    biomedparse_patches = batch_data["biomedparse_patch"].to(device)
    gt_patches = batch_data["nnunet_seg"].to(device)
    B = nnunet_patches.shape[0]

    ae_mods = {k: fusion_modules[k] for k in fusion_modules if k.startswith("ae_")}
    dis_mods = {k: fusion_modules[k] for k in fusion_modules if k.startswith("dis_")}
    class_query_pooler = fusion_modules["class_query_pooler"]
    gate_net = fusion_modules["gate_net"]

    # Feature extraction
    enc_feats, F_enc = extract_nnunet_features(model_nnunet, nnunet_patches, device)
    img_embeds_base, res3d = extract_biomedparse_backbone_embeds_and_res_levels_3d(
        model_biomedparse, biomedparse_patches, device, res_names=("res2", "res3", "res4", "res5")
    )
    for rn in ["res2", "res3", "res4", "res5"]:
        img_embeds_base.pop(rn, None)

    # Disentangle
    feats = _disentangle_and_inject(enc_feats, res3d, img_embeds_base, ae_mods, dis_mods, device)

    # AE / Ortho / CKA losses
    loss_ae_z, loss_ortho, loss_ka = _compute_ae_ortho_ka_losses(feats, device)

    # Pixel decoder for p and s branches
    Dm = res3d["res3"].shape[2]
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

    eps = 1e-8
    loss_ae_pd_mask = mse_loss(mask_features_full, mask_features_p + mask_features_s) / (mask_features_full.pow(2).mean() + eps)
    loss_ae_pd_ms = []
    for mf, mp, ms_ in zip(ms_full, ms_p, ms_s):
        loss_ae_pd_ms.append(mse_loss(mf, mp + ms_) / (mf.pow(2).mean() + eps))
    loss_ae_pd = (loss_ae_pd_mask + sum(loss_ae_pd_ms)) / (1 + len(loss_ae_pd_ms))
    loss_ae = (loss_ae_z + loss_ae_pd) / 9.0  # 8 Z-layer terms + 1 PD term in loss_ae_z

    # Gate
    mu, _ = class_query_pooler(F_enc)
    tau_out = gate_net(mu)
    tau_mask = tau_out["mask"]
    tau_ms_list = tau_out.get("ms", [])

    # Per-class segmentation
    loss_seg_total = torch.tensor(0.0, device=device)
    loss_seg_denom = torch.tensor(0.0, device=device)
    dice_list = []
    dice_per_class = {}

    for p_idx in range(P):
        pf1 = slice_prompt_features(prompt_features, p_idx)
        for k, v in pf1.items():
            if torch.is_tensor(v):
                pf1[k] = v.to(device)

        tau_mask_bp = tau_mask[:, p_idx, :]
        C_mask = tau_mask_bp.shape[-1]
        tau_2d = tau_mask_bp.unsqueeze(1).expand(B, Dm, C_mask).reshape(N, C_mask, 1, 1)
        mask_features = tau_2d * mask_features_p + (1 - tau_2d) * mask_features_s

        multi_scale_features = []
        for i, (mp, ms_, tau_ms_i) in enumerate(zip(ms_p, ms_s, tau_ms_list)):
            tau_ms_bp = tau_ms_i[:, p_idx, :]
            C_i = tau_ms_bp.shape[-1]
            tau_ms_2d = tau_ms_bp.unsqueeze(1).expand(B, Dm, C_i).reshape(N, C_i, 1, 1)
            multi_scale_features.append(tau_ms_2d * mp + (1 - tau_ms_2d) * ms_)

        pred_out = run_biomedparse_predictor_override(
            model_biomedparse.sem_seg_head, multi_scale_features, mask_features, pf1
        )
        pred_gmasks = pred_out["pred_gmasks"]
        obj_exist = pred_out.get("object_existence", None)
        mask_logits = select_best_mask_from_queries(pred_gmasks, obj_exist)

        class_id = prompt_to_class_id[p_idx]
        valid = (gt_patches != -1).float().squeeze(1)
        gt_bin = (gt_patches == class_id).float().squeeze(1)

        mask_logits_3d = mask_logits.view(B, Dm, mask_logits.shape[-2], mask_logits.shape[-1])
        mask_logits_3d_rs = F.interpolate(
            mask_logits_3d.unsqueeze(1), size=(pd, ph, pw), mode="trilinear", align_corners=False
        ).squeeze(1)

        # Empty-GT protection (per sample)
        valid_d = valid.detach()
        valid_m = valid_d > 0.5
        valid_vox = valid_m.flatten(1).sum(1).float()
        thr = valid_vox * 0.0005
        gt_fg = ((gt_bin.detach() > 0.5) & valid_m).flatten(1).sum(1).float()
        gt_empty = gt_fg < thr

        bce_map = F.binary_cross_entropy_with_logits(mask_logits_3d_rs, gt_bin, reduction="none")
        bce_map = bce_map * valid_d
        denom = valid_d.sum(dim=(1, 2, 3)).clamp_min(1.0)
        l_bce = bce_map.sum(dim=(1, 2, 3)) / denom

        probs = torch.sigmoid(mask_logits_3d_rs) * valid_d
        tgt = gt_bin * valid_d
        inter = (probs * tgt).sum(dim=(1, 2, 3))
        union = probs.sum(dim=(1, 2, 3)) + tgt.sum(dim=(1, 2, 3))
        dice_soft = (2.0 * inter + 1e-6) / (union + 1e-6)
        l_dice = 1.0 - dice_soft

        l_seg_per = torch.where(gt_empty, l_bce, l_bce + l_dice)
        w_per = torch.where(gt_empty, torch.full_like(l_seg_per, 0.5), torch.ones_like(l_seg_per))
        loss_seg_total = loss_seg_total + (w_per * l_seg_per).sum()
        loss_seg_denom = loss_seg_denom + w_per.sum()

        # Metrics
        with torch.no_grad():
            pred_bin = (torch.sigmoid(mask_logits_3d_rs) > 0.5).float()
            nonempty = ~gt_empty
            if nonempty.any():
                inter_h = ((pred_bin > 0.5) & (gt_bin.detach() > 0.5) & valid_m).flatten(1).sum(1).float()
                union_h = ((pred_bin > 0.5) & valid_m).flatten(1).sum(1).float() + gt_fg
                dice_h = (2.0 * inter_h + 1e-6) / (union_h + 1e-6)
                d = float(dice_h[nonempty].mean().item())
                dice_list.append(d)
                dice_per_class[p_idx] = d
            else:
                dice_per_class[p_idx] = None

    loss_seg = loss_seg_total / loss_seg_denom.clamp_min(1e-6)
    dice_mean = float(np.mean(dice_list)) if dice_list else 0.0

    # Tau entropy regularization
    if w_p_reg > 0:
        tau_all = [tau_mask] + list(tau_ms_list)
        tau_cat = torch.cat(tau_all, dim=-1) if len(tau_all) > 1 else tau_mask
        loss_p_reg = entropy_loss(tau_cat.reshape(-1, tau_cat.shape[-1]))
    else:
        loss_p_reg = torch.tensor(0.0, device=device)

    total_loss = w_seg * loss_seg + w_ae * loss_ae + w_ort * loss_ortho + w_ka * loss_ka + w_p_reg * loss_p_reg

    tau_per_class_mean = {i: float(tau_mask[:, i, :].detach().mean().item()) for i in range(P)}

    logs = {
        "loss_total": float(total_loss.detach().item()),
        "loss_seg": float(loss_seg.detach().item()),
        "loss_ae": float(loss_ae.detach().item()),
        "loss_ortho": float(loss_ortho.detach().item()),
        "loss_ka": float(loss_ka.detach().item()),
        "loss_p_reg": float(loss_p_reg.detach().item()),
        "dice_mean": dice_mean,
        "dice_per_class": dice_per_class,
        "tau_mean": float(tau_mask.detach().mean().item()),
        "tau_std": float(tau_mask.detach().std().item()),
        "tau_per_class_mean": tau_per_class_mean,
        "tau_distribution": {
            i: tau_mask[:, i, :].detach().cpu().numpy().flatten().tolist() for i in range(P)
        },
        "tau_per_class_channel": tau_mask.detach().cpu().numpy().mean(axis=0).tolist(),
    }
    return total_loss, logs


@torch.no_grad()
def predict_patch_logits_per_class(
    nnunet_patch_5d: torch.Tensor,
    biomedparse_patch_4d: torch.Tensor,
    patch_size: List[int],
    prompt_features: dict,
    P: int,
    model_nnunet: nn.Module,
    model_biomedparse: nn.Module,
    fusion_modules: Dict[str, nn.Module],
    device: torch.device,
) -> torch.Tensor:
    """
    Inference-only forward on one 3D patch.

    Returns: [P, Z, H, W] logits.
    """
    pd, ph, pw = [int(x) for x in patch_size]

    dis_mods = {k: fusion_modules[k] for k in fusion_modules if k.startswith("dis_")}
    class_query_pooler = fusion_modules["class_query_pooler"]
    gate_net = fusion_modules["gate_net"]

    nnunet_patch_5d = nnunet_patch_5d.to(device)
    biomedparse_patch_4d = biomedparse_patch_4d.to(device)

    _enc_feats, F_enc = extract_nnunet_features(model_nnunet, nnunet_patch_5d, device)
    img_embeds_base, res3d = extract_biomedparse_backbone_embeds_and_res_levels_3d(
        model_biomedparse, biomedparse_patch_4d, device, res_names=("res2", "res3", "res4", "res5")
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

    mu, _ = class_query_pooler(F_enc)
    tau_out = gate_net(mu)
    tau_mask = tau_out["mask"]
    tau_ms_list = tau_out["ms"]

    B = 1
    Dm = res3d["res3"].shape[2]
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

    patch_logits_pc = []
    for p_idx in range(P):
        pf1 = slice_prompt_features(prompt_features, p_idx)
        for k, v in pf1.items():
            if torch.is_tensor(v):
                pf1[k] = v.to(device)

        tau_mask_bp = tau_mask[:, p_idx, :]
        C_mask = tau_mask_bp.shape[-1]
        tau_2d = tau_mask_bp.unsqueeze(1).expand(B, Dm, C_mask).reshape(N, C_mask, 1, 1)
        mask_features = tau_2d * mask_features_p + (1 - tau_2d) * mask_features_s

        multi_scale_features = []
        for i_ms, (mp, ms_, tau_ms_i) in enumerate(zip(ms_p, ms_s, tau_ms_list)):
            C_i = tau_ms_i[:, p_idx, :].shape[-1]
            tau_ms_2d = tau_ms_i[:, p_idx, :].unsqueeze(1).expand(B, Dm, C_i).reshape(N, C_i, 1, 1)
            multi_scale_features.append(tau_ms_2d * mp + (1 - tau_ms_2d) * ms_)

        pred_out = run_biomedparse_predictor_override(
            model_biomedparse.sem_seg_head, multi_scale_features, mask_features, pf1
        )
        pred_gmasks = pred_out["pred_gmasks"]
        mask_logits = select_best_mask_from_queries(pred_gmasks, pred_out.get("object_existence"))

        mask_logits_3d = mask_logits.view(B, Dm, mask_logits.shape[-2], mask_logits.shape[-1])
        mask_logits_3d_rs = F.interpolate(
            mask_logits_3d.unsqueeze(1), size=(pd, ph, pw), mode="trilinear", align_corners=False
        ).squeeze(1)
        patch_logits_pc.append(mask_logits_3d_rs[0])

    return torch.stack(patch_logits_pc, dim=0)
