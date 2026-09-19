#!/usr/bin/env python3
"""Router policy views: prior residual, class residual, 8x8 coefficients, CT overlay.

Optional --with_ablation also decodes this slice with g=1, g=0, and the learned gate.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/oodka_router_policy_mpl")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk
import torch
import torch.nn.functional as F
from matplotlib.colors import TwoSlopeNorm

from oodka.config import TrainConfig
from oodka.data.slice_dataset import FullSliceBlockDataset
from oodka.models.beta_router import PromptBetaRouter
from oodka.models.feature_extraction import (
    extract_biomedparse_backbone_features_2p5d,
    extract_nnunet_features,
)
from oodka.models.prompts import build_text_prompts_for_dataset
from oodka.train.forward import _predict_all_prompt_logits, _run_pixel_decoder
from oodka.train.model_builder import (
    build_fusion_modules,
    build_prompt_features,
    load_frozen_backbones,
    load_frozen_biomedparse,
)
from oodka.utils.io_utils import find_raw_image_files
from visualize_mechanism_v3 import CLASS_COLORS, LEVELS, _ct_limits, _select_slice

NAMES = ("LV", "RV", "LA", "RA", "MYO", "AO", "PA")
IDS = (1, 2, 3, 4, 5, 6, 7)
PRIOR_MEAN = 0.7


def _upsample(value: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    tensor = torch.from_numpy(np.asarray(value, dtype=np.float32))
    if tensor.ndim == 2:
        tensor = tensor[None, None]
    elif tensor.ndim == 3:
        tensor = tensor[:, None]
    else:
        raise ValueError(value.shape)
    resized = F.interpolate(tensor, size=size, mode="bilinear", align_corners=False)
    return resized[:, 0].numpy()


def _class_grid(n_class: int) -> tuple[int, int]:
    columns = 4
    rows = int(np.ceil(n_class / columns))
    return rows, columns


def _imshow_grid(
    path: Path,
    maps: np.ndarray,
    *,
    title: str,
    cmap: str,
    vmax: float,
    names: tuple[str, ...],
    present: set[int],
    cbar_label: str,
    diverging: bool = True,
) -> None:
    rows, columns = _class_grid(len(names))
    figure, axes = plt.subplots(
        rows, columns, figsize=(3.3 * columns, 3.15 * rows + 0.6), constrained_layout=True
    )
    axes = np.atleast_2d(axes)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax) if diverging else None
    last = None
    for index, name in enumerate(names):
        axis = axes[index // columns, index % columns]
        cid = IDS[index]
        kwargs = {"cmap": cmap, "interpolation": "nearest"}
        if diverging:
            kwargs["norm"] = norm
        else:
            kwargs["vmin"] = -vmax
            kwargs["vmax"] = vmax
        last = axis.imshow(maps[index], **kwargs)
        suffix = "" if cid in present else "  (absent on slice)"
        axis.set_title(f"{name}{suffix}", fontsize=11)
        axis.set_xticks([])
        axis.set_yticks([])
    for index in range(len(names), rows * columns):
        axes[index // columns, index % columns].axis("off")
    figure.colorbar(last, ax=axes, shrink=0.72, label=cbar_label)
    figure.suptitle(title, fontsize=13)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=170)
    plt.close(figure)


def _plot_coefficients(
    path: Path,
    alpha_w: np.ndarray,
    beta_w: np.ndarray,
    names: tuple[str, ...],
) -> None:
    vmax = max(float(np.max(np.abs(alpha_w))), float(np.max(np.abs(beta_w))), 1e-6)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    figure, axes = plt.subplots(
        len(names), 2, figsize=(6.4, 2.15 * len(names) + 0.7), constrained_layout=True
    )
    last = None
    for row, name in enumerate(names):
        last = axes[row, 0].imshow(alpha_w[row], cmap="coolwarm", norm=norm)
        axes[row, 1].imshow(beta_w[row], cmap="coolwarm", norm=norm)
        axes[row, 0].set_ylabel(name, fontsize=11)
        if row == 0:
            axes[row, 0].set_title(r"$w^{(\alpha)}$  (raises P if $>$ 0)", fontsize=10)
            axes[row, 1].set_title(r"$w^{(\beta)}$  (raises S if $>$ 0)", fontsize=10)
        for axis in axes[row]:
            axis.set_xticks([])
            axis.set_yticks([])
    figure.colorbar(last, ax=axes, shrink=0.62, label="RBF coefficient")
    figure.suptitle("8×8 Gaussian-center weights  ·  one pair per prompt", fontsize=13)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=170)
    plt.close(figure)


def _plot_overlay(
    path: Path,
    image: np.ndarray,
    gt: np.ndarray,
    gate: np.ndarray,
    *,
    present: set[int],
    vmax: float,
) -> None:
    gate512 = _upsample(gate, image.shape[-2:])
    lo, hi = _ct_limits(image)
    rows, columns = _class_grid(len(NAMES) + 1)
    figure, axes = plt.subplots(
        rows, columns, figsize=(3.4 * columns, 3.2 * rows + 0.5), constrained_layout=True
    )
    axes = np.atleast_2d(axes)
    axes[0, 0].imshow(image, cmap="gray", vmin=lo, vmax=hi)
    for cid in IDS:
        mask = gt == cid
        if mask.any():
            axes[0, 0].contour(mask, levels=[0.5], colors=[CLASS_COLORS[cid - 1]], linewidths=1.1)
    axes[0, 0].set_title("CT + GT", fontsize=11)
    axes[0, 0].axis("off")
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    last = None
    for index, name in enumerate(NAMES):
        axis = axes.ravel()[index + 1]
        axis.imshow(image, cmap="gray", vmin=lo, vmax=hi)
        last = axis.imshow(gate512[index] - PRIOR_MEAN, cmap="coolwarm", norm=norm, alpha=0.62)
        mask = gt == IDS[index]
        if mask.any():
            axis.contour(mask, levels=[0.5], colors=["#111111"], linewidths=1.3)
        suffix = "" if IDS[index] in present else "  absent"
        axis.set_title(f"{name}  g−0.7{suffix}", fontsize=10)
        axis.axis("off")
    for axis in axes.ravel()[len(NAMES) + 1 :]:
        axis.axis("off")
    figure.colorbar(last, ax=axes, shrink=0.62, label="P-gate minus prior 0.7")
    figure.suptitle("Learned gate on this slice  ·  warm = more P than prior", fontsize=13)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=170)
    plt.close(figure)


def _organ_stats(
    gate: np.ndarray, gt: np.ndarray
) -> list[dict]:
    gate512 = _upsample(gate, gt.shape[-2:])
    heart = gt > 0
    rows = []
    for index, (name, cid) in enumerate(zip(NAMES, IDS)):
        organ = gt == cid
        rec = {
            "name": name,
            "present": bool(organ.any()),
            "pixels": int(organ.sum()),
            "g_all": float(gate[index].mean()),
            "g_heart": float(gate512[index][heart].mean()) if heart.any() else float("nan"),
            "g_bg": float(gate512[index][~heart].mean()),
        }
        if organ.any():
            rec["g_organ"] = float(gate512[index][organ].mean())
            rec["g_organ_minus_prior"] = rec["g_organ"] - PRIOR_MEAN
        else:
            rec["g_organ"] = float("nan")
            rec["g_organ_minus_prior"] = float("nan")
        rows.append(rec)
    return rows


def _plot_organ_table(path: Path, rows: list[dict]) -> None:
    figure, axis = plt.subplots(figsize=(10.8, 3.6))
    axis.axis("off")
    cells = []
    for row in rows:
        cells.append(
            [
                row["name"],
                "yes" if row["present"] else "no",
                f"{row['pixels']}",
                f"{row['g_organ']:.3f}" if row["present"] else "—",
                f"{row['g_organ_minus_prior']:+.3f}" if row["present"] else "—",
                f"{row['g_heart']:.3f}",
                f"{row['g_bg']:.3f}",
                f"{row['g_all']:.3f}",
            ]
        )
    table = axis.table(
        cellText=cells,
        colLabels=[
            "Class",
            "On slice",
            "GT px",
            "g on organ",
            "vs prior 0.7",
            "g on heart",
            "g on bg",
            "g whole field",
        ],
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.15, 1.45)
    axis.set_title(
        "P-gate on this slice  ·  lower g = more S   ·  prior mean = 0.7",
        fontsize=12,
        pad=16,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(figure)


def _plot_organ_energy(
    path: Path,
    gt: np.ndarray,
    decoder_p: np.ndarray,
    decoder_s: np.ndarray,
    rows: list[dict],
) -> None:
    present = [row for row in rows if row["present"]]
    if not present:
        return
    labels = [row["name"] for row in present]
    p_on = []
    s_on = []
    g_on = []
    for row in present:
        mask = gt == IDS[NAMES.index(row["name"])]
        p_on.append(float(decoder_p[mask].mean()))
        s_on.append(float(decoder_s[mask].mean()))
        g_on.append(row["g_organ"])
    figure, axes = plt.subplots(1, 2, figsize=(9.6, 3.6), constrained_layout=True)
    x = np.arange(len(labels))
    axes[0].bar(x - 0.18, p_on, 0.36, label="P decoder RMS", color="#3b6ea5")
    axes[0].bar(x + 0.18, s_on, 0.36, label="S decoder RMS", color="#c47b2b")
    axes[0].set_xticks(x, labels)
    axes[0].set_ylabel("Mean RMS on GT organ")
    axes[0].legend(fontsize=8)
    axes[0].set_title("Decoder energy on this slice's organs")
    colors = ["#2a9d8f" if value >= PRIOR_MEAN else "#e76f51" for value in g_on]
    axes[1].bar(x, g_on, color=colors)
    axes[1].axhline(PRIOR_MEAN, color="0.3", ls="--", lw=1, label="prior 0.7")
    axes[1].set_xticks(x, labels)
    axes[1].set_ylim(0.0, 1.0)
    axes[1].set_ylabel("P gate on organ")
    axes[1].legend(fontsize=8)
    axes[1].set_title("Router preference on the same masks")
    figure.suptitle("What the gate does on anatomy that is actually present", fontsize=12)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=170)
    plt.close(figure)


def _plot_ablation(
    path: Path,
    image: np.ndarray,
    gt: np.ndarray,
    probs: dict[str, np.ndarray],
    present_ids: list[int],
) -> None:
    lo, hi = _ct_limits(image)
    columns = ("learned", "pure_p", "pure_s")
    titles = ("Learned gate", "Forced g=1 (pure P)", "Forced g=0 (pure S)")
    figure, axes = plt.subplots(
        len(present_ids),
        4,
        figsize=(13.2, 3.05 * len(present_ids) + 0.6),
        constrained_layout=True,
    )
    if len(present_ids) == 1:
        axes = np.atleast_2d(axes)
    for row, cid in enumerate(present_ids):
        name = NAMES[cid - 1]
        axes[row, 0].imshow(image, cmap="gray", vmin=lo, vmax=hi)
        axes[row, 0].contour(gt == cid, levels=[0.5], colors=["#00ff3b"], linewidths=1.2)
        axes[row, 0].set_ylabel(name, fontsize=11)
        axes[row, 0].set_title("GT" if row == 0 else "")
        axes[row, 0].axis("off")
        for column, key in enumerate(columns, start=1):
            prob = probs[key][cid - 1]
            axes[row, column].imshow(prob, cmap="viridis", vmin=0.0, vmax=1.0)
            if (gt == cid).any():
                axes[row, column].contour(
                    gt == cid, levels=[0.5], colors=["#00ff3b"], linewidths=1.0
                )
            if (prob >= 0.5).any():
                axes[row, column].contour(
                    prob >= 0.5, levels=[0.5], colors=["#ff2d2d"], linewidths=0.9
                )
            if row == 0:
                axes[row, column].set_title(titles[column - 1], fontsize=10)
            axes[row, column].axis("off")
    figure.suptitle(
        "Class probability  ·  green=GT  ·  red=pred≥0.5",
        fontsize=13,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=170)
    plt.close(figure)


def _extract_coefficients(router: PromptBetaRouter, text_embedding: torch.Tensor) -> dict:
    hidden = router.trunk(router.norm(text_embedding.float()))
    alpha_w = router.alpha_coeff(hidden).detach().cpu().numpy().reshape(-1, 8, 8)
    beta_w = router.beta_coeff(hidden).detach().cpu().numpy().reshape(-1, 8, 8)
    return {
        "alpha": alpha_w,
        "beta": beta_w,
        "alpha_bias": float(router.alpha_bias.detach().cpu()),
        "beta_bias": float(router.beta_bias.detach().cpu()),
    }


def _run_ablation(
    *,
    checkpoint: dict,
    saved: dict,
    cfg: TrainConfig,
    device: torch.device,
    case_id: str,
    split: str,
    slice_index: int,
    prompt_features: dict,
    prompts: dict,
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, int]:
    images_dir = cfg.imagesTr_dir if split == "val" else cfg.imagesTs_dir
    labels_dir = cfg.labelsTr_dir if split == "val" else cfg.labelsTs_dir
    with open(cfg.dataset_json_path, encoding="utf-8") as handle:
        dataset_json = json.load(handle)
    ending = dataset_json.get("file_ending", ".nii.gz")
    gt_volume = np.asarray(
        sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(labels_dir, case_id + ending)))
    )
    center = _select_slice(gt_volume, "all_classes", slice_index)
    dataset = FullSliceBlockDataset(
        [case_id],
        nnunet_preproc_dir=cfg.nnunet_preproc_dir,
        images_dir=images_dir,
        labels_dir=labels_dir,
        file_ending=ending,
        image_size=cfg.image_size,
        block_z=cfg.block_z,
        norm_mode=cfg.norm_mode,
        window_level=cfg.window_level,
        window_width=cfg.window_width,
        low_percentile=cfg.low_percentile,
        high_percentile=cfg.high_percentile,
        raw_cache_cases=1,
        require_no_crop=cfg.require_no_crop,
        biomedparse_modality=cfg.biomedparse_modality,
        biomedparse_preproc_dir=cfg.biomedparse_preproc_dir,
        pseudo_rgb_mode=cfg.pseudo_rgb_mode,
    )
    record_index = next(
        index
        for index, (_case_id, z_start, valid_count) in enumerate(dataset.records)
        if z_start <= center < z_start + valid_count
    )
    item = dataset[record_index]
    center_local = center - int(item["z_start"])
    bp = item["biomedparse_image"].unsqueeze(0)
    nn_input = item["nnunet_image"].unsqueeze(0).permute(0, 2, 1, 3, 4).contiguous()
    gt_slice = np.asarray(item["gt"][center_local].cpu())
    image = np.asarray(item["nnunet_image"][center_local, 0].cpu())
    model_nnunet, model_biomedparse = load_frozen_backbones(
        cfg.nnunet_model_dir, cfg.fold, device
    )
    modules = build_fusion_modules(
        model_nnunet,
        model_biomedparse,
        len(prompts),
        device,
        text_dim=int(prompt_features["class_emb"].shape[-1]),
        route_prior_p_mean=float(saved.get("route_prior_p_mean", cfg.route_prior_p_mean)),
        route_prior_concentration=float(
            saved.get("route_prior_concentration", cfg.route_prior_concentration)
        ),
        route_spatial_basis_grid_size=int(
            saved.get("route_spatial_basis_grid_size", cfg.route_spatial_basis_grid_size)
        ),
        route_spatial_basis_sigma=float(
            saved.get("route_spatial_basis_sigma", cfg.route_spatial_basis_sigma)
        ),
        ot_feature_weight=float(saved.get("ot_feature_weight", cfg.ot_feature_weight)),
        ot_coordinate_weight=float(saved.get("ot_coordinate_weight", cfg.ot_coordinate_weight)),
        ot_coordinate_radius=float(saved.get("ot_coordinate_radius", 0.0)),
        p_ot_semantic_weight=float(saved.get("p_ot_semantic_weight", cfg.p_ot_semantic_weight)),
        s_gain_mode=str(saved.get("s_gain_mode", "hard_positive")),
        s_gain_temperature=float(saved.get("s_gain_temperature", cfg.s_gain_temperature)),
        p_ot_epsilon=float(saved.get("p_ot_epsilon", cfg.p_ot_epsilon)),
        s_ot_epsilon=float(saved.get("s_ot_epsilon", cfg.s_ot_epsilon)),
        s_ot_rho_base=float(saved.get("s_ot_rho_base", cfg.s_ot_rho_base)),
        s_ot_rho_expert=float(saved.get("s_ot_rho_expert", cfg.s_ot_rho_expert)),
        ot_sinkhorn_iterations=int(saved.get("ot_sinkhorn_iterations", cfg.ot_sinkhorn_iterations)),
        ot_max_grid_size=int(saved.get("ot_max_grid_size", cfg.ot_max_grid_size)),
        s_transport_mode=str(saved.get("s_transport_mode", "unbalanced")),
        s_partial_mass_fraction=float(saved.get("s_partial_mass_fraction", 0.5)),
        expert_adapter_variant=str(saved.get("expert_adapter_variant", "legacy")),
        remove_res5_expert_branch_norm=bool(
            saved.get("remove_res5_expert_branch_norm", False)
        ),
    )
    for name, module in modules.items():
        if name in checkpoint:
            module.load_state_dict(checkpoint[name])
        module.eval()

    output_hw = tuple(int(value) for value in gt_volume.shape[-2:])
    with torch.no_grad():
        _expert_raw, _deepest, _expert_logits = extract_nnunet_features(
            model_nnunet, nn_input.to(device), device, return_logits=True
        )
        embeds, student_raw = extract_biomedparse_backbone_features_2p5d(
            model_biomedparse,
            bp.to(device),
            device,
            res_names=("res2", "res3", "res4", "res5"),
        )
        features: dict[str, torch.Tensor] = {}
        for level in LEVELS:
            student = student_raw[f"res{level}"]
            student_p, student_s = modules[f"dis_b_res{level}"](student)
            features[f"Zb{level}_p"] = student_p
            features[f"Zb{level}_s"] = student_s
        embeds_base = dict(embeds)
        for level in LEVELS:
            embeds_base.pop(f"res{level}", None)
        mask_p, multi_p = _run_pixel_decoder(
            model_biomedparse, embeds_base, features, "p", B=1, Dm=bp.shape[1]
        )
        mask_s, multi_s = _run_pixel_decoder(
            model_biomedparse, embeds_base, features, "s", B=1, Dm=bp.shape[1]
        )
        route = modules["beta_router"](
            prompt_features["class_emb"].detach(),
            spatial_size=mask_p.shape[-2:],
            batch_size=1,
            sample=False,
        )
        learned = route["gate"]
        ones = torch.ones_like(learned)
        zeros = torch.zeros_like(learned)
        probs = {}
        for key, gate in (("learned", learned), ("pure_p", ones), ("pure_s", zeros)):
            logits = _predict_all_prompt_logits(
                sem_seg_head=model_biomedparse.sem_seg_head,
                mask_features_p=mask_p,
                mask_features_s=mask_s,
                ms_p=multi_p,
                ms_s=multi_s,
                gate=gate,
                prompt_features=prompt_features,
                B=1,
                Z=bp.shape[1],
                P=len(prompts),
                output_shape=(bp.shape[1], *output_hw),
            )
            probs[key] = torch.sigmoid(logits[0, :, center_local]).detach().cpu().numpy()
    return probs, image, gt_slice, center


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--case_dir", required=True)
    parser.add_argument("--case_id", default="heart_1004")
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--slice_index", type=int, default=79)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--block_z", type=int, default=4)
    parser.add_argument("--with_ablation", action="store_true")
    args = parser.parse_args()

    case_dir = Path(args.case_dir)
    decision = np.load(case_dir / "decision" / "decision_maps.npz")
    representation = np.load(case_dir / "representation" / "representation_maps.npz")
    gate = np.asarray(decision["gate_mean"], dtype=np.float64)
    gt = np.asarray(representation["gt"])
    present = {int(value) for value in np.unique(gt) if int(value) > 0}

    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    saved = checkpoint.get("config", {})
    cfg = TrainConfig(
        device=args.device,
        block_z=args.block_z,
        num_workers=0,
        dataset_name=str(saved.get("dataset_name", "Dataset009_CT_OOD")),
        image_size=int(saved.get("image_size", 512)),
        norm_mode=str(saved.get("norm_mode", "ct")),
        pseudo_rgb_mode=str(saved.get("pseudo_rgb_mode", "adjacent")),
        low_percentile=float(saved.get("low_percentile", 1.0)),
        high_percentile=float(saved.get("high_percentile", 99.0)),
        require_no_crop=bool(saved.get("require_no_crop", True)),
        biomedparse_modality=int(saved.get("biomedparse_modality", 0)),
    )
    cfg.resolve_paths()
    images_dir = cfg.imagesTr_dir if args.split == "val" else cfg.imagesTs_dir
    labels_dir = cfg.labelsTr_dir if args.split == "val" else cfg.labelsTs_dir
    with open(cfg.dataset_json_path, encoding="utf-8") as handle:
        ending = json.load(handle).get("file_ending", ".nii.gz")
    image_files = find_raw_image_files(images_dir, args.case_id, ending)
    if not image_files:
        raise FileNotFoundError(args.case_id)
    image_volume = np.asarray(sitk.GetArrayFromImage(sitk.ReadImage(image_files[0])))
    gt_volume = np.asarray(
        sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(labels_dir, args.case_id + ending)))
    )
    center = _select_slice(gt_volume, "all_classes", args.slice_index)
    image = image_volume[center]

    prior_delta = gate - PRIOR_MEAN
    residual = gate - gate.mean(axis=0, keepdims=True)
    prior_vmax = max(float(np.max(np.abs(prior_delta))), 1e-6)
    residual_vmax = max(float(np.max(np.abs(residual))), 1e-6)
    out = case_dir / "decision" / "router_policy"
    _imshow_grid(
        out / "01_gate_minus_prior.png",
        prior_delta,
        title="P-gate minus prior 0.7   ·   red=more P   blue=more S",
        cmap="coolwarm",
        vmax=prior_vmax,
        names=NAMES,
        present=present,
        cbar_label="g − 0.7",
    )
    _imshow_grid(
        out / "02_gate_class_residual.png",
        residual,
        title="Class residual  g_c − mean_c(g)   ·   shared radial field removed",
        cmap="coolwarm",
        vmax=residual_vmax,
        names=NAMES,
        present=present,
        cbar_label="class-specific leftover",
    )
    _plot_overlay(
        out / "03_gate_on_ct.png",
        image,
        gt,
        gate,
        present=present,
        vmax=prior_vmax,
    )
    rows = _organ_stats(gate, gt)
    _plot_organ_table(out / "04_organ_gate_table.png", rows)
    _plot_organ_energy(
        out / "05_organ_decoder_vs_gate.png",
        gt,
        np.asarray(representation["decoder_res2_p"]),
        np.asarray(representation["decoder_res2_s"]),
        rows,
    )

    model_text = load_frozen_biomedparse(device)
    prompts, _prompt_to_class_id = build_text_prompts_for_dataset(
        dataset_name=cfg.dataset_name
    )
    prompt_features = build_prompt_features(model_text, prompts, device)
    router = PromptBetaRouter(
        text_dim=int(prompt_features["class_emb"].shape[-1]),
        prior_p_mean=float(saved.get("route_prior_p_mean", 0.7)),
        prior_concentration=float(saved.get("route_prior_concentration", 10.0)),
        basis_grid_size=int(saved.get("route_spatial_basis_grid_size", 8)),
        basis_sigma=float(saved.get("route_spatial_basis_sigma", 0.0)),
    ).to(device)
    router.load_state_dict(checkpoint["beta_router"])
    router.eval()
    with torch.no_grad():
        coeffs = _extract_coefficients(router, prompt_features["class_emb"].detach())
    _plot_coefficients(out / "06_basis_coefficients.png", coeffs["alpha"], coeffs["beta"], NAMES)

    files = [
        "01_gate_minus_prior.png",
        "02_gate_class_residual.png",
        "03_gate_on_ct.png",
        "04_organ_gate_table.png",
        "05_organ_decoder_vs_gate.png",
        "06_basis_coefficients.png",
    ]
    if args.with_ablation:
        probs, ab_image, ab_gt, _center = _run_ablation(
            checkpoint=checkpoint,
            saved=saved,
            cfg=cfg,
            device=device,
            case_id=args.case_id,
            split=args.split,
            slice_index=args.slice_index,
            prompt_features=prompt_features,
            prompts=prompts,
        )
        present_ids = [cid for cid in IDS if (ab_gt == cid).any()]
        _plot_ablation(out / "07_gate_ablation.png", ab_image, ab_gt, probs, present_ids)
        files.append("07_gate_ablation.png")

    (out / "README.json").write_text(
        json.dumps(
            {
                "case_id": args.case_id,
                "z": int(center),
                "prior_p_mean": PRIOR_MEAN,
                "present_labels": sorted(present),
                "organ_gate": rows,
                "alpha_bias": coeffs["alpha_bias"],
                "beta_bias": coeffs["beta_bias"],
                "files": files,
            },
            indent=2,
        )
    )
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
