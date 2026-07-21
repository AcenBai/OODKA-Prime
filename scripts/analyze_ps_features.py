#!/usr/bin/env python3
"""Visualize and quantify student P/S roles without loading nnUNet."""

from __future__ import annotations

import argparse
import json
import os
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/oodka_ps_mpl")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk
import torch
import torch.nn.functional as F

from oodka.config import EvalConfig
from oodka.data.slice_dataset import normalize_biomedparse_volume
from oodka.eval.eval_oodka import _make_block_batch
from oodka.models.feature_extraction import extract_biomedparse_backbone_features_2p5d
from oodka.models.prompts import build_text_prompts_for_dataset
from oodka.train.model_builder import (
    build_fusion_modules,
    build_prompt_features,
    load_frozen_biomedparse,
)
from oodka.utils.io_utils import find_raw_image_files


def _boundary(mask: torch.Tensor) -> torch.Tensor:
    value = mask.float().unsqueeze(0).unsqueeze(0)
    dilation = F.max_pool2d(value, 3, 1, 1)
    erosion = -F.max_pool2d(-value, 3, 1, 1)
    return (dilation - erosion).squeeze() > 0


def _safe_mean(value: torch.Tensor, mask: torch.Tensor) -> float:
    return float(value[mask].mean().item()) if mask.any() else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--case_id", default="heart_1004")
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--block_z", type=int, default=6)
    parser.add_argument(
        "--output_dir",
        default="outputs/oodka_ot_experiments/interpretability/ps_features",
    )
    args = parser.parse_args()

    cfg = EvalConfig(device=args.device, block_z=args.block_z, split=args.split)
    cfg.resolve_paths()
    images_dir = cfg.imagesTr_dir if args.split == "val" else cfg.imagesTs_dir
    labels_dir = cfg.labelsTr_dir if args.split == "val" else cfg.labelsTs_dir
    with open(cfg.dataset_json_path, encoding="utf-8") as file_handle:
        dataset_json = json.load(file_handle)
    ending = dataset_json.get("file_ending", ".nii.gz")
    image_files = find_raw_image_files(images_dir, args.case_id, ending)
    if not image_files:
        raise FileNotFoundError(args.case_id)
    raw = np.asarray(sitk.GetArrayFromImage(sitk.ReadImage(image_files[0])))
    gt = np.asarray(
        sitk.GetArrayFromImage(
            sitk.ReadImage(os.path.join(labels_dir, args.case_id + ending))
        )
    )
    foreground_area = (gt > 0).reshape(gt.shape[0], -1).sum(axis=1)
    center = int(foreground_area.argmax())
    z_start = max(0, min(center - args.block_z // 2, gt.shape[0] - args.block_z))
    normalized = normalize_biomedparse_volume(
        raw,
        norm_mode=cfg.norm_mode,
        window_level=cfg.window_level,
        window_width=cfg.window_width,
        low_percentile=cfg.low_percentile,
        high_percentile=cfg.high_percentile,
    )
    bp, valid_z, counts = _make_block_batch(
        normalized,
        [z_start],
        block_z=args.block_z,
        image_size=cfg.image_size,
    )

    device = torch.device(args.device)
    model = load_frozen_biomedparse(device)
    prompts, _mapping = build_text_prompts_for_dataset(dataset_name=cfg.dataset_name)
    prompt_features = build_prompt_features(model, prompts, device)
    modules = build_fusion_modules(
        None,
        model,
        len(prompts),
        device,
        text_dim=int(prompt_features["class_emb"].shape[-1]),
    )
    checkpoint = torch.load(args.checkpoint, map_location=device)
    for name, module in modules.items():
        module.load_state_dict(checkpoint[name])
        module.eval()

    with torch.no_grad():
        _embeds, features = extract_biomedparse_backbone_features_2p5d(
            model, bp.to(device), device
        )
        maps = {}
        metrics = {}
        center_local = center - z_start
        gt_center = torch.from_numpy(gt[center] > 0).to(device)
        gt_boundary = _boundary(gt_center)
        gt_interior = gt_center & ~gt_boundary
        gt_background = ~gt_center
        for level in [2, 3, 4, 5]:
            p_feature, s_feature = modules[f"dis_b_res{level}"](
                features[f"res{level}"]
            )
            p_energy = p_feature.square().sum(dim=1).sqrt()[0, center_local]
            s_energy = s_feature.square().sum(dim=1).sqrt()[0, center_local]
            specificity = s_energy / (p_energy + s_energy + 1e-8)
            stacked = torch.stack((p_energy, s_energy, specificity)).unsqueeze(0)
            resized = F.interpolate(
                stacked,
                size=gt.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )[0]
            p_map, s_map, specificity_map = resized
            p_fg = _safe_mean(p_map, gt_center)
            p_bg = _safe_mean(p_map, gt_background)
            s_boundary = _safe_mean(specificity_map, gt_boundary)
            s_interior = _safe_mean(specificity_map, gt_interior)
            metrics[f"res{level}"] = {
                "p_foreground_enrichment": p_fg / max(p_bg, 1e-8),
                "s_boundary_specificity_enrichment": s_boundary
                / max(s_interior, 1e-8),
                "p_foreground_mean": p_fg,
                "p_background_mean": p_bg,
                "s_boundary_specificity_mean": s_boundary,
                "s_interior_specificity_mean": s_interior,
            }
            maps[f"res{level}_p_energy"] = p_map.cpu().numpy()
            maps[f"res{level}_s_energy"] = s_map.cpu().numpy()
            maps[f"res{level}_s_specificity"] = specificity_map.cpu().numpy()

    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    np.savez_compressed(
        os.path.join(output_dir, f"{args.case_id}_z{center:04d}_maps.npz"),
        gt=gt[center],
        **maps,
    )
    with open(
        os.path.join(output_dir, f"{args.case_id}_z{center:04d}_metrics.json"),
        "w",
        encoding="utf-8",
    ) as file_handle:
        json.dump(
            {
                "case_id": args.case_id,
                "z": center,
                "checkpoint": os.path.abspath(args.checkpoint),
                "metrics": metrics,
            },
            file_handle,
            indent=2,
        )

    fig, axes = plt.subplots(4, 4, figsize=(16, 16))
    for row, level in enumerate([2, 3, 4, 5]):
        axes[row, 0].imshow(raw[center], cmap="gray")
        axes[row, 0].contour(gt[center] > 0, levels=[0.5], colors="lime")
        axes[row, 0].set_title(f"res{level}: image/GT")
        axes[row, 1].imshow(maps[f"res{level}_p_energy"], cmap="magma")
        axes[row, 1].set_title("P energy")
        axes[row, 2].imshow(maps[f"res{level}_s_energy"], cmap="magma")
        axes[row, 2].set_title("S energy")
        axes[row, 3].imshow(
            maps[f"res{level}_s_specificity"], cmap="viridis", vmin=0, vmax=1
        )
        axes[row, 3].set_title("S specificity")
        for axis in axes[row]:
            axis.axis("off")
    plt.tight_layout()
    figure_path = os.path.join(output_dir, f"{args.case_id}_z{center:04d}.png")
    plt.savefig(figure_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(json.dumps(metrics, indent=2))
    print(f"saved={figure_path}")


if __name__ == "__main__":
    main()
