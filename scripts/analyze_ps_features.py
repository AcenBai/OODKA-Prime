#!/usr/bin/env python3
"""Visualize expert/student raw and decomposed P/S feature roles."""

from __future__ import annotations

import argparse
import json
import os
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/oodka_ps_mpl")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import SimpleITK as sitk
import torch
import torch.nn.functional as F

from oodka.config import TrainConfig
from oodka.data.slice_dataset import FullSliceBlockDataset
from oodka.models.feature_extraction import (
    extract_biomedparse_backbone_features_2p5d,
    extract_nnunet_features,
)
from oodka.models.prompts import build_text_prompts_for_dataset
from oodka.train.model_builder import (
    build_fusion_modules,
    build_prompt_features,
    load_frozen_backbones,
)
from oodka.train.forward import _run_pixel_decoder
from oodka.utils.io_utils import find_raw_image_files


LEVELS = (2, 3, 4, 5)
CLASS_COLORS = ("#00ff3b", "#ff2d2d", "#00c8ff", "#ffd400", "#d12dff", "#ff7a00", "#ffffff")


def _boundary(mask: torch.Tensor) -> torch.Tensor:
    value = mask.float().unsqueeze(0).unsqueeze(0)
    dilation = F.max_pool2d(value, 3, 1, 1)
    erosion = -F.max_pool2d(-value, 3, 1, 1)
    return (dilation - erosion).squeeze() > 0


def _safe_mean(value: torch.Tensor, mask: torch.Tensor) -> float:
    return float(value[mask].mean().item()) if mask.any() else 0.0


def _select_slice(gt: np.ndarray, mode: str, slice_index: int | None) -> int:
    if slice_index is not None:
        if not 0 <= slice_index < gt.shape[0]:
            raise ValueError(f"slice_index={slice_index} outside [0,{gt.shape[0]})")
        return int(slice_index)
    foreground_area = (gt > 0).reshape(gt.shape[0], -1).sum(axis=1)
    if mode == "largest_foreground":
        return int(foreground_area.argmax())
    all_classes = np.array(
        [all(np.any(gt[z] == class_id) for class_id in range(1, 8)) for z in range(gt.shape[0])]
    )
    if not all_classes.any():
        raise RuntimeError("No slice contains all seven foreground classes")
    candidates = np.flatnonzero(all_classes)
    return int(candidates[np.argmax(foreground_area[candidates])])


def _resize_scalar_map(value: torch.Tensor, output_hw: tuple[int, int]) -> torch.Tensor:
    return F.interpolate(
        value[None, None].float(), size=output_hw, mode="bilinear", align_corners=False
    )[0, 0]


def _energy_map(feature: torch.Tensor, z_index: int, output_hw: tuple[int, int]) -> torch.Tensor:
    if feature.ndim != 5:
        raise ValueError(f"feature must be [B,C,Z,H,W], got {feature.shape}")
    energy = feature.float().square().sum(dim=1).add(1e-8).sqrt()[0, z_index]
    return _resize_scalar_map(energy, output_hw)


def _decoder_energy_map(
    feature: torch.Tensor, z_index: int, output_hw: tuple[int, int]
) -> torch.Tensor:
    """Channel-L2 energy for pixel-decoder output shaped [B*Z,C,H,W]."""
    if feature.ndim != 4:
        raise ValueError(f"decoder feature must be [B*Z,C,H,W], got {feature.shape}")
    energy = feature[z_index].float().square().sum(dim=0).add(1e-8).sqrt()
    return _resize_scalar_map(energy, output_hw)


def _pearson(left: torch.Tensor, right: torch.Tensor) -> float:
    x = left.float().flatten()
    y = right.float().flatten()
    x = x - x.mean()
    y = y - y.mean()
    denominator = x.norm() * y.norm()
    return float((x * y).sum().div(denominator.clamp_min(1e-8)).item())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Full fusion checkpoint containing expert adapters and student decomposers",
    )
    parser.add_argument("--case_id", default="heart_1004")
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--block_z", type=int, default=6)
    parser.add_argument("--slice_index", type=int, default=None)
    parser.add_argument(
        "--selection",
        choices=("largest_foreground", "all_classes"),
        default="largest_foreground",
    )
    parser.add_argument(
        "--output_dir",
        default="outputs/oodka_ot_experiments/interpretability/ps_features",
    )
    args = parser.parse_args()

    cfg = TrainConfig(device=args.device, block_z=args.block_z, num_workers=0)
    cfg.resolve_paths()
    images_dir = cfg.imagesTr_dir if args.split == "val" else cfg.imagesTs_dir
    labels_dir = cfg.labelsTr_dir if args.split == "val" else cfg.labelsTs_dir
    with open(cfg.dataset_json_path, encoding="utf-8") as file_handle:
        dataset_json = json.load(file_handle)
    ending = dataset_json.get("file_ending", ".nii.gz")
    label_names = {
        int(class_id): str(name)
        for name, class_id in dataset_json.get("labels", {}).items()
        if int(class_id) > 0
    }

    image_files = find_raw_image_files(images_dir, args.case_id, ending)
    if not image_files:
        raise FileNotFoundError(args.case_id)
    label_path = os.path.join(labels_dir, args.case_id + ending)
    raw = np.asarray(sitk.GetArrayFromImage(sitk.ReadImage(image_files[0])))
    gt = np.asarray(sitk.GetArrayFromImage(sitk.ReadImage(label_path)))
    center = _select_slice(gt, args.selection, args.slice_index)

    dataset = FullSliceBlockDataset(
        [args.case_id],
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

    device = torch.device(args.device)
    model_nnunet, model_biomedparse = load_frozen_backbones(
        cfg.nnunet_model_dir, cfg.fold, device
    )
    prompts, prompt_to_class_id = build_text_prompts_for_dataset(dataset_name=cfg.dataset_name)
    prompt_features = build_prompt_features(model_biomedparse, prompts, device)
    modules = build_fusion_modules(
        model_nnunet,
        model_biomedparse,
        len(prompts),
        device,
        text_dim=int(prompt_features["class_emb"].shape[-1]),
    )
    checkpoint = torch.load(args.checkpoint, map_location=device)
    required = [
        *(f"ae_enc{level}_to_res{level}" for level in LEVELS),
        *(f"dis_b_res{level}" for level in LEVELS),
    ]
    missing = [name for name in required if name not in checkpoint]
    if missing:
        raise KeyError(
            "Expert/student comparison requires a full fusion checkpoint; "
            f"missing {missing}"
        )
    for name, module in modules.items():
        if name in checkpoint:
            module.load_state_dict(checkpoint[name])
        module.eval()

    output_hw = tuple(int(value) for value in gt.shape[-2:])
    with torch.no_grad():
        expert_raw, _deepest = extract_nnunet_features(
            model_nnunet, nn_input, device
        )
        embeds, student_raw = extract_biomedparse_backbone_features_2p5d(
            model_biomedparse, bp.to(device), device
        )
        maps = {}
        metrics = {}
        student_branches = {}
        gt_center = torch.from_numpy(gt[center] > 0).to(device)
        gt_boundary = _boundary(gt_center)
        gt_interior = gt_center & ~gt_boundary
        gt_background = ~gt_center

        for level in LEVELS:
            student_feature = student_raw[f"res{level}"]
            expert_feature_native = expert_raw[f"enc{level}"]
            expert_feature = expert_feature_native
            if expert_feature.shape[-3:] != student_feature.shape[-3:]:
                expert_feature = F.interpolate(
                    expert_feature,
                    size=student_feature.shape[-3:],
                    mode="trilinear",
                    align_corners=False,
                )
            expert_p, expert_s, _p_rec, _s_rec = modules[
                f"ae_enc{level}_to_res{level}"
            ](expert_feature)
            student_p, student_s = modules[f"dis_b_res{level}"](student_feature)
            student_branches[f"Zb{level}_p"] = student_p
            student_branches[f"Zb{level}_s"] = student_s

            level_maps = {
                "expert_raw": _energy_map(expert_feature_native, center_local, output_hw),
                "student_raw": _energy_map(student_feature, center_local, output_hw),
                "expert_p": _energy_map(expert_p, center_local, output_hw),
                "expert_s": _energy_map(expert_s, center_local, output_hw),
                "student_p": _energy_map(student_p, center_local, output_hw),
                "student_s": _energy_map(student_s, center_local, output_hw),
            }
            level_maps["student_s_share"] = level_maps["student_s"] / (
                level_maps["student_p"] + level_maps["student_s"] + 1e-8
            )
            maps.update(
                {f"res{level}_{name}": value.cpu().numpy() for name, value in level_maps.items()}
            )

            p_fg = _safe_mean(level_maps["student_p"], gt_center)
            p_bg = _safe_mean(level_maps["student_p"], gt_background)
            s_boundary = _safe_mean(level_maps["student_s_share"], gt_boundary)
            s_interior = _safe_mean(level_maps["student_s_share"], gt_interior)
            metrics[f"res{level}"] = {
                "expert_student_raw_energy_correlation": _pearson(
                    level_maps["expert_raw"], level_maps["student_raw"]
                ),
                "expert_student_p_energy_correlation": _pearson(
                    level_maps["expert_p"], level_maps["student_p"]
                ),
                "expert_student_s_energy_correlation": _pearson(
                    level_maps["expert_s"], level_maps["student_s"]
                ),
                "student_p_foreground_enrichment": p_fg / max(p_bg, 1e-8),
                "student_s_boundary_share_enrichment": s_boundary / max(s_interior, 1e-8),
            }

        # Run the two student pixel-decoder branches exactly as training does.
        # Predictor-scale order is mask/res2 plus [res5,res4,res3].
        embeds_base = dict(embeds)
        for level in LEVELS:
            embeds_base.pop(f"res{level}", None)
        mask_p, multi_scale_p = _run_pixel_decoder(
            model_biomedparse,
            embeds_base,
            student_branches,
            "p",
            B=1,
            Dm=bp.shape[1],
        )
        mask_s, multi_scale_s = _run_pixel_decoder(
            model_biomedparse,
            embeds_base,
            student_branches,
            "s",
            B=1,
            Dm=bp.shape[1],
        )
        decoder_p = {
            2: mask_p,
            3: multi_scale_p[2],
            4: multi_scale_p[1],
            5: multi_scale_p[0],
        }
        decoder_s = {
            2: mask_s,
            3: multi_scale_s[2],
            4: multi_scale_s[1],
            5: multi_scale_s[0],
        }
        route = modules["beta_router"](
            prompt_features["class_emb"].detach(), batch_size=1, sample=False
        )
        prompt_gate_means = route["mean"].detach()
        prompt_fusion_maps = {}
        for prompt_index in range(len(prompts)):
            class_id = int(prompt_to_class_id[prompt_index])
            class_maps = {}
            for level_index, level in enumerate(LEVELS):
                gate_value = prompt_gate_means[prompt_index, level_index]
                p_feature = decoder_p[level]
                s_feature = decoder_s[level]
                fused_feature = gate_value * p_feature + (1.0 - gate_value) * s_feature
                class_maps[level] = {
                    "p": _decoder_energy_map(p_feature, center_local, output_hw).cpu().numpy(),
                    "s": _decoder_energy_map(s_feature, center_local, output_hw).cpu().numpy(),
                    "fused": _decoder_energy_map(
                        fused_feature, center_local, output_hw
                    ).cpu().numpy(),
                    "gate_p": float(gate_value.item()),
                }
            prompt_fusion_maps[class_id] = class_maps

    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    stem = f"{args.case_id}_z{center:04d}_{args.selection}"
    np.savez_compressed(
        os.path.join(output_dir, stem + "_maps.npz"), gt=gt[center], **maps
    )
    report = {
        "case_id": args.case_id,
        "z": center,
        "selection": args.selection,
        "labels_present": sorted(int(value) for value in np.unique(gt[center]) if value > 0),
        "checkpoint": os.path.abspath(args.checkpoint),
        "metrics": metrics,
    }
    with open(os.path.join(output_dir, stem + "_metrics.json"), "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    columns = (
        ("image_gt", "Image / class-colored GT"),
        ("expert_raw", "Expert raw energy"),
        ("student_raw", "Student raw energy"),
        ("expert_p", "Expert P energy"),
        ("expert_s", "Expert S energy"),
        ("student_p", "Student P energy"),
        ("student_s", "Student S energy"),
        ("student_s_share", "Student S energy share"),
    )
    fig, axes = plt.subplots(4, len(columns), figsize=(4 * len(columns), 16))
    for row, level in enumerate(LEVELS):
        for column, (name, title) in enumerate(columns):
            axis = axes[row, column]
            if name == "image_gt":
                axis.imshow(raw[center], cmap="gray")
                for class_id, color in zip(range(1, 8), CLASS_COLORS):
                    if np.any(gt[center] == class_id):
                        axis.contour(gt[center] == class_id, levels=[0.5], colors=[color], linewidths=1.2)
                if row == 0:
                    handles = [
                        Line2D([0], [0], color=color, lw=2, label=label_names.get(class_id, str(class_id)))
                        for class_id, color in zip(range(1, 8), CLASS_COLORS)
                        if np.any(gt[center] == class_id)
                    ]
                    axis.legend(handles=handles, fontsize=7, loc="lower left", framealpha=0.75)
            else:
                kwargs = {"vmin": 0.0, "vmax": 1.0} if name == "student_s_share" else {}
                axis.imshow(maps[f"res{level}_{name}"], cmap="viridis" if kwargs else "magma", **kwargs)
            axis.set_title(f"res{level}: {title}" if column == 0 else title)
            axis.axis("off")
    fig.suptitle(
        f"{args.case_id}, z={center}, selection={args.selection}; energy panels use independent color scales",
        fontsize=14,
    )
    plt.tight_layout(rect=(0, 0, 1, 0.975))
    figure_path = os.path.join(output_dir, stem + ".png")
    plt.savefig(figure_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    prompt_report = {
        "case_id": args.case_id,
        "z": center,
        "selection": args.selection,
        "scale_order": ["mask_features/res2", "multi_scale/res3", "multi_scale/res4", "multi_scale/res5"],
        "classes": {},
    }
    scale_labels = {
        2: "mask_features / res2 (128x128)",
        3: "multi_scale / res3 (64x64)",
        4: "multi_scale / res4 (32x32)",
        5: "multi_scale / res5 (16x16)",
    }
    for class_id in sorted(prompt_fusion_maps):
        class_name = label_names.get(class_id, f"class_{class_id}")
        safe_name = "".join(
            character if character.isalnum() else "_" for character in class_name
        ).strip("_")
        class_maps = prompt_fusion_maps[class_id]
        class_stem = f"{args.case_id}_z{center:04d}_class{class_id:02d}_{safe_name}_prompt_fusion"
        np.savez_compressed(
            os.path.join(output_dir, class_stem + "_maps.npz"),
            gt=(gt[center] == class_id).astype(np.uint8),
            **{
                f"res{level}_{branch}": class_maps[level][branch]
                for level in LEVELS
                for branch in ("p", "s", "fused")
            },
        )
        figure, class_axes = plt.subplots(4, 4, figsize=(18, 18))
        for row, level in enumerate(LEVELS):
            gt_axis, p_axis, s_axis, fused_axis = class_axes[row]
            class_mask = gt[center] == class_id
            gt_axis.imshow(raw[center], cmap="gray")
            gt_axis.contourf(
                class_mask,
                levels=[0.5, 1.5],
                colors=[CLASS_COLORS[class_id - 1]],
                alpha=0.28,
            )
            gt_axis.contour(
                class_mask,
                levels=[0.5],
                colors=[CLASS_COLORS[class_id - 1]],
                linewidths=1.6,
            )
            gt_axis.text(
                0.02,
                0.98,
                scale_labels[level],
                transform=gt_axis.transAxes,
                va="top",
                ha="left",
                color="white",
                fontsize=10,
                bbox={"facecolor": "black", "alpha": 0.55, "pad": 3},
            )
            p_map = class_maps[level]["p"]
            s_map = class_maps[level]["s"]
            fused_map = class_maps[level]["fused"]
            common_max = max(
                float(np.percentile(p_map, 99.5)),
                float(np.percentile(s_map, 99.5)),
                float(np.percentile(fused_map, 99.5)),
                1e-8,
            )
            p_axis.imshow(p_map, cmap="magma", vmin=0.0, vmax=common_max)
            s_axis.imshow(s_map, cmap="magma", vmin=0.0, vmax=common_max)
            fused_axis.imshow(fused_map, cmap="magma", vmin=0.0, vmax=common_max)
            if row == 0:
                gt_axis.set_title(f"GT: {class_name}")
                p_axis.set_title("P decoder feature energy")
                s_axis.set_title("S decoder feature energy")
                fused_axis.set_title("Prompt-gated fused energy")
            fused_axis.text(
                0.02,
                0.98,
                f"P gate={class_maps[level]['gate_p']:.3f}",
                transform=fused_axis.transAxes,
                va="top",
                ha="left",
                color="white",
                fontsize=10,
                bbox={"facecolor": "black", "alpha": 0.55, "pad": 3},
            )
            for axis in (gt_axis, p_axis, s_axis, fused_axis):
                axis.axis("off")
        figure.suptitle(
            f"{args.case_id}, z={center}: class-specific pixel-decoder fusion for {class_name}",
            fontsize=16,
        )
        figure.tight_layout(rect=(0, 0, 1, 0.975))
        class_figure_path = os.path.join(output_dir, class_stem + ".png")
        figure.savefig(class_figure_path, dpi=170, bbox_inches="tight")
        plt.close(figure)
        prompt_report["classes"][str(class_id)] = {
            "name": class_name,
            "figure": class_figure_path,
            "gate_p": {
                f"res{level}": class_maps[level]["gate_p"] for level in LEVELS
            },
        }
    with open(
        os.path.join(output_dir, stem + "_prompt_fusion.json"),
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(prompt_report, handle, indent=2)
    print(json.dumps(report, indent=2))
    print(f"saved={figure_path}")
    print(f"saved_prompt_figures={len(prompt_fusion_maps)}")


if __name__ == "__main__":
    main()
