#!/usr/bin/env python3
"""Mechanism-v3 style audit for the two-pass LGE ROI model.

The script visualizes Pass 1 in full-image coordinates, Pass 2 both inside
the resized ROI and restored to full-image coordinates, and the final hard
spatial switch.  It intentionally uses the exact checkpoint prompts and ROI
construction used by deployment inference.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/oodka_lge_roi_mechanism_mpl")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle
import numpy as np
import SimpleITK as sitk
import torch
import torch.nn.functional as F

from oodka.config import EvalConfig
from oodka.data.aligned_preprocessing import AlignedBiomedParsePreprocessor
from oodka.data.lge_roi import (
    ROICoordinates,
    ROIGenerator,
    hard_switch_foreground_logits,
    restore_roi_logits,
)
from oodka.data.slice_dataset import make_biomedparse_block
from oodka.models.feature_extraction import (
    extract_biomedparse_backbone_features_2p5d,
)
from oodka.models.prompts import (
    MYOPS_LGE_ROI_ANATOMY_PROMPTS,
    MYOPS_LGE_ROI_V2_REFINEMENT_PROMPTS,
)
from oodka.train.forward import _predict_all_prompt_logits, _run_pixel_decoder
from oodka.train.model_builder import (
    build_fusion_modules,
    build_prompt_features,
    load_frozen_biomedparse,
)
from oodka.utils.io_utils import find_raw_image_files


LEVELS = (2, 3, 4, 5)
ANATOMY_NAMES = ("LV", "RV", "total_myo")
REFINEMENT_NAMES = ("LV", "RV", "normal_myo", "scar_edema")
FINAL_NAMES = ("background", "LV", "RV", "normal_myo", "scar_edema")
SEGMENTATION_COLORS = (
    "#000000", "#ef4444", "#38bdf8", "#22c55e", "#f59e0b"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resize(value: torch.Tensor, size: tuple[int, int]) -> np.ndarray:
    if value.ndim == 2:
        value = value[None, None]
    elif value.ndim == 3:
        value = value[None]
    result = F.interpolate(
        value.float(), size=size, mode="bilinear", align_corners=False
    )
    return result[0, 0].detach().cpu().numpy()


def _rms(value: torch.Tensor, size: tuple[int, int]) -> np.ndarray:
    return _resize(value.float().square().mean(dim=1, keepdim=True).sqrt(), size)


def _restore_map(
    value: np.ndarray,
    roi: ROICoordinates,
    full_size: tuple[int, int],
) -> np.ndarray:
    tensor = torch.from_numpy(value)[None, None].float()
    resized = F.interpolate(
        tensor,
        size=(roi.height, roi.width),
        mode="bilinear",
        align_corners=False,
    )[0, 0].numpy()
    output = np.full(full_size, np.nan, dtype=np.float32)
    output[roi.y0 : roi.y1, roi.x0 : roi.x1] = resized
    return output


def _remap_final_gt(gt: np.ndarray) -> np.ndarray:
    output = np.zeros_like(gt, dtype=np.int16)
    output[gt == 3] = 1
    output[gt == 5] = 2
    output[gt == 4] = 3
    output[np.isin(gt, (1, 2))] = 4
    return output


def _run_pass(
    block: torch.Tensor,
    prompts: dict[str, str],
    *,
    model: torch.nn.Module,
    modules: dict[str, torch.nn.Module],
    device: torch.device,
    output_size: tuple[int, int],
) -> dict:
    """Return logits plus finest-level P/S/gate/fused mechanism maps."""
    prompt_features = build_prompt_features(model, prompts, device)
    block = block.to(device)
    batch_size, slices = block.shape[:2]
    with torch.no_grad():
        embeds, student_raw = extract_biomedparse_backbone_features_2p5d(
            model, block, device, res_names=("res2", "res3", "res4", "res5")
        )
        features = {}
        for level in LEVELS:
            p_value, s_value = modules[f"dis_b_res{level}"](
                student_raw[f"res{level}"]
            )
            features[f"Zb{level}_p"] = p_value
            features[f"Zb{level}_s"] = s_value
        embeds_base = dict(embeds)
        for level in LEVELS:
            embeds_base.pop(f"res{level}", None)
        mask_p, multi_p = _run_pixel_decoder(
            model, embeds_base, features, "p", B=batch_size, Dm=slices
        )
        mask_s, multi_s = _run_pixel_decoder(
            model, embeds_base, features, "s", B=batch_size, Dm=slices
        )
        route = modules["beta_router"](
            prompt_features["class_emb"].detach(),
            spatial_size=mask_p.shape[-2:],
            batch_size=batch_size,
            sample=False,
        )
        logits = _predict_all_prompt_logits(
            sem_seg_head=model.sem_seg_head,
            mask_features_p=mask_p,
            mask_features_s=mask_s,
            ms_p=multi_p,
            ms_s=multi_s,
            gate=route["gate"],
            prompt_features=prompt_features,
            B=batch_size,
            Z=slices,
            P=len(prompts),
            output_shape=(slices, *output_size),
        )
        p_energy = _rms(mask_p, output_size)
        s_energy = _rms(mask_s, output_size)
        prompt_maps = []
        for index in range(len(prompts)):
            gate = route["gate"][0, index]
            fused = gate[None, None] * mask_p + (1.0 - gate[None, None]) * mask_s
            prompt_maps.append(
                {
                    "gate": _resize(gate, output_size),
                    "fused": _rms(fused, output_size),
                    "probability": torch.sigmoid(logits[0, index, 0])
                    .float().cpu().numpy(),
                }
            )
    return {
        "logits": logits,
        "p_energy": p_energy,
        "s_energy": s_energy,
        "prompt_maps": prompt_maps,
        "gate_mean": route["mean"].detach().float().cpu().numpy(),
        "gate_concentration": route["concentration"].detach().float().cpu().numpy(),
    }


def _draw_roi(axis, roi: ROICoordinates, color: str = "#facc15") -> None:
    axis.add_patch(
        Rectangle(
            (roi.x0, roi.y0), roi.width, roi.height,
            fill=False, edgecolor=color, linewidth=2.0,
        )
    )


def _show_heat(
    axis,
    image: np.ndarray,
    value: np.ndarray,
    title: str,
    *,
    cmap: str = "magma",
    vmin: float | None = None,
    vmax: float | None = None,
    roi: ROICoordinates | None = None,
) -> None:
    axis.imshow(image, cmap="gray")
    masked = np.ma.masked_invalid(value)
    artist = axis.imshow(masked, cmap=cmap, alpha=0.72, vmin=vmin, vmax=vmax)
    if roi is not None:
        _draw_roi(axis, roi)
    axis.set_title(title, fontsize=10)
    axis.axis("off")
    plt.colorbar(artist, ax=axis, fraction=0.045, pad=0.02)


def _plot_mechanism(
    output_path: Path,
    *,
    image: np.ndarray,
    names: tuple[str, ...],
    data: dict,
    title: str,
    roi: ROICoordinates | None = None,
    restore_roi: bool = False,
) -> None:
    rows = len(names)
    figure, axes = plt.subplots(
        rows, 5, figsize=(18, 3.35 * rows), constrained_layout=True
    )
    axes = np.atleast_2d(axes)
    p_values = data["p_energy"]
    s_values = data["s_energy"]
    fused_values = [entry["fused"] for entry in data["prompt_maps"]]
    if restore_roi:
        assert roi is not None
        full_size = image.shape
        p_values = _restore_map(p_values, roi, full_size)
        s_values = _restore_map(s_values, roi, full_size)
        fused_values = [_restore_map(value, roi, full_size) for value in fused_values]
    energy_values = [p_values, s_values, *fused_values]
    finite = np.concatenate([v[np.isfinite(v)] for v in energy_values])
    vmax = float(np.percentile(finite, 99.0)) if finite.size else 1.0
    for index, name in enumerate(names):
        prompt = data["prompt_maps"][index]
        gate = prompt["gate"]
        probability = prompt["probability"]
        if restore_roi:
            gate = _restore_map(gate, roi, image.shape)
            probability = _restore_map(probability, roi, image.shape)
        _show_heat(
            axes[index, 0], image, p_values, f"{name} · P decoder energy",
            vmin=0.0, vmax=vmax, roi=roi,
        )
        _show_heat(
            axes[index, 1], image, s_values, "S decoder energy",
            vmin=0.0, vmax=vmax, roi=roi,
        )
        _show_heat(
            axes[index, 2], image, gate,
            f"P gate · mean={data['gate_mean'][index].mean():.3f}",
            cmap="coolwarm", vmin=0.0, vmax=1.0, roi=roi,
        )
        _show_heat(
            axes[index, 3], image, fused_values[index], "Prompt-gated fusion",
            vmin=0.0, vmax=vmax, roi=roi,
        )
        _show_heat(
            axes[index, 4], image, probability, "Sigmoid response",
            cmap="viridis", vmin=0.0, vmax=1.0, roi=roi,
        )
    figure.suptitle(title, fontsize=15, fontweight="bold")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=190, bbox_inches="tight")
    plt.close(figure)


def _segmentation_overlay(
    axis, image: np.ndarray, labels: np.ndarray, title: str,
    *, roi: ROICoordinates | None = None,
) -> None:
    axis.imshow(image, cmap="gray")
    masked = np.ma.masked_where(labels == 0, labels)
    axis.imshow(
        masked,
        cmap=ListedColormap(SEGMENTATION_COLORS),
        vmin=0,
        vmax=4,
        alpha=0.65,
        interpolation="nearest",
    )
    if roi is not None:
        _draw_roi(axis, roi)
    axis.set_title(title, fontsize=10)
    axis.axis("off")


def _plot_global_summary(
    output_path: Path,
    *,
    image: np.ndarray,
    gt: np.ndarray,
    roi: ROICoordinates,
    roi_image: np.ndarray,
    anatomy: dict,
    restored_refinement: torch.Tensor,
    final_prediction: np.ndarray,
) -> None:
    anatomy_prob = torch.sigmoid(anatomy["logits"])[0, :, 0].cpu().numpy()
    refinement_prob = torch.sigmoid(restored_refinement)[0, :, 0].cpu().numpy()
    figure, axes = plt.subplots(3, 5, figsize=(18, 10), constrained_layout=True)

    axes[0, 0].imshow(image, cmap="gray")
    axes[0, 0].set_title("Full LGE input")
    axes[0, 0].axis("off")
    _segmentation_overlay(axes[0, 1], image, gt, "GT · full coordinates")
    for index, name in enumerate(ANATOMY_NAMES):
        _show_heat(
            axes[0, index + 2], image, anatomy_prob[index],
            f"Pass 1 · {name}", cmap="viridis", vmin=0.0, vmax=1.0,
            roi=roi if name == "total_myo" else None,
        )

    axes[1, 0].imshow(image, cmap="gray")
    _draw_roi(axes[1, 0], roi)
    axes[1, 0].set_title("Predicted ROI on full image")
    axes[1, 0].axis("off")
    for index, name in enumerate(REFINEMENT_NAMES):
        column = index + 1
        _show_heat(
            axes[1, column], image, refinement_prob[index],
            f"Pass 2 restored · {name}", cmap="viridis",
            vmin=0.0, vmax=1.0, roi=roi,
        )

    axes[2, 0].imshow(roi_image, cmap="gray")
    axes[2, 0].set_title("Pass 2 ROI input · resized")
    axes[2, 0].axis("off")
    pass1_labels = torch.cat(
        [torch.zeros_like(anatomy["logits"][:, :1]), anatomy["logits"]], dim=1
    ).argmax(dim=1)[0, 0].cpu().numpy()
    pass1_visible = np.where(pass1_labels == 3, 0, pass1_labels)
    _segmentation_overlay(
        axes[2, 1], image, pass1_visible, "Pass 1 deployable LV/RV", roi=roi
    )
    pass2_labels = torch.cat(
        [torch.zeros_like(restored_refinement[:, :1]), restored_refinement], dim=1
    ).argmax(dim=1)[0, 0].cpu().numpy()
    _segmentation_overlay(
        axes[2, 2], image, pass2_labels, "Pass 2 restored argmax", roi=roi
    )
    _segmentation_overlay(
        axes[2, 3], image, final_prediction, "Final hard spatial switch", roi=roi
    )
    error = np.where(final_prediction != gt, 1.0, np.nan).astype(np.float32)
    _show_heat(
        axes[2, 4], image, error, "Final disagreement with GT",
        cmap="Reds", vmin=0.0, vmax=1.0, roi=roi,
    )
    figure.suptitle(
        "LGE ROI two-pass Mechanism-v3 · before and after ROI in global coordinates",
        fontsize=15,
        fontweight="bold",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--case_id", default="myo_3042")
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--slice_index", type=int, default=1)
    parser.add_argument("--device", default="cuda:2")
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint).resolve()
    output_dir = Path(args.output_dir).resolve()
    device = torch.device(args.device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if checkpoint.get("format") != "oodka_lge_roi_v2":
        raise ValueError("Expected an oodka_lge_roi_v2 checkpoint")
    saved = checkpoint["config"]
    cfg = EvalConfig(
        dataset_name="Dataset011_MYO_LGE_BC_OOD",
        fold=int(saved.get("fold", 0)),
        block_z=1,
        batch_size=1,
        image_size=int(saved.get("image_size", 256)),
        norm_mode="mri",
        pseudo_rgb_mode="center_repeat",
        device=args.device,
        split=args.split,
        out_dir=str(output_dir),
        use_aligned_biomedparse_preprocessing=True,
    )
    cfg.resolve_paths()
    with open(cfg.dataset_json_path, encoding="utf-8") as handle:
        dataset_json = json.load(handle)
    ending = dataset_json.get("file_ending", ".nii.gz")
    images_dir = cfg.imagesTr_dir if args.split == "val" else cfg.imagesTs_dir
    labels_dir = cfg.labelsTr_dir if args.split == "val" else cfg.labelsTs_dir
    image_files = find_raw_image_files(images_dir, args.case_id, ending)
    label_path = os.path.join(labels_dir, args.case_id + ending)
    if not image_files or not os.path.isfile(label_path):
        raise FileNotFoundError(args.case_id)

    preprocessor = AlignedBiomedParsePreprocessor(
        plans_path=cfg.plans_path,
        dataset_json_path=cfg.dataset_json_path,
        configuration_name="2d",
        low_percentile=float(saved.get("low_percentile", 1.0)),
        high_percentile=float(saved.get("high_percentile", 99.0)),
    )
    bp_u8, aligned_seg, _properties = preprocessor.run_case(
        image_files, label_path, modality=0
    )
    z_index = int(args.slice_index)
    if not 0 <= z_index < bp_u8.shape[0]:
        raise IndexError(f"slice_index={z_index}, depth={bp_u8.shape[0]}")
    image_size = (cfg.image_size, cfg.image_size)
    full_block = make_biomedparse_block(
        bp_u8, [z_index], cfg.image_size, pseudo_rgb_mode="center_repeat"
    )[None]
    image = full_block[0, 0, 0].numpy()
    gt_raw = F.interpolate(
        torch.from_numpy(aligned_seg[z_index])[None, None].float(),
        size=image_size,
        mode="nearest",
    )[0, 0].numpy().astype(np.int16)
    gt = _remap_final_gt(gt_raw)

    model = load_frozen_biomedparse(device)
    anatomy_features = build_prompt_features(model, MYOPS_LGE_ROI_ANATOMY_PROMPTS, device)
    modules = build_fusion_modules(
        None,
        model,
        len(MYOPS_LGE_ROI_ANATOMY_PROMPTS),
        device,
        text_dim=int(anatomy_features["class_emb"].shape[-1]),
        route_prior_p_mean=float(saved.get("route_prior_p_mean", 0.7)),
        route_prior_concentration=float(saved.get("route_prior_concentration", 10.0)),
        route_spatial_basis_grid_size=int(saved.get("route_spatial_basis_grid_size", 8)),
        route_spatial_basis_sigma=float(saved.get("route_spatial_basis_sigma", 0.0)),
    )
    for name, module in modules.items():
        module.load_state_dict(checkpoint[name])
        module.eval()

    anatomy = _run_pass(
        full_block,
        MYOPS_LGE_ROI_ANATOMY_PROMPTS,
        model=model,
        modules=modules,
        device=device,
        output_size=image_size,
    )
    roi_generator = ROIGenerator(
        threshold=float(saved.get("roi_threshold", 0.3)),
        expand=float(saved.get("roi_expand", 1.25)),
        fallback=str(saved.get("roi_fallback", "full")),
    )
    roi = roi_generator.from_probability(
        torch.sigmoid(anatomy["logits"][0, 2, 0]).detach()
    )
    crop = full_block[0, 0, :, roi.y0 : roi.y1, roi.x0 : roi.x1]
    roi_image_tensor = F.interpolate(
        crop[None], size=image_size, mode="bilinear", align_corners=False
    )[0]
    roi_block = roi_image_tensor[None, None]
    refinement = _run_pass(
        roi_block,
        MYOPS_LGE_ROI_V2_REFINEMENT_PROMPTS,
        model=model,
        modules=modules,
        device=device,
        output_size=image_size,
    )
    restored = restore_roi_logits(refinement["logits"], [roi], image_size)
    foreground = hard_switch_foreground_logits(anatomy["logits"], restored, [roi])
    final_prediction = torch.cat(
        [torch.zeros_like(foreground[:, :1]), foreground], dim=1
    ).argmax(dim=1)[0, 0].cpu().numpy().astype(np.int16)

    output_dir.mkdir(parents=True, exist_ok=True)
    _plot_mechanism(
        output_dir / "01_pass1_full_global_mechanism.png",
        image=image,
        names=ANATOMY_NAMES,
        data=anatomy,
        title="Pass 1 · full-image global anatomy and ROI localization",
    )
    _plot_mechanism(
        output_dir / "02_pass2_roi_local_mechanism.png",
        image=roi_image_tensor[0].cpu().numpy(),
        names=REFINEMENT_NAMES,
        data=refinement,
        title="Pass 2 · ROI-local refinement before coordinate restoration",
    )
    _plot_mechanism(
        output_dir / "03_pass2_restored_global_mechanism.png",
        image=image,
        names=REFINEMENT_NAMES,
        data=refinement,
        title="Pass 2 · ROI refinement restored to full-image global coordinates",
        roi=roi,
        restore_roi=True,
    )
    _plot_global_summary(
        output_dir / "04_before_after_global_summary.png",
        image=image,
        gt=gt,
        roi=roi,
        roi_image=roi_image_tensor[0].cpu().numpy(),
        anatomy=anatomy,
        restored_refinement=restored,
        final_prediction=final_prediction,
    )

    manifest = {
        "schema": "oodka-lge-roi-mechanism-v3",
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "case_id": args.case_id,
        "split": args.split,
        "slice_index": z_index,
        "labels_present_raw": sorted(int(v) for v in np.unique(gt_raw)),
        "roi": {
            "x0": roi.x0, "y0": roi.y0, "x1": roi.x1, "y1": roi.y1,
            "width": roi.width, "height": roi.height,
            "area_fraction": roi.width * roi.height / float(cfg.image_size ** 2),
            "fallback": roi.fallback,
            "threshold": float(saved.get("roi_threshold", 0.3)),
            "expand": float(saved.get("roi_expand", 1.25)),
        },
        "anatomy_gate_statistics": {
            name: {
                "mean": float(anatomy["gate_mean"][index].mean()),
                "min": float(anatomy["gate_mean"][index].min()),
                "max": float(anatomy["gate_mean"][index].max()),
            }
            for index, name in enumerate(ANATOMY_NAMES)
        },
        "refinement_gate_statistics": {
            name: {
                "mean": float(refinement["gate_mean"][index].mean()),
                "min": float(refinement["gate_mean"][index].min()),
                "max": float(refinement["gate_mean"][index].max()),
            }
            for index, name in enumerate(REFINEMENT_NAMES)
        },
        "outputs": [
            "01_pass1_full_global_mechanism.png",
            "02_pass2_roi_local_mechanism.png",
            "03_pass2_restored_global_mechanism.png",
            "04_before_after_global_summary.png",
        ],
    }
    with (output_dir / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    np.savez_compressed(
        output_dir / "mechanism_maps.npz",
        image=image,
        gt=gt,
        final_prediction=final_prediction,
        anatomy_logits=anatomy["logits"].float().cpu().numpy(),
        restored_refinement_logits=restored.float().cpu().numpy(),
    )
    print(json.dumps(manifest, indent=2))
    print(f"saved={output_dir}")


if __name__ == "__main__":
    main()
