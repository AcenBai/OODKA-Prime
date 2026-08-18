#!/usr/bin/env python3
"""Restore ROI-local mechanism-v3 maps onto the full LGE model canvas."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/oodka_mechanism_v3_restore_mpl")

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import torch
import torch.nn.functional as F


LEVELS = (2, 3, 4, 5)
CLASS_NAMES = ("LV", "RV", "normal_myo", "scar_edema")
CLASS_COLORS = ("#00ff3b", "#ff2d2d", "#00c8ff", "#ffd400")


def resize(value: np.ndarray, size: tuple[int, int], *, nearest: bool = False) -> np.ndarray:
    tensor = torch.from_numpy(np.asarray(value))[None, None].float()
    kwargs = {} if nearest else {"align_corners": False}
    result = F.interpolate(
        tensor, size=size, mode="nearest" if nearest else "bilinear", **kwargs
    )[0, 0]
    return result.numpy()


def restore(value: np.ndarray, roi: dict, size: tuple[int, int]) -> np.ndarray:
    output = np.full(size, np.nan, dtype=np.float32)
    local = resize(value, (roi["height"], roi["width"]))
    output[roi["y0"] : roi["y1"], roi["x0"] : roi["x1"]] = local
    return output


def draw_base(axis, image: np.ndarray, gt: np.ndarray, roi: dict, class_id=None):
    axis.imshow(image, cmap="gray", vmin=np.percentile(image, 1), vmax=np.percentile(image, 99))
    ids = range(1, 5) if class_id is None else (class_id,)
    for cid in ids:
        mask = gt == cid
        if mask.any():
            axis.contour(mask, [0.5], colors=[CLASS_COLORS[cid - 1]], linewidths=1.1)
    axis.add_patch(Rectangle(
        (roi["x0"], roi["y0"]), roi["width"], roi["height"],
        fill=False, edgecolor="#facc15", linewidth=1.5,
    ))
    axis.axis("off")


def heat(axis, image, value, roi, title, vmax=None, cmap="magma"):
    axis.imshow(image, cmap="gray", vmin=np.percentile(image, 1), vmax=np.percentile(image, 99))
    artist = axis.imshow(np.ma.masked_invalid(value), cmap=cmap, vmin=0, vmax=vmax, alpha=0.9)
    axis.add_patch(Rectangle(
        (roi["x0"], roi["y0"]), roi["width"], roi["height"],
        fill=False, edgecolor="#facc15", linewidth=1.2,
    ))
    axis.set_title(title, fontsize=9)
    axis.axis("off")
    return artist


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--roi_case_root", required=True)
    parser.add_argument("--aligned_npz", required=True)
    parser.add_argument("--slice_index", type=int, required=True)
    parser.add_argument("--output_root", required=True)
    args = parser.parse_args()

    source = Path(args.roi_case_root)
    output = Path(args.output_root)
    manifest = json.loads((source / "manifest.json").read_text())
    roi = manifest["predicted_roi"]
    with np.load(args.aligned_npz) as aligned:
        image = resize(aligned["data"][0, args.slice_index].astype(np.float32), (256, 256))
        raw_gt = resize(aligned["seg"][0, args.slice_index].astype(np.float32), (256, 256), nearest=True).astype(np.int16)
    gt = np.zeros_like(raw_gt)
    gt[raw_gt == 3] = 1
    gt[raw_gt == 5] = 2
    gt[raw_gt == 4] = 3
    gt[np.isin(raw_gt, (1, 2))] = 4

    rep = np.load(source / "representation" / "representation_maps.npz")
    scales = json.loads((source / "representation" / "color_scales.json").read_text())
    rep_out = output / "representation"
    rep_out.mkdir(parents=True, exist_ok=True)

    figure, axes = plt.subplots(4, 4, figsize=(16, 16), constrained_layout=True)
    for row, level in enumerate(LEVELS):
        draw_base(axes[row, 0], image, gt, roi)
        axes[row, 0].set_title(f"res{level}: full LGE + GT + predicted ROI")
        heat(axes[row, 1], image, restore(rep[f"raw_res{level}_expert"], roi, image.shape), roi, "Expert raw RMS", scales["raw"]["vmax"])
        heat(axes[row, 2], image, restore(rep[f"raw_res{level}_student_relative"], roi, image.shape), roi, "Student relative", 1.0, "viridis")
        heat(axes[row, 3], image, restore(rep[f"raw_res{level}_student"], roi, image.shape), roi, "Student raw RMS", scales["raw"]["vmax"])
    figure.suptitle("Pass 2 backbone representations restored to full-image coordinates", fontsize=15)
    figure.savefig(rep_out / "01_backbone_raw_energy.png", dpi=180)
    plt.close(figure)

    figure, axes = plt.subplots(4, 6, figsize=(22, 16), constrained_layout=True)
    keys = ("expert_p", "expert_s", "student_p", "student_s", "expert_s_share", "student_s_share")
    names = ("Expert P", "Expert S", "Student P", "Student S", "Expert S share", "Student S share")
    for row, level in enumerate(LEVELS):
        for column, (key, name) in enumerate(zip(keys, names)):
            vmax = 1.0 if "share" in key else scales["decomposed"]["vmax"]
            cmap = "viridis" if "share" in key else "magma"
            heat(axes[row, column], image, restore(rep[f"branch_res{level}_{key}"], roi, image.shape), roi, f"res{level}: {name}", vmax, cmap)
    figure.suptitle("Pass 2 decomposed expert/student representations restored globally", fontsize=15)
    figure.savefig(rep_out / "02_disentangled_energy.png", dpi=180)
    plt.close(figure)

    figure, axes = plt.subplots(4, 4, figsize=(16, 16), constrained_layout=True)
    for row, level in enumerate(LEVELS):
        draw_base(axes[row, 0], image, gt, roi)
        for column, (key, name) in enumerate((("p", "P decoder RMS"), ("s", "S decoder RMS"), ("s_share", "S share")), start=1):
            vmax = 1.0 if key == "s_share" else scales["pixel_decoder"]["vmax"]
            heat(axes[row, column], image, restore(rep[f"decoder_res{level}_{key}"], roi, image.shape), roi, name, vmax, "viridis" if key == "s_share" else "magma")
    figure.suptitle("Pass 2 pixel-decoder P/S representations restored globally", fontsize=15)
    figure.savefig(rep_out / "03_pixel_decoder_energy.png", dpi=180)
    plt.close(figure)

    decision = np.load(source / "decision" / "decision_maps.npz")
    decision_out = output / "decision"
    decision_out.mkdir(parents=True, exist_ok=True)
    gate_mean = decision["gate_mean"]
    figure, axes = plt.subplots(4, 3, figsize=(12, 15), constrained_layout=True)
    for row, name in enumerate(CLASS_NAMES):
        for column, (key, title, cmap, vmax) in enumerate((
            ("gate_mean", "P gate mean", "viridis", 1.0),
            ("gate_concentration", "Beta concentration", "magma", None),
            ("gate_uncertainty", "Gate standard deviation", "magma", None),
        )):
            value = restore(decision[key][row], roi, image.shape)
            heat(axes[row, column], image, value, roi, f"{name}: {title}", vmax, cmap)
    figure.suptitle("Pass 2 prompt-conditioned Beta router restored globally", fontsize=15)
    figure.savefig(decision_out / "02_router_heatmaps.png", dpi=190)
    plt.close(figure)

    figure, axes = plt.subplots(4, 5, figsize=(17, 13), constrained_layout=True)
    vmax = scales["pixel_decoder"]["vmax"]
    for row, name in enumerate(CLASS_NAMES):
        draw_base(axes[row, 0], image, gt, roi, row + 1)
        axes[row, 0].set_title(f"{name} GT")
        for column, level in enumerate(LEVELS, start=1):
            value = restore(decision[f"class{row + 1:02d}_res{level}_fused"], roi, image.shape)
            title = f"res{level}" if row == 0 else f"mean P gate={float(gate_mean[row].mean()):.3f}"
            heat(axes[row, column], image, value, roi, title, vmax)
    figure.suptitle("All Pass 2 prompt-gated fused representations restored globally", fontsize=15)
    figure.savefig(decision_out / "03_all_class_fusion.png", dpi=180)
    plt.close(figure)

    np.savez_compressed(
        output / "restored_global_maps.npz",
        image=image,
        gt=gt,
        probabilities=np.stack([restore(value, roi, image.shape) for value in decision["probabilities"]]),
        roi=np.asarray([roi["x0"], roi["y0"], roi["x1"], roi["y1"]]),
    )
    (output / "manifest.json").write_text(json.dumps({
        "schema": "oodka-mechanism-visualization-v3-restored-global",
        "source": str(source.resolve()),
        "predicted_roi": roi,
        "coordinate_system": "full 256x256 model canvas; NaN outside predicted ROI",
        "ot_note": "OT matrices remain in the pass2_predicted_roi directory because transport is defined in ROI-token coordinates.",
    }, indent=2))


if __name__ == "__main__":
    main()
