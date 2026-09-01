#!/usr/bin/env python3
"""Plot paired metrics and representative cases for selective refinement."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from scipy.stats import wilcoxon


CLASS_NAMES = ("LV", "RV", "LA", "RA", "Myo", "AO", "PA")
CLASS_COLORS = (
    "#4c78a8",
    "#72b7b2",
    "#b279a2",
    "#f58518",
    "#54a24b",
    "#eeca3b",
    "#e45756",
)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _read_volume(path: Path) -> np.ndarray:
    return sitk.GetArrayFromImage(sitk.ReadImage(str(path)))


def _normalize_image(image: np.ndarray) -> np.ndarray:
    low, high = np.percentile(image, (1.0, 99.0))
    return np.clip((image - low) / max(1e-6, high - low), 0.0, 1.0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--merged_dir", type=Path, required=True)
    parser.add_argument("--shard_root", type=Path, required=True)
    parser.add_argument("--global_pred_dir", type=Path, required=True)
    parser.add_argument("--images_dir", type=Path, required=True)
    parser.add_argument("--labels_dir", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--scope", default="any")
    parser.add_argument("--bootstrap_samples", type=int, default=100000)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    global_rows = _read_csv(args.merged_dir / "selective_global.csv")
    fused_rows = _read_csv(args.merged_dir / "selective_sweep.csv")
    global_by_key = {(row["case_id"], row["postprocess"]): row for row in global_rows}
    paired = {}
    rng = np.random.default_rng(20260902)
    for postprocess in ("none", "keep_largest_per_class"):
        rows = [row for row in fused_rows if row["postprocess"] == postprocess]
        rows.sort(key=lambda row: row["case_id"])
        global_values = np.asarray(
            [
                float(global_by_key[(row["case_id"], postprocess)]["dice_mean_gt"])
                for row in rows
            ]
        )
        fused_values = np.asarray([float(row["dice_mean_gt"]) for row in rows])
        delta = fused_values - global_values
        bootstrap = rng.choice(
            delta,
            size=(args.bootstrap_samples, len(delta)),
            replace=True,
        ).mean(axis=1)
        paired[postprocess] = {
            "rows": rows,
            "global": global_values,
            "fused": fused_values,
            "delta": delta,
            "mean_delta": float(delta.mean()),
            "median_delta": float(np.median(delta)),
            "bootstrap_ci95": [
                float(value) for value in np.quantile(bootstrap, (0.025, 0.975))
            ],
            "wilcoxon_p": float(wilcoxon(delta).pvalue),
            "wins": int((delta > 0).sum()),
            "losses": int((delta < 0).sum()),
        }

    raw = paired["none"]
    raw_rows = raw["rows"]
    class_delta = []
    for class_id in range(1, 8):
        values = [
            float(row[f"dice_{class_id}"])
            - float(global_by_key[(row["case_id"], "none")][f"dice_{class_id}"])
            for row in raw_rows
        ]
        class_delta.append(float(np.mean(values)))

    fused_paths = {
        path.name: path
        for path in args.shard_root.glob(
            f"shard_gpu*/selective_pred_{args.scope}_none/*.nii.gz"
        )
    }
    changed_outcomes = {"beneficial": 0, "harmful": 0, "wrong_to_wrong": 0}
    for filename, fused_path in fused_paths.items():
        global_labels = _read_volume(args.global_pred_dir / filename)
        fused_labels = _read_volume(fused_path)
        target = _read_volume(args.labels_dir / filename)
        changed = fused_labels != global_labels
        before_correct = global_labels == target
        after_correct = fused_labels == target
        changed_outcomes["beneficial"] += int(
            (changed & ~before_correct & after_correct).sum()
        )
        changed_outcomes["harmful"] += int(
            (changed & before_correct & ~after_correct).sum()
        )
        changed_outcomes["wrong_to_wrong"] += int(
            (changed & ~before_correct & ~after_correct).sum()
        )

    result = {
        "raw": {key: value for key, value in raw.items() if key != "rows"},
        "keep_largest_per_class": {
            key: value
            for key, value in paired["keep_largest_per_class"].items()
            if key != "rows"
        },
        "raw_class_mean_delta": dict(zip(CLASS_NAMES, class_delta)),
        "changed_voxel_outcomes": changed_outcomes,
        "case_delta_raw": {
            row["case_id"]: float(delta) for row, delta in zip(raw_rows, raw["delta"])
        },
    }
    for section in ("raw", "keep_largest_per_class"):
        for key in ("global", "fused", "delta"):
            result[section].pop(key, None)
    with (args.out_dir / "analysis_summary.json").open("w") as handle:
        json.dump(result, handle, indent=2)

    figure, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    ax = axes[0, 0]
    ax.scatter(raw["global"], raw["fused"], color="#4c78a8", alpha=0.8)
    low = min(raw["global"].min(), raw["fused"].min()) - 0.01
    high = max(raw["global"].max(), raw["fused"].max()) + 0.01
    ax.plot((low, high), (low, high), "--", color="0.4", linewidth=1)
    ax.set(xlabel="Global Dice", ylabel="Selective Dice", title="Paired OOD cases")
    ax.set_xlim(low, high)
    ax.set_ylim(low, high)

    ax = axes[0, 1]
    colors = ["#54a24b" if value >= 0 else "#e45756" for value in class_delta]
    ax.bar(CLASS_NAMES, np.asarray(class_delta) * 100.0, color=colors)
    ax.axhline(0.0, color="0.25", linewidth=1)
    ax.set(ylabel="Mean Dice change (points)", title="Per-class effect")

    ax = axes[1, 0]
    order = np.argsort(raw["delta"])
    ordered_delta = raw["delta"][order] * 100.0
    ordered_cases = np.asarray([row["case_id"] for row in raw_rows])[order]
    ax.barh(
        np.arange(len(order)),
        ordered_delta,
        color=["#54a24b" if value >= 0 else "#e45756" for value in ordered_delta],
    )
    ax.set_yticks(np.arange(len(order)), ordered_cases, fontsize=7)
    ax.axvline(0.0, color="0.25", linewidth=1)
    ax.set(xlabel="Dice change (points)", title="Case-level raw Dice change")

    ax = axes[1, 1]
    outcome_names = ("beneficial", "harmful", "wrong_to_wrong")
    outcome_values = [changed_outcomes[name] for name in outcome_names]
    ax.bar(
        ("Corrected", "Broke correct", "Wrong→wrong"),
        np.asarray(outcome_values) / 1000.0,
        color=("#54a24b", "#e45756", "#b279a2"),
    )
    ax.set(ylabel="Changed voxels (thousands)", title="Overwrite outcomes")
    figure.suptitle(
        "MRI OOD selective AO/PA refinement "
        f"(mean Δ={raw['mean_delta'] * 100:+.3f} points, "
        f"wins={raw['wins']}/{len(raw_rows)})"
    )
    figure.savefig(args.out_dir / "selective_summary.png", dpi=180)
    plt.close(figure)

    best_index = int(np.argmax(raw["delta"]))
    worst_index = int(np.argmin(raw["delta"]))
    example_indices = (best_index, worst_index)
    cmap = ListedColormap(CLASS_COLORS)
    figure, axes = plt.subplots(2, 3, figsize=(13, 8), constrained_layout=True)
    for row_index, paired_index in enumerate(example_indices):
        row = raw_rows[paired_index]
        case_id = row["case_id"]
        filename = f"{case_id}.nii.gz"
        image_path = next(args.images_dir.glob(f"{case_id}_0000.nii.gz"))
        image = _read_volume(image_path)
        target = _read_volume(args.labels_dir / filename)
        global_labels = _read_volume(args.global_pred_dir / filename)
        fused_labels = _read_volume(fused_paths[filename])
        changed_per_slice = (global_labels != fused_labels).sum(axis=(1, 2))
        slice_index = int(changed_per_slice.argmax())
        gray = _normalize_image(image[slice_index])
        segmentations = (target, global_labels, fused_labels)
        titles = ("Ground truth", "Global", "Selective AO/PA")
        for column, (segmentation, title) in enumerate(zip(segmentations, titles)):
            ax = axes[row_index, column]
            ax.imshow(gray, cmap="gray")
            overlay = np.ma.masked_where(
                segmentation[slice_index] == 0,
                segmentation[slice_index] - 1,
            )
            ax.imshow(overlay, cmap=cmap, vmin=0, vmax=6, alpha=0.55)
            ax.set_title(title)
            ax.axis("off")
        delta = raw["delta"][paired_index] * 100.0
        axes[row_index, 0].text(
            -0.04,
            0.5,
            f"{case_id}\nz={slice_index}, Δ={delta:+.2f}",
            transform=axes[row_index, 0].transAxes,
            rotation=90,
            va="center",
            ha="right",
            fontsize=10,
        )
    handles = [
        Patch(facecolor=color, label=name)
        for name, color in zip(CLASS_NAMES, CLASS_COLORS)
    ]
    figure.legend(handles=handles, loc="lower center", ncol=7)
    figure.suptitle("Best and worst raw-Dice selective-refinement cases")
    figure.savefig(args.out_dir / "best_worst_overlays.png", dpi=180)
    plt.close(figure)


if __name__ == "__main__":
    main()
