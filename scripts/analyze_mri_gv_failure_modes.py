#!/usr/bin/env python3
"""Offline MRI GV-ROI failure-mode analysis and publication-style plots."""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import SimpleITK as sitk
from scipy.stats import spearmanr


CLASS_NAMES = {1: "LV", 2: "RV", 3: "LA", 4: "RA", 5: "Myo", 6: "AO", 7: "PA"}
CLASS_COLORS = {
    1: "#e41a1c", 2: "#377eb8", 3: "#4daf4a", 4: "#984ea3",
    5: "#ff7f00", 6: "#00c8c8", 7: "#f4d03f",
}


def _read(path: str) -> tuple[np.ndarray, sitk.Image]:
    image = sitk.ReadImage(path)
    return sitk.GetArrayFromImage(image), image


def _case_path(directory: str, case_id: str, image: bool = False) -> str:
    suffix = "_0000.nii.gz" if image else ".nii.gz"
    return os.path.join(directory, case_id + suffix)


def _robust_display(image: np.ndarray) -> np.ndarray:
    finite = image[np.isfinite(image)]
    low, high = np.percentile(finite, [1, 99]) if finite.size else (0.0, 1.0)
    return np.clip((image - low) / max(high - low, 1e-6), 0.0, 1.0)


def _overlay(ax, labels: np.ndarray, alpha: float = 0.38) -> None:
    rgba = np.zeros((*labels.shape, 4), dtype=np.float32)
    for class_id, color in CLASS_COLORS.items():
        rgb = matplotlib.colors.to_rgb(color)
        mask = labels == class_id
        rgba[mask, :3] = rgb
        rgba[mask, 3] = alpha
    ax.imshow(rgba, interpolation="nearest")


def _case_features(case_id: str, domain: str, image_dir: str, label_dir: str) -> dict:
    image, image_itk = _read(_case_path(image_dir, case_id, image=True))
    gt, _ = _read(_case_path(label_dir, case_id))
    spacing_xyz = np.asarray(image_itk.GetSpacing(), dtype=np.float64)
    voxel_ml = float(np.prod(spacing_xyz) / 1000.0)
    depth, height, width = gt.shape
    norm_image = _robust_display(image.astype(np.float32))
    row = {
        "case_id": case_id, "domain": domain,
        "depth": depth, "height": height, "width": width,
        "spacing_x": spacing_xyz[0], "spacing_y": spacing_xyz[1],
        "spacing_z": spacing_xyz[2], "voxel_ml": voxel_ml,
    }
    foreground = gt > 0
    row["foreground_volume_ml"] = float(foreground.sum() * voxel_ml)
    row["foreground_fraction"] = float(foreground.mean())
    for class_id, name in CLASS_NAMES.items():
        mask = gt == class_id
        coords = np.argwhere(mask)
        row[f"{name}_volume_ml"] = float(mask.sum() * voxel_ml)
        row[f"{name}_fraction"] = float(mask.mean())
        row[f"{name}_slice_fraction"] = float(mask.any(axis=(1, 2)).mean())
        row[f"{name}_intensity_mean"] = float(norm_image[mask].mean()) if mask.any() else np.nan
        if coords.size:
            lo, hi = coords.min(axis=0), coords.max(axis=0) + 1
            centroid = coords.mean(axis=0)
            row[f"{name}_centroid_z"] = float(centroid[0] / depth)
            row[f"{name}_centroid_y"] = float(centroid[1] / height)
            row[f"{name}_centroid_x"] = float(centroid[2] / width)
            row[f"{name}_extent_z"] = float((hi[0] - lo[0]) / depth)
            row[f"{name}_extent_y"] = float((hi[1] - lo[1]) / height)
            row[f"{name}_extent_x"] = float((hi[2] - lo[2]) / width)
        else:
            for metric in ("centroid_z", "centroid_y", "centroid_x", "extent_z", "extent_y", "extent_x"):
                row[f"{name}_{metric}"] = np.nan
    for left, right in (("AO", "PA"), ("AO", "LV"), ("PA", "RV")):
        a = np.array([row[f"{left}_centroid_z"], row[f"{left}_centroid_y"], row[f"{left}_centroid_x"]])
        b = np.array([row[f"{right}_centroid_z"], row[f"{right}_centroid_y"], row[f"{right}_centroid_x"]])
        row[f"distance_{left}_{right}"] = float(np.linalg.norm(a - b))
    return row


def _distribution_report(rows: list[dict]) -> dict:
    numeric = [key for key, value in rows[0].items() if key not in {"case_id", "domain"} and isinstance(value, (int, float, np.number))]
    report = {"n_in_domain": sum(r["domain"] == "in_domain" for r in rows), "n_ood": sum(r["domain"] == "ood" for r in rows), "metrics": {}}
    for key in numeric:
        a = np.asarray([r[key] for r in rows if r["domain"] == "in_domain"], dtype=float)
        b = np.asarray([r[key] for r in rows if r["domain"] == "ood"], dtype=float)
        a, b = a[np.isfinite(a)], b[np.isfinite(b)]
        if not len(a) or not len(b):
            continue
        pooled = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / max(1, len(a) + len(b) - 2))
        smd = float((b.mean() - a.mean()) / pooled) if pooled > 1e-12 else 0.0
        report["metrics"][key] = {
            "in_mean": float(a.mean()), "in_std": float(a.std(ddof=1)) if len(a) > 1 else 0.0,
            "ood_mean": float(b.mean()), "ood_std": float(b.std(ddof=1)) if len(b) > 1 else 0.0,
            "absolute_standardized_mean_difference": abs(smd), "signed_smd_ood_minus_in": smd,
        }
    report["top_shifts"] = [
        {"metric": key, **value}
        for key, value in sorted(
            report["metrics"].items(), key=lambda item: item[1]["absolute_standardized_mean_difference"], reverse=True
        )[:30]
    ]
    return report


def _domain_montage(rows, raw_base, out_path):
    selected = {}
    for domain in ("in_domain", "ood"):
        pool = sorted([r for r in rows if r["domain"] == domain], key=lambda r: r["AO_volume_ml"] + r["PA_volume_ml"])
        indices = np.linspace(0, len(pool) - 1, 4).round().astype(int)
        selected[domain] = [pool[i] for i in indices]
    fig, axes = plt.subplots(2, 4, figsize=(18, 9), constrained_layout=True)
    for row_index, domain in enumerate(("in_domain", "ood")):
        image_dir = os.path.join(raw_base, "imagesTr" if domain == "in_domain" else "imagesTs")
        label_dir = os.path.join(raw_base, "labelsTr" if domain == "in_domain" else "labelsTs")
        for ax, row in zip(axes[row_index], selected[domain]):
            image, _ = _read(_case_path(image_dir, row["case_id"], image=True))
            gt, _ = _read(_case_path(label_dir, row["case_id"]))
            gv_area = np.isin(gt, [6, 7]).sum(axis=(1, 2))
            z = int(gv_area.argmax())
            ax.imshow(_robust_display(image)[z], cmap="gray")
            _overlay(ax, gt[z])
            ax.set_title(f"{row['case_id']}  z={z}\nGV={row['AO_volume_ml'] + row['PA_volume_ml']:.1f} ml")
            ax.axis("off")
        axes[row_index, 0].set_ylabel("in-domain" if domain == "in_domain" else "OOD", fontsize=14)
    handles = [plt.Line2D([0], [0], color=CLASS_COLORS[i], lw=6, label=CLASS_NAMES[i]) for i in CLASS_NAMES]
    fig.legend(handles=handles, loc="lower center", ncol=7)
    fig.suptitle("MRI anatomy at the slice with maximal AO+PA area", fontsize=16)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _shift_plot(report, out_path):
    preferred = []
    for name in CLASS_NAMES.values():
        preferred.extend([f"{name}_volume_ml", f"{name}_centroid_x", f"{name}_centroid_y", f"{name}_centroid_z"])
    preferred += ["spacing_x", "spacing_z", "distance_AO_PA"]
    values = [(key, report["metrics"][key]) for key in preferred if key in report["metrics"]]
    values.sort(key=lambda item: item[1]["absolute_standardized_mean_difference"])
    fig, ax = plt.subplots(figsize=(10, max(7, 0.27 * len(values))))
    labels = [key for key, _ in values]
    smd = [value["signed_smd_ood_minus_in"] for _, value in values]
    ax.barh(np.arange(len(labels)), smd, color=["#d95f02" if x > 0 else "#1b9e77" for x in smd])
    ax.axvline(0, color="black", lw=0.8)
    ax.axvline(0.8, color="gray", ls="--", lw=0.8)
    ax.axvline(-0.8, color="gray", ls="--", lw=0.8)
    ax.set_yticks(np.arange(len(labels)), labels=labels, fontsize=8)
    ax.set_xlabel("standardized mean difference (OOD − in-domain)")
    ax.set_title("MRI anatomical/acquisition domain shifts")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _sum_confusions(records, key, predicate=lambda _: True):
    matrices = [np.asarray(r[key], dtype=np.int64) for r in records if predicate(r)]
    return np.sum(matrices, axis=0) if matrices else np.zeros((8, 8), dtype=np.int64)


def _roi_report(records):
    gv_positive = [r for r in records if r["class_voxels"]["6"] + r["class_voxels"]["7"] > 0]
    nonfallback = [r for r in records if not r["roi"]["fallback"]]
    def summarize(pool):
        if not pool:
            return {}
        return {
            "n_blocks": len(pool),
            "fallback_rate": float(np.mean([r["roi"]["fallback"] for r in pool])),
            "roi_area_mean": float(np.mean([r["roi"]["area_fraction"] for r in pool])),
            "roi_area_median": float(np.median([r["roi"]["area_fraction"] for r in pool])),
            "class_coverage_mean": {
                str(c): float(np.mean([r["class_coverage"][str(c)] for r in pool if r["class_coverage"][str(c)] is not None]))
                if any(r["class_coverage"][str(c)] is not None for r in pool) else None
                for c in range(1, 8)
            },
        }
    confusion = _sum_confusions(records, "confusion_final_inside_roi")
    local_confusion = _sum_confusions(records, "confusion_local_inside_roi")
    foreground_inside = np.array([sum(r["class_voxels_inside_roi"].values()) for r in records], dtype=float)
    composition = {
        str(c): int(sum(r["class_voxels_inside_roi"][str(c)] for r in records)) for c in range(1, 8)
    }
    total_fg = max(1, sum(composition.values()))
    return {
        "all_blocks": summarize(records), "gv_positive_blocks": summarize(gv_positive),
        "nonfallback_blocks": summarize(nonfallback),
        "foreground_composition_inside_roi": composition,
        "foreground_composition_fraction": {k: v / total_fg for k, v in composition.items()},
        "confusion_final_inside_roi": confusion.tolist(),
        "confusion_local_inside_roi": local_confusion.tolist(),
        "mean_foreground_voxels_inside_roi": float(foreground_inside.mean()) if len(foreground_inside) else 0.0,
    }


def _confusion_plot(matrix, out_path, title):
    matrix = np.asarray(matrix, dtype=float)
    normalized = matrix / np.maximum(matrix.sum(axis=1, keepdims=True), 1.0)
    fig, ax = plt.subplots(figsize=(9, 8))
    image = ax.imshow(normalized, cmap="magma", vmin=0, vmax=1)
    labels = ["BG"] + list(CLASS_NAMES.values())
    ax.set_xticks(range(8), labels=labels, rotation=45, ha="right")
    ax.set_yticks(range(8), labels=labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Ground truth")
    ax.set_title(title)
    for y in range(8):
        for x in range(8):
            value = normalized[y, x]
            if value >= 0.01:
                ax.text(x, y, f"{value:.2f}", ha="center", va="center", color="white" if value < 0.55 else "black", fontsize=8)
    fig.colorbar(image, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _prediction_montage(metrics_path, pred_dir, raw_base, records, out_path):
    with open(metrics_path, newline="") as handle:
        rows = list(csv.DictReader(handle))
    rows.sort(key=lambda row: float(row["dice_mean_gt"]))
    selected = rows[:3] + rows[-3:]
    by_case = defaultdict(list)
    for record in records:
        by_case[record["case_id"]].append(record)
    fig, axes = plt.subplots(2, 6, figsize=(24, 8), constrained_layout=True)
    for column, metric in enumerate(selected):
        case_id = metric["case_id"]
        image, _ = _read(_case_path(os.path.join(raw_base, "imagesTs"), case_id, image=True))
        gt, _ = _read(_case_path(os.path.join(raw_base, "labelsTs"), case_id))
        pred, _ = _read(_case_path(pred_dir, case_id))
        pool = by_case[case_id]
        record = max(pool, key=lambda r: r["class_voxels"]["6"] + r["class_voxels"]["7"])
        preproc_depth = max(r["z_start"] + r["valid_count"] for r in pool)
        z_pre = record["z_start"] + record["valid_count"] // 2
        z = min(gt.shape[0] - 1, int(round((z_pre + 0.5) / preproc_depth * gt.shape[0] - 0.5)))
        display = _robust_display(image)[z]
        for row_index, labels in enumerate((gt[z], pred[z])):
            ax = axes[row_index, column]
            ax.imshow(display, cmap="gray")
            _overlay(ax, labels)
            if row_index == 0:
                roi = record["roi"]
                x0, x1 = roi["x0"] / 320 * gt.shape[2], roi["x1"] / 320 * gt.shape[2]
                y0, y1 = roi["y0"] / 320 * gt.shape[1], roi["y1"] / 320 * gt.shape[1]
                rect = plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor="lime", lw=2)
                ax.add_patch(rect)
            ax.axis("off")
        axes[0, column].set_title(f"{case_id}  mean={float(metric['dice_mean_gt']):.3f}\nAO={float(metric['dice_6']):.3f}, PA={float(metric['dice_7']):.3f}")
    axes[0, 0].set_ylabel("GT + predicted ROI", fontsize=13)
    axes[1, 0].set_ylabel("Final prediction", fontsize=13)
    fig.suptitle("Three worst and three best OOD cases", fontsize=16)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _failure_correlations(anatomy_rows, metrics_path, records, output_dir):
    with open(metrics_path, newline="") as handle:
        metric_rows = {row["case_id"]: row for row in csv.DictReader(handle)}
    by_case = defaultdict(list)
    for record in records:
        by_case[record["case_id"]].append(record)
    joined = []
    for anatomy in anatomy_rows:
        if anatomy["domain"] != "ood" or anatomy["case_id"] not in metric_rows:
            continue
        case_id = anatomy["case_id"]
        row = dict(anatomy)
        row.update({
            "dice_mean": float(metric_rows[case_id]["dice_mean_gt"]),
            "dice_AO": float(metric_rows[case_id]["dice_6"]),
            "dice_PA": float(metric_rows[case_id]["dice_7"]),
        })
        pool = by_case[case_id]
        row["roi_fallback_rate"] = float(np.mean([r["roi"]["fallback"] for r in pool]))
        row["roi_area_mean"] = float(np.mean([r["roi"]["area_fraction"] for r in pool]))
        for class_id, name in ((6, "AO"), (7, "PA")):
            total = sum(r["class_voxels"][str(class_id)] for r in pool)
            inside = sum(r["class_voxels_inside_roi"][str(class_id)] for r in pool)
            row[f"roi_{name}_coverage"] = inside / total if total else 1.0
        joined.append(row)
    predictors = [
        key for key, value in joined[0].items()
        if key not in {"case_id", "domain", "dice_mean", "dice_AO", "dice_PA"}
        and isinstance(value, (int, float, np.number))
    ]
    correlations = []
    for target in ("dice_mean", "dice_AO", "dice_PA"):
        for predictor in predictors:
            x = np.asarray([row[predictor] for row in joined], dtype=float)
            y = np.asarray([row[target] for row in joined], dtype=float)
            valid = np.isfinite(x) & np.isfinite(y)
            if valid.sum() < 5 or np.unique(x[valid]).size < 3:
                continue
            rho, p_value = spearmanr(x[valid], y[valid])
            correlations.append({
                "target": target, "predictor": predictor, "spearman_rho": float(rho),
                "p_value_unadjusted": float(p_value), "n": int(valid.sum()),
            })
    correlations.sort(key=lambda row: abs(row["spearman_rho"]), reverse=True)
    report = {
        "n_cases": len(joined),
        "note": "Exploratory univariate correlations; p-values are unadjusted.",
        "top_correlations": correlations[:40],
        "all_correlations": correlations,
    }
    with open(os.path.join(output_dir, "failure_correlations.json"), "w") as handle:
        json.dump(report, handle, indent=2)
    with open(os.path.join(output_dir, "ood_case_failure_features.csv"), "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=joined[0].keys())
        writer.writeheader(); writer.writerows(joined)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), constrained_layout=True)
    for ax, target in zip(axes, ("dice_mean", "dice_AO", "dice_PA")):
        best = next(row for row in correlations if row["target"] == target)
        predictor = best["predictor"]
        x = np.asarray([row[predictor] for row in joined], dtype=float)
        y = np.asarray([row[target] for row in joined], dtype=float)
        ax.scatter(x, y, c="#4c78a8", edgecolor="white", s=55)
        ax.set_xlabel(predictor); ax.set_ylabel(target)
        ax.set_title(f"Spearman ρ={best['spearman_rho']:.2f}, p={best['p_value_unadjusted']:.3g}")
    fig.suptitle("Strongest exploratory OOD failure correlates")
    fig.savefig(os.path.join(output_dir, "failure_correlations.png"), dpi=180)
    plt.close(fig)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_base", required=True)
    parser.add_argument("--pred_dir", required=True)
    parser.add_argument("--metrics_csv", required=True)
    parser.add_argument("--block_diagnostics", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    in_cases = sorted(name.removesuffix(".nii.gz") for name in os.listdir(os.path.join(args.raw_base, "labelsTr")) if name.endswith(".nii.gz"))
    ood_cases = sorted(name.removesuffix(".nii.gz") for name in os.listdir(os.path.join(args.raw_base, "labelsTs")) if name.endswith(".nii.gz"))
    rows = []
    for case_id in in_cases:
        rows.append(_case_features(case_id, "in_domain", os.path.join(args.raw_base, "imagesTr"), os.path.join(args.raw_base, "labelsTr")))
    for case_id in ood_cases:
        rows.append(_case_features(case_id, "ood", os.path.join(args.raw_base, "imagesTs"), os.path.join(args.raw_base, "labelsTs")))
    with open(os.path.join(args.output_dir, "case_anatomy_metrics.csv"), "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader(); writer.writerows(rows)
    distribution = _distribution_report(rows)
    with open(os.path.join(args.output_dir, "domain_shift_summary.json"), "w") as handle:
        json.dump(distribution, handle, indent=2)
    _domain_montage(rows, args.raw_base, os.path.join(args.output_dir, "in_vs_ood_gt_montage.png"))
    _shift_plot(distribution, os.path.join(args.output_dir, "domain_shift_effect_sizes.png"))
    with open(args.block_diagnostics) as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    roi = _roi_report(records)
    with open(os.path.join(args.output_dir, "roi_quantification.json"), "w") as handle:
        json.dump(roi, handle, indent=2)
    _confusion_plot(roi["confusion_final_inside_roi"], os.path.join(args.output_dir, "roi_final_confusion.png"), "Final hard-switch confusion inside predicted ROI")
    _confusion_plot(roi["confusion_local_inside_roi"], os.path.join(args.output_dir, "roi_local_confusion.png"), "Local branch confusion inside predicted ROI")
    _prediction_montage(args.metrics_csv, args.pred_dir, args.raw_base, records, os.path.join(args.output_dir, "ood_failure_montage.png"))
    correlations = _failure_correlations(rows, args.metrics_csv, records, args.output_dir)
    manifest = {
        "in_domain_cases": in_cases, "ood_cases": ood_cases,
        "class_names": CLASS_NAMES,
        "note": "ROI rectangles are mapped by normalized XY coordinates from the aligned 320x320 inference grid.",
    }
    with open(os.path.join(args.output_dir, "manifest.json"), "w") as handle:
        json.dump(manifest, handle, indent=2)
    print(json.dumps({"output_dir": os.path.abspath(args.output_dir), "top_shifts": distribution["top_shifts"][:10], "roi": {k: roi[k] for k in ("all_blocks", "gv_positive_blocks", "foreground_composition_fraction")}, "failure_correlations": correlations["top_correlations"][:10]}, indent=2))


if __name__ == "__main__":
    main()
