#!/usr/bin/env python3
"""Quantify ROI warp severity from evaluator block diagnostics."""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _quantiles(values: np.ndarray) -> dict[str, float]:
    return {
        name: float(np.quantile(values, value))
        for name, value in (
            ("min", 0.0), ("p05", 0.05), ("median", 0.5),
            ("p95", 0.95), ("max", 1.0),
        )
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--block_diagnostics", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--image_size", type=int, default=320)
    parser.add_argument(
        "--source_classes",
        default="6,7",
        help="Comma-separated GT classes whose union defines coverage.",
    )
    parser.add_argument("--title", default="MRI ROI geometry audit")
    args = parser.parse_args()
    source_classes = tuple(
        int(value) for value in args.source_classes.split(",") if value
    )
    if not source_classes:
        raise ValueError("--source_classes must not be empty")

    with open(args.block_diagnostics, encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    if not records:
        raise ValueError("No block diagnostics found")

    rows = []
    for record in records:
        roi = record["roi"]
        width = int(roi["x1"]) - int(roi["x0"])
        height = int(roi["y1"]) - int(roi["y0"])
        target = sum(
            int(record["class_voxels"].get(str(class_id), 0))
            for class_id in source_classes
        )
        inside = sum(
            int(record["class_voxels_inside_roi"].get(str(class_id), 0))
            for class_id in source_classes
        )
        scale_x = args.image_size / max(1, width)
        scale_y = args.image_size / max(1, height)
        rows.append(
            {
                "case_id": record["case_id"],
                "z_start": int(record["z_start"]),
                "valid_count": int(record["valid_count"]),
                "fallback": bool(roi["fallback"]),
                "width": width,
                "height": height,
                "area_fraction": width * height / float(args.image_size**2),
                "resize_scale_x": scale_x,
                "resize_scale_y": scale_y,
                "resize_max_scale": max(scale_x, scale_y),
                "resize_area_scale": scale_x * scale_y,
                "resize_anisotropy": max(scale_x, scale_y)
                / max(1e-12, min(scale_x, scale_y)),
                "pad_scale_x": 1.0,
                "pad_scale_y": 1.0,
                "pad_content_fraction": width * height
                / float(args.image_size**2),
                "source_voxels": target,
                "source_voxels_inside_roi": inside,
                "source_coverage": inside / target if target else None,
            }
        )

    os.makedirs(args.output_dir, exist_ok=True)
    csv_path = os.path.join(args.output_dir, "roi_geometry_blocks.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    positive = [row for row in rows if row["source_voxels"] > 0]
    nonfallback = [row for row in positive if not row["fallback"]]
    analysis = nonfallback or positive
    scales = np.asarray([row["resize_max_scale"] for row in analysis])
    scale_x = np.asarray([row["resize_scale_x"] for row in analysis])
    scale_y = np.asarray([row["resize_scale_y"] for row in analysis])
    anisotropy = np.asarray([row["resize_anisotropy"] for row in analysis])
    coverage = np.asarray([row["source_coverage"] for row in analysis])
    areas = np.asarray([row["area_fraction"] for row in analysis])

    case_rows: dict[str, list[dict]] = defaultdict(list)
    for row in positive:
        case_rows[row["case_id"]].append(row)
    case_summary = []
    for case_id, values in sorted(case_rows.items()):
        total = sum(row["source_voxels"] for row in values)
        inside = sum(row["source_voxels_inside_roi"] for row in values)
        current = np.asarray(
            [row["resize_max_scale"] for row in values if not row["fallback"]]
        )
        case_summary.append(
            {
                "case_id": case_id,
                "source_coverage": inside / max(1, total),
                "nonfallback_blocks": int(current.size),
                "resize_max_scale_median": (
                    float(np.median(current)) if current.size else 1.0
                ),
                "resize_max_scale_p95": (
                    float(np.quantile(current, 0.95)) if current.size else 1.0
                ),
                "fraction_nonfallback_over_4x": (
                    float(np.mean(current > 4.0)) if current.size else 0.0
                ),
            }
        )
    with open(
        os.path.join(args.output_dir, "roi_geometry_cases.csv"),
        "w", newline="", encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=case_summary[0].keys())
        writer.writeheader()
        writer.writerows(case_summary)

    summary = {
        "input": os.path.abspath(args.block_diagnostics),
        "image_size": args.image_size,
        "source_classes": source_classes,
        "n_blocks": len(rows),
        "n_source_positive_blocks": len(positive),
        "n_source_positive_nonfallback_blocks": len(nonfallback),
        "fallback_rate_all": float(np.mean([row["fallback"] for row in rows])),
        "resize_scale_x": _quantiles(scale_x),
        "resize_scale_y": _quantiles(scale_y),
        "resize_max_scale": _quantiles(scales),
        "resize_anisotropy": _quantiles(anisotropy),
        "area_fraction": _quantiles(areas),
        "fraction_over_2x": float(np.mean(scales > 2.0)),
        "fraction_over_4x": float(np.mean(scales > 4.0)),
        "fraction_over_8x": float(np.mean(scales > 8.0)),
        "source_coverage_mean_per_block": float(np.mean(coverage)),
        "source_zero_coverage_blocks": int(np.sum(coverage == 0.0)),
        "interpretation": (
            "resize_scale measures the current crop-to-square magnification; "
            "pad keeps both scales at 1 and uses area_fraction of the canvas."
        ),
    }
    with open(
        os.path.join(args.output_dir, "roi_geometry_summary.json"),
        "w", encoding="utf-8",
    ) as handle:
        json.dump(summary, handle, indent=2)

    fig, axes = plt.subplots(2, 2, figsize=(13, 10), constrained_layout=True)
    axes[0, 0].hist(scales, bins=35, color="#4c78a8", alpha=0.88)
    for threshold in (2, 4, 8):
        axes[0, 0].axvline(threshold, color="#d62728", ls="--", lw=1)
    axes[0, 0].set_xlabel("current resize: max axis magnification")
    axes[0, 0].set_ylabel("source-positive nonfallback blocks")
    axes[0, 0].set_title("ROI zoom severity")

    ordered = np.sort(scales)
    axes[0, 1].plot(ordered, np.arange(1, len(ordered) + 1) / len(ordered))
    axes[0, 1].set_xscale("log", base=2)
    axes[0, 1].set_xlabel("max axis magnification (log2)")
    axes[0, 1].set_ylabel("empirical CDF")
    axes[0, 1].set_title("How often ROI is aggressively enlarged")

    scatter = axes[1, 0].scatter(
        scales,
        coverage,
        c=areas,
        cmap="viridis",
        s=28,
        alpha=0.75,
    )
    axes[1, 0].set_xscale("log", base=2)
    axes[1, 0].set_xlabel("max axis magnification (log2)")
    axes[1, 0].set_ylabel("GT source coverage")
    axes[1, 0].set_title("Localization loss versus geometric warp")
    fig.colorbar(scatter, ax=axes[1, 0], label="ROI area fraction")

    axes[1, 1].scatter(
        [row["width"] for row in analysis],
        [row["height"] for row in analysis],
        c=anisotropy,
        cmap="magma",
        s=28,
        alpha=0.75,
    )
    axes[1, 1].plot([0, args.image_size], [0, args.image_size], "k--", lw=1)
    axes[1, 1].set_xlim(0, args.image_size)
    axes[1, 1].set_ylim(0, args.image_size)
    axes[1, 1].set_xlabel("ROI width on model grid")
    axes[1, 1].set_ylabel("ROI height on model grid")
    axes[1, 1].set_title("Non-square crops are stretched to a square")

    fig.suptitle(args.title, fontsize=16)
    fig.savefig(
        os.path.join(args.output_dir, "roi_geometry_diagnostics.png"), dpi=180
    )
    plt.close(fig)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
