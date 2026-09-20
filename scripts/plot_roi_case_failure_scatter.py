#!/usr/bin/env python3
"""Join case metrics with ROI geometry and plot coverage-versus-Dice failures."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr


def _read_unique_rows(path: str, case_column: str) -> tuple[list[str], dict[str, dict[str, str]]]:
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or case_column not in reader.fieldnames:
            raise ValueError(f"{path} does not contain required column {case_column!r}")
        rows: dict[str, dict[str, str]] = {}
        for row in reader:
            case_id = row[case_column]
            if case_id in rows:
                raise ValueError(f"Duplicate {case_column}={case_id!r} in {path}")
            rows[case_id] = row
    return list(reader.fieldnames), rows


def _float(row: dict[str, str], column: str) -> float:
    try:
        return float(row[column])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            f"Invalid numeric value for {column!r} in case {row.get('case_id')!r}"
        ) from error


def _correlation(x: np.ndarray, y: np.ndarray) -> dict[str, float | int | None]:
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 3 or np.unique(x[valid]).size < 2 or np.unique(y[valid]).size < 2:
        return {"spearman_rho": None, "p_value": None, "n": int(valid.sum())}
    rho, p_value = spearmanr(x[valid], y[valid])
    return {
        "spearman_rho": float(rho),
        "p_value": float(p_value),
        "n": int(valid.sum()),
    }


def _quantiles(values: np.ndarray) -> dict[str, float]:
    return {
        name: float(np.quantile(values, quantile))
        for name, quantile in (
            ("min", 0.0),
            ("p05", 0.05),
            ("median", 0.5),
            ("p95", 0.95),
            ("max", 1.0),
        )
    }


def _output_paths(output: str) -> tuple[Path, Path, Path]:
    requested = Path(output)
    if requested.suffix.lower() == ".png":
        requested.parent.mkdir(parents=True, exist_ok=True)
        stem = requested.with_suffix("")
        return requested, Path(f"{stem}_merged.csv"), Path(f"{stem}_summary.json")
    requested.mkdir(parents=True, exist_ok=True)
    return (
        requested / "roi_case_failure_scatter.png",
        requested / "roi_case_failure_merged.csv",
        requested / "roi_case_failure_summary.json",
    )


def _parse_csv_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Join per-case segmentation metrics and ROI geometry, then plot "
            "coverage versus class Dice with geometric zoom encoded by color."
        )
    )
    parser.add_argument("--metrics_csv", required=True)
    parser.add_argument("--geometry_cases_csv", required=True)
    parser.add_argument(
        "--output",
        required=True,
        help="Output directory, or an explicit .png path.",
    )
    parser.add_argument("--case_column", default="case_id")
    parser.add_argument("--class_ids", default="6,7")
    parser.add_argument("--class_names", default="AO,PA")
    parser.add_argument("--dice_prefix", default="dice_")
    parser.add_argument("--coverage_column", default="source_coverage")
    parser.add_argument("--color_column", default="resize_max_scale_median")
    parser.add_argument(
        "--annotate_cases",
        default="",
        help="Comma-separated case IDs to label in both panels.",
    )
    parser.add_argument(
        "--title",
        default="ROI localization versus class failure",
    )
    args = parser.parse_args()

    class_ids = _parse_csv_list(args.class_ids)
    class_names = _parse_csv_list(args.class_names)
    if not class_ids or len(class_ids) != len(class_names):
        raise ValueError("--class_ids and --class_names must have equal non-zero length")

    metric_fields, metric_rows = _read_unique_rows(args.metrics_csv, args.case_column)
    geometry_fields, geometry_rows = _read_unique_rows(
        args.geometry_cases_csv, args.case_column
    )
    dice_columns = [f"{args.dice_prefix}{class_id}" for class_id in class_ids]
    required_metric = [args.case_column, *dice_columns]
    required_geometry = [args.case_column, args.coverage_column, args.color_column]
    for column in required_metric:
        if column not in metric_fields:
            raise ValueError(f"Missing metrics column {column!r}")
    for column in required_geometry:
        if column not in geometry_fields:
            raise ValueError(f"Missing geometry column {column!r}")

    shared_cases = sorted(set(metric_rows) & set(geometry_rows))
    if not shared_cases:
        raise ValueError("The input CSVs have no shared case IDs")

    merged_rows: list[dict[str, str]] = []
    geometry_output_fields: list[str] = []
    for field in geometry_fields:
        if field == args.case_column:
            continue
        output_field = field if field not in metric_fields else f"geometry_{field}"
        geometry_output_fields.append(output_field)
    for case_id in shared_cases:
        row = dict(metric_rows[case_id])
        for field, output_field in zip(
            (field for field in geometry_fields if field != args.case_column),
            geometry_output_fields,
        ):
            row[output_field] = geometry_rows[case_id][field]
        merged_rows.append(row)

    png_path, csv_path, json_path = _output_paths(args.output)
    merged_fields = list(metric_fields) + geometry_output_fields
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=merged_fields)
        writer.writeheader()
        writer.writerows(merged_rows)

    coverage = np.asarray(
        [_float(geometry_rows[case_id], args.coverage_column) for case_id in shared_cases]
    )
    color = np.asarray(
        [_float(geometry_rows[case_id], args.color_column) for case_id in shared_cases]
    )
    dice = {
        class_name: np.asarray(
            [_float(metric_rows[case_id], column) for case_id in shared_cases]
        )
        for class_name, column in zip(class_names, dice_columns)
    }
    requested_annotations = _parse_csv_list(args.annotate_cases)
    annotated_cases = [case_id for case_id in requested_annotations if case_id in shared_cases]
    missing_annotations = sorted(set(requested_annotations) - set(shared_cases))

    n_panels = len(class_names)
    fig, axes = plt.subplots(
        1,
        n_panels,
        figsize=(6.4 * n_panels, 5.7),
        sharex=True,
        sharey=True,
        constrained_layout=True,
        squeeze=False,
    )
    artists = []
    index_by_case = {case_id: index for index, case_id in enumerate(shared_cases)}
    for panel_index, (class_name, values) in enumerate(dice.items()):
        axis = axes[0, panel_index]
        artist = axis.scatter(
            coverage,
            values,
            c=color,
            cmap="magma",
            s=68,
            edgecolor="white",
            linewidth=0.7,
            alpha=0.9,
        )
        artists.append(artist)
        correlation = _correlation(coverage, values)
        rho = correlation["spearman_rho"]
        p_value = correlation["p_value"]
        statistic = (
            f"Spearman rho={rho:+.3f}, p={p_value:.3g}"
            if rho is not None and p_value is not None
            else "Spearman correlation unavailable"
        )
        axis.set_title(f"{class_name}: {statistic}")
        axis.set_xlabel(args.coverage_column.replace("_", " "))
        axis.grid(alpha=0.2)
        axis.axvline(0.95, color="#777777", linestyle="--", linewidth=1)
        for annotation_index, case_id in enumerate(annotated_cases):
            index = index_by_case[case_id]
            offset_x = -6 if coverage[index] >= 0.93 else 6
            vertical_offsets = (8, 16, -14, -20, 24)
            offset_y = vertical_offsets[annotation_index % len(vertical_offsets)]
            horizontal_alignment = "left" if offset_x > 0 else "right"
            axis.annotate(
                case_id.removeprefix("heart_"),
                (coverage[index], values[index]),
                xytext=(offset_x, offset_y),
                textcoords="offset points",
                ha=horizontal_alignment,
                va="bottom",
                fontsize=8,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72, "pad": 1},
            )
    axes[0, 0].set_ylabel("Dice")
    finite_coverage = coverage[np.isfinite(coverage)]
    x_min = max(0.0, float(finite_coverage.min()) - 0.04)
    for axis in axes.ravel():
        axis.set_xlim(x_min, 1.015)
        axis.set_ylim(0.0, 1.0)
    fig.colorbar(
        artists[0],
        ax=axes.ravel().tolist(),
        label=args.color_column.replace("_", " "),
        fraction=0.035,
        pad=0.02,
    )
    fig.suptitle(f"{args.title} (n={len(shared_cases)})", fontsize=15)
    fig.savefig(png_path, dpi=200, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)

    geometry_numeric_columns = [
        column
        for column in geometry_fields
        if column != args.case_column
        and all(
            _is_finite_number(geometry_rows[case_id].get(column))
            for case_id in shared_cases
        )
    ]
    correlations: dict[str, dict[str, dict[str, float | int | None]]] = {}
    for class_name, values in dice.items():
        correlations[class_name] = {}
        for column in geometry_numeric_columns:
            geometry_values = np.asarray(
                [_float(geometry_rows[case_id], column) for case_id in shared_cases]
            )
            correlations[class_name][column] = _correlation(geometry_values, values)

    annotated_values: dict[str, dict[str, Any]] = {}
    for case_id in annotated_cases:
        annotated_values[case_id] = {
            args.coverage_column: _float(geometry_rows[case_id], args.coverage_column),
            args.color_column: _float(geometry_rows[case_id], args.color_column),
            **{
                class_name: _float(metric_rows[case_id], column)
                for class_name, column in zip(class_names, dice_columns)
            },
        }
        for optional_column in (
            "resize_max_scale_p95",
            "fraction_nonfallback_over_4x",
        ):
            if optional_column in geometry_rows[case_id]:
                annotated_values[case_id][optional_column] = _float(
                    geometry_rows[case_id], optional_column
                )

    summary = {
        "inputs": {
            "metrics_csv": os.path.abspath(args.metrics_csv),
            "geometry_cases_csv": os.path.abspath(args.geometry_cases_csv),
        },
        "outputs": {
            "png": str(png_path.resolve()),
            "merged_csv": str(csv_path.resolve()),
            "summary_json": str(json_path.resolve()),
        },
        "join": {
            "n_metrics_cases": len(metric_rows),
            "n_geometry_cases": len(geometry_rows),
            "n_shared_cases": len(shared_cases),
            "metrics_only_cases": sorted(set(metric_rows) - set(geometry_rows)),
            "geometry_only_cases": sorted(set(geometry_rows) - set(metric_rows)),
        },
        "plot": {
            "class_ids": class_ids,
            "class_names": class_names,
            "dice_columns": dice_columns,
            "coverage_column": args.coverage_column,
            "color_column": args.color_column,
            "annotated_cases": annotated_cases,
            "missing_annotated_cases": missing_annotations,
        },
        "distributions": {
            args.coverage_column: _quantiles(coverage),
            args.color_column: _quantiles(color),
            **{class_name: _quantiles(values) for class_name, values in dice.items()},
        },
        "spearman_correlations": correlations,
        "annotated_case_values": annotated_values,
    }
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, allow_nan=False)

    print(json.dumps(summary, indent=2, allow_nan=False))


def _is_finite_number(value: str | None) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


if __name__ == "__main__":
    main()
