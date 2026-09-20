#!/usr/bin/env python3
"""Strict, paired comparison of per-case segmentation metric CSV files.

Example:
    python scripts/compare_paired_segmentation_runs.py \
        --baseline experiments/baseline/metrics.csv \
        --run oracle_resize=experiments/oracle_resize/metrics.csv \
        --run oracle_pad=experiments/oracle_pad/metrics.csv \
        --output-dir experiments/mri_roi_comparison
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np


_CLASS_METRIC_RE = re.compile(r"^dice_(\d+)$")
_RUN_NAME_RE = re.compile(r"^[A-Za-z0-9_.-]+$")


@dataclass(frozen=True)
class MetricsTable:
    """The columns needed for a paired Dice comparison."""

    path: Path
    case_ids: tuple[str, ...]
    metric: str
    class_metrics: tuple[str, ...]
    values: dict[str, dict[str, float]]


def _metric_sort_key(column: str) -> int:
    match = _CLASS_METRIC_RE.fullmatch(column)
    if match is None:
        raise ValueError(f"Unsupported class metric column: {column!r}")
    return int(match.group(1))


def _parse_float(
    raw: str | None,
    *,
    path: Path,
    case_id: str,
    column: str,
    allow_missing: bool,
) -> float:
    if allow_missing and (raw is None or not raw.strip()):
        return float("nan")
    try:
        return float(raw)  # type: ignore[arg-type]
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"Invalid numeric value for {column!r} in case {case_id!r} at {path}: {raw!r}"
        ) from error


def read_metrics(path: str | Path, metric: str = "dice_mean_gt") -> MetricsTable:
    """Read one metrics CSV and reject duplicate/empty case identifiers."""

    csv_path = Path(path)
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Metrics CSV has no header: {csv_path}")
        if "case_id" not in reader.fieldnames:
            raise ValueError(f"Missing required column 'case_id' in {csv_path}")
        if metric not in reader.fieldnames:
            raise ValueError(f"Missing required metric column {metric!r} in {csv_path}")

        class_metrics = tuple(
            sorted(
                (column for column in reader.fieldnames if _CLASS_METRIC_RE.fullmatch(column)),
                key=_metric_sort_key,
            )
        )
        if not class_metrics:
            raise ValueError(f"No per-class Dice columns (dice_1, dice_2, ...) in {csv_path}")

        case_ids: list[str] = []
        values = {column: {} for column in (metric, *class_metrics)}
        for row_number, row in enumerate(reader, start=2):
            case_id = (row.get("case_id") or "").strip()
            if not case_id:
                raise ValueError(f"Empty case_id at row {row_number} in {csv_path}")
            if case_id in values[metric]:
                raise ValueError(f"Duplicate case_id={case_id!r} in {csv_path}")
            case_ids.append(case_id)
            for column in values:
                values[column][case_id] = _parse_float(
                    row.get(column),
                    path=csv_path,
                    case_id=case_id,
                    column=column,
                    allow_missing=column != metric,
                )

    if not case_ids:
        raise ValueError(f"Metrics CSV has no data rows: {csv_path}")
    return MetricsTable(
        path=csv_path.resolve(),
        case_ids=tuple(case_ids),
        metric=metric,
        class_metrics=class_metrics,
        values=values,
    )


def _strictly_validate_join(baseline: MetricsTable, run: MetricsTable, name: str) -> None:
    baseline_cases = set(baseline.case_ids)
    run_cases = set(run.case_ids)
    if baseline_cases != run_cases:
        missing = sorted(baseline_cases - run_cases)
        unexpected = sorted(run_cases - baseline_cases)
        raise ValueError(
            f"Run {name!r} does not exactly match baseline case_ids; "
            f"missing={missing}, unexpected={unexpected}"
        )
    if baseline.class_metrics != run.class_metrics:
        raise ValueError(
            f"Run {name!r} class Dice columns do not match baseline; "
            f"baseline={list(baseline.class_metrics)}, run={list(run.class_metrics)}"
        )


def _finite_mean(values: np.ndarray) -> tuple[float | None, int]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return None, 0
    return float(finite.mean()), int(finite.size)


def _summarize_table(table: MetricsTable, case_ids: Sequence[str]) -> dict[str, object]:
    overall = np.asarray([table.values[table.metric][case_id] for case_id in case_ids])
    if not np.all(np.isfinite(overall)):
        bad_cases = [
            case_id for case_id, value in zip(case_ids, overall) if not np.isfinite(value)
        ]
        raise ValueError(
            f"Non-finite {table.metric!r} values in {table.path} for cases {bad_cases}"
        )

    per_class_mean: dict[str, float | None] = {}
    per_class_valid_cases: dict[str, int] = {}
    for column in table.class_metrics:
        class_values = np.asarray([table.values[column][case_id] for case_id in case_ids])
        mean, count = _finite_mean(class_values)
        per_class_mean[column] = mean
        per_class_valid_cases[column] = count
    return {
        "path": str(table.path),
        "overall_mean": float(overall.mean()),
        "per_class_mean": per_class_mean,
        "per_class_valid_cases": per_class_valid_cases,
    }


def paired_bootstrap_ci(
    deltas: np.ndarray,
    *,
    samples: int,
    seed: int,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Percentile CI from case-level paired bootstrap resampling."""

    if samples <= 0:
        raise ValueError("bootstrap samples must be positive")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between 0 and 1")
    if deltas.ndim != 1 or deltas.size == 0 or not np.all(np.isfinite(deltas)):
        raise ValueError("paired bootstrap requires a non-empty finite 1D delta array")

    rng = np.random.default_rng(seed)
    bootstrap_means = np.empty(samples, dtype=np.float64)
    # Bound the temporary index matrix while retaining vectorized sampling.
    batch_size = max(1, min(samples, 1_000_000 // deltas.size))
    for start in range(0, samples, batch_size):
        stop = min(start + batch_size, samples)
        indices = rng.integers(0, deltas.size, size=(stop - start, deltas.size))
        bootstrap_means[start:stop] = deltas[indices].mean(axis=1)

    alpha = (1.0 - confidence) / 2.0
    lower, upper = np.quantile(bootstrap_means, [alpha, 1.0 - alpha])
    return float(lower), float(upper)


def compare_paired_runs(
    baseline_path: str | Path,
    named_run_paths: Sequence[tuple[str, str | Path]],
    *,
    baseline_name: str = "baseline",
    metric: str = "dice_mean_gt",
    bootstrap_samples: int = 10_000,
    seed: int = 20260920,
    tie_tolerance: float = 1e-12,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    """Compare runs and return the JSON summary plus per-case output rows."""

    if not _RUN_NAME_RE.fullmatch(baseline_name):
        raise ValueError(f"Invalid baseline name: {baseline_name!r}")
    if not named_run_paths:
        raise ValueError("At least one named run is required")
    if tie_tolerance < 0.0:
        raise ValueError("tie_tolerance must be non-negative")

    names = [name for name, _ in named_run_paths]
    if any(not _RUN_NAME_RE.fullmatch(name) for name in names):
        invalid = [name for name in names if not _RUN_NAME_RE.fullmatch(name)]
        raise ValueError(f"Invalid run names: {invalid}; use letters, numbers, '.', '_' or '-'")
    if len(names) != len(set(names)):
        raise ValueError(f"Run names must be unique: {names}")
    if baseline_name in names:
        raise ValueError(f"Run name {baseline_name!r} conflicts with the baseline name")

    baseline = read_metrics(baseline_path, metric)
    case_ids = baseline.case_ids
    baseline_values = np.asarray(
        [baseline.values[metric][case_id] for case_id in case_ids], dtype=np.float64
    )
    baseline_summary = _summarize_table(baseline, case_ids)

    per_case_rows: list[dict[str, object]] = [
        {
            "case_id": case_id,
            f"{baseline_name}__{metric}": float(baseline.values[metric][case_id]),
        }
        for case_id in case_ids
    ]
    run_summaries: dict[str, object] = {}

    for name, path in named_run_paths:
        run = read_metrics(path, metric)
        _strictly_validate_join(baseline, run, name)
        run_summary = _summarize_table(run, case_ids)
        run_values = np.asarray(
            [run.values[metric][case_id] for case_id in case_ids], dtype=np.float64
        )
        deltas = run_values - baseline_values
        ci_low, ci_high = paired_bootstrap_ci(
            deltas, samples=bootstrap_samples, seed=seed
        )
        wins = int(np.count_nonzero(deltas > tie_tolerance))
        losses = int(np.count_nonzero(deltas < -tie_tolerance))
        ties = int(deltas.size - wins - losses)
        run_summary["comparison_to_baseline"] = {
            "paired_mean_delta": float(deltas.mean()),
            "bootstrap_95_ci": [ci_low, ci_high],
            "wins": wins,
            "ties": ties,
            "losses": losses,
            "tie_tolerance": tie_tolerance,
        }
        run_summaries[name] = run_summary

        for row, case_id, value, delta in zip(
            per_case_rows, case_ids, run_values, deltas
        ):
            row[f"{name}__{metric}"] = float(value)
            row[f"{name}__delta_vs_{baseline_name}"] = float(delta)

    summary: dict[str, object] = {
        "schema_version": 1,
        "metric": metric,
        "case_count": len(case_ids),
        "class_metrics": list(baseline.class_metrics),
        "baseline_name": baseline_name,
        "baseline": baseline_summary,
        "runs": run_summaries,
        "bootstrap": {
            "method": "case-level paired percentile bootstrap",
            "samples": bootstrap_samples,
            "seed": seed,
            "confidence": 0.95,
            "shared_resamples_across_runs": True,
        },
    }
    return summary, per_case_rows


def write_comparison_outputs(
    output_dir: str | Path,
    summary: dict[str, object],
    per_case_rows: Sequence[dict[str, object]],
) -> tuple[Path, Path]:
    """Write ``summary.json`` and ``per_case_deltas.csv``."""

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    summary_path = destination / "summary.json"
    per_case_path = destination / "per_case_deltas.csv"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    if not per_case_rows:
        raise ValueError("Cannot write an empty per-case comparison")
    with per_case_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_case_rows[0]))
        writer.writeheader()
        writer.writerows(per_case_rows)
    return summary_path, per_case_path


def parse_named_run(value: str) -> tuple[str, str]:
    """Parse ``NAME=metrics.csv`` without restricting '=' in the path."""

    if "=" not in value:
        raise argparse.ArgumentTypeError("run must use NAME=PATH syntax")
    name, path = value.split("=", 1)
    name = name.strip()
    path = path.strip()
    if not name or not path:
        raise argparse.ArgumentTypeError("run must have a non-empty NAME and PATH")
    return name, path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, help="Baseline metrics.csv path")
    parser.add_argument("--baseline-name", default="baseline")
    parser.add_argument(
        "--run",
        action="append",
        type=parse_named_run,
        required=True,
        metavar="NAME=PATH",
        help="Named metrics.csv to compare; repeat for multiple runs",
    )
    parser.add_argument("--output-dir", "--out_dir", required=True)
    parser.add_argument("--metric", default="dice_mean_gt")
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260920)
    parser.add_argument("--tie-tolerance", type=float, default=1e-12)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    summary, per_case_rows = compare_paired_runs(
        args.baseline,
        args.run,
        baseline_name=args.baseline_name,
        metric=args.metric,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
        tie_tolerance=args.tie_tolerance,
    )
    summary_path, per_case_path = write_comparison_outputs(
        args.output_dir, summary, per_case_rows
    )
    print(f"Wrote {summary_path}")
    print(f"Wrote {per_case_path}")


if __name__ == "__main__":
    main()
