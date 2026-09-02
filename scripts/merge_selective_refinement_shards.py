#!/usr/bin/env python3
"""Merge independently evaluated selective-refinement case shards."""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict

import numpy as np


CONFIG_KEYS = (
    "confidence_threshold",
    "ambiguity_margin",
    "background_confidence_threshold",
    "background_ambiguity_margin",
    "gv_write_threshold",
    "gv_write_dilation",
    "forbid_background_fallback",
    "overwrite_scope",
    "postprocess",
)
COUNT_KEYS = (
    "proposed_voxels",
    "accepted_voxels",
    "changed_voxels",
    "ambiguous_voxels",
    "ineligible_voxels",
    "background_vetoed_voxels",
    "beneficial_changes",
    "harmful_changes",
)


def _read_csv(path: str) -> list[dict[str, str]]:
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: str, rows: list[dict[str, str]]) -> None:
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def _optional_mean(rows: list[dict[str, str]], key: str) -> float | None:
    values = [float(row[key]) for row in rows if row.get(key, "") != ""]
    return float(np.mean(values)) if values else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard_dirs", nargs="+", required=True)
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()

    selective_rows: list[dict[str, str]] = []
    global_rows: list[dict[str, str]] = []
    metadata = None
    seen_cases: set[str] = set()
    for shard_dir in args.shard_dirs:
        sweep_path = os.path.join(shard_dir, "selective_sweep.csv")
        global_path = os.path.join(shard_dir, "selective_global.csv")
        summary_path = os.path.join(shard_dir, "selective_sweep_summary.json")
        shard_rows = _read_csv(sweep_path)
        shard_global_rows = _read_csv(global_path)
        shard_cases = {row["case_id"] for row in shard_global_rows}
        overlap = seen_cases & shard_cases
        if overlap:
            raise ValueError(f"Duplicate cases across shards: {sorted(overlap)}")
        seen_cases |= shard_cases
        selective_rows.extend(shard_rows)
        global_rows.extend(shard_global_rows)
        with open(summary_path) as handle:
            shard_metadata = json.load(handle)
        current = {
            "global_pred_dir": shard_metadata["global_pred_dir"],
            "child_class_ids": shard_metadata["child_class_ids"],
        }
        if metadata is None:
            metadata = current
        elif metadata != current:
            raise ValueError("Shard metadata mismatch")

    os.makedirs(args.out_dir, exist_ok=True)
    _write_csv(os.path.join(args.out_dir, "selective_sweep.csv"), selective_rows)
    _write_csv(os.path.join(args.out_dir, "selective_global.csv"), global_rows)

    dice_keys = sorted(
        (
            key
            for key in selective_rows[0]
            if key.startswith("dice_") and key[5:].isdigit()
        ),
        key=lambda value: int(value.split("_")[1]),
    )
    groups: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    for row in selective_rows:
        groups[tuple(row[key] for key in CONFIG_KEYS)].append(row)
    results = []
    for key, rows in groups.items():
        aggregate = dict(zip(CONFIG_KEYS, key))
        aggregate["confidence_threshold"] = float(aggregate["confidence_threshold"])
        aggregate["ambiguity_margin"] = float(aggregate["ambiguity_margin"])
        aggregate["background_confidence_threshold"] = float(
            aggregate["background_confidence_threshold"]
        )
        aggregate["background_ambiguity_margin"] = float(
            aggregate["background_ambiguity_margin"]
        )
        aggregate["gv_write_threshold"] = float(aggregate["gv_write_threshold"])
        aggregate["gv_write_dilation"] = int(aggregate["gv_write_dilation"])
        aggregate["forbid_background_fallback"] = (
            aggregate["forbid_background_fallback"] == "True"
        )
        aggregate["n_cases"] = len(rows)
        aggregate["mean_dice_gt_present"] = _optional_mean(rows, "dice_mean_gt")
        for count_key in COUNT_KEYS:
            aggregate[count_key] = sum(int(row[count_key]) for row in rows)
        for dice_key in dice_keys:
            aggregate[f"{dice_key}_mean"] = _optional_mean(rows, dice_key)
        results.append(aggregate)
    results.sort(key=lambda row: row["mean_dice_gt_present"], reverse=True)

    global_groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in global_rows:
        global_groups[row["postprocess"]].append(row)
    global_baselines = []
    for postprocess, rows in global_groups.items():
        aggregate = {
            "postprocess": postprocess,
            "n_cases": len(rows),
            "mean_dice_gt_present": _optional_mean(rows, "dice_mean_gt"),
        }
        for dice_key in dice_keys:
            aggregate[f"{dice_key}_mean"] = _optional_mean(rows, dice_key)
        global_baselines.append(aggregate)

    with open(
        os.path.join(args.out_dir, "selective_sweep_summary.json"), "w"
    ) as handle:
        json.dump(
            {
                **(metadata or {}),
                "n_unique_cases": len(seen_cases),
                "global_baselines": global_baselines,
                "results": results,
            },
            handle,
            indent=2,
        )


if __name__ == "__main__":
    main()
