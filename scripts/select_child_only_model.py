#!/usr/bin/env python3
"""Select a child-only ROI training run using validation Dice only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dirs", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    candidates = []
    for run_dir in args.run_dirs:
        history_path = run_dir / "history.json"
        checkpoint = run_dir / "fusion_whs_mri_gv_selective_best.pth"
        if not history_path.is_file() or not checkpoint.is_file():
            raise FileNotFoundError(
                f"Incomplete training run: {run_dir} (history/checkpoint required)"
            )
        with history_path.open() as handle:
            history = json.load(handle)
        evaluated = [row for row in history if row.get("val") is not None]
        if not evaluated:
            raise ValueError(f"No validation epochs in {history_path}")
        best = max(
            evaluated,
            key=lambda row: float(row["val"]["exclusive_macro_dice"]),
        )
        with (run_dir / "resolved_config.json").open() as handle:
            config = json.load(handle)
        candidates.append(
            {
                "run_dir": str(run_dir.resolve()),
                "checkpoint": str(checkpoint.resolve()),
                "best_epoch": int(best["epoch"]),
                "best_val_child_dice": float(best["val"]["exclusive_macro_dice"]),
                "val_dice_per_child": best["val"]["exclusive_dice_per_class"],
                "roi_threshold": float(config["roi_threshold"]),
                "roi_expand": float(config["roi_expand"]),
                "roi_fallback": config["roi_fallback"],
            }
        )
    candidates.sort(key=lambda row: row["best_val_child_dice"], reverse=True)
    report = {"selection_source": "in_domain_validation", "candidates": candidates}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as handle:
        json.dump(report, handle, indent=2)
    print(candidates[0]["checkpoint"])


if __name__ == "__main__":
    main()
