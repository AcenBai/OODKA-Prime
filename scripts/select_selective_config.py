#!/usr/bin/env python3
"""Lock selective fusion parameters using a validation sweep."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    with args.summary.open() as handle:
        summary = json.load(handle)
    candidates = summary["results"]
    if not candidates:
        raise ValueError("Validation sweep contains no candidates")
    selected = max(
        candidates,
        key=lambda row: (
            float(row["mean_dice_gt_present"]),
            float(row["confidence_threshold"]),
            float(row["ambiguity_margin"]),
            row["overwrite_scope"] == "background_children",
        ),
    )
    report = {
        "selection_source": "in_domain_validation",
        "selected": selected,
        "global_baselines": summary["global_baselines"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as handle:
        json.dump(report, handle, indent=2)
    print(
        selected["confidence_threshold"],
        selected["ambiguity_margin"],
        selected["overwrite_scope"],
        selected["postprocess"],
    )


if __name__ == "__main__":
    main()
