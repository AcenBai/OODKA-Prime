#!/usr/bin/env python3
"""Build a balanced sparse dataset for the MRI AO/PA correction gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import SimpleITK as sitk

from oodka.eval.correction_gate import (
    FEATURE_NAMES,
    build_correction_feature_context,
)


def _read_volume(path: Path) -> np.ndarray:
    return sitk.GetArrayFromImage(sitk.ReadImage(str(path)))


def _sample(
    rng: np.random.Generator,
    values: np.ndarray,
    count: int,
) -> np.ndarray:
    if len(values) <= int(count):
        return values
    return rng.choice(values, size=int(count), replace=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence_dir", type=Path, required=True)
    parser.add_argument("--labels_dir", type=Path, required=True)
    parser.add_argument("--case_ids", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--candidate_min_probability", type=float, default=0.5)
    parser.add_argument("--gv_core_threshold", type=float, default=0.9)
    parser.add_argument("--max_positive_per_case", type=int, default=30000)
    parser.add_argument("--negative_ratio", type=float, default=4.0)
    parser.add_argument("--hard_negative_fraction", type=float, default=0.75)
    parser.add_argument("--hard_negative_probability", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=20260902)
    args = parser.parse_args()
    if args.max_positive_per_case <= 0 or args.negative_ratio <= 0:
        raise ValueError("Sample counts and ratios must be positive")
    if not 0.0 <= args.hard_negative_fraction <= 1.0:
        raise ValueError("hard_negative_fraction must be in [0,1]")

    rng = np.random.default_rng(args.seed)
    case_ids = tuple(value.strip() for value in args.case_ids.split(",") if value)
    all_features = []
    all_targets = []
    all_case_indices = []
    report = []
    for case_index, case_id in enumerate(case_ids):
        evidence_path = args.evidence_dir / f"{case_id}.npz"
        with np.load(evidence_path) as evidence:
            local_logits = evidence["local_logits"]
            gv_logit = evidence["gv_logit"]
            fallback_mask = evidence["fallback_mask"]
        target = _read_volume(args.labels_dir / f"{case_id}.nii.gz")
        context = build_correction_feature_context(
            local_logits,
            gv_logit,
            fallback_mask,
            candidate_min_probability=args.candidate_min_probability,
            gv_core_threshold=args.gv_core_threshold,
        )
        candidates = context.candidate_indices()
        proposals = context.proposal_labels(candidates)
        target_at_candidates = target.ravel()[candidates]
        correct = proposals == target_at_candidates
        positive = candidates[correct]
        negative = candidates[~correct]

        selected_positive_parts = []
        positive_proposals = proposals[correct]
        per_child_cap = max(1, args.max_positive_per_case // 2)
        for child_id in (6, 7):
            selected_positive_parts.append(
                _sample(
                    rng,
                    positive[positive_proposals == child_id],
                    per_child_cap,
                )
            )
        selected_positive = np.concatenate(selected_positive_parts)
        negative_count = int(round(len(selected_positive) * args.negative_ratio))

        max_probability = np.maximum(context.p_ao, context.p_pa).ravel()
        negative_target = target.ravel()[negative]
        hard_mask = (negative_target == 0) & (
            (max_probability[negative] >= args.hard_negative_probability)
            | context.fallback_mask.ravel()[negative]
        )
        hard = negative[hard_mask]
        other = negative[~hard_mask]
        hard_count = min(
            len(hard), int(round(negative_count * args.hard_negative_fraction))
        )
        selected_hard = _sample(rng, hard, hard_count)
        selected_other = _sample(rng, other, negative_count - len(selected_hard))
        selected_negative = np.concatenate([selected_hard, selected_other])

        selected = np.concatenate([selected_positive, selected_negative])
        labels = np.concatenate(
            [
                np.ones(len(selected_positive), dtype=np.uint8),
                np.zeros(len(selected_negative), dtype=np.uint8),
            ]
        )
        order = rng.permutation(len(selected))
        selected = selected[order]
        labels = labels[order]
        all_features.append(context.features(selected).astype(np.float16))
        all_targets.append(labels)
        all_case_indices.append(np.full(len(labels), case_index, dtype=np.uint8))
        report.append(
            {
                "case_id": case_id,
                "candidate_voxels": int(len(candidates)),
                "correct_proposals": int(len(positive)),
                "incorrect_proposals": int(len(negative)),
                "sampled_positive": int(len(selected_positive)),
                "sampled_negative": int(len(selected_negative)),
                "sampled_hard_negative": int(len(selected_hard)),
            }
        )
        print(json.dumps(report[-1]))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        features=np.concatenate(all_features),
        targets=np.concatenate(all_targets),
        case_index=np.concatenate(all_case_indices),
        case_ids=np.asarray(case_ids),
        feature_names=np.asarray(FEATURE_NAMES),
    )
    with args.output.with_suffix(".json").open("w") as handle:
        json.dump(
            {
                "case_ids": case_ids,
                "feature_names": FEATURE_NAMES,
                "config": vars(args) | {"output": str(args.output)},
                "cases": report,
            },
            handle,
            indent=2,
            default=str,
        )


if __name__ == "__main__":
    main()
