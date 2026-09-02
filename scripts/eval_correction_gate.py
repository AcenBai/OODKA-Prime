#!/usr/bin/env python3
"""Evaluate learned protected correction gates from saved raw-space evidence."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import torch

from oodka.eval.correction_gate import (
    ProtectedFusionMetricContext,
    build_correction_feature_context,
    load_gate_checkpoint,
    predict_gate_probabilities,
)
from oodka.eval.selective_refinement import dilate_mask_in_plane
from oodka.utils.metrics import dice_no_ignore


def _read_volume(path: Path) -> np.ndarray:
    return sitk.GetArrayFromImage(sitk.ReadImage(str(path)))


def _float_grid(value: str) -> tuple[float, ...]:
    values = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    if not values or any(not 0.0 < item < 1.0 for item in values):
        raise ValueError("thresholds must be a non-empty list in (0,1)")
    return values


def _optional_mean(rows: list[dict], key: str) -> float | None:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    return float(np.mean(values)) if values else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence_dir", type=Path, required=True)
    parser.add_argument("--global_pred_dir", type=Path, required=True)
    parser.add_argument("--labels_dir", type=Path, required=True)
    parser.add_argument("--gate_checkpoints", type=Path, nargs="+", required=True)
    parser.add_argument("--thresholds", default="0.3,0.5,0.7,0.8,0.9,0.95")
    parser.add_argument(
        "--fallback_modes",
        default="allow,forbid_background",
        help="Comma-separated allow and/or forbid_background.",
    )
    parser.add_argument(
        "--proposal_modes",
        default="gate_only",
        help="Comma-separated gate_only and/or gv_veto_intersection.",
    )
    parser.add_argument("--case_ids", default="")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=262144)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument(
        "--save_predictions",
        action="store_true",
        help="Save fused NIfTIs; requires exactly one gate and fusion setting.",
    )
    args = parser.parse_args()
    thresholds = _float_grid(args.thresholds)
    fallback_modes = tuple(
        item.strip() for item in args.fallback_modes.split(",") if item.strip()
    )
    if not fallback_modes or any(
        item not in {"allow", "forbid_background"} for item in fallback_modes
    ):
        raise ValueError("fallback_modes supports allow,forbid_background")
    proposal_modes = tuple(
        item.strip() for item in args.proposal_modes.split(",") if item.strip()
    )
    if not proposal_modes or any(
        item not in {"gate_only", "gv_veto_intersection"} for item in proposal_modes
    ):
        raise ValueError("proposal_modes supports gate_only,gv_veto_intersection")
    if args.save_predictions and (
        len(args.gate_checkpoints) != 1
        or len(thresholds) != 1
        or len(fallback_modes) != 1
        or len(proposal_modes) != 1
    ):
        raise ValueError("save_predictions requires one gate, threshold, and mode")
    case_ids = (
        tuple(item.strip() for item in args.case_ids.split(",") if item.strip())
        if args.case_ids.strip()
        else tuple(path.stem for path in sorted(args.evidence_dir.glob("*.npz")))
    )
    if not case_ids:
        raise ValueError("No evidence cases found")
    device = torch.device(args.device)
    gates = []
    settings = set()
    for checkpoint_path in args.gate_checkpoints:
        model, checkpoint = load_gate_checkpoint(str(checkpoint_path), device)
        settings.add(
            (
                float(checkpoint["candidate_min_probability"]),
                float(checkpoint["gv_core_threshold"]),
            )
        )
        gates.append((checkpoint_path.stem, model, checkpoint))
    if len(settings) != 1:
        raise ValueError("All gates must use the same feature-context settings")
    candidate_min_probability, gv_core_threshold = next(iter(settings))

    rows = []
    global_rows = []
    class_ids = tuple(range(1, 8))
    prediction_dir = args.output_dir / "pred_raw"
    if args.save_predictions:
        prediction_dir.mkdir(parents=True, exist_ok=True)
    for case_id in case_ids:
        with np.load(args.evidence_dir / f"{case_id}.npz") as evidence:
            local_logits = evidence["local_logits"]
            gv_logit = evidence["gv_logit"]
            fallback_mask = evidence["fallback_mask"].astype(bool)
        global_labels = _read_volume(args.global_pred_dir / f"{case_id}.nii.gz")
        target = _read_volume(args.labels_dir / f"{case_id}.nii.gz")
        context = build_correction_feature_context(
            local_logits,
            gv_logit,
            fallback_mask,
            candidate_min_probability=candidate_min_probability,
            gv_core_threshold=gv_core_threshold,
        )
        indices = context.candidate_indices(global_labels)
        proposals = context.proposal_labels(indices)
        local_ao = context.p_ao.ravel()[indices]
        local_pa = context.p_pa.ravel()[indices]
        winner_probability = np.maximum(local_ao, local_pa)
        local_margin = np.abs(local_ao - local_pa)
        global_at_candidates = global_labels.ravel()[indices]
        background = global_at_candidates == 0
        gv_write_mask = dilate_mask_in_plane(context.p_gv >= 0.7, 5)
        gv_veto_accept = (winner_probability >= 0.9) & (local_margin >= 0.1)
        gv_veto_accept &= ~background | (
            (winner_probability >= 0.95)
            & (local_margin >= 0.2)
            & gv_write_mask.ravel()[indices]
            & ~fallback_mask.ravel()[indices]
        )
        global_dice, global_mean, _ = dice_no_ignore(global_labels, target, class_ids)
        metric_context = ProtectedFusionMetricContext.from_arrays(
            global_labels,
            target,
            class_ids,
        )
        global_row = {"case_id": case_id, "dice_mean_gt": global_mean}
        for class_id in class_ids:
            global_row[f"dice_{class_id}"] = global_dice.get(class_id)
        global_rows.append(global_row)

        for gate_name, model, _ in gates:
            gate_probability = predict_gate_probabilities(
                model,
                context,
                indices,
                device=device,
                batch_size=args.batch_size,
            )
            for threshold in thresholds:
                threshold_accept = gate_probability >= threshold
                for proposal_mode in proposal_modes:
                    proposal_accept = (
                        gv_veto_accept
                        if proposal_mode == "gv_veto_intersection"
                        else np.ones(len(indices), dtype=bool)
                    )
                    for fallback_mode in fallback_modes:
                        accept = threshold_accept & proposal_accept
                        if fallback_mode == "forbid_background":
                            background_fallback = (
                                background & fallback_mask.ravel()[indices]
                            )
                            accept &= ~background_fallback
                        metrics = metric_context.evaluate(
                            indices,
                            proposals,
                            accept,
                        )
                        row = {
                            "case_id": case_id,
                            "gate": gate_name,
                            "threshold": threshold,
                            "proposal_mode": proposal_mode,
                            "fallback_mode": fallback_mode,
                            "dice_mean_gt": metrics.mean_dice_gt_present,
                            "candidate_voxels": int(len(indices)),
                            "accepted_voxels": int(accept.sum()),
                            "changed_voxels": metrics.changed_voxels,
                            "beneficial_changes": metrics.beneficial_changes,
                            "harmful_changes": metrics.harmful_changes,
                        }
                        for class_id in class_ids:
                            row[f"dice_{class_id}"] = metrics.dice_per_class.get(
                                class_id
                            )
                        rows.append(row)
                        if args.save_predictions:
                            fused = global_labels.copy()
                            fused.ravel()[indices[accept]] = proposals[accept]
                            output = sitk.GetImageFromArray(fused.astype(np.int16))
                            output.CopyInformation(
                                sitk.ReadImage(
                                    str(args.global_pred_dir / f"{case_id}.nii.gz")
                                )
                            )
                            sitk.WriteImage(
                                output,
                                str(prediction_dir / f"{case_id}.nii.gz"),
                            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for filename, values in (
        ("correction_gate.csv", rows),
        ("global.csv", global_rows),
    ):
        with (args.output_dir / filename).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=values[0].keys())
            writer.writeheader()
            writer.writerows(values)

    grouped: dict[tuple[str, float, str, str], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[
            (
                row["gate"],
                row["threshold"],
                row["proposal_mode"],
                row["fallback_mode"],
            )
        ].append(row)
    results = []
    for (gate_name, threshold, proposal_mode, fallback_mode), group in grouped.items():
        result = {
            "gate": gate_name,
            "threshold": threshold,
            "proposal_mode": proposal_mode,
            "fallback_mode": fallback_mode,
            "n_cases": len(group),
            "mean_dice_gt_present": _optional_mean(group, "dice_mean_gt"),
            "candidate_voxels": sum(row["candidate_voxels"] for row in group),
            "accepted_voxels": sum(row["accepted_voxels"] for row in group),
            "changed_voxels": sum(row["changed_voxels"] for row in group),
            "beneficial_changes": sum(row["beneficial_changes"] for row in group),
            "harmful_changes": sum(row["harmful_changes"] for row in group),
        }
        for class_id in class_ids:
            result[f"dice_{class_id}_mean"] = _optional_mean(group, f"dice_{class_id}")
        results.append(result)
    results.sort(key=lambda row: row["mean_dice_gt_present"], reverse=True)
    global_summary = {
        "n_cases": len(global_rows),
        "mean_dice_gt_present": _optional_mean(global_rows, "dice_mean_gt"),
    }
    for class_id in class_ids:
        global_summary[f"dice_{class_id}_mean"] = _optional_mean(
            global_rows, f"dice_{class_id}"
        )
    with (args.output_dir / "summary.json").open("w") as handle:
        json.dump(
            {
                "selection_source": "provided_split",
                "global": global_summary,
                "results": results,
            },
            handle,
            indent=2,
        )
    print(json.dumps({"global": global_summary, "best": results[0]}, indent=2))


if __name__ == "__main__":
    main()
