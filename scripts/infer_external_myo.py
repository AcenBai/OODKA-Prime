#!/usr/bin/env python3
"""Run pure-student OODKA inference and export binary 3D MYO masks."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import SimpleITK as sitk
import torch
from tqdm import tqdm

from oodka.config import EvalConfig
from oodka.data.slice_dataset import normalize_biomedparse_volume
from oodka.eval.eval_oodka import _block_logits_to_raw_labels, _make_block_batch
from oodka.models.prompts import build_text_prompts_for_dataset
from oodka.train.forward import predict_block_logits_per_class
from oodka.train.model_builder import (
    build_fusion_modules,
    build_prompt_features,
    load_frozen_biomedparse,
)
from oodka.utils.postprocessing import keep_largest_component_per_class


OUTPUT_SUFFIX = "_OODKA_OT30_MYO.nii.gz"
MYO_CLASS_ID = 5


def _discover_inputs(input_root: Path) -> list[Path]:
    files = [
        path
        for path in input_root.rglob("*.nii.gz")
        if not path.name.endswith(OUTPUT_SUFFIX)
    ]
    return sorted(path for path in files if path.is_file())


def _output_path(input_path: Path) -> Path:
    stem = input_path.name[:-7] if input_path.name.endswith(".nii.gz") else input_path.stem
    return input_path.with_name(stem + OUTPUT_SUFFIX)


def _predict_case(
    image_path: Path,
    *,
    cfg: EvalConfig,
    model_biomedparse,
    fusion_modules,
    prompt_features: dict,
    prompt_to_class_id: dict[int, int],
    prompt_count: int,
) -> tuple[np.ndarray, sitk.Image]:
    reference = sitk.ReadImage(str(image_path))
    raw = np.asarray(sitk.GetArrayFromImage(reference))
    if raw.ndim != 3:
        raise ValueError(f"{image_path}: expected 3D image, got {raw.shape}")

    normalized = normalize_biomedparse_volume(
        raw,
        norm_mode=cfg.norm_mode,
        window_level=cfg.window_level,
        window_width=cfg.window_width,
        low_percentile=cfg.low_percentile,
        high_percentile=cfg.high_percentile,
    )
    prediction = np.zeros(raw.shape, dtype=np.int16)
    starts = list(range(0, raw.shape[0], cfg.block_z))
    for batch_offset in tqdm(
        range(0, len(starts), cfg.batch_size),
        desc=image_path.parent.name,
        leave=False,
    ):
        batch_starts = starts[batch_offset : batch_offset + cfg.batch_size]
        bp_blocks, valid_z, valid_counts = _make_block_batch(
            normalized,
            batch_starts,
            block_z=cfg.block_z,
            image_size=cfg.image_size,
        )
        logits = predict_block_logits_per_class(
            biomedparse_images=bp_blocks,
            valid_z=valid_z,
            output_size=(cfg.image_size, cfg.image_size),
            prompt_features=prompt_features,
            P=prompt_count,
            model_biomedparse=model_biomedparse,
            fusion_modules=fusion_modules,
            device=torch.device(cfg.device),
        )
        for block_index, (z_start, valid_count) in enumerate(
            zip(batch_starts, valid_counts)
        ):
            labels = _block_logits_to_raw_labels(
                logits[block_index, :, :valid_count],
                raw.shape[1:],
                prompt_to_class_id,
            )
            prediction[z_start : z_start + valid_count] = labels

    prediction = keep_largest_component_per_class(
        prediction, sorted(prompt_to_class_id.values())
    )
    return (prediction == MYO_CLASS_ID).astype(np.uint8), reference


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--block_z", type=int, default=6)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--report",
        default="outputs/oodka_ot_experiments/external_CCTA0722_myo_report.json",
    )
    args = parser.parse_args()

    input_root = Path(args.input_root).resolve()
    checkpoint_path = Path(args.checkpoint).resolve()
    inputs = _discover_inputs(input_root)
    if not inputs:
        raise FileNotFoundError(f"No input NIfTI files under {input_root}")

    cfg = EvalConfig(
        dataset_name="Dataset009_CT_OOD",
        device=args.device,
        block_z=args.block_z,
        batch_size=args.batch_size,
        image_size=args.image_size,
        norm_mode="ct",
    )
    cfg.resolve_paths()
    device = torch.device(cfg.device)

    model_biomedparse = load_frozen_biomedparse(device)
    prompts, prompt_to_class_id = build_text_prompts_for_dataset(
        dataset_name=cfg.dataset_name
    )
    prompt_features = build_prompt_features(model_biomedparse, prompts, device)
    fusion_modules = build_fusion_modules(
        None,
        model_biomedparse,
        len(prompts),
        device,
        text_dim=int(prompt_features["class_emb"].shape[-1]),
    )
    checkpoint = torch.load(str(checkpoint_path), map_location=device)
    for name, module in fusion_modules.items():
        if name not in checkpoint:
            raise KeyError(f"{checkpoint_path}: missing module {name}")
        module.load_state_dict(checkpoint[name])
        module.eval()

    report = {
        "input_root": str(input_root),
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "myo_class_id": MYO_CLASS_ID,
        "output_suffix": OUTPUT_SUFFIX,
        "cases": [],
    }
    for image_path in inputs:
        output_path = _output_path(image_path)
        if output_path.exists() and not args.overwrite:
            raise FileExistsError(
                f"{output_path} already exists; pass --overwrite to replace it"
            )
        myo_mask, reference = _predict_case(
            image_path,
            cfg=cfg,
            model_biomedparse=model_biomedparse,
            fusion_modules=fusion_modules,
            prompt_features=prompt_features,
            prompt_to_class_id=prompt_to_class_id,
            prompt_count=len(prompts),
        )
        mask_image = sitk.GetImageFromArray(myo_mask)
        mask_image.CopyInformation(reference)
        sitk.WriteImage(mask_image, str(output_path), useCompression=True)

        spacing = reference.GetSpacing()
        voxel_volume_ml = float(np.prod(spacing) / 1000.0)
        voxel_count = int(myo_mask.sum())
        case_result = {
            "case": image_path.parent.name,
            "input": str(image_path),
            "output": str(output_path),
            "shape_zyx": [int(value) for value in myo_mask.shape],
            "spacing_xyz": [float(value) for value in spacing],
            "foreground_voxels": voxel_count,
            "foreground_volume_ml": voxel_count * voxel_volume_ml,
        }
        report["cases"].append(case_result)
        print(json.dumps(case_result, ensure_ascii=False))

    report_path = Path(args.report).resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"saved_report={report_path}")


if __name__ == "__main__":
    main()
