#!/usr/bin/env python3
"""One-block integration smoke test for raw-space OODKA evaluation."""

from __future__ import annotations

import argparse
import os
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np
import SimpleITK as sitk
import torch

from oodka.config import EvalConfig
from oodka.data.slice_dataset import normalize_biomedparse_volume
from oodka.eval.eval_oodka import (
    _block_logits_to_raw_labels,
    _make_block_batch,
)
from oodka.models.prompts import build_text_prompts_for_dataset
from oodka.train.forward import predict_block_logits_per_class
from oodka.train.model_builder import (
    build_fusion_modules,
    build_prompt_features,
    load_frozen_biomedparse,
)
from oodka.utils.io_utils import find_raw_image_files


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--block_z", type=int, default=4)
    parser.add_argument("--block_index", type=int, default=-1)
    parser.add_argument("--case_id", default="heart_2001")
    parser.add_argument("--ckpt", default="")
    args = parser.parse_args()

    cfg = EvalConfig(device=args.device, block_z=args.block_z)
    cfg.resolve_paths()
    import json
    with open(cfg.dataset_json_path, encoding="utf-8") as file_handle:
        dataset_json = json.load(file_handle)
    file_ending = dataset_json.get("file_ending", ".nii.gz")
    image_files = find_raw_image_files(cfg.imagesTs_dir, args.case_id, file_ending)
    raw_image = np.asarray(sitk.GetArrayFromImage(sitk.ReadImage(image_files[0])))
    raw_shape = tuple(int(value) for value in raw_image.shape)
    bp_u8 = normalize_biomedparse_volume(
        raw_image,
        norm_mode=cfg.norm_mode,
        window_level=cfg.window_level,
        window_width=cfg.window_width,
        low_percentile=cfg.low_percentile,
        high_percentile=cfg.high_percentile,
    )
    starts = list(range(0, raw_shape[0], cfg.block_z))
    start = starts[args.block_index]
    bp_blocks, valid_z, valid_counts = _make_block_batch(
        bp_u8,
        [start],
        block_z=cfg.block_z,
        image_size=cfg.image_size,
    )
    print(
        f"case={args.case_id} raw_shape={raw_shape} "
        f"z_start={start} valid_count={valid_counts[0]}"
    )
    print(
        f"bp={tuple(bp_blocks.shape)} "
        f"valid_z={valid_z.tolist()}"
    )

    device = torch.device(cfg.device)
    model_biomedparse = load_frozen_biomedparse(device)
    text_prompts, prompt_to_class_id = build_text_prompts_for_dataset(
        dataset_name=cfg.dataset_name
    )
    prompt_features = build_prompt_features(model_biomedparse, text_prompts, device)
    checkpoint = (
        torch.load(args.ckpt, map_location=device) if args.ckpt else None
    )
    checkpoint_cfg = checkpoint.get("config", {}) if checkpoint else {}
    use_query_guided_injection = bool(
        checkpoint_cfg.get("use_query_guided_injection", not args.ckpt)
    )
    use_beta_router = bool(
        checkpoint_cfg.get(
            "use_beta_router", not use_query_guided_injection
        )
    )
    fusion_modules = build_fusion_modules(
        None,
        model_biomedparse,
        len(text_prompts),
        device,
        text_dim=int(prompt_features["class_emb"].shape[-1]),
        use_beta_router=use_beta_router,
    )
    if checkpoint is not None:
        for name, module in fusion_modules.items():
            module.load_state_dict(checkpoint[name])
    for module in fusion_modules.values():
        module.eval()

    logits = predict_block_logits_per_class(
        biomedparse_images=bp_blocks,
        valid_z=valid_z,
        output_size=(cfg.image_size, cfg.image_size),
        prompt_features=prompt_features,
        P=len(text_prompts),
        model_biomedparse=model_biomedparse,
        fusion_modules=fusion_modules,
        device=device,
        use_query_guided_injection=use_query_guided_injection,
        query_guided_s_floor=float(
            checkpoint_cfg.get("query_guided_s_floor", 0.2)
        ),
        query_guided_topk=int(
            checkpoint_cfg.get("query_guided_topk", 4)
        ),
        use_beta_router=use_beta_router,
    )
    raw_labels = _block_logits_to_raw_labels(
        logits[0, :, : valid_counts[0]], raw_shape[1:], prompt_to_class_id
    )
    assert logits.shape == (
        1,
        len(text_prompts),
        cfg.block_z,
        cfg.image_size,
        cfg.image_size,
    )
    assert raw_labels.shape == (valid_counts[0], raw_shape[1], raw_shape[2])
    print(
        f"logits={tuple(logits.shape)} raw_labels={raw_labels.shape} "
        f"predictor_calls={predictor_calls}"
    )
    if device.type == "cuda":
        print(
            f"max_cuda_memory_gib="
            f"{torch.cuda.max_memory_allocated(device) / 1024**3:.3f}"
        )
    print("2.5D eval block smoke test: OK")


if __name__ == "__main__":
    main()
