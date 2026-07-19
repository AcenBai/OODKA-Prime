#!/usr/bin/env python3
"""One-block integration smoke test for the 2.5D training path."""

from __future__ import annotations

import argparse
import json
import os
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import torch
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate

from oodka.config import TrainConfig
from oodka.data.slice_dataset import FullSliceBlockDataset, CaseBlockBatchSampler
from oodka.models.prompts import build_text_prompts_for_dataset
from oodka.train.forward import forward_one_batch
from oodka.train.model_builder import (
    build_fusion_modules,
    build_prompt_features,
    load_frozen_backbones,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--block_z", type=int, default=2)
    parser.add_argument("--backward", action="store_true")
    parser.add_argument("--block_index", type=int, default=0)
    args = parser.parse_args()

    cfg = TrainConfig(
        batch_size=args.batch_size,
        block_z=args.block_z,
        device=args.device,
        num_workers=0,
    )
    cfg.resolve_paths()
    with open(cfg.dataset_json_path, encoding="utf-8") as f:
        dataset_json = json.load(f)
    with open(cfg.splits_final_json, encoding="utf-8") as f:
        case_id = json.load(f)[cfg.fold]["train"][0]
    dataset = FullSliceBlockDataset(
        [case_id],
        nnunet_preproc_dir=cfg.nnunet_preproc_dir,
        images_dir=cfg.imagesTr_dir,
        labels_dir=cfg.labelsTr_dir,
        file_ending=dataset_json.get("file_ending", ".nii.gz"),
        image_size=cfg.image_size,
        block_z=cfg.block_z,
        norm_mode=cfg.norm_mode,
        raw_cache_cases=1,
    )
    batch_sampler = CaseBlockBatchSampler(
        dataset, cfg.batch_size, shuffle=False, drop_last=False
    )
    if args.block_index:
        batch = default_collate([
            dataset[min(args.block_index + offset, len(dataset) - 1)]
            for offset in range(args.batch_size)
        ])
    else:
        batch = next(iter(DataLoader(dataset, batch_sampler=batch_sampler, num_workers=0)))
    print("data shapes:", {k: tuple(v.shape) for k, v in batch.items() if torch.is_tensor(v)})

    device = torch.device(cfg.device)
    print("loading frozen backbones...")
    model_nnunet, model_biomedparse = load_frozen_backbones(
        cfg.nnunet_model_dir, cfg.fold, device
    )
    text_prompts, prompt_to_class_id = build_text_prompts_for_dataset(
        dataset_name=cfg.dataset_name
    )
    prompt_features = build_prompt_features(
        model_biomedparse, text_prompts, device
    )
    P = len(text_prompts)
    fusion_modules = build_fusion_modules(
        model_nnunet, model_biomedparse, P, device
    )
    predictor_calls = 0

    def count_predictor_call(_module, _inputs):
        nonlocal predictor_calls
        predictor_calls += 1

    predictor_hook = model_biomedparse.sem_seg_head.predictor.register_forward_pre_hook(
        count_predictor_call
    )
    print("running forward...")
    loss, logs = forward_one_batch(
        batch_data=batch,
        block_shape=[cfg.block_z, cfg.image_size, cfg.image_size],
        prompt_features=prompt_features,
        P=P,
        prompt_to_class_id=prompt_to_class_id,
        w_seg=cfg.w_seg,
        w_ae=cfg.w_ae,
        w_ort=cfg.w_ort,
        w_ka=cfg.w_ka,
        model_nnunet=model_nnunet,
        model_biomedparse=model_biomedparse,
        fusion_modules=fusion_modules,
        device=device,
    )
    predictor_hook.remove()
    assert predictor_calls == 1, f"Expected one predictor call, got {predictor_calls}"
    print(f"predictor_calls={predictor_calls}")
    print("loss:", float(loss.detach()))
    print("logs:", {
        key: value for key, value in logs.items()
        if key not in {"tau_distribution", "tau_per_class_channel"}
    })
    if args.backward:
        print("running backward...")
        loss.backward()
        grads = sum(
            parameter.grad is not None
            for module in fusion_modules.values()
            for parameter in module.parameters()
        )
        print(f"backward: OK ({grads} tensors with gradients)")
    if device.type == "cuda":
        print(f"max_cuda_memory_gib={torch.cuda.max_memory_allocated(device) / 1024**3:.3f}")
    print("2.5D train block smoke test: OK")


if __name__ == "__main__":
    main()
