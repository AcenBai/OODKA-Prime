#!/usr/bin/env python3
"""Entry point: OODKA sliding-window evaluation."""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch

from oodka.config import EvalConfig, ensure_nnunet_on_path
from oodka.models.prompts import build_text_prompts_for_dataset
from oodka.train.model_builder import load_frozen_backbones, build_fusion_modules, build_prompt_features
from oodka.eval.eval_oodka import evaluate_oodka_sliding_window


def main():
    parser = argparse.ArgumentParser(description="OODKA sliding-window evaluation")
    parser.add_argument("--dataset_name", type=str, default="Dataset009_CT_OOD")
    parser.add_argument("--nnunet_trainer_tag", type=str, default="nnUNetTrainer_500epochs__nnUNetPlans__2d")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--block_z", type=int, default=4)
    parser.add_argument("--norm_mode", type=str, default="ct", choices=("ct", "mri"))
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--distangler_ckpt", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="")
    parser.add_argument("--tile_step_size", type=float, default=0.5)
    parser.add_argument("--case_limit", type=int, default=0)
    args = parser.parse_args()

    cfg = EvalConfig(
        dataset_name=args.dataset_name,
        nnunet_trainer_tag=args.nnunet_trainer_tag,
        fold=args.fold,
        block_z=args.block_z,
        norm_mode=args.norm_mode,
        device=args.device,
        distangler_ckpt=args.distangler_ckpt,
        out_dir=args.out_dir or os.path.join("outputs", f"oodka_eval_{args.dataset_name}"),
        tile_step_size=args.tile_step_size,
        case_limit=args.case_limit,
    )
    cfg.resolve_paths()
    device = torch.device(cfg.device)

    ensure_nnunet_on_path()
    from batchgenerators.utilities.file_and_folder_operations import load_json
    from nnunetv2.utilities.plans_handling.plans_handler import PlansManager

    plans = load_json(cfg.plans_path)
    pm = PlansManager(plans)
    cm = pm.get_configuration(cfg.nnunet_configuration)
    patch_size = [cfg.block_z] + list(cm.patch_size)

    print("Loading frozen backbones...")
    model_nnunet, model_biomedparse = load_frozen_backbones(cfg.nnunet_model_dir, cfg.fold, device)

    text_prompts, prompt_to_class_id = build_text_prompts_for_dataset(dataset_name=cfg.dataset_name)
    P = len(text_prompts)

    print("Building prompt features...")
    prompt_features, class_emb = build_prompt_features(model_biomedparse, text_prompts, device)

    print("Building fusion modules...")
    fusion_modules = build_fusion_modules(model_nnunet, model_biomedparse, P, device)

    # Load checkpoint
    print(f"Loading checkpoint: {cfg.distangler_ckpt}")
    ckpt = torch.load(cfg.distangler_ckpt, map_location=device)
    for name, m in fusion_modules.items():
        if name in ckpt:
            m.load_state_dict(ckpt[name])
            print(f"  Loaded {name}")
        else:
            print(f"  WARNING: {name} not found in checkpoint")

    evaluate_oodka_sliding_window(
        cfg=cfg,
        model_nnunet=model_nnunet,
        model_biomedparse=model_biomedparse,
        fusion_modules=fusion_modules,
        prompt_features=prompt_features,
        prompt_to_class_id=prompt_to_class_id,
        P=P,
        patch_size=patch_size,
    )


if __name__ == "__main__":
    main()
