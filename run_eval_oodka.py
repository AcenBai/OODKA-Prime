#!/usr/bin/env python3
"""Entry point: OODKA contiguous 2.5D block evaluation."""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch

from oodka.config import EvalConfig
from oodka.models.prompts import build_text_prompts_for_dataset
from oodka.train.model_builder import load_frozen_backbones, build_fusion_modules, build_prompt_features
from oodka.eval.eval_oodka import evaluate_oodka_blocks


def main():
    parser = argparse.ArgumentParser(description="OODKA contiguous-block evaluation")
    parser.add_argument("--dataset_name", type=str, default="Dataset009_CT_OOD")
    parser.add_argument("--nnunet_trainer_tag", type=str, default="nnUNetTrainer_500epochs__nnUNetPlans__2d")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--block_z", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Number B of independent contiguous-Z blocks")
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--norm_mode", type=str, default="ct", choices=("ct", "mri"))
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--distangler_ckpt", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="")
    parser.add_argument("--case_limit", type=int, default=0)
    args = parser.parse_args()

    cfg = EvalConfig(
        dataset_name=args.dataset_name,
        nnunet_trainer_tag=args.nnunet_trainer_tag,
        fold=args.fold,
        block_z=args.block_z,
        batch_size=args.batch_size,
        image_size=args.image_size,
        norm_mode=args.norm_mode,
        device=args.device,
        distangler_ckpt=args.distangler_ckpt,
        out_dir=args.out_dir or os.path.join("outputs", f"oodka_eval_{args.dataset_name}"),
        case_limit=args.case_limit,
    )
    cfg.resolve_paths()
    device = torch.device(cfg.device)

    print("Loading frozen backbones...")
    model_nnunet, model_biomedparse = load_frozen_backbones(cfg.nnunet_model_dir, cfg.fold, device)

    text_prompts, prompt_to_class_id = build_text_prompts_for_dataset(dataset_name=cfg.dataset_name)
    P = len(text_prompts)

    print("Building prompt features...")
    prompt_features = build_prompt_features(model_biomedparse, text_prompts, device)

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

    evaluate_oodka_blocks(
        cfg=cfg,
        model_nnunet=model_nnunet,
        model_biomedparse=model_biomedparse,
        fusion_modules=fusion_modules,
        prompt_features=prompt_features,
        prompt_to_class_id=prompt_to_class_id,
        P=P,
    )


if __name__ == "__main__":
    main()
