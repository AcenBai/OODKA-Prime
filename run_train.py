#!/usr/bin/env python3
"""Entry point: OODKA training."""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch

from oodka.config import TrainConfig, ensure_nnunet_on_path, ensure_biomedparse_on_path
from oodka.models.prompts import build_text_prompts_for_dataset
from oodka.train.model_builder import load_frozen_backbones, build_fusion_modules, build_prompt_features
from oodka.train.engine import OODKATrainer


def main():
    parser = argparse.ArgumentParser(description="OODKA training")
    parser.add_argument("--dataset_name", type=str, default="Dataset009_CT_OOD")
    parser.add_argument("--nnunet_trainer_tag", type=str, default="nnUNetTrainer_500epochs__nnUNetPlans__2d")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--block_z", type=int, default=4)
    parser.add_argument("--norm_mode", type=str, default="ct", choices=("ct", "mri"))
    parser.add_argument("--n_epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--output_dir", type=str, default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--w_seg", type=float, default=3.0)
    parser.add_argument("--w_ae", type=float, default=0.2)
    parser.add_argument("--w_ort", type=float, default=0.3)
    parser.add_argument("--w_ka", type=float, default=0.5)
    parser.add_argument("--val_every_epochs", type=int, default=5)
    parser.add_argument("--num_epoch_cycles", type=int, default=4)
    args = parser.parse_args()

    cfg = TrainConfig(
        dataset_name=args.dataset_name,
        nnunet_trainer_tag=args.nnunet_trainer_tag,
        fold=args.fold,
        block_z=args.block_z,
        norm_mode=args.norm_mode,
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
        output_dir=args.output_dir,
        seed=args.seed,
        w_seg=args.w_seg,
        w_ae=args.w_ae,
        w_ort=args.w_ort,
        w_ka=args.w_ka,
        val_every_epochs=args.val_every_epochs,
        num_epoch_cycles=args.num_epoch_cycles,
    )
    cfg.resolve_paths()
    device = torch.device(cfg.device)

    print("=" * 60)
    print("OODKA Training")
    print("=" * 60)
    print(f"Dataset: {cfg.dataset_name}")
    print(f"Device:  {cfg.device}")
    print(f"Output:  {cfg.output_dir}")
    print()

    print("Loading frozen backbones...")
    model_nnunet, model_biomedparse = load_frozen_backbones(cfg.nnunet_model_dir, cfg.fold, device)

    text_prompts, prompt_to_class_id = build_text_prompts_for_dataset(dataset_name=cfg.dataset_name)
    P = len(text_prompts)
    print(f"Prompts: {P} classes")

    print("Building prompt features...")
    prompt_features, class_emb = build_prompt_features(model_biomedparse, text_prompts, device)

    print("Building fusion modules...")
    fusion_modules = build_fusion_modules(model_nnunet, model_biomedparse, P, device)
    n_params = sum(p.numel() for m in fusion_modules.values() for p in m.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_params:,}")

    trainer = OODKATrainer(
        cfg=cfg,
        model_nnunet=model_nnunet,
        model_biomedparse=model_biomedparse,
        fusion_modules=fusion_modules,
        prompt_features=prompt_features,
        class_emb=class_emb,
        prompt_to_class_id=prompt_to_class_id,
        P=P,
    )
    trainer.train()


if __name__ == "__main__":
    main()
