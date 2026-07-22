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
    parser.add_argument("--block_z", type=int, default=4,
                        help="Number Z of consecutive slices per block")
    parser.add_argument("--norm_mode", type=str, default="ct", choices=("ct", "mri"))
    parser.add_argument("--n_epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Number B of independent contiguous-Z blocks")
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--output_dir", type=str, default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--w_seg", type=float, default=3.0)
    parser.add_argument("--w_ae", type=float, default=0.2)
    parser.add_argument("--w_ort", type=float, default=0.3)
    parser.add_argument("--w_route", type=float, default=1e-3)
    parser.add_argument("--route_warmup_epochs", type=int, default=5)
    parser.add_argument("--w_p_ot", type=float, default=0.1)
    parser.add_argument("--w_s_ot", type=float, default=0.1)
    parser.add_argument("--p_ot_start_epoch", type=int, default=2)
    parser.add_argument("--s_ot_start_epoch", type=int, default=3)
    parser.add_argument("--ot_warmup_epochs", type=int, default=5)
    parser.add_argument("--no_amp", action="store_true")
    parser.add_argument("--resume_checkpoint", type=str, default="")
    parser.add_argument("--val_every_epochs", type=int, default=5)
    parser.add_argument("--train_case_limit", type=int, default=0)
    parser.add_argument("--val_case_limit", type=int, default=0)
    parser.add_argument("--max_train_batches", type=int, default=0)
    parser.add_argument("--max_val_batches", type=int, default=0)
    args = parser.parse_args()

    cfg = TrainConfig(
        dataset_name=args.dataset_name,
        nnunet_trainer_tag=args.nnunet_trainer_tag,
        fold=args.fold,
        block_z=args.block_z,
        norm_mode=args.norm_mode,
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
        image_size=args.image_size,
        num_workers=args.num_workers,
        lr=args.lr,
        device=args.device,
        output_dir=args.output_dir,
        seed=args.seed,
        w_seg=args.w_seg,
        w_ae=args.w_ae,
        w_ort=args.w_ort,
        w_route=args.w_route,
        route_warmup_epochs=args.route_warmup_epochs,
        w_p_ot=args.w_p_ot,
        w_s_ot=args.w_s_ot,
        p_ot_start_epoch=args.p_ot_start_epoch,
        s_ot_start_epoch=args.s_ot_start_epoch,
        ot_warmup_epochs=args.ot_warmup_epochs,
        amp=not args.no_amp,
        resume_checkpoint=args.resume_checkpoint,
        val_every_epochs=args.val_every_epochs,
        train_case_limit=args.train_case_limit,
        val_case_limit=args.val_case_limit,
        max_train_batches=args.max_train_batches,
        max_val_batches=args.max_val_batches,
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
    prompt_features = build_prompt_features(model_biomedparse, text_prompts, device)

    print("Building fusion modules...")
    text_dim = int(prompt_features["class_emb"].shape[-1])
    fusion_modules = build_fusion_modules(
        model_nnunet,
        model_biomedparse,
        P,
        device,
        text_dim=text_dim,
        route_prior_p_means=cfg.route_prior_p_means,
        route_prior_concentration=cfg.route_prior_concentration,
        ot_feature_weight=cfg.ot_feature_weight,
        ot_coordinate_weight=cfg.ot_coordinate_weight,
        p_ot_semantic_weight=cfg.p_ot_semantic_weight,
        p_ot_epsilon=cfg.p_ot_epsilon,
        s_ot_epsilon=cfg.s_ot_epsilon,
        s_ot_rho_base=cfg.s_ot_rho_base,
        s_ot_rho_expert=cfg.s_ot_rho_expert,
        ot_sinkhorn_iterations=cfg.ot_sinkhorn_iterations,
    )
    n_params = sum(p.numel() for m in fusion_modules.values() for p in m.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_params:,}")

    trainer = OODKATrainer(
        cfg=cfg,
        model_nnunet=model_nnunet,
        model_biomedparse=model_biomedparse,
        fusion_modules=fusion_modules,
        prompt_features=prompt_features,
        prompt_to_class_id=prompt_to_class_id,
        P=P,
    )
    trainer.train()


if __name__ == "__main__":
    main()
