#!/usr/bin/env python3
"""Entry point: nnUNet 2D baseline evaluation."""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from oodka.config import EvalConfig
from oodka.eval.eval_nnunet import evaluate_nnunet_2d


def main():
    parser = argparse.ArgumentParser(description="nnUNet 2D evaluation")
    parser.add_argument("--dataset_name", type=str, default="Dataset009_CT_OOD")
    parser.add_argument("--nnunet_trainer_tag", type=str, default="nnUNetTrainer_500epochs__nnUNetPlans__2d")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--out_dir", type=str, default="")
    parser.add_argument("--case_limit", type=int, default=0)
    args = parser.parse_args()

    cfg = EvalConfig(
        dataset_name=args.dataset_name,
        nnunet_trainer_tag=args.nnunet_trainer_tag,
        fold=args.fold,
        device=args.device,
        out_dir=args.out_dir or os.path.join("outputs", f"nnunet_eval_{args.dataset_name}"),
        case_limit=args.case_limit,
    )
    cfg.resolve_paths()

    evaluate_nnunet_2d(cfg)


if __name__ == "__main__":
    main()
