#!/usr/bin/env python3
"""Entry point: BiomedParse preprocessing for train+val+test."""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from oodka.config import ensure_nnunet_on_path, biomedparse_output_dir

ensure_nnunet_on_path()

from oodka.preprocess import preprocess_dataset


def main():
    parser = argparse.ArgumentParser(description="OODKA BiomedParse preprocessing")
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="")
    parser.add_argument("--configuration", type=str, default="auto")
    parser.add_argument("--plans_identifier", type=str, default="nnUNetPlans")
    parser.add_argument("--norm_mode", type=str, default="ct", choices=("ct", "mri"))
    parser.add_argument("--window_level", type=int, default=40)
    parser.add_argument("--window_width", type=int, default=400)
    parser.add_argument("--low_percentile", type=float, default=1.0)
    parser.add_argument("--high_percentile", type=float, default=99.0)
    parser.add_argument("--num_processes", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--include_test", action="store_true",
                        help="Also preprocess imagesTs + labelsTs into test/")
    parser.add_argument("--test_only", action="store_true",
                        help="Only preprocess test cases into test/")
    parser.add_argument("--fold", type=int, default=0,
                        help="Fold index in splits_final.json for train/val split")
    parser.add_argument("--splits_final_json", type=str, default="",
                        help="Path to splits_final.json (auto-detected if omitted)")
    args = parser.parse_args()

    output_dir = args.output_dir or biomedparse_output_dir(args.dataset_name)

    preprocess_dataset(
        dataset_name_or_id=args.dataset_name,
        output_dir=output_dir,
        configuration_name=args.configuration,
        plans_identifier=args.plans_identifier,
        norm_mode=args.norm_mode,
        window_level=args.window_level,
        window_width=args.window_width,
        low_percentile=args.low_percentile,
        high_percentile=args.high_percentile,
        num_processes=args.num_processes,
        overwrite_existing=args.overwrite,
        include_test=args.include_test,
        test_only=args.test_only,
        splits_final_json=args.splits_final_json or None,
        fold=args.fold,
    )


if __name__ == "__main__":
    main()
