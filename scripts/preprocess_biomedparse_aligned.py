#!/usr/bin/env python3
"""Create an nnUNet-geometry-aligned BiomedParse MRI/LGE store."""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys

import numpy as np
from tqdm import tqdm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from oodka.config import TrainConfig
from oodka.data.aligned_preprocessing import AlignedBiomedParsePreprocessor
from oodka.utils.io_utils import (
    discover_case_ids_from_dir,
    find_raw_image_files,
    maybe_mkdir_p,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Preprocess MRI/LGE for BiomedParse with the frozen nnUNet "
            "expert's crop/resample geometry"
        )
    )
    parser.add_argument(
        "--dataset_name",
        default="Dataset011_MYO_LGE_BC_OOD",
    )
    parser.add_argument("--configuration", default="2d")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--low_percentile", type=float, default=1.0)
    parser.add_argument("--high_percentile", type=float, default=99.0)
    parser.add_argument("--modality", type=int, default=0)
    parser.add_argument(
        "--split",
        choices=("train", "test", "all"),
        default="train",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    cfg = TrainConfig(
        dataset_name=args.dataset_name,
        nnunet_configuration=args.configuration,
    )
    cfg.resolve_paths()
    with open(cfg.dataset_json_path, encoding="utf-8") as file_handle:
        dataset_json = json.load(file_handle)
    file_ending = dataset_json.get("file_ending", ".nii.gz")

    preprocessor = AlignedBiomedParsePreprocessor(
        plans_path=cfg.plans_path,
        dataset_json_path=cfg.dataset_json_path,
        configuration_name=args.configuration,
        low_percentile=args.low_percentile,
        high_percentile=args.high_percentile,
    )
    maybe_mkdir_p(args.output_dir)

    split_dirs = []
    if args.split in ("train", "all"):
        split_dirs.append(("train", cfg.imagesTr_dir, cfg.labelsTr_dir))
    if args.split in ("test", "all"):
        split_dirs.append(("test", cfg.imagesTs_dir, cfg.labelsTs_dir))

    jobs = []
    for split_name, images_dir, labels_dir in split_dirs:
        case_ids = discover_case_ids_from_dir(labels_dir, file_ending)
        for case_id in case_ids:
            jobs.append((split_name, case_id, images_dir, labels_dir))
    if not jobs:
        raise FileNotFoundError(
            f"No labeled cases found for requested split={args.split}"
        )

    for split_name, case_id, images_dir, labels_dir in tqdm(
        jobs, desc="Aligned BiomedParse preprocessing"
    ):
        output_npz = os.path.join(args.output_dir, case_id + ".npz")
        output_pkl = os.path.join(args.output_dir, case_id + ".pkl")
        if (
            not args.overwrite
            and os.path.isfile(output_npz)
            and os.path.isfile(output_pkl)
        ):
            continue
        image_files = find_raw_image_files(images_dir, case_id, file_ending)
        label_path = os.path.join(labels_dir, case_id + file_ending)
        bp_u8, seg, properties = preprocessor.run_case(
            image_files,
            label_path,
            modality=args.modality,
        )
        if seg is None:
            raise RuntimeError(f"{split_name}/{case_id}: label was not loaded")
        np.savez_compressed(
            output_npz,
            data=bp_u8[None],
            seg=seg[None],
        )
        with open(output_pkl, "wb") as file_handle:
            pickle.dump(properties, file_handle)

    print(f"Saved {len(jobs)} cases to {args.output_dir}")


if __name__ == "__main__":
    main()
