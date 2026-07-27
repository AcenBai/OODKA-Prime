#!/usr/bin/env python3
"""Read-only smoke test for the full-slice dual-input data path."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

from torch.utils.data import DataLoader

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from oodka.config import TrainConfig
from oodka.data.slice_dataset import FullSliceBlockDataset, CaseBlockBatchSampler


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", default="Dataset009_CT_OOD")
    parser.add_argument("--case_id", default="")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--block_z", type=int, default=4)
    parser.add_argument("--all_train", action="store_true")
    args = parser.parse_args()

    cfg = TrainConfig(dataset_name=args.dataset_name)
    cfg.resolve_paths()
    with open(cfg.dataset_json_path, encoding="utf-8") as f:
        dataset_json = json.load(f)
    with open(cfg.splits_final_json, encoding="utf-8") as f:
        splits = json.load(f)
    case_id = args.case_id or splits[cfg.fold]["train"][0]
    case_ids = list(splits[cfg.fold]["train"]) if args.all_train else [case_id]

    dataset = FullSliceBlockDataset(
        case_ids,
        nnunet_preproc_dir=cfg.nnunet_preproc_dir,
        images_dir=cfg.imagesTr_dir,
        labels_dir=cfg.labelsTr_dir,
        file_ending=dataset_json.get("file_ending", ".nii.gz"),
        image_size=cfg.image_size,
        block_z=args.block_z,
        norm_mode=cfg.norm_mode,
        window_level=cfg.window_level,
        window_width=cfg.window_width,
        low_percentile=cfg.low_percentile,
        high_percentile=cfg.high_percentile,
        raw_cache_cases=cfg.raw_cache_cases,
        require_no_crop=cfg.require_no_crop,
        biomedparse_modality=cfg.biomedparse_modality,
    )
    sampler = CaseBlockBatchSampler(
        dataset, args.batch_size, shuffle=False, drop_last=False
    )
    epoch_indices = [index for indices in sampler for index in indices]
    assert len(epoch_indices) == len(dataset)
    assert sorted(epoch_indices) == list(range(len(dataset)))
    loader_iter = iter(DataLoader(dataset, batch_sampler=sampler, num_workers=0))
    start = time.perf_counter()
    batch = next(loader_iter)
    first_seconds = time.perf_counter() - start
    start = time.perf_counter()
    cached_batch = next(loader_iter)
    cached_seconds = time.perf_counter() - start
    print(f"cases={len(case_ids)} first_case={case_id} blocks={len(dataset)} "
          f"real_slices={dataset.total_real_slices}")
    print(f"case_ids={batch['case_id']}")
    print(f"z_starts={batch['z_start'].tolist()}")
    print(f"nnunet_image={tuple(batch['nnunet_image'].shape)} {batch['nnunet_image'].dtype}")
    print(f"biomedparse_image={tuple(batch['biomedparse_image'].shape)} {batch['biomedparse_image'].dtype}")
    print(f"gt={tuple(batch['gt'].shape)} {batch['gt'].dtype}")
    print(f"first_batch_seconds={first_seconds:.3f}")
    print(f"cached_batch_seconds={cached_seconds:.3f}")
    print(f"valid_z={batch['valid_z'].tolist()}")
    print(f"cached_z_starts={cached_batch['z_start'].tolist()}")
    print(f"epoch_blocks={len(epoch_indices)}/{len(dataset)} unique={len(set(epoch_indices))}")
    valid_slices = sum(dataset.records[i][2] for i in epoch_indices)
    print(f"epoch_real_slice_coverage={valid_slices}/{dataset.total_real_slices}")
    assert batch["nnunet_image"].shape[-2:] == (cfg.image_size, cfg.image_size)
    assert batch["biomedparse_image"].shape[1:] == (args.block_z, 3, cfg.image_size, cfg.image_size)
    assert batch["gt"].shape[-2:] == (cfg.image_size, cfg.image_size)
    assert valid_slices == dataset.total_real_slices
    print("slice data smoke test: OK")


if __name__ == "__main__":
    main()
