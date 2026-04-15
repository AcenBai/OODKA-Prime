"""Patch sampling utilities for training and validation."""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from ..config import ensure_nnunet_on_path

ensure_nnunet_on_path()
from batchgenerators.utilities.file_and_folder_operations import (
    join,
    load_json,
    isfile,
    load_pickle,
)
from nnunetv2.paths import nnUNet_preprocessed
from nnunetv2.training.dataloading.nnunet_dataset import infer_dataset_class
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager


def load_fold_cases_from_splits_final(
    splits_final_json: str, fold: int = 0
) -> Tuple[List[str], List[str]]:
    splits = load_json(splits_final_json)
    if not isinstance(splits, list) or len(splits) == 0:
        raise ValueError(f"Invalid splits_final.json: {type(splits)}")
    if fold < 0 or fold >= len(splits):
        raise ValueError(f"fold={fold} out of range (n_folds={len(splits)})")
    f = splits[fold]
    train_ids = list(f.get("train", []))
    val_ids = list(f.get("val", []))
    if not train_ids:
        raise ValueError(f"No train cases in fold {fold}")
    if not val_ids:
        raise ValueError(f"No val cases in fold {fold}")
    return train_ids, val_ids


class PatchSampler:
    """Sample paired nnUNet + BiomedParse patches following nnUNet's bbox logic."""

    def __init__(
        self,
        nnunet_preproc_dir: str,
        biomedparse_preproc_dir: str,
        plans_path: str,
        dataset_json_path: str,
        patch_size: List[int],
        final_patch_size: Optional[List[int]] = None,
    ):
        self.nnunet_preproc_dir = nnunet_preproc_dir
        self.biomedparse_preproc_dir = biomedparse_preproc_dir
        self.patch_size = patch_size
        self.final_patch_size = final_patch_size or patch_size
        self.plans = load_json(plans_path)
        self.dataset_json = load_json(dataset_json_path)
        self.plans_manager = PlansManager(self.plans)
        self.label_manager = self.plans_manager.get_label_manager(self.dataset_json)
        self.nnunet_dataset_class = infer_dataset_class(nnunet_preproc_dir)
        self.nnunet_dataset = self.nnunet_dataset_class(nnunet_preproc_dir)
        self.need_to_pad = (
            np.array(patch_size) - np.array(self.final_patch_size)
        ).astype(int)

    def get_bbox(
        self,
        data_shape: np.ndarray,
        force_fg: bool,
        class_locations: Optional[Dict] = None,
        *,
        valid_bbox_lbs: Optional[List[int]] = None,
        valid_bbox_ubs: Optional[List[int]] = None,
        cycle_idx: Optional[int] = None,
        case_rank: int = 0,
        fg_round_robin_idx: int = 0,
    ) -> Tuple[List[int], List[int]]:
        need_to_pad = self.need_to_pad.copy()
        dim = len(data_shape)
        for d in range(dim):
            if need_to_pad[d] + data_shape[d] < self.patch_size[d]:
                need_to_pad[d] = self.patch_size[d] - data_shape[d]
        lbs = [-need_to_pad[i] // 2 for i in range(dim)]
        ubs = [
            data_shape[i] + need_to_pad[i] // 2 + need_to_pad[i] % 2 - self.patch_size[i]
            for i in range(dim)
        ]

        lbs_eff, ubs_eff = [], []
        for i in range(dim):
            if int(data_shape[i]) >= int(self.patch_size[i]):
                lbs_eff.append(0)
                ubs_eff.append(int(data_shape[i]) - int(self.patch_size[i]))
            else:
                lbs_eff.append(int(lbs[i]))
                ubs_eff.append(int(ubs[i]))

        if valid_bbox_lbs is not None and valid_bbox_ubs is not None:
            for i in range(dim):
                lo = max(int(lbs_eff[i]), int(valid_bbox_lbs[i]))
                hi = min(int(ubs_eff[i]), int(valid_bbox_ubs[i]) - int(self.patch_size[i]))
                if hi >= lo:
                    lbs_eff[i], ubs_eff[i] = lo, hi

        def _clamp(lbs_in):
            return [
                int(np.clip(int(lbs_in[i]), int(lbs_eff[i]), int(ubs_eff[i])))
                for i in range(dim)
            ]

        def _choose_voxel_uniform_z(vox_list):
            if vox_list is None or len(vox_list) == 0:
                return None
            vox = np.asarray(vox_list)
            if vox.ndim != 2 or vox.shape[1] < (dim + 1) or dim < 3:
                return vox[np.random.choice(len(vox))]
            z = vox[:, 1].astype(np.int64)
            uniq_z = np.unique(z)
            z_sel = uniq_z[np.random.randint(0, len(uniq_z))]
            idxs = np.where(z == z_sel)[0]
            return vox[idxs[np.random.randint(0, len(idxs))]]

        if not force_fg:
            bbox_lbs = [np.random.randint(lbs_eff[i], ubs_eff[i] + 1) for i in range(dim)]
        else:
            if not class_locations:
                bbox_lbs = [np.random.randint(lbs_eff[i], ubs_eff[i] + 1) for i in range(dim)]
            else:
                annotated_key = tuple([-1])
                eligible = [
                    k for k, v in class_locations.items()
                    if len(v) > 0 and k != annotated_key
                ]
                if not eligible:
                    bbox_lbs = [np.random.randint(lbs_eff[i], ubs_eff[i] + 1) for i in range(dim)]
                else:
                    eligible = sorted(eligible, key=str)
                    rr = int(cycle_idx or 0) + int(case_rank) + int(fg_round_robin_idx)
                    sel_cls = eligible[rr % len(eligible)]
                    voxel = _choose_voxel_uniform_z(class_locations[sel_cls])
                    if voxel is None:
                        bbox_lbs = [np.random.randint(lbs_eff[i], ubs_eff[i] + 1) for i in range(dim)]
                    else:
                        center = [int(voxel[i + 1]) - int(self.patch_size[i]) // 2 for i in range(dim)]
                        bbox_lbs = _clamp(center)

        bbox_ubs = [bbox_lbs[i] + self.patch_size[i] for i in range(dim)]
        return [int(x) for x in bbox_lbs], [int(x) for x in bbox_ubs]

    def sample_patches_from_case(
        self,
        case_id: str,
        num_samples: int = 2,
        sampling_types: Optional[List[str]] = None,
        *,
        cycle_idx: Optional[int] = None,
        case_rank: int = 0,
    ) -> List[Dict]:
        patches_info = []
        try:
            data_nnunet, seg_nnunet, _, properties = self.nnunet_dataset.load_case(case_id)
            shape = np.array(data_nnunet).shape[1:]
            class_locations = properties.get("class_locations", {})

            valid_bbox_lbs = valid_bbox_ubs = None
            try:
                seg0 = np.asarray(seg_nnunet, dtype=np.int16)[0]
                valid = seg0 != -1
                if valid.any():
                    coords = np.where(valid)
                    valid_bbox_lbs = [int(c.min()) for c in coords]
                    valid_bbox_ubs = [int(c.max()) + 1 for c in coords]
            except Exception:
                pass

            bp_file = join(self.biomedparse_preproc_dir, case_id + ".npz")
            if not isfile(bp_file):
                return []

            if sampling_types is None:
                sampling_types = (
                    ["random"] * (num_samples // 2)
                    + ["foreground"] * (num_samples - num_samples // 2)
                )

            fg_counter = 0
            for patch_idx, stype in enumerate(sampling_types):
                force_fg = stype == "foreground"
                fg_rr = fg_counter if force_fg else 0
                if force_fg:
                    fg_counter += 1

                bbox_lbs, bbox_ubs = self.get_bbox(
                    shape, force_fg, class_locations,
                    valid_bbox_lbs=valid_bbox_lbs,
                    valid_bbox_ubs=valid_bbox_ubs,
                    cycle_idx=cycle_idx,
                    case_rank=case_rank,
                    fg_round_robin_idx=fg_rr,
                )
                patches_info.append({
                    "patch_id": f"{case_id}_patch_{patch_idx:03d}",
                    "case_id": case_id,
                    "patch_idx": patch_idx,
                    "bbox_lbs": str(bbox_lbs),
                    "bbox_ubs": str(bbox_ubs),
                    "patch_size": str(self.patch_size),
                    "force_fg": force_fg,
                    "sampling_type": stype,
                })
        except Exception as e:
            import traceback
            print(f"Error sampling from {case_id}: {e}")
            traceback.print_exc()
            return []
        return patches_info

    def sample_train_patches(
        self,
        case_ids: List[str],
        num_epoch_cycles: int = 8,
        seed: Optional[int] = None,
    ) -> pd.DataFrame:
        if seed is not None:
            np.random.seed(seed)
        all_patches = []
        for cycle_idx in range(num_epoch_cycles):
            for case_rank, case_id in enumerate(
                tqdm(case_ids, desc=f"Cycle {cycle_idx+1}/{num_epoch_cycles}", leave=False)
            ):
                patches = self.sample_patches_from_case(
                    case_id, num_samples=2,
                    sampling_types=["random", "foreground"],
                    cycle_idx=cycle_idx, case_rank=case_rank,
                )
                for p in patches:
                    p["cycle_idx"] = cycle_idx
                    p["batch_idx"] = cycle_idx
                    p["patch_id"] = f"{case_id}_cycle{cycle_idx:02d}_{p['patch_idx']:d}"
                    p["split"] = "train"
                all_patches.extend(patches)
        return pd.DataFrame(all_patches)

    def sample_val_patches(
        self,
        case_ids: List[str],
        num_random: int = 4,
        num_foreground: int = 4,
        seed: int = 42,
    ) -> pd.DataFrame:
        np.random.seed(seed)
        all_patches = []
        for case_rank, case_id in enumerate(tqdm(case_ids, desc="Val patches", leave=False)):
            stypes = ["random"] * num_random + ["foreground"] * num_foreground
            patches = self.sample_patches_from_case(
                case_id,
                num_samples=len(stypes),
                sampling_types=stypes,
                case_rank=case_rank,
            )
            for p in patches:
                p["split"] = "test"
                p["cycle_idx"] = -1
                p["batch_idx"] = -1
            all_patches.extend(patches)
        return pd.DataFrame(all_patches)
