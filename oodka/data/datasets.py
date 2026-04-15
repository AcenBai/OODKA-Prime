"""PyTorch Dataset classes for OODKA training and validation."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .loading import load_patch_from_nnunet_preproc, load_patch_from_biomedparse_preproc
from ..config import ensure_nnunet_on_path

ensure_nnunet_on_path()
from nnunetv2.training.dataloading.nnunet_dataset import infer_dataset_class


class PatchDataset(Dataset):
    """Fixed-patch dataset from a CSV (used for validation)."""

    def __init__(
        self,
        patches_df: pd.DataFrame,
        nnunet_preproc_dir: str,
        biomedparse_preproc_dir: str,
        patch_size: List[int],
    ):
        self.df = patches_df.reset_index(drop=True)
        self.nn_dir = nnunet_preproc_dir
        self.bp_dir = biomedparse_preproc_dir
        self.patch_size = patch_size

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        case_id = row["case_id"]
        bbox_lbs = eval(row["bbox_lbs"]) if isinstance(row["bbox_lbs"], str) else list(row["bbox_lbs"])
        bbox_ubs = eval(row["bbox_ubs"]) if isinstance(row["bbox_ubs"], str) else list(row["bbox_ubs"])

        nn_patch, nn_seg, _ = load_patch_from_nnunet_preproc(
            self.nn_dir, case_id, bbox_lbs, bbox_ubs, self.patch_size
        )
        bp_patch, bp_seg, _ = load_patch_from_biomedparse_preproc(
            self.bp_dir, case_id, bbox_lbs, bbox_ubs, self.patch_size
        )
        return {
            "patch_id": row["patch_id"],
            "case_id": case_id,
            "nnunet_patch": torch.from_numpy(nn_patch).float(),
            "nnunet_seg": torch.from_numpy(nn_seg).long(),
            "biomedparse_patch": torch.from_numpy(bp_patch.astype(np.float32)),
            "biomedparse_seg": torch.from_numpy(bp_seg).long() if bp_seg is not None else None,
        }


class DynamicPatchDataset(Dataset):
    """Dynamic-sampling dataset: re-samples patches each epoch."""

    def __init__(
        self,
        case_ids: List[str],
        nnunet_preproc_dir: str,
        biomedparse_preproc_dir: str,
        patch_size: List[int],
        plans_path: str,
        dataset_json_path: str,
        num_epoch_cycles: int = 8,
    ):
        from ..utils.patch_sampling import PatchSampler as _PS
        from batchgenerators.utilities.file_and_folder_operations import load_json

        self.case_ids = case_ids
        self.nn_dir = nnunet_preproc_dir
        self.bp_dir = biomedparse_preproc_dir
        self.patch_size = patch_size
        self.num_epoch_cycles = num_epoch_cycles
        self.length = num_epoch_cycles * len(case_ids) * 2

        self.nn_ds_cls = infer_dataset_class(nnunet_preproc_dir)
        self.nn_ds = self.nn_ds_cls(nnunet_preproc_dir)
        self.need_to_pad = np.zeros(len(patch_size), dtype=int)

        # Lightweight PatchSampler for bbox computation
        self._sampler = _PS(
            nnunet_preproc_dir=nnunet_preproc_dir,
            biomedparse_preproc_dir=biomedparse_preproc_dir,
            plans_path=plans_path,
            dataset_json_path=dataset_json_path,
            patch_size=patch_size,
        )

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        patches_per_cycle = len(self.case_ids) * 2
        cycle_idx = idx // patches_per_cycle
        idx_in_cycle = idx % patches_per_cycle
        case_idx = idx_in_cycle // 2
        patch_in_case = idx_in_cycle % 2
        case_id = self.case_ids[case_idx]

        force_fg = patch_in_case == 1
        data_nn, seg_nn, _, props = self.nn_ds.load_case(case_id)
        shape = np.array(data_nn).shape[1:]
        class_locs = props.get("class_locations", {})

        valid_lbs = valid_ubs = None
        try:
            seg0 = np.asarray(seg_nn, dtype=np.int16)[0]
            valid = seg0 != -1
            if valid.any():
                coords = np.where(valid)
                valid_lbs = [int(c.min()) for c in coords]
                valid_ubs = [int(c.max()) + 1 for c in coords]
        except Exception:
            pass

        bbox_lbs, bbox_ubs = self._sampler.get_bbox(
            shape, force_fg, class_locs,
            valid_bbox_lbs=valid_lbs, valid_bbox_ubs=valid_ubs,
            cycle_idx=cycle_idx, case_rank=case_idx,
        )

        nn_patch, nn_seg, _ = load_patch_from_nnunet_preproc(
            self.nn_dir, case_id, bbox_lbs, bbox_ubs, self.patch_size
        )
        bp_patch, bp_seg, _ = load_patch_from_biomedparse_preproc(
            self.bp_dir, case_id, bbox_lbs, bbox_ubs, self.patch_size
        )

        return {
            "patch_id": f"{case_id}_c{cycle_idx:02d}_p{patch_in_case}",
            "case_id": case_id,
            "nnunet_patch": torch.from_numpy(nn_patch).float(),
            "nnunet_seg": torch.from_numpy(nn_seg).long(),
            "biomedparse_patch": torch.from_numpy(bp_patch.astype(np.float32)),
            "biomedparse_seg": torch.from_numpy(bp_seg).long() if bp_seg is not None else None,
        }


def custom_collate_fn(batch):
    """Handle None fields and string fields in collation."""
    collated = {}
    for key in batch[0]:
        values = [item[key] for item in batch]
        if key in ("patch_id", "case_id"):
            collated[key] = values
        elif values[0] is None:
            collated[key] = None
        elif torch.is_tensor(values[0]):
            collated[key] = torch.stack(values, dim=0)
        else:
            collated[key] = values
    return collated
