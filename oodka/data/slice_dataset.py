"""Contiguous full-slice-block dual-input data path for OODKA.

BiomedParse reads adjacent slices from the raw NIfTI volume, while nnUNet
reads the corresponding center slice directly from its existing .b2nd store.
Both inputs are resized to a fixed square without spatial patch sampling.
"""

from __future__ import annotations

import os
import pickle
import random
from collections import OrderedDict
from typing import Dict, Iterator, List, Sequence

import blosc2
import numpy as np
import SimpleITK as sitk
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, Sampler

from ..utils.io_utils import find_raw_image_files
from ..utils.normalization import (
    BiomedParseMRINormalization,
)


def _resize_images(x: torch.Tensor, image_size: int) -> torch.Tensor:
    """Resize an image stack shaped [N, C, H, W]."""
    return F.interpolate(
        x,
        size=(image_size, image_size),
        mode="bilinear",
        align_corners=False,
    )


def _resize_labels(x: torch.Tensor, image_size: int) -> torch.Tensor:
    """Resize a label stack shaped [N, H, W]."""
    return F.interpolate(
        x[:, None].float(),
        size=(image_size, image_size),
        mode="nearest",
    )[:, 0].long()


def normalize_biomedparse_volume(
    image: np.ndarray,
    *,
    norm_mode: str,
    window_level: float,
    window_width: float,
    low_percentile: float,
    high_percentile: float,
) -> np.ndarray:
    """Normalize one raw volume to the BiomedParse 0..255 intensity range."""
    norm_mode = str(norm_mode).lower()
    if norm_mode == "mri":
        normalizer = BiomedParseMRINormalization(low_percentile, high_percentile)
        return np.rint(normalizer.run(image)).astype(np.uint8)
    if norm_mode != "ct":
        raise ValueError(f"norm_mode must be 'ct' or 'mri', got {norm_mode!r}")

    lower = float(window_level) - float(window_width) / 2.0
    upper = float(window_level) + float(window_width) / 2.0
    if upper <= lower:
        raise ValueError(f"window_width must be positive, got {window_width}")
    work = image.astype(np.float32)
    np.clip(work, lower, upper, out=work)
    work -= lower
    work *= 255.0 / (upper - lower)
    return np.rint(work).astype(np.uint8)


def make_biomedparse_block(
    image_u8: np.ndarray,
    centers: Sequence[int],
    image_size: int,
) -> torch.Tensor:
    """Build adjacent-slice pseudo-RGB images as ``[Z,3,H,W]``."""
    block = []
    for z in centers:
        z = int(z)
        z_indices = (max(z - 1, 0), z, min(z + 1, image_u8.shape[0] - 1))
        block.append(np.stack([image_u8[i] for i in z_indices]))
    return _resize_images(torch.from_numpy(np.stack(block)).float(), image_size)


class FullSliceBlockDataset(Dataset):
    """Return aligned contiguous Z blocks for 2.5D OODKA training.

    Each record is one non-overlapping block of ``block_z`` center slices from
    a single case. Every center slice becomes pseudo-RGB ``z-1, z, z+1``.
    Tail blocks replicate their last center slice for tensor shape stability;
    replicated positions have ``valid_z=False`` and target value ``-1``.
    """

    def __init__(
        self,
        case_ids: Sequence[str],
        *,
        nnunet_preproc_dir: str,
        images_dir: str,
        labels_dir: str,
        file_ending: str = ".nii.gz",
        image_size: int = 512,
        block_z: int = 4,
        norm_mode: str = "ct",
        window_level: float = 40.0,
        window_width: float = 400.0,
        low_percentile: float = 1.0,
        high_percentile: float = 99.0,
        raw_cache_cases: int = 2,
        require_no_crop: bool = True,
        biomedparse_modality: int = 0,
    ):
        self.case_ids = [str(x) for x in case_ids]
        self.nnunet_preproc_dir = str(nnunet_preproc_dir)
        self.images_dir = str(images_dir)
        self.labels_dir = str(labels_dir)
        self.file_ending = str(file_ending)
        self.image_size = int(image_size)
        self.block_z = int(block_z)
        self.norm_mode = str(norm_mode).lower()
        self.window_level = float(window_level)
        self.window_width = float(window_width)
        self.low_percentile = float(low_percentile)
        self.high_percentile = float(high_percentile)
        self.raw_cache_cases = max(1, int(raw_cache_cases))
        self.require_no_crop = bool(require_no_crop)
        self.biomedparse_modality = int(biomedparse_modality)

        if self.image_size <= 0:
            raise ValueError(f"image_size must be positive, got {self.image_size}")
        if self.block_z <= 0:
            raise ValueError(f"block_z must be positive, got {self.block_z}")
        if self.norm_mode not in {"ct", "mri"}:
            raise ValueError(f"norm_mode must be 'ct' or 'mri', got {self.norm_mode!r}")
        if not self.case_ids:
            raise ValueError("FullSliceBlockDataset needs at least one case")

        self._case_info: Dict[str, dict] = {}
        self.records: List[tuple[str, int, int]] = []
        self.case_to_indices: Dict[str, List[int]] = {}
        self.total_real_slices = 0
        self._raw_cache: OrderedDict[str, tuple[np.ndarray, np.ndarray]] = OrderedDict()
        self._b2nd_cache: OrderedDict[str, object] = OrderedDict()
        self._index_cases()

    def _index_cases(self) -> None:
        for case_id in self.case_ids:
            image_files = find_raw_image_files(self.images_dir, case_id, self.file_ending)
            if not image_files:
                raise FileNotFoundError(f"No raw images found for {case_id} in {self.images_dir}")
            if not 0 <= self.biomedparse_modality < len(image_files):
                raise IndexError(
                    f"{case_id}: biomedparse_modality={self.biomedparse_modality}, "
                    f"but only {len(image_files)} modalities were found"
                )
            label_path = os.path.join(self.labels_dir, case_id + self.file_ending)
            b2nd_path = os.path.join(self.nnunet_preproc_dir, case_id + ".b2nd")
            props_path = os.path.join(self.nnunet_preproc_dir, case_id + ".pkl")
            for path in (label_path, b2nd_path, props_path):
                if not os.path.isfile(path):
                    raise FileNotFoundError(path)

            raw_size_xyz = sitk.ReadImage(image_files[self.biomedparse_modality]).GetSize()
            label_size_xyz = sitk.ReadImage(label_path).GetSize()
            raw_shape = tuple(int(x) for x in reversed(raw_size_xyz))
            label_shape = tuple(int(x) for x in reversed(label_size_xyz))
            if raw_shape != label_shape:
                raise ValueError(f"{case_id}: raw shape {raw_shape} != label shape {label_shape}")

            with open(props_path, "rb") as f:
                props = pickle.load(f)
            shape_before_crop = tuple(int(x) for x in props.get("shape_before_cropping", raw_shape))
            bbox = props.get("bbox_used_for_cropping")
            expected_bbox = [[0, int(x)] for x in shape_before_crop]
            if self.require_no_crop and bbox != expected_bbox:
                raise ValueError(
                    f"{case_id}: nnUNet crop {bbox} does not cover full shape "
                    f"{shape_before_crop}; full-slice index alignment is unsafe"
                )
            if shape_before_crop != raw_shape:
                raise ValueError(
                    f"{case_id}: nnUNet shape_before_cropping={shape_before_crop} "
                    f"!= raw shape={raw_shape}"
                )

            nn_array = blosc2.open(
                urlpath=b2nd_path,
                mode="r",
                dparams={"nthreads": 1},
            )
            nn_shape = tuple(int(x) for x in nn_array.shape)
            if len(nn_shape) != 4:
                raise ValueError(f"{case_id}: expected .b2nd [C,Z,H,W], got {nn_shape}")
            if nn_shape[1] != raw_shape[0]:
                raise ValueError(
                    f"{case_id}: raw Z={raw_shape[0]} != nnUNet Z={nn_shape[1]}"
                )

            self._case_info[case_id] = {
                "image_path": image_files[self.biomedparse_modality],
                "label_path": label_path,
                "b2nd_path": b2nd_path,
                "raw_shape": raw_shape,
                "nn_shape": nn_shape,
            }
            indices = []
            for z_start in range(0, raw_shape[0], self.block_z):
                valid_count = min(self.block_z, raw_shape[0] - z_start)
                indices.append(len(self.records))
                self.records.append((case_id, z_start, valid_count))
                self.total_real_slices += valid_count
            self.case_to_indices[case_id] = indices

    def __getstate__(self):
        state = self.__dict__.copy()
        # Blosc2 handles and cached volumes are opened independently per worker.
        state["_raw_cache"] = OrderedDict()
        state["_b2nd_cache"] = OrderedDict()
        return state

    def _normalize_to_u8(self, image: np.ndarray) -> np.ndarray:
        return normalize_biomedparse_volume(
            image,
            norm_mode=self.norm_mode,
            window_level=self.window_level,
            window_width=self.window_width,
            low_percentile=self.low_percentile,
            high_percentile=self.high_percentile,
        )

    def _load_raw_case(self, case_id: str) -> tuple[np.ndarray, np.ndarray]:
        cached = self._raw_cache.pop(case_id, None)
        if cached is not None:
            self._raw_cache[case_id] = cached
            return cached

        info = self._case_info[case_id]
        image = sitk.GetArrayFromImage(sitk.ReadImage(info["image_path"]))
        label = sitk.GetArrayFromImage(sitk.ReadImage(info["label_path"]))
        if tuple(image.shape) != info["raw_shape"] or image.shape != label.shape:
            raise ValueError(
                f"{case_id}: volume geometry changed after indexing: "
                f"image={image.shape}, label={label.shape}"
            )
        bp_u8 = self._normalize_to_u8(image)
        cached = (bp_u8, np.asarray(label, dtype=np.int16))
        self._raw_cache[case_id] = cached
        while len(self._raw_cache) > self.raw_cache_cases:
            self._raw_cache.popitem(last=False)
        return cached

    def _open_b2nd(self, case_id: str):
        cached = self._b2nd_cache.pop(case_id, None)
        if cached is not None:
            self._b2nd_cache[case_id] = cached
            return cached
        arr = blosc2.open(
            urlpath=self._case_info[case_id]["b2nd_path"],
            mode="r",
            dparams={"nthreads": 1},
        )
        self._b2nd_cache[case_id] = arr
        while len(self._b2nd_cache) > self.raw_cache_cases:
            self._b2nd_cache.popitem(last=False)
        return arr

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict:
        case_id, z_start, valid_count = self.records[int(index)]
        image, label = self._load_raw_case(case_id)

        valid_centers = list(range(z_start, z_start + valid_count))
        centers = valid_centers + [valid_centers[-1]] * (self.block_z - valid_count)
        bp_images = make_biomedparse_block(image, centers, self.image_size)

        nn_array = self._open_b2nd(case_id)
        nn_valid = np.asarray(
            nn_array[:, z_start : z_start + valid_count, :, :], dtype=np.float32
        ).transpose(1, 0, 2, 3)
        if valid_count < self.block_z:
            nn_pad = np.repeat(nn_valid[-1:], self.block_z - valid_count, axis=0)
            nn_valid = np.concatenate([nn_valid, nn_pad], axis=0)
        nn_images = _resize_images(torch.from_numpy(nn_valid.copy()), self.image_size)

        gt_valid = label[z_start : z_start + valid_count].astype(np.int64)
        gt = _resize_labels(torch.from_numpy(gt_valid), self.image_size)
        if valid_count < self.block_z:
            gt_pad = torch.full(
                (self.block_z - valid_count, self.image_size, self.image_size),
                -1,
                dtype=torch.long,
            )
            gt = torch.cat([gt, gt_pad], dim=0)
        valid_z = torch.arange(self.block_z) < valid_count

        return {
            "case_id": case_id,
            "z_start": int(z_start),
            "nnunet_image": nn_images,
            "biomedparse_image": bp_images,
            "gt": gt,
            "valid_z": valid_z,
        }


class CaseBlockBatchSampler(Sampler[List[int]]):
    """Yield batches of independent contiguous-Z block records.

    ``batch_size`` is B (number of independent blocks); ``dataset.block_z`` is
    Z (number of consecutive slices inside each block). Every record appears
    exactly once per epoch when ``drop_last=False``.
    """

    def __init__(
        self,
        dataset: FullSliceBlockDataset,
        batch_size: int,
        *,
        shuffle: bool = True,
        drop_last: bool = False,
        seed: int = 0,
    ):
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.drop_last = bool(drop_last)
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _batches(self) -> List[List[int]]:
        batches: List[List[int]] = []
        rng = random.Random(self.seed + self.epoch)
        case_ids = list(self.dataset.case_ids)
        if self.shuffle:
            rng.shuffle(case_ids)

        # Keep each shuffled group within the Dataset's per-worker LRU budget.
        # This avoids globally shuffled batches repeatedly re-reading NIfTI
        # volumes while still mixing slices from more than one case.
        group_size = max(1, self.dataset.raw_cache_cases)
        for group_start in range(0, len(case_ids), group_size):
            group_indices: List[int] = []
            for case_id in case_ids[group_start : group_start + group_size]:
                group_indices.extend(self.dataset.case_to_indices[case_id])
            if self.shuffle:
                rng.shuffle(group_indices)
            for start in range(0, len(group_indices), self.batch_size):
                batch = group_indices[start : start + self.batch_size]
                if len(batch) == self.batch_size or not self.drop_last:
                    batches.append(batch)
        return batches

    def __iter__(self) -> Iterator[List[int]]:
        yield from self._batches()

    def __len__(self) -> int:
        return len(self._batches())
