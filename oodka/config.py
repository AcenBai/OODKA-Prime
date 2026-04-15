"""
Centralized path resolution and default configuration for OODKA.

All paths are computed relative to OODKA_ROOT (this repo's root directory).
External dependencies (BiomedParse, nnUNet) are resolved relative to OODKA_ROOT's
parent (SegMan/) by default but can be overridden via environment variables.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List, Optional


OODKA_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEGMAN_ROOT = os.path.dirname(OODKA_ROOT)

NNUNET_CODE_DIR = os.path.join(OODKA_ROOT, "nnUNet")
BIOMEDPARSE_DIR = os.environ.get(
    "BIOMEDPARSE_DIR",
    os.path.join(SEGMAN_ROOT, "BiomedParse"),
)    # To find where the raw Biomedparse weight
BIOMEDPARSE_CKPT = os.environ.get(
    "BIOMEDPARSE_CKPT",
    os.path.join(BIOMEDPARSE_DIR, "biomedparse_v2.ckpt"),
)

NNUNET_FRAME_BASE = os.path.join(NNUNET_CODE_DIR, "nnUNetFrame", "DATASET") # Raw Data was stored in this directory
NNUNET_RAW = os.path.join(NNUNET_FRAME_BASE, "nnUNet_raw", "nnUNet_raw_data")
NNUNET_PREPROCESSED = os.path.join(NNUNET_FRAME_BASE, "nnUNet_preprocessed")
NNUNET_RESULTS = os.path.join(NNUNET_FRAME_BASE, "nnUNet_results")

OUTPUT_BASE = os.path.join(OODKA_ROOT, "outputs")


def ensure_nnunet_on_path():
    """Insert the vendored nnUNet source into sys.path (idempotent)."""
    import sys
    if NNUNET_CODE_DIR not in sys.path:
        sys.path.insert(0, NNUNET_CODE_DIR)


def ensure_biomedparse_on_path():
    """Insert BiomedParse source into sys.path (idempotent)."""
    import sys
    if BIOMEDPARSE_DIR not in sys.path:
        sys.path.insert(0, BIOMEDPARSE_DIR)


def nnunet_raw_dir(dataset_name: str) -> str:
    return os.path.join(NNUNET_RAW, dataset_name)


def nnunet_preprocessed_dir(dataset_name: str) -> str:
    return os.path.join(NNUNET_PREPROCESSED, dataset_name)


def nnunet_results_dir(dataset_name: str, trainer_tag: str) -> str:
    return os.path.join(NNUNET_RESULTS, dataset_name, trainer_tag)


def biomedparse_output_dir(dataset_name: str, split: str = "") -> str:
    """Return BiomedParse preprocessed directory, optionally under a split subfolder.

    Args:
        split: one of "", "train", "val", "test".
               Empty string returns the dataset root (backward compat).
    """
    base = os.path.join(OODKA_ROOT, "biomedparse_preprocessed", dataset_name)
    if split:
        return os.path.join(base, split)
    return base


# ---------------------------------------------------------------------------
# Training configuration dataclass
# ---------------------------------------------------------------------------

@dataclass
class TrainConfig:
    dataset_name: str = "Dataset009_CT_OOD"
    nnunet_trainer_tag: str = "nnUNetTrainer_500epochs__nnUNetPlans__2d"
    nnunet_configuration: str = "2d"
    fold: int = 0
    block_z: int = 4

    norm_mode: str = "ct"
    window_level: float = 40.0
    window_width: float = 400.0
    low_percentile: float = 1.0
    high_percentile: float = 99.0

    n_epochs: int = 100
    batch_size: int = 2
    lr: float = 1e-4
    weight_decay: float = 1e-4
    seed: int = 42

    w_seg: float = 3.0
    w_ae: float = 0.2
    w_ort: float = 0.3
    w_ka: float = 0.5
    w_p_reg: float = 0.0
    p_reg_warmup_epochs: int = 5
    p_reg_decay_epochs: int = 0

    num_epoch_cycles: int = 4
    val_every_epochs: int = 5
    tile_step_size: float = 0.5

    device: str = "cuda:0"

    # Resolved at runtime
    output_dir: str = ""

    def resolve_paths(self):
        """Fill in derived paths from dataset_name / trainer_tag."""
        self.nnunet_preproc_dir = os.path.join(
            nnunet_preprocessed_dir(self.dataset_name),
            f"nnUNetPlans_{self.nnunet_configuration}",
        )
        self.nnunet_model_dir = nnunet_results_dir(
            self.dataset_name, self.nnunet_trainer_tag
        )
        self.biomedparse_preproc_dir = biomedparse_output_dir(self.dataset_name)
        self.biomedparse_preproc_train = biomedparse_output_dir(self.dataset_name, "train")
        self.biomedparse_preproc_val = biomedparse_output_dir(self.dataset_name, "val")
        self.splits_final_json = os.path.join(
            nnunet_preprocessed_dir(self.dataset_name), "splits_final.json"
        )
        self.plans_path = os.path.join(self.nnunet_model_dir, "plans.json")
        self.dataset_json_path = os.path.join(self.nnunet_model_dir, "dataset.json")
        self.nnunet_checkpoint = os.path.join(
            self.nnunet_model_dir, f"fold_{self.fold}", "checkpoint_best.pth"
        )
        if not self.output_dir:
            self.output_dir = os.path.join(
                OUTPUT_BASE, f"oodka_{self.dataset_name}"
            )


@dataclass
class EvalConfig:
    dataset_name: str = "Dataset009_CT_OOD"
    nnunet_trainer_tag: str = "nnUNetTrainer_500epochs__nnUNetPlans__2d"
    nnunet_configuration: str = "2d"
    fold: int = 0
    block_z: int = 4

    norm_mode: str = "ct"
    window_level: float = 40.0
    window_width: float = 400.0
    low_percentile: float = 1.0
    high_percentile: float = 99.0

    tile_step_size: float = 0.5
    device: str = "cuda:0"
    case_limit: int = 0

    distangler_ckpt: str = ""
    out_dir: str = ""

    def resolve_paths(self):
        self.nnunet_preproc_dir = os.path.join(
            nnunet_preprocessed_dir(self.dataset_name),
            f"nnUNetPlans_{self.nnunet_configuration}",
        )
        self.nnunet_model_dir = nnunet_results_dir(
            self.dataset_name, self.nnunet_trainer_tag
        )
        self.biomedparse_preproc_dir = biomedparse_output_dir(self.dataset_name)
        self.biomedparse_preproc_train = biomedparse_output_dir(self.dataset_name, "train")
        self.biomedparse_preproc_val = biomedparse_output_dir(self.dataset_name, "val")
        self.biomedparse_preproc_test = biomedparse_output_dir(self.dataset_name, "test")
        self.plans_path = os.path.join(self.nnunet_model_dir, "plans.json")
        self.dataset_json_path = os.path.join(self.nnunet_model_dir, "dataset.json")
        self.nnunet_checkpoint = os.path.join(
            self.nnunet_model_dir, f"fold_{self.fold}", "checkpoint_best.pth"
        )
        raw_base = nnunet_raw_dir(self.dataset_name)
        self.imagesTs_dir = os.path.join(raw_base, "imagesTs")
        self.labelsTs_dir = os.path.join(raw_base, "labelsTs")
        self.imagesTr_dir = os.path.join(raw_base, "imagesTr")
        self.labelsTr_dir = os.path.join(raw_base, "labelsTr")
