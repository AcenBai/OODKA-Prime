"""
Centralized path resolution and default configuration for OODKA.

All paths are computed relative to OODKA_ROOT (this repo's root directory).
External dependencies (BiomedParse, nnUNet) are resolved relative to OODKA_ROOT's
parent (SegMan/) by default but can be overridden via environment variables.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Tuple


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

    # Contiguous full-slice-block dual-input data path.
    image_size: int = 512
    raw_cache_cases: int = 2
    require_no_crop: bool = True
    biomedparse_modality: int = 0

    norm_mode: str = "ct"
    window_level: float = 40.0
    window_width: float = 400.0
    low_percentile: float = 1.0
    high_percentile: float = 99.0

    n_epochs: int = 100
    batch_size: int = 1
    lr: float = 1e-4
    weight_decay: float = 1e-4
    seed: int = 42

    w_seg: float = 3.0
    w_ae: float = 0.2
    w_ort: float = 0.3
    w_route: float = 1e-3
    route_warmup_epochs: int = 5
    # P prior means ordered as res2,res3,res4,res5; S always equals 1-P.
    route_prior_p_means: Tuple[float, float, float, float] = (0.5, 0.6, 0.7, 0.8)
    route_prior_concentration: float = 10.0

    w_p_ot: float = 0.1
    w_s_ot: float = 0.1
    p_ot_start_epoch: int = 2
    s_ot_start_epoch: int = 3
    ot_warmup_epochs: int = 5
    ot_sinkhorn_iterations: int = 30
    ot_feature_weight: float = 1.0
    ot_coordinate_weight: float = 0.1
    p_ot_semantic_weight: float = 0.25
    p_ot_epsilon: float = 0.1
    s_ot_epsilon: float = 0.1
    s_ot_rho_base: float = 1.0
    s_ot_rho_expert: float = 0.2

    amp: bool = True
    amp_dtype: str = "float16"
    resume_checkpoint: str = ""

    num_workers: int = 2
    val_every_epochs: int = 5
    train_case_limit: int = 0
    val_case_limit: int = 0
    max_train_batches: int = 0
    max_val_batches: int = 0

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
        self.splits_final_json = os.path.join(
            nnunet_preprocessed_dir(self.dataset_name), "splits_final.json"
        )
        self.plans_path = os.path.join(self.nnunet_model_dir, "plans.json")
        self.nnunet_checkpoint = os.path.join(
            self.nnunet_model_dir, f"fold_{self.fold}", "checkpoint_best.pth"
        )
        raw_base = nnunet_raw_dir(self.dataset_name)
        self.dataset_json_path = os.path.join(raw_base, "dataset.json")
        self.imagesTr_dir = os.path.join(raw_base, "imagesTr")
        self.labelsTr_dir = os.path.join(raw_base, "labelsTr")
        self.imagesTs_dir = os.path.join(raw_base, "imagesTs")
        self.labelsTs_dir = os.path.join(raw_base, "labelsTs")
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
    batch_size: int = 1
    image_size: int = 512
    require_no_crop: bool = True
    biomedparse_modality: int = 0

    norm_mode: str = "ct"
    window_level: float = 40.0
    window_width: float = 400.0
    low_percentile: float = 1.0
    high_percentile: float = 99.0

    tile_step_size: float = 0.5
    device: str = "cuda:0"
    case_limit: int = 0
    split: str = "test"

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
        self.plans_path = os.path.join(self.nnunet_model_dir, "plans.json")
        self.nnunet_checkpoint = os.path.join(
            self.nnunet_model_dir, f"fold_{self.fold}", "checkpoint_best.pth"
        )
        raw_base = nnunet_raw_dir(self.dataset_name)
        self.splits_final_json = os.path.join(
            nnunet_preprocessed_dir(self.dataset_name), "splits_final.json"
        )
        self.dataset_json_path = os.path.join(raw_base, "dataset.json")
        self.imagesTs_dir = os.path.join(raw_base, "imagesTs")
        self.labelsTs_dir = os.path.join(raw_base, "labelsTs")
        self.imagesTr_dir = os.path.join(raw_base, "imagesTr")
        self.labelsTr_dir = os.path.join(raw_base, "labelsTr")
