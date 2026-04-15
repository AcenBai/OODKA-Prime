# OODKA: Out-of-Distribution Knowledge Amalgamation — Pipeline Reference

```
OODKA/
├── oodka/                          # Core Python package
│   ├── config.py                   # Centralized paths & hyperparameter dataclasses
│   ├── preprocess/
│   │   └── biomedparse_preprocessor.py   # Offline & online BiomedParse preprocessing
│   ├── models/
│   │   ├── disentangle.py          # TwoBranchDisentangle (private/shared split)
│   │   ├── gate.py                 # GateNet (per-class, per-channel gating τ)
│   │   ├── feature_extraction.py   # Frozen backbone feature extraction
│   │   ├── biomedparse_helpers.py  # BiomedParse interaction utilities
│   │   ├── losses.py               # Seg, AE, Orthogonal, CKA, τ-entropy losses
│   │   └── prompts.py              # Dataset-specific text prompts
│   ├── data/
│   │   ├── loading.py              # Low-level patch I/O (.b2nd / .npz)
│   │   └── datasets.py            # PyTorch Dataset & collate_fn
│   ├── train/
│   │   ├── model_builder.py        # Load frozen backbones & build fusion modules
│   │   ├── forward.py              # forward_one_batch & predict_patch_logits_per_class
│   │   └── engine.py               # OODKATrainer loop
│   ├── eval/
│   │   ├── eval_nnunet.py          # nnUNet 2D baseline evaluation
│   │   └── eval_oodka.py           # OODKA sliding-window evaluation
│   └── utils/
│       ├── io_utils.py             # NIfTI I/O, case discovery, directory helpers
│       ├── metrics.py              # Dice, Precision, Recall, HD95
│       ├── normalization.py        # CT (WL/WW) and MRI (percentile) normalizers
│       ├── patch_sampling.py       # nnUNet-style random + foreground-aware sampling
│       ├── postprocessing.py       # Connected-component refinement utilities
│       └── visualization.py        # Training curve plotting
├── run_preprocess.py               # Entry point: BiomedParse preprocessing
├── run_train.py                    # Entry point: OODKA training
├── run_eval_nnunet.py              # Entry point: nnUNet baseline evaluation
├── run_eval_oodka.py               # Entry point: OODKA evaluation
├── shell/                          # One-click shell wrappers
│   ├── preprocess.sh
│   ├── train.sh
│   ├── eval_nnunet.sh
│   └── eval_oodka.sh
├── nnUNet/                         # Vendored nnUNet source (not modified)
├── biomedparse_preprocessed/       # Generated: offline preprocessed data
└── outputs/                        # Generated: checkpoints, logs, predictions
```

## Prerequisites

Before running the OODKA pipeline, the following must already be in place:

| Dependency | Default Location | Override |
|---|---|---|
| nnUNet source (vendored) | `OODKA/nnUNet/` | — |
| BiomedParse source | `../BiomedParse/` | `BIOMEDPARSE_DIR` env var |
| BiomedParse checkpoint | `../BiomedParse/biomedparse_v2.ckpt` | `BIOMEDPARSE_CKPT` env var |
| nnUNet raw data | `OODKA/nnUNet/nnUNetFrame/DATASET/nnUNet_raw/nnUNet_raw_data/<Dataset>/` | — |
| nnUNet preprocessed data | `OODKA/nnUNet/nnUNetFrame/DATASET/nnUNet_preprocessed/<Dataset>/` | — |
| nnUNet trained model | `OODKA/nnUNet/nnUNetFrame/DATASET/nnUNet_results/<Dataset>/<Trainer>/` | — |

> **Note:** nnUNet training and its own preprocessing must be completed beforehand. OODKA operates on top of a pre-trained nnUNet model.

## Pipeline Overview

```
┌──────────────────────────────────────────────────────────────┐
│  Stage 1: BiomedParse Preprocessing (offline, run once)      │
│  run_preprocess.py → biomedparse_preprocessed/{train,val,test}│
└──────────────────────┬───────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────┐
│  Stage 2: OODKA Training                                     │
│  run_train.py → outputs/oodka_<Dataset>/                     │
│    • Frozen nnUNet encoder + Frozen BiomedParse               │
│    • Trainable fusion modules (Disentangle, AE, Gate, Pooler)│
│    • Validates on val/ split every N epochs                  │
│    • Saves fusion_disentangle_best.pth                       │
└──────────────────────┬───────────────────────────────────────┘
                       │
          ┌────────────┴────────────┐
          ▼                         ▼
┌───────────────────┐   ┌──────────────────────────────────────┐
│ Stage 3a: nnUNet  │   │ Stage 3b: OODKA Evaluation            │
│ Baseline Eval     │   │ run_eval_oodka.py                     │
│ run_eval_nnunet.py│   │   • Online dual-branch preprocessing  │
│                   │   │   • Sliding-window + Gaussian blend   │
└───────────────────┘   └──────────────────────────────────────┘
```

Stages 3a and 3b are independent and can run in parallel.

---

## Stage 1: BiomedParse Preprocessing

**Purpose:** Convert raw NIfTI volumes into BiomedParse-compatible format (crop, resample, custom normalization) and organize them into train/val/test splits.

### Command

```bash
bash shell/preprocess.sh
```

Or equivalently:

```bash
python run_preprocess.py \
  --dataset_name Dataset009_CT_OOD \
  --norm_mode ct \
  --num_processes 8 \
  --include_test
```

### Key Arguments

| Argument | Default | Description |
|---|---|---|
| `--dataset_name` | (required) | nnUNet dataset identifier |
| `--norm_mode` | `ct` | `ct` for WL/WW normalization; `mri` for percentile normalization |
| `--window_level` | `40` | CT window level (only used when `norm_mode=ct`) |
| `--window_width` | `400` | CT window width (only used when `norm_mode=ct`) |
| `--fold` | `0` | Which fold in `splits_final.json` defines train/val |
| `--include_test` | off | Also preprocess `imagesTs/labelsTs` into `test/` |
| `--test_only` | off | Only preprocess test cases |
| `--num_processes` | `8` | Parallel workers |
| `--overwrite` | off | Overwrite existing preprocessed files |

### Processing Steps

1. Read nnUNet's `splits_final.json` to determine train/val split for the given fold.
2. For each case, reuse nnUNet's `DefaultPreprocessor` for spatial operations (cropping to non-zero region, resampling to target spacing).
3. Replace the normalization step with a custom normalizer:
   - **CT mode:** Apply window level/width, then scale to \[0, 255\].
   - **MRI mode:** Clip to \[low_percentile, high_percentile\], then scale to \[0, 255\].
4. Save each case as `.npz` (data + segmentation arrays) and `.pkl` (metadata).

### Output

```
OODKA/biomedparse_preprocessed/Dataset009_CT_OOD/
├── train/          # Training cases (.npz + .pkl)
├── val/            # Validation cases (.npz + .pkl)
└── test/           # Test cases (.npz + .pkl), if --include_test
```

---

## Stage 2: OODKA Training

**Purpose:** Train the lightweight fusion modules that bridge nnUNet and BiomedParse features, while keeping both backbone encoders frozen.

### Command

```bash
bash shell/train.sh
```

Or equivalently:

```bash
python run_train.py \
  --dataset_name Dataset009_CT_OOD \
  --device cuda:0 \
  --n_epochs 100 \
  --batch_size 2 \
  --block_z 4 \
  --norm_mode ct
```

### Key Arguments

| Argument | Default | Description |
|---|---|---|
| `--dataset_name` | `Dataset009_CT_OOD` | nnUNet dataset identifier |
| `--n_epochs` | `100` | Total training epochs |
| `--batch_size` | `2` | Batch size |
| `--lr` | `1e-4` | Learning rate |
| `--block_z` | `4` | Number of slices stacked as the z-dimension of each 3D patch |
| `--w_seg` | `3.0` | Segmentation loss weight |
| `--w_ae` | `0.2` | Autoencoder reconstruction loss weight |
| `--w_ort` | `0.3` | Orthogonal correlation loss weight |
| `--w_ka` | `0.5` | Spatial CKA loss weight |
| `--val_every_epochs` | `5` | Validate every N epochs |
| `--device` | `cuda:0` | GPU device |

### Training Pipeline

1. **Load frozen backbones:**
   - nnUNet encoder from `checkpoint_best.pth`
   - BiomedParse from `biomedparse_v2.ckpt`
2. **Build text prompt features:** Encode dataset-specific class names (e.g., "left ventricle", "myocardium") via BiomedParse's text encoder.
3. **Build trainable fusion modules:**
   - `TwoBranchDisentangle` — 1×1 conv splitting features into private/shared components
   - `DualBranchAutoEncoder` — Channel alignment, disentanglement, and reconstruction
   - `ClassQueryPooler` — Cross-attention from nnUNet features to class queries
   - `GateNet` — Generates per-class, per-channel gating values τ
4. **Patch sampling:** From the offline-preprocessed train split, use `PatchSampler` (random + foreground-aware) to generate 3D patches of size `(block_z, H, W)`.
5. **Forward pass per batch:**
   - Extract features from both frozen backbones
   - Pass through fusion modules
   - Compute multi-objective loss: Segmentation (BCE + Dice) + AE reconstruction + Orthogonal correlation + Spatial CKA + τ entropy regularization
6. **Validation:** Every `val_every_epochs`, run sliding-window inference on the val split and report mean Dice.
7. **Checkpoint:** Save the best model (by validation Dice) as `fusion_disentangle_best.pth`.

### Output

```
OODKA/outputs/oodka_Dataset009_CT_OOD/
├── fusion_disentangle_best.pth     # Best checkpoint
├── training_log.json               # Per-epoch metrics
└── ...
```

---

## Stage 3a: nnUNet Baseline Evaluation

**Purpose:** Run the standalone nnUNet 2D model on test cases to establish a baseline.

### Command

```bash
bash shell/eval_nnunet.sh
```

Or equivalently:

```bash
python run_eval_nnunet.py \
  --dataset_name Dataset009_CT_OOD \
  --device cuda:0
```

### Key Arguments

| Argument | Default | Description |
|---|---|---|
| `--dataset_name` | `Dataset009_CT_OOD` | nnUNet dataset identifier |
| `--fold` | `0` | Model fold |
| `--device` | `cuda:0` | GPU device |
| `--case_limit` | `0` | Limit number of test cases (0 = all) |

### Evaluation Pipeline

1. Initialize `nnUNetPredictor` from the trained model folder.
2. For each test case in `imagesTs/`:
   - Read raw image(s), run `predict_single_npy_array`
   - `argmax` to produce the final segmentation
   - Export prediction as NIfTI
   - Compute per-class Dice, Precision, Recall, and HD95 against ground truth in `labelsTs/`
3. Save results.

### Output

```
OODKA/outputs/nnunet_eval_Dataset009_CT_OOD/
├── predictions/        # Predicted .nii.gz files
├── metrics.csv         # Per-case, per-class metrics
└── summary.json        # Aggregated metrics
```

---

## Stage 3b: OODKA Evaluation

**Purpose:** Run the full OODKA fusion model (frozen backbones + trained fusion modules) with sliding-window inference on test cases.

### Command

```bash
CKPT=outputs/oodka_Dataset009_CT_OOD/fusion_disentangle_best.pth \
bash shell/eval_oodka.sh
```

Or equivalently:

```bash
python run_eval_oodka.py \
  --dataset_name Dataset009_CT_OOD \
  --device cuda:0 \
  --block_z 4 \
  --norm_mode ct \
  --distangler_ckpt outputs/oodka_Dataset009_CT_OOD/fusion_disentangle_best.pth
```

### Key Arguments

| Argument | Default | Description |
|---|---|---|
| `--dataset_name` | `Dataset009_CT_OOD` | nnUNet dataset identifier |
| `--distangler_ckpt` | (required) | Path to trained fusion checkpoint |
| `--block_z` | `4` | z-dimension of 3D patch |
| `--norm_mode` | `ct` | Normalization mode |
| `--tile_step_size` | `0.5` | Sliding-window step as fraction of patch size |
| `--device` | `cuda:0` | GPU device |
| `--case_limit` | `0` | Limit number of test cases (0 = all) |

### Evaluation Pipeline

1. Load frozen backbones and restore fusion module weights from checkpoint.
2. For each test case:
   - **Online preprocessing:** Run both nnUNet and BiomedParse preprocessing on-the-fly via `preprocess_case_online` (no dependency on offline-cached test data).
   - **Sliding-window inference:** Iterate over overlapping 3D patches, forward through both branches and fusion modules, accumulate logits with Gaussian blending.
   - **Segmentation assembly:** `argmax` over fused logits, then map prompt indices back to original class IDs.
   - Compute per-class Dice, Precision, Recall, and HD95.
3. Save results.

### Output

```
OODKA/outputs/oodka_eval_Dataset009_CT_OOD/
├── pred_nii/           # Predicted .nii.gz files
├── metrics.csv         # Per-case, per-class metrics
└── summary.json        # Aggregated metrics
```

---

## Quick-Start Summary

```bash
# Step 1 — Preprocess BiomedParse data (train + val + test)
bash shell/preprocess.sh

# Step 2 — Train fusion modules
bash shell/train.sh

# Step 3a — Evaluate nnUNet baseline (independent, can run in parallel with 3b)
bash shell/eval_nnunet.sh

# Step 3b — Evaluate OODKA fusion model
CKPT=outputs/oodka_Dataset009_CT_OOD/fusion_disentangle_best.pth \
bash shell/eval_oodka.sh
```
