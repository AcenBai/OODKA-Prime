# OODKA 2.5D Pipeline

OODKA combines a frozen nnUNet encoder and a frozen BiomedParse model through
trainable disentanglement, autoencoder, class-query, and gating modules.

The data path deliberately does not spatially co-preprocess BiomedParse with
nnUNet and does not sample random patches.

## Input contract

Every training and inference batch follows the same contract:

```text
nnUNet:      [B, Z, C_nn, 512, 512]
BiomedParse: [B, Z, 3,    512, 512]
valid_z:     [B, Z]
```

- `B` is the number of independent slice blocks.
- `Z` is the number of ordered, consecutive center slices inside each block.
- BiomedParse pseudo-RGB channels are raw slices `(z-1, z, z+1)`.
- nnUNet reads its existing `.b2nd` data during training.
- Both branches are resized independently to `512×512`.
- Feature maps are aligned by interpolation at fusion time.
- A tail block repeats its final center input to keep fixed `Z`; repeated
  positions are excluded with `valid_z=False`.
- Every real center slice is visited exactly once per epoch.

The 2D backbones process `B*Z` images. Their outputs are restored to
`[B,C,Z,Hf,Wf]` before the Conv3d fusion adapters.

For `P` prompts, OODKA builds class-specific gated visual features in
`[B,Z,P,C,H,W]` order and flattens them to `[B*Z*P,C,H,W]`. Prompt embeddings
are expanded in the same `[B,Z,P]` order, so all visual-prompt pairs run in a
single BiomedParse predictor call without sharing the wrong class gate.

## Prerequisites

The following must already exist:

- vendored nnUNet source under `OODKA/nnUNet/`;
- nnUNet raw dataset under its `nnUNet_raw` directory;
- nnUNet preprocessed `.b2nd` training data;
- a trained nnUNet 2D checkpoint;
- BiomedParse source and `biomedparse_v2.ckpt`.

There is no separate OODKA/BiomedParse preprocessing command or cached
BiomedParse feature bank.

## Training

```bash
bash shell/train.sh
```

Equivalent command:

```bash
python run_train.py \
  --dataset_name Dataset009_CT_OOD \
  --device cuda:0 \
  --n_epochs 100 \
  --batch_size 1 \
  --block_z 4 \
  --image_size 512 \
  --num_workers 2 \
  --norm_mode ct
```

Important arguments:

| Argument | Meaning |
|---|---|
| `--batch_size` | `B`, the number of independent Z-blocks |
| `--block_z` | `Z`, consecutive center slices per block |
| `--image_size` | independent in-plane input size for both branches |
| `--num_workers` | block-loading workers |
| `--norm_mode` | raw BiomedParse normalization: `ct` or `mri` |

With the current 24GB GPU probe, `B=1,Z=4,P=7,512×512` completes the fully
parallel prompt forward and backward with a PyTorch peak allocation of about
11.78 GiB. `B=2,Z=4` does
not fit under the observed GPU load, so `B=1` is the default.

Training uses all non-overlapping blocks for both train and validation.
Segmentation and auxiliary feature losses exclude padded tail positions.

## OODKA evaluation

```bash
CKPT=outputs/oodka_Dataset009_CT_OOD/fusion_disentangle_best.pth \
bash shell/eval_oodka.sh
```

Equivalent command:

```bash
python run_eval_oodka.py \
  --dataset_name Dataset009_CT_OOD \
  --device cuda:0 \
  --distangler_ckpt /path/to/fusion_disentangle_best.pth \
  --batch_size 1 \
  --block_z 4 \
  --image_size 512 \
  --norm_mode ct
```

For each test case:

1. BiomedParse reads the raw selected modality, normalizes it, constructs
   adjacent-slice pseudo-RGB, and resizes each slice to `512×512`.
2. nnUNet's official preprocessor runs once in memory for the nnUNet branch;
   no OODKA preprocessing files are written.
3. The case is divided into non-overlapping consecutive Z-blocks.
4. Block logits are resized per slice to the raw image H/W before argmax.
5. Blocks are assembled into the original Z order and exported with the
   reference NIfTI geometry.

The evaluator asserts that nnUNet did not crop or alter the Z count. This is
required because the two branches share center-slice indices.

Outputs are written under the configured evaluation directory:

```text
pred_nii/
metrics.csv
summary.json
```

## nnUNet baseline

The independent nnUNet baseline remains available:

```bash
bash shell/eval_nnunet.sh
```

Its `tile_step_size` option is unrelated to the OODKA block data path.

## Relevant source files

```text
oodka/data/slice_dataset.py       contiguous block construction
oodka/models/feature_extraction.py  B*Z backbone execution and 5D restoration
oodka/train/forward.py            training and block inference forward passes
oodka/train/engine.py             full-slice epoch traversal
oodka/eval/eval_oodka.py          raw-space block evaluation
run_train.py                      training entry point
run_eval_oodka.py                 OODKA evaluation entry point
```
