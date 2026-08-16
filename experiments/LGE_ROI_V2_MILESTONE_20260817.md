# LGE ROI V2 milestone (2026-08-17)

This document records the reproducible code/configuration milestone for the
two-pass LGE experiment. Large checkpoints, predictions, and datasets are not
tracked by Git.

## Code

- Branch: `crop-comparison_v2`
- Core implementation commit: `dc2a907`
- Parameterized launcher commit: `c1e0aba`
- Dataset: `Dataset011_MYO_LGE_BC_OOD`
- Frozen backbones with shared OODKA fusion/decoder parameters

## Method

- Pass 1, full slice: `LV / RV / total_myo`
- ROI: threshold predicted `total_myo` at 0.3, expand bbox by 1.25
- Pass 2, ROI: `LV / RV / normal_myo / scar_edema_on_myo`
- Inference: Pass 1 outside the ROI and Pass 2 inside the ROI
- No GT ROI is used during validation or test inference
- `Z=1`, batch size 18, MRI normalization, center-repeat pseudo-RGB
- Geometry/intensity augmentation and ROI jitter are enabled during training
- Validation uses mutually exclusive hard spatial switching every 5 epochs

## Reproduction

```bash
shell/run_lge_roi_v2_full.sh \
  experiments/lge_roi_v2_z1_b18_100ep_reproduction \
  100
```

The launcher trains, selects the validation-best checkpoint, evaluates it on
validation/test, and writes the training summary plot.

## Results

All test numbers use the same 45 cases, four merged foreground classes, and
mean case macro Dice over GT-present classes.

| Model | Test macro Dice |
|---|---:|
| nnUNet merged-4 baseline | 0.5842 |
| ROI V2, 30-epoch schedule best | 0.6506 |
| ROI V2, 100-epoch schedule validation-best (epoch 30) | **0.6700** |
| ROI V2, 100-epoch schedule final epoch 100 | 0.6428 |

Validation-best 100-epoch-schedule test Dice by class:

| Class | Dice |
|---|---:|
| LV | 0.8640 |
| RV | 0.6981 |
| normal_myo | 0.6032 |
| scar_edema_on_myo | 0.5145 |

The validation-best checkpoint occurs at epoch 30 (`val=0.7305`). Training to
epoch 100 lowers the test macro, so downstream use must load
`fusion_lge_roi_v2_best.pth`, not the final epoch checkpoint.

## Local artifact locations

- 30-epoch run: `experiments/lge_roi_v2_z1_b18_30ep_rerun_20260817/`
- 100-epoch run: `experiments/lge_roi_v2_z1_b18_100ep_rerun_20260817/`
- Canonical test summary: `eval_test_best/summary.json`
- Final-epoch comparison: `eval_test_epoch100/summary.json`

These artifact paths are local references and are intentionally not part of
the Git history.
