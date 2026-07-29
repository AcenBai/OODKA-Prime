# Mechanism visualization v3

This directory contains the canonical audited mechanism visualization.

## Cases

- `heart_1004_z0079`
- `heart_1014_z0191`

Each case contains complete `representation/`, `ot/`, and `decision/` outputs.

## v3 refinements

1. `representation/01_backbone_raw_energy.png`
   - Expert and Student absolute RMS retain one shared absolute color scale.
   - A fourth column shows Student within-level relative contrast:
     `clip((RMS - P1) / (P99 - P1), 0, 1)`.
   - P1/P99 are computed jointly across the two selected cases for each level.

2. `ot/res*_s_transport.png`
   - The previous barycentric-teacher RMS panel is replaced by the true received
     expert signal `U_i = sum_j pi_ij E_j`.
   - `RMS(U_i)` has its own color scale because it includes transported mass.

3. `ot/res*_s_token_layout.png`
   - The middle panel remains the barycentric teacher
     `T_i = U_i / (received_i + eps)`.
   - Teacher and Student visibility are both controlled by received mass.
   - PCA is fitted jointly with received-mass weighting. Low-received tokens fade
     toward neutral gray.

## Reproducibility

- Fixed cross-case scales: `shared_color_scales.json`
- Per-case color scales: `*/representation/color_scales.json`
- Derived numeric maps: `*/representation/representation_maps.npz` and
  `*/ot/ot_derived_maps.npz`
- OT definitions and PCA metadata: `*/ot/ot_summary.json`
- Checkpoint, Git commit, generator, and slice metadata: `*/manifest.json`

Generated with:

```bash
python scripts/visualize_mechanism_v3.py \
  --checkpoint experiments/ot_uot_scale_gate_f0_30ep_20260722/models/fusion_disentangle_best.pth \
  --case_id <case_id> --split val --slice_index <z> \
  --selection all_classes --device cuda:0 --block_z 6 \
  --shared_color_scales experiments/ot_uot_scale_gate_f0_30ep_20260722/analysis/mechanism_v3/shared_color_scales.json \
  --output_root experiments/ot_uot_scale_gate_f0_30ep_20260722/analysis/mechanism_v3
```
