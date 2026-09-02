# CT-OOD Direct-Expert alignment ablations

All variants use `Dataset009_CT_OOD`, fold 0, Direct-Expert, no ROI, no data
augmentation, `B=1`, `Z=4`, adjacent pseudo-RGB, and 30 epochs.  Each detached
pipeline trains, evaluates the best checkpoint on the independent test split,
and generates mechanism-v3 visualizations for `heart_1004/z0079` and
`heart_1014/z0191`.

Launch one variant on a physical GPU:

```bash
bash shell/launch_ct_direct_expert_alignment_variant.sh GPU VARIANT
```

Variants:

- `no_relative`: original one-way Expert-to-Student KD.
- `relative_no_expert_ortho`: bidirectional KD, but Expert orthogonality is off.
- `relative_rms`: bidirectional KD plus transported reverse log-RMS alignment.
- `relative_capacity_s`: bidirectional KD plus capacity-constrained S partial OT.
- `relative_p_only`: reverse KD is active for P only; S remains one-way.

Optional environment overrides:

```bash
BATCH_SIZE=1 RELATIVE_KD_RMS_WEIGHT=0.5 \
  bash shell/launch_ct_direct_expert_alignment_variant.sh 0 relative_rms

S_PARTIAL_MASS_FRACTION=0.5 \
  bash shell/launch_ct_direct_expert_alignment_variant.sh 1 relative_capacity_s
```

The capacity-constrained variant transports exactly the configured fraction
of the normalized mass while enforcing both real marginal capacities.  Its
rejection is therefore `b - transported` without hidden overuse.

Re-run the detailed visualization independently:

```bash
bash shell/visualize_ct_direct_expert_checkpoint.sh \
  GPU /absolute/path/to/experiment heart_1004 79
```
