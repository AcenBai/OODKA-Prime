# MRI ROI refinement stage report

Date: 2026-09-21

Branch: `codex/lge-roi-five-class`

Scope: no data augmentation; MRI whole-heart ROI geometry and deployable two-pass refinement

## Executive result

The ROI direction is viable. The first controlled Oracle study shows that the
historical crop-to-square resize is a material source of OOD degradation, and
that preserving the ROI pixel scale with padding raises the empirical GT-box
reference well above the historical global baseline.

| Oracle ROI geometry | Raw mean Dice | Largest-per-class postprocessed mean Dice | Best ID val exclusive-macro Dice |
| --- | ---: | ---: | ---: |
| resize | 0.687459 | 0.692739 | 0.840595 (epoch 20) |
| letterbox | 0.689266 | 0.697568 | 0.817862 (epoch 25) |
| **pad** | **0.715459** | **0.719778** | 0.825317 (epoch 20) |

Against the historical global predictions on the same 26 OOD cases:

| Comparison | Paired mean delta | Bootstrap 95% CI | Case wins |
| --- | ---: | ---: | ---: |
| Oracle-pad raw vs global raw 0.662654 | **+0.052805** | [+0.039800, +0.065102] | 24/26 |
| Oracle-pad LCC vs global LCC 0.670067 | **+0.049711** | [+0.034931, +0.063183] | 22/26 |
| Oracle-pad raw vs Oracle-resize | **+0.028000** | [+0.018152, +0.038334] | 21/26 |
| Oracle-pad LCC vs Oracle-resize | **+0.027039** | [+0.015415, +0.040197] | 21/26 |

The Oracle result is an empirical reference under this defined GT-box
protocol, not a mathematical upper bound or a deployable score. A looser box
or a differently trained refiner can still outperform it in an individual
case. In the inference input path, test labels are used to define the
test-time box; as usual, they are also used after inference to score the
prediction. All three Oracle runs use seed 42, Z4-block unions of labels 1--7,
expansion 1.4, and a full-canvas fallback for an empty union. They provide
strong single-seed evidence that, with a sufficiently accurate ROI and
scale-preserving geometry, the current refinement model can exceed the global
baseline.

The exact matched global rerun is still training, so the table above is a
strict same-case comparison against the historical global predictions, not
yet the final matched-training comparison. Because the three geometries were
also inspected on these 26 cases, their ranking is exploratory until it is
confirmed on an independent OOD cohort.

The paired bootstrap intervals resample the 26 test cases. They quantify case
sampling uncertainty only; they do not include model-training seed
uncertainty.

## What the geometry experiment says

MRI uses a `320 x 320` model canvas, not `512 x 512`.

The historical path is:

1. aligned/native MRI is resized to `320 x 320`;
2. an XY ROI is extracted on that grid;
3. the ROI is stretched again to `320 x 320`;
4. local logits are resized back into the ROI.

The new `pad` mode removes steps 3 and 4 when the ROI already fits the canvas.
It does not remove the first aligned-image-to-320 resampling.

For the prior predicted great-vessel run, the median maximum-axis zoom was
3.52x, 32.7% of source-positive blocks exceeded 4x, the 95th percentile was
10.67x, and the maximum was 20x. The case audit further showed:

- AO Dice is associated with localization coverage and aggressive resizing;
- PA Dice is almost unrelated to box coverage, so geometry is not the only
  failure mechanism;
- high-coverage cases still fail when the local branch replaces all seven
  global classes;
- strong-resize counterexamples also exist, so interpolation is a risk factor,
  not a complete causal explanation.

The Oracle geometry ranking reverses between in-domain validation and OOD
test: resize has the best ID validation Dice, while pad is much better OOD.
This is evidence that scale-preserving padding improves domain-shift
robustness, but geometry selection must remain predeclared or be confirmed on
another OOD validation set rather than tuned repeatedly on this test set.

Local audit artifacts:

- geometry distributions:
  `experiments/mri_roi_geometry_audit_20260920/existing_gv_predicted/roi_geometry_diagnostics.png`;
- coverage/resize versus failure scatter:
  `experiments/mri_roi_case_failure_scatter_20260920/existing_gv_predicted/roi_case_failure_scatter.png`.

## Per-class Oracle result

Raw OOD Dice:

| Geometry | LV | RV | LA | RA | Myo | AO | PA |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| resize | 0.7931 | 0.6211 | 0.8217 | 0.7381 | 0.6740 | 0.6644 | **0.4998** |
| letterbox | 0.8033 | 0.6134 | 0.8282 | 0.7484 | 0.6694 | 0.6824 | 0.4799 |
| **pad** | **0.8397** | 0.6189 | **0.8533** | **0.7779** | **0.6915** | **0.7433** | 0.4836 |

Pad improves five of seven classes over resize and raises the overall
seven-class mean substantially, but it does not solve PA. The next
architecture should therefore preserve the strong global prediction and use
local refinement as a class-selective correction rather than a complete
seven-class replacement.

## First predicted-box transfer result

The raw zero-training transfer result is now complete: the Oracle-pad
checkpoint uses its jointly trained first pass to predict the box, while all
weights remain unchanged.

| Raw protocol | Mean Dice | Delta vs historical global | Paired 95% CI | Case wins vs global |
| --- | ---: | ---: | ---: | ---: |
| historical global | 0.662654 | - | - | - |
| Oracle-pad with GT box | 0.715459 | +0.052805 | [+0.040034, +0.065131] | 24/26 |
| **Oracle-pad checkpoint with predicted box** | **0.653926** | **-0.008728** | **[-0.017255, +0.000079]** | **7/26** |

Predicted-box transfer loses `0.061533` Dice against the same checkpoint with
a GT box (paired CI `[-0.073349, -0.049820]`) and loses on all 26 cases. Its
empirical gap capture `(T - G) / (O - G)` is `-0.165`, so merely substituting
the learned box does not deploy the Oracle gain.

The failure is not explained by gross anatomy coverage. Across positive
blocks, mean seven-class-union coverage is 0.9887; AO voxel-weighted coverage
is 0.99998 and PA coverage is 1.0. Only 6/991 positive blocks have zero union
coverage, and fallback is 2.26%. Predicted boxes are actually looser than the
GT boxes: median canvas area fraction is 0.262 versus 0.197 for Oracle.

Class-wise transfer deltas versus historical global are LV -0.0063, RV
+0.0327, LA -0.0368, RA -0.0208, Myo -0.0059, AO +0.0259, and PA -0.0500.
The evidence therefore shifts priority away from simple union-recall tuning
and toward predicted-box training-distribution robustness, local background
rejection, and protected global/local fusion. Raw transfer geometry is saved
at
`experiments/whs_mri_oracle_pad_noaug_relative_capacity_s_z4_b1_30ep_20260920/test_best_predicted_transfer_none/geometry/roi_geometry_diagnostics.png`.
Largest-component transfer and the resize/letterbox transfer controls are
still running, so this is a raw, single-checkpoint readout.

## Existing deployable evidence

The parallel selective AO/PA worktree already validates this principle using
predicted ROIs and frozen global predictions:

| Method | Evidence status | Raw Dice | LCC Dice | Raw delta vs global |
| --- | --- | ---: | ---: | ---: |
| historical global | reference | 0.662654 | 0.670067 | - |
| protected child-only AO/PA | post-hoc mechanism diagnostic | 0.666945 | **0.673215** | +0.004291 |
| protected child-only + GV veto | validation-locked | **0.667394** | not reported in the locked raw ablation | +0.004740 |

In its raw diagnostic, protected child-only copies LV/RV/LA/RA/Myo bit-for-bit
from the global model and only lets the ROI branch correct background/AO/PA.
Its raw gain has bootstrap CI `[+0.002136, +0.006763]`, but this specific
child-only rule was inspected post hoc. The validation-locked GV-veto variant
improves 21/26 cases with CI `[+0.002729, +0.007106]`.

The GV-veto result is the current deployable evidence that ROI refinement can
exceed baseline; child-only is supporting mechanism evidence. The Oracle-pad
result indicates that substantially more gain is available if predicted
localization and local-background rejection are improved. Full provenance is
documented in
`/data4/baihexiang/SegMan/worktrees/mri-gv-selective-refinement/docs/mri_selective_ao_pa_refinement.md`.

## Code delivered and synchronized

The following are soft switches and remain reversible:

- `--roi_train_source {predicted,ground_truth,full}`;
- `--roi_transform {resize,pad,letterbox}`;
- shared `--seed` control across all three training entry points;
- `--postprocess {none,largest_per_class}` in both evaluators;
- default Relative KD enabled;
- default capacity-partial S transport enabled.

Padding pixels in the segmentation target use label `-1` and are excluded
from BCE/Dice. Image padding remains zero. Bounds, reversible placement, odd
geometry, Oracle Z-block union, and checkpoint fallback behavior are covered
by tests. Current test result: `70 passed`.

Relevant pushed commits:

- `2257292` - Oracle ROI, geometry transforms, diagnostics, raw/LCC switches;
- `a5ff934` - strict paired comparison and bootstrap tool;
- `3816613` / `82b55fa` - predicted-pad experiment pipeline and sweep knobs;
- `255aab0` - Oracle-checkpoint to predicted-ROI transfer evaluation.
- `a8f4ab5` - shared seed control, replication queue, automatic comparisons,
  and this stage report.

## Running and queued experiments

All experiment directories are under `spatial_combination/experiments/`.

1. Matched global baseline, Relative KD + capacity S, B1/Z4/30 epochs:
   `whs_mri_global_noaug_relative_capacity_s_z4_b1_30ep_20260920/`.
2. Full-canvas two-branch control:
   `whs_mri_full_roi_control_noaug_relative_capacity_s_z4_b1_30ep_20260920/`.
3. Predicted-pad, warmup 5:
   `whs_mri_predicted_pad_noaug_relative_capacity_s_z4_b1_w5_30ep_20260921/`.
4. Predicted-pad, warmup 10, queued after item 3:
   `whs_mri_predicted_pad_noaug_relative_capacity_s_z4_b1_w10_30ep_20260921/`.
5. Predicted-pad, warmup 5, expansion 1.8, queued after the full control:
   `whs_mri_predicted_pad_noaug_relative_capacity_s_z4_b1_w5_t02_e18_30ep_20260921/`.
6. Predicted-pad, warmup 5, threshold 0.1, queued after the matched baseline:
   `whs_mri_predicted_pad_noaug_relative_capacity_s_z4_b1_w5_t01_e14_30ep_20260921/`.
7. Zero-training transfer diagnostic: evaluate Oracle-trained pad, resize, and
   letterbox checkpoints using predicted boxes. These write
   `test_best_predicted_transfer_{none,largest_per_class}/` inside each Oracle
   experiment.
8. Automatic raw/LCC paired aggregation:
   `mri_roi_geometry_comparison_20260920/`.
9. Predicted-pad warmup-5 replication with independent seed 43, queued after
   the warmup-10 run:
   `whs_mri_predicted_pad_noaug_relative_capacity_s_z4_b1_w5_seed43_30ep_20260921/`.
10. Automatic predicted-pad matrix and seed-replication comparisons:
    `mri_predicted_pad_comparison_20260921/`.
11. Matched global seed-43 replication, queued after the threshold-0.1 run so
    each ROI seed is compared with the same global seed:
    `whs_mri_global_noaug_relative_capacity_s_z4_b1_seed43_30ep_20260921/`.

Every pipeline writes a completion marker. A result is complete only when the
corresponding `PIPELINE_COMPLETE`, `PREDICTED_TRANSFER_EVAL_COMPLETE`, or
comparison-complete marker exists. Downstream GPU queues now verify the
upstream marker before starting instead of relying only on process exit.

The automatic OOD matrix is an exploratory diagnostic. Warmup, threshold, and
expansion must be selected from ID validation and then locked before an OOD
score is treated as confirmatory. Warmup 5 and warmup 10 also do not have the
same mixed-training budget: in a 30-epoch run they receive 25 versus 20 mixed
epochs. A lower warmup-10 result would therefore be ambiguous; only if that
comparison matters should it be repeated for 35 total epochs.

Snapshot at 2026-09-21 01:55 CST:

- matched global has completed epoch 16; best validation Dice is 0.7718 at
  epoch 15;
- full-canvas control has completed epoch 8; best validation Dice is 0.7770 at
  epoch 5;
- predicted-pad warmup-5 has completed epoch 5; it has not yet completed a
  predicted-ROI mixed epoch, so its warmup validation score is not evidence
  about refinement;
- Oracle-pad to predicted-ROI raw transfer is complete; largest-component
  transfer is running;
- no active job has reported OOM, traceback, NaN, or a killed process.

GPU 7 has only about 2 GiB free because of an unrelated resident workload.
The transfer evaluator is currently progressing, but this is the main runtime
risk; a failed transfer should be requeued on a roomier GPU without changing
its checkpoint or evaluation options.

## Known caveat in simple padding

The `-1` target mask removes padded pixels from segmentation loss. The
remaining auxiliary paths are not all equivalent: AE and orthogonality mask
invalid Z slices but still see the XY-padded canvas; P-OT excludes `-1` pixels
from its GT structural mass; S-OT can still assign mass through feature energy
on padding; router KL is prompt-only, while its spatial gate is parameterized
over the entire padded canvas. BiomedParse also mixes spatial context
internally, so a loss mask cannot make the forward pass completely blind to
black borders.

This does not invalidate the current result: it measures the actual simple
padding pipeline proposed for deployment. It does mean that a failed pad run
would not isolate geometry alone. A follow-up masked-regularizer ablation is
worth doing only after the predicted-pad score is known.

## Causal matrix readout

Use the queued runs as a five-point decomposition:

- `G`: matched global baseline;
- `F`: full-canvas two-branch control;
- `O`: Oracle-pad empirical GT-box reference;
- `T`: Oracle-pad checkpoint evaluated with predicted boxes;
- `P`: predicted-pad training with the ID-validation-locked setting.

For any run `X`, report empirical gap capture as
`R_X = (X - G) / (O - G)` alongside the absolute Dice delta. This makes the
failure location easier to distinguish:

- `F > G` measures the value of the second branch/prompts without cropping;
- `T - O` mixes box error, box-distribution shift, and seven-class local
  replacement at inference;
- `P - T` measures whether training on predicted/noisy boxes helps;
- `P - F` is the remaining measurable value of cropping under the current
  full-replacement architecture.

## Optimization priorities after the queued matrix

### Priority 1: close the Oracle-to-predicted gap

This gap is not localization alone. It jointly contains box accuracy, the
GT-box versus predicted-box training-distribution mismatch, and the harm from
replacing all seven global classes with local predictions.

- choose warmup 5 vs 10 from in-domain validation, then report locked OOD;
- compare expansion 1.4 vs 1.8 and threshold 0.2 vs 0.1;
- smooth boxes across adjacent Z blocks or derive one connected 3-D support to
  reduce scale/position jitter;
- add a training-only Oracle-to-predicted box curriculum rather than exposing
  the refiner to perfect boxes for all epochs.

### Priority 2: stop replacing correct global anatomy

- preserve the global seven-class output by default;
- let the ROI branch emit only AO/PA corrections or class-wise residual logits;
- directly optimize the composed final output, not just independent anchor and
  refinement losses;
- protect LV/RV/LA/RA/Myo exactly, as the successful selective experiment does.

### Priority 3: improve rejection, not just foreground recognition

- add an explicit local background/reject prompt;
- use a loose ROI for reading context but a tighter connected support for
  writing corrections;
- score connected AO/PA proposals instead of independent voxels;
- anchor proposals to existing global vessels or high-confidence GV support.

### Priority 4: geometry refinements if pad is too conservative

- test capped aspect-preserving zoom (for example max 1.5x or 2x) followed by
  padding; current letterbox zooms without a cap and is not equivalent;
- provide ROI scale and original-center coordinates to the router;
- optionally mask XY padding in AE/OT/orthogonality losses.

## Decision rule

- If `F > G + 0.005` and the paired CI lower bound is positive, future ROI
  methods must beat `F`, not merely `G`.
- If `abs(P - F) < 0.003`, stop the threshold/expansion/warmup sweep: cropping
  has no measurable extra value under full replacement.
- If `T` captures at least 70% of `O - G`, prioritize protected fusion; if its
  GT-box recall is below 0.95 or fallback exceeds 10%, fix localization first.
- If `P - T >= 0.005`, predicted-box training is useful. If `T - P >= 0.005`,
  test Oracle pretraining followed by predicted-box finetuning instead.
- If predicted-pad exceeds the matched raw baseline by at least 0.005 and its
  paired bootstrap lower bound is positive for both matched seed pairs, make
  it the deployable default. If the two seed deltas differ by more than 0.005
  or flip sign, add seed 44 before deciding.
- If all predicted full-replacement settings remain within 0.003 of `G`, stop
  that sweep and combine pad with validation-locked protected AO/PA residual
  fusion.
- If it remains far below the empirical Oracle reference, decompose locator
  recall, box-distribution shift, and full-replacement fusion; do not spend the
  next budget on another blind seven-class ROI sweep.
