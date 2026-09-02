# MRI selective AO/PA refinement experiment

Date: 2026-09-02  
Branch: `codex/mri-gv-selective-refinement`

## Question

Can the strong global seven-class MRI model remain the default prediction,
while a great-vessel ROI branch is allowed to overwrite only confident AO/PA
voxels? This preserves the global model's open prompt semantics and avoids
forcing all seven classes to compete inside the ROI.

## Fusion rule

For every raw-space voxel:

1. retain the global seven-class label by default;
2. evaluate the local AO and PA prompts as independent sigmoid scores;
3. accept the winning child only above a validation-locked confidence
   threshold and top-1/top-2 probability margin;
4. optionally protect global LV/RV/LA/RA/Myo labels;
5. never use OOD test labels to select a model, threshold, margin, overwrite
   scope, or postprocessing rule.

## Proof of concept with the existing seven-prompt ROI checkpoint

The existing local checkpoint was not retrained. Only its AO/PA channels were
used. In-domain validation selected confidence `0.95`, margin `0.05`, overwrite
scope `any`, and largest-component-per-class postprocessing.

| MRI OOD metric (26 cases) | Global | Selective | Delta |
| --- | ---: | ---: | ---: |
| Raw seven-class mean Dice | 0.662654 | 0.665213 | +0.002559 |
| Keep-largest mean Dice | 0.670067 | 0.671193 | +0.001127 |

Raw Dice improved in 19/26 cases. The paired mean delta bootstrap 95% CI was
`[+0.000837, +0.004384]`; paired Wilcoxon `p=0.01295`. With largest-component
postprocessing the result was less stable: 16/26 wins, 95% CI
`[-0.001719, +0.003823]`, `p=0.22696`.

Raw class-wise mean Dice changes:

| LV | RV | LA | RA | Myo | AO | PA |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| +0.000055 | +0.000169 | -0.001963 | -0.000361 | +0.000005 | +0.003790 | +0.016216 |

The proof of concept therefore produces a small real OOD gain, dominated by
PA, but does not reproduce the large LGE bridge-prompt improvement.

## Failure mode

The local overwrite changed 503,919 voxels:

| Outcome | Voxels | Fraction of changes |
| --- | ---: | ---: |
| Corrected wrong global label | 128,002 | 25.4% |
| Replaced a correct global label | 275,786 | 54.7% |
| Wrong label to another wrong label | 100,131 | 19.9% |

Of 393,615 changes originating from global background, 122,857 recovered a
missed AO/PA voxel, but 260,464 created a new AO/PA false positive on true
background. A further 7,715 correct non-child foreground voxels were overwritten,
which explains the small LA/RA regression under the `any` scope.

The upstream GV localization threshold mask is much less precise OOD than on
validation. Validation precision/recall were `0.721/0.822`; the five OOD shard
precision values ranged `0.283-0.356` and recall ranged `0.588-0.743`. Across
OOD blocks, 25.85% used fallback and the mean ROI area fraction was 0.320. The
local branch therefore receives many oversized or full-image crops and has no
background prompt with which to veto an AO/PA false positive.

The prior router audit does not show saturated gates, but the centered spatial
gate maps have effective rank about `1.03`. The router is therefore highly
low-rank across prompts, while the unwarped-gate affine mismatch remains
measurable (maximum about `0.092` for the audited transforms). Spatial affine
unity remains a plausible secondary issue; the direct failure observed here is
ROI/local-background suppression.

## Child-only experiment

The definitive experiment trains only:

- one GV-union localization prompt, supervised by AO union PA;
- two refinement prompts, AO and PA;
- no local LV/RV/LA/RA/Myo competition.

Five ROI-conservatism variants were fine-tuned from the same existing fusion
weights on GPUs 0-4. Model selection and fusion calibration used only the four
in-domain validation cases. The controller selected:

- ROI threshold `0.2`, expansion `1.4`, center fallback;
- best epoch `10`, local child Dice `0.799932` (AO `0.778296`, PA `0.821567`);
- overwrite confidence `0.9`, ambiguity margin `0.1`, scope `any`;
- largest-component-per-class as the validation-selected primary output.

The validation-selected fused seven-class Dice was `0.839401`, versus
`0.836354` for the matched global baseline.

### Locked OOD result

| MRI OOD metric (26 cases) | Global | Child-only selective | Delta |
| --- | ---: | ---: | ---: |
| Raw seven-class mean Dice | 0.662654 | 0.665847 | +0.003193 |
| Keep-largest mean Dice | 0.670067 | 0.672412 | +0.002345 |

The raw result improved in 19/26 cases, with bootstrap 95% CI
`[+0.000877, +0.005576]` and paired Wilcoxon `p=0.01105`. The keep-largest
result improved in 16/26 cases, with CI `[-0.001309, +0.005834]` and
`p=0.14272`.

Raw child-only class deltas were LV `-0.000113`, RV `+0.000290`, LA
`-0.005234`, RA `-0.001242`, Myo `+0.000005`, AO `+0.012427`, and PA
`+0.016220`. Child-only training therefore materially strengthens AO, but an
unrestricted positive overwrite damages non-child open semantics.

### Principle-constrained protected-scope diagnostic

The `background_children` scope was declared and swept before OOD testing, but
validation selected `any` by `0.000779` Dice. Reconstructing the protected
scope from exactly the same accepted proposals is reported as a diagnostic,
not as the locked primary result.

| MRI OOD metric (26 cases) | Global | Protected child-only | Delta |
| --- | ---: | ---: | ---: |
| Raw seven-class mean Dice | 0.662654 | 0.666945 | +0.004291 |
| Keep-largest mean Dice | 0.670067 | 0.673215 | +0.003148 |

Protected raw Dice improved in 19/26 cases, with bootstrap 95% CI
`[+0.002136, +0.006763]` and Wilcoxon `p=0.00059`. It copies
LV/RV/LA/RA/Myo bit-for-bit from the global model. AO improves from `0.449620`
to `0.463572` and PA from `0.400927` to `0.417013` without sacrificing the
other five prompts.

This protected result best matches the intended semantics: global prediction
remains authoritative for every non-child class, while the ROI branch acts as
an AO/PA recall booster.

### Remaining failure

The unrestricted child-only output changed 818,275 voxels. It recovered
200,702 AO/PA voxels that the global model called background, but also changed
429,521 true-background voxels into AO/PA. Mean AO precision/recall changes
from `0.3630/0.6269` globally to `0.3500/0.7279` with protected refinement;
PA changes from `0.2997/0.7048` to `0.3041/0.7536`.

Thus the bridge is now a useful and statistically supported OOD recall path,
but AO background suppression remains the main limitation. The next change
should add an explicit background veto or connected anatomical support rather
than adding more foreground prompts.

All child-only artifacts are under
`experiments/selective_refinement_20260902/child_only_pipeline_finetune10/`.
