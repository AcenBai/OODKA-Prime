# OT/30: Initial-Proposal-Guided S Injection

## Scope

This experiment changes only OODKA code. The BiomedParse repository and its
frozen weights are not modified.

The default OT/30 predictor path is:

```text
P/S backbone features
        |
        v
frozen Pixel Decoder, run separately on P and S
        |
        +--> P mask feature + prompt-conditioned mask queries
        |         |
        |         v
        |    initial P query masks (16 hypotheses)
        |         |
        |         v
        |    detached pixel-wise top-k mean R
        |
        v
F_l = P_l + [epsilon + (1 - epsilon) R_l] * S_l
        |
        v
the original frozen BoltzFormer iterative decoder
        |
        v
final query masks -> query mean -> per-prompt logits
```

`R` is recomputed for every image slice and prompt. It is fixed only within
one decoder forward. The official per-layer attention mask continues to update
after every decoder layer.

## Default choices

- `query_guided_topk = 4`
- `query_guided_s_floor = 0.2`
- `R` is detached before it gates S
- the official object-existence classifier is not used
- the old `PromptBetaRouter` remains checkpoint-compatible but is frozen
- `w_route = 0`

The S floor keeps a weak S residual everywhere:

```text
R = 0 -> P + 0.2 S
R = 1 -> P + 1.0 S
```

## Train

The new path is the default:

```bash
cd /data4/baihexiang/SegMan/OODKA

/data4/baihexiang/conda_envs/biomedparse_v2/bin/python run_train.py \
  --device cuda:0 \
  --n_epochs 30 \
  --output_dir outputs/oodka_ot30_query_guided_30ep
```

For a short runtime smoke test:

```bash
/data4/baihexiang/conda_envs/biomedparse_v2/bin/python run_train.py \
  --device cuda:0 \
  --n_epochs 1 \
  --train_case_limit 1 \
  --val_case_limit 1 \
  --max_train_batches 1 \
  --max_val_batches 1 \
  --num_workers 0 \
  --output_dir outputs/smoke_ot30_query_guided
```

To reproduce the old fusion path:

```bash
/data4/baihexiang/conda_envs/biomedparse_v2/bin/python run_train.py \
  --legacy_beta_router \
  --w_route 0.001 \
  --output_dir outputs/legacy_beta_ablation
```

## Checkpoint behavior

- New full and student-deploy checkpoints save the architecture fields in
  `config`.
- Evaluation reads those fields before building modules.
- A checkpoint without the new fields is treated as a legacy Beta checkpoint.
- `--resume_checkpoint` always preserves the checkpoint's original fusion
  architecture; it does not silently convert a legacy run into OT/30.

## Runtime diagnostics

Training logs report the proposal mean and standard deviation as `R=mean±std`.
This is an early collapse check:

- nearly constant high `R`: S is almost globally injected;
- nearly constant low `R`: the model mostly uses `P + 0.2 S`;
- spatially and prompt-varying `R`: the intended retrieval behavior is active.
