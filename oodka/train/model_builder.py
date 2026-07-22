"""Build frozen backbones and trainable fusion modules."""

from __future__ import annotations

import os
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from ..config import (
    ensure_nnunet_on_path,
    ensure_biomedparse_on_path,
    BIOMEDPARSE_DIR,
    BIOMEDPARSE_CKPT,
)
from ..models.disentangle import TwoBranchDisentangle, DualBranchAutoEncoder
from ..models.beta_router import PromptBetaRouter
from ..models.ot import MultiScaleOTDistillation


# ---------------------------------------------------------------------------
# Frozen backbones
# ---------------------------------------------------------------------------


def load_frozen_nnunet(
    model_dir: str, fold: int, device: torch.device,
) -> nn.Module:
    """Load a trained nnUNet model and freeze all parameters."""
    ensure_nnunet_on_path()
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
    from batchgenerators.utilities.file_and_folder_operations import load_json

    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=False,
        device=device,
        verbose=False,
    )
    predictor.initialize_from_trained_model_folder(
        model_dir,
        use_folds=(fold,),
        checkpoint_name="checkpoint_best.pth",
    )
    model = predictor.network.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


def load_frozen_biomedparse(device: torch.device) -> nn.Module:
    """Load the BiomedParse model and freeze all parameters."""
    ensure_biomedparse_on_path()
    import hydra
    from hydra import compose
    from hydra.core.global_hydra import GlobalHydra

    GlobalHydra.instance().clear()
    hydra.initialize_config_dir(
        config_dir=os.path.join(BIOMEDPARSE_DIR, "configs", "model"),
        job_name="oodka",
        version_base=None,
    )
    cfg = compose(config_name="biomedparse_3D")
    model = hydra.utils.instantiate(cfg, _convert_="object")

    model.load_pretrained(checkpoint_path=BIOMEDPARSE_CKPT)
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


def load_frozen_backbones(
    nnunet_model_dir: str, fold: int, device: torch.device,
) -> Tuple[nn.Module, nn.Module]:
    """Load and return (model_nnunet, model_biomedparse), both frozen."""
    model_nnunet = load_frozen_nnunet(nnunet_model_dir, fold, device)
    model_biomedparse = load_frozen_biomedparse(device)
    return model_nnunet, model_biomedparse


# ---------------------------------------------------------------------------
# Trainable fusion modules
# ---------------------------------------------------------------------------


def _detect_biomedparse_res_channels(model_biomedparse: nn.Module, device: torch.device) -> Dict[str, int]:
    """Probe BiomedParse backbone to detect output channels for res2..res5."""
    import torch
    dummy = torch.randn(1, 3, 256, 256, device=device)
    pixel_mean = model_biomedparse.pixel_mean.view(1, 3, 1, 1).to(device)
    pixel_std = model_biomedparse.pixel_std.view(1, 3, 1, 1).to(device)
    dummy = (dummy - pixel_mean) / pixel_std
    with torch.no_grad():
        out = model_biomedparse.backbone(dummy)
    return {k: v.shape[1] for k, v in out.items() if k.startswith("res")}


def _detect_nnunet_enc_channels(model_nnunet: nn.Module, device: torch.device) -> Dict[str, int]:
    """Probe nnUNet encoder to detect output channels at stages 2..5."""
    feat = {}

    def _hook(name):
        def _h(m, i, o):
            feat[name] = o
        return _h

    handles = []
    for si in [2, 3, 4, 5]:
        handles.append(model_nnunet.encoder.stages[si].register_forward_hook(_hook(f"enc{si}")))

    is_2d = any(isinstance(m, nn.Conv2d) for m in model_nnunet.modules()) and \
            not any(isinstance(m, nn.Conv3d) for m in model_nnunet.modules())
    if is_2d:
        dummy = torch.randn(1, 1, 256, 256, device=device)
    else:
        dummy = torch.randn(1, 1, 32, 256, 256, device=device)

    with torch.no_grad():
        model_nnunet(dummy)

    for h in handles:
        h.remove()

    return {k: v.shape[1] for k, v in feat.items()}


def build_fusion_modules(
    model_nnunet: Optional[nn.Module],
    model_biomedparse: nn.Module,
    P: int,
    device: torch.device,
    text_dim: int = 512,
    route_hidden_dim: int = 256,
    route_prior_p_means: Tuple[float, float, float, float] = (0.5, 0.6, 0.7, 0.8),
    route_prior_concentration: float = 10.0,
    ot_feature_weight: float = 1.0,
    ot_coordinate_weight: float = 0.1,
    p_ot_semantic_weight: float = 0.25,
    p_ot_epsilon: float = 0.1,
    s_ot_epsilon: float = 0.1,
    s_ot_rho_base: float = 1.0,
    s_ot_rho_expert: float = 0.2,
    ot_sinkhorn_iterations: int = 30,
) -> Dict[str, nn.Module]:
    """
    Build all trainable fusion modules.

    Returns dict with keys:
        During training, ae_enc2_to_res2..ae_enc5_to_res5
        (4 DualBranchAutoEncoder),
        dis_b_res2..dis_b_res5 (4 TwoBranchDisentangle),
        beta_router.

        Passing ``model_nnunet=None`` builds only modules required for pure
        student inference and does not import, initialize, or probe nnUNet.
    """
    if P <= 0:
        raise ValueError(f"P must be positive, got {P}")
    res_ch = _detect_biomedparse_res_channels(model_biomedparse, device)
    modules = {}

    enc_ch = (
        _detect_nnunet_enc_channels(model_nnunet, device)
        if model_nnunet is not None
        else {}
    )
    for si in [2, 3, 4, 5]:
        c_out = res_ch.get(f"res{si}", 256)
        modules[f"dis_b_res{si}"] = TwoBranchDisentangle(c_out).to(device)
        if model_nnunet is not None:
            c_in = enc_ch.get(f"enc{si}", 64)
            c_mid = min(max(c_in, c_out) // 2, 256)
            modules[f"ae_enc{si}_to_res{si}"] = DualBranchAutoEncoder(
                c_in, c_mid, c_out
            ).to(device)

    modules["beta_router"] = PromptBetaRouter(
        text_dim=text_dim,
        hidden_dim=route_hidden_dim,
        prior_p_means=route_prior_p_means,
        prior_concentration=route_prior_concentration,
    ).to(device)
    if model_nnunet is not None:
        modules["ot_distillation"] = MultiScaleOTDistillation(
            feature_weight=ot_feature_weight,
            coordinate_weight=ot_coordinate_weight,
            p_semantic_weight=p_ot_semantic_weight,
            p_epsilon=p_ot_epsilon,
            s_epsilon=s_ot_epsilon,
            rho_base=s_ot_rho_base,
            rho_expert=s_ot_rho_expert,
            sinkhorn_iterations=ot_sinkhorn_iterations,
        ).to(device)

    return modules


# ---------------------------------------------------------------------------
# Prompt features
# ---------------------------------------------------------------------------


def build_prompt_features(
    model_biomedparse: nn.Module,
    text_prompts: Dict[str, str],
    device: torch.device,
) -> dict:
    """Encode text prompts through BiomedParse's text encoder."""
    ids = sorted([int(k) for k in text_prompts.keys() if k != "instance_label"])
    text = "[SEP]".join([text_prompts[str(i)] for i in ids])

    with torch.no_grad():
        prompt_features = model_biomedparse.sem_seg_head.encode_prompts(
            text=text, eval=True
        )

    if isinstance(prompt_features, dict):
        for k, v in prompt_features.items():
            if torch.is_tensor(v):
                prompt_features[k] = v.to(device)
    return prompt_features
