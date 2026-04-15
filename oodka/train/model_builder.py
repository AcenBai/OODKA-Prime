"""Build frozen backbones and trainable fusion modules."""

from __future__ import annotations

import os
from typing import Dict, List, Tuple

import torch
import torch.nn as nn

from ..config import (
    ensure_nnunet_on_path,
    ensure_biomedparse_on_path,
    BIOMEDPARSE_DIR,
    BIOMEDPARSE_CKPT,
)
from ..models.disentangle import TwoBranchDisentangle, DualBranchAutoEncoder
from ..models.gate import ClassQueryPooler, GateNet


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


def _detect_pixel_decoder_ms_channels(model_biomedparse: nn.Module, device: torch.device) -> Tuple[int, List[int]]:
    """Probe pixel decoder to detect mask_features channels and multi_scale_features channels."""
    from ..models.biomedparse_helpers import parse_pixel_decoder_out
    import torch

    dummy = torch.randn(1, 3, 256, 256, device=device)
    pixel_mean = model_biomedparse.pixel_mean.view(1, 3, 1, 1).to(device)
    pixel_std = model_biomedparse.pixel_std.view(1, 3, 1, 1).to(device)
    dummy = (dummy - pixel_mean) / pixel_std

    with torch.no_grad():
        img_embeds = model_biomedparse.backbone(dummy)
        pd_out = model_biomedparse.sem_seg_head.pixel_decoder.forward_features(img_embeds)

    mf, ms = parse_pixel_decoder_out(pd_out)
    return mf.shape[1], [m.shape[1] for m in ms]


def build_fusion_modules(
    model_nnunet: nn.Module,
    model_biomedparse: nn.Module,
    P: int,
    device: torch.device,
    d_q: int = 256,
    n_heads: int = 8,
) -> Dict[str, nn.Module]:
    """
    Build all trainable fusion modules.

    Returns dict with keys:
        ae_enc2_to_res2..ae_enc5_to_res5 (4 DualBranchAutoEncoder),
        dis_b_res2..dis_b_res5 (4 TwoBranchDisentangle),
        class_query_pooler,
        gate_net
    """
    enc_ch = _detect_nnunet_enc_channels(model_nnunet, device)
    res_ch = _detect_biomedparse_res_channels(model_biomedparse, device)
    mask_ch, ms_ch_list = _detect_pixel_decoder_ms_channels(model_biomedparse, device)

    # Detect deepest encoder channels for class query pooler
    feat = {}
    def _hook(m, i, o):
        feat["deepest"] = o
    h = model_nnunet.encoder.stages[-1].register_forward_hook(_hook)
    is_2d = any(isinstance(m, nn.Conv2d) for m in model_nnunet.modules()) and \
            not any(isinstance(m, nn.Conv3d) for m in model_nnunet.modules())
    dummy = torch.randn(1, 1, 256, 256, device=device) if is_2d else torch.randn(1, 1, 32, 256, 256, device=device)
    with torch.no_grad():
        model_nnunet(dummy)
    h.remove()
    C_enc_deepest = feat["deepest"].shape[1]

    modules = {}

    for si in [2, 3, 4, 5]:
        c_in = enc_ch.get(f"enc{si}", 64)
        c_out = res_ch.get(f"res{si}", 256)
        c_mid = min(max(c_in, c_out) // 2, 256)
        modules[f"ae_enc{si}_to_res{si}"] = DualBranchAutoEncoder(c_in, c_mid, c_out).to(device)
        modules[f"dis_b_res{si}"] = TwoBranchDisentangle(c_out).to(device)

    modules["class_query_pooler"] = ClassQueryPooler(
        P=P, C_e=C_enc_deepest, d_q=d_q, n_heads=n_heads
    ).to(device)

    modules["gate_net"] = GateNet(
        d_q=d_q, out_ch_mask=mask_ch, out_ch_ms=ms_ch_list
    ).to(device)

    return modules


# ---------------------------------------------------------------------------
# Prompt features
# ---------------------------------------------------------------------------


def build_prompt_features(
    model_biomedparse: nn.Module,
    text_prompts: Dict[str, str],
    device: torch.device,
) -> Tuple[dict, torch.Tensor]:
    """Encode text prompts through BiomedParse's text encoder.

    Returns (prompt_features, class_emb).
    """
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
    class_emb = prompt_features.get(
        "class_emb", torch.zeros(len(ids), 256, device=device)
    )
    return prompt_features, class_emb
