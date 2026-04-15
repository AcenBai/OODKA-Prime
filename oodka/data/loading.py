"""Low-level patch loading from preprocessed data stores."""

from __future__ import annotations

import os
import pickle
from typing import List, Tuple

import numpy as np

from ..config import ensure_nnunet_on_path

ensure_nnunet_on_path()
import blosc2


def load_patch_from_nnunet_preproc(
    nnunet_preproc_dir: str,
    case_id: str,
    bbox_lbs: List[int],
    bbox_ubs: List[int],
    patch_size: List[int],
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """Load a 3D patch from nnUNet blosc2 preprocessed data."""
    data_arr = blosc2.open(
        os.path.join(nnunet_preproc_dir, f"{case_id}.b2nd"),
        mode="r", dparams={"nthreads": 1},
    )
    seg_arr = blosc2.open(
        os.path.join(nnunet_preproc_dir, f"{case_id}_seg.b2nd"),
        mode="r", dparams={"nthreads": 1},
    )
    with open(os.path.join(nnunet_preproc_dir, f"{case_id}.pkl"), "rb") as f:
        props = pickle.load(f)

    data = np.array(data_arr, dtype=np.float32)
    seg = np.array(seg_arr, dtype=np.int16)

    sd, sh, sw = [int(x) for x in bbox_lbs]
    pd, ph, pw = [int(x) for x in patch_size]
    D, H, W = data.shape[1], data.shape[2], data.shape[3]

    patch_data = np.zeros((data.shape[0], pd, ph, pw), dtype=np.float32)
    patch_seg = -np.ones((1, pd, ph, pw), dtype=np.int16)

    src_d0, src_d1 = max(0, sd), min(D, sd + pd)
    src_h0, src_h1 = max(0, sh), min(H, sh + ph)
    src_w0, src_w1 = max(0, sw), min(W, sw + pw)

    dst_d0 = src_d0 - sd
    dst_h0 = src_h0 - sh
    dst_w0 = src_w0 - sw

    if src_d1 > src_d0 and src_h1 > src_h0 and src_w1 > src_w0:
        patch_data[:, dst_d0:dst_d0 + (src_d1 - src_d0),
                   dst_h0:dst_h0 + (src_h1 - src_h0),
                   dst_w0:dst_w0 + (src_w1 - src_w0)] = \
            data[:, src_d0:src_d1, src_h0:src_h1, src_w0:src_w1]
        patch_seg[:, dst_d0:dst_d0 + (src_d1 - src_d0),
                  dst_h0:dst_h0 + (src_h1 - src_h0),
                  dst_w0:dst_w0 + (src_w1 - src_w0)] = \
            seg[:, src_d0:src_d1, src_h0:src_h1, src_w0:src_w1]

    return patch_data, patch_seg, props


def load_patch_from_biomedparse_preproc(
    biomedparse_preproc_dir: str,
    case_id: str,
    bbox_lbs: List[int],
    bbox_ubs: List[int],
    patch_size: List[int],
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """Load a 3D patch from BiomedParse npz preprocessed data."""
    data_dict = np.load(os.path.join(biomedparse_preproc_dir, f"{case_id}.npz"))
    patch_data = data_dict["data"]
    patch_seg = data_dict["seg"]
    with open(os.path.join(biomedparse_preproc_dir, f"{case_id}.pkl"), "rb") as f:
        props = pickle.load(f)

    if patch_seg is not None and patch_seg.ndim == 4:
        patch_seg = patch_seg[0]

    sd, sh, sw = [int(x) for x in bbox_lbs]
    pd, ph, pw = [int(x) for x in patch_size]

    is_4d = patch_data.ndim == 4
    if is_4d:
        D, H, W = patch_data.shape[1], patch_data.shape[2], patch_data.shape[3]
        out_data = np.zeros((patch_data.shape[0], pd, ph, pw), dtype=patch_data.dtype)
    else:
        D, H, W = patch_data.shape
        out_data = np.zeros((pd, ph, pw), dtype=patch_data.dtype)

    out_seg = -np.ones((pd, ph, pw), dtype=np.int16) if patch_seg is not None else None

    src_d0, src_d1 = max(0, sd), min(D, sd + pd)
    src_h0, src_h1 = max(0, sh), min(H, sh + ph)
    src_w0, src_w1 = max(0, sw), min(W, sw + pw)

    dst_d0 = src_d0 - sd
    dst_h0 = src_h0 - sh
    dst_w0 = src_w0 - sw

    if src_d1 > src_d0 and src_h1 > src_h0 and src_w1 > src_w0:
        sl_src = (slice(src_d0, src_d1), slice(src_h0, src_h1), slice(src_w0, src_w1))
        sl_dst = (
            slice(dst_d0, dst_d0 + src_d1 - src_d0),
            slice(dst_h0, dst_h0 + src_h1 - src_h0),
            slice(dst_w0, dst_w0 + src_w1 - src_w0),
        )
        if is_4d:
            out_data[(slice(None),) + sl_dst] = patch_data[(slice(None),) + sl_src]
        else:
            out_data[sl_dst] = patch_data[sl_src]
        if out_seg is not None and patch_seg is not None:
            out_seg[sl_dst] = patch_seg[sl_src]

    return out_data, out_seg, props
