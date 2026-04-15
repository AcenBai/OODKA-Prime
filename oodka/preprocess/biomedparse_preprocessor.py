"""
Unified BiomedParse preprocessing: train + val + test.

Reuses nnUNet DefaultPreprocessor for cropping/resampling, overriding only
the normalization step with CT (WL/WW -> 0-255) or MRI (percentile -> 0-255).
"""

from __future__ import annotations

import os
from multiprocessing import Pool
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from tqdm import tqdm

from ..config import (
    ensure_nnunet_on_path,
    nnunet_preprocessed_dir,
    nnunet_raw_dir,
    biomedparse_output_dir,
    NNUNET_PREPROCESSED,
    NNUNET_RAW,
)
from ..utils.normalization import BiomedParseCTNormalization, BiomedParseMRINormalization
from ..utils.io_utils import maybe_mkdir_p, strip_modality_suffix

ensure_nnunet_on_path()
from batchgenerators.utilities.file_and_folder_operations import (
    join,
    load_json,
    isfile,
    write_pickle,
)
from nnunetv2.paths import nnUNet_preprocessed, nnUNet_raw
from nnunetv2.preprocessing.preprocessors.default_preprocessor import DefaultPreprocessor
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager, ConfigurationManager
from nnunetv2.utilities.dataset_name_id_conversion import maybe_convert_to_dataset_name
from nnunetv2.utilities.utils import (
    get_filenames_of_train_images_and_targets,
    get_identifiers_from_splitted_dataset_folder,
    create_lists_from_splitted_dataset_folder,
)


# ---------------------------------------------------------------------------
# nnUNet-based preprocessors with custom normalization
# ---------------------------------------------------------------------------


def resolve_configuration_name(plans_manager: PlansManager, configuration_name: str) -> str:
    cfg = configuration_name.strip()
    available = list(plans_manager.available_configurations)
    if not available:
        raise RuntimeError("No configurations found in plans")
    if cfg.lower() == "auto":
        for pref in ("2d", "3d_fullres"):
            if pref in available:
                return pref
        return available[0]
    if cfg not in available:
        raise RuntimeError(f"Configuration {cfg!r} not in plans. Available: {available}")
    return cfg


class BiomedParseCTWindowPreprocessor(DefaultPreprocessor):
    """CT: WL/WW -> 0-255; rest identical to DefaultPreprocessor."""

    def __init__(self, window_level=40, window_width=400, verbose=True):
        super().__init__(verbose=verbose)
        self.normalizer = BiomedParseCTNormalization(window_level, window_width)

    def _normalize(self, data, seg, configuration_manager, foreground_intensity_properties_per_channel):
        for c in range(data.shape[0]):
            data[c] = self.normalizer.run(data[c], seg[0] if seg is not None else None)
        return data

    def run_case_npy(self, data, seg, properties, plans_manager, configuration_manager, dataset_json):
        original_shape = data.shape[1:]
        original_spacing = properties.get("spacing", [1.0, 1.0, 1.0])
        data, seg, properties = super().run_case_npy(
            data, seg, properties, plans_manager, configuration_manager, dataset_json
        )
        data = data.astype(np.uint8)
        properties["original_shape_before_transpose"] = original_shape
        properties["original_spacing_before_transpose"] = original_spacing
        return data, seg, properties


class BiomedParseMRIPreprocessor(DefaultPreprocessor):
    """MRI: percentile stretch -> 0-255; rest identical to DefaultPreprocessor."""

    def __init__(self, low_percentile=1.0, high_percentile=99.0, verbose=True):
        super().__init__(verbose=verbose)
        self.normalizer = BiomedParseMRINormalization(low_percentile, high_percentile)

    def _normalize(self, data, seg, configuration_manager, foreground_intensity_properties_per_channel):
        for c in range(data.shape[0]):
            data[c] = self.normalizer.run(data[c], seg[0] if seg is not None else None)
        return data

    def run_case_npy(self, data, seg, properties, plans_manager, configuration_manager, dataset_json):
        original_shape = data.shape[1:]
        original_spacing = properties.get("spacing", [1.0, 1.0, 1.0])
        data, seg, properties = super().run_case_npy(
            data, seg, properties, plans_manager, configuration_manager, dataset_json
        )
        data = data.astype(np.uint8)
        properties["original_shape_before_transpose"] = original_shape
        properties["original_spacing_before_transpose"] = original_spacing
        return data, seg, properties


# ---------------------------------------------------------------------------
# Single-case processing (for multiprocessing)
# ---------------------------------------------------------------------------


def _process_single_case(args):
    (
        idx, image_file, seg_file, output_dir, overwrite_existing,
        window_level, window_width, low_percentile, high_percentile,
        dataset_name, configuration_name, plans_identifier, norm_mode,
    ) = args
    try:
        plans_file = join(nnUNet_preprocessed, dataset_name, plans_identifier + ".json")
        plans = load_json(plans_file)
        plans_manager = PlansManager(plans)
        cfg_name = resolve_configuration_name(plans_manager, configuration_name)
        configuration_manager = plans_manager.get_configuration(cfg_name)

        dataset_json_file = join(nnUNet_preprocessed, dataset_name, "dataset.json")
        dataset_json = load_json(dataset_json_file)

        if str(norm_mode).lower() == "mri":
            preprocessor = BiomedParseMRIPreprocessor(
                low_percentile=low_percentile, high_percentile=high_percentile, verbose=False
            )
        else:
            preprocessor = BiomedParseCTWindowPreprocessor(
                window_level=window_level, window_width=window_width, verbose=False
            )

        fn = os.path.basename(image_file[0])
        case_id = strip_modality_suffix(fn.replace(dataset_json.get("file_ending", ".nii.gz"), ""))
        output_filename = join(output_dir, case_id)
        if not overwrite_existing and isfile(output_filename + ".npz"):
            return f"Skipped {case_id} (exists)"

        data, seg, properties = preprocessor.run_case(
            image_file, seg_file, plans_manager, configuration_manager, dataset_json
        )
        np.savez_compressed(output_filename + ".npz", data=data, seg=seg)
        write_pickle(properties, output_filename + ".pkl")
        return f"OK {case_id}: data {data.shape}, seg {seg.shape}"
    except Exception as e:
        return f"ERROR [{idx}]: {e}"


# ---------------------------------------------------------------------------
# Discover train + optional test cases
# ---------------------------------------------------------------------------


def _collect_cases(
    dataset_name: str, dataset_json: dict, include_test: bool
) -> dict:
    raw_base = join(nnUNet_raw, dataset_name)
    dataset = get_filenames_of_train_images_and_targets(raw_base, dataset_json)
    if not include_test:
        return dataset

    im_ts = join(raw_base, "imagesTs")
    lb_ts = join(raw_base, "labelsTs")
    if not (os.path.isdir(im_ts) and os.path.isdir(lb_ts)):
        print("[preprocess] --include_test set but imagesTs/labelsTs not found; skipping.")
        return dataset

    fe = dataset_json["file_ending"]
    try:
        identifiers = list(get_identifiers_from_splitted_dataset_folder(im_ts, fe))
        images = create_lists_from_splitted_dataset_folder(im_ts, fe, identifiers)
    except Exception as e:
        print(f"[preprocess] Failed to read imagesTs: {e}")
        return dataset

    merged = dict(dataset)
    n_add = 0
    for case_id, im_list in zip(identifiers, images):
        seg_p = join(lb_ts, case_id + fe)
        if isfile(seg_p):
            merged[case_id] = {"images": im_list, "label": seg_p}
            n_add += 1
    print(f"[preprocess] Added {n_add} test cases (total {len(merged)}).")
    return merged


def _collect_test_only_cases(
    dataset_name: str,
    dataset_json: dict,
    imagesTs_dir: Optional[str] = None,
    labelsTs_dir: Optional[str] = None,
) -> dict:
    """Collect only imagesTs + labelsTs cases (for dedicated test preprocess)."""
    raw_base = join(nnUNet_raw, dataset_name)
    im_ts = imagesTs_dir or join(raw_base, "imagesTs")
    lb_ts = labelsTs_dir or join(raw_base, "labelsTs")
    if not (os.path.isdir(im_ts) and os.path.isdir(lb_ts)):
        raise FileNotFoundError(f"Need both imagesTs ({im_ts}) and labelsTs ({lb_ts})")

    fe = dataset_json["file_ending"]
    identifiers = list(get_identifiers_from_splitted_dataset_folder(im_ts, fe))
    images = create_lists_from_splitted_dataset_folder(im_ts, fe, identifiers)
    dataset = {}
    for case_id, im_list in zip(identifiers, images):
        seg_p = join(lb_ts, case_id + fe)
        if isfile(seg_p):
            dataset[case_id] = {"images": im_list, "label": seg_p}
    print(f"[preprocess] Found {len(dataset)} test cases.")
    return dataset


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _run_cases_parallel(args_list, num_processes: int):
    """Run _process_single_case in parallel or sequentially."""
    if num_processes > 1:
        with Pool(num_processes) as pool:
            results = list(tqdm(pool.imap(_process_single_case, args_list), total=len(args_list)))
    else:
        results = [_process_single_case(a) for a in tqdm(args_list)]
    for r in results:
        print(r)


def _make_args_list(dataset, out_dir, overwrite_existing, window_level, window_width,
                    low_percentile, high_percentile, dataset_name, cfg_name, plans_identifier, norm_mode):
    image_files = [dataset[k]["images"] for k in dataset]
    seg_files = [dataset[k]["label"] for k in dataset]
    return [
        (
            idx, img, seg, out_dir, overwrite_existing,
            window_level, window_width, low_percentile, high_percentile,
            dataset_name, cfg_name, plans_identifier, norm_mode,
        )
        for idx, (img, seg) in enumerate(zip(image_files, seg_files))
    ]


def preprocess_dataset(
    dataset_name_or_id: Union[int, str],
    output_dir: str,
    *,
    configuration_name: str = "auto",
    plans_identifier: str = "nnUNetPlans",
    norm_mode: str = "ct",
    window_level: int = 40,
    window_width: int = 400,
    low_percentile: float = 1.0,
    high_percentile: float = 99.0,
    num_processes: int = 8,
    overwrite_existing: bool = False,
    include_test: bool = False,
    test_only: bool = False,
    imagesTs_dir: Optional[str] = None,
    labelsTs_dir: Optional[str] = None,
    splits_final_json: Optional[str] = None,
    fold: int = 0,
):
    """
    Preprocess a dataset for BiomedParse branch.

    Output is organized into subdirectories:
        output_dir/train/   — training split cases
        output_dir/val/     — validation split cases
        output_dir/test/    — test cases (imagesTs/labelsTs)

    The train/val split is read from splits_final.json (nnUNet fold).
    If splits_final_json is not provided, all imagesTr cases go to train/.

    Args:
        test_only: if True, only preprocess imagesTs/labelsTs into test/.
        include_test: if True and not test_only, also preprocess test cases.
        splits_final_json: path to nnUNet splits_final.json for train/val split.
        fold: which fold to use from splits_final.json.
    """
    dataset_name = maybe_convert_to_dataset_name(dataset_name_or_id)

    plans_file = join(nnUNet_preprocessed, dataset_name, plans_identifier + ".json")
    if not isfile(plans_file):
        raise FileNotFoundError(f"Plans not found: {plans_file}")
    plans = load_json(plans_file)
    plans_manager = PlansManager(plans)
    cfg_name = resolve_configuration_name(plans_manager, configuration_name)

    dataset_json_file = join(nnUNet_preprocessed, dataset_name, "dataset.json")
    if not isfile(dataset_json_file):
        raise FileNotFoundError(f"dataset.json not found: {dataset_json_file}")
    dataset_json = load_json(dataset_json_file)

    common_kwargs = dict(
        overwrite_existing=overwrite_existing,
        window_level=window_level, window_width=window_width,
        low_percentile=low_percentile, high_percentile=high_percentile,
        dataset_name=dataset_name, cfg_name=cfg_name,
        plans_identifier=plans_identifier, norm_mode=norm_mode,
    )

    print(f"norm_mode={norm_mode}, WL={window_level}, WW={window_width}")

    # ---- test only mode ----
    if test_only:
        test_dir = join(output_dir, "test")
        maybe_mkdir_p(test_dir)
        dataset = _collect_test_only_cases(
            dataset_name, dataset_json,
            imagesTs_dir=imagesTs_dir, labelsTs_dir=labelsTs_dir,
        )
        print(f"\n[test] Preprocessing {len(dataset)} cases -> {test_dir}")
        args_list = _make_args_list(dataset, test_dir, **common_kwargs)
        _run_cases_parallel(args_list, num_processes)
        print(f"\nDone! test/ -> {test_dir}")
        return

    # ---- collect all training cases (imagesTr/labelsTr) ----
    all_train_dataset = _collect_cases(dataset_name, dataset_json, include_test=False)

    # Split into train / val using splits_final.json
    train_ids: Optional[set] = None
    val_ids: Optional[set] = None

    if splits_final_json is None:
        candidate = join(
            nnUNet_preprocessed, dataset_name, "splits_final.json"
        )
        if isfile(candidate):
            splits_final_json = candidate

    if splits_final_json and isfile(splits_final_json):
        splits = load_json(splits_final_json)
        if isinstance(splits, list) and len(splits) > fold:
            train_ids = set(splits[fold].get("train", []))
            val_ids = set(splits[fold].get("val", []))
            print(f"Using splits_final.json fold={fold}: "
                  f"{len(train_ids)} train, {len(val_ids)} val")

    if train_ids is not None and val_ids is not None:
        train_dataset = {k: v for k, v in all_train_dataset.items() if k in train_ids}
        val_dataset = {k: v for k, v in all_train_dataset.items() if k in val_ids}
        orphan = {k: v for k, v in all_train_dataset.items()
                  if k not in train_ids and k not in val_ids}
        if orphan:
            print(f"  {len(orphan)} cases not in any split -> added to train/")
            train_dataset.update(orphan)
    else:
        print("No splits_final.json found; all imagesTr cases go to train/")
        train_dataset = all_train_dataset
        val_dataset = {}

    # Process train/
    train_dir = join(output_dir, "train")
    maybe_mkdir_p(train_dir)
    if train_dataset:
        print(f"\n[train] Preprocessing {len(train_dataset)} cases -> {train_dir}")
        args_list = _make_args_list(train_dataset, train_dir, **common_kwargs)
        _run_cases_parallel(args_list, num_processes)

    # Process val/
    val_dir = join(output_dir, "val")
    maybe_mkdir_p(val_dir)
    if val_dataset:
        print(f"\n[val] Preprocessing {len(val_dataset)} cases -> {val_dir}")
        args_list = _make_args_list(val_dataset, val_dir, **common_kwargs)
        _run_cases_parallel(args_list, num_processes)

    # Process test/ (optional)
    if include_test:
        test_dir = join(output_dir, "test")
        maybe_mkdir_p(test_dir)
        test_dataset = _collect_test_only_cases(
            dataset_name, dataset_json,
            imagesTs_dir=imagesTs_dir, labelsTs_dir=labelsTs_dir,
        )
        if test_dataset:
            print(f"\n[test] Preprocessing {len(test_dataset)} cases -> {test_dir}")
            args_list = _make_args_list(test_dataset, test_dir, **common_kwargs)
            _run_cases_parallel(args_list, num_processes)

    print(f"\nDone! Output: {output_dir}/{{train,val,test}}")


# ---------------------------------------------------------------------------
# Online preprocessing (for sliding-window eval without cached .npz)
# ---------------------------------------------------------------------------


def preprocess_case_online(
    image_files: List[str],
    seg_file: Optional[str],
    plans_path: str,
    dataset_json_path: str,
    norm_mode: str = "ct",
    window_level: float = 40,
    window_width: float = 400,
    low_percentile: float = 1.0,
    high_percentile: float = 99.0,
    configuration_name: str = "auto",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict, dict]:
    """
    On-the-fly preprocess a single case for both nnUNet and BiomedParse branches.

    Returns:
        data_nn, seg_nn, data_bp, seg_bp, props_nn, props_bp
    """
    plans = load_json(plans_path)
    dataset_json = load_json(dataset_json_path)
    plans_manager = PlansManager(plans)
    cfg_name = resolve_configuration_name(plans_manager, configuration_name)
    configuration_manager = plans_manager.get_configuration(cfg_name)

    nn_preprocessor = DefaultPreprocessor(verbose=False)
    data_nn, seg_nn, props_nn = nn_preprocessor.run_case(
        image_files, seg_file, plans_manager, configuration_manager, dataset_json
    )

    if norm_mode.lower() == "mri":
        bp_preprocessor = BiomedParseMRIPreprocessor(
            low_percentile=low_percentile, high_percentile=high_percentile, verbose=False
        )
    else:
        bp_preprocessor = BiomedParseCTWindowPreprocessor(
            window_level=window_level, window_width=window_width, verbose=False
        )
    data_bp, seg_bp, props_bp = bp_preprocessor.run_case(
        image_files, seg_file, plans_manager, configuration_manager, dataset_json
    )

    return data_nn, seg_nn, data_bp, seg_bp, props_nn, props_bp
