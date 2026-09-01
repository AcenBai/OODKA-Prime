"""nnUNet-geometry-aligned BiomedParse preprocessing for CT and MRI."""

from __future__ import annotations

import json
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..config import ensure_nnunet_on_path
from ..utils.normalization import BiomedParseMRINormalization


class AlignedBiomedParsePreprocessor:
    """Apply nnUNet crop/resampling with modality-aware BiomedParse normalization.

    Geometry is inherited from the frozen nnUNet expert's plans. Only the
    intensity normalization is replaced by CT windowing or MRI percentiles.
    This gives the expert and student exactly aligned spatial tensors.
    """

    def __init__(
        self,
        *,
        plans_path: str,
        dataset_json_path: str,
        configuration_name: str = "2d",
        low_percentile: float = 1.0,
        high_percentile: float = 99.0,
        norm_mode: str = "mri",
        window_level: float = 40.0,
        window_width: float = 400.0,
    ):
        ensure_nnunet_on_path()
        from nnunetv2.preprocessing.preprocessors.default_preprocessor import (
            DefaultPreprocessor,
        )
        from nnunetv2.utilities.plans_handling.plans_handler import PlansManager

        with open(plans_path, encoding="utf-8") as file_handle:
            plans = json.load(file_handle)
        with open(dataset_json_path, encoding="utf-8") as file_handle:
            self.dataset_json = json.load(file_handle)
        self.plans_manager = PlansManager(plans)
        self.configuration_manager = self.plans_manager.get_configuration(
            configuration_name
        )
        norm_mode = str(norm_mode).lower()
        if norm_mode == "mri":
            normalizer = BiomedParseMRINormalization(
                low_percentile,
                high_percentile,
            )
        elif norm_mode == "ct":
            lower = float(window_level) - float(window_width) / 2.0
            upper = float(window_level) + float(window_width) / 2.0
            if upper <= lower:
                raise ValueError("window_width must be positive")

            class _CTWindowNormalization:
                @staticmethod
                def run(image):
                    work = np.asarray(image, dtype=np.float32).copy()
                    np.clip(work, lower, upper, out=work)
                    work -= lower
                    work *= 255.0 / (upper - lower)
                    return work

            normalizer = _CTWindowNormalization()
        else:
            raise ValueError("norm_mode must be 'ct' or 'mri'")

        class _AlignedPreprocessor(DefaultPreprocessor):
            def __init__(self):
                super().__init__(verbose=False)

            def _normalize(
                self,
                data,
                seg,
                configuration_manager,
                foreground_intensity_properties_per_channel,
            ):
                del (
                    seg,
                    configuration_manager,
                    foreground_intensity_properties_per_channel,
                )
                for channel_index in range(data.shape[0]):
                    data[channel_index] = normalizer.run(data[channel_index])
                return data

        self.preprocessor = _AlignedPreprocessor()

    def run_case(
        self,
        image_files: List[str],
        seg_file: Optional[str],
        *,
        modality: int = 0,
    ) -> Tuple[np.ndarray, Optional[np.ndarray], dict]:
        """Return aligned ``bp_u8[Z,H,W]``, optional seg, and properties."""
        data, seg, properties = self.preprocessor.run_case(
            image_files,
            seg_file,
            self.plans_manager,
            self.configuration_manager,
            self.dataset_json,
        )
        data = np.asarray(data)
        if data.ndim != 4 or not 0 <= modality < data.shape[0]:
            raise ValueError(
                f"Expected aligned data [C,Z,H,W], got {data.shape} "
                f"for modality={modality}"
            )
        bp_u8 = np.clip(data[modality], 0.0, 255.0).astype(np.uint8)
        seg_zyx = None
        if seg is not None:
            seg_array = np.asarray(seg)
            seg_zyx = (seg_array[0] if seg_array.ndim == 4 else seg_array).astype(
                np.int16, copy=False
            )
        return bp_u8, seg_zyx, properties

    def prompt_logits_to_raw_segmentation(
        self,
        prompt_logits: np.ndarray,
        prompt_to_class_id: Dict[int, int],
        properties: dict,
    ) -> np.ndarray:
        """Restore prompt logits from preprocessed geometry to raw geometry."""
        ensure_nnunet_on_path()
        from nnunetv2.inference.export_prediction import (
            convert_predicted_logits_to_segmentation_with_correct_shape,
        )

        if prompt_logits.ndim != 4:
            raise ValueError(
                f"prompt_logits must be [P,Z,H,W], got {prompt_logits.shape}"
            )
        label_manager = self.plans_manager.get_label_manager(self.dataset_json)
        full_logits = np.zeros(
            (label_manager.num_segmentation_heads, *prompt_logits.shape[1:]),
            dtype=np.float32,
        )
        for prompt_index, class_id in prompt_to_class_id.items():
            if not 0 <= int(class_id) < full_logits.shape[0]:
                raise ValueError(
                    f"class_id={class_id} exceeds {full_logits.shape[0]} "
                    "segmentation heads"
                )
            full_logits[int(class_id)] = prompt_logits[int(prompt_index)]
        restored = convert_predicted_logits_to_segmentation_with_correct_shape(
            full_logits,
            self.plans_manager,
            self.configuration_manager,
            label_manager,
            properties,
            return_probabilities=False,
            num_threads_torch=1,
        )
        return np.asarray(restored, dtype=np.int16)

    def prompt_values_to_raw_geometry(
        self,
        prompt_values: np.ndarray,
        properties: dict,
    ) -> np.ndarray:
        """Restore continuous ``[P,Z,H,W]`` values without a task nonlinearity.

        This follows nnUNet's probability-resampling and geometry restoration
        path but deliberately does not apply softmax. It is therefore suitable
        for independent open-vocabulary prompt logits.
        """
        if prompt_values.ndim != 4:
            raise ValueError(
                f"prompt_values must be [P,Z,H,W], got {prompt_values.shape}"
            )
        spacing_transposed = [
            properties["spacing"][index]
            for index in self.plans_manager.transpose_forward
        ]
        target_shape = properties["shape_after_cropping_and_before_resampling"]
        configured_spacing = self.configuration_manager.spacing
        current_spacing = (
            configured_spacing
            if len(configured_spacing) == len(target_shape)
            else [spacing_transposed[0], *configured_spacing]
        )
        restored = self.configuration_manager.resampling_fn_probabilities(
            prompt_values,
            target_shape,
            current_spacing,
            spacing_transposed,
        )
        if not hasattr(restored, "device"):
            import torch

            restored = torch.as_tensor(restored)
        label_manager = self.plans_manager.get_label_manager(self.dataset_json)
        restored = label_manager.revert_cropping_on_probabilities(
            restored,
            properties["bbox_used_for_cropping"],
            properties["shape_before_cropping"],
        )
        if hasattr(restored, "detach"):
            restored = restored.detach().cpu().numpy()
        else:
            restored = np.asarray(restored)
        axes = [0] + [index + 1 for index in self.plans_manager.transpose_backward]
        return restored.transpose(axes).astype(np.float32, copy=False)
