"""nnUNet-geometry-aligned BiomedParse preprocessing for MRI/LGE data."""

from __future__ import annotations

import json
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..config import ensure_nnunet_on_path
from ..utils.normalization import BiomedParseMRINormalization


class AlignedBiomedParsePreprocessor:
    """Apply nnUNet crop/resampling with BiomedParse MRI normalization.

    Geometry is inherited from the frozen nnUNet expert's plans. Only the
    intensity normalization is replaced by the LGE/MRI percentile mapping.
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
        normalizer = BiomedParseMRINormalization(
            low_percentile,
            high_percentile,
        )

        class _MRIAlignedPreprocessor(DefaultPreprocessor):
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

        self.preprocessor = _MRIAlignedPreprocessor()

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
        # Match the reference offline preprocessor exactly: clip, then uint8
        # cast (truncation rather than rounding).
        bp_u8 = np.clip(data[modality], 0.0, 255.0).astype(np.uint8)
        seg_zyx = None
        if seg is not None:
            seg_array = np.asarray(seg)
            seg_zyx = (
                seg_array[0] if seg_array.ndim == 4 else seg_array
            ).astype(np.int16, copy=False)
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
