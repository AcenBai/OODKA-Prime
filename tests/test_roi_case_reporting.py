"""Small end-to-end scoring fixtures for ROI evaluation exports."""

import numpy as np
import SimpleITK as sitk

from oodka.eval.roi_case_reporting import (
    remap_gt,
    score_exclusive_case,
    score_independent_case,
)


class _IdentityPreprocessor:
    @staticmethod
    def prompt_logits_to_raw_segmentation(logits, mapping, properties):
        del properties
        foreground = logits[0] > 0
        return np.where(foreground, mapping[0], 0).astype(np.int16)


def test_case_reporting_preserves_metrics_order_and_nifti_shape(tmp_path):
    target = np.zeros((1, 4, 4), dtype=np.int16)
    target[0, 1:3, 1:3] = 1
    source = target.copy()
    assert np.array_equal(remap_gt(source, ((1,),)), target)
    logits = np.full((1, 1, 4, 4), -5.0, dtype=np.float32)
    logits[0, 0, 1:3, 1:3] = 5.0
    reference_path = tmp_path / "case.nii.gz"
    sitk.WriteImage(sitk.GetImageFromArray(target), str(reference_path))
    pred_dir = tmp_path / "exclusive"
    pred_dir.mkdir()
    binary_dir = tmp_path / "binary"
    binary_dir.mkdir()
    common = dict(
        case_id="case",
        final_prompt_logits=logits,
        target=target,
        spacing=(1.0, 1.0, 1.0),
        preprocessor=_IdentityPreprocessor(),
        properties={},
        raw_shape=target.shape,
        postprocess="none",
        label_path=str(reference_path),
        ending=".nii.gz",
    )
    confusion = np.zeros((2, 2), dtype=np.int64)
    exclusive = score_exclusive_case(
        **common, confusion=confusion, pred_dir=str(pred_dir)
    )
    assert exclusive["dice_mean_gt"] == 1.0
    assert list(exclusive)[:3] == ["case_id", "dice_mean_gt", "dice_1"]
    assert confusion.tolist() == [[12, 0], [0, 4]]
    assert sitk.GetArrayFromImage(sitk.ReadImage(str(pred_dir / "case.nii.gz"))).shape == target.shape

    predicted = np.zeros(2, dtype=np.int64)
    actual = np.zeros(2, dtype=np.int64)
    independent = score_independent_case(
        **common,
        threshold=0.5,
        independent_dirs={1: str(binary_dir)},
        independent_predicted=predicted,
        independent_target=actual,
    )
    assert independent["dice_mean_gt"] == 1.0
    assert list(independent)[:3] == ["case_id", "dice_mean_gt", "dice_1"]
    assert predicted.tolist() == actual.tolist() == [0, 4]
    assert sitk.GetArrayFromImage(sitk.ReadImage(str(binary_dir / "case.nii.gz"))).shape == target.shape
