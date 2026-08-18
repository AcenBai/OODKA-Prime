import json

import torch

from oodka.data.lge_roi import (
    ROICache,
    ROICoordinates,
    ROIGenerator,
    augment_lge_batch,
    crop_and_resize_batch,
    hard_switch_foreground_logits,
    remap_grouped_labels,
    restore_roi_logits,
    roi_prompt_visibility,
)
from run_eval_lge_roi import _hierarchical_foreground_logits


def test_roi_generation_expands_about_center_and_clips():
    probability = torch.zeros(32, 40)
    probability[8:16, 2:10] = 0.9
    roi = ROIGenerator(threshold=0.3, expand=1.25).from_probability(probability)
    assert not roi.fallback
    assert 0 <= roi.x0 < roi.x1 <= 40
    assert 0 <= roi.y0 < roi.y1 <= 32
    assert roi.x0 <= 2 and roi.x1 >= 10
    assert roi.y0 <= 8 and roi.y1 >= 16


def test_empty_roi_has_full_image_fallback():
    roi = ROIGenerator(fallback="full").from_probability(torch.zeros(17, 19))
    assert roi == ROICoordinates(0, 0, 19, 17, fallback=True)


def test_lge_group_remapping():
    labels = torch.tensor([[[[0, 1, 2], [3, 4, 5], [-1, 0, 0]]]])
    final = remap_grouped_labels(labels, ((3,), (5,), (4,), (1, 2)))
    expected = torch.tensor([[[[0, 4, 4], [1, 3, 2], [-1, 0, 0]]]])
    assert torch.equal(final, expected)


def test_lge_split_pathology_group_remapping():
    labels = torch.tensor([[[[0, 1, 2], [3, 4, 5], [-1, 0, 0]]]])
    final = remap_grouped_labels(labels, ((3,), (5,), (4,), (1,), (2,)))
    expected = torch.tensor([[[[0, 4, 5], [1, 3, 2], [-1, 0, 0]]]])
    assert torch.equal(final, expected)


def test_crop_and_restore_coordinate_contract():
    image = torch.arange(16 * 16).reshape(1, 1, 1, 16, 16).float()
    batch = {
        "nnunet_image": image,
        "biomedparse_image": image.repeat(1, 1, 3, 1, 1),
        "gt": torch.zeros((1, 1, 16, 16), dtype=torch.long),
        "valid_z": torch.ones((1, 1), dtype=torch.bool),
    }
    roi = ROICoordinates(4, 5, 12, 13)
    cropped = crop_and_resize_batch(batch, [roi])
    assert cropped["nnunet_image"].shape == image.shape
    logits = torch.ones((1, 2, 1, 16, 16))
    restored = restore_roi_logits(logits, [roi], (16, 16))
    assert restored.shape == logits.shape
    assert torch.all(restored[:, :, :, :5] < 0)
    assert torch.allclose(restored[0, :, 0, 5:13, 4:12], torch.ones(2, 8, 8))


def test_roi_cache_roundtrip(tmp_path):
    cache = ROICache()
    cache.set("case", 3, ROICoordinates(1, 2, 8, 9, fallback=False))
    path = tmp_path / "cache.json"
    cache.save(path, metadata={"threshold": 0.3})
    payload = json.loads(path.read_text())
    assert payload["metadata"]["threshold"] == 0.3
    restored = ROICache.load(path)
    assert restored.get("case", 3) == cache.get("case", 3)


def test_hierarchical_decision_only_refines_total_myo():
    anatomy = torch.full((1, 3, 1, 1, 4), -5.0)
    # coarse outputs: background, LV, RV, total-MYO
    anatomy[0, 0, 0, 0, 1] = 4.0
    anatomy[0, 1, 0, 0, 2] = 4.0
    anatomy[0, 2, 0, 0, 3] = 4.0
    refinement = torch.zeros((1, 2, 1, 1, 4))
    refinement[0, 1, 0, 0] = 10.0

    foreground = _hierarchical_foreground_logits(anatomy, refinement)
    background = torch.zeros_like(foreground[:, :1])
    labels = torch.cat([background, foreground], dim=1).argmax(dim=1)
    assert labels.flatten().tolist() == [0, 1, 2, 4]


def test_v2_hard_switch_uses_pass1_outside_and_pass2_inside():
    anatomy = torch.full((1, 3, 1, 4, 4), -5.0)
    anatomy[:, 0] = 6.0
    refinement = torch.full((1, 4, 1, 4, 4), -5.0)
    refinement[:, 3] = 8.0
    roi = ROICoordinates(1, 1, 3, 3)
    foreground = hard_switch_foreground_logits(anatomy, refinement, [roi])
    labels = torch.cat([torch.zeros_like(foreground[:, :1]), foreground], dim=1)
    labels = labels.argmax(dim=1)[0, 0]
    assert labels[0, 0] == 1
    assert torch.all(labels[1:3, 1:3] == 4)


def test_v3_hard_switch_supports_separate_edema_channel():
    anatomy = torch.full((1, 3, 1, 4, 4), -5.0)
    anatomy[:, 1] = 6.0
    refinement = torch.full((1, 5, 1, 4, 4), -5.0)
    refinement[:, 4] = 8.0
    roi = ROICoordinates(1, 1, 3, 3)
    foreground = hard_switch_foreground_logits(anatomy, refinement, [roi])
    labels = torch.cat([torch.zeros_like(foreground[:, :1]), foreground], dim=1)
    labels = labels.argmax(dim=1)[0, 0]
    assert labels[0, 0] == 2
    assert torch.all(labels[1:3, 1:3] == 5)


def test_roi_visibility_skips_truncated_positive_but_keeps_true_negative():
    gt = torch.zeros((1, 1, 8, 8), dtype=torch.long)
    gt[0, 0, 1:5, 1:5] = 3
    roi = ROICoordinates(6, 0, 8, 8)
    valid = roi_prompt_visibility(gt, [roi], ((3,), (5,)), min_coverage=0.01)
    assert valid.tolist() == [[False, True]]
    partial = roi_prompt_visibility(
        gt, [ROICoordinates(0, 0, 2, 8)], ((3,),), min_coverage=0.01
    )
    assert partial.tolist() == [[True]]


def test_lge_augmentation_preserves_shapes_and_discrete_labels():
    image = torch.randn(2, 1, 1, 16, 16)
    gt = torch.randint(0, 6, (2, 1, 16, 16))
    batch = {
        "nnunet_image": image,
        "biomedparse_image": image.repeat(1, 1, 3, 1, 1),
        "gt": gt,
    }
    augmented = augment_lge_batch(
        batch,
        rotation_degrees=10.0,
        scale_min=0.95,
        scale_max=1.05,
        translation_fraction=0.05,
        horizontal_flip_probability=0.5,
        vertical_flip_probability=0.2,
        intensity_probability=0.0,
    )
    assert augmented["nnunet_image"].shape == image.shape
    assert augmented["biomedparse_image"].shape == (2, 1, 3, 16, 16)
    assert augmented["gt"].shape == gt.shape
    assert set(augmented["gt"].unique().tolist()) <= set(gt.unique().tolist())
