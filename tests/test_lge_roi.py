import json

import torch

from oodka.data.lge_roi import (
    ROICache,
    ROICoordinates,
    ROIGenerator,
    crop_and_resize_batch,
    remap_grouped_labels,
    restore_roi_logits,
)


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
