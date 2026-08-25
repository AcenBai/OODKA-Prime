import torch

from oodka.train.forward import _compute_segmentation_loss_and_metrics


def _seg_loss(prompt_count: int, reduction: str) -> torch.Tensor:
    logits = torch.zeros((2, prompt_count, 1, 4, 4))
    gt = torch.zeros((2, 1, 4, 4), dtype=torch.long)
    valid_z = torch.ones((2, 1), dtype=torch.bool)
    class_ids = torch.arange(1, prompt_count + 1)
    loss, _, _ = _compute_segmentation_loss_and_metrics(
        logits,
        gt,
        valid_z,
        class_ids,
        prompt_reduction=reduction,
    )
    return loss


def test_prompt_mean_is_invariant_to_prompt_count():
    one = _seg_loss(1, "prompt_mean")
    seven = _seg_loss(7, "prompt_mean")
    assert torch.allclose(one, seven)


def test_prompt_sum_scales_with_prompt_count():
    one = _seg_loss(1, "sum")
    seven = _seg_loss(7, "sum")
    assert torch.allclose(seven, 7.0 * one)
