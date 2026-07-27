import numpy as np
import pytest

from oodka.eval.eval_nnunet import _coerce_nnunet_segmentation


def test_coerce_nnunet_segmentation_accepts_public_api_shapes():
    expected = (5, 7, 9)
    direct = np.zeros(expected, dtype=np.uint8)
    leading_singleton = direct[None]

    assert _coerce_nnunet_segmentation(direct, expected).shape == expected
    assert (
        _coerce_nnunet_segmentation((leading_singleton, {}), expected).shape
        == expected
    )


def test_coerce_nnunet_segmentation_rejects_logits_or_2d_argmax():
    expected = (5, 7, 9)
    with pytest.raises(ValueError, match="unexpected shape"):
        _coerce_nnunet_segmentation(np.zeros((8, *expected)), expected)
    with pytest.raises(ValueError, match="unexpected shape"):
        _coerce_nnunet_segmentation(np.zeros(expected[1:]), expected)
