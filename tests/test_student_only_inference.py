import importlib
import sys

import numpy as np

from oodka.eval.eval_oodka import _make_block_batch


def test_oodka_eval_import_does_not_import_nnunetv2():
    sys.modules.pop("nnunetv2", None)
    importlib.import_module("run_eval_oodka")
    assert "nnunetv2" not in sys.modules


def test_student_block_builder_has_no_expert_input_and_masks_tail():
    volume = np.arange(5 * 8 * 8, dtype=np.uint8).reshape(5, 8, 8)
    bp, valid, counts = _make_block_batch(
        volume, [0, 4], block_z=3, image_size=16
    )
    assert bp.shape == (2, 3, 3, 16, 16)
    assert valid.tolist() == [[True, True, True], [True, False, False]]
    assert counts == [3, 1]
