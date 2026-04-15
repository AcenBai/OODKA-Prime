"""Image normalization strategies for BiomedParse preprocessing."""

from __future__ import annotations

import numpy as np


class BiomedParseCTNormalization:
    """Map HU values to [0, 255] using window level / window width."""

    def __init__(self, window_level: float = 40, window_width: float = 400):
        self.window_level = window_level
        self.window_width = window_width

    def run(self, image: np.ndarray, seg: np.ndarray = None) -> np.ndarray:
        image = image.astype(np.float32, copy=False)
        lower = self.window_level - self.window_width / 2.0
        upper = self.window_level + self.window_width / 2.0
        image = np.clip(image, lower, upper)
        image = (image - lower) / (upper - lower) * 255.0
        return np.clip(image, 0, 255)


class BiomedParseMRINormalization:
    """Map MRI intensities to [0, 255] via foreground percentile stretching."""

    def __init__(self, low_percentile: float = 1.0, high_percentile: float = 99.0):
        self.low_percentile = float(low_percentile)
        self.high_percentile = float(high_percentile)

    def run(self, image: np.ndarray, seg: np.ndarray = None) -> np.ndarray:
        image = image.astype(np.float32, copy=False)
        fg_threshold = np.percentile(image, 0.1)
        mask = image > fg_threshold
        foreground = image[mask] if np.any(mask) else image

        lo = np.percentile(foreground, self.low_percentile)
        hi = np.percentile(foreground, self.high_percentile)
        if hi <= lo:
            hi = lo + 1.0

        image = np.clip(image, lo, hi)
        image = (image - lo) / (hi - lo) * 255.0
        return np.clip(image, 0, 255)
