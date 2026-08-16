"""Predicted-myocardium ROI utilities for the LGE mixed-training experiment."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, Sequence

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class ROICoordinates:
    """Exclusive XYXY coordinates in the model-input pixel grid."""

    x0: int
    y0: int
    x1: int
    y1: int
    fallback: bool = False

    @property
    def width(self) -> int:
        return self.x1 - self.x0

    @property
    def height(self) -> int:
        return self.y1 - self.y0


class ROIGenerator:
    def __init__(
        self,
        threshold: float = 0.3,
        expand: float = 1.25,
        fallback: str = "full",
        min_size: int = 8,
    ) -> None:
        if not 0.0 < threshold < 1.0:
            raise ValueError("ROI threshold must be in (0,1)")
        if expand < 1.0:
            raise ValueError("ROI expansion must be >= 1")
        if fallback not in {"full", "center"}:
            raise ValueError("ROI fallback must be 'full' or 'center'")
        self.threshold = float(threshold)
        self.expand = float(expand)
        self.fallback = fallback
        self.min_size = int(min_size)

    @staticmethod
    def _expanded_interval(
        low: int,
        high: int,
        limit: int,
        factor: float,
        min_size: int,
    ) -> tuple[int, int]:
        center = 0.5 * (float(low) + float(high))
        size = max(float(high - low) * factor, float(min_size))
        out_low = int(round(center - 0.5 * size))
        out_high = int(round(center + 0.5 * size))
        if out_low < 0:
            out_high -= out_low
            out_low = 0
        if out_high > limit:
            out_low -= out_high - limit
            out_high = limit
        out_low = max(0, out_low)
        out_high = min(limit, max(out_low + 1, out_high))
        return out_low, out_high

    def from_probability(self, probability: torch.Tensor) -> ROICoordinates:
        if probability.ndim != 2:
            raise ValueError(f"Expected probability [H,W], got {probability.shape}")
        height, width = (int(v) for v in probability.shape)
        locations = torch.nonzero(probability >= self.threshold, as_tuple=False)
        if locations.numel() == 0:
            if self.fallback == "full":
                return ROICoordinates(0, 0, width, height, fallback=True)
            side_h = max(self.min_size, int(round(height * 0.75)))
            side_w = max(self.min_size, int(round(width * 0.75)))
            y0 = max(0, (height - side_h) // 2)
            x0 = max(0, (width - side_w) // 2)
            return ROICoordinates(
                x0,
                y0,
                min(width, x0 + side_w),
                min(height, y0 + side_h),
                fallback=True,
            )
        y0 = int(locations[:, 0].min().item())
        y1 = int(locations[:, 0].max().item()) + 1
        x0 = int(locations[:, 1].min().item())
        x1 = int(locations[:, 1].max().item()) + 1
        x0, x1 = self._expanded_interval(
            x0, x1, width, self.expand, self.min_size
        )
        y0, y1 = self._expanded_interval(
            y0, y1, height, self.expand, self.min_size
        )
        return ROICoordinates(x0, y0, x1, y1, fallback=False)


class ROICache:
    """Slice-keyed fixed ROI cache generated after anatomy warm-up."""

    def __init__(self, coordinates: Dict[str, ROICoordinates] | None = None):
        self.coordinates = dict(coordinates or {})

    @staticmethod
    def key(case_id: str, z_index: int) -> str:
        return f"{case_id}:{int(z_index)}"

    def set(self, case_id: str, z_index: int, roi: ROICoordinates) -> None:
        self.coordinates[self.key(case_id, z_index)] = roi

    def get(self, case_id: str, z_index: int) -> ROICoordinates:
        key = self.key(case_id, z_index)
        if key not in self.coordinates:
            raise KeyError(f"ROI cache has no entry for {key}")
        return self.coordinates[key]

    def save(self, path: str | Path, metadata: dict | None = None) -> None:
        payload = {
            "metadata": dict(metadata or {}),
            "coordinates": {
                key: asdict(value) for key, value in sorted(self.coordinates.items())
            },
        }
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "ROICache":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            {
                key: ROICoordinates(**value)
                for key, value in payload["coordinates"].items()
            }
        )


def remap_grouped_labels(
    labels: torch.Tensor,
    groups: Sequence[Sequence[int]],
) -> torch.Tensor:
    """Map disjoint source-label groups to compact labels 1..P."""
    output = torch.zeros_like(labels)
    output[labels < 0] = -1
    occupied = torch.zeros_like(labels, dtype=torch.bool)
    for target_id, source_ids in enumerate(groups, start=1):
        mask = torch.zeros_like(labels, dtype=torch.bool)
        for source_id in source_ids:
            mask |= labels == int(source_id)
        if (occupied & mask).any():
            raise ValueError("Grouped labels must be disjoint")
        output[mask] = int(target_id)
        occupied |= mask
    return output


def crop_and_resize_batch(
    batch_data: dict,
    rois: Sequence[ROICoordinates],
) -> dict:
    """Apply per-slice ROIs identically to both visual branches and GT.

    The experiment is intentionally Z=1, so each batch element has one ROI.
    Crops are resized back to the original model-input H/W.
    """
    nn_image = batch_data["nnunet_image"]
    bp_image = batch_data["biomedparse_image"]
    gt = batch_data["gt"]
    if nn_image.ndim != 5 or bp_image.ndim != 5 or gt.ndim != 4:
        raise ValueError("Unexpected batch tensor rank for ROI crop")
    batch_size, block_z = nn_image.shape[:2]
    if block_z != 1 or len(rois) != batch_size:
        raise ValueError("ROI mixed training currently requires block_z=1")
    out_h, out_w = gt.shape[-2:]
    nn_crops = []
    bp_crops = []
    gt_crops = []
    for index, roi in enumerate(rois):
        if roi.width <= 0 or roi.height <= 0:
            raise ValueError(f"Invalid ROI: {roi}")
        nn_crop = nn_image[index, 0, :, roi.y0 : roi.y1, roi.x0 : roi.x1]
        bp_crop = bp_image[index, 0, :, roi.y0 : roi.y1, roi.x0 : roi.x1]
        gt_crop = gt[index, 0, roi.y0 : roi.y1, roi.x0 : roi.x1]
        nn_crops.append(
            F.interpolate(
                nn_crop[None], size=(out_h, out_w), mode="bilinear",
                align_corners=False,
            )[0]
        )
        bp_crops.append(
            F.interpolate(
                bp_crop[None], size=(out_h, out_w), mode="bilinear",
                align_corners=False,
            )[0]
        )
        gt_crops.append(
            F.interpolate(
                gt_crop[None, None].float(), size=(out_h, out_w),
                mode="nearest",
            )[0, 0].to(gt.dtype)
        )
    output = dict(batch_data)
    output["nnunet_image"] = torch.stack(nn_crops, dim=0)[:, None]
    output["biomedparse_image"] = torch.stack(bp_crops, dim=0)[:, None]
    output["gt"] = torch.stack(gt_crops, dim=0)[:, None]
    return output


def restore_roi_logits(
    roi_logits: torch.Tensor,
    rois: Sequence[ROICoordinates],
    output_size: tuple[int, int],
    outside_logit: float = -20.0,
) -> torch.Tensor:
    """Restore ``[B,P,1,h,w]`` ROI logits to full-image coordinates."""
    if roi_logits.ndim != 5 or roi_logits.shape[2] != 1:
        raise ValueError("ROI logits must be [B,P,1,H,W]")
    batch_size, prompts = roi_logits.shape[:2]
    if len(rois) != batch_size:
        raise ValueError("ROI count does not match batch")
    height, width = output_size
    output = roi_logits.new_full(
        (batch_size, prompts, 1, height, width), float(outside_logit)
    )
    for index, roi in enumerate(rois):
        resized = F.interpolate(
            roi_logits[index, :, 0][None],
            size=(roi.height, roi.width),
            mode="bilinear",
            align_corners=False,
        )[0]
        output[index, :, 0, roi.y0 : roi.y1, roi.x0 : roi.x1] = resized
    return output


def roi_diagnostics(
    rois: Iterable[ROICoordinates],
    image_size: tuple[int, int],
) -> dict:
    rois = list(rois)
    height, width = image_size
    if not rois:
        return {"n": 0}
    areas = torch.tensor(
        [roi.width * roi.height / float(height * width) for roi in rois]
    )
    widths = torch.tensor([roi.width for roi in rois], dtype=torch.float32)
    heights = torch.tensor([roi.height for roi in rois], dtype=torch.float32)
    return {
        "n": len(rois),
        "fallback_count": sum(int(roi.fallback) for roi in rois),
        "fallback_rate": sum(int(roi.fallback) for roi in rois) / len(rois),
        "area_fraction_mean": float(areas.mean()),
        "area_fraction_median": float(areas.median()),
        "width_mean": float(widths.mean()),
        "height_mean": float(heights.mean()),
    }
