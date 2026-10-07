"""ROI localization, oracle boxes, jitter, visibility, and output policy."""

from __future__ import annotations

from typing import Sequence

import torch

from .roi_geometry import ROICoordinates

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


def oracle_block_rois(
    labels: torch.Tensor,
    source_labels: Sequence[int],
    generator: ROIGenerator,
    valid_z: torch.Tensor | None = None,
) -> list[ROICoordinates]:
    """Create one GT-union XY ROI for every contiguous Z block."""
    if labels.ndim != 4:
        raise ValueError("labels must be [B,Z,H,W]")
    targets = torch.zeros_like(labels, dtype=torch.bool)
    for source_id in source_labels:
        targets |= labels == int(source_id)
    if valid_z is not None:
        if valid_z.shape != labels.shape[:2]:
            raise ValueError("valid_z must be [B,Z]")
        targets &= valid_z.to(targets.device)[:, :, None, None].bool()
    return [
        generator.from_probability(block_target.any(dim=0).float())
        for block_target in targets
    ]


def jitter_roi(
    roi: ROICoordinates,
    image_size: tuple[int, int],
    *,
    center_fraction: float,
    scale_min: float,
    scale_max: float,
) -> ROICoordinates:
    """Randomly perturb a cached ROI without changing its coordinate space."""
    if center_fraction <= 0.0 and scale_min == 1.0 and scale_max == 1.0:
        return roi
    if not 0.0 < scale_min <= scale_max:
        raise ValueError("ROI jitter scales must satisfy 0 < min <= max")
    height, width = image_size
    scale = float(torch.empty(()).uniform_(scale_min, scale_max))
    dx = float(torch.empty(()).uniform_(-center_fraction, center_fraction))
    dy = float(torch.empty(()).uniform_(-center_fraction, center_fraction))
    center_x = 0.5 * (roi.x0 + roi.x1) + dx * roi.width
    center_y = 0.5 * (roi.y0 + roi.y1) + dy * roi.height
    out_w = max(2, int(round(roi.width * scale)))
    out_h = max(2, int(round(roi.height * scale)))
    x0 = int(round(center_x - out_w / 2.0))
    y0 = int(round(center_y - out_h / 2.0))
    x0 = min(max(0, x0), max(0, width - out_w))
    y0 = min(max(0, y0), max(0, height - out_h))
    return ROICoordinates(
        x0=x0,
        y0=y0,
        x1=min(width, x0 + out_w),
        y1=min(height, y0 + out_h),
        fallback=roi.fallback,
    )


def roi_prompt_visibility(
    original_gt: torch.Tensor,
    rois: Sequence[ROICoordinates],
    groups: Sequence[Sequence[int]],
    *,
    min_coverage: float,
) -> torch.Tensor:
    """Return [B,P] validity, skipping classes truncated by ROI crops.

    A class absent on the full slice is a genuine negative. A class present on
    the full block is supervised only when its shared block ROI retains enough
    of it. The same XY crop is used for every Z slice in a block.
    """
    if original_gt.ndim != 4:
        raise ValueError("original_gt must be [B,Z,H,W]")
    batch_size, block_z = original_gt.shape[:2]
    if len(rois) != batch_size:
        raise ValueError("ROI count must equal B")
    visible = torch.ones((batch_size, len(groups)), dtype=torch.bool)
    for batch_index in range(batch_size):
        for prompt_index, source_ids in enumerate(groups):
            mask = torch.zeros_like(original_gt[batch_index], dtype=torch.bool)
            for source_id in source_ids:
                mask |= original_gt[batch_index] == int(source_id)
            total = int(mask.sum())
            if total == 0:
                continue
            roi = rois[batch_index]
            inside = int(
                mask[:, roi.y0 : roi.y1, roi.x0 : roi.x1].sum()
            )
            visible[batch_index, prompt_index] = inside / total >= min_coverage
    return visible


def hard_switch_foreground_logits(
    anatomy_logits: torch.Tensor,
    restored_roi_logits: torch.Tensor,
    rois: Sequence[ROICoordinates],
    outside_prompt_mapping: Sequence[tuple[int, int]] = ((0, 0), (1, 1)),
) -> torch.Tensor:
    """Use mapped Pass-1 classes outside each ROI and Pass 2 inside.

    ``outside_prompt_mapping`` contains ``(anatomy_index, output_index)``
    pairs. Auxiliary localization prompts can therefore produce the ROI
    without ever becoming deployable output classes.
    """
    if anatomy_logits.ndim != 5 or anatomy_logits.shape[1] < 2:
        raise ValueError("anatomy_logits must be [B,>=2,Z,H,W]")
    if restored_roi_logits.ndim != 5 or restored_roi_logits.shape[1] < 4:
        raise ValueError("V2/V3 ROI logits must be [B,>=4,Z,H,W]")
    batch_size, prompt_count, block_z, height, width = restored_roi_logits.shape
    if len(rois) != batch_size:
        raise ValueError("ROI count must equal B")
    outside = restored_roi_logits.new_full(
        (batch_size, prompt_count, block_z, height, width), -20.0
    )
    for anatomy_index, output_index in outside_prompt_mapping:
        anatomy_index = int(anatomy_index)
        output_index = int(output_index)
        if not 0 <= anatomy_index < anatomy_logits.shape[1]:
            raise ValueError(
                f"Invalid anatomy prompt index {anatomy_index} for "
                f"{anatomy_logits.shape[1]} prompts"
            )
        if not 0 <= output_index < prompt_count:
            raise ValueError(
                f"Invalid output prompt index {output_index} for "
                f"{prompt_count} prompts"
            )
        outside[:, output_index] = anatomy_logits[:, anatomy_index]
    mask = torch.zeros(
        (batch_size, 1, block_z, height, width),
        dtype=torch.bool,
        device=restored_roi_logits.device,
    )
    for batch_index, roi in enumerate(rois):
        mask[
            batch_index, :, :, roi.y0 : roi.y1, roi.x0 : roi.x1
        ] = True
    return torch.where(mask, restored_roi_logits, outside)



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
