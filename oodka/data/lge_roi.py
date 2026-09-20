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


ROI_TRANSFORMS = ("resize", "pad", "letterbox")


@dataclass(frozen=True)
class ROIPlacement:
    """Location of transformed ROI content inside a fixed model canvas."""

    top: int
    left: int
    height: int
    width: int
    canvas_height: int
    canvas_width: int


def roi_placement(
    roi: ROICoordinates,
    canvas_size: tuple[int, int],
    transform: str,
) -> ROIPlacement:
    """Return the reversible crop-to-canvas geometry for one ROI."""
    if transform not in ROI_TRANSFORMS:
        raise ValueError(
            f"roi transform must be one of {ROI_TRANSFORMS}, got {transform!r}"
        )
    canvas_height, canvas_width = (int(value) for value in canvas_size)
    if min(canvas_height, canvas_width, roi.height, roi.width) <= 0:
        raise ValueError(f"Invalid ROI/canvas geometry: {roi}, {canvas_size}")
    if transform == "resize":
        content_height, content_width = canvas_height, canvas_width
    elif transform == "pad":
        if roi.height > canvas_height or roi.width > canvas_width:
            raise ValueError(
                "pad transform cannot place an ROI larger than its canvas: "
                f"roi={roi.height}x{roi.width}, canvas={canvas_size}"
            )
        content_height, content_width = roi.height, roi.width
    else:
        scale = min(
            canvas_height / float(roi.height),
            canvas_width / float(roi.width),
        )
        content_height = min(
            canvas_height, max(1, int(round(roi.height * scale)))
        )
        content_width = min(
            canvas_width, max(1, int(round(roi.width * scale)))
        )
    return ROIPlacement(
        top=(canvas_height - content_height) // 2,
        left=(canvas_width - content_width) // 2,
        height=content_height,
        width=content_width,
        canvas_height=canvas_height,
        canvas_width=canvas_width,
    )


def _resize_spatial(
    value: torch.Tensor,
    size: tuple[int, int],
    *,
    mode: str,
) -> torch.Tensor:
    if value.shape[-2:] == size:
        return value
    kwargs = {"size": size, "mode": mode}
    if mode in {"linear", "bilinear", "bicubic", "trilinear"}:
        kwargs["align_corners"] = False
    return F.interpolate(value, **kwargs)


def _validate_roi_bounds(
    roi: ROICoordinates,
    image_size: tuple[int, int],
) -> None:
    height, width = (int(value) for value in image_size)
    if not (
        0 <= roi.x0 < roi.x1 <= width
        and 0 <= roi.y0 < roi.y1 <= height
    ):
        raise ValueError(
            f"ROI {roi} lies outside image canvas {height}x{width}"
        )


def transform_roi_tensor(
    tensor: torch.Tensor,
    rois: Sequence[ROICoordinates],
    *,
    transform: str,
    mode: str,
    padding_value: float = 0.0,
) -> torch.Tensor:
    """Crop ``[B,N,C,H,W]`` and map each ROI to the original-size canvas."""
    if tensor.ndim != 5:
        raise ValueError("ROI tensor must be [B,N,C,H,W]")
    if len(rois) != tensor.shape[0]:
        raise ValueError("ROI count must equal B")
    canvas_size = (int(tensor.shape[-2]), int(tensor.shape[-1]))
    transformed = []
    for batch_index, roi in enumerate(rois):
        _validate_roi_bounds(roi, canvas_size)
        placement = roi_placement(roi, canvas_size, transform)
        crop = tensor[
            batch_index, :, :, roi.y0 : roi.y1, roi.x0 : roi.x1
        ]
        crop = _resize_spatial(
            crop,
            (placement.height, placement.width),
            mode=mode,
        )
        pad_right = placement.canvas_width - placement.left - placement.width
        pad_bottom = placement.canvas_height - placement.top - placement.height
        transformed.append(
            F.pad(
                crop,
                (placement.left, pad_right, placement.top, pad_bottom),
                value=float(padding_value),
            )
        )
    return torch.stack(transformed, dim=0)


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


def augment_lge_batch(
    batch_data: dict,
    *,
    rotation_degrees: float,
    scale_min: float,
    scale_max: float,
    translation_fraction: float,
    horizontal_flip_probability: float,
    vertical_flip_probability: float,
    intensity_probability: float,
) -> dict:
    """Apply synchronized affine and mild modality-aware intensity jitter."""
    output = dict(batch_data)
    nn_image = batch_data["nnunet_image"]
    bp_image = batch_data["biomedparse_image"]
    gt = batch_data["gt"]
    if nn_image.ndim != 5 or bp_image.ndim != 5 or gt.ndim != 4:
        raise ValueError("Unexpected batch tensor rank for augmentation")
    batch_size, block_z = nn_image.shape[:2]
    angles = torch.empty(batch_size).uniform_(-rotation_degrees, rotation_degrees)
    angles = angles * torch.pi / 180.0
    scales = torch.empty(batch_size).uniform_(scale_min, scale_max)
    flip_x = torch.where(
        torch.rand(batch_size) < horizontal_flip_probability, -1.0, 1.0
    )
    flip_y = torch.where(
        torch.rand(batch_size) < vertical_flip_probability, -1.0, 1.0
    )
    tx = torch.empty(batch_size).uniform_(-translation_fraction, translation_fraction) * 2.0
    ty = torch.empty(batch_size).uniform_(-translation_fraction, translation_fraction) * 2.0
    theta = torch.zeros((batch_size, 2, 3), dtype=nn_image.dtype)
    theta[:, 0, 0] = torch.cos(angles) * flip_x / scales
    theta[:, 0, 1] = -torch.sin(angles) * flip_y / scales
    theta[:, 1, 0] = torch.sin(angles) * flip_x / scales
    theta[:, 1, 1] = torch.cos(angles) * flip_y / scales
    theta[:, 0, 2] = tx
    theta[:, 1, 2] = ty

    # Preserve 3-D block coherence by sharing one spatial transform across all
    # Z slices of a block. Each slice's pseudo-RGB channels remain synchronized.
    theta_bz = theta.repeat_interleave(block_z, dim=0)

    def spatial(images: torch.Tensor, mode: str) -> torch.Tensor:
        flat = images.flatten(0, 1)
        grid = F.affine_grid(theta_bz, flat.shape, align_corners=False)
        transformed = F.grid_sample(
            flat, grid, mode=mode, padding_mode="zeros", align_corners=False
        )
        return transformed.unflatten(0, (batch_size, block_z))

    nn_aug = spatial(nn_image, "bilinear")
    bp_aug = spatial(bp_image, "bilinear")
    gt_aug = spatial(gt[:, :, None].float(), "nearest")[:, :, 0].to(gt.dtype)

    for index in range(batch_size * block_z):
        if float(torch.rand(())) >= intensity_probability:
            continue
        contrast = float(torch.empty(()).uniform_(0.8, 1.2))
        shift = float(torch.empty(()).uniform_(-0.1, 0.1))
        noise = float(torch.empty(()).uniform_(0.0, 0.04))
        batch_index, z_index = divmod(index, block_z)
        for images, clamp in ((nn_aug, False), (bp_aug, True)):
            value = images[batch_index, z_index]
            mean = value.mean()
            std = value.std().clamp_min(1e-6)
            value = mean + contrast * (value - mean) + shift * std
            value = value + torch.randn_like(value) * (noise * std)
            if clamp:
                gamma = float(torch.empty(()).uniform_(0.75, 1.35))
                value = (value.clamp(0.0, 255.0) / 255.0).pow(gamma) * 255.0
            images[batch_index, z_index] = value
    output["nnunet_image"] = nn_aug
    output["biomedparse_image"] = bp_aug
    output["gt"] = gt_aug
    return output


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
    *,
    transform: str = "resize",
) -> dict:
    """Crop one coherent cuboid and map it to the fixed model canvas."""
    nn_image = batch_data["nnunet_image"]
    bp_image = batch_data["biomedparse_image"]
    gt = batch_data["gt"]
    if nn_image.ndim != 5 or bp_image.ndim != 5 or gt.ndim != 4:
        raise ValueError("Unexpected batch tensor rank for ROI crop")
    batch_size, block_z = nn_image.shape[:2]
    if len(rois) != batch_size:
        raise ValueError("ROI count must equal B")
    output = dict(batch_data)
    output["nnunet_image"] = transform_roi_tensor(
        nn_image, rois, transform=transform, mode="bilinear"
    )
    output["biomedparse_image"] = transform_roi_tensor(
        bp_image, rois, transform=transform, mode="bilinear"
    )
    output["gt"] = transform_roi_tensor(
        gt[:, :, None].float(),
        rois,
        transform=transform,
        mode="nearest",
        # Padding is artificial canvas, not anatomical background.  The
        # segmentation loss already treats -1 as spatially invalid.
        padding_value=-1.0,
    )[:, :, 0].to(gt.dtype)
    return output


def restore_roi_logits(
    roi_logits: torch.Tensor,
    rois: Sequence[ROICoordinates],
    output_size: tuple[int, int],
    outside_logit: float = -20.0,
    *,
    transform: str = "resize",
) -> torch.Tensor:
    """Restore ``[B,P,Z,h,w]`` ROI logits to full-image coordinates."""
    if roi_logits.ndim != 5:
        raise ValueError("ROI logits must be [B,P,Z,H,W]")
    batch_size, prompts, block_z = roi_logits.shape[:3]
    if len(rois) != batch_size:
        raise ValueError("ROI count must equal B")
    height, width = output_size
    if roi_logits.shape[-2:] != (height, width):
        raise ValueError(
            "ROI-logit canvas must match output_size for reversible "
            f"placement, got {tuple(roi_logits.shape[-2:])} and "
            f"{(height, width)}"
        )
    output = roi_logits.new_full(
        (batch_size, prompts, block_z, height, width), float(outside_logit)
    )
    for batch_index, roi in enumerate(rois):
        _validate_roi_bounds(roi, (height, width))
        placement = roi_placement(roi, (height, width), transform)
        content = roi_logits[
            batch_index,
            :,
            :,
            placement.top : placement.top + placement.height,
            placement.left : placement.left + placement.width,
        ]
        resized = _resize_spatial(
            content.permute(1, 0, 2, 3),
            (roi.height, roi.width),
            mode="bilinear",
        ).permute(1, 0, 2, 3)
        output[
            batch_index, :, :, roi.y0 : roi.y1, roi.x0 : roi.x1
        ] = resized
    return output


def roi_diagnostics(
    rois: Iterable[ROICoordinates],
    image_size: tuple[int, int],
    *,
    transform: str = "resize",
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
    placements = [roi_placement(roi, image_size, transform) for roi in rois]
    scale_x = torch.tensor(
        [placement.width / roi.width for roi, placement in zip(rois, placements)]
    )
    scale_y = torch.tensor(
        [placement.height / roi.height for roi, placement in zip(rois, placements)]
    )
    anisotropy = torch.maximum(scale_x, scale_y) / torch.minimum(
        scale_x, scale_y
    ).clamp_min(1e-12)
    return {
        "n": len(rois),
        "fallback_count": sum(int(roi.fallback) for roi in rois),
        "fallback_rate": sum(int(roi.fallback) for roi in rois) / len(rois),
        "area_fraction_mean": float(areas.mean()),
        "area_fraction_median": float(areas.median()),
        "width_mean": float(widths.mean()),
        "height_mean": float(heights.mean()),
        "transform": transform,
        "scale_x_mean": float(scale_x.mean()),
        "scale_y_mean": float(scale_y.mean()),
        "pixel_area_scale_mean": float((scale_x * scale_y).mean()),
        "anisotropy_mean": float(anisotropy.mean()),
        "anisotropy_max": float(anisotropy.max()),
    }
