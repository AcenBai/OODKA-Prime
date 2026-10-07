"""Modality-neutral ROI canvas geometry, transforms, and diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

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
