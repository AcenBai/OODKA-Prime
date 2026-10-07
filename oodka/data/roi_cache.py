"""Persistent predicted-ROI cache for training blocks."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict

from .roi_geometry import ROICoordinates

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
