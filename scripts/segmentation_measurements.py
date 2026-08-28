"""Fiziksel hasta koordinatlarında segmentasyon ölçümleri.

Bu ölçümler model maskesini tarif eder; radyolog ölçümü veya tanı değildir.
NIfTI affine matrisinin RAS hasta koordinatlarında olduğu varsayılır.
"""

from __future__ import annotations

from itertools import product
from typing import Any

import numpy as np


def _world_points(indices: np.ndarray, affine: np.ndarray) -> np.ndarray:
    return indices @ affine[:3, :3].T + affine[:3, 3]


def _world_bbox(indices: np.ndarray, affine: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Voksel kenarlarını da içeren RAS eksenli fiziksel kutuyu döndürür."""
    lower = indices.min(axis=0).astype(float) - 0.5
    upper = indices.max(axis=0).astype(float) + 0.5
    corners = np.asarray(list(product(*zip(lower, upper))), dtype=float)
    world = _world_points(corners, affine)
    return world.min(axis=0), world.max(axis=0)


def _estimated_pancreas_location(
    pancreas_indices: np.ndarray,
    tumor_indices: np.ndarray,
    affine: np.ndarray,
) -> dict[str, Any] | None:
    """Tümörün RAS sağ-sol ekseninde pankreas içindeki kaba bölgesini tahmin eder."""
    if not pancreas_indices.size or not tumor_indices.size:
        return None

    pancreas_x = _world_points(pancreas_indices, affine)[:, 0]
    tumor_x = _world_points(tumor_indices, affine)[:, 0]
    gland_min, gland_max = float(pancreas_x.min()), float(pancreas_x.max())
    gland_span = gland_max - gland_min
    if gland_span <= 1e-6:
        return None

    # RAS +X hastanın sağıdır: pankreas başı sağda, kuyruk soldadır.
    tumor_min = (float(tumor_x.min()) - gland_min) / gland_span
    tumor_max = (float(tumor_x.max()) - gland_min) / gland_span
    thirds = (
        (0.0, 1.0 / 3.0, "kuyruk"),
        (1.0 / 3.0, 2.0 / 3.0, "gövde"),
        (2.0 / 3.0, 1.0, "baş"),
    )
    involved = []
    for start, stop, name in thirds:
        overlap = max(0.0, min(tumor_max, stop) - max(tumor_min, start))
        if overlap > 0.03:
            involved.append(name)
    involved = list(reversed(involved))  # klinik yazım: baş-gövde-kuyruk
    return {
        "estimated_region": "-".join(involved) if involved else "belirsiz",
        "method": "pankreasın RAS sağ-sol uzanımındaki göreli tümör aralığı",
        "tumor_range_fraction_left_to_right": [
            round(max(0.0, tumor_min), 3), round(min(1.0, tumor_max), 3)
        ],
        "warning": "Kaba anatomik tahmindir; radyolog lokalizasyonunun yerini tutmaz.",
    }


def measure_segmentation(mask: np.ndarray, affine: np.ndarray) -> dict[str, Any]:
    """Etiket 1=pankreas, 2=tümör maskesini RAS milimetrelerinde özetler."""
    data = np.squeeze(np.asarray(mask))
    affine = np.asarray(affine, dtype=float)
    if data.ndim != 3 or affine.shape != (4, 4):
        raise ValueError("Maske 3B ve affine 4x4 olmalıdır.")

    voxel_volume_ml = abs(float(np.linalg.det(affine[:3, :3]))) / 1000.0
    pancreas_indices = np.argwhere(data == 1)
    gland_indices = np.argwhere(data > 0)
    tumor_indices = np.argwhere(data == 2)
    result: dict[str, Any] = {
        "coordinate_system": "NIfTI RAS hasta koordinatları",
        "voxel_volume_ml": round(voxel_volume_ml, 7),
        "pancreas": None,
        "tumor": None,
        "estimated_location": None,
    }

    for name, indices in (("pancreas", pancreas_indices), ("tumor", tumor_indices)):
        if not indices.size:
            continue
        world = _world_points(indices, affine)
        lower, upper = _world_bbox(indices, affine)
        dimensions = upper - lower
        result[name] = {
            "voxels": int(indices.shape[0]),
            "volume_ml": round(indices.shape[0] * voxel_volume_ml, 3),
            "dimensions_rl_ap_si_mm": [round(float(v), 1) for v in dimensions],
            "centroid_ras_mm": [round(float(v), 1) for v in world.mean(axis=0)],
            "bbox_ras_mm": {
                "min": [round(float(v), 1) for v in lower],
                "max": [round(float(v), 1) for v in upper],
            },
        }

    result["estimated_location"] = _estimated_pancreas_location(
        gland_indices, tumor_indices, affine
    )
    return result
