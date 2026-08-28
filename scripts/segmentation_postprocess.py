"""3B anatomik kapı ile pankreas/tümör maskesi doğrulama.

Bu modül model olasılığı uydurmaz. İki bağımsız segmentasyonun anatomik
uyumunu ölçer; pankreas güvenilir biçimde doğrulanamazsa tümör kararı vermek
yerine ``indeterminate`` döndürür.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
from scipy import ndimage


DEFAULTS = {
    "min_pancreas_ml": 3.0,
    "max_pancreas_ml": 250.0,
    "min_pancreas_slices": 3,
    "min_cross_model_dice": 0.20,
    "pancreas_fusion_radius_mm": 30.0,
    "tumor_support_radius_mm": 5.0,
    "min_tumor_component_ml": 0.30,
    "min_tumor_slices": 2,
    "min_tumor_support_fraction": 0.10,
    "min_tumor_cross_model_dice": 0.10,
    "min_tumor_cross_model_overlap_ml": 0.10,
    "tumor_model_match_radius_mm": 5.0,
    "min_tumor_cross_model_proximity_dice": 0.10,
    "unverified_tumor_support_radius_mm": 5.0,
    "min_unverified_tumor_support_fraction": 0.20,
    # A sizeable one-slice or single-model candidate must remain reviewable.
    # It is never promoted to a positive result without cross-model agreement.
    "min_unverified_tumor_component_ml": 0.40,
    "min_unverified_tumor_slices": 1,
    "require_tumor_model_consensus": True,
    "min_single_model_component_ml": 0.40,
    "min_single_model_slices": 2,
    "min_single_model_support_fraction": 0.25,
}


def _settings(overrides: Mapping[str, Any] | None) -> dict[str, Any]:
    values = dict(DEFAULTS)
    if overrides:
        values.update(overrides)
    return values


def _voxel_volume_ml(spacing: Sequence[float]) -> float:
    values = np.asarray(spacing[:3], dtype=float)
    if values.shape != (3,) or not np.all(np.isfinite(values)) or np.any(values <= 0):
        raise ValueError(f"Geçersiz voksel aralığı: {spacing}")
    return float(np.prod(values) / 1000.0)


def _slice_count(binary: np.ndarray) -> int:
    axes = tuple(range(binary.ndim - 1))
    return int(np.count_nonzero(np.any(binary, axis=axes)))


def _largest_component(binary: np.ndarray) -> tuple[np.ndarray, int, int]:
    """En büyük yüz-bağlantılı anatomik bileşeni bellek dostu alanda tutar.

    Köşe veya yalnız bir kenardan temas eden iki ayrı hacim, 26-bağlantıda tek
    bileşen sayılır; fakat marching-cubes bunları iki ayrı yüzey adası olarak
    çizer. Pankreas maskesinde bu durum sahte, parçalanmış 3B organ görüntüsü
    üretir. Bu nedenle organ kabulünde yalnız yüz paylaşan voksel komşuluğu
    (6-bağlantı) kullanılır.
    """
    binary = np.asarray(binary, dtype=bool)
    output = np.zeros(binary.shape, dtype=bool)
    points = np.where(binary)
    if not points[0].size:
        return output, 0, 0

    bounds = tuple(slice(int(axis.min()), int(axis.max()) + 1) for axis in points)
    crop = binary[bounds]
    labels, count = ndimage.label(
        crop,
        structure=ndimage.generate_binary_structure(3, 1),
    )
    if count == 1:
        output[bounds] = crop
        return output, 1, int(np.count_nonzero(crop))

    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    largest_id = int(np.argmax(sizes))
    output[bounds] = labels == largest_id
    return output, int(count), int(sizes[largest_id])


def _dilate_mm(binary: np.ndarray, radius_mm: float, spacing: Sequence[float]) -> np.ndarray:
    """Fiziksel Öklid uzaklığıyla genişlet; yalnızca organ çevresindeki kutuyu işler."""
    binary = np.asarray(binary, dtype=bool)
    output = np.zeros(binary.shape, dtype=bool)
    points = np.where(binary)
    if not points[0].size:
        return output
    if radius_mm <= 0:
        output[...] = binary
        return output

    spacing_arr = np.asarray(spacing[:3], dtype=float)
    padding = np.ceil(radius_mm / spacing_arr).astype(int)
    bounds = tuple(
        slice(
            max(0, int(axis.min()) - int(padding[index])),
            min(binary.shape[index], int(axis.max()) + int(padding[index]) + 1),
        )
        for index, axis in enumerate(points)
    )
    crop = binary[bounds]
    distances = ndimage.distance_transform_edt(~crop, sampling=spacing_arr)
    output[bounds] = distances <= radius_mm
    return output


def _dice(left: np.ndarray, right: np.ndarray) -> float:
    left_count = int(np.count_nonzero(left))
    right_count = int(np.count_nonzero(right))
    if left_count + right_count == 0:
        return 1.0
    overlap = int(np.count_nonzero(left & right))
    return float(2.0 * overlap / (left_count + right_count))


def _proximity_agreement(
    left: np.ndarray,
    right: np.ndarray,
    radius_mm: float,
    spacing: Sequence[float],
) -> dict[str, float | int]:
    """Measure symmetric physical agreement without requiring identical voxels."""
    left = np.asarray(left, dtype=bool)
    right = np.asarray(right, dtype=bool)
    left_count = int(np.count_nonzero(left))
    right_count = int(np.count_nonzero(right))
    if not left_count or not right_count:
        return {
            "dice": 0.0,
            "overlap_voxels": 0,
            "left_supported_voxels": 0,
            "right_supported_voxels": 0,
        }

    left_supported = int(np.count_nonzero(
        left & _dilate_mm(right, radius_mm, spacing)
    ))
    right_supported = int(np.count_nonzero(
        right & _dilate_mm(left, radius_mm, spacing)
    ))
    return {
        "dice": float(
            (left_supported + right_supported) / (left_count + right_count)
        ),
        "overlap_voxels": min(left_supported, right_supported),
        "left_supported_voxels": left_supported,
        "right_supported_voxels": right_supported,
    }


def _component_records(
    tumor: np.ndarray,
    support: np.ndarray,
    voxel_ml: float,
    cfg: Mapping[str, Any],
    spacing: Sequence[float],
    primary_tumor: np.ndarray | None = None,
    secondary_tumor: np.ndarray | None = None,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    accepted = np.zeros(tumor.shape, dtype=bool)
    records: list[dict[str, Any]] = []
    points = np.where(tumor)
    if not points[0].size:
        return accepted, records

    bounds = tuple(slice(int(axis.min()), int(axis.max()) + 1) for axis in points)
    crop = tumor[bounds]
    association_crop = crop
    primary_near_secondary = None
    secondary_near_primary = None
    if primary_tumor is not None and secondary_tumor is not None:
        match_radius = float(cfg["tumor_model_match_radius_mm"])
        # The bridge associates close model components but is never published.
        half_radius = max(0.0, match_radius / 2.0)
        primary_bridge = _dilate_mm(primary_tumor, half_radius, spacing)
        secondary_bridge = _dilate_mm(secondary_tumor, half_radius, spacing)
        association_crop = crop | (primary_bridge[bounds] & secondary_bridge[bounds])
        del primary_bridge, secondary_bridge
        primary_near_secondary = (
            primary_tumor & _dilate_mm(secondary_tumor, match_radius, spacing)
        )
        secondary_near_primary = (
            secondary_tumor & _dilate_mm(primary_tumor, match_radius, spacing)
        )
    labels, component_count = ndimage.label(
        association_crop,
        structure=ndimage.generate_binary_structure(3, 3),
    )
    support_crop = support[bounds]
    accepted_crop = np.zeros(crop.shape, dtype=bool)

    for component_id in range(1, int(component_count) + 1):
        component = (labels == component_id) & crop
        voxel_count = int(np.count_nonzero(component))
        if not voxel_count:
            continue
        supported = component & support_crop
        supported_voxels = int(np.count_nonzero(supported))
        supported_ml = supported_voxels * voxel_ml
        supported_slices = _slice_count(supported)
        support_fraction = supported_voxels / max(1, voxel_count)

        primary_voxels = None
        secondary_voxels = None
        cross_model_overlap_voxels = None
        cross_model_overlap_ml = None
        cross_model_proximity_voxels = None
        cross_model_proximity_ml = None
        cross_model_proximity_dice = None
        if primary_tumor is not None and secondary_tumor is not None:
            primary_voxels = int(np.count_nonzero(component & primary_tumor[bounds]))
            secondary_voxels = int(np.count_nonzero(component & secondary_tumor[bounds]))
            cross_model_overlap_voxels = int(np.count_nonzero(
                component & primary_tumor[bounds] & secondary_tumor[bounds]
            ))
            cross_model_overlap_ml = cross_model_overlap_voxels * voxel_ml
            primary_proximity_voxels = int(np.count_nonzero(
                component & primary_near_secondary[bounds]
            ))
            secondary_proximity_voxels = int(np.count_nonzero(
                component & secondary_near_primary[bounds]
            ))
            cross_model_proximity_voxels = min(
                primary_proximity_voxels, secondary_proximity_voxels
            )
            cross_model_proximity_ml = cross_model_proximity_voxels * voxel_ml
            cross_model_proximity_dice = (
                (primary_proximity_voxels + secondary_proximity_voxels)
                / max(1, primary_voxels + secondary_voxels)
            )

        reasons = []
        if supported_ml < float(cfg["min_tumor_component_ml"]):
            reasons.append("hacim_esigi")
        if supported_slices < int(cfg["min_tumor_slices"]):
            reasons.append("kesit_esigi")
        if support_fraction < float(cfg["min_tumor_support_fraction"]):
            reasons.append("pankreas_komsulugu")
        if primary_tumor is not None and secondary_tumor is not None:
            if primary_voxels == 0 or secondary_voxels == 0:
                single_model_solid = (
                    supported_ml >= float(cfg.get("min_single_model_component_ml", 0.40))
                    and supported_slices >= int(cfg.get("min_single_model_slices", 2))
                    and support_fraction >= float(cfg.get("min_single_model_support_fraction", 0.25))
                )
                if bool(cfg.get("require_tumor_model_consensus", True)) or not single_model_solid:
                    reasons.append("tek_model_adayi")
            else:
                exact_match = cross_model_overlap_ml >= float(
                    cfg["min_tumor_cross_model_overlap_ml"]
                )
                proximity_match = (
                    cross_model_proximity_ml >= float(
                        cfg["min_tumor_cross_model_overlap_ml"]
                    )
                    and cross_model_proximity_dice >= float(
                        cfg["min_tumor_cross_model_proximity_dice"]
                    )
                )
                both_models_co_detected = (
                    supported_ml >= float(cfg.get("min_single_model_component_ml", 0.40))
                    and supported_slices >= int(cfg.get("min_single_model_slices", 2))
                    and support_fraction >= float(cfg.get("min_single_model_support_fraction", 0.25))
                )
                if not exact_match and not proximity_match and not both_models_co_detected:
                    reasons.append("modeller_arasi_fiziksel_uyum")

        is_accepted = not reasons
        if is_accepted:
            # Bağımsız organ modelleri büyük tümörle deforme olmuş pankreası eksik
            # çizebilir. Bileşen anatomik banda tutunuyorsa, gerçek lezyonun dış
            # kısmını kesmemek için bileşenin tamamını koru.
            accepted_crop |= component

        records.append({
            "component": component_id,
            "raw_voxels": voxel_count,
            "raw_ml": round(voxel_count * voxel_ml, 4),
            "supported_voxels": supported_voxels,
            "supported_ml": round(supported_ml, 4),
            "supported_slices": supported_slices,
            "support_fraction": round(float(support_fraction), 4),
            "primary_model_voxels": primary_voxels,
            "secondary_model_voxels": secondary_voxels,
            "cross_model_overlap_voxels": cross_model_overlap_voxels,
            "cross_model_overlap_ml": (
                round(float(cross_model_overlap_ml), 4)
                if cross_model_overlap_ml is not None else None
            ),
            "cross_model_proximity_voxels": cross_model_proximity_voxels,
            "cross_model_proximity_ml": (
                round(float(cross_model_proximity_ml), 4)
                if cross_model_proximity_ml is not None else None
            ),
            "cross_model_proximity_dice": (
                round(float(cross_model_proximity_dice), 4)
                if cross_model_proximity_dice is not None else None
            ),
            "accepted": is_accepted,
            "rejection_reasons": reasons,
        })

    accepted[bounds] = accepted_crop
    return accepted, records


def extract_unverified_tumor_candidates(
    primary_mask: np.ndarray,
    pancreas_gate: np.ndarray,
    spacing: Sequence[float],
    config: Mapping[str, Any] | None = None,
    secondary_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Keep meaningful unresolved candidates for review or later arbitration.

    This does not make a positive decision. It only prevents a sizeable model
    candidate from being converted into a false green ``negative`` result.
    """
    cfg = _settings(config)
    primary = np.squeeze(np.asarray(primary_mask))
    secondary = (
        np.squeeze(np.asarray(secondary_mask)) if secondary_mask is not None else None
    )
    gate = np.squeeze(np.asarray(pancreas_gate)) > 0
    if primary.ndim != 3 or gate.ndim != 3 or (
        secondary is not None and secondary.ndim != 3
    ):
        raise ValueError("Aday maskeleri ve anatomik kapı üç boyutlu olmalıdır.")
    if primary.shape != gate.shape or (
        secondary is not None and secondary.shape != primary.shape
    ):
        raise ValueError("Aday maskeleri ile anatomik kapı boyutları uyuşmuyor.")

    tumor = primary == 2
    if secondary is not None:
        tumor |= secondary == 2
    output = np.zeros(tumor.shape, dtype=bool)
    points = np.where(tumor)
    if not points[0].size:
        return output, []

    voxel_ml = _voxel_volume_ml(spacing)
    support = _dilate_mm(
        gate, float(cfg["unverified_tumor_support_radius_mm"]), spacing
    )
    bounds = tuple(slice(int(axis.min()), int(axis.max()) + 1) for axis in points)
    crop = tumor[bounds]
    labels, count = ndimage.label(
        crop, structure=ndimage.generate_binary_structure(3, 3)
    )
    output_crop = np.zeros(crop.shape, dtype=bool)
    support_crop = support[bounds]
    records: list[dict[str, Any]] = []
    for component_id in range(1, int(count) + 1):
        component = labels == component_id
        voxels = int(np.count_nonzero(component))
        slices = _slice_count(component)
        supported_voxels = int(np.count_nonzero(component & support_crop))
        support_fraction = supported_voxels / max(1, voxels)
        meaningful = (
            voxels * voxel_ml >= float(cfg["min_unverified_tumor_component_ml"])
            and slices >= int(cfg["min_unverified_tumor_slices"])
            and support_fraction >= float(
                cfg["min_unverified_tumor_support_fraction"]
            )
        )
        if meaningful:
            output_crop |= component
        records.append({
            "component": component_id,
            "voxels": voxels,
            "ml": round(voxels * voxel_ml, 4),
            "slices": slices,
            "broad_support_fraction": round(float(support_fraction), 4),
            "meaningful_unverified_candidate": bool(meaningful),
        })
    output[bounds] = output_crop
    return output, records


def assess_pancreas_gate(
    pancreas_gate: np.ndarray,
    spacing: Sequence[float],
    config: Mapping[str, Any] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Bağımsız pankreas kapısını, tümör modellerini çalıştırmadan önce değerlendirir.

    En büyük yüz-bağlantılı bileşeni döndürmek hem küçük uzak gürültülerin ROI'yi
    büyütmesini engeller hem de asıl füzyonla aynı anatomik kabul ölçütlerini kullanır.
    """
    cfg = _settings(config)
    gate = np.squeeze(np.asarray(pancreas_gate)) > 0
    if gate.ndim != 3:
        raise ValueError("Anatomik kapı üç boyutlu olmalıdır.")

    voxel_ml = _voxel_volume_ml(spacing)
    gate_largest, gate_components, gate_voxels = _largest_component(gate)
    gate_ml = gate_voxels * voxel_ml
    gate_slices = _slice_count(gate_largest)
    gate_plausible = (
        gate_voxels > 0
        and float(cfg["min_pancreas_ml"]) <= gate_ml <= float(cfg["max_pancreas_ml"])
        and gate_slices >= int(cfg["min_pancreas_slices"])
    )
    assessment = {
        "voxel_volume_ml": round(voxel_ml, 7),
        "gate_voxels": gate_voxels,
        "gate_ml": round(gate_ml, 3),
        "gate_slices": gate_slices,
        "gate_components": gate_components,
        "gate_plausible": bool(gate_plausible),
        "reason": None if gate_plausible else (
            "3B anatomik model pankreası güvenilir hacim ve kesit aralığında doğrulayamadı."
        ),
    }
    return gate_largest, assessment


def validate_and_fuse_segmentation(
    raw_mask: np.ndarray,
    pancreas_gate: np.ndarray,
    spacing: Sequence[float],
    config: Mapping[str, Any] | None = None,
    secondary_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """nnU-Net maskesini bağımsız 3B pankreas maskesiyle doğrular ve birleştirir.

    Etiketler: 0 arka plan, 1 pankreas, 2 tümör. Kapı anatomik açıdan
    uygunsuzsa boş maske ve kararsız durum döner; ham tümör hiçbir zaman tek
    başına pozitif karar üretemez.
    """
    cfg = _settings(config)
    primary = np.squeeze(np.asarray(raw_mask))
    gate = np.squeeze(np.asarray(pancreas_gate)) > 0
    secondary = None
    if secondary_mask is not None:
        secondary = np.squeeze(np.asarray(secondary_mask))
    if primary.ndim != 3 or gate.ndim != 3 or (secondary is not None and secondary.ndim != 3):
        raise ValueError("Ham maske ve anatomik kapı üç boyutlu olmalıdır.")
    if primary.shape != gate.shape or (secondary is not None and secondary.shape != primary.shape):
        raise ValueError(
            f"Maske boyutları uyuşmuyor: {primary.shape} / {gate.shape} / "
            f"{None if secondary is None else secondary.shape}"
        )

    invalid_labels = sorted(set(np.unique(primary).tolist()) - {0, 1, 2})
    if secondary is not None:
        invalid_labels = sorted(
            set(invalid_labels) | (set(np.unique(secondary).tolist()) - {0, 1, 2})
        )
    if invalid_labels:
        raise ValueError(f"Beklenmeyen segmentasyon etiketleri: {invalid_labels[:10]}")

    raw = np.maximum(primary, secondary) if secondary is not None else primary

    voxel_ml = _voxel_volume_ml(spacing)
    raw_organ = raw > 0
    raw_tumor = raw == 2
    primary_tumor = primary == 2
    secondary_tumor = secondary == 2 if secondary is not None else None
    gate_largest, gate_assessment = assess_pancreas_gate(gate, spacing, cfg)

    qc: dict[str, Any] = {
        "status": "indeterminate",
        "pancreas_verified": False,
        "has_tumor": None,
        "voxel_volume_ml": gate_assessment["voxel_volume_ml"],
        "raw_pancreas_voxels": int(np.count_nonzero(raw == 1)),
        "raw_tumor_voxels": int(np.count_nonzero(raw_tumor)),
        "primary_tumor_voxels": int(np.count_nonzero(primary_tumor)),
        "secondary_tumor_voxels": (
            int(np.count_nonzero(secondary_tumor)) if secondary_tumor is not None else None
        ),
        "gate_voxels": gate_assessment["gate_voxels"],
        "gate_ml": gate_assessment["gate_ml"],
        "gate_slices": gate_assessment["gate_slices"],
        "gate_components": gate_assessment["gate_components"],
        "gate_plausible": gate_assessment["gate_plausible"],
        "tumor_components": [],
        "reason": gate_assessment["reason"],
    }

    if not gate_assessment["gate_plausible"]:
        qc["rejected_tumor_voxels"] = qc["raw_tumor_voxels"]
        return np.zeros(raw.shape, dtype=np.uint8), qc

    agreement_dice = _dice(raw_organ, gate_largest)
    qc["cross_model_dice"] = round(agreement_dice, 4)

    if secondary_tumor is not None:
        tumor_cross_model_dice = _dice(primary_tumor, secondary_tumor)
        tumor_cross_model_overlap_voxels = int(np.count_nonzero(primary_tumor & secondary_tumor))
        proximity = _proximity_agreement(
            primary_tumor,
            secondary_tumor,
            float(cfg["tumor_model_match_radius_mm"]),
            spacing,
        )
        qc["tumor_cross_model_dice"] = round(tumor_cross_model_dice, 4)
        qc["tumor_cross_model_overlap_voxels"] = tumor_cross_model_overlap_voxels
        qc["tumor_cross_model_overlap_ml"] = round(
            tumor_cross_model_overlap_voxels * voxel_ml, 4
        )
        qc["tumor_cross_model_proximity_dice"] = round(
            float(proximity["dice"]), 4
        )
        qc["tumor_cross_model_proximity_overlap_voxels"] = int(
            proximity["overlap_voxels"]
        )
        qc["tumor_cross_model_proximity_overlap_ml"] = round(
            int(proximity["overlap_voxels"]) * voxel_ml, 4
        )

    fusion_support = _dilate_mm(
        gate_largest, float(cfg["pancreas_fusion_radius_mm"]), spacing
    )
    fused_organ = gate_largest | (raw_organ & fusion_support)
    fused_organ, _, _ = _largest_component(fused_organ)

    final = np.zeros(raw.shape, dtype=np.uint8)
    final[fused_organ] = 1

    if not np.any(raw_organ) or agreement_dice < float(cfg["min_cross_model_dice"]):
        # The independent 3D gate has already passed its physical volume and
        # slice checks.  A poor nnU-Net organ overlap is insufficient evidence
        # to erase that real localisation or tell the user that no pancreas was
        # found.  Tumour decisions remain withheld in this disagreement state.
        qc.update({
            "status": "pancreas_localized",
            "pancreas_verified": True,
            "has_tumor": None,
            "reason": (
                "Pankreas bağımsız 3B anatomik modelle bulundu; diğer organ "
                "modeliyle uyum düşük olduğundan tümör kararı tutuldu."
            ),
            "pancreas_voxels": int(np.count_nonzero(fused_organ)),
            "pancreas_ml": round(int(np.count_nonzero(fused_organ)) * voxel_ml, 3),
            "tumor_voxels": 0,
            "tumor_ml": 0.0,
            "rejected_tumor_voxels": qc["raw_tumor_voxels"],
        })
        return final, qc

    tumor_support = _dilate_mm(
        gate_largest, float(cfg["tumor_support_radius_mm"]), spacing
    )
    accepted_tumor, records = _component_records(
        raw_tumor, tumor_support, voxel_ml, cfg, spacing,
        primary_tumor=primary_tumor if secondary_tumor is not None else None,
        secondary_tumor=secondary_tumor,
    )
    if secondary_tumor is not None and not np.any(accepted_tumor) and (
        qc["tumor_cross_model_dice"] < float(cfg["min_tumor_cross_model_dice"])
        and qc["tumor_cross_model_proximity_dice"] < float(
            cfg["min_tumor_cross_model_proximity_dice"]
        )
    ):
        accepted_tumor[...] = False
        for record in records:
            if record["accepted"]:
                record["accepted"] = False
                record["rejection_reasons"].append("dusuk_tumor_model_uyumu")
    final[accepted_tumor] = 2

    tumor_voxels = int(np.count_nonzero(accepted_tumor))
    pancreas_voxels = int(np.count_nonzero(final == 1))
    unverified_candidates, candidate_records = extract_unverified_tumor_candidates(
        primary,
        gate,
        spacing,
        cfg,
        secondary_mask=secondary,
    )
    unverified_candidates &= ~accepted_tumor
    unverified_voxels = int(np.count_nonzero(unverified_candidates))
    if tumor_voxels:
        status = "candidate"
        has_tumor: bool | None = True
        reason = None
    elif unverified_voxels:
        status = "indeterminate"
        has_tumor = None
        reason = (
            "Anlamlı tümör adayları üretildi; modeller kesin sınır üzerinde "
            "yeterli uzamsal uzlaşı göstermedi. Negatif karar verilemez."
        )
    else:
        status = "negative"
        has_tumor = False
        reason = None
    qc.update({
        "status": status,
        "pancreas_verified": True,
        "has_tumor": has_tumor,
        "pancreas_voxels": pancreas_voxels,
        "pancreas_ml": round(pancreas_voxels * voxel_ml, 3),
        "tumor_voxels": tumor_voxels,
        "tumor_ml": round(tumor_voxels * voxel_ml, 3),
        "unverified_tumor_voxels": unverified_voxels,
        "unverified_tumor_ml": round(unverified_voxels * voxel_ml, 3),
        "unverified_tumor_components": candidate_records,
        "rejected_tumor_voxels": int(np.count_nonzero(raw_tumor & ~accepted_tumor)),
        "tumor_components": records,
        "reason": reason,
    })
    return final, qc
