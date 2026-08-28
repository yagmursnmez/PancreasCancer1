"""Fuse validated PanTS probabilities with the existing pancreas/tumor mask."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np
from nibabel.processing import resample_from_to
from scipy import ndimage

from segmentation_measurements import measure_segmentation


def component_supported_by(mask: np.ndarray, seed: np.ndarray | None = None) -> np.ndarray:
    labels, count = ndimage.label(mask)
    if count == 0:
        return np.zeros(mask.shape, dtype=bool)
    if seed is not None and np.any(seed):
        overlaps = np.bincount(labels[seed].ravel(), minlength=count + 1)
        overlaps[0] = 0
        if overlaps.max() > 0:
            chosen = int(overlaps.argmax())
            return labels == chosen
        return np.zeros(mask.shape, dtype=bool)
    else:
        sizes = np.bincount(labels.ravel(), minlength=count + 1)
        sizes[0] = 0
        if sizes.max() > 0:
            chosen = int(sizes.argmax())
            return labels == chosen
        return np.zeros(mask.shape, dtype=bool)


def resampled(path: Path | None, target: nib.spatialimages.SpatialImage, order: int) -> np.ndarray | None:
    if path is None:
        return None
    image = nib.load(str(path))
    return resample_from_to(image, target, order=order).get_fdata(dtype=np.float32)


def dice(a: np.ndarray, b: np.ndarray) -> float:
    return 2.0 * int(np.count_nonzero(a & b)) / max(int(a.sum() + b.sum()), 1)


def _within_distance(seed: np.ndarray, radius_mm: float, spacing) -> np.ndarray:
    """Return a memory-bounded physical dilation around ``seed``."""
    seed = np.asarray(seed, dtype=bool)
    output = np.zeros(seed.shape, dtype=bool)
    points = np.where(seed)
    if not points[0].size:
        return output
    if radius_mm <= 0:
        output[...] = seed
        return output

    spacing_array = np.asarray(spacing, dtype=float)
    if spacing_array.shape != (3,) or np.any(spacing_array <= 0):
        raise ValueError(f"Invalid voxel spacing: {spacing}")
    padding = np.ceil(float(radius_mm) / spacing_array).astype(int)
    bounds = tuple(
        slice(
            max(0, int(axis.min()) - int(padding[index])),
            min(seed.shape[index], int(axis.max()) + int(padding[index]) + 1),
        )
        for index, axis in enumerate(points)
    )
    crop = seed[bounds]
    distance = ndimage.distance_transform_edt(~crop, sampling=spacing_array)
    output[bounds] = distance <= float(radius_mm)
    return output


def screening_consensus_candidate(
    med_prob: np.ndarray,
    rs_prob: np.ndarray,
    pancreas_seed: np.ndarray,
    spacing,
    *,
    medformer_threshold: float = 0.45,
    rsuper_threshold: float = 0.45,
    support_radius_mm: float = 10.0,
    match_radius_mm: float = 5.0,
) -> tuple[np.ndarray, dict]:
    """Extract a review-only lesion seed from two PanTS probability maps.

    The seed is confined to the independently localised pancreas and requires
    physical proximity between both tumour models.  It is deliberately not a
    diagnosis or a published tumour mask; ``refine_with_consensus`` decides
    whether its stricter calibrated rules can confirm it.
    """
    med = np.asarray(med_prob, dtype=np.float32)
    rs = np.asarray(rs_prob, dtype=np.float32)
    pancreas = np.asarray(pancreas_seed, dtype=bool)
    if med.ndim != 3 or med.shape != rs.shape or pancreas.shape != med.shape:
        raise ValueError("PanTS tarama olasılıkları ve pankreas tohumu aynı 3B boyutta olmalıdır.")

    output = np.zeros(med.shape, dtype=bool)
    if not np.any(pancreas):
        return output, {"proximity_dice": 0.0, "voxels": 0}

    support = _within_distance(pancreas, support_radius_mm, spacing)
    med_candidate = (med >= float(medformer_threshold)) & support
    rs_candidate = (rs >= float(rsuper_threshold)) & support
    med_near_rs = med_candidate & _within_distance(rs_candidate, match_radius_mm, spacing)
    rs_near_med = rs_candidate & _within_distance(med_candidate, match_radius_mm, spacing)
    # Keep the mutually close evidence, not either model's unpaired field.
    paired = med_near_rs | rs_near_med
    output = component_supported_by(paired)
    proximity_dice = float(
        (int(np.count_nonzero(med_near_rs)) + int(np.count_nonzero(rs_near_med)))
        / max(1, int(np.count_nonzero(med_candidate)) + int(np.count_nonzero(rs_candidate)))
    )
    return output, {
        "proximity_dice": proximity_dice,
        "voxels": int(np.count_nonzero(output)),
        "medformer_supported_voxels": int(np.count_nonzero(med_near_rs)),
        "rsuper_supported_voxels": int(np.count_nonzero(rs_near_med)),
    }


def refine_with_consensus(
    base: np.ndarray,
    med_prob: np.ndarray,
    rs_prob: np.ndarray,
    med_pancreas: np.ndarray,
    rs_pancreas: np.ndarray,
    spacing,
    *,
    medformer_threshold: float = 0.45,
    rsuper_threshold: float = 0.45,
    core_threshold: float = 0.60,
    envelope_threshold: float = 0.30,
    min_cross_model_dice: float = 0.50,
    max_expansion_mm: float = 5.0,
    candidate_seed: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Refine a verified seed or arbitrate an explicitly unverified candidate."""
    base = np.asarray(base, dtype=np.uint8)
    med_prob = np.asarray(med_prob, dtype=np.float32)
    rs_prob = np.asarray(rs_prob, dtype=np.float32)
    med_pancreas = np.asarray(med_pancreas, dtype=bool)
    rs_pancreas = np.asarray(rs_pancreas, dtype=bool)
    candidate_seed = (
        np.zeros(base.shape, dtype=bool)
        if candidate_seed is None else np.asarray(candidate_seed, dtype=bool)
    )
    arrays = (med_prob, rs_prob, med_pancreas, rs_pancreas, candidate_seed)
    if base.ndim != 3 or any(array.shape != base.shape for array in arrays):
        raise ValueError("Base mask and PanTS outputs must be matching 3D arrays.")

    base_tumor = base == 2
    primary = np.zeros(base.shape, dtype=bool)
    core = np.zeros(base.shape, dtype=bool)
    envelope = np.zeros(base.shape, dtype=bool)
    agreement = 0.0
    expansion_allowed = False
    rejection_reason = None
    candidate_arbitration = bool(np.any(candidate_seed) and not np.any(base_tumor))
    candidate_confirmed = False

    if np.any(base_tumor):
        # Only compare lesion components attached to the independently verified seed.
        med_primary = component_supported_by(
            med_prob >= float(medformer_threshold), base_tumor
        )
        rs_primary = component_supported_by(
            rs_prob >= float(rsuper_threshold), base_tumor
        )
        agreement = dice(med_primary, rs_primary)
        near_seed = _within_distance(base_tumor, max_expansion_mm, spacing)
        expansion_allowed = bool(
            np.any(med_primary)
            and np.any(rs_primary)
            and agreement >= float(min_cross_model_dice)
        )
        agreed_addition = med_primary & rs_primary & near_seed
        primary_raw = base_tumor | (agreed_addition if expansion_allowed else False)
        primary = component_supported_by(primary_raw, base_tumor)

        med_core = component_supported_by(med_prob >= float(core_threshold), base_tumor)
        rs_core = component_supported_by(rs_prob >= float(core_threshold), base_tumor)
        core = base_tumor | (med_core & rs_core & near_seed & primary)

        # A unilateral low-threshold extension is shown only as uncertainty.
        med_envelope = component_supported_by(
            med_prob >= float(envelope_threshold), primary
        )
        rs_envelope = component_supported_by(
            rs_prob >= float(envelope_threshold), primary
        )
        near_primary = _within_distance(primary, max_expansion_mm, spacing)
        envelope = component_supported_by(
            primary | ((med_envelope | rs_envelope) & near_primary), primary
        )
        if not expansion_allowed:
            rejection_reason = (
                "PanTS modelleri doğrulanmış aday çevresinde yeterli uzamsal "
                "uyum göstermedi; ana maske genişletilmedi."
            )
    elif np.any(candidate_seed):
        # Unverified candidates are never copied into the published mask. They
        # become a positive candidate only when both PanTS models independently
        # agree at the calibrated primary thresholds near the same seed.
        near_candidate = _within_distance(
            candidate_seed, max_expansion_mm, spacing
        )
        med_primary = component_supported_by(
            med_prob >= float(medformer_threshold), near_candidate
        )
        rs_primary = component_supported_by(
            rs_prob >= float(rsuper_threshold), near_candidate
        )
        agreement = dice(med_primary, rs_primary)
        agreed_primary = med_primary & rs_primary & near_candidate
        candidate_confirmed = bool(
            np.any(med_primary)
            and np.any(rs_primary)
            and np.any(agreed_primary)
            and agreement >= float(min_cross_model_dice)
        )
        expansion_allowed = candidate_confirmed
        if candidate_confirmed:
            primary = component_supported_by(agreed_primary, near_candidate)
            med_core = component_supported_by(
                med_prob >= float(core_threshold), near_candidate
            )
            rs_core = component_supported_by(
                rs_prob >= float(core_threshold), near_candidate
            )
            core = primary & med_core & rs_core
            med_envelope = component_supported_by(
                med_prob >= float(envelope_threshold), primary
            )
            rs_envelope = component_supported_by(
                rs_prob >= float(envelope_threshold), primary
            )
            near_primary = _within_distance(primary, max_expansion_mm, spacing)
            envelope = component_supported_by(
                primary | ((med_envelope | rs_envelope) & near_primary), primary
            )
            envelope |= candidate_seed
        else:
            med_envelope = component_supported_by(
                med_prob >= float(envelope_threshold), near_candidate
            )
            rs_envelope = component_supported_by(
                rs_prob >= float(envelope_threshold), near_candidate
            )
            envelope = candidate_seed | (
                (med_envelope | rs_envelope) & near_candidate
            )
            rejection_reason = (
                "PanTS hakem modelleri belirsiz aday üzerinde yeterli ortak "
                "sınır üretmedi; sonuç negatif değil, kararsız bırakıldı."
            )

    gland = component_supported_by(
        (base > 0) | med_pancreas | rs_pancreas | primary,
        base > 0 if np.any(base > 0) else None,
    )
    refined = np.zeros(base.shape, dtype=np.uint8)
    refined[gland & ~primary] = 1
    refined[primary] = 2
    uncertainty = np.zeros(base.shape, dtype=np.uint8)
    uncertainty[core] = 1
    uncertainty[primary & ~core] = 2
    uncertainty[envelope & ~primary] = 3

    metrics = {
        "cross_model_dice": float(agreement),
        "expansion_allowed": bool(expansion_allowed),
        "rejection_reason": rejection_reason,
        "base_tumor_voxels": int(base_tumor.sum()),
        "primary_tumor_voxels": int(primary.sum()),
        "added_tumor_voxels": int(np.count_nonzero(primary & ~base_tumor)),
        "core_voxels": int(core.sum()),
        "envelope_voxels": int(envelope.sum()),
        "candidate_arbitration": candidate_arbitration,
        "candidate_confirmed": bool(candidate_confirmed),
        "candidate_seed_voxels": int(candidate_seed.sum()),
    }
    return refined, uncertainty, metrics


def save_nifti(data: np.ndarray, reference: nib.spatialimages.SpatialImage, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    output = nib.Nifti1Image(data, reference.affine, reference.header.copy())
    output.set_data_dtype(np.uint8)
    output.set_qform(reference.affine, code=1)
    output.set_sform(reference.affine, code=1)
    nib.save(output, str(path))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-mask", required=True, type=Path)
    parser.add_argument("--medformer-probability", required=True, type=Path)
    parser.add_argument("--medformer-pancreas", required=True, type=Path)
    parser.add_argument("--rsuper-probability", type=Path)
    parser.add_argument("--rsuper-pancreas", type=Path)
    parser.add_argument("--medformer-threshold", type=float, default=0.40)
    parser.add_argument("--rsuper-threshold", type=float, default=0.30)
    parser.add_argument("--core-medformer-threshold", type=float, default=0.50)
    parser.add_argument("--core-rsuper-threshold", type=float, default=0.40)
    parser.add_argument("--envelope-medformer-threshold", type=float, default=0.25)
    parser.add_argument("--envelope-rsuper-threshold", type=float, default=0.20)
    parser.add_argument("--output-mask", required=True, type=Path)
    parser.add_argument("--output-uncertainty", required=True, type=Path)
    parser.add_argument("--output-audit", required=True, type=Path)
    args = parser.parse_args()

    base_nii = nib.load(str(args.base_mask))
    base = np.asanyarray(base_nii.dataobj).astype(np.uint8)
    base_tumor = base == 2
    med_prob = resampled(args.medformer_probability, base_nii, order=1)
    med_pan = resampled(args.medformer_pancreas, base_nii, order=0) > 0.5
    rs_prob = resampled(args.rsuper_probability, base_nii, order=1)
    rs_pan_data = resampled(args.rsuper_pancreas, base_nii, order=0)
    rs_pan = rs_pan_data > 0.5 if rs_pan_data is not None else np.zeros_like(base_tumor)

    primary_raw = base_tumor | (med_prob >= args.medformer_threshold)
    core_raw = base_tumor | (med_prob >= args.core_medformer_threshold)
    envelope_raw = base_tumor | (med_prob >= args.envelope_medformer_threshold)
    if rs_prob is not None:
        primary_raw |= rs_prob >= args.rsuper_threshold
        core_raw |= rs_prob >= args.core_rsuper_threshold
        envelope_raw |= rs_prob >= args.envelope_rsuper_threshold

    primary = component_supported_by(primary_raw, base_tumor)
    core = component_supported_by(core_raw, base_tumor) & primary
    envelope = component_supported_by(envelope_raw | primary, primary) | primary
    gland = component_supported_by((base > 0) | med_pan | rs_pan | primary, primary)

    final = np.zeros(base.shape, dtype=np.uint8)
    final[gland & ~primary] = 1
    final[primary] = 2
    uncertainty = np.zeros(base.shape, dtype=np.uint8)
    uncertainty[core] = 1
    uncertainty[primary & ~core] = 2
    uncertainty[envelope & ~primary] = 3
    save_nifti(final, base_nii, args.output_mask)
    save_nifti(uncertainty, base_nii, args.output_uncertainty)

    voxel_ml = abs(float(np.linalg.det(base_nii.affine[:3, :3]))) / 1000.0
    audit = {
        "thresholds": {
            "primary": {"medformer": args.medformer_threshold, "rsuper": args.rsuper_threshold if rs_prob is not None else None},
            "core": {"medformer": args.core_medformer_threshold, "rsuper": args.core_rsuper_threshold if rs_prob is not None else None},
            "envelope": {"medformer": args.envelope_medformer_threshold, "rsuper": args.envelope_rsuper_threshold if rs_prob is not None else None},
        },
        "cross_model_dice_at_primary_thresholds": (
            dice(med_prob >= args.medformer_threshold, rs_prob >= args.rsuper_threshold)
            if rs_prob is not None else None
        ),
        "volumes_ml": {
            "core": round(float(core.sum()) * voxel_ml, 3),
            "primary": round(float(primary.sum()) * voxel_ml, 3),
            "envelope": round(float(envelope.sum()) * voxel_ml, 3),
            "pancreas_and_tumor": round(float(gland.sum()) * voxel_ml, 3),
        },
        "measurements": measure_segmentation(final, base_nii.affine),
        "output_mask": str(args.output_mask),
        "output_uncertainty": str(args.output_uncertainty),
    }
    args.output_audit.parent.mkdir(parents=True, exist_ok=True)
    args.output_audit.write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(audit, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
