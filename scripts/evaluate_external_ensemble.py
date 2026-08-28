"""Calibrate an external pancreatic-lesion probability model on nnU-Net validation data."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import nibabel as nib
import numpy as np
from nibabel.processing import resample_from_to
from scipy import ndimage


EPS = 1e-8


def largest_component(mask: np.ndarray) -> np.ndarray:
    labels, count = ndimage.label(mask)
    if count == 0:
        return mask
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    return labels == int(sizes.argmax())


def scores(pred: np.ndarray, target: np.ndarray) -> dict[str, float]:
    pred = pred.astype(bool, copy=False)
    target = target.astype(bool, copy=False)
    tp = int(np.count_nonzero(pred & target))
    fp = int(np.count_nonzero(pred & ~target))
    fn = int(np.count_nonzero(~pred & target))
    return {
        "dice": 2.0 * tp / (2 * tp + fp + fn + EPS),
        "precision": tp / (tp + fp + EPS),
        "recall": tp / (tp + fn + EPS),
        "pred_voxels": int(pred.sum()),
        "target_voxels": int(target.sum()),
        "overlap_voxels": tp,
        "detected": float(tp > 0),
    }


def candidate_masks(prob: np.ndarray, nnunet: np.ndarray) -> dict[str, np.ndarray]:
    result = {"nnunet": largest_component(nnunet)}
    for threshold in np.arange(0.25, 0.76, 0.05):
        key = f"medformer_lcc_t{threshold:.2f}"
        med = largest_component(prob >= threshold)
        result[key] = med
        result[f"union_t{threshold:.2f}"] = largest_component(nnunet | med)
        result[f"intersection_t{threshold:.2f}"] = largest_component(nnunet & med)
    for weight in (0.40, 0.50, 0.60, 0.70, 0.80):
        fused = weight * prob + (1.0 - weight) * nnunet.astype(np.float32)
        for threshold in np.arange(0.35, 0.66, 0.05):
            result[f"fusion_w{weight:.2f}_t{threshold:.2f}"] = largest_component(
                fused >= threshold
            )
    return result


def summarize(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["candidate"])].append(row)
    summary = []
    for candidate, items in grouped.items():
        entry: dict[str, object] = {"candidate": candidate, "cases": len(items)}
        for metric in ("dice", "precision", "recall", "detected"):
            values = np.asarray([float(x[metric]) for x in items])
            entry[f"mean_{metric}"] = float(values.mean())
            entry[f"median_{metric}"] = float(np.median(values))
        ratios = np.asarray(
            [float(x["pred_voxels"]) / max(float(x["target_voxels"]), 1.0) for x in items]
        )
        entry["median_volume_ratio"] = float(np.median(ratios))
        summary.append(entry)
    return sorted(summary, key=lambda x: float(x["mean_dice"]), reverse=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ground-truth-dir", required=True, type=Path)
    parser.add_argument("--nnunet-dir", required=True, type=Path)
    parser.add_argument("--external-root", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-csv", required=True, type=Path)
    args = parser.parse_args()

    rows: list[dict[str, object]] = []
    for pred_file in sorted(args.nnunet_dir.glob("case_*.nii.gz")):
        case_id = pred_file.name.removesuffix(".nii.gz")
        gt_file = args.ground_truth_dir / pred_file.name
        raw_file = args.external_root / case_id / "predictions_raw" / "pancreatic_lesion.nii.gz"
        if not raw_file.exists():
            print(f"Skipping {case_id}: external probability missing")
            continue
        gt_nii = nib.load(str(gt_file))
        nn_nii = nib.load(str(pred_file))
        raw_nii = nib.load(str(raw_file))
        target_src = np.asanyarray(gt_nii.dataobj) == 2
        nnunet_src = np.asanyarray(nn_nii.dataobj) == 2
        target = resample_from_to(
            nib.Nifti1Image(target_src.astype(np.uint8), gt_nii.affine), raw_nii, order=0
        ).get_fdata() > 0.5
        nnunet = resample_from_to(
            nib.Nifti1Image(nnunet_src.astype(np.uint8), nn_nii.affine), raw_nii, order=0
        ).get_fdata() > 0.5
        prob = np.clip(np.asanyarray(raw_nii.dataobj).astype(np.float32), 0.0, 1.0)

        # Connected-component sweeps are equivalent inside a crop containing every
        # nonzero candidate and avoid repeatedly labelling a mostly empty CT volume.
        support = target | nnunet | (prob >= 0.05)
        coords = np.argwhere(support)
        if coords.size:
            lo = np.maximum(coords.min(axis=0) - 3, 0)
            hi = np.minimum(coords.max(axis=0) + 4, support.shape)
            crop = tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))
            target, nnunet, prob = target[crop], nnunet[crop], prob[crop]
        for candidate, mask in candidate_masks(prob, nnunet).items():
            rows.append({"case_id": case_id, "candidate": candidate, **scores(mask, target)})
        print(f"Evaluated {case_id}")

    summary = summarize(rows)
    payload = {"evaluated_cases": len({str(x['case_id']) for x in rows}), "summary": summary}
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(summary[:12], indent=2))


if __name__ == "__main__":
    main()
