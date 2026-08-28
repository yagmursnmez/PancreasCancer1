"""Evaluate PanTS pancreas masks against the MSD fold-0 expert labels."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import nibabel as nib
import numpy as np
from nibabel.processing import resample_from_to
from scipy import ndimage


def largest(mask: np.ndarray) -> np.ndarray:
    labels, count = ndimage.label(mask)
    if count == 0:
        return mask
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    return labels == sizes.argmax()


def dice(pred: np.ndarray, target: np.ndarray) -> float:
    total = int(pred.sum() + target.sum())
    return 2.0 * int(np.count_nonzero(pred & target)) / max(total, 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ground-truth-dir", required=True, type=Path)
    parser.add_argument("--nnunet-dir", required=True, type=Path)
    parser.add_argument("--external-root", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    args = parser.parse_args()

    rows = []
    for nn_file in sorted(args.nnunet_dir.glob("case_*.nii.gz")):
        case_id = nn_file.name.removesuffix(".nii.gz")
        case_root = args.external_root / case_id
        pancreas_file = case_root / "predictions" / "pancreas.nii.gz"
        lesion_file = case_root / "predictions_raw" / "pancreatic_lesion.nii.gz"
        if not pancreas_file.exists() or not lesion_file.exists():
            continue
        pan_nii = nib.load(str(pancreas_file))
        lesion_nii = nib.load(str(lesion_file))
        gt_nii = nib.load(str(args.ground_truth_dir / nn_file.name))
        nn_nii = nib.load(str(nn_file))
        gt = resample_from_to(
            nib.Nifti1Image((np.asanyarray(gt_nii.dataobj) > 0).astype(np.uint8), gt_nii.affine),
            pan_nii,
            order=0,
        ).get_fdata() > 0.5
        nn = resample_from_to(
            nib.Nifti1Image((np.asanyarray(nn_nii.dataobj) > 0).astype(np.uint8), nn_nii.affine),
            pan_nii,
            order=0,
        ).get_fdata() > 0.5
        pan = np.asanyarray(pan_nii.dataobj) > 0
        lesion = resample_from_to(lesion_nii, pan_nii, order=1).get_fdata() >= 0.4
        candidates = {
            "nnunet": largest(nn),
            "pants_pancreas": largest(pan),
            "pants_pancreas_plus_lesion": largest(pan | lesion),
            "union": largest(nn | pan | lesion),
        }
        for name, mask in candidates.items():
            rows.append({"case_id": case_id, "candidate": name, "dice": dice(mask, gt)})

    grouped = defaultdict(list)
    for row in rows:
        grouped[row["candidate"]].append(row["dice"])
    summary = [
        {
            "candidate": name,
            "cases": len(values),
            "mean_dice": float(np.mean(values)),
            "median_dice": float(np.median(values)),
        }
        for name, values in grouped.items()
    ]
    summary.sort(key=lambda x: x["mean_dice"], reverse=True)
    payload = {"summary": summary, "cases": rows}
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
