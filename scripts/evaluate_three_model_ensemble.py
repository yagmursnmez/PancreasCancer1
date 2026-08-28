"""Grid-search nnU-Net + MedFormer PanTS + R-Super PanTS lesion unions."""

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


def metrics(pred: np.ndarray, target: np.ndarray) -> dict[str, float]:
    tp = int(np.count_nonzero(pred & target))
    fp = int(np.count_nonzero(pred & ~target))
    fn = int(np.count_nonzero(~pred & target))
    return {
        "dice": 2 * tp / max(2 * tp + fp + fn, 1),
        "precision": tp / max(tp + fp, 1),
        "recall": tp / max(tp + fn, 1),
        "volume_ratio": int(pred.sum()) / max(int(target.sum()), 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ground-truth-dir", required=True, type=Path)
    parser.add_argument("--nnunet-dir", required=True, type=Path)
    parser.add_argument("--medformer-root", required=True, type=Path)
    parser.add_argument("--rsuper-root", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    args = parser.parse_args()

    rows = []
    for nn_file in sorted(args.nnunet_dir.glob("case_*.nii.gz")):
        case_id = nn_file.name.removesuffix(".nii.gz")
        med_file = args.medformer_root / case_id / "predictions_raw" / "pancreatic_lesion.nii.gz"
        rs_file = args.rsuper_root / case_id / "predictions_raw" / "pancreatic_lesion.nii.gz"
        if not med_file.exists() or not rs_file.exists():
            continue
        med_nii = nib.load(str(med_file))
        rs_nii = nib.load(str(rs_file))
        gt_nii = nib.load(str(args.ground_truth_dir / nn_file.name))
        nn_nii = nib.load(str(nn_file))
        gt_src = np.asanyarray(gt_nii.dataobj) == 2
        nn_src = np.asanyarray(nn_nii.dataobj) == 2
        target = resample_from_to(
            nib.Nifti1Image(gt_src.astype(np.uint8), gt_nii.affine), med_nii, order=0
        ).get_fdata() > 0.5
        nn = resample_from_to(
            nib.Nifti1Image(nn_src.astype(np.uint8), nn_nii.affine), med_nii, order=0
        ).get_fdata() > 0.5
        med = np.clip(np.asanyarray(med_nii.dataobj).astype(np.float32), 0, 1)
        rs = np.clip(resample_from_to(rs_nii, med_nii, order=1).get_fdata(dtype=np.float32), 0, 1)
        support = target | nn | (med >= 0.05) | (rs >= 0.05)
        coords = np.argwhere(support)
        if coords.size:
            lo = np.maximum(coords.min(0) - 3, 0)
            hi = np.minimum(coords.max(0) + 4, support.shape)
            crop = tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))
            target, nn, med, rs = target[crop], nn[crop], med[crop], rs[crop]

        for med_t in np.arange(0.30, 0.56, 0.05):
            for rs_t in np.arange(0.20, 0.66, 0.05):
                pred = largest(nn | (med >= med_t) | (rs >= rs_t))
                rows.append({
                    "case_id": case_id,
                    "candidate": f"union_med{med_t:.2f}_rs{rs_t:.2f}",
                    **metrics(pred, target),
                })

    grouped = defaultdict(list)
    for row in rows:
        grouped[row["candidate"]].append(row)
    summary = []
    for candidate, items in grouped.items():
        entry = {"candidate": candidate, "cases": len(items)}
        for key in ("dice", "precision", "recall", "volume_ratio"):
            values = [item[key] for item in items]
            entry[f"mean_{key}"] = float(np.mean(values))
            entry[f"median_{key}"] = float(np.median(values))
        summary.append(entry)
    summary.sort(key=lambda x: x["mean_dice"], reverse=True)
    payload = {"evaluated_cases": len({x["case_id"] for x in rows}), "summary": summary}
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(summary[:15], indent=2))


if __name__ == "__main__":
    main()
