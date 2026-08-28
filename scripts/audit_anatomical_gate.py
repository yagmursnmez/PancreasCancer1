"""Bağımsız 3B pankreas kapısının gerçek doğrulama maskelerindeki etkisini denetle."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import nibabel as nib
import numpy as np

from segmentation_postprocess import validate_and_fuse_segmentation


BASE_PATH = Path(__file__).resolve().parents[1]


def dice(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=bool)
    right = np.asarray(right, dtype=bool)
    denominator = int(np.count_nonzero(left)) + int(np.count_nonzero(right))
    if denominator == 0:
        return 1.0
    return float(2 * np.count_nonzero(left & right) / denominator)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pred-dir", type=Path, required=True)
    parser.add_argument("--label-dir", type=Path, required=True)
    parser.add_argument("--gate-dir", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=BASE_PATH / "metrics" / "anatomical_gate_audit.json",
    )
    args = parser.parse_args()

    config = json.loads((BASE_PATH / "config.json").read_text(encoding="utf-8"))[
        "anatomical_gate"
    ]
    records = []
    for case_dir in sorted(path for path in args.gate_dir.iterdir() if path.is_dir()):
        case_id = case_dir.name
        pred_path = args.pred_dir / f"{case_id}.nii.gz"
        label_path = args.label_dir / f"{case_id}.nii.gz"
        gate_path = case_dir / "pancreas.nii.gz"
        if not (pred_path.exists() and label_path.exists() and gate_path.exists()):
            continue

        pred_img = nib.load(str(pred_path))
        pred = np.asanyarray(pred_img.dataobj).astype(np.uint8, copy=False)
        truth = np.asanyarray(nib.load(str(label_path)).dataobj).astype(np.uint8, copy=False)
        gate = np.asanyarray(nib.load(str(gate_path)).dataobj)
        final, qc = validate_and_fuse_segmentation(
            pred, gate, nib.affines.voxel_sizes(pred_img.affine), config
        )
        records.append({
            "case_id": case_id,
            "raw_organ_dice": round(dice(pred > 0, truth > 0), 4),
            "final_organ_dice": round(dice(final > 0, truth > 0), 4),
            "gate_organ_dice": round(dice(gate > 0, truth > 0), 4),
            "raw_tumor_dice": round(dice(pred == 2, truth == 2), 4),
            "final_tumor_dice": round(dice(final == 2, truth == 2), 4),
            "raw_tumor_detected": bool(np.any(pred == 2)),
            "final_tumor_detected": bool(np.any(final == 2)),
            "truth_tumor_voxels": int(np.count_nonzero(truth == 2)),
            "quality": qc,
        })

    if not records:
        print("Eşleşen doğrulama vakası bulunamadı.")
        return 1

    def mean(field: str) -> float:
        return round(float(np.mean([record[field] for record in records])), 4)

    report = {
        "created_at": datetime.now().isoformat(),
        "case_count": len(records),
        "scope": "Seçilmiş pozitif MSD fold-0 vakaları; negatif özgüllük ölçmez.",
        "summary": {
            "raw_organ_dice_mean": mean("raw_organ_dice"),
            "final_organ_dice_mean": mean("final_organ_dice"),
            "gate_organ_dice_mean": mean("gate_organ_dice"),
            "raw_tumor_dice_mean": mean("raw_tumor_dice"),
            "final_tumor_dice_mean": mean("final_tumor_dice"),
            "raw_tumor_detected": sum(r["raw_tumor_detected"] for r in records),
            "final_tumor_detected": sum(r["final_tumor_detected"] for r in records),
        },
        "cases": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    print(f"Rapor: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
