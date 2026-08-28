"""2B nnU-Net + 3B DiNTS + anatomik kapı tümör ensemble denetimi."""

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
    denominator = int(np.count_nonzero(left)) + int(np.count_nonzero(right))
    return float(2 * np.count_nonzero(left & right) / denominator) if denominator else 1.0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nnunet-dir", type=Path, required=True)
    parser.add_argument("--dints-dir", type=Path, required=True)
    parser.add_argument("--gate-dir", type=Path, required=True)
    parser.add_argument("--label-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    cfg = json.loads((BASE_PATH / "config.json").read_text(encoding="utf-8"))[
        "anatomical_gate"
    ]
    records = []
    for case_dir in sorted(path for path in args.gate_dir.iterdir() if path.is_dir()):
        case_id = case_dir.name
        paths = {
            "nnunet": args.nnunet_dir / f"{case_id}.nii.gz",
            "dints": args.dints_dir / f"{case_id}_0000" / f"{case_id}_0000_trans.nii.gz",
            "gate": case_dir / "pancreas.nii.gz",
            "truth": args.label_dir / f"{case_id}.nii.gz",
        }
        if not all(path.exists() for path in paths.values()):
            continue

        pred_img = nib.load(str(paths["nnunet"]))
        pred = np.asanyarray(pred_img.dataobj).astype(np.uint8, copy=False)
        dints = np.asanyarray(nib.load(str(paths["dints"])).dataobj).astype(np.uint8, copy=False)
        gate = np.asanyarray(nib.load(str(paths["gate"])).dataobj)
        truth = np.asanyarray(nib.load(str(paths["truth"])).dataobj).astype(np.uint8, copy=False)
        ensemble = np.maximum(pred, dints)
        final, quality = validate_and_fuse_segmentation(
            ensemble, gate, nib.affines.voxel_sizes(pred_img.affine), cfg
        )
        records.append({
            "case_id": case_id,
            "nnunet_tumor_dice": round(dice(pred == 2, truth == 2), 4),
            "dints_tumor_dice": round(dice(dints == 2, truth == 2), 4),
            "verified_ensemble_tumor_dice": round(dice(final == 2, truth == 2), 4),
            "verified_tumor_detected": bool(np.any(final == 2)),
            "quality": quality,
        })

    if not records:
        print("Eşleşen vaka bulunamadı.")
        return 1

    def mean(field: str) -> float:
        return round(float(np.mean([record[field] for record in records])), 4)

    report = {
        "created_at": datetime.now().isoformat(),
        "scope": "Seçilmiş 8 pozitif MSD fold-0 vakası; negatif özgüllük ölçmez.",
        "case_count": len(records),
        "summary": {
            "nnunet_tumor_dice_mean": mean("nnunet_tumor_dice"),
            "dints_tumor_dice_mean": mean("dints_tumor_dice"),
            "verified_ensemble_tumor_dice_mean": mean("verified_ensemble_tumor_dice"),
            "verified_tumor_detected": sum(r["verified_tumor_detected"] for r in records),
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
