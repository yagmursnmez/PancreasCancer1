"""Create an affine-preserving, generously padded pancreas ROI for external models."""

from __future__ import annotations

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--mask", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--margin-mm", type=float, default=80.0)
    args = parser.parse_args()

    image_nii = nib.load(str(args.image))
    mask_nii = nib.load(str(args.mask))
    image = np.asanyarray(image_nii.dataobj)
    mask = np.asanyarray(mask_nii.dataobj) > 0
    if image.shape != mask.shape:
        raise ValueError(f"Image/mask shape mismatch: {image.shape} vs {mask.shape}")
    coords = np.argwhere(mask)
    if coords.size == 0:
        raise ValueError("The seed mask is empty")

    spacing = np.sqrt((image_nii.affine[:3, :3] ** 2).sum(axis=0))
    margin_vox = np.ceil(args.margin_mm / spacing).astype(int)
    lo = np.maximum(coords.min(axis=0) - margin_vox, 0)
    hi = np.minimum(coords.max(axis=0) + margin_vox + 1, image.shape)
    slices = tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))
    cropped = image[slices]

    affine = image_nii.affine.copy()
    affine[:3, 3] = (image_nii.affine @ np.r_[lo, 1.0])[:3]
    header = image_nii.header.copy()
    output = nib.Nifti1Image(cropped, affine, header)
    output.set_qform(affine, int(image_nii.header["qform_code"]))
    output.set_sform(affine, int(image_nii.header["sform_code"]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    nib.save(output, str(args.output))
    print(
        f"ROI saved: shape={cropped.shape}, start={lo.tolist()}, end={hi.tolist()}, "
        f"margin_mm={args.margin_mm:g}"
    )


if __name__ == "__main__":
    main()
