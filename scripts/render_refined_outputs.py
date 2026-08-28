"""Render calibrated 2D and layered 3D outputs for a refined patient mask."""

from __future__ import annotations

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np

from reconstruct_3d import (
    create_interactive_html,
    extract_surface_mesh,
    render_3d_visualization,
)


def render_montage(ct: np.ndarray, mask: np.ndarray, uncertainty: np.ndarray, affine: np.ndarray, output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.patches as patches
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    tumor_counts = (mask == 2).sum(axis=(0, 1))
    gland_counts = (mask > 0).sum(axis=(0, 1))
    tumor_slices = np.flatnonzero(tumor_counts)
    gland_slices = np.flatnonzero(gland_counts)
    ranked = tumor_slices[np.argsort(-tumor_counts[tumor_slices])][:5].tolist()
    distributed = (
        gland_slices[np.linspace(0, len(gland_slices) - 1, min(8, len(gland_slices)), dtype=int)].tolist()
        if len(gland_slices) else []
    )
    selected = list(dict.fromkeys(int(x) for x in ranked + distributed))[:8]
    while len(selected) < 8:
        selected.append(selected[-1] if selected else 0)

    fig, axes = plt.subplots(2, 4, figsize=(20, 10), facecolor="#0d0d1a")
    overlay_cmap = ListedColormap(["none", "#27ae60", "#e74c3c"])
    axcodes = nib.aff2axcodes(affine)
    opposite = {"L": "R", "R": "L", "A": "P", "P": "A"}
    right_marker = axcodes[1]
    left_marker = opposite.get(right_marker, "?")
    for axis, z in zip(axes.flat, selected):
        image = np.clip(np.nan_to_num(ct[:, :, z], nan=-150.0), -150.0, 250.0)
        image = (image + 150.0) / 400.0
        labels = mask[:, :, z]
        axis.imshow(image, cmap="gray", origin="upper", vmin=0, vmax=1)
        axis.imshow(
            np.ma.masked_where(labels == 0, labels), cmap=overlay_cmap,
            origin="upper", vmin=0, vmax=2, alpha=0.42,
        )
        core = uncertainty[:, :, z] == 1
        envelope = uncertainty[:, :, z] > 0
        if core.any():
            axis.contour(core, levels=[0.5], colors=["#8e241b"], linewidths=1.5)
        if envelope.any():
            axis.contour(envelope, levels=[0.5], colors=["#f39c12"], linewidths=1.2)
        axis.text(0.01, 0.5, left_marker, transform=axis.transAxes, color="white", weight="bold")
        axis.text(0.96, 0.5, right_marker, transform=axis.transAxes, color="white", weight="bold")
        axis.set_title(
            f"Kesit {z} · pankreas {int((labels == 1).sum())} · tümör {int((labels == 2).sum())}",
            color="white", fontsize=10,
        )
        axis.axis("off")
    fig.suptitle("PanTS ile kalibre edilmiş pankreas ve tümör sınırları", color="#ff6b61", fontsize=19, weight="bold")
    fig.legend(
        handles=[
            patches.Patch(color="#27ae60", alpha=0.55, label="Pankreas"),
            patches.Patch(color="#e74c3c", alpha=0.65, label="Kalibre ana tümör sınırı"),
            patches.Patch(color="#8e241b", label="Yüksek güvenli çekirdek"),
            patches.Patch(color="#f39c12", alpha=0.45, label="Duyarlı belirsizlik sınırı"),
        ],
        loc="lower center", ncol=4, facecolor="#1a1a2e", labelcolor="white",
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.96))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=170, bbox_inches="tight", facecolor="#0d0d1a")
    plt.close(fig)
    print(f"Montage saved: {output}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ct", required=True, type=Path)
    parser.add_argument("--mask", required=True, type=Path)
    parser.add_argument("--uncertainty", required=True, type=Path)
    parser.add_argument("--output-montage", required=True, type=Path)
    parser.add_argument("--output-html", required=True, type=Path)
    parser.add_argument("--output-3d-png", required=True, type=Path)
    args = parser.parse_args()

    ct_nii = nib.load(str(args.ct))
    mask_nii = nib.load(str(args.mask))
    unc_nii = nib.load(str(args.uncertainty))
    ct = np.asanyarray(ct_nii.dataobj)
    mask = np.asanyarray(mask_nii.dataobj).astype(np.uint8)
    uncertainty = np.asanyarray(unc_nii.dataobj).astype(np.uint8)
    if ct.shape != mask.shape or mask.shape != uncertainty.shape:
        raise ValueError(f"Shape mismatch: {ct.shape}, {mask.shape}, {uncertainty.shape}")
    render_montage(ct, mask, uncertainty, mask_nii.affine, args.output_montage)

    spacing = tuple(float(x) for x in nib.affines.voxel_sizes(mask_nii.affine))
    pancreas_mesh = extract_surface_mesh(mask, 1, spacing, 2, affine=mask_nii.affine)
    tumor_mesh = extract_surface_mesh(mask, 2, spacing, 1, affine=mask_nii.affine)
    core_labels = (uncertainty == 1).astype(np.uint8)
    envelope_labels = (uncertainty > 0).astype(np.uint8)
    core_mesh = extract_surface_mesh(core_labels, 1, spacing, 1, affine=mask_nii.affine)
    envelope_mesh = extract_surface_mesh(envelope_labels, 1, spacing, 1, affine=mask_nii.affine)
    create_interactive_html(
        pancreas_mesh, tumor_mesh, True, args.output_html, coordinate_system="RAS",
        tumor_core_mesh=core_mesh, tumor_envelope_mesh=envelope_mesh,
    )
    render_3d_visualization(pancreas_mesh, tumor_mesh, True, args.output_3d_png)


if __name__ == "__main__":
    main()
