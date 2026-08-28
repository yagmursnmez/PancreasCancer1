import sys
import os
import time
import json
import numpy as np
import nibabel as nib
from pathlib import Path

base_path = Path(r"c:\Users\monster\Documents\PythonProjects\PancreasCancer")
sys.path.insert(0, str(base_path))

from web.app import convert_dicom_to_nifti, _run_inference

patient2_dir = Path(r"C:\Users\monster\Downloads\tumorlu denemeleri 1. rapor atıldı\tumorlu denemeleri 1. rapor atıldı\2.hasta")
scratch_dir = base_path / ".gemini" / "antigravity" / "brain" / "14e53c1a-25ff-494e-92b2-a72a1c439894" / "scratch"

nifti_2hasta = scratch_dir / "2hasta.nii.gz"

print("--- Converting 2.hasta DICOM to NIfTI ---")
ok = convert_dicom_to_nifti(patient2_dir, nifti_2hasta)
print(f"DICOM conversion success={ok}: {nifti_2hasta}")

if nifti_2hasta.exists():
    img = nib.load(str(nifti_2hasta))
    print(f"NIfTI shape: {img.shape}, spacing: {img.header.get_zooms()[:3]}")

print("\n--- Running _run_inference on 2.hasta ---")
t0 = time.time()
mask_data, err, quality = _run_inference(
    ct_path=nifti_2hasta,
    case_id="2.hasta_TEST",
    job_id="test_2hasta_job",
    force_model_rerun=True,
    analysis_profile="balanced"
)
elapsed = time.time() - t0

print(f"\nInference completed in {elapsed:.2f}s")
print(f"Error: {err}")
print(f"Status: {quality.get('status')}")
print(f"Has tumor: {quality.get('has_tumor')}")
print(f"Pancreas voxels: {quality.get('pancreas_voxels')} ({quality.get('pancreas_ml')} ml)")
print(f"Tumor voxels: {quality.get('tumor_voxels')} ({quality.get('tumor_ml')} ml)")
print(f"Unverified voxels: {quality.get('unverified_tumor_voxels')} ({quality.get('unverified_tumor_ml')} ml)")
print(f"Reason: {quality.get('reason')}")

print("\nTumor Components Detailed Records:")
for c in quality.get("tumor_components", []):
    print(f"  {c}")

print("\nUnverified Candidate Records:")
for c in quality.get("unverified_tumor_components", []):
    print(f"  {c}")
