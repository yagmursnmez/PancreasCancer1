import sys
import os
import json
import numpy as np
import nibabel as nib
from pathlib import Path

base_path = Path(r"c:\Users\monster\Documents\PythonProjects\PancreasCancer")
sys.path.insert(0, str(base_path / "scripts"))
sys.path.insert(0, str(base_path))

from segmentation_postprocess import (
    validate_and_fuse_segmentation,
    extract_unverified_tumor_candidates,
    _settings
)

# Test with modified parameters
custom_config = {
    "unverified_tumor_support_radius_mm": 5.0,        # 5 mm max distance (was 30 mm!)
    "min_unverified_tumor_support_fraction": 0.20,     # 20% pancreas anchoring (was 1%!)
    "min_unverified_tumor_component_ml": 0.50,         # 0.5 mL volume threshold (was 0.3 mL)
    "min_unverified_tumor_slices": 2,                  # At least 2 slices (was 1 slice!)
}

print("Testing strict candidate extraction parameters:")
for k, v in custom_config.items():
    print(f"  {k}: {v}")

# Create synthetic volume with:
# 1. Pancreas (center)
# 2. Real Tumor inside pancreas (1.0 mL)
# 3. Single-model candidate inside pancreas (0.8 mL, 4 slices, 80% support)
# 4. Noise blob 25mm away from pancreas (0.4 mL, 1 slice)

shape = (100, 100, 50)
spacing = (1.0, 1.0, 2.0) # 1x1x2 mm per voxel -> 2 mm^3 per voxel -> 0.002 mL per voxel

gate = np.zeros(shape, dtype=bool)
gate[40:60, 40:60, 20:30] = True # Pancreas: 20x20x10 voxels = 4000 voxels = 8.0 mL

primary_tumor = np.zeros(shape, dtype=bool)
# Candidate A inside pancreas: 8x8x4 = 256 voxels = 0.512 mL (supported by primary model)
primary_tumor[45:53, 45:53, 22:26] = True

# Noise blob B far away (25mm away in Z): 10x10x1 = 100 voxels = 0.2 mL (1 slice)
primary_tumor[10:20, 10:20, 5:6] = True

secondary_tumor = np.zeros(shape, dtype=bool)
# Secondary model detects Candidate A partially: 6x6x4 = 144 voxels
secondary_tumor[46:52, 46:52, 22:26] = True

raw_primary = np.zeros(shape, dtype=np.uint8)
raw_primary[gate] = 1
raw_primary[primary_tumor] = 2

raw_secondary = np.zeros(shape, dtype=np.uint8)
raw_secondary[gate] = 1
raw_secondary[secondary_tumor] = 2

print("\n--- Test 1: Noise Blob Filtering ---")
cand_mask, cand_recs = extract_unverified_tumor_candidates(
    raw_primary, gate, spacing, config=custom_config, secondary_mask=raw_secondary
)
print(f"Candidates extracted count: {len(cand_recs)}")
for r in cand_recs:
    print(f"  Comp {r['component']}: voxels={r['voxels']}, ml={r['ml']}, slices={r['slices']}, support_frac={r['broad_support_fraction']}, meaningful={r['meaningful_unverified_candidate']}")

print(f"\nTotal candidate voxels kept: {np.count_nonzero(cand_mask)}")

print("\n--- Test 2: Validation & Fusion Output ---")
final_mask, qc = validate_and_fuse_segmentation(
    raw_primary, gate, spacing, config=custom_config, secondary_mask=raw_secondary
)

print(f"Final status: {qc['status']}")
print(f"Has tumor: {qc['has_tumor']}")
print(f"Verified tumor voxels: {qc['tumor_voxels']} ({qc['tumor_ml']} mL)")
print(f"Unverified candidate voxels: {qc['unverified_tumor_voxels']}")
print(f"Reason: {qc['reason']}")
