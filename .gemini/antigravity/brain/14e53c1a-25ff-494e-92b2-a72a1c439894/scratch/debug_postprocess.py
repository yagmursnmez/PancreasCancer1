import sys
import os
import json
import numpy as np
import nibabel as nib
from pathlib import Path

base_path = Path(r"c:\Users\monster\Documents\PythonProjects\PancreasCancer")
sys.path.insert(0, str(base_path / "scripts"))
sys.path.insert(0, str(base_path))

from segmentation_postprocess import validate_and_fuse_segmentation, extract_unverified_tumor_candidates

scratch_dir = base_path / ".gemini" / "antigravity" / "brain" / "14e53c1a-25ff-494e-92b2-a72a1c439894" / "scratch"
test_nifti = scratch_dir / "PATIENT2557311.nii.gz"

with open(base_path / "config.json") as f:
    config = json.load(f)

gate_cfg = config["anatomical_gate"]

print("Config anatomical_gate settings:")
for k, v in gate_cfg.items():
    if "tumor" in k or "unverified" in k:
        print(f"  {k}: {v}")

# Let's inspect test inference outputs if available
nnunet_out = scratch_dir / "nnunet_out_npp2" / "testcase.nii.gz"
if nnunet_out.exists():
    raw_mask = np.asanyarray(nib.load(str(nnunet_out)).dataobj)
    print(f"\nRaw nnU-Net mask shape: {raw_mask.shape}")
    print(f"Pancreas voxels: {np.count_nonzero(raw_mask==1)}")
    print(f"Tumor voxels: {np.count_nonzero(raw_mask==2)}")
