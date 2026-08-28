import sys
import os
import time
import subprocess
import json
from pathlib import Path

base_path = Path(r"c:\Users\monster\Documents\PythonProjects\PancreasCancer")
sys.path.insert(0, str(base_path))

with open(base_path / "config.json") as f:
    cfg = json.load(f)

for k, v in cfg.get("paths", {}).items():
    if k.startswith("nnunet_"):
        env_key = "nnUNet_" + k.split("_")[1]
        os.environ[env_key] = v

os.environ["nnUNet_raw"] = str(base_path / "data" / "nnunet_raw")
os.environ["nnUNet_preprocessed"] = str(base_path / "data" / "nnunet_preprocessed")
os.environ["nnUNet_results"] = str(base_path / "data" / "nnunet_results")

scratch_dir = base_path / ".gemini" / "antigravity" / "brain" / "14e53c1a-25ff-494e-92b2-a72a1c439894" / "scratch"
test_nifti = scratch_dir / "PATIENT2557311.nii.gz"
model_input = scratch_dir / "nnunet_in"
model_input.mkdir(exist_ok=True)

import shutil
input_file = model_input / "testcase_0000.nii.gz"
shutil.copy2(test_nifti, input_file)

checkpoint_path = base_path / "data" / "nnunet_results" / "Dataset007_Pancreas" / "nnUNetTrainer__nnUNetPlans__2d" / "fold_0" / "checkpoint_best.pth"

def test_nnunet(npp=1):
    out_dir = scratch_dir / f"nnunet_out_npp{npp}"
    out_dir.mkdir(exist_ok=True)
    cmd = [
        sys.executable, "-c",
        "from nnunetv2.inference.predict_from_raw_data import predict_entry_point; predict_entry_point()",
        "-i", str(model_input), "-o", str(out_dir),
        "-d", "007", "-c", "2d", "-f", "0",
        "-step_size", "0.5", "-npp", str(npp), "-nps", str(npp),
        "-chk", str(checkpoint_path), "-device", "cuda", "--disable_tta"
    ]
    print(f"\n--- Testing nnU-Net (npp={npp}, nps={npp}) ---")
    t0 = time.time()
    res = subprocess.run(cmd, capture_output=True, text=True, env=os.environ)
    elapsed = time.time() - t0
    print(f"Elapsed time: {elapsed:.2f}s | Return code: {res.returncode}")
    return elapsed

if __name__ == "__main__":
    t1 = test_nnunet(npp=1)
    t2 = test_nnunet(npp=2)
    t5 = test_nnunet(npp=5)
    print(f"\nSummary: npp=1: {t1:.2f}s | npp=2: {t2:.2f}s | npp=5: {t5:.2f}s")
