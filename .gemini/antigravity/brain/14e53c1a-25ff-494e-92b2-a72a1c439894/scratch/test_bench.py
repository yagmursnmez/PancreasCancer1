import sys
import os
import time
import subprocess
from pathlib import Path

sys.path.insert(0, r"c:\Users\monster\Documents\PythonProjects\PancreasCancer")

from web.app import convert_dicom_to_nifti, ANATOMICAL_GATE_CONFIG

patient_dir = Path(r"C:\Users\monster\Downloads\ESOGU TUMORLU\PATIENT2557311")
scratch_dir = Path(r"c:\Users\monster\Documents\PythonProjects\PancreasCancer\.gemini\antigravity\brain\14e53c1a-25ff-494e-92b2-a72a1c439894\scratch")
scratch_dir.mkdir(parents=True, exist_ok=True)

test_nifti = scratch_dir / "PATIENT2557311.nii.gz"

if not test_nifti.exists():
    print("Converting DICOM to NIfTI...")
    t0 = time.time()
    ok = convert_dicom_to_nifti(patient_dir, test_nifti)
    print(f"DICOM conversion success={ok} in {time.time()-t0:.2f}s: {test_nifti}")
else:
    print("NIfTI test file already exists:", test_nifti)

totalseg_exe = Path(r"c:\Users\monster\Documents\PythonProjects\PancreasCancer\.venv_totalseg\Scripts\TotalSegmentator.exe")

def benchmark_totalseg(fast=False):
    out_dir = scratch_dir / ("out_fast" if fast else "out_full")
    out_dir.mkdir(exist_ok=True)
    cmd = [
        str(totalseg_exe), "-i", str(test_nifti), "-o", str(out_dir),
        "--roi_subset", "pancreas", "--device", "gpu", "--nr_thr_saving", "1"
    ]
    if fast:
        cmd.append("--fast")
    
    print(f"\n--- Testing TotalSegmentator (fast={fast}) ---")
    print("Command:", " ".join(cmd))
    t0 = time.time()
    res = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - t0
    print(f"Elapsed time: {elapsed:.2f}s")
    print("Return code:", res.returncode)
    if res.returncode != 0:
        print("Stderr:", res.stderr[-1000:])
    else:
        print("Success! Output files:", list(out_dir.glob("*.nii.gz")))

if __name__ == "__main__":
    benchmark_totalseg(fast=True)
