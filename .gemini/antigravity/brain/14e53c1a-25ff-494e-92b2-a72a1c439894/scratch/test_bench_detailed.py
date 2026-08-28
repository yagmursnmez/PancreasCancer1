import sys
import os
import time
import subprocess
from pathlib import Path

scratch_dir = Path(r"c:\Users\monster\Documents\PythonProjects\PancreasCancer\.gemini\antigravity\brain\14e53c1a-25ff-494e-92b2-a72a1c439894\scratch")
test_nifti = scratch_dir / "PATIENT2557311.nii.gz"
totalseg_exe = Path(r"c:\Users\monster\Documents\PythonProjects\PancreasCancer\.venv_totalseg\Scripts\TotalSegmentator.exe")

def benchmark_totalseg_detailed(fast=True):
    out_dir = scratch_dir / ("out_fast_run2" if fast else "out_full_run2")
    out_dir.mkdir(exist_ok=True)
    cmd = [
        str(totalseg_exe), "-i", str(test_nifti), "-o", str(out_dir),
        "--roi_subset", "pancreas", "--device", "gpu", "--nr_thr_saving", "1"
    ]
    if fast:
        cmd.append("--fast")
    
    print(f"\n--- Detailed TotalSegmentator Test (fast={fast}) ---")
    t0 = time.time()
    res = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - t0
    print(f"Elapsed time: {elapsed:.2f}s")
    print("Return code:", res.returncode)
    print("Stdout:", res.stdout)
    print("Stderr:", res.stderr)

if __name__ == "__main__":
    benchmark_totalseg_detailed(fast=True)
