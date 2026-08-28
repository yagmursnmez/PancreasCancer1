import sys
import os
import time
import json
from pathlib import Path

base_path = Path(r"c:\Users\monster\Documents\PythonProjects\PancreasCancer")
sys.path.insert(0, str(base_path))

from web.app import _run_inference

scratch_dir = base_path / ".gemini" / "antigravity" / "brain" / "14e53c1a-25ff-494e-92b2-a72a1c439894" / "scratch"
test_nifti = scratch_dir / "PATIENT2557311.nii.gz"

print("--- Running Full Pipeline Benchmark on PATIENT2557311 ---")
t_start = time.time()

# Run inference directly with force_model_rerun=True to test actual GPU execution speed
mask_data, err, quality = _run_inference(
    ct_path=test_nifti,
    case_id="PATIENT2557311_BENCH",
    job_id="test_job_123",
    force_model_rerun=True,
    analysis_profile="balanced"
)

t_total = time.time() - t_start

print(f"\n==========================================")
print(f"BENCHMARK COMPLETE IN {t_total:.2f} SECONDS")
print(f"Error: {err}")
if quality:
    print("\nTimings Breakdown:")
    timings = quality.get("timings", {})
    for stage, s in timings.items():
        print(f"  - {stage}: {s:.2f}s")
    print(f"\nModels evaluated: {quality.get('models')}")
    print(f"Pancreas voxels: {quality.get('pancreas_voxels')} ({quality.get('pancreas_ml'):.2f} ml)")
    print(f"Tumor voxels: {quality.get('tumor_voxels')} ({quality.get('tumor_ml'):.2f} ml)")
    print(f"Has tumor: {quality.get('has_tumor')}")
print(f"==========================================")
