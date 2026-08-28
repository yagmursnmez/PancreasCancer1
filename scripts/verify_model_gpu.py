"""Run the real segmentation chain and capture NVIDIA driver telemetry.

This is intentionally separate from ``verify_nvidia_gpu.py``: that script
proves that CUDA matrix operations work, while this one proves that the real
TotalSegmentator/nnU-Net/DiNTS/PanTS inference path uses the selected NVIDIA adapter.
No model, threshold, ROI, mirroring, segmentation, or mask setting is changed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import threading
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WEB_ROOT = PROJECT_ROOT / "web"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify the real PancreasAI model chain on NVIDIA CUDA."
    )
    parser.add_argument("input", type=Path, help="Input 3D NIfTI (.nii or .nii.gz).")
    parser.add_argument(
        "--case-id", default="gpu_model_verification", help="Output mask case id."
    )
    parser.add_argument(
        "--bypass-cache",
        action="store_true",
        help="Use a one-off cache namespace so every heavy model really runs.",
    )
    parser.add_argument(
        "--analysis-profile",
        choices=("balanced", "full_ensemble"),
        default="balanced",
        help="Bounded inference profile to verify (default: balanced).",
    )
    parser.add_argument(
        "--compare-mask",
        type=Path,
        help="Optional existing mask for exact voxel-by-voxel comparison.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "metrics" / "gpu_model_verification.json",
        help="JSON evidence file.",
    )
    parser.add_argument("--sample-seconds", type=float, default=0.5)
    return parser.parse_args()


def _nvidia_sample(web_app) -> dict | None:
    """Use the same zero-subprocess NVML path as the web service."""
    payload = web_app._read_nvidia_gpu_snapshot()
    if not payload:
        return None
    return {
        "sampled_at": datetime.now(timezone.utc).isoformat(),
        "gpu_index": payload.get("gpu_index"),
        "name": payload["name"],
        "gpu_uuid": payload.get("gpu_uuid"),
        "utilization_percent": int(payload["utilization_percent"]),
        "memory_used_mib": int(payload["memory_used_mib"]),
        "memory_total_mib": int(payload["memory_total_mib"]),
        "temperature_c": payload.get("temperature_c"),
        "power_w": (
            round(float(payload["power_w"]), 2)
            if payload.get("power_w") is not None else None
        ),
        "performance_state": payload.get("performance_state"),
        "sm_clock_mhz": int(payload.get("sm_clock_mhz", 0) or 0),
        "source": payload.get("source"),
    }


def _logical_mask_digest(mask) -> str:
    import numpy as np

    contiguous = np.ascontiguousarray(mask, dtype=np.uint8)
    return hashlib.sha256(memoryview(contiguous)).hexdigest()


def _summarize_samples(samples: list[dict]) -> dict:
    usable = [sample for sample in samples if sample.get("name")]
    active = [sample for sample in usable if sample["utilization_percent"] > 0]
    by_stage: dict[str, list[dict]] = defaultdict(list)
    for sample in usable:
        by_stage[sample.get("stage") or "unlabelled"].append(sample)

    def summary(rows: list[dict]) -> dict:
        active_rows = [row for row in rows if row["utilization_percent"] > 0]
        return {
            "samples": len(rows),
            "active_samples": len(active_rows),
            "average_utilization_percent": round(
                sum(row["utilization_percent"] for row in rows) / len(rows), 1
            ) if rows else 0.0,
            "average_active_utilization_percent": round(
                sum(row["utilization_percent"] for row in active_rows) / len(active_rows), 1
            ) if active_rows else 0.0,
            "peak_utilization_percent": max(
                (row["utilization_percent"] for row in rows), default=0
            ),
            "peak_memory_used_mib": max(
                (row["memory_used_mib"] for row in rows), default=0
            ),
            "peak_power_w": max(
                (row["power_w"] for row in rows if row.get("power_w") is not None),
                default=0.0,
            ),
            "peak_sm_clock_mhz": max(
                (row["sm_clock_mhz"] for row in rows), default=0
            ),
        }

    return {
        **summary(usable),
        "active_sample_ratio": round(len(active) / len(usable), 3) if usable else 0.0,
        "by_stage": {stage: summary(rows) for stage, rows in by_stage.items()},
    }


def main() -> int:
    args = _parse_args()
    input_path = args.input.resolve()
    if not input_path.exists():
        raise FileNotFoundError(input_path)

    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    os.environ["CUDA_REQUIRED"] = "True"
    os.environ["CUDA_MODULE_LOADING"] = "LAZY"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "backend:cudaMallocAsync"
    os.environ["SHIM_MCCOMPAT_ENABLE_GPU"] = "1"
    sys.path.insert(0, str(WEB_ROOT))

    import app as web_app
    import nibabel as nib
    import numpy as np

    cuda = web_app._require_cuda()
    if not cuda.get("name") or not cuda.get("cuda"):
        raise RuntimeError(f"NVIDIA CUDA runtime could not be verified: {cuda!r}")

    original_cache_version = web_app.INFERENCE_CACHE_VERSION
    if args.bypass_cache:
        nonce = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        web_app.INFERENCE_CACHE_VERSION = f"{original_cache_version}-proof-{nonce}"

    job_id = f"verify-{int(time.time())}"
    stop = threading.Event()
    samples: list[dict] = []

    def monitor() -> None:
        while not stop.is_set():
            sample = _nvidia_sample(web_app)
            if sample is not None:
                with web_app._JOB_LOCK:
                    progress = dict(web_app._PROGRESS.get(job_id, {}))
                sample["stage"] = progress.get("stage", "starting")
                sample["progress_percent"] = progress.get("percent")
                sample["model_process_active"] = web_app._GPU_MODEL_PROCESS_ACTIVE.is_set()
                sample["model_export_active"] = web_app._GPU_MODEL_EXPORT_ACTIVE.is_set()
                samples.append(sample)
            stop.wait(max(0.2, float(args.sample_seconds)))

    monitor_thread = threading.Thread(target=monitor, name="nvidia-telemetry", daemon=True)
    started_wall = datetime.now(timezone.utc)
    started = time.monotonic()
    monitor_thread.start()
    try:
        mask, error, quality = web_app._run_inference(
            input_path,
            args.case_id,
            job_id=job_id,
            analysis_profile=args.analysis_profile,
        )
    finally:
        stop.set()
        monitor_thread.join(timeout=6)
        web_app.INFERENCE_CACHE_VERSION = original_cache_version
    elapsed = time.monotonic() - started

    evidence = {
        "status": "ok" if mask is not None and not error else "error",
        "started_at": started_wall.isoformat(),
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": round(elapsed, 3),
        "input": str(input_path),
        "case_id": args.case_id,
        "cache_bypassed": bool(args.bypass_cache),
        "analysis_profile": args.analysis_profile,
        "cuda": cuda,
        "selected_gpu": {
            "name": cuda.get("name"),
            "uuid": cuda.get("uuid"),
            "cuda_device": cuda.get("device", "cuda:0"),
            "physical_selector": cuda.get("physical_selector"),
        },
        "cuda_device": cuda.get("device", "cuda:0"),
        "error": error,
        "quality": quality,
        "telemetry": _summarize_samples(samples),
        "samples": samples,
    }

    if mask is not None:
        mask_array = np.asarray(mask, dtype=np.uint8)
        evidence["mask"] = {
            "shape": list(mask_array.shape),
            "sha256_uint8_voxels": _logical_mask_digest(mask_array),
            "pancreas_voxels": int(np.count_nonzero(mask_array == 1)),
            "tumor_voxels": int(np.count_nonzero(mask_array == 2)),
        }
        if args.compare_mask:
            reference_path = args.compare_mask.resolve()
            reference = np.asanyarray(nib.load(str(reference_path)).dataobj).astype(
                np.uint8, copy=False
            )
            same_shape = reference.shape == mask_array.shape
            exact = bool(same_shape and np.array_equal(reference, mask_array))
            evidence["comparison"] = {
                "reference": str(reference_path),
                "same_shape": same_shape,
                "exact_voxel_match": exact,
                "different_voxels": int(np.count_nonzero(reference != mask_array))
                if same_shape else None,
                "reference_sha256_uint8_voxels": _logical_mask_digest(reference),
            }

    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(evidence, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({
        "status": evidence["status"],
        "elapsed_seconds": evidence["elapsed_seconds"],
        "cuda": evidence["cuda"],
        "telemetry": evidence["telemetry"],
        "mask": evidence.get("mask"),
        "comparison": evidence.get("comparison"),
        "output": str(output_path),
        "error": error,
    }, ensure_ascii=False, indent=2))
    return 0 if evidence["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
