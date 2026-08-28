"""Verify that this project performs sustained CUDA work on the NVIDIA GPU."""

from __future__ import annotations

import argparse
import json
import os
import time


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=float, default=30.0)
    args = parser.parse_args()

    # CUDA_VISIBLE_DEVICES selects the physical NVIDIA adapter. Windows Task
    # Manager numbering is unrelated and is deliberately not assumed.
    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    os.environ["CUDA_MODULE_LOADING"] = "LAZY"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "backend:cudaMallocAsync"
    os.environ["SHIM_MCCOMPAT_ENABLE_GPU"] = "1"

    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        raise RuntimeError(
            "NVIDIA CUDA aygiti bulunamadi; CPU testi yapilmadi. "
            f"cuda_available={torch.cuda.is_available()} "
            f"device_count={torch.cuda.device_count()}"
        )

    device = torch.device("cuda:0")
    properties = torch.cuda.get_device_properties(device)
    if not torch.version.cuda:
        raise RuntimeError("PyTorch CUDA derlemesi kullanilmiyor.")

    # Keep the matrices large enough to produce a clearly visible Compute/CUDA
    # graph without consuming all 6 GB of VRAM.
    size = 6144
    left = torch.randn((size, size), device=device, dtype=torch.float16)
    right = torch.randn((size, size), device=device, dtype=torch.float16)
    torch.cuda.synchronize(device)
    started = time.monotonic()
    iterations = 0
    checksum = 0.0
    while time.monotonic() - started < max(1.0, args.seconds):
        output = torch.mm(left, right)
        left, right = right, output
        # Keep values finite during a long repeated matrix workload.
        left.mul_(0.0001).clamp_(-1.0, 1.0)
        iterations += 1
    checksum = float(left[0, 0].item())
    torch.cuda.synchronize(device)
    elapsed = time.monotonic() - started

    print(json.dumps({
        "status": "ok",
        "pid": os.getpid(),
        "cuda_device": str(device),
        "gpu_uuid": str(getattr(properties, "uuid", "") or ""),
        "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "name": properties.name,
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "vram_total_gb": round(properties.total_memory / (1024 ** 3), 2),
        "vram_peak_mib": round(torch.cuda.max_memory_allocated(device) / (1024 ** 2), 1),
        "elapsed_seconds": round(elapsed, 2),
        "iterations": iterations,
        "checksum": checksum,
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
