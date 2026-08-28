"""
============================================================
ADIM 9: WEB UYGULAMASI — Flask
============================================================
Kullanım:
    python web/app.py
    Tarayıcı: http://localhost:5000
============================================================
"""

import os
import sys
import json
import time
import uuid
import ctypes
import shutil
import subprocess
import gc
import hashlib
import logging
import re
import math
import tempfile
import threading
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from pathlib import Path
from datetime import datetime
from logging.handlers import RotatingFileHandler
import numpy as np
import nibabel as nib

from flask import (
    Flask, Request as FlaskRequest, render_template, request, redirect,
    url_for, flash, jsonify, send_file, send_from_directory
)
from werkzeug.utils import secure_filename
from werkzeug.exceptions import RequestEntityTooLarge

# ============================================================
# PATH AYARLARI
# ⚠️ BU SATIRI DEĞİŞTİR
# ============================================================
BASE_PATH  = Path(__file__).parent.parent
UPLOAD_DIR = BASE_PATH / "web" / "static" / "uploads"
RESULT_DIR = BASE_PATH / "web" / "static" / "results"
MASK_DIR   = BASE_PATH / "data" / "inference_output" / "segmentation_masks"
VIZ_DIR    = BASE_PATH / "data" / "inference_output" / "visualizations"
RECON_3D_DIR = BASE_PATH / "data" / "inference_output" / "3d_reconstructions"
INFERENCE_CACHE_DIR = BASE_PATH / "data" / "inference_output" / "inference_cache"
DICOM_CONVERSION_CACHE_DIR = (
    BASE_PATH / "data" / "inference_output" / "dicom_conversion_cache"
)
GPU_RUN_DIR = BASE_PATH / "metrics" / "gpu_runs"
WEB_LOG_DIR = BASE_PATH / "logs"

for d in [
    UPLOAD_DIR, RESULT_DIR, MASK_DIR, VIZ_DIR, RECON_3D_DIR,
    INFERENCE_CACHE_DIR, DICOM_CONVERSION_CACHE_DIR, GPU_RUN_DIR, WEB_LOG_DIR,
]:
    d.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(BASE_PATH / "scripts"))


def _parse_bool(value, default=False):
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on", "evet"}


def _load_env_file(path: Path):
    """Ek paket gerektirmeden .env yükler; web ayarlarında proje dosyasını esas alır."""
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        value = value.split("#", 1)[0].strip().strip('"').strip("'")
        key = key.strip()
        # Web çalışma ayarlarının tek kaynağı proje .env dosyasıdır. Böylece
        # IDE/önceki terminalden kalan 100 MB gibi eski değerler yeni servisi ezemez.
        if key in {
            "PANCREAS_DEBUG", "FLASK_DEBUG", "MAX_UPLOAD_MB", "MAX_UPLOAD_FILES",
            "MODEL_TIMEOUT_SECONDS", "MODEL_CHECKPOINT", "FLASK_PORT",
            "CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER", "CUDA_REQUIRED",
            "CUDA_MODULE_LOADING", "PYTORCH_CUDA_ALLOC_CONF",
            "SHIM_MCCOMPAT_ENABLE_GPU", "GPU_EVIDENCE_REQUIRED",
            "GPU_TELEMETRY_INTERVAL_SECONDS", "DEFAULT_EXECUTION_MODE",
            "PACS_CONFIG_PATH",
        }:
            os.environ[key] = value
        else:
            os.environ.setdefault(key, value)


_load_env_file(BASE_PATH / ".env")
PACS_CONFIG_PATH = Path(
    os.environ.get("PACS_CONFIG_PATH", str(BASE_PATH / "pacs_config.json"))
)
try:
    with open(BASE_PATH / "config.json", encoding="utf-8") as config_file:
        RUNTIME_CONFIG = json.load(config_file)
except (OSError, json.JSONDecodeError):
    RUNTIME_CONFIG = {}

WEB_CONFIG = RUNTIME_CONFIG.get("web", {})
DEBUG_ENABLED = _parse_bool(
    os.environ.get("PANCREAS_DEBUG"),
    WEB_CONFIG.get("debug", False),
)
FLASK_DEBUG_ENABLED = _parse_bool(os.environ.get("FLASK_DEBUG"), False)
MAX_UPLOAD_MB = int(os.environ.get("MAX_UPLOAD_MB", WEB_CONFIG.get("max_upload_mb", 8192)))
MAX_UPLOAD_FILES = int(os.environ.get("MAX_UPLOAD_FILES", WEB_CONFIG.get("max_upload_files", 5000)))
MODEL_TIMEOUT_SECONDS = int(os.environ.get("MODEL_TIMEOUT_SECONDS", 1800))
MODEL_CHECKPOINT = os.environ.get("MODEL_CHECKPOINT", "checkpoint_final.pth").strip()
TUMOR_MODEL_3D_CONFIG = RUNTIME_CONFIG.get("tumor_model_3d", {})
PANTS_REFINEMENT_CONFIG = RUNTIME_CONFIG.get("pants_refinement", {})
ANATOMICAL_GATE_CONFIG = RUNTIME_CONFIG.get("anatomical_gate", {})
INFERENCE_CONFIG = RUNTIME_CONFIG.get("inference", {})
INFERENCE_CACHE_VERSION = "face-connected-pancreas-v12"
DICOM_CONVERSION_CACHE_VERSION = "dicom-nifti-r3-selected-series-content-sha256"
ARTIFACT_VERSION = "candidate-aware-uncertainty-v4"
CUDA_REQUIRED = _parse_bool(os.environ.get("CUDA_REQUIRED"), True)
CUDA_VISIBLE_DEVICE = os.environ.get("CUDA_VISIBLE_DEVICES", "0").strip() or "0"
DEFAULT_EXECUTION_MODE = str(
    os.environ.get(
        "DEFAULT_EXECUTION_MODE",
        WEB_CONFIG.get("default_execution_mode", "cache_allowed"),
    )
).strip().lower()
if DEFAULT_EXECUTION_MODE not in {"cache_allowed", "fresh_gpu"}:
    DEFAULT_EXECUTION_MODE = "cache_allowed"

ANALYSIS_PROFILES = INFERENCE_CONFIG.get("profiles", {})
DEFAULT_ANALYSIS_PROFILE = str(
    INFERENCE_CONFIG.get("default_profile", "balanced")
).strip().lower()
if DEFAULT_ANALYSIS_PROFILE not in ANALYSIS_PROFILES:
    DEFAULT_ANALYSIS_PROFILE = (
        "full_ensemble" if "full_ensemble" in ANALYSIS_PROFILES
        else next(iter(ANALYSIS_PROFILES), "full_ensemble")
    )


def _analysis_profile(value=None) -> tuple[str, dict]:
    """Return a bounded, cache-safe model profile; never accept free-form flags."""
    name = str(value or DEFAULT_ANALYSIS_PROFILE).strip().lower()
    if name not in ANALYSIS_PROFILES:
        name = DEFAULT_ANALYSIS_PROFILE
    defaults = {
        "label": "Tam 5-model doğrulama",
        "nnunet_tta": True,
        "dints_overlap": 0.625,
        "pants_enabled": True,
    }
    defaults.update(dict(ANALYSIS_PROFILES.get(name, {})))
    defaults["nnunet_tta"] = _parse_bool(defaults.get("nnunet_tta"), True)
    defaults["pants_enabled"] = _parse_bool(defaults.get("pants_enabled"), True)
    defaults["dints_overlap"] = min(
        0.875, max(0.0, float(defaults.get("dints_overlap", 0.625)))
    )
    return name, defaults
GPU_EVIDENCE_REQUIRED = _parse_bool(os.environ.get("GPU_EVIDENCE_REQUIRED"), True)
GPU_TELEMETRY_INTERVAL_SECONDS = max(
    0.5, float(os.environ.get("GPU_TELEMETRY_INTERVAL_SECONDS", "1.0"))
)
DICOM_HEADER_WORKERS = max(1, min(32, int(WEB_CONFIG.get("dicom_header_workers", 32))))
IO_BUFFER_BYTES = max(1, int(WEB_CONFIG.get("io_buffer_mb", 8))) * 1024 * 1024

logger = logging.getLogger("pancreas_ai")
logger.handlers.clear()
logger.propagate = False
logger.setLevel(logging.DEBUG if DEBUG_ENABLED else logging.WARNING)
_log_handler = logging.StreamHandler()
_log_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
logger.addHandler(_log_handler)
_file_log_handler = RotatingFileHandler(
    WEB_LOG_DIR / "pancreas_web.log", maxBytes=10 * 1024 * 1024,
    backupCount=5, encoding="utf-8",
)
_file_log_handler.setFormatter(
    logging.Formatter("%(asctime)s | %(levelname)s | %(process)d | %(threadName)s | %(message)s")
)
logger.addHandler(_file_log_handler)
# The browser polls progress while a model runs. Werkzeug's default access log
# redraws the terminal for every poll and was measurably loading the Intel iGPU.
# Detailed model/progress evidence is already persisted by this module, so normal
# HTTP 200 access lines add noise and contention without diagnostic value.
logging.getLogger("werkzeug").setLevel(logging.WARNING)


def debug_log(step: str, message: str, *args):
    """DEBUG=False olduğunda hiçbir adım/debug satırı üretmez."""
    if DEBUG_ENABLED:
        logger.debug("[%s] " + message, step, *args)


_CUDA_INFO = None
_CUDA_RUNTIME_PROBES = {}
_GPU_MODEL_PROCESS_ACTIVE = threading.Event()
_GPU_MODEL_EXPORT_ACTIVE = threading.Event()
_GPU_TELEMETRY_LOCK = threading.Lock()
_GPU_TELEMETRY_CACHE = {"sampled_at": 0.0, "payload": None}
_GPU_AUDIT_LOCK = threading.Lock()
_GPU_RUN_AUDITS = {}
_JOB_GPU_AUDITS = {}
_NVML_LOCK = threading.Lock()
_NVML_INIT_ATTEMPTED = False
_NVML_LIBRARY = None
_NVML_DEVICE = None
_NVML_DEVICE_NAME = None
_NVML_DEVICE_UUID = None
_NVML_DEVICE_INDEX = None
_NVML_POWER_LIMIT_W = None


class _NvmlUtilization(ctypes.Structure):
    _fields_ = [("gpu", ctypes.c_uint), ("memory", ctypes.c_uint)]


class _NvmlMemory(ctypes.Structure):
    _fields_ = [
        ("total", ctypes.c_ulonglong),
        ("free", ctypes.c_ulonglong),
        ("used", ctypes.c_ulonglong),
    ]


class _NvmlProcessInfoV3(ctypes.Structure):
    _fields_ = [
        ("pid", ctypes.c_uint),
        ("usedGpuMemory", ctypes.c_ulonglong),
        ("gpuInstanceId", ctypes.c_uint),
        ("computeInstanceId", ctypes.c_uint),
    ]


def _gpu_environment(base=None, python_paths=None):
    """Return a child environment pinned to the project's NVIDIA CUDA device."""
    environment = dict(base or os.environ)
    # CUDA_VISIBLE_DEVICES selects an NVIDIA CUDA device. Windows Task Manager's
    # GPU number is a presentation label and is intentionally not used here.
    environment["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    environment["CUDA_VISIBLE_DEVICES"] = CUDA_VISIBLE_DEVICE
    # These change allocation/loading mechanics only. They leave model weights,
    # precision, TTA, thresholds and segmentation decisions untouched.
    environment["CUDA_MODULE_LOADING"] = "LAZY"
    environment["PYTORCH_CUDA_ALLOC_CONF"] = "backend:cudaMallocAsync"
    environment["SHIM_MCCOMPAT_ENABLE_GPU"] = "1"
    if python_paths is not None:
        environment["PYTHONPATH"] = os.pathsep.join(str(path) for path in python_paths)
    return environment


def _probe_cuda_runtime(python_executable, python_paths=None):
    """Verify one model runtime on the selected NVIDIA GPU without retaining it."""
    executable = str(Path(python_executable).resolve())
    paths = tuple(str(Path(path).resolve()) for path in (python_paths or []))
    cache_key = (executable, paths, CUDA_VISIBLE_DEVICE)
    if cache_key in _CUDA_RUNTIME_PROBES:
        return dict(_CUDA_RUNTIME_PROBES[cache_key])
    try:
        probe_code = (
            "import json, os, torch; "
            "assert torch.cuda.is_available() and torch.cuda.device_count() > 0, "
            "'CUDA aygiti bulunamadi'; "
            "assert torch.version.cuda, 'PyTorch CUDA derlemesi kullanilmiyor'; "
            "d=torch.device('cuda:0'); x=torch.ones(1,device=d)*2; "
            "torch.cuda.synchronize(d); assert float(x.item())==2.0; "
            "p=torch.cuda.get_device_properties(d); "
            "print(json.dumps({'device':str(d),'name':p.name,"
            "'uuid':str(getattr(p,'uuid','') or ''),"
            "'pci_bus_id':str(getattr(p,'pci_bus_id','') or ''),"
            "'memory_gb':round(p.total_memory/(1024**3),2),"
            "'torch':torch.__version__,'cuda':torch.version.cuda,"
            "'allocator':torch.cuda.memory.get_allocator_backend(),"
            "'module_loading':os.environ.get('CUDA_MODULE_LOADING')}))"
        )
        completed = subprocess.run(
            [executable, "-c", probe_code], cwd=str(BASE_PATH),
            env=_gpu_environment(python_paths=paths if paths else None), capture_output=True,
            text=True, timeout=45, creationflags=(
                subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
            ),
        )
        if completed.returncode != 0:
            details = (completed.stderr or completed.stdout).strip()[-1500:]
            raise RuntimeError(details or "CUDA alt surec denetimi basarisiz oldu.")
        output_line = next(
            (line for line in reversed(completed.stdout.splitlines()) if line.strip()), ""
        )
        info = json.loads(output_line)
        info["visible_devices"] = CUDA_VISIBLE_DEVICE
        info["python"] = executable
        info["vendor"] = "NVIDIA"
        info["physical_selector"] = CUDA_VISIBLE_DEVICE.split(",", 1)[0].strip()
        _CUDA_RUNTIME_PROBES[cache_key] = info
        return dict(info)
    except Exception as exc:
        if CUDA_REQUIRED:
            raise RuntimeError(f"NVIDIA CUDA zorunlu denetimi basarisiz: {exc}") from exc
        debug_log("CUDA", "CUDA kullanilamiyor; CUDA_REQUIRED=False: %s", exc)
        return {"device": "cpu", "error": str(exc)}


def _require_cuda():
    """Verify the selected NVIDIA CUDA runtime; never silently fall back to CPU."""
    global _CUDA_INFO
    if _CUDA_INFO is None:
        _CUDA_INFO = _probe_cuda_runtime(sys.executable)
    return dict(_CUDA_INFO)


def _normalize_gpu_uuid(value) -> str:
    text = str(value or "").strip()
    if text.lower().startswith("gpu-"):
        text = text[4:]
    return text.lower()


def _selected_gpu_identity() -> dict:
    info = dict(_CUDA_INFO or {})
    selector = str(info.get("physical_selector") or CUDA_VISIBLE_DEVICE).split(",", 1)[0].strip()
    return {
        "name": info.get("name"),
        "uuid": _normalize_gpu_uuid(info.get("uuid")),
        "selector": selector or "0",
        "cuda_device": info.get("device", "cuda:0"),
    }


def _validated_power_w(raw_value, utilization_percent=0, power_limit_w=None):
    """Reject transient/sentinel driver readings without inventing a replacement."""
    try:
        value = float(raw_value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(value) or value <= 0:
        return None
    try:
        limit = float(power_limit_w) if power_limit_w is not None else None
    except (TypeError, ValueError):
        limit = None
    if limit and math.isfinite(limit) and limit > 0:
        if value > max(limit * 1.25, limit + 30.0):
            return None
    # A device reporting no work cannot plausibly draw hundreds of watts. This
    # catches the observed 752.7 W NVML transition artifact while preserving
    # legitimate high-power accelerator samples under load.
    if int(utilization_percent or 0) <= 1 and value > 250.0:
        return None
    if value > 1500.0:
        return None
    return round(value, 1)


def _initialize_nvml():
    """Load NVIDIA's in-process management API without creating a CUDA context."""
    global _NVML_INIT_ATTEMPTED, _NVML_LIBRARY, _NVML_DEVICE, _NVML_DEVICE_NAME
    global _NVML_DEVICE_UUID, _NVML_DEVICE_INDEX, _NVML_POWER_LIMIT_W
    if os.name != "nt":
        return False
    with _NVML_LOCK:
        if _NVML_INIT_ATTEMPTED:
            return _NVML_LIBRARY is not None and _NVML_DEVICE is not None
        _NVML_INIT_ATTEMPTED = True
        try:
            library = ctypes.WinDLL("nvml.dll")
            library.nvmlInit_v2.restype = ctypes.c_int
            library.nvmlDeviceGetHandleByIndex_v2.argtypes = [
                ctypes.c_uint, ctypes.POINTER(ctypes.c_void_p),
            ]
            library.nvmlDeviceGetHandleByIndex_v2.restype = ctypes.c_int
            handle_by_uuid = getattr(library, "nvmlDeviceGetHandleByUUID", None)
            if handle_by_uuid is not None:
                handle_by_uuid.argtypes = [
                    ctypes.c_char_p, ctypes.POINTER(ctypes.c_void_p),
                ]
                handle_by_uuid.restype = ctypes.c_int
            library.nvmlDeviceGetName.argtypes = [
                ctypes.c_void_p, ctypes.c_char_p, ctypes.c_uint,
            ]
            library.nvmlDeviceGetName.restype = ctypes.c_int
            get_uuid = getattr(library, "nvmlDeviceGetUUID", None)
            if get_uuid is not None:
                get_uuid.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_uint]
                get_uuid.restype = ctypes.c_int
            get_index = getattr(library, "nvmlDeviceGetIndex", None)
            if get_index is not None:
                get_index.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint)]
                get_index.restype = ctypes.c_int
            library.nvmlDeviceGetUtilizationRates.argtypes = [
                ctypes.c_void_p, ctypes.POINTER(_NvmlUtilization),
            ]
            library.nvmlDeviceGetUtilizationRates.restype = ctypes.c_int
            library.nvmlDeviceGetMemoryInfo.argtypes = [
                ctypes.c_void_p, ctypes.POINTER(_NvmlMemory),
            ]
            library.nvmlDeviceGetMemoryInfo.restype = ctypes.c_int
            library.nvmlDeviceGetTemperature.argtypes = [
                ctypes.c_void_p, ctypes.c_uint, ctypes.POINTER(ctypes.c_uint),
            ]
            library.nvmlDeviceGetTemperature.restype = ctypes.c_int
            library.nvmlDeviceGetPowerUsage.argtypes = [
                ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint),
            ]
            library.nvmlDeviceGetPowerUsage.restype = ctypes.c_int
            get_power_limit = getattr(library, "nvmlDeviceGetPowerManagementLimit", None)
            if get_power_limit is not None:
                get_power_limit.argtypes = [
                    ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint),
                ]
                get_power_limit.restype = ctypes.c_int
            library.nvmlDeviceGetPerformanceState.argtypes = [
                ctypes.c_void_p, ctypes.POINTER(ctypes.c_int),
            ]
            library.nvmlDeviceGetPerformanceState.restype = ctypes.c_int
            library.nvmlDeviceGetClockInfo.argtypes = [
                ctypes.c_void_p, ctypes.c_uint, ctypes.POINTER(ctypes.c_uint),
            ]
            library.nvmlDeviceGetClockInfo.restype = ctypes.c_int
            process_query = getattr(
                library, "nvmlDeviceGetComputeRunningProcesses_v3", None
            )
            if process_query is not None:
                process_query.argtypes = [
                    ctypes.c_void_p,
                    ctypes.POINTER(ctypes.c_uint),
                    ctypes.POINTER(_NvmlProcessInfoV3),
                ]
                process_query.restype = ctypes.c_int
            process_name = getattr(library, "nvmlSystemGetProcessName", None)
            if process_name is not None:
                process_name.argtypes = [
                    ctypes.c_uint, ctypes.c_char_p, ctypes.c_uint,
                ]
                process_name.restype = ctypes.c_int

            if library.nvmlInit_v2() != 0:
                return False
            device = ctypes.c_void_p()
            identity = _selected_gpu_identity()
            selected = False
            if identity["uuid"] and handle_by_uuid is not None:
                uuid_bytes = f"GPU-{identity['uuid']}".encode("ascii", errors="ignore")
                selected = handle_by_uuid(uuid_bytes, ctypes.byref(device)) == 0
            if not selected and str(identity["selector"]).lower().startswith("gpu-"):
                if handle_by_uuid is not None:
                    selected = handle_by_uuid(
                        str(identity["selector"]).encode("ascii", errors="ignore"),
                        ctypes.byref(device),
                    ) == 0
            if not selected:
                try:
                    physical_index = int(identity["selector"])
                except (TypeError, ValueError):
                    physical_index = 0
                selected = library.nvmlDeviceGetHandleByIndex_v2(
                    physical_index, ctypes.byref(device)
                ) == 0
            if not selected:
                return False
            name_buffer = ctypes.create_string_buffer(128)
            if library.nvmlDeviceGetName(device, name_buffer, len(name_buffer)) != 0:
                return False
            device_name = name_buffer.value.decode("utf-8", errors="replace")
            uuid_text = ""
            if get_uuid is not None:
                uuid_buffer = ctypes.create_string_buffer(96)
                if get_uuid(device, uuid_buffer, len(uuid_buffer)) == 0:
                    uuid_text = uuid_buffer.value.decode("ascii", errors="replace")
            device_index = None
            if get_index is not None:
                index_value = ctypes.c_uint()
                if get_index(device, ctypes.byref(index_value)) == 0:
                    device_index = int(index_value.value)
            power_limit_w = None
            if get_power_limit is not None:
                power_limit = ctypes.c_uint()
                if get_power_limit(device, ctypes.byref(power_limit)) == 0:
                    power_limit_w = round(power_limit.value / 1000.0, 1)
            _NVML_LIBRARY = library
            _NVML_DEVICE = device
            _NVML_DEVICE_NAME = device_name
            _NVML_DEVICE_UUID = uuid_text or (
                f"GPU-{identity['uuid']}" if identity["uuid"] else None
            )
            _NVML_DEVICE_INDEX = device_index
            _NVML_POWER_LIMIT_W = power_limit_w
            return True
        except (AttributeError, OSError, ValueError):
            _NVML_LIBRARY = None
            _NVML_DEVICE = None
            _NVML_DEVICE_NAME = None
            _NVML_DEVICE_UUID = None
            _NVML_DEVICE_INDEX = None
            _NVML_POWER_LIMIT_W = None
            return False


def _read_nvml_gpu_snapshot():
    """Read selected NVIDIA counters without spawning nvidia-smi each second."""
    if not _initialize_nvml():
        return None
    with _NVML_LOCK:
        utilization = _NvmlUtilization()
        memory = _NvmlMemory()
        temperature = ctypes.c_uint()
        power = ctypes.c_uint()
        performance_state = ctypes.c_int()
        sm_clock = ctypes.c_uint()
        memory_clock = ctypes.c_uint()
        if _NVML_LIBRARY.nvmlDeviceGetUtilizationRates(
            _NVML_DEVICE, ctypes.byref(utilization)
        ) != 0:
            return None
        if _NVML_LIBRARY.nvmlDeviceGetMemoryInfo(
            _NVML_DEVICE, ctypes.byref(memory)
        ) != 0:
            return None
        temperature_ok = _NVML_LIBRARY.nvmlDeviceGetTemperature(
            _NVML_DEVICE, 0, ctypes.byref(temperature)
        ) == 0
        power_ok = _NVML_LIBRARY.nvmlDeviceGetPowerUsage(
            _NVML_DEVICE, ctypes.byref(power)
        ) == 0
        pstate_ok = _NVML_LIBRARY.nvmlDeviceGetPerformanceState(
            _NVML_DEVICE, ctypes.byref(performance_state)
        ) == 0
        sm_clock_ok = _NVML_LIBRARY.nvmlDeviceGetClockInfo(
            _NVML_DEVICE, 1, ctypes.byref(sm_clock)
        ) == 0
        memory_clock_ok = _NVML_LIBRARY.nvmlDeviceGetClockInfo(
            _NVML_DEVICE, 2, ctypes.byref(memory_clock)
        ) == 0
    power_w = _validated_power_w(
        power.value / 1000.0 if power_ok else None,
        utilization.gpu,
        _NVML_POWER_LIMIT_W,
    )
    return {
        "gpu_index": _NVML_DEVICE_INDEX,
        "gpu_uuid": _NVML_DEVICE_UUID,
        "cuda_device": _selected_gpu_identity()["cuda_device"],
        "name": _NVML_DEVICE_NAME,
        "utilization_percent": int(utilization.gpu),
        "memory_used_mib": int(memory.used / (1024 ** 2)),
        "memory_total_mib": int(memory.total / (1024 ** 2)),
        "temperature_c": int(temperature.value) if temperature_ok else None,
        "power_w": power_w,
        "power_limit_w": _NVML_POWER_LIMIT_W,
        "power_reading_valid": power_w is not None,
        "performance_state": f"P{performance_state.value}" if pstate_ok else None,
        "sm_clock_mhz": int(sm_clock.value) if sm_clock_ok else None,
        "memory_clock_mhz": int(memory_clock.value) if memory_clock_ok else None,
        "source": "NVIDIA NVML driver API",
    }


def _read_nvidia_smi_gpu_snapshot():
    """Fallback for systems where NVML cannot be loaded."""
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid,name,utilization.gpu,memory.used,memory.total,"
                "temperature.gpu,power.draw,pstate",
                "--format=csv,noheader,nounits",
            ],
            cwd=str(BASE_PATH), capture_output=True, text=True, timeout=3,
            creationflags=(subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0),
        )
        if completed.returncode != 0:
            return None
        identity = _selected_gpu_identity()
        selected_fields = None
        for line in completed.stdout.splitlines():
            fields = [field.strip() for field in line.split(",")]
            if len(fields) != 9:
                continue
            row_uuid = _normalize_gpu_uuid(fields[1])
            uuid_match = bool(identity["uuid"] and row_uuid == identity["uuid"])
            selector_match = (
                fields[0] == identity["selector"] or
                _normalize_gpu_uuid(identity["selector"]) == row_uuid
            )
            if uuid_match or selector_match:
                selected_fields = fields
                break
        if selected_fields is None:
            return None
        fields = selected_fields
        utilization = int(float(fields[3]))
        raw_power = float(fields[7])
        power_w = _validated_power_w(raw_power, utilization)
        payload = {
            "gpu_index": int(fields[0]),
            "gpu_uuid": fields[1],
            "cuda_device": identity["cuda_device"],
            "name": fields[2],
            "utilization_percent": utilization,
            "memory_used_mib": int(float(fields[4])),
            "memory_total_mib": int(float(fields[5])),
            "temperature_c": int(float(fields[6])),
            "power_w": power_w,
            "power_limit_w": None,
            "power_reading_valid": power_w is not None,
            "performance_state": fields[8],
            "source": "NVIDIA driver (nvidia-smi)",
        }
        return payload
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def _read_nvidia_gpu_snapshot():
    """Read one selected-device sample directly from NVIDIA's driver."""
    payload = _read_nvml_gpu_snapshot() or _read_nvidia_smi_gpu_snapshot()
    if payload:
        with _GPU_TELEMETRY_LOCK:
            _GPU_TELEMETRY_CACHE.update({
                "sampled_at": time.monotonic(), "payload": payload,
            })
        return dict(payload)
    return None


def _read_nvml_cuda_processes():
    """Return WDDM CUDA PIDs through NVML, or None when NVML is unavailable."""
    if not _initialize_nvml():
        return None
    process_query = getattr(
        _NVML_LIBRARY, "nvmlDeviceGetComputeRunningProcesses_v3", None
    )
    if process_query is None:
        return None
    with _NVML_LOCK:
        count = ctypes.c_uint(0)
        result = process_query(_NVML_DEVICE, ctypes.byref(count), None)
        # NVML_ERROR_INSUFFICIENT_SIZE=7 means count now contains capacity.
        if result not in (0, 7):
            return None
        if count.value == 0:
            return []
        buffer = (_NvmlProcessInfoV3 * count.value)()
        result = process_query(_NVML_DEVICE, ctypes.byref(count), buffer)
        if result != 0:
            return None
        process_name_query = getattr(_NVML_LIBRARY, "nvmlSystemGetProcessName", None)
        processes = []
        for index in range(count.value):
            item = buffer[index]
            process_name = ""
            if process_name_query is not None:
                name_buffer = ctypes.create_string_buffer(1024)
                if process_name_query(item.pid, name_buffer, len(name_buffer)) == 0:
                    process_name = name_buffer.value.decode("utf-8", errors="replace")
            used_memory = int(item.usedGpuMemory / (1024 ** 2))
            if item.usedGpuMemory == ctypes.c_ulonglong(-1).value:
                used_memory = None
            processes.append({
                "pid": int(item.pid),
                "process_name": process_name or f"PID {int(item.pid)}",
                "used_memory_mib": used_memory,
            })
        return processes


def _read_nvidia_smi_cuda_processes():
    """Fallback CUDA PID query for systems where NVML is unavailable."""
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,process_name,used_memory",
                "--format=csv,noheader,nounits",
            ],
            cwd=str(BASE_PATH), capture_output=True, text=True, timeout=3,
            creationflags=(subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0),
        )
        if completed.returncode != 0:
            return []
        processes = []
        for line in completed.stdout.splitlines():
            fields = [field.strip() for field in line.split(",", 2)]
            if len(fields) != 3:
                continue
            try:
                pid = int(fields[0])
            except ValueError:
                continue
            memory_text = fields[2].replace("MiB", "").strip()
            try:
                used_memory_mib = int(float(memory_text))
            except ValueError:
                used_memory_mib = None
            processes.append({
                "pid": pid,
                "process_name": fields[1],
                "used_memory_mib": used_memory_mib,
            })
        return processes
    except (OSError, subprocess.SubprocessError):
        return []


def _nvidia_cuda_processes():
    """Return CUDA-context PIDs as reported by the NVIDIA driver (WDDM-safe)."""
    processes = _read_nvml_cuda_processes()
    return processes if processes is not None else _read_nvidia_smi_cuda_processes()


def _runtime_resource_snapshot(launcher_pid: int):
    """Sample model-process and whole-system CPU/RAM/I/O without a subprocess."""
    try:
        import psutil

        root = psutil.Process(int(launcher_pid))
        processes = [root]
        try:
            processes.extend(root.children(recursive=True))
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            pass
        unique = {int(process.pid): process for process in processes}
        cpu_time_s = 0.0
        rss_bytes = 0
        read_bytes = 0
        write_bytes = 0
        thread_count = 0
        live_pids = []
        for pid, process in unique.items():
            try:
                cpu_times = process.cpu_times()
                memory = process.memory_info()
                cpu_time_s += float(cpu_times.user + cpu_times.system)
                rss_bytes += int(memory.rss)
                thread_count += int(process.num_threads())
                try:
                    io = process.io_counters()
                    read_bytes += int(io.read_bytes)
                    write_bytes += int(io.write_bytes)
                except (psutil.AccessDenied, psutil.NoSuchProcess, AttributeError):
                    pass
                live_pids.append(pid)
            except (psutil.AccessDenied, psutil.NoSuchProcess):
                continue

        cpu_times = psutil.cpu_times()
        system_total_s = float(sum(cpu_times))
        system_idle_s = float(cpu_times.idle + getattr(cpu_times, "iowait", 0.0))
        disk = psutil.disk_io_counters()
        memory = psutil.virtual_memory()
        return {
            "process_pids": sorted(live_pids),
            "process_count": len(live_pids),
            "process_cpu_time_s": round(cpu_time_s, 4),
            "process_rss_mib": round(rss_bytes / (1024 ** 2), 1),
            "process_read_bytes": read_bytes,
            "process_write_bytes": write_bytes,
            "process_threads": thread_count,
            "system_cpu_total_s": system_total_s,
            "system_cpu_idle_s": system_idle_s,
            "system_available_memory_mib": round(memory.available / (1024 ** 2), 1),
            "system_disk_read_bytes": int(disk.read_bytes) if disk else 0,
            "system_disk_write_bytes": int(disk.write_bytes) if disk else 0,
        }
    except Exception:
        # Telemetry must never interrupt a medical-model subprocess. The audit
        # still retains NVIDIA evidence when psutil is unavailable or access is denied.
        return None


def _runtime_resource_deltas(current: dict, previous: dict, elapsed_s: float):
    """Convert cumulative psutil counters into one lightweight interval sample."""
    if not current:
        return None
    payload = {
        key: value for key, value in current.items()
        if not key.endswith("_bytes") and not key.startswith("system_cpu_")
    }
    if not previous or elapsed_s <= 0:
        payload.update({
            "process_cpu_percent": 0.0,
            "system_cpu_percent": 0.0,
            "process_read_mib_delta": 0.0,
            "process_write_mib_delta": 0.0,
            "system_disk_read_mib_delta": 0.0,
            "system_disk_write_mib_delta": 0.0,
        })
        return payload

    process_cpu_delta = max(
        0.0,
        float(current["process_cpu_time_s"]) - float(previous["process_cpu_time_s"]),
    )
    system_total_delta = max(
        0.0,
        float(current["system_cpu_total_s"]) - float(previous["system_cpu_total_s"]),
    )
    system_idle_delta = max(
        0.0,
        float(current["system_cpu_idle_s"]) - float(previous["system_cpu_idle_s"]),
    )
    payload.update({
        # May exceed 100% because one process tree can use multiple CPU cores.
        "process_cpu_percent": round(100.0 * process_cpu_delta / elapsed_s, 1),
        "system_cpu_percent": round(
            100.0 * max(0.0, system_total_delta - system_idle_delta) /
            max(0.001, system_total_delta),
            1,
        ),
        "process_read_mib_delta": round(max(
            0, current["process_read_bytes"] - previous["process_read_bytes"]
        ) / (1024 ** 2), 3),
        "process_write_mib_delta": round(max(
            0, current["process_write_bytes"] - previous["process_write_bytes"]
        ) / (1024 ** 2), 3),
        "system_disk_read_mib_delta": round(max(
            0, current["system_disk_read_bytes"] - previous["system_disk_read_bytes"]
        ) / (1024 ** 2), 3),
        "system_disk_write_mib_delta": round(max(
            0, current["system_disk_write_bytes"] - previous["system_disk_write_bytes"]
        ) / (1024 ** 2), 3),
    })
    return payload


def _nvidia_gpu_snapshot(force=False):
    """Expose selected-device telemetry only while a model stage is active."""
    if not _GPU_MODEL_PROCESS_ACTIVE.is_set() or _GPU_MODEL_EXPORT_ACTIVE.is_set():
        return None
    now = time.monotonic()
    with _GPU_TELEMETRY_LOCK:
        cached = _GPU_TELEMETRY_CACHE.get("payload")
        cache_window = max(0.7, GPU_TELEMETRY_INTERVAL_SECONDS * 1.5)
        if not force and cached and now - _GPU_TELEMETRY_CACHE["sampled_at"] < cache_window:
            return dict(cached)
    payload = _read_nvidia_gpu_snapshot()
    if payload:
        with _GPU_TELEMETRY_LOCK:
            _GPU_TELEMETRY_CACHE.update({"sampled_at": now, "payload": payload})
        return dict(payload)
    return None


_PROGRESS = {}
_JOB_RESULTS = {}
_JOB_LOCK = threading.Lock()
_MODEL_RUN_LOCK = threading.Lock()
_DICOM_EXPORTS = {}
_DICOM_EXPORT_LOCK = threading.Lock()
_DICOM_WORK_LOCK = threading.Lock()


class _AnalysisProcessPriority:
    """Temporarily favor model-side CPU work without using High/Realtime."""

    def __init__(self):
        self._process = None
        self._previous = None

    def __enter__(self):
        if os.name != "nt":
            return self
        try:
            import psutil
            self._process = psutil.Process(os.getpid())
            self._previous = self._process.nice()
            self._process.nice(psutil.ABOVE_NORMAL_PRIORITY_CLASS)
            debug_log(
                "PERFORMANS", "Web inference CPU önceliği geçici olarak AboveNormal",
            )
        except Exception as exc:
            self._process = None
            debug_log("PERFORMANS", "Web inference önceliği değiştirilemedi: %s", exc)
        return self

    def __exit__(self, _exc_type, _exc, _traceback):
        if self._process is not None and self._previous is not None:
            try:
                self._process.nice(self._previous)
                debug_log("PERFORMANS", "Web inference CPU önceliği geri yüklendi")
            except Exception as exc:
                debug_log("PERFORMANS", "Web CPU önceliği geri yüklenemedi: %s", exc)
        return False


def _atomic_write_json(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f".{uuid.uuid4().hex}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _persist_gpu_audit(audit_id: str):
    with _GPU_AUDIT_LOCK:
        audit = _GPU_RUN_AUDITS.get(audit_id)
        if not audit:
            return
        payload = json.loads(json.dumps(audit, ensure_ascii=False, default=str))
        report_path = Path(audit["report_path"])
    _atomic_write_json(report_path, payload)


def _aggregate_gpu_stages(audit: dict) -> dict:
    summaries = [dict(stage.get("summary", {})) for stage in audit.get("stages", [])]
    cuda_pids = sorted({
        int(pid)
        for summary in summaries
        for pid in summary.get("observed_model_cuda_pids", [])
    })
    total_samples = sum(int(item.get("sample_count", 0)) for item in summaries)
    utilization_sum = sum(
        float(item.get("average_utilization_percent", 0.0)) *
        int(item.get("sample_count", 0))
        for item in summaries
    )
    return {
        "stage_count": len(summaries),
        "verified_stage_count": sum(bool(item.get("gpu_evidence")) for item in summaries),
        "gpu_verified": bool(summaries) and all(
            bool(item.get("gpu_evidence")) for item in summaries
        ),
        "evidence_policy": "model_process_cuda_pid_and_device_activity",
        "sample_count": total_samples,
        "total_stage_elapsed_s": round(sum(
            float(item.get("elapsed_s", 0.0)) for item in summaries
        ), 3),
        "average_utilization_percent": round(
            utilization_sum / max(1, total_samples), 1
        ),
        "peak_utilization_percent": max(
            [int(item.get("peak_utilization_percent", 0)) for item in summaries] or [0]
        ),
        "peak_memory_used_mib": max(
            [int(item.get("peak_memory_used_mib", 0)) for item in summaries] or [0]
        ),
        "peak_memory_delta_mib": max(
            [int(item.get("peak_memory_delta_mib", 0)) for item in summaries] or [0]
        ),
        "peak_power_w": max(
            [float(item.get("peak_power_w", 0.0)) for item in summaries] or [0.0]
        ),
        "peak_sm_clock_mhz": max(
            [int(item.get("peak_sm_clock_mhz", 0) or 0) for item in summaries] or [0]
        ),
        "gpu_utilization_active_samples": sum(
            int(item.get("utilization_active_samples", 0)) for item in summaries
        ),
        "observed_model_cuda_pids": cuda_pids,
    }


def _public_gpu_audit(audit_id: str):
    with _GPU_AUDIT_LOCK:
        audit = _GPU_RUN_AUDITS.get(audit_id)
        if not audit:
            return None
        aggregate = _aggregate_gpu_stages(audit)
        active_record = next(
            (stage for stage in reversed(audit.get("stages", []))
             if stage.get("status") == "running"),
            None,
        )
        completed_record = next(
            (stage for stage in reversed(audit.get("stages", []))
             if stage.get("status") in {"completed", "failed"}),
            None,
        )

        def public_stage(stage_record, include_live=False):
            if not stage_record:
                return None
            summary = dict(stage_record.get("summary") or {})
            for internal_key in (
                "_utilization_sum_percent", "_active_utilization_sum_percent",
                "_process_cpu_sum_percent", "_process_cpu_samples",
            ):
                summary.pop(internal_key, None)
            payload = {
                "stage_id": stage_record.get("stage_id"),
                "name": stage_record.get("name"),
                "detail": stage_record.get("detail"),
                "status": stage_record.get("status"),
                "launcher_pid": stage_record.get("launcher_pid"),
                "summary": summary,
            }
            if include_live:
                payload["latest_sample"] = dict(stage_record.get("latest_sample") or {})
            return payload

        return {
            "audit_id": audit_id,
            "status": audit.get("status"),
            "execution_mode": audit.get("execution_mode"),
            "analysis_profile": audit.get("analysis_profile"),
            "model_chain_executed": bool(audit.get("model_chain_executed")),
            "started_at": audit.get("started_at"),
            "finished_at": audit.get("finished_at"),
            "report_path": audit.get("report_path"),
            "report_url": f"/api/gpu_audit/{audit_id}",
            "selected_gpu": dict(audit.get("selected_gpu") or {}),
            "active_stage": public_stage(active_record, include_live=True),
            "last_completed_stage": public_stage(completed_record),
            **aggregate,
        }


def _start_gpu_audit(
    job_id: str, case_id: str, cache_key: str, force_model_rerun: bool,
    cuda_info: dict, analysis_profile: str = "full_ensemble",
) -> str:
    safe_job = secure_filename(job_id)[:80] or "direct"
    safe_case = secure_filename(case_id)[:80] or "case"
    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    audit_id = f"{timestamp}_{safe_job}_{uuid.uuid4().hex[:8]}"
    audit_dir = GPU_RUN_DIR / audit_id
    audit_dir.mkdir(parents=True, exist_ok=False)
    audit = {
        "schema_version": 3,
        "audit_id": audit_id,
        "job_id": safe_job,
        "case_id": safe_case,
        "cache_key": cache_key,
        "status": "running",
        "execution_mode": "fresh_gpu" if force_model_rerun else "cache_allowed",
        "analysis_profile": analysis_profile,
        "model_chain_executed": False,
        "gpu_evidence_required": GPU_EVIDENCE_REQUIRED,
        "gpu_evidence_policy": (
            "A stage is verified only when a CUDA PID owned by the launched model "
            "process tree is observed together with utilization or VRAM activity."
        ),
        "telemetry_scope": (
            "Whole-device NVIDIA counters plus model-process CUDA PID ownership"
        ),
        "telemetry_interval_seconds": GPU_TELEMETRY_INTERVAL_SECONDS,
        "selected_gpu": {
            "name": cuda_info.get("name"),
            "uuid": cuda_info.get("uuid"),
            "cuda_device": cuda_info.get("device", "cuda:0"),
            "physical_selector": cuda_info.get(
                "physical_selector", CUDA_VISIBLE_DEVICE.split(",", 1)[0]
            ),
        },
        "cuda_device": cuda_info.get("device", "cuda:0"),
        "cuda_runtime": dict(cuda_info),
        "started_at": datetime.now().astimezone().isoformat(timespec="milliseconds"),
        "finished_at": None,
        "error": None,
        "stages": [],
        "report_path": str(audit_dir / "gpu_run_report.json"),
    }
    with _GPU_AUDIT_LOCK:
        _GPU_RUN_AUDITS[audit_id] = audit
        if job_id:
            _JOB_GPU_AUDITS[safe_job] = audit_id
    _persist_gpu_audit(audit_id)
    debug_log("GPU AUDIT", "GPU çalışma kaydı açıldı: %s", audit["report_path"])
    return audit_id


def _begin_gpu_stage(
    audit_id: str, stage: str, detail: str, command, cwd, launcher_pid: int,
    environment: dict, baseline_gpu: dict, baseline_processes: list,
):
    if not audit_id:
        return None
    with _GPU_AUDIT_LOCK:
        audit = _GPU_RUN_AUDITS.get(audit_id)
        if not audit:
            return None
        index = len(audit["stages"]) + 1
        stage_slug = secure_filename(stage).lower()[:60] or "model"
        stage_id = f"{index:02d}_{stage_slug}"
        audit_dir = Path(audit["report_path"]).parent
        baseline_memory = int((baseline_gpu or {}).get("memory_used_mib", 0))
        stage_record = {
            "stage_id": stage_id,
            "name": stage,
            "detail": detail,
            "status": "running",
            "started_at": datetime.now().astimezone().isoformat(timespec="milliseconds"),
            "finished_at": None,
            "launcher_pid": int(launcher_pid),
            "command": [str(item) for item in command],
            "cwd": str(cwd or BASE_PATH),
            "cuda_environment": {
                key: environment.get(key) if environment else os.environ.get(key)
                for key in (
                    "CUDA_DEVICE_ORDER", "CUDA_VISIBLE_DEVICES", "CUDA_MODULE_LOADING",
                    "PYTORCH_CUDA_ALLOC_CONF", "SHIM_MCCOMPAT_ENABLE_GPU",
                )
            },
            "baseline_gpu": baseline_gpu,
            "baseline_cuda_processes": baseline_processes,
            "stdout_log": str(audit_dir / f"{stage_id}.stdout.log"),
            "telemetry_log": str(audit_dir / f"{stage_id}.telemetry.jsonl"),
            "summary": {
                "sample_count": 0,
                "peak_utilization_percent": 0,
                "peak_memory_used_mib": baseline_memory,
                "peak_memory_delta_mib": 0,
                "peak_power_w": float((baseline_gpu or {}).get("power_w") or 0.0),
                "peak_sm_clock_mhz": int(
                    (baseline_gpu or {}).get("sm_clock_mhz", 0) or 0
                ),
                "utilization_active_samples": 0,
                "first_utilization_s": None,
                "last_utilization_s": None,
                "first_vram_activity_s": None,
                "last_vram_activity_s": None,
                "first_cuda_process_s": None,
                "phase_sample_counts": {},
                "peak_process_cpu_percent": 0.0,
                "peak_process_rss_mib": 0.0,
                "peak_process_count": 0,
                "peak_system_cpu_percent": 0.0,
                "minimum_available_memory_mib": None,
                "process_read_mib": 0.0,
                "process_write_mib": 0.0,
                "system_disk_read_mib": 0.0,
                "system_disk_write_mib": 0.0,
                "_utilization_sum_percent": 0.0,
                "_active_utilization_sum_percent": 0.0,
                "_process_cpu_sum_percent": 0.0,
                "_process_cpu_samples": 0,
                "observed_model_cuda_pids": [],
                "owned_cuda_process_sample_count": 0,
                "owned_cuda_activity_sample_count": 0,
                "gpu_evidence": False,
            },
        }
        audit["stages"].append(stage_record)
        audit["model_chain_executed"] = True
    _persist_gpu_audit(audit_id)
    return stage_id


def _append_gpu_stage_sample(audit_id: str, stage_id: str, sample: dict):
    if not audit_id or not stage_id:
        return
    telemetry_path = None
    persist_snapshot = False
    with _GPU_AUDIT_LOCK:
        audit = _GPU_RUN_AUDITS.get(audit_id)
        if not audit:
            return
        stage_record = next(
            (item for item in audit["stages"] if item["stage_id"] == stage_id), None
        )
        if not stage_record:
            return
        telemetry_path = Path(stage_record["telemetry_log"])
        summary = stage_record["summary"]
        gpu = sample.get("gpu") or {}
        baseline_memory = int((stage_record.get("baseline_gpu") or {}).get("memory_used_mib", 0))
        elapsed_s = float(sample.get("elapsed_s", 0.0) or 0.0)
        utilization = int(gpu.get("utilization_percent", 0) or 0)
        summary["sample_count"] += 1
        summary["_utilization_sum_percent"] += utilization
        summary["peak_utilization_percent"] = max(
            summary["peak_utilization_percent"], utilization
        )
        if utilization > 0:
            summary["utilization_active_samples"] += 1
            summary["_active_utilization_sum_percent"] += utilization
            if summary["first_utilization_s"] is None:
                summary["first_utilization_s"] = round(elapsed_s, 3)
            summary["last_utilization_s"] = round(elapsed_s, 3)
        memory_used = int(gpu.get("memory_used_mib", 0) or 0)
        summary["peak_memory_used_mib"] = max(summary["peak_memory_used_mib"], memory_used)
        summary["peak_memory_delta_mib"] = max(
            summary["peak_memory_delta_mib"], max(0, memory_used - baseline_memory)
        )
        if memory_used - baseline_memory >= 64:
            if summary["first_vram_activity_s"] is None:
                summary["first_vram_activity_s"] = round(elapsed_s, 3)
            summary["last_vram_activity_s"] = round(elapsed_s, 3)
        summary["peak_power_w"] = max(
            summary["peak_power_w"], float(gpu.get("power_w", 0.0) or 0.0)
        )
        summary["peak_sm_clock_mhz"] = max(
            summary["peak_sm_clock_mhz"], int(gpu.get("sm_clock_mhz", 0) or 0)
        )
        observed = set(summary["observed_model_cuda_pids"])
        owned_cuda_processes = list(sample.get("owned_cuda_processes") or [])
        observed.update(int(item["pid"]) for item in owned_cuda_processes)
        if sample.get("launcher_cuda_context"):
            observed.add(int(stage_record["launcher_pid"]))
        if owned_cuda_processes or sample.get("launcher_cuda_context"):
            summary["owned_cuda_process_sample_count"] += 1
            if utilization > 0 or memory_used - baseline_memory >= 64:
                summary["owned_cuda_activity_sample_count"] += 1
        if observed and summary["first_cuda_process_s"] is None:
            summary["first_cuda_process_s"] = round(elapsed_s, 3)
        summary["observed_model_cuda_pids"] = sorted(observed)
        phase = str(sample.get("phase") or "unknown")
        summary["phase_sample_counts"][phase] = (
            int(summary["phase_sample_counts"].get(phase, 0)) + 1
        )
        resources = sample.get("resources") or {}
        if resources:
            process_cpu = float(resources.get("process_cpu_percent", 0.0) or 0.0)
            system_cpu = float(resources.get("system_cpu_percent", 0.0) or 0.0)
            available_memory = resources.get("system_available_memory_mib")
            summary["peak_process_cpu_percent"] = max(
                summary["peak_process_cpu_percent"], process_cpu
            )
            summary["peak_process_rss_mib"] = max(
                summary["peak_process_rss_mib"],
                float(resources.get("process_rss_mib", 0.0) or 0.0),
            )
            summary["peak_process_count"] = max(
                summary["peak_process_count"],
                int(resources.get("process_count", 0) or 0),
            )
            summary["peak_system_cpu_percent"] = max(
                summary["peak_system_cpu_percent"], system_cpu
            )
            if available_memory is not None:
                current_minimum = summary["minimum_available_memory_mib"]
                summary["minimum_available_memory_mib"] = round(
                    min(
                        float(available_memory),
                        float(current_minimum) if current_minimum is not None else float(available_memory),
                    ),
                    1,
                )
            summary["process_read_mib"] += float(
                resources.get("process_read_mib_delta", 0.0) or 0.0
            )
            summary["process_write_mib"] += float(
                resources.get("process_write_mib_delta", 0.0) or 0.0
            )
            summary["system_disk_read_mib"] += float(
                resources.get("system_disk_read_mib_delta", 0.0) or 0.0
            )
            summary["system_disk_write_mib"] += float(
                resources.get("system_disk_write_mib_delta", 0.0) or 0.0
            )
            summary["_process_cpu_sum_percent"] += process_cpu
            summary["_process_cpu_samples"] += 1
        stage_record["latest_sample"] = {
            "sample": sample.get("sample"),
            "sampled_at": sample.get("sampled_at"),
            "elapsed_s": sample.get("elapsed_s"),
            "phase": phase,
            "gpu": dict(gpu),
            "owned_cuda_processes": owned_cuda_processes,
            "process_gpu_verified": bool(
                owned_cuda_processes or sample.get("launcher_cuda_context")
            ),
        }
        persist_snapshot = summary["sample_count"] % 10 == 0
    telemetry_path.parent.mkdir(parents=True, exist_ok=True)
    with open(telemetry_path, "a", encoding="utf-8") as stream:
        stream.write(json.dumps(sample, ensure_ascii=False, default=str) + "\n")
    if persist_snapshot:
        _persist_gpu_audit(audit_id)


def _end_gpu_stage(
    audit_id: str, stage_id: str, source_log: Path, returncode: int,
    elapsed: float, error: str = None, evidence_required: bool = False,
):
    if not audit_id or not stage_id:
        return None
    stdout_destination = None
    with _GPU_AUDIT_LOCK:
        audit = _GPU_RUN_AUDITS.get(audit_id)
        if not audit:
            return None
        stage_record = next(
            (item for item in audit["stages"] if item["stage_id"] == stage_id), None
        )
        if not stage_record:
            return None
        stdout_destination = Path(stage_record["stdout_log"])
    try:
        if source_log.exists():
            shutil.copy2(str(source_log), str(stdout_destination))
    except OSError as exc:
        error = f"{error}; stdout arşivlenemedi: {exc}" if error else str(exc)

    with _GPU_AUDIT_LOCK:
        audit = _GPU_RUN_AUDITS[audit_id]
        stage_record = next(item for item in audit["stages"] if item["stage_id"] == stage_id)
        summary = stage_record["summary"]
        evidence_reasons = []
        owned_process_seen = bool(summary["observed_model_cuda_pids"])
        device_activity_seen = bool(summary["owned_cuda_activity_sample_count"])
        if owned_process_seen:
            evidence_reasons.append("model süreç ağacına ait NVIDIA CUDA PID gözlendi")
        if summary["peak_utilization_percent"] > 0:
            evidence_reasons.append("seçili NVIDIA aygıtında kullanım > 0")
        if summary["peak_memory_delta_mib"] >= 64:
            evidence_reasons.append("seçili NVIDIA VRAM artışı >= 64 MiB")
        if device_activity_seen:
            evidence_reasons.append("CUDA PID sahipliği ve aygıt etkinliği aynı örnekte görüldü")
        summary["gpu_evidence"] = bool(owned_process_seen and device_activity_seen)
        summary["evidence_policy"] = "owned_cuda_pid_and_device_activity"
        summary["evidence_reasons"] = evidence_reasons
        summary["elapsed_s"] = round(float(elapsed), 3)
        summary["returncode"] = int(returncode)
        sample_count = int(summary.get("sample_count", 0))
        active_samples = int(summary.get("utilization_active_samples", 0))
        summary["average_utilization_percent"] = round(
            float(summary.pop("_utilization_sum_percent", 0.0)) /
            max(1, sample_count),
            1,
        )
        summary["average_active_utilization_percent"] = round(
            float(summary.pop("_active_utilization_sum_percent", 0.0)) /
            max(1, active_samples),
            1,
        )
        summary["utilization_active_sample_ratio"] = round(
            active_samples / max(1, sample_count), 3
        )
        process_cpu_samples = int(summary.pop("_process_cpu_samples", 0))
        summary["average_process_cpu_percent"] = round(
            float(summary.pop("_process_cpu_sum_percent", 0.0)) /
            max(1, process_cpu_samples),
            1,
        )
        for key in (
            "process_read_mib", "process_write_mib",
            "system_disk_read_mib", "system_disk_write_mib",
        ):
            summary[key] = round(float(summary.get(key, 0.0)), 3)
        summary["estimated_phase_seconds"] = {
            phase: round(min(float(elapsed), count * GPU_TELEMETRY_INTERVAL_SECONDS), 3)
            for phase, count in summary.get("phase_sample_counts", {}).items()
        }
        first_activity_values = [
            value for value in (
                summary.get("first_utilization_s"),
                summary.get("first_vram_activity_s"),
                summary.get("first_cuda_process_s"),
            ) if value is not None
        ]
        last_activity_values = [
            value for value in (
                summary.get("last_utilization_s"),
                summary.get("last_vram_activity_s"),
            ) if value is not None
        ]
        first_activity = min(first_activity_values) if first_activity_values else None
        last_activity = max(last_activity_values) if last_activity_values else None
        summary["cpu_pre_gpu_s"] = round(
            float(first_activity) if first_activity is not None else float(elapsed), 3
        )
        summary["gpu_resident_window_s"] = round(
            max(0.0, float(last_activity) - float(first_activity))
            if first_activity is not None and last_activity is not None else 0.0,
            3,
        )
        summary["cpu_post_gpu_s"] = round(
            max(0.0, float(elapsed) - float(last_activity))
            if last_activity is not None else 0.0,
            3,
        )
        if returncode == 0 and evidence_required and not summary["gpu_evidence"]:
            error = (
                "Model tamamlandı ancak model süreç ağacına bağlanabilen NVIDIA CUDA "
                "kanıtı alınamadı (süreç CUDA PID'i + aygıt etkinliği birlikte görülmedi)."
            )
        stage_record["status"] = "completed" if returncode == 0 and not error else "failed"
        stage_record["finished_at"] = datetime.now().astimezone().isoformat(timespec="milliseconds")
        stage_record["error"] = error
        result = dict(summary)
    _persist_gpu_audit(audit_id)
    return result


def _finish_gpu_audit(audit_id: str, status: str, error: str = None):
    if not audit_id:
        return None
    changed = False
    with _GPU_AUDIT_LOCK:
        audit = _GPU_RUN_AUDITS.get(audit_id)
        if not audit:
            return None
        if audit.get("status") == "running":
            audit["status"] = status
            audit["error"] = error
            audit["finished_at"] = datetime.now().astimezone().isoformat(timespec="milliseconds")
            audit["summary"] = _aggregate_gpu_stages(audit)
            changed = True
    if changed:
        _persist_gpu_audit(audit_id)
    return _public_gpu_audit(audit_id)


def _attach_gpu_audit(quality: dict, audit_id: str):
    audit = _public_gpu_audit(audit_id)
    if audit:
        quality["gpu_audit"] = audit
        quality["gpu_audit_id"] = audit_id
    return quality


def update_progress(job_id: str, percent: int, stage: str, detail: str = ""):
    if not job_id:
        return
    safe_id = secure_filename(job_id)[:80]
    if not safe_id:
        return
    now = time.time()
    payload = {
        "job_id": safe_id,
        "percent": max(0, min(100, int(percent))),
        "stage": stage,
        "detail": detail,
        "updated_at": now,
    }
    with _JOB_LOCK:
        _PROGRESS[safe_id] = payload
        cutoff = now - 21600
        for store in (_PROGRESS, _JOB_RESULTS):
            stale = [key for key, value in store.items() if value.get("updated_at", now) < cutoff]
            for key in stale:
                store.pop(key, None)
    debug_log("İLERLEME", "%s%% - %s%s", payload["percent"], stage,
              f" ({detail})" if detail else "")


def _latest_model_fraction(log_path: Path):
    """nnU-Net tqdm çıktısındaki en büyük işi bulur (küçük export sayaçlarını yok sayar)."""
    try:
        with open(log_path, "rb") as stream:
            stream.seek(0, os.SEEK_END)
            size = stream.tell()
            stream.seek(max(0, size - 512 * 1024))
            text = stream.read().decode("utf-8", errors="ignore")
        matches = re.findall(r"(\d{1,9})/(\d{1,9})", text)
        candidates = [(int(done), int(total)) for done, total in matches if int(total) >= 20]
        if not candidates:
            return None
        largest_total = max(total for _, total in candidates)
        done, total = [item for item in candidates if item[1] == largest_total][-1]
        return max(0.0, min(1.0, done / max(1, total)))
    except OSError:
        return None


def _model_gpu_prediction_finished(log_path: Path) -> bool:
    """Detect nnU-Net's CPU resampling/export phase after CUDA inference."""
    try:
        with open(log_path, "rb") as stream:
            stream.seek(0, os.SEEK_END)
            stream.seek(max(0, stream.tell() - 64 * 1024))
            tail = stream.read().decode("utf-8", errors="ignore")
        return "GPU prediction completed" in tail
    except OSError:
        return False


class _ModelProcessActivity:
    def __enter__(self):
        _GPU_MODEL_PROCESS_ACTIVE.set()
        return self

    def __exit__(self, _exc_type, _exc, _traceback):
        _GPU_MODEL_PROCESS_ACTIVE.clear()
        _GPU_MODEL_EXPORT_ACTIVE.clear()
        return False


def _run_logged_process(
    command, log_path: Path, timeout_seconds: int, *, job_id: str = "",
    stage: str, detail: str, progress_start: int, progress_end: int,
    cwd=None, env=None, parse_model_progress: bool = False,
    expected_seconds: float = 120.0, audit_id: str = "",
    require_gpu_evidence: bool = True,
) -> tuple[int, float]:
    """Run one model while persisting PID, stdout and NVIDIA telemetry evidence."""
    started = time.monotonic()
    last_reported = progress_start
    process = None
    stage_id = None
    stage_summary = None
    run_error = None
    baseline_gpu = _read_nvidia_gpu_snapshot() if audit_id else None
    baseline_processes = _nvidia_cuda_processes() if audit_id else []
    creationflags = 0
    if os.name == "nt":
        # Model subprocesses also perform CPU resampling and export. Above-normal
        # priority reduces desktop contention without using dangerous realtime/high
        # priority and is inherited by their worker processes.
        creationflags = (
            subprocess.CREATE_NO_WINDOW | subprocess.ABOVE_NORMAL_PRIORITY_CLASS
        )
    try:
        with _ModelProcessActivity(), open(
            log_path, "w", encoding="utf-8", errors="replace"
        ) as log_file:
            process = subprocess.Popen(
                command, cwd=cwd, stdout=log_file, stderr=subprocess.STDOUT,
                text=True, env=env, creationflags=creationflags,
            )
            stage_id = _begin_gpu_stage(
                audit_id, stage, detail, command, cwd, process.pid,
                env or dict(os.environ), baseline_gpu, baseline_processes,
            )
            debug_log(
                "GPU TELEMETRİ", "%s başladı; PID=%s, CUDA=%s, NVIDIA=%s, UUID=%s",
                stage, process.pid,
                (_CUDA_INFO or {}).get("device", "cuda:0"),
                (_CUDA_INFO or {}).get("name", "seçili NVIDIA GPU"),
                (_CUDA_INFO or {}).get("uuid", "bilinmiyor"),
            )
            next_sample_at = 0.0
            next_process_sample_at = 0.0
            sample_index = 0
            cuda_processes = list(baseline_processes)
            known_model_pids = {int(process.pid)}
            previous_resources = None
            previous_resource_elapsed = None
            while process.poll() is None:
                elapsed = time.monotonic() - started
                if elapsed > timeout_seconds:
                    process.kill()
                    process.wait(timeout=10)
                    raise subprocess.TimeoutExpired(command, timeout_seconds)

                fraction = _latest_model_fraction(log_path) if parse_model_progress else None
                export_phase = (
                    parse_model_progress and _model_gpu_prediction_finished(log_path)
                )
                if export_phase:
                    _GPU_MODEL_EXPORT_ACTIVE.set()

                if stage_id and elapsed >= next_sample_at:
                    gpu = _read_nvidia_gpu_snapshot()
                    current_resources = _runtime_resource_snapshot(process.pid)
                    if current_resources:
                        known_model_pids.update(
                            int(pid) for pid in current_resources.get("process_pids", [])
                        )
                    # CUDA contexts persist long enough to capture at 5 s cadence.
                    # On supported NVIDIA drivers both this and the 1 s counters use
                    # in-process NVML; nvidia-smi remains only a compatibility fallback.
                    if elapsed >= next_process_sample_at:
                        cuda_processes = _nvidia_cuda_processes()
                        next_process_sample_at = elapsed + 5.0
                    current_pids = {int(item["pid"]) for item in cuda_processes}
                    owned_cuda_processes = [
                        item for item in cuda_processes
                        if int(item["pid"]) in known_model_pids
                    ]
                    unrelated_cuda_processes = [
                        item for item in cuda_processes
                        if int(item["pid"]) not in known_model_pids
                    ]
                    resource_interval = (
                        elapsed - previous_resource_elapsed
                        if previous_resource_elapsed is not None else 0.0
                    )
                    resources = _runtime_resource_deltas(
                        current_resources, previous_resources, resource_interval
                    )
                    previous_resources = current_resources
                    previous_resource_elapsed = elapsed
                    gpu_payload = gpu or {}
                    utilization = int(gpu_payload.get("utilization_percent", 0) or 0)
                    memory_delta = (
                        int(gpu_payload.get("memory_used_mib", 0) or 0) -
                        int((baseline_gpu or {}).get("memory_used_mib", 0) or 0)
                    )
                    if export_phase:
                        phase = "cpu_export"
                    elif utilization > 0:
                        phase = (
                            "cuda_compute" if owned_cuda_processes
                            else "unrelated_gpu_activity"
                        )
                    elif owned_cuda_processes or memory_delta >= 64:
                        phase = "cuda_context_cpu_or_io"
                    else:
                        phase = "cpu_preprocess_or_model_load"
                    sample_index += 1
                    _append_gpu_stage_sample(audit_id, stage_id, {
                        "sample": sample_index,
                        "sampled_at": datetime.now().astimezone().isoformat(timespec="milliseconds"),
                        "elapsed_s": round(elapsed, 3),
                        "phase": phase,
                        "launcher_pid": int(process.pid),
                        "launcher_running": True,
                        "launcher_cuda_context": int(process.pid) in current_pids,
                        "gpu": gpu,
                        "cuda_processes": cuda_processes,
                        "owned_cuda_processes": owned_cuda_processes,
                        "unrelated_cuda_processes": unrelated_cuda_processes,
                        "resources": resources,
                    })
                    next_sample_at = max(
                        next_sample_at + GPU_TELEMETRY_INTERVAL_SECONDS,
                        elapsed + 0.01,
                    )

                if fraction is None:
                    fraction = min(0.95, elapsed / max(1.0, expected_seconds))
                target = progress_start + int((progress_end - progress_start) * fraction)
                target = min(progress_end - 1, max(last_reported, target))
                if target != last_reported or int(elapsed) % 10 == 0:
                    last_reported = target
                    reported_stage = (
                        "nnU-Net GPU tahmini tamamlandı" if export_phase else stage
                    )
                    reported_detail = (
                        "GPU hesabı bitti; maske CPU ile yeniden örneklenip diske kaydediliyor"
                        if export_phase else detail
                    )
                    update_progress(
                        job_id, target, reported_stage,
                        f"{reported_detail} — geçen süre {int(elapsed // 60):02d}:{int(elapsed % 60):02d}",
                    )
                time.sleep(min(1.0, GPU_TELEMETRY_INTERVAL_SECONDS))
    except Exception as exc:
        run_error = f"{type(exc).__name__}: {exc}"
        if process is not None and process.poll() is None:
            try:
                process.kill()
                process.wait(timeout=10)
            except (OSError, subprocess.SubprocessError):
                pass
        raise
    finally:
        elapsed = time.monotonic() - started
        if process is not None and stage_id:
            stage_summary = _end_gpu_stage(
                audit_id, stage_id, log_path, int(process.returncode or 0), elapsed,
                error=run_error,
                evidence_required=bool(require_gpu_evidence and GPU_EVIDENCE_REQUIRED),
            )
            if stage_summary:
                debug_log(
                    "GPU DOĞRULAMA",
                    "%s bitti; PID=%s, tepe_yük=%%%s, tepe_VRAM=%s MiB, "
                    "ort_yük=%%%s, CPU-ön/GPU/CPU-son=%.1f/%.1f/%.1f sn, "
                    "CUDA_PID=%s, kanıt=%s",
                    stage, process.pid,
                    stage_summary.get("peak_utilization_percent", 0),
                    stage_summary.get("peak_memory_used_mib", 0),
                    stage_summary.get("average_utilization_percent", 0),
                    stage_summary.get("cpu_pre_gpu_s", 0),
                    stage_summary.get("gpu_resident_window_s", 0),
                    stage_summary.get("cpu_post_gpu_s", 0),
                    stage_summary.get("observed_model_cuda_pids", []),
                    stage_summary.get("gpu_evidence", False),
                )
    elapsed = time.monotonic() - started
    if (
        int(process.returncode or 0) == 0 and require_gpu_evidence and
        GPU_EVIDENCE_REQUIRED and stage_summary and not stage_summary.get("gpu_evidence")
    ):
        message = (
            f"{stage}: model sürecine bağlanabilen NVIDIA CUDA kanıtı alınamadı. "
            "Çalışma güvenli biçimde durduruldu; ayrıntılar kalıcı GPU audit kaydında."
        )
        _finish_gpu_audit(audit_id, "failed", message)
        raise RuntimeError(message)
    update_progress(job_id, progress_end, stage, f"{detail} tamamlandı ({elapsed:.1f} sn)")
    return int(process.returncode or 0), elapsed

# ============================================================
# FLASK UYGULAMASI
# ============================================================
class DiskSpoolingRequest(FlaskRequest):
    """Çoklu DICOM yüklemesinde dosyaları 64 KB sonrasında RAM yerine diske taşır."""

    def _get_file_stream(
        self, total_content_length, content_type, filename=None, content_length=None
    ):
        return tempfile.SpooledTemporaryFile(max_size=64 * 1024, mode="rb+")


app = Flask(__name__, template_folder="templates", static_folder="static")
app.request_class = DiskSpoolingRequest
app.secret_key = os.urandom(24)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024
app.config["MAX_FORM_PARTS"] = MAX_UPLOAD_FILES
app.config["MAX_FORM_MEMORY_SIZE"] = 10 * 1024 * 1024
app.config["UPLOAD_FOLDER"]      = str(UPLOAD_DIR)

ALLOWED_EXTENSIONS = {".nii", ".nii.gz", ".dcm", ".dicom", ".zip", ".png", ".jpg", ".jpeg"}


def allowed_file(filename: str) -> bool:
    fn = filename.lower()
    return (fn.endswith(".nii.gz") or
            any(fn.endswith(ext) for ext in [".nii", ".dcm", ".dicom", ".zip", ".png", ".jpg", ".jpeg"]))


def _normalize_dicom_text(*values) -> str:
    """Türkçe karakterleri de kapsayan, seri karşılaştırmasına uygun metin üretir."""
    joined = " ".join(str(value or "") for value in values)
    normalized = unicodedata.normalize("NFKD", joined)
    return " ".join(
        "".join(char for char in normalized if not unicodedata.combining(char))
        .lower().replace("_", " ").replace("-", " ").split()
    )


def _score_dicom_series(headers):
    """Bir DICOM serisinin portal-venöz abdomen CT olma uygunluğunu puanlar."""
    import numpy as np

    dataset = headers[0][1]
    modality = str(getattr(dataset, "Modality", "")).upper()
    if modality != "CT":
        return None

    sop_uid = str(getattr(dataset, "SOPClassUID", ""))
    ct_sop_uids = {
        "1.2.840.10008.5.1.4.1.1.2",    # CT Image Storage
        "1.2.840.10008.5.1.4.1.1.2.1",  # Enhanced CT Image Storage
        "1.2.840.10008.5.1.4.1.1.2.2",  # Legacy Converted Enhanced CT
    }
    if sop_uid and sop_uid not in ct_sop_uids:
        return None

    description = _normalize_dicom_text(getattr(dataset, "SeriesDescription", ""))
    body_part = _normalize_dicom_text(getattr(dataset, "BodyPartExamined", ""))
    image_type = _normalize_dicom_text(*getattr(dataset, "ImageType", []))
    kernel = _normalize_dicom_text(getattr(dataset, "ConvolutionKernel", ""))
    reject_text = f"{description} {image_type}"
    rejected_terms = (
        "topogram", "localizer", "scout", "premonitoring", "monitoring",
        "patient protocol", "dose report", "examination report",
    )
    if any(term in reject_text for term in rejected_terms):
        return None

    score = min(len(headers), 500) * 0.4
    reasons = [f"kesit={len(headers)}"]

    if any(term in body_part for term in ("abdomen", "abdominal", "batin", "karin")):
        score += 2000
        reasons.append("BodyPart=ABDOMEN")
    elif "pelvis" in body_part:
        score += 800
        reasons.append("BodyPart=PELVIS")
    if any(term in body_part for term in ("chest", "thorax", "toraks", "neck", "boyun", "head", "brain")):
        score -= 1600
        reasons.append(f"uygunsuz BodyPart={body_part}")

    abdomen_terms = ("abdomen", "abdominal", "tum batin", "batin", "karin", "pankreas", "pancreas")
    if any(term in description for term in abdomen_terms):
        score += 1200
        reasons.append("abdomen açıklaması")
    non_abdomen_terms = (
        "thorax", "toraks", "chest", "lung", "akciger", "neck", "boyun",
        "head", "brain", "kranium", "servikal",
    )
    if any(term in description for term in non_abdomen_terms):
        score -= 1300
        reasons.append("abdomen dışı açıklama")

    if any(term in description for term in ("venoz", "venous", "portal", "70 sn", "70 sec")):
        score += 250
        reasons.append("portal/venöz faz")
    elif any(term in description for term in ("arterial", "arteryel", "25 sn", "25 sec")):
        score += 80
        reasons.append("arteryel faz")

    if "axial" in image_type:
        score += 50
        reasons.append("aksiyel")
    if str(getattr(dataset, "ContrastBolusAgent", "")).strip():
        score += 50
        reasons.append("IV kontrast")
    if any(term in kernel for term in ("br", "b30", "b31", "b36", "soft", "standard")):
        score += 100
        reasons.append("yumuşak doku kerneli")
    if any(term in kernel for term in ("lung", "bl60", "b70", "sharp")):
        score -= 500
        reasons.append("akciğer/keskin kernel")

    try:
        thickness = float(dataset.SliceThickness)
        if 0.5 <= thickness <= 3.5:
            score += 100
            reasons.append(f"kesit kalınlığı={thickness:g}")
        elif thickness > 5:
            score -= 200
    except Exception:
        pass

    positions = []
    for _, item, _, _ in headers:
        try:
            orientation = np.asarray(item.ImageOrientationPatient, dtype=float)
            position = np.asarray(item.ImagePositionPatient, dtype=float)
            normal = np.cross(orientation[:3], orientation[3:])
            positions.append(float(np.dot(position, normal)))
        except Exception:
            continue
    unique_positions = len({round(value, 3) for value in positions})
    if unique_positions > 10:
        score += 100
        reasons.append(f"uzamsal kesit={unique_positions}")
    elif len(headers) > 2 and unique_positions <= 1:
        score -= 1500
        reasons.append("aynı konum tekrarı")

    return score, reasons


def _build_dicom_nifti_affine(first_header, slice_normal, slice_spacing):
    """Map numpy ``(row, column, slice)`` indices from DICOM LPS to NIfTI RAS."""
    import numpy as np

    try:
        orientation = np.asarray(first_header.ImageOrientationPatient, dtype=float)
        origin_lps = np.asarray(first_header.ImagePositionPatient, dtype=float)
        pixel_spacing = np.asarray(first_header.PixelSpacing, dtype=float)
    except Exception as exc:
        raise ValueError(
            "DICOM hasta geometrisi eksik (ImageOrientationPatient, "
            "ImagePositionPatient veya PixelSpacing)."
        ) from exc

    if orientation.shape != (6,) or origin_lps.shape != (3,) or pixel_spacing.shape != (2,):
        raise ValueError("DICOM hasta geometrisi beklenen boyutta değil.")
    if not np.all(np.isfinite(orientation)) or not np.all(np.isfinite(origin_lps)):
        raise ValueError("DICOM yönelim/konum değerleri geçersiz.")
    if not np.all(np.isfinite(pixel_spacing)) or np.any(pixel_spacing <= 0):
        raise ValueError("DICOM PixelSpacing değerleri geçersiz.")

    # DICOM IOP: ilk üç değer sütun indeksi boyunca, son üç değer satır
    # indeksi boyunca hasta yönünü verir. PixelSpacing ise (satır, sütun)
    # sırasındadır. Hacim dizimiz (satır, sütun, kesit) olduğundan affine
    # eksenlerinin bu sırada kurulması gerekir.
    column_index_direction_lps = orientation[:3]
    row_index_direction_lps = orientation[3:]
    normal_lps = np.asarray(slice_normal, dtype=float)
    normal_norm = float(np.linalg.norm(normal_lps))
    if not np.isfinite(normal_norm) or normal_norm < 1e-6:
        raise ValueError("DICOM kesit normali hesaplanamadı.")
    normal_lps /= normal_norm

    affine_lps = np.eye(4, dtype=float)
    affine_lps[:3, 0] = row_index_direction_lps * float(pixel_spacing[0])
    affine_lps[:3, 1] = column_index_direction_lps * float(pixel_spacing[1])
    affine_lps[:3, 2] = normal_lps * float(slice_spacing)
    affine_lps[:3, 3] = origin_lps

    # DICOM dünya koordinatları LPS, NIfTI ise RAS kullanır.
    lps_to_ras = np.diag([-1.0, -1.0, 1.0, 1.0])
    affine_ras = lps_to_ras @ affine_lps
    if not np.all(np.isfinite(affine_ras)) or abs(np.linalg.det(affine_ras[:3, :3])) < 1e-8:
        raise ValueError("DICOM'dan geçerli NIfTI affine üretilemedi.")
    return affine_ras


def _safe_extract_zip(zip_ref, destination: Path):
    """Extract safely using the configured large userspace I/O buffer."""
    destination_resolved = destination.resolve()
    for member in zip_ref.infolist():
        member_path = (destination / member.filename).resolve()
        try:
            member_path.relative_to(destination_resolved)
        except ValueError as exc:
            raise ValueError(f"Güvensiz ZIP yolu: {member.filename}") from exc
        if member.is_dir():
            member_path.mkdir(parents=True, exist_ok=True)
            continue
        member_path.parent.mkdir(parents=True, exist_ok=True)
        with zip_ref.open(member, "r") as source, open(
            member_path, "wb", buffering=IO_BUFFER_BYTES
        ) as target:
            shutil.copyfileobj(source, target, length=IO_BUFFER_BYTES)


def _dicom_series_content_key(slice_headers, selected_uid: str, progress_callback=None):
    """Hash the exact selected DICOM bytes so reuse can never cross CT inputs."""
    digest = hashlib.sha256()
    digest.update(DICOM_CONVERSION_CACHE_VERSION.encode("ascii"))
    digest.update(str(selected_uid or "").encode("utf-8", errors="replace"))
    total = len(slice_headers)
    for index, (dicom_path, _dataset, rows, columns) in enumerate(slice_headers):
        digest.update(f"{rows}x{columns}:".encode("ascii"))
        stat = dicom_path.stat()
        digest.update(str(stat.st_size).encode("ascii"))
        with open(dicom_path, "rb", buffering=IO_BUFFER_BYTES) as stream:
            for block in iter(lambda: stream.read(IO_BUFFER_BYTES), b""):
                digest.update(block)
        if progress_callback and (index % 50 == 0 or index + 1 == total):
            progress_callback(
                index + 1, total, "Seçilen CT serisi güvenli önbellek için doğrulanıyor"
            )
    return digest.hexdigest()


def convert_image_to_nifti(input_path: Path, output_nii_path: Path, progress_callback=None) -> bool:
    """Görsel kesitleri, hacim boyutunda RAM kopyası oluşturmadan NIfTI'ye dönüştürür."""
    import zipfile
    from PIL import Image
    import numpy as np
    import nibabel as nib

    extract_dir = None
    temp_volume_path = None
    volume = None
    try:
        output_nii_path.parent.mkdir(parents=True, exist_ok=True)
        img_files = []

        if input_path.is_dir():
            img_files = sorted(
                f for f in input_path.rglob("*")
                if f.suffix.lower() in [".png", ".jpg", ".jpeg"] and f.is_file()
            )
        elif input_path.name.lower().endswith(".zip"):
            extract_dir = input_path.parent / f"extracted_img_{uuid.uuid4().hex[:8]}"
            extract_dir.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(input_path, "r") as zip_ref:
                _safe_extract_zip(zip_ref, extract_dir)
            img_files = sorted(
                f for f in extract_dir.rglob("*")
                if f.suffix.lower() in [".png", ".jpg", ".jpeg"] and f.is_file()
            )
        elif input_path.suffix.lower() in [".png", ".jpg", ".jpeg"]:
            img_files = [input_path]

        if not img_files:
            return False

        with Image.open(img_files[0]) as first_image:
            width, height = first_image.size

        temp_volume_path = output_nii_path.parent / f".{uuid.uuid4().hex}.img-volume.dat"
        volume = np.memmap(
            temp_volume_path, dtype=np.float32, mode="w+",
            shape=(height, width, len(img_files)),
        )
        total = len(img_files)
        for index, image_path in enumerate(img_files):
            with Image.open(image_path) as pil_image:
                gray = pil_image.convert("L")
                if gray.size != (width, height):
                    raise ValueError("Görsel kesit boyutları birbiriyle uyuşmuyor.")
                slice_array = np.asarray(gray, dtype=np.float32)
            volume[:, :, index] = (slice_array / 255.0) * 400.0 - 150.0
            if progress_callback and (index % 25 == 0 or index + 1 == total):
                progress_callback(index + 1, total, "Görsel kesitler hacme dönüştürülüyor")

        volume.flush()
        nifti_image = nib.Nifti1Image(volume, np.eye(4))
        nifti_image.set_data_dtype(np.float32)
        nib.save(nifti_image, str(output_nii_path))
        del nifti_image, volume
        volume = None
        temp_volume_path.unlink(missing_ok=True)
        return True

    except Exception as exc:
        debug_log("GÖRSEL DÖNÜŞÜM", "Hata: %s", exc)
        return False
    finally:
        if volume is not None:
            del volume
        if temp_volume_path:
            temp_volume_path.unlink(missing_ok=True)
        if extract_dir:
            shutil.rmtree(extract_dir, ignore_errors=True)
        gc.collect()


def convert_dicom_to_nifti(
    input_path: Path, output_nii_path: Path, progress_callback=None, error_detail=None
) -> bool:
    """
    Çok-serili DICOM klasöründen pankreas için uygun abdomen CT serisini seçer.
    Piksel hacmi float32 disk eşlemeli oluşturulur; modelin giriş değerleri değiştirilmez.
    """
    import zipfile
    import pydicom
    import numpy as np
    import nibabel as nib

    extract_dir = None
    temp_volume_path = None
    volume = None

    def fail(message):
        if error_detail is not None:
            error_detail["error"] = message
        debug_log("DICOM DÖNÜŞÜM", "%s", message)
        return False

    try:
        output_nii_path.parent.mkdir(parents=True, exist_ok=True)
        if input_path.is_dir():
            dcm_files = [
                path for path in input_path.rglob("*")
                if path.suffix.lower() in [".dcm", ".dicom", ""] and path.is_file()
            ]
        elif input_path.name.lower().endswith(".zip"):
            extract_dir = input_path.parent / f"extracted_{uuid.uuid4().hex[:8]}"
            extract_dir.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(input_path, "r") as zip_ref:
                _safe_extract_zip(zip_ref, extract_dir)
            dcm_files = [
                path for path in extract_dir.rglob("*")
                if path.suffix.lower() in [".dcm", ".dicom", ""] and path.is_file()
            ]
            nested_archives = sorted(
                path for path in extract_dir.rglob("*.zip") if path.is_file()
            )
            if not dcm_files and len(nested_archives) == 1:
                nested_dir = nested_archives[0].parent / f"nested_{uuid.uuid4().hex[:8]}"
                nested_dir.mkdir(parents=True, exist_ok=True)
                with zipfile.ZipFile(nested_archives[0], "r") as zip_ref:
                    _safe_extract_zip(zip_ref, nested_dir)
                dcm_files = [
                    path for path in nested_dir.rglob("*")
                    if path.suffix.lower() in [".dcm", ".dicom", ""] and path.is_file()
                ]
            elif not dcm_files and len(nested_archives) > 1:
                if all("maskeli_goruntu" in path.name.lower() for path in nested_archives):
                    return fail(
                        f"ZIP, {len(nested_archives)} ayrı maskeli sonuç arşivi içeriyor; "
                        "bunlar ham CT değildir. Hastanın orijinal DICOM CT serisini yükleyin."
                    )
                return fail(
                    f"ZIP içinde {len(nested_archives)} ayrı ZIP bulundu. "
                    "Tek hastaya ait ham DICOM arşivini ayrı yükleyin."
                )
        elif input_path.name.lower().endswith((".dcm", ".dicom")):
            dcm_files = [input_path]
        else:
            dcm_files = []

        if not dcm_files:
            return fail("ZIP veya klasör içinde DICOM dosyası bulunamadı.")

        series_groups = {}
        readable_count = 0
        modalities = set()
        rgb_count = 0
        header_total = len(dcm_files)
        header_tags = [
            "InstanceNumber", "SliceLocation", "ImagePositionPatient",
            "ImageOrientationPatient", "Rows", "Columns", "PixelSpacing",
            "SeriesInstanceUID", "SeriesNumber", "SeriesDescription",
            "ProtocolName", "BodyPartExamined", "Modality", "ImageType",
            "SliceThickness", "SpacingBetweenSlices", "ConvolutionKernel",
            "ContrastBolusAgent", "SOPClassUID", "SamplesPerPixel",
            "PhotometricInterpretation",
        ]

        def read_header(dicom_path):
            # Piksel verisi burada hic okunmaz. DICOM etiketleri ilk 1 KB icinde
            # olmak zorunda olmadigindan yalniz gereken metadata etiketleri okunur.
            dataset = pydicom.dcmread(
                str(dicom_path), stop_before_pixels=True, force=True,
                defer_size="1 MB", specific_tags=header_tags,
            )
            return dicom_path, dataset

        worker_count = min(DICOM_HEADER_WORKERS, header_total)
        debug_log(
            "DICOM TARAMA", "%s dosya %s paralel baslik iscisiyle taraniyor",
            header_total, worker_count,
        )
        with ThreadPoolExecutor(
            max_workers=worker_count, thread_name_prefix="dicom-header"
        ) as executor:
            futures = [executor.submit(read_header, path) for path in dcm_files]
            completed_count = 0
            for future in as_completed(futures):
                completed_count += 1
                try:
                    dicom_path, dataset = future.result()
                    readable_count += 1
                    modality = str(getattr(dataset, "Modality", "")).upper()
                    if modality:
                        modalities.add(modality)
                    samples_per_pixel = int(getattr(dataset, "SamplesPerPixel", 1))
                    photometric = str(
                        getattr(dataset, "PhotometricInterpretation", "")
                    ).upper()
                    if samples_per_pixel != 1 or photometric.startswith(("RGB", "YBR")):
                        rgb_count += 1
                        continue
                    rows = int(getattr(dataset, "Rows", 0))
                    columns = int(getattr(dataset, "Columns", 0))
                    if rows and columns:
                        series_uid = str(
                            getattr(dataset, "SeriesInstanceUID", "")
                        ).strip()
                        group_key = series_uid or f"NO_UID::{dicom_path.parent}"
                        series_groups.setdefault(group_key, []).append(
                            (dicom_path, dataset, rows, columns)
                        )
                except Exception:
                    continue
                finally:
                    if progress_callback and (
                        completed_count % 50 == 0 or completed_count == header_total
                    ):
                        progress_callback(
                            completed_count, header_total, "DICOM serileri taranıyor"
                        )

        candidates = []
        for series_uid, headers in series_groups.items():
            dimensions = {(item[2], item[3]) for item in headers}
            if len(dimensions) != 1:
                continue
            description = str(getattr(headers[0][1], "SeriesDescription", "")).lower()
            scored = _score_dicom_series(headers)
            if scored is None:
                continue
            score, reasons = scored
            candidates.append((score, len(headers), series_uid, headers, description, reasons))

        if not candidates:
            if readable_count and rgb_count == readable_count:
                return fail(
                    "ZIP yalnızca RGB/maskeli DICOM görüntüleri içeriyor; "
                    "analiz için orijinal gri-seviye CT DICOM serisi gerekli."
                )
            if modalities and "CT" not in modalities:
                return fail(
                    "DICOM görüntü türü " + "/".join(sorted(modalities)) +
                    "; bu pankreas modeli yalnızca karın CT serilerini analiz eder."
                )
            return fail("Uygun ve tutarlı bir karın CT DICOM serisi bulunamadı.")

        for score, count, _, _, candidate_description, reasons in sorted(candidates, reverse=True):
            debug_log(
                "DICOM SERİ SEÇİMİ", "puan=%.1f, kesit=%s, seri='%s', neden=%s",
                score, count, candidate_description, ", ".join(reasons),
            )

        selected_score, _, selected_uid, slice_headers, description, selected_reasons = max(
            candidates, key=lambda item: (item[0], item[1])
        )
        # Toraks/boyun gibi açıkça uygunsuz bir seriyi, sırf eldeki adayların en
        # iyisi olduğu için pankreas modeline göndermek yalancı pozitif üretir.
        if selected_score < 0:
            debug_log(
                "DICOM DÖNÜŞÜM", "En iyi CT serisi abdomen için uygun değil: "
                "puan=%.1f, seri='%s'", selected_score, description,
            )
            return fail("DICOM serisi karın CT analizi için uygun görünmüyor.")
        debug_log(
            "DICOM DÖNÜŞÜM",
            "%s seri içinden '%s' seçildi: %s kesit; neden=%s (UID=%s)",
            len(series_groups), description, len(slice_headers),
            ", ".join(selected_reasons), selected_uid,
        )

        def get_spatial_position(dataset):
            try:
                orientation = np.asarray(dataset.ImageOrientationPatient, dtype=float)
                position = np.asarray(dataset.ImagePositionPatient, dtype=float)
                normal = np.cross(orientation[:3], orientation[3:])
                return float(np.dot(position, normal))
            except Exception:
                return None

        def get_slice_order(dataset):
            spatial_position = get_spatial_position(dataset)
            if spatial_position is not None:
                return 0, spatial_position
            for attribute in ("SliceLocation", "InstanceNumber"):
                try:
                    return 1, float(getattr(dataset, attribute))
                except Exception:
                    continue
            return 2, 0.0

        slice_headers.sort(key=lambda item: (get_slice_order(item[1]), str(item[0])))
        spatial_slice_count = len({
            round(value, 3) for value in (
                get_spatial_position(item[1]) for item in slice_headers
            ) if value is not None
        })
        if len(slice_headers) < 16 or spatial_slice_count < 10:
            debug_log(
                "DICOM DÖNÜŞÜM", "3B abdomen analizi için yetersiz kesit: "
                "dosya=%s, uzamsal=%s", len(slice_headers), spatial_slice_count,
            )
            return fail(
                f"Seçilen CT serisi yalnızca {len(slice_headers)} dosya ve "
                f"{spatial_slice_count} uzamsal kesit içeriyor; güvenilir 3B analiz için "
                "en az 16 dosya ve 10 farklı kesit konumu gerekli."
            )
        rows, columns = slice_headers[0][2], slice_headers[0][3]

        first_header = slice_headers[0][1]
        try:
            pixel_spacing = [float(value) for value in first_header.PixelSpacing]
            row_spacing, column_spacing = pixel_spacing[0], pixel_spacing[1]
        except Exception:
            row_spacing = column_spacing = 1.0
        spatial_positions = [
            value for value in (get_spatial_position(item[1]) for item in slice_headers)
            if value is not None
        ]
        unique_positions = np.asarray(sorted(set(round(value, 6) for value in spatial_positions)))
        if unique_positions.size > 1:
            slice_spacing = float(np.median(np.abs(np.diff(unique_positions))))
        else:
            try:
                slice_spacing = abs(float(getattr(first_header, "SpacingBetweenSlices")))
            except Exception:
                try:
                    slice_spacing = abs(float(first_header.SliceThickness))
                except Exception:
                    slice_spacing = 1.0
        if not np.isfinite(slice_spacing) or slice_spacing <= 0:
            slice_spacing = 1.0

        try:
            first_orientation = np.asarray(first_header.ImageOrientationPatient, dtype=float)
            slice_normal = np.cross(first_orientation[:3], first_orientation[3:])
            nifti_affine = _build_dicom_nifti_affine(
                first_header, slice_normal=slice_normal, slice_spacing=slice_spacing
            )
        except ValueError as exc:
            debug_log("DICOM GEOMETRİ", "Geçersiz hasta geometrisi: %s", exc)
            return fail(str(exc))
        debug_log(
            "DICOM GEOMETRİ", "boyut=%sx%sx%s, spacing=%.4fx%.4fx%.4f mm, affine=%s",
            rows, columns, len(slice_headers), row_spacing, column_spacing, slice_spacing,
            np.array2string(nifti_affine, precision=3, suppress_small=True),
        )
        conversion_cache_key = _dicom_series_content_key(
            slice_headers, selected_uid, progress_callback
        )
        cached_nifti_path = DICOM_CONVERSION_CACHE_DIR / f"{conversion_cache_key}.nii.gz"
        cached_metadata_path = DICOM_CONVERSION_CACHE_DIR / f"{conversion_cache_key}.json"
        if cached_nifti_path.exists():
            try:
                cached_img = nib.load(str(cached_nifti_path))
                cache_valid = (
                    tuple(cached_img.shape) == (rows, columns, len(slice_headers))
                    and np.allclose(
                        cached_img.affine, nifti_affine, atol=1e-5, rtol=1e-7
                    )
                )
                del cached_img
                if cache_valid:
                    _copy_or_link(cached_nifti_path, output_nii_path)
                    if error_detail is not None:
                        error_detail.update({
                            "cache_hit": True,
                            "cache_key": conversion_cache_key,
                            "persistent_nifti_path": str(cached_nifti_path),
                            "selected_slice_count": len(slice_headers),
                        })
                    if progress_callback:
                        progress_callback(
                            1, 1, "Aynı DICOM serisinin doğrulanmış NIfTI hacmi yeniden kullanıldı"
                        )
                    debug_log(
                        "DICOM ÖNBELLEK", "İçerik eşleşti; dönüşüm atlandı: %s",
                        conversion_cache_key,
                    )
                    return True
            except (OSError, ValueError):
                pass

        temp_volume_path = output_nii_path.parent / f".{uuid.uuid4().hex}.dcm-volume.dat"
        volume = np.memmap(
            temp_volume_path, dtype=np.float32, mode="w+",
            shape=(rows, columns, len(slice_headers)),
        )

        total = len(slice_headers)

        for index, (dicom_path, _, _, _) in enumerate(slice_headers):
            dataset = pydicom.dcmread(str(dicom_path), force=True)
            pixel_array = np.asarray(dataset.pixel_array, dtype=np.float32)
            if pixel_array.ndim != 2 or pixel_array.shape != (rows, columns):
                raise ValueError(f"Uyumsuz DICOM kesiti: {dicom_path.name}")
            pixel_array *= float(getattr(dataset, "RescaleSlope", 1.0))
            pixel_array += float(getattr(dataset, "RescaleIntercept", 0.0))
            if not np.all(np.isfinite(pixel_array)):
                raise ValueError(f"Geçersiz HU değeri içeren DICOM kesiti: {dicom_path.name}")
            # Bazı tarayıcılar FOV dışını -8192 gibi aykırı değerlerle doldurur.
            # Eğitim CT'lerinin tanısal HU aralığını koruyup dolgu/metal uçlarını
            # yumuşak-doku normalizasyonunu bozmayacak aralığa sınırla.
            np.clip(pixel_array, -1024.0, 3071.0, out=pixel_array)
            volume[:, :, index] = pixel_array
            del dataset, pixel_array
            if progress_callback and (index % 25 == 0 or index + 1 == total):
                progress_callback(index + 1, total, "Seçilen CT serisi NIfTI'ye dönüştürülüyor")

        volume.flush()
        nifti_image = nib.Nifti1Image(volume, nifti_affine)
        nifti_image.set_data_dtype(np.float32)
        nifti_image.set_qform(nifti_affine, code=1)
        nifti_image.set_sform(nifti_affine, code=1)
        nib.save(nifti_image, str(output_nii_path))
        del nifti_image, volume
        volume = None
        temp_volume_path.unlink(missing_ok=True)
        cache_temp_path = DICOM_CONVERSION_CACHE_DIR / (
            f".{conversion_cache_key}.{uuid.uuid4().hex}.tmp.nii.gz"
        )
        try:
            _copy_or_link(output_nii_path, cache_temp_path)
            os.replace(cache_temp_path, cached_nifti_path)
            _atomic_write_json(cached_metadata_path, {
                "schema_version": 1,
                "cache_version": DICOM_CONVERSION_CACHE_VERSION,
                "cache_key": conversion_cache_key,
                "shape": [rows, columns, len(slice_headers)],
                "slice_count": len(slice_headers),
                "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            })
            if error_detail is not None:
                error_detail.update({
                    "cache_hit": False,
                    "cache_key": conversion_cache_key,
                    "persistent_nifti_path": str(cached_nifti_path),
                    "selected_slice_count": len(slice_headers),
                })
        finally:
            cache_temp_path.unlink(missing_ok=True)
        return True

    except Exception as exc:
        debug_log("DICOM DÖNÜŞÜM", "Hata: %s", exc)
        return fail("DICOM ZIP okunamadı veya içeriği bozuk.")
    finally:
        if volume is not None:
            del volume
        if temp_volume_path:
            temp_volume_path.unlink(missing_ok=True)
        if extract_dir:
            shutil.rmtree(extract_dir, ignore_errors=True)
        gc.collect()


def load_nnunet_env():
    """nnU-Net ortam değişkenlerini yükler."""
    config_path = BASE_PATH / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
        paths = config.get("paths", {})
        for key, path_key in [
            ("nnUNet_raw",          "nnunet_raw"),
            ("nnUNet_preprocessed", "nnunet_preprocessed"),
            ("nnUNet_results",      "nnunet_results"),
        ]:
            if not os.environ.get(key) and path_key in paths:
                os.environ[key] = paths[path_key]


load_nnunet_env()


# ============================================================
# ROUTES
# ============================================================
@app.route("/")
def index():
    """Ana sayfa."""
    return render_template(
        "index.html",
        default_execution_mode=DEFAULT_EXECUTION_MODE,
        default_analysis_profile=DEFAULT_ANALYSIS_PROFILE,
    )


@app.route("/api/pacs/status")
def pacs_status():
    """Expose configuration readiness without starting or controlling PACS I/O.

    The PACS SCP intentionally runs as a separate process so an inbound C-STORE
    burst cannot block Flask requests or the GPU inference worker.  Do not
    expose remote PACS host details in this unauthenticated web endpoint.
    """
    if not PACS_CONFIG_PATH.exists():
        return jsonify({
            "status": "not_configured",
            "configured": False,
            "listener_enabled": False,
            "message": "pacs_config.json bulunamadı; PACS alımı kapalı.",
        })
    try:
        from pacs_bridge import PacsConfigurationError, load_config

        config = load_config(PACS_CONFIG_PATH)
    except (PacsConfigurationError, RuntimeError) as exc:
        return jsonify({
            "status": "invalid_config",
            "configured": False,
            "listener_enabled": False,
            "message": f"PACS yapılandırması geçersiz: {exc}",
        })
    return jsonify({
        "status": "ready" if config.enabled else "configured_disabled",
        "configured": True,
        "listener_enabled": config.enabled,
        "ae_title": config.local_ae_title,
        "listen_port": config.listen_port,
        "accept_ct_only": config.accept_ct_only,
        "max_cache_mb": config.max_cache_mb,
        "message": (
            "PACS dinleyicisi yapılandırıldı; ayrı PACS worker sürecini başlatın."
            if config.enabled else
            "PACS ayarları hazır, ancak dinleyici güvenlik için etkin değil."
        ),
    })


@app.route("/api/progress/<job_id>")
def progress_status(job_id):
    safe_id = secure_filename(job_id)[:80]
    with _JOB_LOCK:
        payload = dict(_PROGRESS.get(safe_id, {
            "job_id": safe_id,
            "percent": 0,
            "stage": "Yükleme bekleniyor",
            "detail": "",
        }))
    payload.pop("updated_at", None)
    with _GPU_AUDIT_LOCK:
        audit_id = _JOB_GPU_AUDITS.get(safe_id)
    public_audit = _public_gpu_audit(audit_id) if audit_id else None
    payload["gpu_audit"] = public_audit
    active_stage = (public_audit or {}).get("active_stage") or {}
    latest_sample = active_stage.get("latest_sample") or {}
    payload["gpu_state"] = {
        "active": bool(active_stage),
        "stage": active_stage.get("name"),
        "detail": active_stage.get("detail"),
        **latest_sample,
    } if active_stage else None
    # Backward-compatible field: only publish a live sample when it is tied to
    # the launched model process tree. The endpoint never substitutes old peaks.
    payload["gpu"] = (
        latest_sample.get("gpu")
        if latest_sample.get("process_gpu_verified") else None
    )
    return jsonify(payload)


@app.route("/api/gpu_status")
def gpu_status():
    """Report model GPU policy without creating an idle CUDA context."""
    model_process_active = _GPU_MODEL_PROCESS_ACTIVE.is_set()
    export_active = _GPU_MODEL_EXPORT_ACTIVE.is_set()
    gpu_compute_active = model_process_active and not export_active
    cuda_info = dict(_CUDA_INFO or {})
    selected_gpu = {
        "name": cuda_info.get("name"),
        "uuid": cuda_info.get("uuid"),
        "physical_selector": cuda_info.get(
            "physical_selector", CUDA_VISIBLE_DEVICE.split(",", 1)[0]
        ),
        "cuda_device": cuda_info.get("device", "cuda:0"),
    }
    return jsonify({
        "active": gpu_compute_active,
        "model_process_active": model_process_active,
        "mode": (
            "model_output_export_cpu" if export_active else
            "model_gpu_active" if gpu_compute_active else "idle_no_gpu"
        ),
        "idle_gpu_load": False,
        "selected_gpu": selected_gpu,
        "cuda_device": selected_gpu["cuda_device"],
        "name": selected_gpu["name"] or "NVIDIA CUDA GPU",
        "torch": cuda_info.get("torch"),
        "cuda_runtime": cuda_info.get("cuda"),
        "allocator_backend": cuda_info.get("allocator", "cudaMallocAsync"),
        "module_loading": cuda_info.get("module_loading", "LAZY"),
        "cpu_fallback_allowed": not CUDA_REQUIRED,
        "telemetry": _nvidia_gpu_snapshot(force=True) if gpu_compute_active else None,
    })


@app.route("/api/gpu_audit/<audit_id>")
def gpu_audit_report(audit_id):
    """Serve the immutable per-run NVIDIA evidence report."""
    safe_id = secure_filename(audit_id)[:180]
    if safe_id != audit_id:
        return jsonify({"error": "Geçersiz GPU audit kimliği"}), 400
    report_path = GPU_RUN_DIR / safe_id / "gpu_run_report.json"
    if not report_path.exists():
        return jsonify({"error": "GPU audit kaydı bulunamadı"}), 404
    return send_file(str(report_path), mimetype="application/json")


@app.route("/result/<job_id>")
def job_result(job_id):
    safe_id = secure_filename(job_id)[:80]
    with _JOB_LOCK:
        stored = _JOB_RESULTS.get(safe_id)
        payload = dict(stored) if stored else None
    if not payload:
        flash("Analiz sonucu bulunamadı veya süresi doldu.", "warning")
        return redirect(url_for("index"))
    return render_template(
        "result.html",
        result=payload["result"],
        case_id=payload["case_id"],
        original_filename=payload["original_filename"],
    )


@app.errorhandler(RequestEntityTooLarge)
def handle_large_upload(_error):
    message = f"Yükleme sunucu sınırını aşıyor. İzin verilen toplam boyut: {MAX_UPLOAD_MB} MB."
    if request.headers.get("X-Requested-With") == "XMLHttpRequest":
        return jsonify({"error": message}), 413
    flash(message, "danger")
    return redirect(url_for("index"))


def _is_ajax_request() -> bool:
    return request.headers.get("X-Requested-With") == "XMLHttpRequest"


def _safe_relative_upload_path(filename: str) -> Path:
    """Klasör yapısını korur ve gönderilen yolun hedef dışına çıkmasını önler."""
    normalized = (filename or "").replace("\\", "/")
    parts = [secure_filename(part) for part in normalized.split("/")]
    parts = [part for part in parts if part and part not in {".", ".."}]
    return Path(*parts) if parts else Path(f"upload_{uuid.uuid4().hex}")


def _conversion_progress(job_id: str):
    def callback(done: int, total: int, detail: str):
        detail_lower = detail.lower()
        if "taran" in detail_lower or "seri" in detail_lower:
            percent = 47 + int(5 * done / max(1, total))
        else:
            percent = 52 + int(8 * done / max(1, total))
        update_progress(job_id, percent, "Kesitler NIfTI hacmine dönüştürülüyor", detail)
    return callback


def _upload_error(
    job_id: str, message: str, status: int = 400, validation_error: bool = False
):
    stage = "Dosya analiz için uygun değil" if validation_error else "İşlem başarısız"
    update_progress(job_id, 0, stage, message)
    if _is_ajax_request():
        return jsonify({"error": message, "validation_error": validation_error}), status
    flash(message, "warning" if validation_error else "danger")
    return redirect(url_for("index"))


def _upload_success(job_id: str, result: dict, case_id: str, original_filename: str):
    update_progress(job_id, 100, "Analiz tamamlandı", "Sonuç sayfası hazırlanıyor")
    if _is_ajax_request():
        safe_id = secure_filename(job_id)[:80]
        with _JOB_LOCK:
            _JOB_RESULTS[safe_id] = {
                "result": result,
                "case_id": case_id,
                "original_filename": original_filename,
                "updated_at": time.time(),
            }
        return jsonify({"redirect_url": url_for("job_result", job_id=safe_id)})
    return render_template(
        "result.html", result=result, case_id=case_id,
        original_filename=original_filename,
    )


def get_clean_basename(filename: str) -> tuple[str, str]:
    """
    Yüklenen dosya adından uzantısız temel adı (base_name) ve tam orijinal adı (original_filename) döndürür.
    Örn: 'PancreasAI_DICOM_case_291e22b5.dcm' -> ('PancreasAI_DICOM_case_291e22b5', 'PancreasAI_DICOM_case_291e22b5.dcm')
    """
    if not filename:
        return "uploaded_file", "uploaded_file"

    original_filename = secure_filename(filename)
    if not original_filename:
        original_filename = "uploaded_file"

    base_name = original_filename
    for ext in [".nii.gz", ".nii", ".dcm", ".dicom", ".zip", ".png", ".jpg", ".jpeg"]:
        if base_name.lower().endswith(ext):
            base_name = base_name[:-len(ext)]
            break

    if not base_name:
        base_name = "uploaded_file"

    return base_name, original_filename


def _request_nifti_path(case_id: str, request_id: str = "") -> Path:
    """Keep simultaneous/repeated uploads from overwriting one another."""
    safe_case = secure_filename(case_id) or "case"
    safe_request = secure_filename(request_id)[:24] or uuid.uuid4().hex[:12]
    return UPLOAD_DIR / f"{safe_case}_{safe_request}_0000.nii.gz"


@app.route("/upload", methods=["POST"])
def upload_file():
    """Büyük CT klasörlerini disk üzerinden işler ve gerçek ilerlemeyi yayınlar."""
    job_id = secure_filename(request.args.get("job_id", ""))[:80] or uuid.uuid4().hex
    execution_mode = str(
        request.form.get("execution_mode", DEFAULT_EXECUTION_MODE)
    ).strip().lower()
    if execution_mode not in {"cache_allowed", "fresh_gpu"}:
        execution_mode = DEFAULT_EXECUTION_MODE
    force_model_rerun = execution_mode != "cache_allowed"
    analysis_profile, _ = _analysis_profile(
        request.form.get("analysis_profile", DEFAULT_ANALYSIS_PROFILE)
    )
    update_progress(job_id, 1, "Yükleme başlatıldı", "Dosya listesi alınıyor")
    files = request.files.getlist("file") or request.files.getlist("files")
    if not files or not files[0] or not files[0].filename:
        return _upload_error(job_id, "Dosya veya klasör seçilmedi!")
    if len(files) > MAX_UPLOAD_FILES:
        return _upload_error(
            job_id,
            f"En fazla {MAX_UPLOAD_FILES} dosya yüklenebilir; {len(files)} dosya seçildi.",
            413,
        )

    debug_log("YÜKLEME", "%s dosya alındı", len(files))
    if len(files) > 1:
        first_name = files[0].filename
        normalized_first = first_name.replace("\\", "/")
        first_parts = normalized_first.split("/")
        fallback_name, _ = get_clean_basename(first_name)
        folder_name = first_parts[0] if len(first_parts) > 1 else fallback_name
        case_id = secure_filename(folder_name) or "uploaded_folder"
        temp_folder = UPLOAD_DIR / f"folder_{case_id}_{uuid.uuid4().hex[:8]}"
        temp_folder.mkdir(parents=True, exist_ok=True)
        try:
            total_files = len(files)
            for index, storage in enumerate(files):
                if storage and storage.filename:
                    save_path = temp_folder / _safe_relative_upload_path(storage.filename)
                    save_path.parent.mkdir(parents=True, exist_ok=True)
                    storage.save(str(save_path))
                if index % 25 == 0 or index + 1 == total_files:
                    percent = 36 + int(10 * (index + 1) / total_files)
                    update_progress(
                        job_id, percent, "Dosyalar diske kaydediliyor",
                        f"{index + 1}/{total_files} dosya",
                    )

            target_nii_path = _request_nifti_path(case_id, job_id)
            callback = _conversion_progress(job_id)
            update_progress(job_id, 47, "DICOM serileri inceleniyor", "Karın serisi seçiliyor")
            conversion_detail = {}
            converted = convert_dicom_to_nifti(
                temp_folder, target_nii_path, callback, conversion_detail
            )
            if not converted:
                converted = convert_image_to_nifti(temp_folder, target_nii_path, callback)
            if not converted:
                return _upload_error(
                    job_id,
                    conversion_detail.get("error") or
                    "Yüklenen klasörde tutarlı bir DICOM CT serisi veya PNG/JPG kesitleri bulunamadı.",
                    200,
                    True,
                )

            analysis_path = Path(
                conversion_detail.get("persistent_nifti_path") or target_nii_path
            )
            result = analyze_ct(
                analysis_path, case_id, original_filename=folder_name, job_id=job_id,
                force_model_rerun=force_model_rerun,
                analysis_profile=analysis_profile,
            )
            if analysis_path != target_nii_path:
                target_nii_path.unlink(missing_ok=True)
            if result.get("error"):
                return _upload_error(job_id, f"Analiz Hatası: {result['error']}", 500)
            return _upload_success(job_id, result, case_id, folder_name)
        finally:
            shutil.rmtree(temp_folder, ignore_errors=True)

    storage = files[0]
    if not allowed_file(storage.filename):
        return _upload_error(
            job_id,
            "Geçersiz dosya formatı! .nii, .nii.gz, .dcm, .dicom, .zip, .png, .jpg veya .jpeg kullanın.",
        )

    case_id, original_filename = get_clean_basename(storage.filename)
    raw_save_path = UPLOAD_DIR / f"raw_{case_id}_{original_filename}"
    update_progress(job_id, 36, "Dosya diske kaydediliyor", original_filename)
    storage.save(str(raw_save_path))
    update_progress(job_id, 46, "Dosya kaydedildi", "Format denetleniyor")

    target_nii_path = _request_nifti_path(case_id, job_id)
    fn_lower = original_filename.lower()
    callback = _conversion_progress(job_id)
    conversion_detail = {}
    converted = True
    if fn_lower.endswith((".dcm", ".dicom")):
        raw_save_path.unlink(missing_ok=True)
        return _upload_error(
            job_id,
            "Tek DICOM kesiti 3B hacim oluşturamaz. 'Doğrudan Klasör Yükle' ile "
            "aynı serideki tüm DICOM kesitlerini seçin veya seriyi ZIP olarak yükleyin.",
            200,
            True,
        )
    if any(fn_lower.endswith(fmt) for fmt in [".png", ".jpg", ".jpeg"]):
        converted = convert_image_to_nifti(raw_save_path, target_nii_path, callback)
    elif fn_lower.endswith(".zip"):
        converted = convert_dicom_to_nifti(
            raw_save_path, target_nii_path, callback, conversion_detail
        )
        if not converted:
            converted = convert_image_to_nifti(raw_save_path, target_nii_path, callback)
    elif fn_lower.endswith(".nii.gz"):
        shutil.copy2(str(raw_save_path), str(target_nii_path))
    else:
        import gzip
        with open(raw_save_path, "rb") as source:
            with gzip.open(target_nii_path, "wb", compresslevel=6) as destination:
                shutil.copyfileobj(source, destination)
    raw_save_path.unlink(missing_ok=True)
    if not converted:
        return _upload_error(
            job_id,
            conversion_detail.get("error") or "Dosya NIfTI hacmine dönüştürülemedi.",
            200,
            True,
        )

    analysis_path = Path(
        conversion_detail.get("persistent_nifti_path") or target_nii_path
    )
    result = analyze_ct(
        analysis_path, case_id, original_filename=original_filename, job_id=job_id,
        force_model_rerun=force_model_rerun,
        analysis_profile=analysis_profile,
    )
    if analysis_path != target_nii_path:
        target_nii_path.unlink(missing_ok=True)
    if result.get("error"):
        return _upload_error(job_id, f"Analiz Hatası: {result['error']}", 500)
    return _upload_success(job_id, result, case_id, original_filename)


@app.route("/api/analyze", methods=["POST"])
def api_analyze():
    """REST API endpoint — JSON yanıt döndürür."""
    if "file" not in request.files:
        return jsonify({"error": "Dosya bulunamadı"}), 400

    file = request.files["file"]
    if not allowed_file(file.filename):
        return jsonify({"error": "Geçersiz dosya formatı (.nii, .nii.gz, .dcm, .dicom, .zip, .png, .jpg, .jpeg desteklenir)"}), 400

    base_name, original_filename = get_clean_basename(file.filename)
    case_id = base_name

    raw_save_path = UPLOAD_DIR / f"raw_{case_id}_{original_filename}"
    file.save(str(raw_save_path))

    target_nii_path = _request_nifti_path(case_id)
    fn_lower = original_filename.lower()

    if any(fn_lower.endswith(fmt) for fmt in [".png", ".jpg", ".jpeg"]):
        converted = convert_image_to_nifti(raw_save_path, target_nii_path)
        if not converted:
            raw_save_path.unlink(missing_ok=True)
            return jsonify({"error": "Görsel dönüştürme başarısız."}), 400
    elif fn_lower.endswith(".dcm") or fn_lower.endswith(".dicom"):
        raw_save_path.unlink(missing_ok=True)
        return jsonify({
            "error": "Tek DICOM kesiti 3B hacim oluşturamaz; aynı serideki tüm kesitleri ZIP olarak gönderin."
        }), 400
    elif fn_lower.endswith(".zip"):
        conversion_detail = {}
        converted = convert_dicom_to_nifti(
            raw_save_path, target_nii_path, error_detail=conversion_detail
        )
        if not converted:
            converted = convert_image_to_nifti(raw_save_path, target_nii_path)
        if not converted:
            raw_save_path.unlink(missing_ok=True)
            return jsonify({
                "error": conversion_detail.get("error") or "ZIP dönüştürme başarısız."
            }), 400
    elif fn_lower.endswith(".nii.gz"):
        shutil.copy2(str(raw_save_path), str(target_nii_path))
    else:
        import gzip
        with open(raw_save_path, "rb") as f_in:
            with gzip.open(target_nii_path, "wb", compresslevel=6) as f_out:
                shutil.copyfileobj(f_in, f_out)

    raw_save_path.unlink(missing_ok=True)

    execution_mode = str(request.form.get(
        "execution_mode", request.args.get("execution_mode", DEFAULT_EXECUTION_MODE)
    )).strip().lower()
    if execution_mode not in {"cache_allowed", "fresh_gpu"}:
        execution_mode = DEFAULT_EXECUTION_MODE
    analysis_profile, _ = _analysis_profile(
        request.form.get(
            "analysis_profile",
            request.args.get("analysis_profile", DEFAULT_ANALYSIS_PROFILE),
        )
    )
    analysis_path = Path(
        conversion_detail.get("persistent_nifti_path") or target_nii_path
    ) if fn_lower.endswith(".zip") else target_nii_path
    result = analyze_ct(
        analysis_path, case_id, original_filename=original_filename,
        force_model_rerun=(execution_mode != "cache_allowed"),
        analysis_profile=analysis_profile,
    )
    if analysis_path != target_nii_path:
        target_nii_path.unlink(missing_ok=True)
    return jsonify(result)


@app.route("/history")
def history():
    """Geçmiş analizleri listele."""
    results_dir = BASE_PATH / "metrics"
    history_items = []

    for json_file in sorted(results_dir.glob("inference_results_*.json"),
                            reverse=True)[:20]:
        try:
            with open(json_file) as f:
                data = json.load(f)
            history_items.append(data)
        except Exception:
            pass

    return render_template("history.html", history=history_items)


@app.route("/static/results/<filename>")
def serve_result(filename):
    return send_from_directory(str(RESULT_DIR), filename)


# ============================================================
# ANALİZ FONKSİYONU
# ============================================================
def _count_mask_labels(mask_data) -> tuple[int, int]:
    """Büyük maskede hacim boyutunda geçici bool dizileri oluşturmadan sayar."""
    import numpy as np
    mask_view = np.squeeze(mask_data)
    if mask_view.ndim < 3:
        return int(np.count_nonzero(mask_view == 1)), int(np.count_nonzero(mask_view == 2))
    pancreas_voxels = 0
    tumor_voxels = 0
    for slice_index in range(mask_view.shape[2]):
        mask_slice = mask_view[:, :, slice_index]
        pancreas_voxels += int(np.count_nonzero(mask_slice == 1))
        tumor_voxels += int(np.count_nonzero(mask_slice == 2))
    return pancreas_voxels, tumor_voxels


def _postprocess_mask_anatomical_continuity(mask_data):
    """Ana pankreas bloğundan eksen boyunca uzak kopuk yalancı bileşenleri temizler."""
    import numpy as np
    try:
        from scipy import ndimage
    except ImportError:
        return mask_data, {"removed_voxels": 0, "removed_components": 0}

    mask = np.asarray(mask_data, dtype=np.uint8)
    organ = mask > 0
    structure = ndimage.generate_binary_structure(3, 2)
    components, component_count = ndimage.label(organ, structure=structure)
    if component_count <= 1:
        return mask, {"removed_voxels": 0, "removed_components": 0}

    counts = np.bincount(components.ravel())
    counts[0] = 0
    main_component = int(np.argmax(counts))
    main_points = np.argwhere(components == main_component)
    main_z_min = int(main_points[:, 2].min())
    main_z_max = int(main_points[:, 2].max())
    # Pankreas/tümörün hemen komşu, kopuk tahminlerini koru; yalnız anatomik
    # kesit bandından belirgin biçimde uzak bileşenleri çıkar.
    z_margin = 5
    removed_voxels = 0
    removed_components = 0
    for component_id in range(1, component_count + 1):
        if component_id == main_component:
            continue
        points = np.argwhere(components == component_id)
        component_z_min = int(points[:, 2].min())
        component_z_max = int(points[:, 2].max())
        is_distant = (
            component_z_max < main_z_min - z_margin or
            component_z_min > main_z_max + z_margin
        )
        if is_distant:
            selector = components == component_id
            removed_voxels += int(np.count_nonzero(selector))
            removed_components += 1
            mask[selector] = 0

    return mask, {
        "removed_voxels": removed_voxels,
        "removed_components": removed_components,
        "main_z_range": [main_z_min, main_z_max],
    }


def analyze_ct(
    ct_path: Path, case_id: str, original_filename: str = None, job_id: str = "",
    force_model_rerun: bool = False, analysis_profile: str = None,
) -> dict:
    """
    CT dosyasını analiz eder:
    1. Gerçek nnU-Net inference (simülasyon tamamen devre dışı)
    2. Segmentasyon maskesinden tümör kararı
    3. Görselleştirme

    Returns:
        Sonuç sözlüğü
    """
    start_time = time.time()
    if not original_filename:
        original_filename = f"{case_id}.dcm"

    result = {
        "case_id":           case_id,
        "filename":          original_filename,
        "original_filename": original_filename,
        "timestamp":         datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "prediction":        None,
        "has_tumor":         None,
        "error":             None,
        "viz_url":           None,
        "elapsed_s":         None,
    }

    try:
        import numpy as np
        import nibabel as nib

        # CT yükle
        ct_img  = nib.load(str(ct_path))
        result["ct_shape"] = list(ct_img.shape)
        del ct_img
        update_progress(job_id, 61, "Model hazırlanıyor", "CT hacmi doğrulandı")

        # Tek GPU üzerinde iki ağır model zinciri CUDA belleğini taşırmasın;
        # eşzamanlı web istekleri burada güvenli biçimde sıraya alınır.
        with _MODEL_RUN_LOCK:
            with _AnalysisProcessPriority():
                mask_data, err_msg, quality = _run_inference(
                    ct_path, case_id, job_id=job_id,
                    force_model_rerun=force_model_rerun,
                    analysis_profile=analysis_profile,
                )

        if mask_data is None:
            result["error"] = err_msg or "Model segmentasyon çıktısı elde edilemedi."
            result["prediction"] = "Model / Inference Hatası"
            result["quality"] = quality
            result["execution_mode"] = quality.get("execution_mode")
            result["analysis_profile"] = quality.get("analysis_profile")
            result["analysis_profile_label"] = quality.get("analysis_profile_label")
            result["gpu_audit"] = quality.get("gpu_audit")
            result["elapsed_s"] = round(time.time() - start_time, 2)
            return result

        result["mode"] = (
            "ensemble_3d_gate"
            if "MONAI DiNTS 3B" in quality.get("models", [])
            else "anatomical_gate_indeterminate"
        )
        result["quality"] = quality

        # Sonuç, model geometrisi ve modeller arası uzamsal destekle sınırlıdır;
        # uzman doğrulaması anlamına gelmez.
        update_progress(job_id, 90, "Maske analiz ediliyor", "3B kalite ölçümleri okunuyor")
        if quality.get("cache_hit") and all(
            key in quality for key in ("pancreas_voxels", "tumor_voxels")
        ):
            pancreas_voxels = int(quality.get("pancreas_voxels", 0))
            tumor_voxels = int(quality.get("tumor_voxels", 0))
        else:
            pancreas_voxels, tumor_voxels = _count_mask_labels(mask_data)
        has_tumor = quality.get("has_tumor")
        agreement_values = []
        for agreement_name in (
            "cross_model_dice", "tumor_cross_model_dice", "pants_cross_model_dice"
        ):
            agreement_value = quality.get(agreement_name)
            if agreement_value is not None:
                agreement_values.append(float(agreement_value))
        boundary_agreement = min(agreement_values) if agreement_values else None
        if boundary_agreement is None:
            boundary_confidence = "hesaplanamadı"
        elif boundary_agreement < 0.30:
            boundary_confidence = "çok düşük"
        elif boundary_agreement < 0.50:
            boundary_confidence = "düşük"
        elif boundary_agreement < 0.70:
            boundary_confidence = "orta"
        else:
            boundary_confidence = "yüksek model uyumu"

        if quality.get("status") == "pancreas_localized":
            prediction_text = (
                "Pankreas 3B anatomik modelle bulundu; tümör kararı için "
                "diğer modeller yeterli anatomik uyum göstermedi"
            )
            confidence_label = (
                "Pankreas konumu doğrulandı; tümör açısından kesin negatif/pozitif karar üretilmedi"
            )
        elif not quality.get("pancreas_verified"):
            prediction_text = "Pankreas geometrik kapıyı geçmedi — karar verilemedi"
            confidence_label = "Model kalite kapısı geçilemedi"
        elif has_tumor is True:
            if boundary_confidence in {"çok düşük", "düşük", "hesaplanamadı"}:
                prediction_text = (
                    f"Tümör adayı saptandı; sınır güveni {boundary_confidence} — "
                    "uzman konturu gerekli"
                )
                confidence_label = (
                    "Pozitif aday var; kesin sınır üretilemedi ve uzman doğrulaması zorunlu"
                )
            else:
                prediction_text = (
                    f"Tümör segmentasyon adayı; sınır güveni {boundary_confidence}"
                )
                confidence_label = "Model uzamsal uyum ölçütleri geçti; uzman onayı yok"
        elif has_tumor is None:
            prediction_text = (
                "Pankreas çevresinde şüpheli alan saptandı — "
                "kesin sınır için uzman incelemesi gerekli"
            )
            confidence_label = (
                "Negatif sonuç verilmedi; şüpheli aday alanı uzman tarafından incelenmeli"
            )
        else:
            prediction_text = "Model zinciri tümör adayı üretmedi — kanseri dışlamaz"
            confidence_label = "Model kalite kapısı geçti; uzman onayı yok"

        result["has_tumor"]       = has_tumor
        result["prediction"]      = prediction_text
        result["tumor_voxels"]    = tumor_voxels
        result["pancreas_voxels"] = pancreas_voxels
        result["tumor_ml"]         = quality.get("tumor_ml", 0.0)
        result["pancreas_ml"]      = quality.get("pancreas_ml", 0.0)
        result["raw_tumor_voxels"] = quality.get("raw_tumor_voxels", 0)
        result["nnunet_raw_tumor_voxels"] = quality.get("nnunet_raw_tumor_voxels", 0)
        result["dints_raw_tumor_voxels"] = quality.get("dints_raw_tumor_voxels", 0)
        result["rejected_tumor_voxels"] = quality.get("rejected_tumor_voxels", 0)
        result["cross_model_dice"] = quality.get("cross_model_dice")
        result["tumor_cross_model_dice"] = quality.get("tumor_cross_model_dice")
        result["tumor_cross_model_overlap_ml"] = quality.get("tumor_cross_model_overlap_ml")
        result["tumor_cross_model_proximity_dice"] = quality.get(
            "tumor_cross_model_proximity_dice"
        )
        result["tumor_cross_model_proximity_overlap_ml"] = quality.get(
            "tumor_cross_model_proximity_overlap_ml"
        )
        result["unverified_tumor_voxels"] = quality.get(
            "unverified_tumor_voxels", 0
        )
        result["unverified_tumor_ml"] = quality.get("unverified_tumor_ml", 0.0)
        result["pancreas_verified"] = bool(quality.get("pancreas_verified"))
        result["quality_status"]   = quality.get("status", "indeterminate")
        result["quality_reason"]   = (
            quality.get("reason") or quality.get("pants_rejection_reason")
        )
        result["confidence"]       = confidence_label
        result["boundary_confidence"] = boundary_confidence
        result["boundary_agreement"] = (
            round(boundary_agreement, 4) if boundary_agreement is not None else None
        )

        ct_img = nib.load(str(ct_path))
        measurements = quality.get("measurements") if quality.get("cache_hit") else None
        if not measurements:
            from segmentation_measurements import measure_segmentation
            measurements = measure_segmentation(mask_data, ct_img.affine)
            _update_inference_cache_measurements(
                str(quality.get("cache_key", "")), measurements
            )
        result["measurements"] = measurements
        result["tumor_dimensions_mm"] = (
            measurements.get("tumor") or {}
        ).get("dimensions_rl_ap_si_mm")
        result["estimated_tumor_location"] = (
            measurements.get("estimated_location") or {}
        ).get("estimated_region")
        quality["measurements"] = measurements

        # Aynı doğrulanmış hacimde PNG/3B HTML'yi tekrar üretme. Manifest anahtarı
        # model/config önbellek anahtarıyla eşleşmiyorsa eski dosyalar kullanılmaz.
        safe_case = secure_filename(case_id) or "case"
        cache_key = str(quality.get("cache_key", ""))
        artifact_manifest = RESULT_DIR / f"{safe_case}_artifacts.json"
        artifact_state = {}
        try:
            artifact_state = json.loads(artifact_manifest.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            artifact_state = {}
        web_viz = RESULT_DIR / f"{safe_case}_seg.png"
        web_3d = RESULT_DIR / f"{safe_case}_3d_interactive.html"
        artifacts_valid = (
            bool(cache_key)
            and artifact_state.get("cache_key") == cache_key
            and artifact_state.get("artifact_version") == ARTIFACT_VERSION
            and web_viz.exists()
            and (
                not result["pancreas_verified"]
                or (
                    artifact_state.get("interactive_3d") == web_3d.name
                    and web_3d.exists()
                )
            )
        )

        if artifacts_valid:
            result["viz_url"] = f"/static/results/{web_viz.name}"
            if result["pancreas_verified"]:
                result["interactive_3d_url"] = f"/static/results/{web_3d.name}"
            update_progress(job_id, 96, "Sonuç görselleri önbellekten alındı", "PNG ve interaktif 3B model hazır")
        else:
            uncertainty_data = None
            uncertainty_path = MASK_DIR / f"{safe_case}_uncertainty.nii.gz"
            if uncertainty_path.exists():
                candidate_uncertainty = np.asanyarray(
                    nib.load(str(uncertainty_path)).dataobj
                ).astype(np.uint8, copy=False)
                if candidate_uncertainty.shape == mask_data.shape:
                    uncertainty_data = candidate_uncertainty
            ct_data = np.asanyarray(ct_img.dataobj)
            update_progress(job_id, 92, "Sonuç görseli hazırlanıyor", "Pankreas içeren 8 kesit seçiliyor")
            viz_path = VIZ_DIR / f"{safe_case}_seg.png"
            _create_web_viz(
                ct_data, mask_data, result, viz_path, affine=ct_img.affine,
                uncertainty_data=uncertainty_data,
            )
            if viz_path.exists():
                shutil.copy2(str(viz_path), str(web_viz))
                result["viz_url"] = f"/static/results/{web_viz.name}"

            if result["pancreas_verified"]:
                update_progress(job_id, 95, "İnteraktif 3B model hazırlanıyor", "Model maskesi hasta koordinatlarına dönüştürülüyor")
                result["interactive_3d_url"] = _create_interactive_3d(
                    mask_data,
                    nib.affines.voxel_sizes(ct_img.affine),
                    safe_case,
                    bool(has_tumor),
                    affine=ct_img.affine,
                    uncertainty_data=uncertainty_data,
                )

            temp_manifest = artifact_manifest.with_suffix(f".{uuid.uuid4().hex}.tmp")
            temp_manifest.write_text(
                json.dumps({
                    "cache_key": cache_key,
                    "artifact_version": ARTIFACT_VERSION,
                    "viz": web_viz.name if web_viz.exists() else None,
                    "interactive_3d": web_3d.name if result.get("interactive_3d_url") else None,
                }, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            os.replace(temp_manifest, artifact_manifest)
            del ct_data

        # Ağır DICOM dışa aktarımı sonuç sayfasını bekletmez; düğme paket hazır
        # olduğunda sayfa üzerinde otomatik etkinleşir.
        update_progress(job_id, 98, "Sonuç sayfası hazırlanıyor", "DICOM paketi arka plana alınıyor")
        result.update(_schedule_dicom_export(
            ct_path,
            MASK_DIR / f"{safe_case}.nii.gz",
            safe_case,
            cache_key,
        ))

        result["timings"] = quality.get("timings", {})
        result["roi_shape"] = quality.get("roi_shape")
        result["roi_reduction_pct"] = quality.get("roi_reduction_pct")
        result["cache_hit"] = bool(quality.get("cache_hit"))
        result["execution_mode"] = quality.get("execution_mode")
        result["analysis_profile"] = quality.get("analysis_profile")
        result["analysis_profile_label"] = quality.get("analysis_profile_label")
        result["models"] = list(quality.get("models", []))
        result["gpu_audit"] = quality.get("gpu_audit")
        del mask_data
        gc.collect()

    except Exception as e:
        result["error"] = str(e)
        result["prediction"] = "Hata"
        debug_log("ANALİZ", "Hata: %s", e)

    result["elapsed_s"] = round(time.time() - start_time, 2)
    result["report_available"] = False
    if not result.get("error") and result.get("prediction"):
        try:
            _persist_patient_report(result, case_id, original_filename)
            result["report_available"] = True
        except Exception as exc:
            debug_log("PDF RAPOR", "Rapor verisi saklanamadı: %s", exc)
    return result


def _patient_report_data_path(case_id: str) -> Path:
    clean_id = _clean_result_case_id(case_id)
    return RESULT_DIR / f"{clean_id}_patient_report.json"


def _persist_patient_report(result: dict, case_id: str, original_filename: str = None):
    """PDF için yalnız raporda gösterilecek JSON-uyumlu sonuç alanlarını saklar."""
    clean_id = _clean_result_case_id(case_id)
    payload = {
        "case_id": clean_id,
        "original_filename": (
            result.get("original_filename") or result.get("filename")
            or original_filename or clean_id
        ),
        "timestamp": result.get("timestamp"),
        "prediction": result.get("prediction"),
        "has_tumor": result.get("has_tumor"),
        "pancreas_verified": bool(result.get("pancreas_verified")),
        "confidence": result.get("confidence"),
        "quality_reason": result.get("quality_reason"),
        "boundary_confidence": result.get("boundary_confidence"),
        "boundary_agreement": result.get("boundary_agreement"),
        "ct_shape": result.get("ct_shape"),
        "elapsed_s": result.get("elapsed_s"),
        "pancreas_voxels": result.get("pancreas_voxels", 0),
        "tumor_voxels": result.get("tumor_voxels", 0),
        "pancreas_ml": result.get("pancreas_ml", 0),
        "tumor_ml": result.get("tumor_ml", 0),
        "tumor_dimensions_mm": result.get("tumor_dimensions_mm"),
        "estimated_tumor_location": result.get("estimated_tumor_location"),
        "cross_model_dice": result.get("cross_model_dice"),
        "tumor_cross_model_dice": result.get("tumor_cross_model_dice"),
        "tumor_cross_model_proximity_dice": result.get(
            "tumor_cross_model_proximity_dice"
        ),
        "tumor_cross_model_proximity_overlap_ml": result.get(
            "tumor_cross_model_proximity_overlap_ml"
        ),
        "unverified_tumor_voxels": result.get("unverified_tumor_voxels", 0),
        "unverified_tumor_ml": result.get("unverified_tumor_ml", 0.0),
        "rejected_tumor_voxels": result.get("rejected_tumor_voxels", 0),
        "nnunet_raw_tumor_voxels": result.get("nnunet_raw_tumor_voxels", 0),
        "dints_raw_tumor_voxels": result.get("dints_raw_tumor_voxels", 0),
        "model": " + ".join(result.get("models") or [
            "nnU-Net 2D", "DiNTS 3B", "TotalSegmentator 3B",
        ]),
        "viz_filename": f"{clean_id}_seg.png",
    }
    report_path = _patient_report_data_path(clean_id)
    temp_path = report_path.with_suffix(f".{uuid.uuid4().hex}.tmp")
    temp_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    os.replace(temp_path, report_path)


def _build_patient_report_pdf(report: dict) -> BytesIO:
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.platypus import (
        Image as ReportImage, KeepTogether, Paragraph, SimpleDocTemplate,
        Spacer, Table, TableStyle,
    )

    regular_font = Path("C:/Windows/Fonts/arial.ttf")
    bold_font = Path("C:/Windows/Fonts/arialbd.ttf")
    font_name = "ArialReport"
    bold_name = "ArialReport-Bold"
    if font_name not in pdfmetrics.getRegisteredFontNames():
        pdfmetrics.registerFont(TTFont(font_name, str(regular_font)))
        pdfmetrics.registerFont(TTFont(bold_name, str(bold_font)))

    buffer = BytesIO()
    document = SimpleDocTemplate(
        buffer, pagesize=A4, rightMargin=16 * mm, leftMargin=16 * mm,
        topMargin=18 * mm, bottomMargin=17 * mm,
        title="PankreasAI Hasta Analiz Raporu",
        author="PankreasAI",
    )
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "ReportTitle", parent=styles["Title"], fontName=bold_name,
        fontSize=19, leading=23, textColor=colors.HexColor("#0f3d63"),
        alignment=TA_CENTER, spaceAfter=4 * mm,
    )
    section_style = ParagraphStyle(
        "ReportSection", parent=styles["Heading2"], fontName=bold_name,
        fontSize=11.5, leading=14, textColor=colors.HexColor("#0f3d63"),
        spaceBefore=3 * mm, spaceAfter=2 * mm,
    )
    body_style = ParagraphStyle(
        "ReportBody", parent=styles["BodyText"], fontName=font_name,
        fontSize=9, leading=12, textColor=colors.HexColor("#263238"),
    )
    small_style = ParagraphStyle(
        "ReportSmall", parent=body_style, fontSize=7.8, leading=10,
        textColor=colors.HexColor("#52606d"),
    )
    warning_style = ParagraphStyle(
        "ReportWarning", parent=body_style, fontName=bold_name,
        fontSize=8.5, leading=11, textColor=colors.HexColor("#8a3b12"),
    )

    def text(value, fallback="Hesaplanamadı"):
        return str(value) if value not in (None, "") else fallback

    def number(value, decimals=3):
        if value is None:
            return "Hesaplanamadı"
        try:
            return f"{float(value):.{decimals}f}"
        except (TypeError, ValueError):
            return str(value)

    def page_footer(canvas, doc):
        canvas.saveState()
        canvas.setFont(font_name, 7.5)
        canvas.setFillColor(colors.HexColor("#64748b"))
        canvas.drawString(16 * mm, 9 * mm, "PankreasAI - Araştırma amaçlı otomatik model raporu")
        canvas.drawRightString(194 * mm, 9 * mm, f"Sayfa {doc.page}")
        canvas.restoreState()

    story = [
        Paragraph("PANKREASAI HASTA ANALİZ RAPORU", title_style),
        Table(
            [[Paragraph("Bu rapor klinik tanı değildir. Bulgular radyolog veya ilgili uzman tarafından doğrulanmalıdır.", warning_style)]],
            colWidths=[178 * mm],
            style=TableStyle([
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#fff4e5")),
                ("BOX", (0, 0), (-1, -1), 0.8, colors.HexColor("#f59e0b")),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                ("TOPPADDING", (0, 0), (-1, -1), 7),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
            ]),
        ),
        Paragraph("Olgu Bilgileri", section_style),
    ]

    case_rows = [
        ["Olgu kimliği", text(report.get("case_id"))],
        ["Dosya adı", text(report.get("original_filename"))],
        ["Analiz zamanı", text(report.get("timestamp"))],
        ["CT hacim boyutu", " x ".join(map(str, report.get("ct_shape") or [])) or "Hesaplanamadı"],
        ["Analiz süresi", f"{text(report.get('elapsed_s'), '0')} saniye"],
    ]
    summary_rows = [
        ["Model sonucu", text(report.get("prediction"))],
        ["Doğrulama durumu", text(report.get("confidence"))],
        ["Pankreas doğrulandı", "Evet" if report.get("pancreas_verified") else "Hayır"],
        ["Sınır güveni", text(report.get("boundary_confidence"))],
        ["En düşük model uyumu", number(report.get("boundary_agreement"), 4)],
    ]
    measurement_rows = [
        ["Pankreas hacmi", f"{number(report.get('pancreas_ml'))} mL"],
        ["Model tümör adayı hacmi", f"{number(report.get('tumor_ml'))} mL"],
        ["Pankreas vokseli", f"{int(report.get('pancreas_voxels') or 0):,}"],
        ["Tümör adayı vokseli", f"{int(report.get('tumor_voxels') or 0):,}"],
        ["Tümör boyutu (R/L x A/P x S/I)",
         " x ".join(map(str, report.get("tumor_dimensions_mm") or [])) + " mm"
         if report.get("tumor_dimensions_mm") else "Hesaplanamadı"],
        ["Kaba anatomik bölge", text(report.get("estimated_tumor_location"), "Belirsiz")],
        ["Pankreas modelleri 3B Dice", number(report.get("cross_model_dice"), 4)],
        ["Tümör modelleri 3B Dice", number(report.get("tumor_cross_model_dice"), 4)],
        ["Tümör modelleri fiziksel yakınlık uyumu",
         number(report.get("tumor_cross_model_proximity_dice"), 4)],
        ["Belirsiz tümör adayı hacmi",
         f"{number(report.get('unverified_tumor_ml'))} mL"],
        ["Belirsiz tümör adayı vokseli",
         f"{int(report.get('unverified_tumor_voxels') or 0):,}"],
        ["Reddedilen ham tümör vokseli", f"{int(report.get('rejected_tumor_voxels') or 0):,}"],
    ]

    def report_table(rows):
        formatted = [[Paragraph(str(label), body_style), Paragraph(str(value), body_style)] for label, value in rows]
        return Table(
            formatted, colWidths=[67 * mm, 111 * mm], repeatRows=0,
            style=TableStyle([
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#eef5f9")),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#b8c7d1")),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]),
        )

    story.extend([
        report_table(case_rows),
        Paragraph("Model Özeti", section_style),
        report_table(summary_rows),
        Paragraph("Segmentasyon Ölçümleri", section_style),
        report_table(measurement_rows),
    ])
    if report.get("quality_reason"):
        story.extend([
            Paragraph("Kalite Kontrol Açıklaması", section_style),
            Table(
                [[Paragraph(text(report.get("quality_reason")), body_style)]],
                colWidths=[178 * mm],
                style=TableStyle([
                    ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
                    ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
                    ("PADDING", (0, 0), (-1, -1), 7),
                ]),
            ),
        ])

    viz_path = RESULT_DIR / str(report.get("viz_filename") or "")
    if viz_path.is_file() and viz_path.parent == RESULT_DIR:
        report_image = ReportImage(str(viz_path))
        max_width, max_height = 178 * mm, 112 * mm
        scale = min(max_width / report_image.imageWidth, max_height / report_image.imageHeight)
        report_image.drawWidth = report_image.imageWidth * scale
        report_image.drawHeight = report_image.imageHeight * scale
        story.append(
            KeepTogether([
                Paragraph("Maskeli Model Görüntüleri", section_style),
                Paragraph("Yeşil: pankreas maskesi | Kırmızı: doğrulanmış tümör adayı | Turuncu kontur: belirsiz model adayı/sınır", small_style),
                Spacer(1, 2 * mm),
                report_image,
            ])
        )

    story.extend([
        Spacer(1, 3 * mm),
        Paragraph("Yöntem ve Sınırlamalar", section_style),
        Paragraph(
            "Sonuç; nnU-Net 2B, DiNTS 3B ve TotalSegmentator 3B modellerinin otomatik çıktılarından üretilmiştir. "
            "Model uyumu gerçek tanısal doğruluk, patoloji sonucu veya uzman onayı anlamına gelmez. "
            "Negatif model çıktısı kanseri dışlamaz; pozitif çıktı kesin kanser tanısı koymaz. "
            "Tedavi veya klinik karar için özgün CT serisi ve resmi radyoloji değerlendirmesi kullanılmalıdır.",
            small_style,
        ),
        Spacer(1, 2 * mm),
        Paragraph("Model: " + text(report.get("model")), small_style),
    ])

    document.build(story, onFirstPage=page_footer, onLaterPages=page_footer)
    buffer.seek(0)
    return buffer


@app.route("/download_report/<case_id>")
def download_report(case_id):
    clean_id = _clean_result_case_id(case_id)
    report_path = _patient_report_data_path(clean_id)
    if not report_path.is_file():
        flash("Hasta raporu bulunamadı; analizi yeniden çalıştırın.", "warning")
        return redirect(url_for("index"))
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
        pdf_buffer = _build_patient_report_pdf(report)
    except Exception as exc:
        debug_log("PDF RAPOR", "Rapor oluşturulamadı: %s", exc)
        flash("Hasta raporu oluşturulamadı.", "danger")
        return redirect(url_for("index"))
    return send_file(
        pdf_buffer, mimetype="application/pdf", as_attachment=True,
        download_name=f"{clean_id}_hasta_analiz_raporu.pdf",
    )


@app.route("/download_masked_png/<case_id>")
def download_masked_png(case_id):
    clean_id = _clean_result_case_id(case_id)
    png_path = RESULT_DIR / f"{clean_id}_seg.png"
    if not png_path.is_file():
        flash("Maskeli PNG görüntüsü bulunamadı.", "warning")
        return redirect(url_for("index"))
    return send_from_directory(
        str(RESULT_DIR), png_path.name, mimetype="image/png", as_attachment=True,
        download_name=f"{clean_id}_maskeli_goruntuler.png",
    )


@app.route("/download_dicom/<case_id>")
def download_dicom(case_id):
    """Maskeli DICOM paketini (.zip) indir (<Orijinal_Dosya_Adi>_maskeli_goruntu.zip)."""
    clean_id = _clean_result_case_id(case_id)

    zip_filename = f"{clean_id}_maskeli_goruntu.zip"
    zip_path = RESULT_DIR / zip_filename

    if not zip_path.exists():
        fallback_path = RESULT_DIR / f"{case_id}.zip"
        if fallback_path.exists():
            zip_path = fallback_path

    if zip_path.exists():
        return send_from_directory(
            str(RESULT_DIR),
            zip_path.name,
            mimetype="application/zip",
            as_attachment=True,
            download_name=f"{clean_id}_maskeli_goruntu.zip"
        )
    else:
        flash("DICOM dosyası bulunamadı!", "danger")
        return redirect(url_for("index"))


@app.route("/api/dicom_status/<case_id>")
def dicom_status(case_id):
    clean_id = _clean_result_case_id(case_id)
    cache_key = str(request.args.get("key", "")).strip().lower()
    if not re.fullmatch(r"[0-9a-f]{64}", cache_key):
        return jsonify({"status": "error", "detail": "Geçersiz DICOM iş anahtarı"}), 400
    if _dicom_package_ready(clean_id, cache_key):
        return jsonify({
            "status": "ready", "percent": 100,
            "detail": "DICOM ZIP paketi hazır",
            "url": f"/download_dicom/{clean_id}",
        })
    with _DICOM_EXPORT_LOCK:
        state = dict(_DICOM_EXPORTS.get(clean_id, {}))
    if state.get("cache_key") != cache_key:
        return jsonify({"status": "missing", "percent": 0, "detail": "DICOM işi bulunamadı"}), 404
    state.pop("cache_key", None)
    return jsonify(state)


def _run_dints_tumor_model(
    ct_path: Path, work_dir: Path, output_dir: Path, log_path: Path,
    job_id: str = "", audit_id: str = "", overlap: float = None,
):
    """Resmî MONAI DiNTS 3B pankreas+tümör modelini bellek-güvenli çalıştırır."""
    if not _parse_bool(TUMOR_MODEL_3D_CONFIG.get("enabled"), True):
        return None, "3B tümör modeli kapalı; düşük kaliteli tek-model sonucu yayınlanmadı.", 0.0

    python_path = Path(TUMOR_MODEL_3D_CONFIG.get(
        "python", ".venv_totalseg/Scripts/python.exe"
    ))
    bundle_dir = Path(TUMOR_MODEL_3D_CONFIG.get(
        "bundle_dir", "data/monai_bundles/pancreas_ct_dints_segmentation"
    ))
    if not python_path.is_absolute():
        python_path = BASE_PATH / python_path
    if not bundle_dir.is_absolute():
        bundle_dir = BASE_PATH / bundle_dir
    config_file = bundle_dir / TUMOR_MODEL_3D_CONFIG.get(
        "config_file", "configs/inference.yaml"
    )
    for required_path, description in (
        (python_path, "3B tümör modeli Python ortamı"),
        (bundle_dir, "3B tümör model paketi"),
        (config_file, "3B tümör modeli inference ayarı"),
        (bundle_dir / "models" / "model.pt", "3B tümör model ağırlığı"),
    ):
        if not required_path.exists():
            return None, f"{description} bulunamadı: {required_path}", 0.0

    work_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    datalist_path = work_dir / "input.json"
    datalist_path.write_text(
        json.dumps({"testing": [{"image": ct_path.name}]}, ensure_ascii=False),
        encoding="utf-8",
    )
    command = [
        str(python_path), "-m", "monai.bundle", "run",
        "--config_file", str(config_file),
        "--dataset_dir", str(ct_path.parent.resolve()),
        "--data_list_file_path", str(datalist_path.resolve()),
        "--output_dir", str(output_dir.resolve()),
        "--dataloader#num_workers", "0",
        "--inferer#sw_batch_size", str(TUMOR_MODEL_3D_CONFIG.get("sw_batch_size", 4)),
    ]
    if overlap is not None:
        command.extend(["--inferer#overlap", str(float(overlap))])
    timeout_seconds = int(TUMOR_MODEL_3D_CONFIG.get("timeout_seconds", 900))
    extra_pythonpath = [
        str(_project_path(path)) for path in TUMOR_MODEL_3D_CONFIG.get("pythonpath", [])
    ]
    environment = _gpu_environment(
        python_paths=extra_pythonpath if extra_pythonpath else None
    )
    _probe_cuda_runtime(python_path, extra_pythonpath)
    process_returncode, elapsed = _run_logged_process(
        command, log_path, timeout_seconds,
        job_id=job_id,
        stage="3B tümör modeli çalışıyor",
        detail="DiNTS anatomik ROI boyunca pankreas ve tümör adaylarını tarıyor",
        progress_start=81,
        progress_end=90,
        cwd=str(bundle_dir),
        env=environment,
        expected_seconds=float(TUMOR_MODEL_3D_CONFIG.get("expected_seconds", 120)),
        audit_id=audit_id,
    )
    if process_returncode != 0:
        details = "3B tümör modeli hata ayrıntısı üretemedi."
        if log_path.exists():
            details = log_path.read_text(encoding="utf-8", errors="replace")[-2500:].strip()
        return None, f"3B tümör modeli başarısız oldu: {details}", elapsed

    mask_files = sorted(output_dir.rglob("*_trans.nii.gz"))
    if len(mask_files) != 1:
        return None, f"3B tümör modeli tek maske üretmedi (bulunan: {len(mask_files)}).", elapsed
    return mask_files[0], None, elapsed


def _run_anatomical_gate(
    ct_path: Path, output_dir: Path, log_path: Path, job_id: str = "",
    audit_id: str = "",
):
    """Produce a pancreas gate, retrying the fast model at full resolution when needed.

    ``--fast`` is useful for routine cases but its coarse spacing can miss a
    small or distorted pancreas.  A failed fast pass must therefore never be
    the sole reason to suppress the rest of the clinical safety pipeline.
    """
    if not _parse_bool(ANATOMICAL_GATE_CONFIG.get("enabled"), True):
        return None, "3B anatomik doğrulama yapılandırmada kapalı; ham maske yayınlanmadı.", 0.0

    executable = Path(ANATOMICAL_GATE_CONFIG.get(
        "executable", ".venv_totalseg/Scripts/TotalSegmentator.exe"
    ))
    if not executable.is_absolute():
        executable = BASE_PATH / executable
    if not executable.exists():
        return None, f"3B anatomik model çalıştırıcısı bulunamadı: {executable}", 0.0

    gate_python = ANATOMICAL_GATE_CONFIG.get("python")
    if gate_python:
        gate_python_path = _project_path(gate_python)
        gate_pythonpath = [
            str(_project_path(path))
            for path in ANATOMICAL_GATE_CONFIG.get("pythonpath", [])
        ]
        _probe_cuda_runtime(gate_python_path, gate_pythonpath)

    home_dir = Path(ANATOMICAL_GATE_CONFIG.get("home_dir", "data/totalsegmentator"))
    if not home_dir.is_absolute():
        home_dir = BASE_PATH / home_dir
    home_dir.mkdir(parents=True, exist_ok=True)
    extra_pythonpath = [
        str(_project_path(path)) for path in ANATOMICAL_GATE_CONFIG.get("pythonpath", [])
    ]
    environment = _gpu_environment(
        python_paths=extra_pythonpath if extra_pythonpath else None
    )
    environment["TOTALSEG_HOME_DIR"] = str(home_dir)
    # TotalSegmentator 2.7+ resolves pretrained models from
    # ``TOTALSEG_WEIGHTS_PATH`` (rather than ``TOTALSEG_HOME_DIR``).  Set
    # both paths so an installed local model is never mistaken for a missing
    # one and the clinical inference path never depends on a network download.
    environment["TOTALSEG_WEIGHTS_PATH"] = str(home_dir / "nnunet" / "results")
    timeout_seconds = int(ANATOMICAL_GATE_CONFIG.get("timeout_seconds", 900))

    def run_pass(name: str, fast: bool):
        pass_output = output_dir / name
        pass_output.mkdir(parents=True, exist_ok=True)
        command = [
            str(executable), "-i", str(ct_path), "-o", str(pass_output),
            "--roi_subset", "pancreas",
            "--device", str(ANATOMICAL_GATE_CONFIG.get("device", "gpu")),
            "--nr_thr_saving", "1",
        ]
        if fast:
            command.append("--fast")
        returncode, elapsed = _run_logged_process(
            command, log_path, timeout_seconds,
            job_id=job_id,
            stage="3B anatomik konumlandırma",
            detail=(
                "Bağımsız hızlı model pankreas konumunu tarıyor"
                if fast else "Hızlı tarama yetersiz kaldı; tam çözünürlüklü pankreas doğrulaması yapılıyor"
            ),
            progress_start=63,
            progress_end=69,
            env=environment,
            expected_seconds=float(ANATOMICAL_GATE_CONFIG.get("expected_seconds", 75)),
            audit_id=audit_id,
        )
        return pass_output / "pancreas.nii.gz", returncode, elapsed

    def is_plausible(mask_path: Path) -> bool:
        if not mask_path.exists():
            return False
        try:
            from segmentation_postprocess import assess_pancreas_gate

            ct_img = nib.load(str(ct_path))
            mask_img = nib.load(str(mask_path))
            if tuple(mask_img.shape) != tuple(ct_img.shape):
                return False
            if not np.allclose(ct_img.affine, mask_img.affine, atol=1e-3, rtol=1e-5):
                return False
            _, assessment = assess_pancreas_gate(
                np.asanyarray(mask_img.dataobj),
                nib.affines.voxel_sizes(ct_img.affine),
                ANATOMICAL_GATE_CONFIG,
            )
            return bool(assessment["gate_plausible"])
        except (OSError, ValueError):
            return False

    use_fast = _parse_bool(ANATOMICAL_GATE_CONFIG.get("fast"), True)
    total_elapsed = 0.0
    fast_path, fast_returncode, fast_elapsed = run_pass("fast", use_fast)
    total_elapsed += fast_elapsed
    if fast_returncode == 0 and is_plausible(fast_path):
        return fast_path, None, total_elapsed

    # The full-resolution retry is intentionally enabled by default.  It is a
    # second anatomical model pass, not a relaxation of volume safeguards.
    retry_full = use_fast and _parse_bool(
        ANATOMICAL_GATE_CONFIG.get("retry_full_resolution_if_invalid"), True
    )
    if retry_full:
        full_path, full_returncode, full_elapsed = run_pass("full_resolution", False)
        total_elapsed += full_elapsed
        if full_returncode == 0 and is_plausible(full_path):
            return full_path, None, total_elapsed
        if full_returncode != 0:
            details = "3B anatomik tam çözünürlük modeli hata ayrıntısı üretemedi."
            if log_path.exists():
                details = log_path.read_text(encoding="utf-8", errors="replace")[-2000:].strip()
            return None, f"3B anatomik doğrulama başarısız oldu: {details}", total_elapsed
        return None, "3B anatomik modeller güvenilir pankreas maskesi üretemedi.", total_elapsed

    if fast_returncode != 0:
        details = "3B anatomik model hata ayrıntısı üretemedi."
        if log_path.exists():
            details = log_path.read_text(encoding="utf-8", errors="replace")[-2000:].strip()
        return None, f"3B anatomik doğrulama başarısız oldu: {details}", total_elapsed
    return None, "3B anatomik model güvenilir pankreas maskesi üretemedi.", total_elapsed


def _copy_or_link(source: Path, destination: Path):
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(str(source), str(destination))


def _inference_cache_key(
    ct_path: Path, checkpoint_path: Path, analysis_profile: str = "full_ensemble"
) -> str:
    """Girdi, ağırlık sürümü ve karar ayarları değişince kendiliğinden geçersizleşir."""
    checkpoint_stat = checkpoint_path.stat()
    signature = {
        "version": INFERENCE_CACHE_VERSION,
        "checkpoint": checkpoint_path.name,
        "checkpoint_size": checkpoint_stat.st_size,
        "checkpoint_mtime_ns": checkpoint_stat.st_mtime_ns,
        "analysis_profile": analysis_profile,
        "inference": INFERENCE_CONFIG,
        "anatomical_gate": ANATOMICAL_GATE_CONFIG,
        "tumor_model_3d": TUMOR_MODEL_3D_CONFIG,
        "pants_refinement": PANTS_REFINEMENT_CONFIG,
    }
    digest = hashlib.sha256(
        json.dumps(signature, sort_keys=True, ensure_ascii=True).encode("utf-8")
    )
    with open(ct_path, "rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _uncertainty_path_for_mask(mask_path: Path) -> Path:
    name = mask_path.name
    stem = name[:-7] if name.lower().endswith(".nii.gz") else mask_path.stem
    return mask_path.with_name(f"{stem}_uncertainty.nii.gz")


def _load_inference_cache(cache_key: str, ct_path: Path, persistent_path: Path):
    import nibabel as nib
    import numpy as np

    cache_mask = INFERENCE_CACHE_DIR / f"{cache_key}.nii.gz"
    cache_json = INFERENCE_CACHE_DIR / f"{cache_key}.json"
    cache_uncertainty = INFERENCE_CACHE_DIR / f"{cache_key}_uncertainty.nii.gz"
    if not cache_mask.exists() or not cache_json.exists():
        return None, None
    try:
        ct_img = nib.load(str(ct_path))
        mask_img = nib.load(str(cache_mask))
        if tuple(mask_img.shape) != tuple(ct_img.shape):
            return None, None
        if not np.allclose(mask_img.affine, ct_img.affine, atol=1e-3, rtol=1e-5):
            return None, None
        quality = json.loads(cache_json.read_text(encoding="utf-8"))
        if quality.get("uncertainty_cached"):
            if not cache_uncertainty.exists():
                return None, None
            uncertainty_img = nib.load(str(cache_uncertainty))
            if tuple(uncertainty_img.shape) != tuple(ct_img.shape):
                return None, None
            if not np.allclose(
                uncertainty_img.affine, ct_img.affine, atol=1e-3, rtol=1e-5
            ):
                return None, None
            uncertainty_path = _uncertainty_path_for_mask(persistent_path)
            shutil.copy2(str(cache_uncertainty), str(uncertainty_path))
            quality["uncertainty_mask_path"] = str(uncertainty_path)
        shutil.copy2(str(cache_mask), str(persistent_path))
        mask_data = np.asanyarray(mask_img.dataobj).astype(np.uint8, copy=False)
        quality["cache_hit"] = True
        quality["cache_key"] = cache_key
        return mask_data, quality
    except (OSError, ValueError, json.JSONDecodeError):
        return None, None


def _save_inference_cache(cache_key: str, persistent_path: Path, quality: dict):
    cache_mask = INFERENCE_CACHE_DIR / f"{cache_key}.nii.gz"
    cache_json = INFERENCE_CACHE_DIR / f"{cache_key}.json"
    cache_uncertainty = INFERENCE_CACHE_DIR / f"{cache_key}_uncertainty.nii.gz"
    shutil.copy2(str(persistent_path), str(cache_mask))
    payload = dict(quality)
    uncertainty_source = Path(str(payload.get("uncertainty_mask_path", "")))
    if uncertainty_source.is_file():
        shutil.copy2(str(uncertainty_source), str(cache_uncertainty))
        payload["uncertainty_cached"] = True
    else:
        cache_uncertainty.unlink(missing_ok=True)
        payload.pop("uncertainty_mask_path", None)
        payload["uncertainty_cached"] = False
    payload["cache_hit"] = False
    payload["cache_key"] = cache_key
    temp_json = cache_json.with_suffix(f".{uuid.uuid4().hex}.tmp")
    temp_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temp_json, cache_json)


def _update_inference_cache_measurements(cache_key: str, measurements: dict):
    """Persist deterministic mask measurements without rewriting the large mask."""
    if not cache_key or not measurements:
        return
    cache_json = INFERENCE_CACHE_DIR / f"{cache_key}.json"
    try:
        payload = json.loads(cache_json.read_text(encoding="utf-8"))
        payload["measurements"] = measurements
        _atomic_write_json(cache_json, payload)
    except (OSError, ValueError, json.JSONDecodeError):
        return


def _prepare_anatomical_roi(source_img, gate_largest, output_path: Path):
    """Axial eksende pankreas çevresini geniş payla kırpar; X/Y kesitlerini değiştirmez."""
    import numpy as np
    import nibabel as nib

    shape = tuple(int(value) for value in source_img.shape)
    full_slices = (slice(None), slice(None), slice(0, shape[2]))
    if not _parse_bool(ANATOMICAL_GATE_CONFIG.get("roi_crop_enabled"), True):
        _copy_or_link(Path(source_img.file_map["image"].filename), output_path)
        return full_slices, shape

    points = np.where(np.asarray(gate_largest) > 0)
    if not points[0].size:
        _copy_or_link(Path(source_img.file_map["image"].filename), output_path)
        return full_slices, shape

    spacing = np.asarray(nib.affines.voxel_sizes(source_img.affine), dtype=float)
    margin_mm = float(ANATOMICAL_GATE_CONFIG.get("roi_margin_mm", 80.0))
    margin_slices = max(1, int(np.ceil(margin_mm / max(spacing[2], 1e-6))))
    start = max(0, int(points[2].min()) - margin_slices)
    stop = min(shape[2], int(points[2].max()) + margin_slices + 1)

    minimum = min(shape[2], int(ANATOMICAL_GATE_CONFIG.get("roi_min_slices", 96)))
    if stop - start < minimum:
        center = (start + stop) // 2
        start = max(0, center - minimum // 2)
        stop = min(shape[2], start + minimum)
        start = max(0, stop - minimum)

    crop_slices = (slice(None), slice(None), slice(start, stop))
    if start == 0 and stop == shape[2]:
        _copy_or_link(Path(source_img.file_map["image"].filename), output_path)
        return crop_slices, shape

    crop_data = np.asanyarray(source_img.dataobj[crop_slices])
    crop_affine = source_img.affine.copy()
    crop_affine[:3, 3] = (
        source_img.affine[:3, :3] @ np.asarray([0.0, 0.0, float(start)])
        + source_img.affine[:3, 3]
    )
    crop_header = source_img.header.copy()
    crop_header.set_data_shape(crop_data.shape)
    crop_img = nib.Nifti1Image(crop_data, crop_affine, crop_header)
    crop_img.set_qform(crop_affine, code=1)
    crop_img.set_sform(crop_affine, code=1)
    nib.save(crop_img, str(output_path))
    return crop_slices, tuple(int(value) for value in crop_data.shape)


def _prepare_dints_roi(source_img, gate_largest, output_path: Path):
    """Crop all axes around the verified pancreas for the isotropic 3D model.

    The shared nnU-Net input deliberately keeps complete axial planes because it
    is a 2D model. DiNTS, however, resamples every input to 1 mm isotropic
    spacing. Passing a whole-body 512x512 plane with a large field of view can
    therefore create a needlessly huge volume and thousands of sliding windows.
    A wide physical margin around the independently verified pancreas preserves
    the clinically relevant neighbourhood while bounding the 3D workload.
    """
    import nibabel as nib
    import numpy as np

    shape = tuple(int(value) for value in source_img.shape)
    full_slices = tuple(slice(0, size) for size in shape)
    if not _parse_bool(TUMOR_MODEL_3D_CONFIG.get("roi_crop_enabled"), True):
        _copy_or_link(Path(source_img.file_map["image"].filename), output_path)
        return full_slices, shape

    points = np.argwhere(np.asarray(gate_largest) > 0)
    if not points.size:
        _copy_or_link(Path(source_img.file_map["image"].filename), output_path)
        return full_slices, shape

    spacing = np.asarray(nib.affines.voxel_sizes(source_img.affine), dtype=float)
    margin_mm = max(0.0, float(TUMOR_MODEL_3D_CONFIG.get("roi_margin_mm", 80.0)))
    margin = np.ceil(margin_mm / np.maximum(spacing, 1e-6)).astype(int)
    start = np.maximum(points.min(axis=0) - margin, 0)
    stop = np.minimum(points.max(axis=0) + margin + 1, np.asarray(shape))
    crop_slices = tuple(slice(int(a), int(b)) for a, b in zip(start, stop))

    if all(part.start == 0 and part.stop == size for part, size in zip(crop_slices, shape)):
        _copy_or_link(Path(source_img.file_map["image"].filename), output_path)
        return crop_slices, shape

    crop_data = np.asanyarray(source_img.dataobj[crop_slices])
    crop_affine = source_img.affine.copy()
    crop_affine[:3, 3] = (source_img.affine @ np.r_[start, 1.0])[:3]
    crop_header = source_img.header.copy()
    crop_header.set_data_shape(crop_data.shape)
    crop_img = nib.Nifti1Image(crop_data, crop_affine, crop_header)
    crop_img.set_qform(crop_affine, code=1)
    crop_img.set_sform(crop_affine, code=1)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(crop_img, str(output_path))
    return crop_slices, tuple(int(value) for value in crop_data.shape)


def _create_interactive_3d(
    mask_data, spacing, case_id: str, has_tumor: bool, affine=None,
    uncertainty_data=None,
):
    """Model maskesini hasta RAS koordinatlarında döndürülebilen yüzeye çevirir."""
    from reconstruct_3d import extract_surface_mesh, create_interactive_html

    safe_case = secure_filename(case_id) or "case"
    pancreas_mesh = extract_surface_mesh(
        mask_data, label=1, spacing=tuple(spacing), smooth_iterations=2, affine=affine
    )
    if pancreas_mesh is None:
        return None
    tumor_mesh = None
    if has_tumor:
        tumor_mesh = extract_surface_mesh(
            mask_data, label=2, spacing=tuple(spacing), smooth_iterations=1, affine=affine
        )
    tumor_core_mesh = None
    tumor_envelope_mesh = None
    if uncertainty_data is not None:
        uncertainty_array = np.asarray(uncertainty_data)
        core_labels = (uncertainty_array == 1).astype(np.uint8)
        envelope_labels = (uncertainty_array > 0).astype(np.uint8)
        tumor_core_mesh = extract_surface_mesh(
            core_labels, label=1, spacing=tuple(spacing), smooth_iterations=1, affine=affine
        )
        tumor_envelope_mesh = extract_surface_mesh(
            envelope_labels, label=1, spacing=tuple(spacing), smooth_iterations=1, affine=affine
        )
    case_dir = RECON_3D_DIR / safe_case
    case_dir.mkdir(parents=True, exist_ok=True)
    html_name = f"{safe_case}_3d_interactive.html"
    stored_html = case_dir / html_name
    if not create_interactive_html(
        pancreas_mesh, tumor_mesh, bool(has_tumor), stored_html,
        coordinate_system="RAS" if affine is not None else "voxel",
        tumor_core_mesh=tumor_core_mesh,
        tumor_envelope_mesh=tumor_envelope_mesh,
    ):
        return None
    web_html = RESULT_DIR / html_name
    shutil.copy2(str(stored_html), str(web_html))
    return f"/static/results/{html_name}"


def _clean_result_case_id(case_id: str) -> str:
    clean_id = secure_filename(case_id) or "case"
    for ext in [".zip", ".dcm", ".dicom", ".nii.gz", ".nii", ".png", ".jpg", ".jpeg"]:
        if clean_id.lower().endswith(ext):
            clean_id = clean_id[:-len(ext)]
            break
    if clean_id.lower().endswith("_maskeli_goruntu"):
        clean_id = clean_id[:-len("_maskeli_goruntu")]
    return clean_id or "case"


def _dicom_manifest_path(clean_id: str) -> Path:
    return RESULT_DIR / f"{clean_id}_maskeli_goruntu.json"


def _dicom_package_ready(clean_id: str, cache_key: str) -> bool:
    zip_path = RESULT_DIR / f"{clean_id}_maskeli_goruntu.zip"
    manifest_path = _dicom_manifest_path(clean_id)
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        return zip_path.exists() and manifest.get("cache_key") == cache_key
    except (OSError, json.JSONDecodeError):
        return False


def _dicom_export_worker(ct_path: Path, mask_path: Path, clean_id: str, cache_key: str):
    temp_zip = RESULT_DIR / f".{clean_id}_{uuid.uuid4().hex}.tmp.zip"
    try:
        import nibabel as nib
        import numpy as np
        from dicom_export import export_full_dicom_package

        with _DICOM_EXPORT_LOCK:
            current = _DICOM_EXPORTS.get(clean_id, {})
            if current.get("cache_key") != cache_key:
                return
            current.update({
                "status": "queued",
                "detail": "Analiz önceliği için DICOM dışa aktarımı kısa süre bekliyor",
            })

        def report(done, total, detail):
            with _DICOM_EXPORT_LOCK:
                current = _DICOM_EXPORTS.get(clean_id, {})
                if current.get("cache_key") == cache_key:
                    current.update({
                        "status": "running",
                        "percent": int(100 * done / max(1, total)),
                        "detail": detail,
                    })

        # Give an immediately submitted analysis first chance to acquire the
        # model lock, then serialize the memory/disk-heavy DICOM export behind
        # inference. This prevents a previous result's ~500 MiB CT load and ZIP
        # writer from starving the next case's CUDA workers through paging.
        time.sleep(10.0)
        with _MODEL_RUN_LOCK:
            with _DICOM_WORK_LOCK:
                with _DICOM_EXPORT_LOCK:
                    current = _DICOM_EXPORTS.get(clean_id, {})
                    if current.get("cache_key") != cache_key:
                        return
                    current.update({
                        "status": "running",
                        "detail": "2D/3D DICOM kesitleri hazırlanıyor",
                    })
                ct_img = nib.load(str(ct_path))
                mask_img = nib.load(str(mask_path))
                ct_data = np.asanyarray(ct_img.dataobj)
                mask_data = np.asanyarray(mask_img.dataobj).astype(np.uint8, copy=False)
                success = export_full_dicom_package(
                    ct_data, mask_data, clean_id, temp_zip, progress_callback=report
                )
        with _DICOM_EXPORT_LOCK:
            current = _DICOM_EXPORTS.get(clean_id, {})
            if current.get("cache_key") != cache_key:
                return
            if not success or not temp_zip.exists():
                current.update({"status": "error", "detail": "DICOM paketi oluşturulamadı"})
                return
            final_zip = RESULT_DIR / f"{clean_id}_maskeli_goruntu.zip"
            os.replace(temp_zip, final_zip)
            manifest_path = _dicom_manifest_path(clean_id)
            temp_manifest = manifest_path.with_suffix(f".{uuid.uuid4().hex}.tmp")
            temp_manifest.write_text(
                json.dumps({"cache_key": cache_key, "created_at": time.time()}),
                encoding="utf-8",
            )
            os.replace(temp_manifest, manifest_path)
            current.update({
                "status": "ready", "percent": 100,
                "detail": "DICOM ZIP paketi hazır",
                "url": f"/download_dicom/{clean_id}",
            })
    except Exception as exc:
        debug_log("DICOM", "Arka plan paketi oluşturulamadı: %s", exc)
        with _DICOM_EXPORT_LOCK:
            current = _DICOM_EXPORTS.get(clean_id, {})
            if current.get("cache_key") == cache_key:
                current.update({"status": "error", "detail": str(exc)})
    finally:
        temp_zip.unlink(missing_ok=True)
        gc.collect()


def _schedule_dicom_export(ct_path: Path, mask_path: Path, case_id: str, cache_key: str):
    clean_id = _clean_result_case_id(case_id)
    status_url = f"/api/dicom_status/{clean_id}?key={cache_key}"
    if _dicom_package_ready(clean_id, cache_key):
        return {
            "dicom_url": f"/download_dicom/{clean_id}",
            "dicom_status_url": status_url,
        }
    with _DICOM_EXPORT_LOCK:
        current = _DICOM_EXPORTS.get(clean_id)
        if not current or current.get("cache_key") != cache_key or current.get("status") == "error":
            _DICOM_EXPORTS[clean_id] = {
                "cache_key": cache_key,
                "status": "queued",
                "percent": 0,
                "detail": "DICOM paketi sıraya alındı",
            }
            worker = threading.Thread(
                target=_dicom_export_worker,
                args=(Path(ct_path), Path(mask_path), clean_id, cache_key),
                daemon=True,
                name=f"dicom-export-{clean_id[:30]}",
            )
            worker.start()
    return {"dicom_status_url": status_url}


def _project_path(value) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else BASE_PATH / path


def _prepare_pants_roi(
    source_img, seed_mask, output_path: Path, margin_mm: float,
    *, return_slices: bool = False,
):
    """Crop all three axes around the verified gland while preserving the affine."""
    import nibabel as nib
    import numpy as np

    points = np.argwhere(np.asarray(seed_mask) > 0)
    if not points.size:
        raise ValueError("PanTS ROI için pankreas/tümör tohumu boş.")
    spacing = nib.affines.voxel_sizes(source_img.affine)
    margin = np.ceil(float(margin_mm) / np.maximum(spacing, 1e-6)).astype(int)
    start = np.maximum(points.min(axis=0) - margin, 0)
    stop = np.minimum(points.max(axis=0) + margin + 1, source_img.shape)
    slices = tuple(slice(int(a), int(b)) for a, b in zip(start, stop))
    data = np.asanyarray(source_img.dataobj[slices])
    affine = source_img.affine.copy()
    affine[:3, 3] = (source_img.affine @ np.r_[start, 1.0])[:3]
    image = nib.Nifti1Image(data, affine, source_img.header.copy())
    image.set_qform(affine, code=1)
    image.set_sform(affine, code=1)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(image, str(output_path))
    shape = tuple(int(x) for x in data.shape)
    return (shape, slices) if return_slices else shape


def _run_pants_refinement(
    source_img, seed_mask, safe_case: str, job_id: str, audit_id: str = "",
    candidate_mask=None,
):
    """Run both calibrated PanTS models and return final mask + uncertainty labels."""
    import nibabel as nib
    import numpy as np
    from nibabel.processing import resample_from_to
    from refine_with_pants import refine_with_consensus, screening_consensus_candidate

    cfg = PANTS_REFINEMENT_CONFIG
    if not _parse_bool(cfg.get("enabled"), False):
        return np.asarray(seed_mask, dtype=np.uint8), None, {"pants_enabled": False}, {}
    base = np.asarray(seed_mask, dtype=np.uint8)
    candidate = (
        np.zeros(base.shape, dtype=bool)
        if candidate_mask is None else np.asarray(candidate_mask, dtype=bool)
    )
    if candidate.shape != base.shape:
        raise ValueError("PanTS belirsiz aday maskesi ile ana maske boyutları uyuşmuyor.")
    screening_enabled = _parse_bool(cfg.get("screening_enabled"), True)
    if not np.any(base == 2) and not np.any(candidate) and not screening_enabled:
        # The explicitly selected fast profile can retain a fast negative path.
        return base.copy(), np.zeros(base.shape, dtype=np.uint8), {
            "pants_enabled": True,
            "pants_skipped": True,
            "pants_skipped_reason": (
                "Dogrulanmis veya anlamli belirsiz tumor adayi yok."
            ),
            "pants_models": [],
            "pants_cross_model_dice": None,
            "pants_expansion_allowed": False,
            "pants_added_tumor_voxels": 0,
            "pants_candidate_arbitration": False,
            "pants_candidate_confirmed": False,
        }, {}
    python_executable = _project_path(cfg.get("python", sys.executable))
    repo_dir = _project_path(cfg.get("repo_dir"))
    predictor = repo_dir / "predict_abdomenatlas.py"
    class_list = _project_path(cfg.get("class_list"))
    checkpoints = {
        "medformer": _project_path(cfg.get("medformer_checkpoint")),
        "rsuper": _project_path(cfg.get("rsuper_checkpoint")),
    }
    required = [python_executable, predictor, class_list, *checkpoints.values()]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("PanTS çalışma dosyaları eksik: " + ", ".join(missing))

    runtime_pythonpath = [str(_project_path(path)) for path in cfg.get("pythonpath", [])]
    _probe_cuda_runtime(python_executable, runtime_pythonpath)
    timeout = int(cfg.get("timeout_seconds", 600))
    expected = float(cfg.get("expected_seconds_per_model", 45))
    timings = {}
    with tempfile.TemporaryDirectory(prefix="pt_") as temp:
        work = Path(temp)
        input_dir = work / "input"
        input_file = input_dir / "scan.nii.gz"
        roi_seed = base.copy()
        roi_seed[candidate] = np.maximum(roi_seed[candidate], 1)
        roi_shape, roi_slices = _prepare_pants_roi(
            source_img, roi_seed, input_file, float(cfg.get("roi_margin_mm", 80.0)),
            return_slices=True,
        )
        roi_target = nib.load(str(input_file))
        extra_pythonpath = runtime_pythonpath
        # Web sürecinin scripts/inference.py modülü, dış deponun inference/
        # paketini gölgelememeli; PanTS alt sürecine yalnız bağımlılık yollarını ver.
        env = _gpu_environment(python_paths=extra_pythonpath)

        outputs = {}
        for index, (model_name, checkpoint) in enumerate(checkpoints.items()):
            save_root = work / model_name
            log_path = work / f"{model_name}.log"
            command = [
                str(python_executable), str(predictor),
                "--load", str(checkpoint),
                "--img_path", str(input_dir),
                "--class_list", str(class_list),
                "--save_path", str(save_root),
                "--organ_mask_on_lesion", "--save_probabilities_lesions",
                "--save_pancreas_lesion_only", "--gpu", "0", "--overwrite",
            ]
            returncode, elapsed = _run_logged_process(
                command, log_path, timeout, job_id=job_id,
                stage=f"PanTS {model_name} çalışıyor",
                detail=f"3B ROI {roi_shape}", progress_start=89, progress_end=90,
                cwd=repo_dir, env=env, expected_seconds=expected,
                audit_id=audit_id,
            )
            timings[f"pants_{model_name}_s"] = round(elapsed, 3)
            if returncode != 0:
                details = log_path.read_text(encoding="utf-8", errors="replace")[-2000:]
                raise RuntimeError(f"PanTS {model_name} başarısız: {details}")
            model_folder = checkpoint.parent.name
            case_root = save_root / "abdomenatlas" / model_folder / "scan"
            raw_path = case_root / "predictions_raw" / "pancreatic_lesion.nii.gz"
            pancreas_path = case_root / "predictions" / "pancreas.nii.gz"
            if not raw_path.exists() or not pancreas_path.exists():
                raise FileNotFoundError(f"PanTS {model_name} çıktısı bulunamadı: {case_root}")
            raw_img = nib.load(str(raw_path))
            pancreas_img = nib.load(str(pancreas_path))
            outputs[model_name] = {
                "prob": np.clip(
                    resample_from_to(
                        raw_img, roi_target, order=1
                    ).get_fdata(dtype=np.float32),
                    0.0, 1.0,
                ),
                "pancreas": resample_from_to(
                    pancreas_img, roi_target, order=0
                ).get_fdata() > 0.5,
            }

    med_t = float(cfg.get("medformer_threshold", 0.45))
    rs_t = float(cfg.get("rsuper_threshold", 0.45))
    core_t = float(cfg.get("core_threshold", 0.60))
    envelope_t = float(cfg.get("envelope_threshold", 0.30))
    med_prob, rs_prob = outputs["medformer"]["prob"], outputs["rsuper"]["prob"]
    # All seed voxels and all PanTS predictions are inside this exact physical
    # ROI. Running connected components and distance transforms on the full
    # 512x512x497 zero-padded volume produced the same mask but allocated GiB
    # of temporary label arrays and was repeated once in a dead legacy block.
    # Compute the single authoritative consensus in ROI coordinates, then paste
    # it back without changing thresholds, spacing, interpolation or labels.
    base_roi = np.asarray(base[roi_slices], dtype=np.uint8)
    candidate_roi = np.asarray(candidate[roi_slices], dtype=bool)
    screening_metrics = {"proximity_dice": 0.0, "voxels": 0}
    if not np.any(base_roi == 2) and not np.any(candidate_roi):
        # A full-ensemble run must not turn an empty nnU-Net/DiNTS field into
        # a silent negative result.  PanTS supplies a review-only seed when
        # both of its independent lesion maps are physically close to the
        # anatomically localised pancreas.
        candidate_roi, screening_metrics = screening_consensus_candidate(
            med_prob,
            rs_prob,
            base_roi == 1,
            nib.affines.voxel_sizes(source_img.affine),
            medformer_threshold=med_t,
            rsuper_threshold=rs_t,
            support_radius_mm=float(cfg.get("screening_support_radius_mm", 10.0)),
            match_radius_mm=float(cfg.get("screening_match_radius_mm", 5.0)),
        )
    refined_roi, uncertainty_roi, consensus = refine_with_consensus(
        base_roi,
        med_prob,
        rs_prob,
        outputs["medformer"]["pancreas"],
        outputs["rsuper"]["pancreas"],
        nib.affines.voxel_sizes(source_img.affine),
        medformer_threshold=med_t,
        rsuper_threshold=rs_t,
        core_threshold=core_t,
        envelope_threshold=envelope_t,
        min_cross_model_dice=float(cfg.get("min_cross_model_dice", 0.50)),
        max_expansion_mm=float(cfg.get("max_expansion_mm", 5.0)),
        candidate_seed=candidate_roi,
    )
    refined = base.copy()
    refined[roi_slices] = refined_roi
    uncertainty = np.zeros(base.shape, dtype=np.uint8)
    uncertainty[roi_slices] = uncertainty_roi
    primary = refined == 2
    core = uncertainty == 1
    envelope = uncertainty > 0
    voxel_ml = abs(float(np.linalg.det(source_img.affine[:3, :3]))) / 1000.0
    quality = {
        "pants_enabled": True,
        "pants_models": ["MedFormer PanTS", "R-Super PanTS Merlin"],
        "pants_thresholds": {
            "medformer": med_t, "rsuper": rs_t,
            "core": core_t, "envelope": envelope_t,
        },
        "pants_cross_model_dice": round(consensus["cross_model_dice"], 4),
        "pants_expansion_allowed": consensus["expansion_allowed"],
        "pants_rejection_reason": consensus["rejection_reason"],
        "pants_added_tumor_voxels": consensus["added_tumor_voxels"],
        "pants_candidate_arbitration": consensus["candidate_arbitration"],
        "pants_candidate_confirmed": consensus["candidate_confirmed"],
        "pants_candidate_seed_voxels": consensus["candidate_seed_voxels"],
        "pants_screening_candidate_voxels": screening_metrics["voxels"],
        "pants_screening_proximity_dice": round(
            float(screening_metrics["proximity_dice"]), 4
        ),
        "tumor_core_ml": round(float(core.sum()) * voxel_ml, 3),
        "tumor_primary_ml": round(float(primary.sum()) * voxel_ml, 3),
        "tumor_envelope_ml": round(float(envelope.sum()) * voxel_ml, 3),
        "pants_roi_shape": list(roi_shape),
    }
    return refined, uncertainty, quality, timings


def _run_inference(
    ct_path: Path, case_id: str, job_id: str = "", checkpoint_name: str = None,
    force_model_rerun: bool = False, analysis_profile: str = None,
) -> tuple["Optional[np.ndarray]", "Optional[str]", dict]:
    """Anatomik konumlandırma + ROI içinde nnU-Net/DiNTS + güvenli füzyon."""
    import numpy as np
    import nibabel as nib

    profile_name, profile = _analysis_profile(analysis_profile)

    cuda_info = _require_cuda()
    update_progress(
        job_id, 62, "NVIDIA GPU dogrulandi",
        f"{cuda_info.get('name', 'CUDA')} ({cuda_info.get('device', 'cuda:0')})",
    )

    results_dir = Path(os.environ.get(
        "nnUNet_results", str(BASE_PATH / "data" / "nnunet_results")
    ))
    fold_dir = (
        results_dir / "Dataset007_Pancreas" /
        "nnUNetTrainer__nnUNetPlans__2d" / "fold_0"
    )
    selected_checkpoint = checkpoint_name or MODEL_CHECKPOINT
    if selected_checkpoint not in {"checkpoint_final.pth", "checkpoint_best.pth"}:
        return None, f"Geçersiz model checkpoint adı: {selected_checkpoint}", {}
    checkpoint_path = fold_dir / selected_checkpoint
    if not checkpoint_path.exists():
        return None, f"Eğitilmiş nnU-Net checkpoint bulunamadı: {checkpoint_path}", {}

    safe_case = secure_filename(case_id) or "case"
    persistent_path = MASK_DIR / f"{safe_case}.nii.gz"
    cache_started = time.monotonic()
    cache_key = _inference_cache_key(ct_path, checkpoint_path, profile_name)
    audit_id = _start_gpu_audit(
        job_id, safe_case, cache_key, force_model_rerun, cuda_info,
        analysis_profile=profile_name,
    )
    if force_model_rerun:
        update_progress(
            job_id, 62, "NVIDIA model zinciri yeniden çalıştırılıyor",
            f"Önbellek atlandı; {profile.get('label')} seçili NVIDIA GPU'da yeniden çalışacak",
        )
    else:
        update_progress(
            job_id, 62, "Önbellek kontrol ediliyor",
            "Aynı CT hacminin önceki güvenli sonucu aranıyor",
        )
        cached_mask, cached_quality = _load_inference_cache(
            cache_key, ct_path, persistent_path
        )
        if cached_mask is not None:
            cached_quality.setdefault("timings", {})["cache_lookup_s"] = round(
                time.monotonic() - cache_started, 3
            )
            cached_quality["execution_mode"] = "cache_hit_models_skipped"
            cached_quality["analysis_profile"] = profile_name
            cached_quality["analysis_profile_label"] = profile.get("label")
            _finish_gpu_audit(audit_id, "cache_hit")
            _attach_gpu_audit(cached_quality, audit_id)
            update_progress(
                job_id, 90, "Önbellek sonucu — GPU modeli çalıştırılmadı",
                "Ağır modeller bu istekte tekrar çalıştırılmadı; anlık NVIDIA kullanımının %0 olması normal",
            )
            return cached_mask, None, cached_quality

    with _GPU_AUDIT_LOCK:
        if audit_id in _GPU_RUN_AUDITS:
            _GPU_RUN_AUDITS[audit_id]["execution_mode"] = (
                "fresh_gpu_forced" if force_model_rerun else "fresh_gpu_cache_miss"
            )
    _persist_gpu_audit(audit_id)

    run_id = uuid.uuid4().hex[:8]
    tmp_input = UPLOAD_DIR / f"tmp_input_{safe_case}_{run_id}"
    model_input = UPLOAD_DIR / f"tmp_model_input_{safe_case}_{run_id}"
    tmp_output = UPLOAD_DIR / f"tmp_output_{safe_case}_{run_id}"
    log_path = UPLOAD_DIR / f"tmp_model_{safe_case}_{run_id}.log"
    dints_work = UPLOAD_DIR / f"tmp_dints_work_{safe_case}_{run_id}"
    dints_input = UPLOAD_DIR / f"tmp_dints_input_{safe_case}_{run_id}"
    dints_output = UPLOAD_DIR / f"tmp_dints_output_{safe_case}_{run_id}"
    dints_log_path = UPLOAD_DIR / f"tmp_dints_{safe_case}_{run_id}.log"
    gate_output = UPLOAD_DIR / f"tmp_gate_{safe_case}_{run_id}"
    gate_log_path = UPLOAD_DIR / f"tmp_gate_{safe_case}_{run_id}.log"
    tmp_input.mkdir(parents=True, exist_ok=False)
    model_input.mkdir(parents=True, exist_ok=False)
    tmp_output.mkdir(parents=True, exist_ok=False)
    timings = {
        "cache_lookup_s": round(time.monotonic() - cache_started, 3),
        "cache_bypassed": bool(force_model_rerun),
    }
    try:
        in_file_path = tmp_input / f"{safe_case}_0000.nii.gz"
        if ct_path.name.lower().endswith(".nii.gz"):
            _copy_or_link(ct_path, in_file_path)
        else:
            import gzip
            with open(ct_path, "rb") as source:
                with gzip.open(in_file_path, "wb", compresslevel=6) as destination:
                    shutil.copyfileobj(source, destination)

        source_img = nib.load(str(in_file_path))
        if len(source_img.shape) != 3:
            return None, f"Model girdisi üç boyutlu değil: {source_img.shape}", {}

        # Bağımsız anatomik model önce çalışır. Pankreas doğrulanamıyorsa tümör
        # modelleri hiç çalıştırılmaz; doğrulanırsa diğer modeller yalnız geniş,
        # fiziksel olarak güvenli axial ROI'yi tarar.
        gate_path, gate_error, gate_elapsed = _run_anatomical_gate(
            in_file_path, gate_output, gate_log_path, job_id=job_id,
            audit_id=audit_id,
        )
        timings["anatomical_gate_s"] = round(gate_elapsed, 3)
        if gate_path is None:
            return None, gate_error, {}
        gate_img = nib.load(str(gate_path))
        gate_data = np.asanyarray(gate_img.dataobj)
        if tuple(source_img.shape) != tuple(gate_data.shape):
            return None, (
                "3B anatomik kapı ile CT boyutları uyuşmuyor: "
                f"{source_img.shape} / {gate_data.shape}"
            ), {}
        if not np.allclose(source_img.affine, gate_img.affine, atol=1e-3, rtol=1e-5):
            return None, "3B anatomik maske ile CT dünya koordinatları uyuşmuyor.", {}

        from segmentation_postprocess import (
            assess_pancreas_gate, extract_unverified_tumor_candidates,
            validate_and_fuse_segmentation,
        )
        source_spacing = nib.affines.voxel_sizes(source_img.affine)
        gate_largest, gate_assessment = assess_pancreas_gate(
            gate_data, source_spacing, ANATOMICAL_GATE_CONFIG
        )
        # The full TotalSegmentator file proxy is no longer needed once the
        # verified component has been extracted. Releasing it early prevents
        # later 3D model workers from competing with stale full-volume pages.
        del gate_data, gate_img
        gc.collect()
        if not gate_assessment["gate_plausible"]:
            mask_data = np.zeros(source_img.shape, dtype=np.uint8)
            quality = {
                **gate_assessment,
                "status": "indeterminate",
                "pancreas_verified": False,
                "has_tumor": None,
                "raw_pancreas_voxels": 0,
                "raw_tumor_voxels": 0,
                "nnunet_raw_tumor_voxels": 0,
                "dints_raw_tumor_voxels": 0,
                "rejected_tumor_voxels": 0,
                "pancreas_voxels": 0,
                "pancreas_ml": 0.0,
                "tumor_voxels": 0,
                "tumor_ml": 0.0,
                "tumor_components": [],
                "models": ["TotalSegmentator 3B full"],
                "checkpoint": selected_checkpoint,
                "roi_original_shape": list(source_img.shape),
                "roi_shape": None,
                "roi_reduction_pct": 100.0,
                "timings": timings,
                "cache_key": cache_key,
                "cache_hit": False,
                "execution_mode": "fresh_gpu",
                "analysis_profile": profile_name,
                "analysis_profile_label": profile.get("label"),
            }
            persistent_mask = nib.Nifti1Image(
                mask_data, source_img.affine, source_img.header.copy()
            )
            persistent_mask.set_data_dtype(np.uint8)
            persistent_mask.set_qform(source_img.affine, code=1)
            persistent_mask.set_sform(source_img.affine, code=1)
            nib.save(persistent_mask, str(persistent_path))
            _finish_gpu_audit(audit_id, "completed")
            _attach_gpu_audit(quality, audit_id)
            _save_inference_cache(cache_key, persistent_path, quality)
            update_progress(job_id, 90, "Pankreas anatomik olarak doğrulanamadı", "Tümör modelleri güvenlik için çalıştırılmadı")
            return mask_data, None, quality

        model_input_path = model_input / f"{safe_case}_0000.nii.gz"
        crop_slices, roi_shape = _prepare_anatomical_roi(
            source_img, gate_largest, model_input_path
        )
        # Copy only the much smaller axial ROI so the full-size component array
        # can be reclaimed before nnU-Net/DiNTS/PanTS allocate their tensors.
        gate_crop = np.asanyarray(gate_largest[crop_slices]).copy()
        del gate_largest
        gc.collect()
        original_voxels = max(1, int(np.prod(source_img.shape)))
        roi_voxels = int(np.prod(roi_shape))
        roi_reduction = 100.0 * (1.0 - roi_voxels / original_voxels)
        update_progress(
            job_id, 69, "Anatomik ROI hazırlandı",
            f"{source_img.shape[2]} kesitten {roi_shape[2]} kesite güvenli kırpma (%{roi_reduction:.1f} azalma)",
        )

        command = [
            sys.executable, "-c",
            "from nnunetv2.inference.predict_from_raw_data import predict_entry_point; predict_entry_point()",
            "-i", str(model_input), "-o", str(tmp_output),
            "-d", "007", "-c", "2d", "-f", "0",
            "-step_size", "0.5", "-npp", "2", "-nps", "2",
            "-chk", selected_checkpoint, "-device", "cuda",
        ]
        if not profile.get("nnunet_tta", True):
            command.append("--disable_tta")
        debug_log(
            "MODEL", "Inference başladı; giriş=%s, checkpoint=%s",
            ct_path.name, selected_checkpoint,
        )
        process_returncode, nnunet_elapsed = _run_logged_process(
            command, log_path, MODEL_TIMEOUT_SECONDS,
            job_id=job_id,
            stage="2B nnU-Net çalışıyor",
            detail=f"Pankreas ROI'sinde {roi_shape[2]} kesit taranıyor",
            progress_start=70,
            progress_end=80,
            env=_gpu_environment(),
            parse_model_progress=True,
            expected_seconds=float(INFERENCE_CONFIG.get("nnunet_expected_seconds", 90)),
            audit_id=audit_id,
        )
        timings["nnunet_s"] = round(nnunet_elapsed, 3)
        if process_returncode != 0:
            details = "nnU-Net hata ayrıntısı alınamadı."
            if log_path.exists():
                details = log_path.read_text(encoding="utf-8", errors="replace")[-2000:].strip()
            return None, f"Model inference başarısız oldu: {details}", {}

        mask_files = sorted(tmp_output.glob("*.nii.gz"))
        if not mask_files:
            return None, "Model çalıştı fakat segmentasyon maskesi üretmedi.", {}

        update_progress(job_id, 80, "nnU-Net tamamlandı", "Ham 2B segmentasyon maskesi yükleniyor")
        mask_img = nib.load(str(mask_files[0]))
        raw_mask = np.asanyarray(mask_img.dataobj).astype(np.uint8, copy=False)
        model_ct_img = nib.load(str(model_input_path))
        if raw_mask.shape != gate_crop.shape:
            return None, (
                "3B anatomik kapı ile nnU-Net maskesinin boyutları uyuşmuyor: "
                f"{raw_mask.shape} / {gate_crop.shape}"
            ), {}
        if not np.allclose(mask_img.affine, model_ct_img.affine, atol=1e-3, rtol=1e-5):
            return None, "3B anatomik maske ile nnU-Net dünya koordinatları uyuşmuyor.", {}
        spacing = nib.affines.voxel_sizes(mask_img.affine)
        _, nnunet_tumor_voxels = _count_mask_labels(raw_mask)

        # Pankreas kapısı fiziksel olarak geçersizse pahalı tümör modelini hiç
        # çalıştırma ve ham nnU-Net maskesini yayınlamadan kapalı kal.
        mask_data, quality = validate_and_fuse_segmentation(
            raw_mask, gate_crop, spacing, ANATOMICAL_GATE_CONFIG
        )
        dints_img = None
        dints_mask = None
        if quality.get("gate_plausible"):
            dints_input_path = dints_input / f"{safe_case}_0000.nii.gz"
            dints_crop_slices, dints_roi_shape = _prepare_dints_roi(
                model_ct_img, gate_crop, dints_input_path
            )
            dints_original_voxels = max(1, int(np.prod(raw_mask.shape)))
            dints_roi_voxels = int(np.prod(dints_roi_shape))
            dints_reduction = 100.0 * (
                1.0 - dints_roi_voxels / dints_original_voxels
            )
            update_progress(
                job_id, 81, "3B tümör ROI'si hazırlandı",
                f"DiNTS girdisi {tuple(raw_mask.shape)} boyutundan "
                f"{dints_roi_shape} boyutuna kırpıldı (%{dints_reduction:.1f} azalma)",
            )
            dints_path, dints_error, dints_elapsed = _run_dints_tumor_model(
                dints_input_path, dints_work, dints_output, dints_log_path,
                job_id=job_id, audit_id=audit_id,
                overlap=profile.get("dints_overlap"),
            )
            timings["dints_s"] = round(dints_elapsed, 3)
            if dints_path is None:
                return None, dints_error, {}
            dints_img = nib.load(str(dints_path))
            dints_roi_mask = np.asanyarray(dints_img.dataobj).astype(np.uint8, copy=False)
            if tuple(dints_roi_mask.shape) != tuple(dints_roi_shape):
                return None, (
                    "3B DiNTS ROI girdisi ile maskesinin boyutları uyuşmuyor: "
                    f"{dints_roi_shape} / {dints_roi_mask.shape}"
                ), {}
            dints_ct_img = nib.load(str(dints_input_path))
            if not np.allclose(dints_ct_img.affine, dints_img.affine, atol=1e-3, rtol=1e-5):
                return None, "3B DiNTS ROI girdisi ile maskesinin dünya koordinatları uyuşmuyor.", {}
            dints_mask = np.zeros(raw_mask.shape, dtype=np.uint8)
            dints_mask[dints_crop_slices] = dints_roi_mask

            # Duyarlılık odaklı birleşim, bağımsız kapı ve fiziksel bileşen
            # kurallarıyla yalancı pozitiflerden arındırılır.
            mask_data, quality = validate_and_fuse_segmentation(
                raw_mask, gate_crop, spacing, ANATOMICAL_GATE_CONFIG,
                secondary_mask=dints_mask,
            )
            _, dints_tumor_voxels = _count_mask_labels(dints_mask)
            quality["dints_roi_shape"] = list(dints_roi_shape)
            quality["dints_roi_ranges"] = [
                [int(part.start), int(part.stop)] for part in dints_crop_slices
            ]
            quality["dints_roi_reduction_pct"] = round(dints_reduction, 2)
            del dints_roi_mask, dints_ct_img
        else:
            dints_tumor_voxels = 0
            quality["dints_skipped_reason"] = "3B pankreas kapısı fiziksel olarak geçersiz."

        candidate_mask, candidate_records = extract_unverified_tumor_candidates(
            raw_mask,
            gate_crop,
            spacing,
            ANATOMICAL_GATE_CONFIG,
            secondary_mask=dints_mask,
        )
        candidate_mask &= mask_data != 2
        candidate_voxels = int(np.count_nonzero(candidate_mask))
        voxel_ml = abs(float(np.prod(spacing[:3]))) / 1000.0
        quality["unverified_tumor_voxels"] = candidate_voxels
        quality["unverified_tumor_ml"] = round(candidate_voxels * voxel_ml, 3)
        quality["unverified_tumor_components"] = candidate_records
        if candidate_voxels and not np.any(mask_data == 2):
            quality["has_tumor"] = None
            quality["status"] = "indeterminate"
            quality["reason"] = (
                "Anlamlı tümör adayları üretildi; modeller kesin sınır üzerinde "
                "yeterli uzamsal uzlaşı göstermedi. Negatif karar verilemez."
            )

        quality["nnunet_raw_tumor_voxels"] = nnunet_tumor_voxels
        quality["dints_raw_tumor_voxels"] = dints_tumor_voxels
        quality["models"] = (
            ["nnU-Net v2 2D", "MONAI DiNTS 3B", "TotalSegmentator 3B full"]
            if quality.get("gate_plausible")
            else ["nnU-Net v2 2D", "TotalSegmentator 3B full"]
        )
        quality["checkpoint"] = selected_checkpoint
        quality["roi_original_shape"] = list(source_img.shape)
        quality["roi_shape"] = list(roi_shape)
        quality["roi_slice_range"] = [
            int(crop_slices[2].start or 0), int(crop_slices[2].stop or source_img.shape[2])
        ]
        quality["roi_reduction_pct"] = round(roi_reduction, 2)
        quality["timings"] = timings
        quality["cache_key"] = cache_key
        quality["cache_hit"] = False
        quality["analysis_profile"] = profile_name
        quality["analysis_profile_label"] = profile.get("label")
        quality["profile_settings"] = {
            "nnunet_tta": bool(profile.get("nnunet_tta", True)),
            "dints_overlap": float(profile.get("dints_overlap", 0.625)),
            "pants_enabled": bool(profile.get("pants_enabled", True)),
        }
        update_progress(job_id, 89, "3B doğrulama tamamlandı", "Fiziksel bileşen ölçütleri uygulandı")

        full_mask = np.zeros(source_img.shape, dtype=np.uint8)
        full_mask[crop_slices] = mask_data
        full_candidate_mask = np.zeros(source_img.shape, dtype=bool)
        full_candidate_mask[crop_slices] = candidate_mask
        uncertainty_mask = np.zeros(source_img.shape, dtype=np.uint8)
        uncertainty_mask[full_candidate_mask] = 3
        # Validation has already reduced all model outputs to full_mask and
        # scalar quality fields. Drop stale ROI arrays and NIfTI proxies before
        # the two memory-heavy PanTS workers start.
        del raw_mask, gate_crop, candidate_mask, mask_data, mask_img, model_ct_img
        if dints_mask is not None:
            del dints_mask
        if dints_img is not None:
            del dints_img
        gc.collect()
        run_pants = (
            bool(profile.get("pants_enabled", True))
            and _parse_bool(PANTS_REFINEMENT_CONFIG.get("enabled"), False)
        )
        if run_pants:
            update_progress(
                job_id, 89, "PanTS üçlü kalibrasyon başlatıldı",
                "MedFormer ve R-Super bağımsız 3B olasılık haritaları üretiyor",
            )
            full_mask, uncertainty_mask, pants_quality, pants_timings = _run_pants_refinement(
                source_img, full_mask, safe_case, job_id, audit_id=audit_id,
                candidate_mask=full_candidate_mask,
            )
            quality.update(pants_quality)
            timings.update(pants_timings)
            quality["models"] = list(quality.get("models", [])) + list(
                pants_quality.get("pants_models", [])
            )
            quality["tumor_voxels"] = int(np.count_nonzero(full_mask == 2))
            quality["tumor_ml"] = round(
                quality["tumor_voxels"] * abs(float(np.linalg.det(source_img.affine[:3, :3]))) / 1000.0,
                3,
            )
            quality["pancreas_voxels"] = int(np.count_nonzero(full_mask == 1))
            quality["pancreas_ml"] = round(
                quality["pancreas_voxels"] * abs(float(np.linalg.det(source_img.affine[:3, :3]))) / 1000.0,
                3,
            )
            if quality["tumor_voxels"] > 0:
                quality["has_tumor"] = True
                quality["status"] = "candidate"
                quality["reason"] = None
            else:
                pants_screening_voxels = int(
                    pants_quality.get("pants_screening_candidate_voxels", 0) or 0
                )
                if pants_screening_voxels:
                    full_candidate_mask |= uncertainty_mask == 3
                    candidate_voxels = int(np.count_nonzero(full_candidate_mask))
            if quality["tumor_voxels"] == 0 and candidate_voxels:
                quality["has_tumor"] = None
                quality["status"] = "indeterminate"
                quality["reason"] = pants_quality.get("pants_rejection_reason") or (
                    "Belirsiz tümör adayı ek hakem modellerce doğrulanamadı; "
                    "negatif karar verilemez."
                )
            elif quality["tumor_voxels"] == 0:
                quality["has_tumor"] = False
                quality["status"] = "negative"
            update_progress(
                job_id, 90, "PanTS üçlü kalibrasyon tamamlandı",
                "Ana sınır, yüksek güvenli çekirdek ve duyarlı sınır kaydedildi",
            )
        else:
            quality.update({
                "pants_enabled": False,
                "pants_skipped": True,
                "pants_skipped_reason": (
                    "Seçilen hızlı profilde iki ek PanTS modeli çalıştırılmaz."
                ),
                "pants_models": [],
            })
            update_progress(
                job_id, 90, "Hızlı 3-model doğrulama tamamlandı",
                "nnU-Net, DiNTS ve TotalSegmentator sonucu geometrik olarak birleştirildi",
            )
        remaining_unverified = int(np.count_nonzero(
            full_candidate_mask & (full_mask != 2)
        ))
        initially_rejected = int(quality.get("rejected_tumor_voxels", 0) or 0)
        confirmed_initial_candidates = int(np.count_nonzero(
            full_candidate_mask & (full_mask == 2)
        ))
        quality["initially_rejected_tumor_voxels"] = initially_rejected
        quality["rejected_tumor_voxels"] = max(
            0, initially_rejected - confirmed_initial_candidates
        )
        quality["unverified_tumor_voxels"] = remaining_unverified
        quality["unverified_tumor_ml"] = round(remaining_unverified * voxel_ml, 3)
        uncertainty_path = MASK_DIR / f"{safe_case}_uncertainty.nii.gz"
        uncertainty_img = nib.Nifti1Image(
            uncertainty_mask, source_img.affine, source_img.header.copy()
        )
        uncertainty_img.set_data_dtype(np.uint8)
        uncertainty_img.set_qform(source_img.affine, code=1)
        uncertainty_img.set_sform(source_img.affine, code=1)
        nib.save(uncertainty_img, str(uncertainty_path))
        quality["uncertainty_mask_path"] = str(uncertainty_path)
        del full_candidate_mask
        persistent_mask = nib.Nifti1Image(
            full_mask, source_img.affine, source_img.header.copy()
        )
        persistent_mask.set_data_dtype(np.uint8)
        persistent_mask.set_qform(source_img.affine, code=1)
        persistent_mask.set_sform(source_img.affine, code=1)
        nib.save(persistent_mask, str(persistent_path))
        quality["execution_mode"] = "fresh_gpu"
        _finish_gpu_audit(audit_id, "completed")
        _attach_gpu_audit(quality, audit_id)
        _save_inference_cache(cache_key, persistent_path, quality)
        debug_log(
            "MODEL", "Inference tamamlandı; maske=%s, 3B kalite=%s",
            full_mask.shape, quality,
        )
        return full_mask, None, quality
    except subprocess.TimeoutExpired:
        message = "Model zinciri zaman sınırını aştı."
        audit = _finish_gpu_audit(audit_id, "failed", message)
        return None, message, {"gpu_audit": audit, "gpu_audit_id": audit_id}
    except Exception as exc:
        message = f"Inference sırasında hata oluştu: {exc}"
        audit = _finish_gpu_audit(audit_id, "failed", message)
        return None, message, {"gpu_audit": audit, "gpu_audit_id": audit_id}
    finally:
        _finish_gpu_audit(
            audit_id, "failed", "Model zinciri başarı durumuna ulaşmadan sonlandı."
        )
        shutil.rmtree(tmp_input, ignore_errors=True)
        shutil.rmtree(model_input, ignore_errors=True)
        shutil.rmtree(tmp_output, ignore_errors=True)
        shutil.rmtree(dints_work, ignore_errors=True)
        shutil.rmtree(dints_input, ignore_errors=True)
        shutil.rmtree(dints_output, ignore_errors=True)
        shutil.rmtree(gate_output, ignore_errors=True)
        try:
            log_path.unlink(missing_ok=True)
        except Exception:
            pass
        try:
            dints_log_path.unlink(missing_ok=True)
        except Exception:
            pass
        try:
            gate_log_path.unlink(missing_ok=True)
        except Exception:
            pass
        gc.collect()


def _create_web_viz(
    ct_data, mask_data, result, output_path: Path, affine=None,
    uncertainty_data=None,
):
    """Pankreas içeren tam 8 kesitte CT + değişmeden gelen model maskesini gösterir."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.colors import ListedColormap
    import numpy as np

    try:
        ct_view = np.squeeze(ct_data)
        mask_view = np.squeeze(mask_data)
        if ct_view.ndim == 2:
            ct_view = ct_view[:, :, np.newaxis]
        if mask_view.ndim == 2:
            mask_view = mask_view[:, :, np.newaxis]
        if ct_view.ndim > 3:
            ct_view = ct_view[..., 0]
        if mask_view.ndim > 3:
            mask_view = mask_view[..., 0]
        if ct_view.ndim != 3 or mask_view.ndim != 3:
            raise ValueError("CT veya maske üç boyutlu değil.")

        if ct_view.shape != mask_view.shape:
            for axes in ((1, 2, 0), (2, 0, 1), (0, 2, 1), (2, 1, 0)):
                candidate = np.transpose(ct_view, axes)
                if candidate.shape == mask_view.shape:
                    ct_view = candidate
                    break
        if ct_view.shape != mask_view.shape:
            raise ValueError(f"CT ve maske boyutları uyuşmuyor: {ct_view.shape} / {mask_view.shape}")

        uncertainty_view = None
        if uncertainty_data is not None:
            uncertainty_view = np.squeeze(np.asarray(uncertainty_data))
            if uncertainty_view.shape != mask_view.shape:
                uncertainty_view = None

        depth = mask_view.shape[2]
        pan_counts = np.zeros(depth, dtype=np.int64)
        tum_counts = np.zeros(depth, dtype=np.int64)
        for slice_index in range(depth):
            slice_mask = mask_view[:, :, slice_index]
            pan_counts[slice_index] = np.count_nonzero(slice_mask == 1)
            tum_counts[slice_index] = np.count_nonzero(slice_mask == 2)

        pancreas_indices = np.flatnonzero(pan_counts > 0)
        selected = []
        uncertainty_counts = np.zeros(depth, dtype=np.int64)
        if uncertainty_view is not None:
            for slice_index in range(depth):
                uncertainty_counts[slice_index] = np.count_nonzero(
                    uncertainty_view[:, :, slice_index] == 3
                )
            uncertain_indices = np.flatnonzero(uncertainty_counts > 0)
            if uncertain_indices.size:
                ranked_uncertain = uncertain_indices[
                    np.argsort(-uncertainty_counts[uncertain_indices])
                ]
                selected.extend(int(value) for value in ranked_uncertain[:4])
        if pancreas_indices.size:
            tumor_indices = pancreas_indices[tum_counts[pancreas_indices] > 0]
            if tumor_indices.size:
                ranked_tumor = tumor_indices[np.argsort(-tum_counts[tumor_indices])]
                selected.extend(int(value) for value in ranked_tumor[:4])

            distributed_positions = np.linspace(
                0, pancreas_indices.size - 1, min(8, pancreas_indices.size), dtype=int
            )
            selected.extend(int(pancreas_indices[position]) for position in distributed_positions)
            selected = list(dict.fromkeys(selected))[:8]
            while len(selected) < 8:
                selected.append(int(pancreas_indices[len(selected) % pancreas_indices.size]))
        else:
            selected = [int(value) for value in np.linspace(0, max(0, depth - 1), 8, dtype=int)]

        result["visualized_slices"] = selected
        result["visualized_pancreas_slices"] = int(sum(pan_counts[index] > 0 for index in selected))
        figure, axes = plt.subplots(
            2, 4, figsize=(20, 10), facecolor="#0d0d1a", squeeze=False
        )
        mask_cmap = ListedColormap(["none", "#27ae60", "#e74c3c"])
        left_marker, right_marker = "R", "L"
        if affine is not None:
            try:
                import nibabel as nib
                horizontal_positive = nib.aff2axcodes(np.asarray(affine))[1]
                opposite = {"L": "R", "R": "L", "A": "P", "P": "A"}
                right_marker = horizontal_positive
                left_marker = opposite.get(horizontal_positive, "?")
            except Exception:
                pass

        for panel, slice_index in enumerate(selected):
            axis = axes.flat[panel]
            # Dizi eksenleri DICOM (satır, sütun, kesit) sırasındadır. Transpoz
            # aksiyel görüntüyü 90 derece döndürüyordu; standart radyolojik görünüm
            # satır=üst-alt, sütun=sağ-sol olacak şekilde doğrudan çizilir.
            ct_slice = np.asarray(ct_view[:, :, slice_index], dtype=np.float32)
            ct_slice = np.nan_to_num(ct_slice, nan=-150.0, posinf=250.0, neginf=-150.0)
            ct_slice = np.clip(ct_slice, -150.0, 250.0)
            ct_slice = (ct_slice + 150.0) / 400.0
            mask_slice = np.asarray(mask_view[:, :, slice_index])

            axis.imshow(ct_slice, cmap="gray", origin="upper", vmin=0, vmax=1)
            axis.imshow(
                np.ma.masked_where(mask_slice < 0.5, mask_slice),
                cmap=mask_cmap, alpha=0.62, origin="upper", vmin=0, vmax=2,
            )
            if uncertainty_view is not None:
                uncertainty_slice = uncertainty_view[:, :, slice_index]
                envelope = uncertainty_slice > 0
                if np.any(envelope):
                    axis.contour(
                        envelope, levels=[0.5], colors=["#f59e0b"],
                        linewidths=1.8,
                    )
            axis.text(
                0.01, 0.5, left_marker, transform=axis.transAxes, color="white",
                fontsize=9, fontweight="bold", va="center",
            )
            axis.text(
                0.96, 0.5, right_marker, transform=axis.transAxes, color="white",
                fontsize=9, fontweight="bold", va="center",
            )
            tumor_text = f" • Tümör {int(tum_counts[slice_index])}" if tum_counts[slice_index] else ""
            uncertain_text = (
                f" • Belirsiz {int(uncertainty_counts[slice_index])}"
                if uncertainty_counts[slice_index] else ""
            )
            axis.set_title(
                f"Kesit {slice_index} • Pankreas {int(pan_counts[slice_index])}"
                f"{tumor_text}{uncertain_text}",
                color="white", fontsize=10, fontweight="bold",
            )
            axis.axis("off")

        if result.get("has_tumor") is True:
            color = "#e74c3c"
        elif result.get("has_tumor") is False and result.get("pancreas_verified"):
            color = "#2ecc71"
        else:
            color = "#f59e0b"
        figure.suptitle(
            f"MODEL ÇIKTISI: {result.get('prediction', '?')}",
            fontsize=20, fontweight="bold", color=color, y=1.01,
        )
        figure.legend(
            handles=[
                mpatches.Patch(color="#27ae60", label="Pankreas (Label 1)"),
                mpatches.Patch(color="#e74c3c", label="Doğrulanmış tümör çekirdeği"),
                mpatches.Patch(
                    facecolor="none", edgecolor="#f59e0b", linewidth=2,
                    label="Belirsiz model adayı / olası dış sınır",
                ),
            ],
            loc="lower center", ncol=3, facecolor="#1a1a2e",
            labelcolor="white", fontsize=12,
        )
        figure.tight_layout(rect=(0, 0.05, 1, 0.97))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(
            str(output_path), dpi=170, bbox_inches="tight", facecolor="#0d0d1a"
        )
        plt.close(figure)
        debug_log("GÖRSEL", "8 panel hazırlandı; kesitler=%s", selected)
    except Exception as exc:
        plt.close("all")
        debug_log("GÖRSEL", "Görselleştirme hatası: %s", exc)
        raise


# ============================================================
# MAIN
# ============================================================
def run_startup_checks():
    """Başlangıç ayarlarını doğrular; ayrıntıları yalnız DEBUG=True iken yazar."""
    port = int(os.environ.get("FLASK_PORT", WEB_CONFIG.get("port", 5000)))
    checkpoint_dir = (
        Path(os.environ.get("nnUNet_results", BASE_PATH / "data" / "nnunet_results")) /
        "Dataset007_Pancreas" / "nnUNetTrainer__nnUNetPlans__2d" / "fold_0"
    )
    checkpoint_ok = any((checkpoint_dir / name).exists() for name in (
        "checkpoint_final.pth", "checkpoint_best.pth"
    ))
    cuda_info = _require_cuda()
    debug_log(
        "CUDA", "%s | %s GB | torch=%s | CUDA=%s | ayirici=%s | moduller=%s | CUDA_VISIBLE_DEVICES=%s",
        cuda_info.get("name"), cuda_info.get("memory_gb"), cuda_info.get("torch"),
        cuda_info.get("cuda"), cuda_info.get("allocator"),
        cuda_info.get("module_loading"), cuda_info.get("visible_devices"),
    )
    debug_log(
        "GPU",
        "Bos GPU yuku kapali; secili NVIDIA aygiti yalniz model calisirken kullanilacak",
    )

    # Each model family uses its own Python environment. Importing PyTorch and
    # creating the first CUDA context used to happen in the critical path of
    # the first uploaded case. Verify those runtimes once before the server
    # accepts requests. This does not load a model or reuse a segmentation;
    # it only moves invariant CUDA startup work to service startup.
    runtime_specs = []
    if _parse_bool(ANATOMICAL_GATE_CONFIG.get("enabled"), True):
        gate_python = ANATOMICAL_GATE_CONFIG.get("python")
        if gate_python:
            runtime_specs.append((
                "TotalSegmentator/DiNTS",
                _project_path(gate_python),
                [
                    str(_project_path(path))
                    for path in ANATOMICAL_GATE_CONFIG.get("pythonpath", [])
                ],
            ))
    if _parse_bool(TUMOR_MODEL_3D_CONFIG.get("enabled"), True):
        runtime_specs.append((
            "DiNTS",
            _project_path(TUMOR_MODEL_3D_CONFIG.get(
                "python", ".venv_totalseg/Scripts/python.exe"
            )),
            [
                str(_project_path(path))
                for path in TUMOR_MODEL_3D_CONFIG.get("pythonpath", [])
            ],
        ))
    if _parse_bool(PANTS_REFINEMENT_CONFIG.get("enabled"), False):
        runtime_specs.append((
            "MedFormer/R-Super",
            _project_path(PANTS_REFINEMENT_CONFIG.get("python", sys.executable)),
            [
                str(_project_path(path))
                for path in PANTS_REFINEMENT_CONFIG.get("pythonpath", [])
            ],
        ))

    warmed = set()
    for label, executable, python_paths in runtime_specs:
        key = (str(Path(executable).resolve()), tuple(python_paths))
        if key in warmed:
            continue
        info = _probe_cuda_runtime(executable, python_paths)
        warmed.add(key)
        debug_log(
            "CUDA", "%s ortamı hazır: %s (%s)",
            label, info.get("name"), info.get("device"),
        )
    debug_log(
        "BAŞLANGIÇ", "Tanı logları=%s; Flask debug middleware=%s; HTTP erişim seli=kapalı",
        DEBUG_ENABLED, FLASK_DEBUG_ENABLED,
    )
    debug_log(
        "BAŞLANGIÇ", "Port=%s, yükleme=%s MB, dosya sınırı=%s",
        port, MAX_UPLOAD_MB, MAX_UPLOAD_FILES,
    )
    debug_log("BAŞLANGIÇ", "Model checkpoint=%s", "hazır" if checkpoint_ok else "bulunamadı")
    try:
        import psutil
        memory_gb = psutil.virtual_memory().total / (1024 ** 3)
        debug_log("BAŞLANGIÇ", "Sistem RAM=%.1f GB; disk tabanlı yükleme etkin", memory_gb)
    except Exception:
        debug_log("BAŞLANGIÇ", "RAM bilgisi okunamadı; disk tabanlı yükleme etkin")
    return port


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=run_startup_checks(),
        debug=FLASK_DEBUG_ENABLED,
        threaded=True,
        use_reloader=False,
    )
