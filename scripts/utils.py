"""
============================================================
UTILS - Ortak Yardımcı Fonksiyonlar
============================================================
Bu modül projedeki tüm scriptler tarafından paylaşılan
ortak fonksiyonları içerir.

Kullanım:
    from utils import load_config, setup_logging, get_device
============================================================
"""

import os
import sys
import json
import logging
import platform
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, Tuple

import numpy as np


# ============================================================
# KONFİGÜRASYON
# ============================================================
def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    config.json dosyasını yükler.

    Args:
        config_path: config.json dosyasının yolu.
                     None ise otomatik olarak proje kökünde arar.

    Returns:
        Config sözlüğü
    """
    if config_path is None:
        # Scriptin bulunduğu dizinden yukarı git, config.json ara
        script_dir = Path(__file__).parent
        config_path = script_dir / "config.json"
        if not config_path.exists():
            config_path = script_dir.parent / "config.json"

    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(
            f"config.json bulunamadı: {config_path}\n"
            "Önce setup_project.py çalıştırın."
        )

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    return config


# ============================================================
# LOGLAMA
# ============================================================
def setup_logging(
    log_name: str,
    log_dir: Optional[str] = None,
    level: int = logging.INFO
) -> logging.Logger:
    """
    Hem dosyaya hem terminale yazan logger oluşturur.

    Args:
        log_name:  Logger adı (örn: "training", "inference")
        log_dir:   Log dosyasının yazılacağı klasör
        level:     Loglama seviyesi

    Returns:
        Yapılandırılmış Logger nesnesi
    """
    logger = logging.getLogger(log_name)
    logger.setLevel(level)

    # Formatter
    formatter = logging.Formatter(
        fmt="[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Terminal handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Dosya handler
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f"{log_name}_{timestamp}.log")
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logger.info(f"Log dosyası: {log_file}")

    return logger


# ============================================================
# GPU / DEVICE YÖNETİMİ
# ============================================================
def get_device(prefer_gpu: bool = True) -> "torch.device":
    """
    Kullanılabilir en iyi hesaplama cihazını döndürür.

    Args:
        prefer_gpu: True ise GPU tercih edilir, False ise CPU

    Returns:
        torch.device nesnesi
    """
    try:
        import torch
        # GPU kontrolü
        device = torch.device("cuda" if (prefer_gpu and torch.cuda.is_available()) else "cpu")

        if device.type == "cuda":
            gpu_name   = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"  ✓ GPU: {gpu_name} ({gpu_memory:.1f} GB)")
        else:
            print("  ⚠ GPU bulunamadı, CPU kullanılıyor")

        return device

    except ImportError:
        raise ImportError("PyTorch kurulu değil! setup.bat çalıştırın.")


def get_device_str() -> str:
    """String olarak device döndürür: 'cuda' veya 'cpu'"""
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


# ============================================================
# NIfTI DOSYA İŞLEME
# ============================================================
def load_nifti(file_path: str) -> Tuple[np.ndarray, Any]:
    """
    NIfTI (.nii veya .nii.gz) dosyasını yükler.

    Args:
        file_path: NIfTI dosyasının yolu

    Returns:
        (numpy array, nibabel image nesnesi)
    """
    try:
        import nibabel as nib
        img = nib.load(file_path)
        data = img.get_fdata()
        return data, img
    except ImportError:
        raise ImportError("nibabel kurulu değil! pip install nibabel")
    except Exception as e:
        raise IOError(f"NIfTI dosyası okunamadı: {file_path}\nHata: {e}")


def save_nifti(
    data: np.ndarray,
    output_path: str,
    reference_img=None,
    affine: Optional[np.ndarray] = None
) -> None:
    """
    Numpy array'i NIfTI formatında kaydeder.

    Args:
        data:          Kaydedilecek numpy array
        output_path:   Çıktı dosyası yolu (.nii.gz önerilir)
        reference_img: Referans NIfTI görüntüsü (affine için)
        affine:        Affine dönüşüm matrisi (reference_img yoksa)
    """
    try:
        import nibabel as nib

        if reference_img is not None:
            affine = reference_img.affine
            header = reference_img.header
            img = nib.Nifti1Image(data, affine, header)
        elif affine is not None:
            img = nib.Nifti1Image(data, affine)
        else:
            # Varsayılan identity affine
            img = nib.Nifti1Image(data, np.eye(4))

        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        nib.save(img, output_path)

    except ImportError:
        raise ImportError("nibabel kurulu değil! pip install nibabel")


# ============================================================
# SEGMENTASYON ANALİZİ
# ============================================================
def analyze_segmentation(mask: np.ndarray) -> Dict[str, Any]:
    """
    Segmentasyon maskesini analiz eder ve tümör kararı verir.

    Args:
        mask: 3D segmentasyon maskesi (0=background, 1=pancreas, 2=tumor)

    Returns:
        Analiz sonuçlarını içeren sözlük
    """
    total_voxels     = mask.size
    background_voxels = np.sum(mask == 0)
    pancreas_voxels  = np.sum(mask == 1)
    tumor_voxels     = np.sum(mask == 2)

    has_tumor = tumor_voxels > 0

    return {
        "total_voxels":     int(total_voxels),
        "background":       int(background_voxels),
        "pancreas_voxels":  int(pancreas_voxels),
        "tumor_voxels":     int(tumor_voxels),
        "pancreas_ratio":   float(pancreas_voxels / total_voxels),
        "tumor_ratio":      float(tumor_voxels / total_voxels),
        "has_tumor":        bool(has_tumor),
        "prediction":       "Tümör Var" if has_tumor else "Tümör Yok",
        "confidence":       "Yüksek" if tumor_voxels > 500 else ("Orta" if tumor_voxels > 50 else "Düşük"),
    }


# ============================================================
# GENEL YARDIMCILAR
# ============================================================
def ensure_dirs(*paths: str) -> None:
    """Birden fazla klasörü oluşturur (varsa geçer)."""
    for path in paths:
        os.makedirs(path, exist_ok=True)


def format_time(seconds: float) -> str:
    """Saniyeyi okunabilir formata çevirir."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds // 60
        secs = seconds % 60
        return f"{int(minutes)}m {int(secs)}s"
    else:
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        return f"{int(hours)}h {int(minutes)}m"


def print_system_info() -> None:
    """Sistem bilgilerini yazdırır."""
    print("\n" + "="*60)
    print("  SİSTEM BİLGİLERİ")
    print("="*60)
    print(f"  OS:              {platform.system()} {platform.release()}")
    print(f"  Python:          {sys.version.split()[0]}")
    print(f"  Proje:           PancreasCancerDetection v1.0.0")

    try:
        import torch
        print(f"  PyTorch:         {torch.__version__}")
        print(f"  CUDA Available:  {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  CUDA Version:    {torch.version.cuda}")
            print(f"  GPU:             {torch.cuda.get_device_name(0)}")
    except ImportError:
        print("  PyTorch:         KURULU DEĞİL")

    try:
        import nibabel
        print(f"  nibabel:         {nibabel.__version__}")
    except ImportError:
        print("  nibabel:         KURULU DEĞİL")

    print("="*60 + "\n")
