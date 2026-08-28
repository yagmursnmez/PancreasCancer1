"""
============================================================
ADIM 1 - DOĞRULAMA TESTİ
============================================================
Bu script, ADIM 1 kurulumunun başarılı olduğunu doğrular.
Çalıştırmadan önce setup.bat veya setup.sh çalıştırın.

Kullanım:
    python verify_step1.py
============================================================
"""

import os
import sys
import json
from pathlib import Path

# ============================================================
# RENK KODLARI
# ============================================================
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

def ok(msg):   print(f"  {GREEN}✓{RESET}  {msg}")
def warn(msg): print(f"  {YELLOW}⚠{RESET}  {msg}")
def fail(msg): print(f"  {RED}✗{RESET}  {msg}")
def info(msg): print(f"  {CYAN}ℹ{RESET}  {msg}")


def main():
    print(f"\n{BOLD}{CYAN}{'='*60}")
    print("  ADIM 1 - DOĞRULAMA TESTİ")
    print(f"{'='*60}{RESET}\n")

    BASE = Path(__file__).parent
    errors = 0
    warnings = 0

    # -------------------------------------------------------
    # TEST 1: Zorunlu dosyalar
    # -------------------------------------------------------
    print(f"{BOLD}[TEST 1] Zorunlu dosyalar:{RESET}")
    required_files = [
        "requirements.txt",
        "setup_project.py",
        "setup.bat",
        "config.json",
        ".env",
        "scripts/utils.py",
    ]
    for f in required_files:
        path = BASE / f
        if path.exists():
            ok(f)
        else:
            fail(f"{f} eksik!")
            errors += 1

    # -------------------------------------------------------
    # TEST 2: Zorunlu klasörler
    # -------------------------------------------------------
    print(f"\n{BOLD}[TEST 2] Zorunlu klasörler:{RESET}")
    required_dirs = [
        "data/raw",
        "data/nnunet_raw/Dataset007_Pancreas/imagesTr",
        "data/nnunet_raw/Dataset007_Pancreas/labelsTr",
        "data/nnunet_raw/Dataset007_Pancreas/imagesTs",
        "data/nnunet_preprocessed",
        "data/nnunet_results",
        "data/inference_output",
        "web/templates",
        "web/static/uploads",
        "scripts",
        "logs",
        "metrics",
    ]
    for d in required_dirs:
        path = BASE / d
        if path.exists():
            ok(d)
        else:
            fail(f"{d} eksik!")
            errors += 1

    # -------------------------------------------------------
    # TEST 3: config.json içeriği
    # -------------------------------------------------------
    print(f"\n{BOLD}[TEST 3] config.json:{RESET}")
    config_path = BASE / "config.json"
    if config_path.exists():
        try:
            with open(config_path) as f:
                config = json.load(f)
            required_keys = ["project", "paths", "dataset", "nnunet", "web"]
            for key in required_keys:
                if key in config:
                    ok(f"config['{key}'] mevcut")
                else:
                    fail(f"config['{key}'] eksik!")
                    errors += 1
        except json.JSONDecodeError as e:
            fail(f"config.json geçersiz JSON: {e}")
            errors += 1
    else:
        fail("config.json bulunamadı!")
        errors += 1

    # -------------------------------------------------------
    # TEST 4: nnU-Net ortam değişkenleri
    # -------------------------------------------------------
    print(f"\n{BOLD}[TEST 4] nnU-Net ortam değişkenleri:{RESET}")
    env_vars = ["nnUNet_raw", "nnUNet_preprocessed", "nnUNet_results"]
    for var in env_vars:
        val = os.environ.get(var)
        if val:
            ok(f"{var} = {val}")
        else:
            warn(f"{var} ayarlanmamış! (setup.bat çalıştırın veya manuel ayarlayın)")
            warnings += 1

    # -------------------------------------------------------
    # TEST 5: Python paketleri
    # -------------------------------------------------------
    print(f"\n{BOLD}[TEST 5] Python paketleri:{RESET}")

    critical_packages = {
        "torch":      "PyTorch",
        "nibabel":    "NIfTI Okuma",
        "SimpleITK":  "CT İşleme",
        "sklearn":    "Metrikler",
        "numpy":      "Sayısal İşlemler",
        "flask":      "Web Sunucu",
    }

    optional_packages = {
        "nnunetv2":   "nnU-Net v2",
        "rich":       "Terminal Çıktısı",
        "plotly":     "İnteraktif Grafikler",
    }

    for pkg, desc in critical_packages.items():
        try:
            mod = __import__(pkg)
            version = getattr(mod, "__version__", "?")
            ok(f"{pkg} v{version} ({desc})")
        except ImportError:
            fail(f"{pkg} kurulu değil! ({desc}) → pip install {pkg}")
            errors += 1

    for pkg, desc in optional_packages.items():
        try:
            mod = __import__(pkg)
            version = getattr(mod, "__version__", "?")
            ok(f"{pkg} v{version} ({desc})")
        except ImportError:
            warn(f"{pkg} kurulu değil (opsiyonel) ({desc})")
            warnings += 1

    # -------------------------------------------------------
    # TEST 6: PyTorch + GPU
    # -------------------------------------------------------
    print(f"\n{BOLD}[TEST 6] GPU Durumu:{RESET}")
    try:
        import torch

        # device kontrolü
        device = "cuda" if torch.cuda.is_available() else "cpu"

        if torch.cuda.is_available():
            ok(f"CUDA kullanılabilir")
            ok(f"GPU: {torch.cuda.get_device_name(0)}")
            mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            ok(f"GPU Belleği: {mem:.1f} GB")
            if mem < 8:
                warn("GPU belleği 8GB altında. Batch size 2 kullanın.")
                warnings += 1
        else:
            warn("GPU bulunamadı. CPU ile çalışacak (yavaş).")
            warnings += 1

    except ImportError:
        fail("PyTorch kurulu değil!")
        errors += 1

    # -------------------------------------------------------
    # SONUÇ
    # -------------------------------------------------------
    print(f"\n{BOLD}{'='*60}")
    print("  TEST SONUÇLARI")
    print(f"{'='*60}{RESET}")

    if errors == 0 and warnings == 0:
        print(f"\n  {GREEN}{BOLD}🎉 TÜM TESTLER BAŞARILI!{RESET}")
        print(f"  {GREEN}ADIM 1 tamamlandı. ADIM 2'ye geçebilirsiniz.{RESET}")
    elif errors == 0:
        print(f"\n  {YELLOW}{BOLD}⚠ Uyarılar var ama devam edebilirsiniz.{RESET}")
        print(f"  {YELLOW}Hata: {errors}, Uyarı: {warnings}{RESET}")
        print(f"  {GREEN}ADIM 2'ye geçebilirsiniz.{RESET}")
    else:
        print(f"\n  {RED}{BOLD}✗ HATALAR VAR! Devam etmeden önce düzeltin.{RESET}")
        print(f"  {RED}Hata: {errors}, Uyarı: {warnings}{RESET}")
        print(f"\n  Çözüm: setup.bat (Windows) veya setup.sh (Linux) çalıştırın.")

    print()
    return 0 if errors == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
