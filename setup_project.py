"""
============================================================
ADIM 1: PROJE KURULUM SCRIPTI
============================================================
Proje: CT Görüntülerinden Pankreas Tümörü Tespiti
Yöntem: nnU-Net v2 2D Segmentasyon + Kural Tabanlı Sınıflandırma

Kullanım:
    python setup_project.py

Bu script:
1. Proje klasör yapısını oluşturur
2. Ortam değişkenlerini (.env) ayarlar
3. GPU kullanılabilirliğini kontrol eder
4. Kurulum doğrulaması yapar
============================================================
"""

import os
import sys
import subprocess
import json
from pathlib import Path
from datetime import datetime

# ============================================================
# RENK KODLARI (Terminal çıktısı için)
# ============================================================
class Colors:
    GREEN  = "\033[92m"
    YELLOW = "\033[93m"
    RED    = "\033[91m"
    BLUE   = "\033[94m"
    CYAN   = "\033[96m"
    BOLD   = "\033[1m"
    RESET  = "\033[0m"

def print_header(text):
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}  {text}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.RESET}\n")

def print_success(text):
    print(f"  {Colors.GREEN}✓{Colors.RESET} {text}")

def print_warning(text):
    print(f"  {Colors.YELLOW}⚠{Colors.RESET}  {text}")

def print_error(text):
    print(f"  {Colors.RED}✗{Colors.RESET} {text}")

def print_info(text):
    print(f"  {Colors.CYAN}ℹ{Colors.RESET} {text}")


# ============================================================
# BÖLÜM 1: PROJE KLASÖR YAPISI
# ============================================================
def create_project_structure(base_path: str) -> dict:
    """
    Projenin tüm klasör yapısını oluşturur.

    Klasör Açıklamaları:
    --------------------
    data/raw/                    : Kaggle'dan indirilen orijinal veriler
    data/nnunet_raw/             : nnU-Net formatına çevrilmiş veriler
    data/nnunet_preprocessed/    : nnU-Net'in ön işlediği veriler
    data/nnunet_results/         : Eğitim sonuçları ve modeller
    data/inference_output/       : Tahmin sonuçları
    web/                         : Flask web uygulaması
    scripts/                     : Yardımcı Python scriptleri
    logs/                        : Eğitim ve inference logları
    notebooks/                   : Jupyter notebook'lar (analiz için)
    metrics/                     : Metrik sonuçları (CSV, JSON)
    """

    print_header("KLASÖR YAPISI OLUŞTURULUYOR")

    folders = {
        # Ham veri
        "data/raw": "Kaggle'dan indirilen orijinal NIfTI dosyaları",

        # nnU-Net klasör hiyerarşisi
        "data/nnunet_raw/Dataset007_Pancreas/imagesTr": "Eğitim CT görüntüleri (_0000.nii.gz)",
        "data/nnunet_raw/Dataset007_Pancreas/labelsTr": "Eğitim segmentasyon maskeleri (.nii.gz)",
        "data/nnunet_raw/Dataset007_Pancreas/imagesTs": "Test CT görüntüleri (etiketlenmemiş)",

        # nnU-Net işlenmiş veriler (nnU-Net otomatik oluşturur)
        "data/nnunet_preprocessed": "nnU-Net'in ön işlediği veriler (otomatik)",

        # Model çıktıları
        "data/nnunet_results": "Eğitilmiş model ağırlıkları ve checkpoint'ler",

        # Tahmin çıktıları
        "data/inference_output/segmentation_masks": "3D segmentasyon maskeleri (.nii.gz)",
        "data/inference_output/visualizations":     "Renkli segmentasyon görselleştirmeleri",
        "data/inference_output/3d_reconstructions": "3D yüzey dosyaları (.stl, .vtk)",

        # Web uygulaması
        "web/static/css":     "CSS stil dosyaları",
        "web/static/js":      "JavaScript dosyaları",
        "web/static/uploads": "Kullanıcı tarafından yüklenen CT dosyaları",
        "web/static/results": "Web'de gösterilecek sonuç görselleri",
        "web/templates":      "HTML şablonları (Jinja2)",

        # Yardımcı scriptler
        "scripts": "Veri hazırlama, eğitim, inference Python scriptleri",

        # Log dosyaları
        "logs/training":   "nnU-Net eğitim logları",
        "logs/inference":  "Inference logları",
        "logs/web":        "Flask web sunucu logları",

        # Jupyter notebook'lar
        "notebooks": "Veri analizi ve görselleştirme notebook'ları",

        # Metrikler
        "metrics": "Dice, IoU, Accuracy, F1, ROC-AUC sonuçları",

        # Checkpoint
        "checkpoints": "Model checkpoint'leri (opsiyonel yedek)",
    }

    created_count = 0
    for folder_rel, description in folders.items():
        folder_abs = os.path.join(base_path, folder_rel)
        try:
            os.makedirs(folder_abs, exist_ok=True)
            # Açıklama dosyası oluştur
            readme_path = os.path.join(folder_abs, "README.txt")
            if not os.path.exists(readme_path):
                with open(readme_path, "w", encoding="utf-8") as f:
                    f.write(f"Klasör: {folder_rel}\n")
                    f.write(f"Açıklama: {description}\n")
                    f.write(f"Oluşturma tarihi: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            print_success(f"/{folder_rel}")
            created_count += 1
        except Exception as e:
            print_error(f"/{folder_rel} oluşturulamadı: {e}")

    print(f"\n  {Colors.GREEN}{Colors.BOLD}Toplam {created_count} klasör oluşturuldu.{Colors.RESET}")
    return folders


# ============================================================
# BÖLÜM 2: ORTAM DEĞİŞKENLERİ (.env)
# ============================================================
def create_env_file(base_path: str):
    """
    .env dosyasını oluşturur.
    nnU-Net zorunlu ortam değişkenlerini içerir.

    ⚠️ DİKKAT: Bu path'leri kendi sisteminize göre düzenleyin!
    """

    print_header("ORTAM DEĞİŞKENLERİ (.env) OLUŞTURULUYOR")

    env_content = f"""# ============================================================
# Pancreas Cancer Detection - Ortam Değişkenleri
# ============================================================
# ⚠️ BU SATIRI DEĞİŞTİR: Projenin tam yolunu girin
BASE_PATH={base_path.replace(os.sep, "/")}

# ============================================================
# nnU-Net v2 ZORUNLU ORTAM DEĞİŞKENLERİ
# Kaynak: https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/setting_up_paths.md
# ============================================================

# Ham veri klasörü (imagesTr, labelsTr, dataset.json buraya gidecek)
nnUNet_raw={base_path.replace(os.sep, "/")}/data/nnunet_raw

# nnU-Net'in ön işleyeceği veri klasörü
nnUNet_preprocessed={base_path.replace(os.sep, "/")}/data/nnunet_preprocessed

# Eğitim sonuçları ve model ağırlıkları
nnUNet_results={base_path.replace(os.sep, "/")}/data/nnunet_results

# ============================================================
# MODEL AYARLARI
# ============================================================
# Dataset ID (nnU-Net için 3 haneli)
DATASET_ID=007

# Dataset adı
DATASET_NAME=Dataset007_Pancreas

# Konfigürasyon: 2D (GPU kısıtı nedeniyle)
NNUNET_CONFIG=2d

# Trainer
NNUNET_TRAINER=nnUNetTrainer

# Planner
NNUNET_PLANNER=nnUNetPlannerResEncM

# Fold sayısı (0-4, sadece fold 0 kullanıyoruz)
FOLD=0

# ============================================================
# WEB UYGULAMASI
# ============================================================
FLASK_PORT=5000
FLASK_DEBUG=True
PANCREAS_DEBUG=True
MAX_UPLOAD_MB=8192
MAX_UPLOAD_FILES=5000
MODEL_TIMEOUT_SECONDS=1800
MODEL_CHECKPOINT=checkpoint_best.pth

# ============================================================
# GPU AYARLARI
# ============================================================
# ⚠️ BU SATIRI DEĞİŞTİR: Birden fazla GPU varsa hangisini kullanacağınızı belirtin
CUDA_VISIBLE_DEVICES=0
"""

    env_path = os.path.join(base_path, ".env")
    try:
        with open(env_path, "w", encoding="utf-8") as f:
            f.write(env_content)
        print_success(f".env dosyası oluşturuldu: {env_path}")
        print_warning("⚠️  BASE_PATH ve CUDA_VISIBLE_DEVICES değerlerini kontrol edin!")
    except Exception as e:
        print_error(f".env oluşturulamadı: {e}")


# ============================================================
# BÖLÜM 3: GPU KONTROLÜ
# ============================================================
def check_gpu():
    """
    GPU kullanılabilirliğini kontrol eder.
    CUDA, GPU belleği ve cuDNN bilgilerini raporlar.
    """

    print_header("GPU KONTROLÜ")

    try:
        import torch

        # GPU kullanılabilirlik
        cuda_available = torch.cuda.is_available()
        device = "cuda" if cuda_available else "cpu"

        if cuda_available:
            gpu_count   = torch.cuda.device_count()
            gpu_name    = torch.cuda.get_device_name(0)
            gpu_memory  = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            cuda_version = torch.version.cuda
            cudnn_version = torch.backends.cudnn.version()

            print_success(f"CUDA Kullanılabilir: {cuda_available}")
            print_success(f"GPU Sayısı:          {gpu_count}")
            print_success(f"GPU Adı:             {gpu_name}")
            print_success(f"GPU Belleği:         {gpu_memory:.1f} GB")
            print_success(f"CUDA Versiyonu:      {cuda_version}")
            print_success(f"cuDNN Versiyonu:     {cudnn_version}")

            if gpu_memory < 8:
                print_warning("⚠️  GPU belleği 8GB'ın altında!")
                print_warning("    nnU-Net 2D çalışır fakat batch size küçük tutun.")
                print_warning("    Eğitim scriptinde: --batch_size 2 kullanın")
            elif gpu_memory >= 16:
                print_success("GPU belleği 16GB+, optimal eğitim için uygundur.")

        else:
            print_warning("CUDA bulunamadı. CPU modunda çalışılacak.")
            print_warning("CPU eğitimi çok yavaş olacak. Eğitim için GPU önerilir.")
            print_info("Google Colab veya Kaggle Notebook kullanmayı düşünün.")

        print_info(f"\n  Kullanılacak device: {device}")
        return device

    except ImportError:
        print_error("PyTorch kurulu değil!")
        print_info("Kurulum: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")
        return "cpu"


# ============================================================
# BÖLÜM 4: KURULUM DOĞRULAMA
# ============================================================
def verify_installation():
    """
    Kritik kütüphanelerin kurulumunu doğrular.
    """

    print_header("KURULUM DOĞRULAMA")

    packages = {
        "torch":       "PyTorch (Derin Öğrenme)",
        "nibabel":     "NIfTI Görüntü Okuma",
        "SimpleITK":   "CT Ön İşleme",
        "sklearn":     "Metrik Hesaplama",
        "numpy":       "Sayısal İşlemler",
        "scipy":       "Bilimsel Hesaplamalar",
        "matplotlib":  "Görselleştirme",
        "flask":       "Web Uygulaması",
        "tqdm":        "İlerleme Çubuğu",
        "rich":        "Terminal Çıktısı",
    }

    all_ok = True
    for pkg, desc in packages.items():
        try:
            __import__(pkg)
            print_success(f"{pkg:<15} ({desc})")
        except ImportError:
            print_warning(f"{pkg:<15} ({desc}) - KURULU DEĞİL")
            all_ok = False

    # nnU-Net özel kontrol
    try:
        result = subprocess.run(
            ["nnUNetv2_train", "--help"],
            capture_output=True, text=True, timeout=10
        )
        print_success(f"nnUNetv2       (nnU-Net v2 CLI)")
    except (FileNotFoundError, subprocess.TimeoutExpired):
        print_warning("nnUNetv2       - CLI bulunamadı (pip install nnunetv2 gerekli)")
        all_ok = False

    return all_ok


# ============================================================
# BÖLÜM 5: PROJE KONFİGÜRASYONU (config.json)
# ============================================================
def create_config_json(base_path: str):
    """
    Tüm proje ayarlarını içeren merkezi config dosyasını oluşturur.
    Diğer tüm scriptler bu dosyayı okuyacak.
    """

    print_header("PROJE KONFİGÜRASYONU (config.json)")

    config = {
        "project": {
            "name":        "PancreasCancerDetection",
            "version":     "1.0.0",
            "description": "CT görüntülerinden pankreas tümörü tespiti",
            "created_at":  datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "author":      "YOUR_NAME_HERE"  # ⚠️ BU SATIRI DEĞİŞTİR
        },

        "paths": {
            # ⚠️ BU SATIRLARI DEĞİŞTİR: Tüm path'leri kendi sisteminize göre ayarlayın
            "base":               "YOUR_PATH_HERE",
            "data_raw":           "YOUR_PATH_HERE/data/raw",
            "nnunet_raw":         "YOUR_PATH_HERE/data/nnunet_raw",
            "nnunet_preprocessed":"YOUR_PATH_HERE/data/nnunet_preprocessed",
            "nnunet_results":     "YOUR_PATH_HERE/data/nnunet_results",
            "inference_output":   "YOUR_PATH_HERE/data/inference_output",
            "web_uploads":        "YOUR_PATH_HERE/web/static/uploads",
            "web_results":        "YOUR_PATH_HERE/web/static/results",
            "logs":               "YOUR_PATH_HERE/logs",
            "metrics":            "YOUR_PATH_HERE/metrics",
        },

        "dataset": {
            "id":             "007",
            "name":           "Dataset007_Pancreas",
            "num_classes":    3,
            "labels": {
                "0": "background",
                "1": "pancreas",
                "2": "tumor"
            },
            "modality": {
                "0": "CT"
            },
            "file_ending":    ".nii.gz",
            "original_source":"Kaggle MSD Task07"
        },

        "nnunet": {
            "configuration":  "2d",
            "trainer":        "nnUNetTrainer",
            "planner":        "nnUNetPlannerResEncM",
            "fold":           0,
            "max_epochs":     1000,
            "num_processes":  4,
            "device":         "cuda"   # "cpu" eğer GPU yoksa
        },

        "training": {
            "batch_size":     2,       # GPU belleği 8GB altındaysa 2 kullan
            "patch_size":     [512, 512],
            "learning_rate":  1e-3,
            "weight_decay":   3e-5,
        },

        "inference": {
            "step_size":      0.5,
            "use_gaussian":   True,
            "use_mirroring":  True,
        },

        "classification": {
            "tumor_label":     2,
            "min_tumor_voxels":50,     # Gürültüyü filtrelemek için minimum voksel sayısı
        },

        "web": {
            "port":           5000,
            "debug":          True,
            "max_upload_mb":  8192,
            "max_upload_files": 5000,
            "allowed_ext":    [".nii", ".nii.gz"]
        }
    }

    # Gerçek path'leri config'e yaz
    config["paths"]["base"] = base_path.replace(os.sep, "/")
    for key in config["paths"]:
        if key != "base":
            rel = config["paths"][key].replace("YOUR_PATH_HERE", "")
            config["paths"][key] = (base_path + rel).replace(os.sep, "/")

    config_path = os.path.join(base_path, "config.json")
    try:
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=4, ensure_ascii=False)
        print_success(f"config.json oluşturuldu: {config_path}")
    except Exception as e:
        print_error(f"config.json oluşturulamadı: {e}")

    return config


# ============================================================
# BÖLÜM 6: SETUP.BAT (Windows için otomatik kurulum)
# ============================================================
def create_setup_scripts(base_path: str):
    """Windows (.bat) ve Linux/Mac (.sh) kurulum scriptleri oluşturur."""

    print_header("KURULUM SCRİPTLERİ OLUŞTURULUYOR")

    # ---- Windows ----
    bat_content = """@echo off
REM ============================================================
REM Pancreas Cancer Detection - Windows Kurulum Scripti
REM ============================================================

echo [1/6] Python sanal ortami olusturuluyor...
python -m venv venv
if errorlevel 1 goto :error

echo [2/6] Sanal ortam aktif ediliyor...
call venv\\Scripts\\activate.bat

echo [3/6] pip guncelleniyor...
python -m pip install --upgrade pip

echo [4/6] PyTorch kuruluyor (CUDA 11.8)...
REM !! BU SATIRI DEGISTIR: CUDA surumunuze gore degistirin !!
REM CUDA 11.8 icin:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
REM CUDA 12.1 icin (yorumu kaldir):
REM pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
REM CPU icin (GPU yoksa):
REM pip install torch torchvision

echo [5/6] Diger gereksinimler kuruluyor...
pip install -r requirements.txt

echo [6/6] nnU-Net ortam degiskenleri ayarlaniyor...
REM !! BU SATIRI DEGISTIR: Kendi proje yolunuzu girin !!
setx nnUNet_raw      "%~dp0data\\nnunet_raw"
setx nnUNet_preprocessed "%~dp0data\\nnunet_preprocessed"
setx nnUNet_results  "%~dp0data\\nnunet_results"

echo.
echo ============================================================
echo KURULUM TAMAMLANDI!
echo ============================================================
echo Ortam aktif etmek icin: venv\\Scripts\\activate
echo Kurulumu dogrulamak icin: python setup_project.py
echo.
goto :end

:error
echo HATA! Kurulum basarisiz.
exit /b 1

:end
pause
"""

    # ---- Linux/Mac ----
    sh_content = """#!/bin/bash
# ============================================================
# Pancreas Cancer Detection - Linux/Mac Kurulum Scripti
# ============================================================

set -e  # Hata durumunda dur

echo "[1/6] Python sanal ortamı oluşturuluyor..."
python3 -m venv venv

echo "[2/6] Sanal ortam aktif ediliyor..."
source venv/bin/activate

echo "[3/6] pip güncelleniyor..."
pip install --upgrade pip

echo "[4/6] PyTorch kuruluyor (CUDA 11.8)..."
# !! BU SATIRI DEĞİŞTİR: CUDA sürümünüze göre değiştirin !!
# CUDA 11.8:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
# CUDA 12.1 (yorumu kaldır):
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
# CPU (GPU yoksa):
# pip install torch torchvision

echo "[5/6] Diğer gereksinimler kuruluyor..."
pip install -r requirements.txt

echo "[6/6] nnU-Net ortam değişkenleri ayarlanıyor..."
# !! BU SATIRI DEĞİŞTİR: Kendi proje yolunuzu girin !!
export nnUNet_raw="$(pwd)/data/nnunet_raw"
export nnUNet_preprocessed="$(pwd)/data/nnunet_preprocessed"
export nnUNet_results="$(pwd)/data/nnunet_results"

# .bashrc veya .zshrc'ye ekle (kalıcı)
echo "export nnUNet_raw='$(pwd)/data/nnunet_raw'" >> ~/.bashrc
echo "export nnUNet_preprocessed='$(pwd)/data/nnunet_preprocessed'" >> ~/.bashrc
echo "export nnUNet_results='$(pwd)/data/nnunet_results'" >> ~/.bashrc

echo ""
echo "============================================================"
echo "KURULUM TAMAMLANDI!"
echo "============================================================"
echo "Ortam aktif etmek için: source venv/bin/activate"
echo "Kurulumu doğrulamak için: python setup_project.py"
"""

    # Windows .bat
    bat_path = os.path.join(base_path, "setup.bat")
    try:
        with open(bat_path, "w", encoding="utf-8") as f:
            f.write(bat_content)
        print_success(f"setup.bat oluşturuldu (Windows)")
    except Exception as e:
        print_error(f"setup.bat oluşturulamadı: {e}")

    # Linux/Mac .sh
    sh_path = os.path.join(base_path, "setup.sh")
    try:
        with open(sh_path, "w", encoding="utf-8") as f:
            f.write(sh_content)
        # Linux'ta executable yap
        if sys.platform != "win32":
            os.chmod(sh_path, 0o755)
        print_success(f"setup.sh oluşturuldu (Linux/Mac)")
    except Exception as e:
        print_error(f"setup.sh oluşturulamadı: {e}")


# ============================================================
# BÖLÜM 7: PROJE ÖZET RAPORU
# ============================================================
def print_final_summary(base_path: str, device: str):
    """Kurulum özet raporunu yazdırır."""

    print_header("PROJE KURULUMU TAMAMLANDI - ÖZET")

    print(f"""
  {Colors.BOLD}[>>] Proje Yolu:{Colors.RESET}
     {base_path}

  {Colors.BOLD}[>>] Onemli Klasorler:{Colors.RESET}
     data/raw/                    -> Kaggle'dan indirilen veriler
     data/nnunet_raw/             -> nnU-Net formatina cevrilmis veriler
     data/nnunet_preprocessed/    -> nnU-Net on isleme ciktisi
     data/nnunet_results/         -> Egitilmis modeller
     data/inference_output/       -> Segmentasyon sonuclari
     web/                         -> Flask web uygulamasi

  {Colors.BOLD}[>>] nnU-Net Ortam Degiskenleri:{Colors.RESET}
     {Colors.YELLOW}Windows PowerShell:{Colors.RESET}
       $env:nnUNet_raw = "{base_path}/data/nnunet_raw"
       $env:nnUNet_preprocessed = "{base_path}/data/nnunet_preprocessed"
       $env:nnUNet_results = "{base_path}/data/nnunet_results"

     {Colors.YELLOW}Linux/Mac (.bashrc):{Colors.RESET}
       export nnUNet_raw="{base_path}/data/nnunet_raw"
       export nnUNet_preprocessed="{base_path}/data/nnunet_preprocessed"
       export nnUNet_results="{base_path}/data/nnunet_results"

  {Colors.BOLD}[>>] GPU Durumu:{Colors.RESET}
     {Colors.GREEN if device == "cuda" else Colors.YELLOW}Device: {device.upper()}{Colors.RESET}

  {Colors.BOLD}[>>] Sonraki Adimlar:{Colors.RESET}
     {Colors.CYAN}ADIM 1:{Colors.RESET} setup.bat (Windows) veya setup.sh (Linux) calistirin
     {Colors.CYAN}ADIM 2:{Colors.RESET} Kaggle'dan veriyi data/raw/ klasorune indirin
     {Colors.CYAN}ADIM 3:{Colors.RESET} ADIM2_prepare_dataset.py scriptini calistirin
""")


# ============================================================
# MAIN
# ============================================================
def main():
    """Ana kurulum fonksiyonu."""

    print(f"""
{Colors.BOLD}{Colors.CYAN}
==============================================================
  PANKREAS KANSERI TESPIT PROJESI - KURULUM
  nnU-Net v2 2D Segmentasyon + Kural Tabanli Siniflandirma
==============================================================
{Colors.RESET}
""")

    # ⚠️ BU SATIRI DEĞİŞTİR: Projenin tam yolunu buraya girin
    BASE_PATH = os.path.dirname(os.path.abspath(__file__))
    # Alternatif: BASE_PATH = "YOUR_PATH_HERE"

    print_info(f"Proje yolu: {BASE_PATH}")

    try:
        # 1. Klasör yapısını oluştur
        create_project_structure(BASE_PATH)

        # 2. Ortam değişkenlerini oluştur
        create_env_file(BASE_PATH)

        # 3. Config dosyasını oluştur
        config = create_config_json(BASE_PATH)

        # 4. Kurulum scriptlerini oluştur
        create_setup_scripts(BASE_PATH)

        # 5. GPU kontrol et
        device = check_gpu()

        # 6. Kurulum doğrula
        installation_ok = verify_installation()

        # 7. Özet rapor
        print_final_summary(BASE_PATH, device)

        if not installation_ok:
            print_warning("Bazı paketler eksik! setup.bat veya setup.sh çalıştırın.")
            return 1

        print_success("Kurulum başarıyla tamamlandı!")
        return 0

    except Exception as e:
        print_error(f"Beklenmeyen hata: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
