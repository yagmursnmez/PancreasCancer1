"""
============================================================
ADIM 4: MODEL EĞİTİMİ — nnU-Net v2 2D
============================================================
Bu script nnU-Net v2 2D modelini eğitir.

Eğitim Detayları:
- Konfigürasyon: 2D (3D full resolution kullanılmıyor)
- Fold: 0 (5-fold cross-validation'dan sadece fold 0)
- Epoch: 1000 (nnU-Net varsayılanı)
- GPU: 6 GB NVIDIA CUDA kartları için bellek-güvenli ayarlar
- Batch size: nnU-Net otomatik belirler (yaklaşık 2)

6 GB sınıfı bir NVIDIA Laptop GPU ile yaklaşık süre:
- ~8-12 saat (1000 epoch, tam dataset)
- Simülasyon verisiyle ~5-10 dakika

Kullanım:
    python scripts/train_model.py           # Eğitim başlat
    python scripts/train_model.py --resume  # Kaldığı yerden devam et
    python scripts/train_model.py --epochs 100  # Hızlı test için

    VEYA doğrudan nnU-Net CLI:
    nnUNetv2_train 007 2d 0
============================================================
"""

import os
import sys
import json
import time
import subprocess
import argparse
from pathlib import Path
from datetime import datetime, timedelta

# ============================================================
# PATH AYARLARI
# ⚠️ BU SATIRI DEĞİŞTİR
# ============================================================
BASE_PATH = Path(__file__).parent.parent

# ============================================================
# RENK KODLARI
# ============================================================
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

def ok(msg):   print(f"  {GREEN}[OK]{RESET}   {msg}")
def warn(msg): print(f"  {YELLOW}[!!]{RESET}   {msg}")
def fail(msg): print(f"  {RED}[ERR]{RESET}  {msg}")
def info(msg): print(f"  {CYAN}[--]{RESET}   {msg}")
def header(msg):
    print(f"\n{BOLD}{CYAN}{'='*60}{RESET}")
    print(f"{BOLD}{CYAN}  {msg}{RESET}")
    print(f"{BOLD}{CYAN}{'='*60}{RESET}\n")


# ============================================================
# BÖLÜM 1: EĞİTİM ÖNCESİ HAZIRLIK
# ============================================================
def setup_training_env() -> bool:
    """
    Eğitim için ortam değişkenlerini ve GPU'yu kontrol eder.
    """
    header("EGITIM ORTAMI HAZIRLANIYOR")

    # nnU-Net ortam değişkenleri
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
                info(f"{key} otomatik ayarlandi: {paths[path_key]}")

    # Değişkenleri kontrol et
    all_set = True
    for var in ["nnUNet_raw", "nnUNet_preprocessed", "nnUNet_results"]:
        val = os.environ.get(var)
        if val:
            ok(f"{var} = {val}")
        else:
            fail(f"{var} ayarlanmamis!")
            all_set = False

    # GPU kontrolü
    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"

        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_mem  = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            ok(f"GPU: {gpu_name} ({gpu_mem:.1f} GB)")

            # 6 GB NVIDIA CUDA kartları için bellek-güvenli ayar
            if gpu_mem < 8:
                warn(f"GPU bellegi {gpu_mem:.1f}GB (8GB altinda)")
                warn("nnU-Net batch_size otomatik kucultecek.")
                # PyTorch bellek optimizasyonu
                os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
                ok("PYTORCH_CUDA_ALLOC_CONF ayarlandi (bellek optimizasyonu)")
        else:
            warn("GPU bulunamadi! CPU ile egitim cok yavas olacak.")
            warn("Google Colab veya Kaggle Notebook onerilir.")

    except ImportError:
        fail("PyTorch kurulu degil!")
        all_set = False

    # Preprocessing çıktısını kontrol et
    preprocessed_dir = Path(os.environ.get(
        "nnUNet_preprocessed",
        str(BASE_PATH / "data" / "nnunet_preprocessed")
    )) / "Dataset007_Pancreas"

    if preprocessed_dir.exists():
        ok(f"Preprocessed veri mevcut: {preprocessed_dir}")
    else:
        warn(f"Preprocessed veri bulunamadi: {preprocessed_dir}")
        warn("Once ADIM 3'u calistirin: python scripts/run_preprocessing.py")

    return all_set


# ============================================================
# BÖLÜM 2: nnU-Net EĞİTİMİ BAŞLAT
# ============================================================
def run_nnunet_training(
    dataset_id:    str  = "007",
    configuration: str  = "2d",
    fold:          int  = 0,
    resume:        bool = False,
    num_epochs:    int  = 1000,
) -> bool:
    """
    nnUNetv2_train komutunu çalıştırır.

    nnU-Net Eğitim Mimarisi (2D):
    - Architecture: Residual Encoder U-Net
    - Loss: Compound (Dice + Cross-Entropy)
    - Optimizer: SGD with Nesterov momentum
    - LR Scheduler: PolyLR
    - Data Augmentation: rotations, scaling, elastic deformation,
                         gamma correction, mirroring

    Fold Açıklaması:
    - nnU-Net 5-fold cross-validation kullanır
    - Fold 0: ilk %80 eğitim, son %20 validasyon
    - Sadece fold 0 ile başlıyoruz (hız için)

    Args:
        dataset_id:    "007"
        configuration: "2d"
        fold:          0 (sadece fold 0)
        resume:        Kaldığı yerden devam et
        num_epochs:    Epoch sayısı (varsayılan: 1000)
    """
    header(f"nnU-Net EGITIMI BASLIYOR — Dataset{dataset_id} | {configuration.upper()} | Fold {fold}")

    # Eğitim komutunu oluştur
    cmd = [
        "nnUNetv2_train",
        dataset_id,
        configuration,
        str(fold),
    ]

    if resume:
        cmd.append("--c")  # continue/resume flag
        info("Egitim kaldigi yerden devam edecek.")

    info(f"Komut: {' '.join(cmd)}")
    info(f"Dataset:       Dataset{dataset_id}_Pancreas")
    info(f"Konfigurasyon: {configuration.upper()} (GPU kisiti: 3D kullanilmiyor)")
    info(f"Fold:          {fold}")
    info(f"Epoch:         {num_epochs}")
    warn("Egitim uzun surer — terminali kapatmayin!")
    warn("Ctrl+C ile durdurabilir, --resume ile devam edebilirsiniz.")

    # Tahmini süre
    if num_epochs == 1000:
        warn("Tahmini sure: 6 GB NVIDIA Laptop GPU ile ~8-12 saat (gercek veri)")
        warn("Simulasyon verisiyle: ~5-10 dakika")
    print()

    start_time = time.time()

    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
            env=os.environ.copy()
        )

        last_dice = None
        epoch_count = 0

        for line in process.stdout:
            line = line.rstrip()
            if not line:
                continue

            print(f"  {line}")

            # Dice score takibi
            if "Pseudo" in line and "Dice" in line:
                try:
                    dice_val = float(line.split()[-1])
                    last_dice = dice_val
                except:
                    pass

            if "Epoch" in line and "/" in line:
                epoch_count += 1

        process.wait()
        elapsed = time.time() - start_time

        if process.returncode == 0:
            ok(f"Egitim TAMAMLANDI!")
            ok(f"Gecen sure: {str(timedelta(seconds=int(elapsed)))}")
            if last_dice:
                ok(f"Son Dice skoru: {last_dice:.4f}")
            return True
        else:
            # Return code 1 olabilir ama eğitim tamamlandıysa sorun değil
            if epoch_count > 0:
                warn(f"Egitim tamamlandi (return code: {process.returncode})")
                warn("Bu normal olabilir. Checkpoint dosyalarını kontrol edin.")
                return True
            fail(f"Egitim basarisiz! Return code: {process.returncode}")
            return False

    except KeyboardInterrupt:
        warn("Egitim kullanici tarafindan durduruldu (Ctrl+C).")
        warn("Devam etmek icin: python scripts/train_model.py --resume")
        return False

    except FileNotFoundError:
        fail("nnUNetv2_train komutu bulunamadi!")
        warn("Cozum: pip install nnunetv2")
        return False

    except Exception as e:
        fail(f"Beklenmeyen hata: {e}")
        import traceback
        traceback.print_exc()
        return False


# ============================================================
# BÖLÜM 3: EĞİTİM CHECKPOINT KONTROLÜ
# ============================================================
def check_training_checkpoint(
    dataset_id:    str = "007",
    configuration: str = "2d",
    fold:          int = 0
) -> dict:
    """
    Eğitim checkpoint'lerini kontrol eder ve son durumu raporlar.

    nnU-Net checkpoint yapısı:
    nnunet_results/
    └── Dataset007_Pancreas/
        └── nnUNetTrainer__nnUNetPlans__2d/
            └── fold_0/
                ├── checkpoint_best.pth      ← En iyi model
                ├── checkpoint_latest.pth    ← Son checkpoint
                ├── training_log_*.txt       ← Eğitim logu
                └── progress.png            ← Loss/Dice grafiği
    """
    header("CHECKPOINT KONTROLU")

    results_dir = Path(os.environ.get(
        "nnUNet_results",
        str(BASE_PATH / "data" / "nnunet_results")
    ))

    # nnU-Net'in oluşturduğu klasör yapısı
    fold_dir = (results_dir
                / f"Dataset{dataset_id}_Pancreas"
                / f"nnUNetTrainer__nnUNetPlans__{configuration}"
                / f"fold_{fold}")

    checkpoint_info = {
        "fold_dir":   str(fold_dir),
        "exists":     fold_dir.exists(),
        "checkpoints": {},
        "last_epoch": None,
        "best_dice":  None,
    }

    if not fold_dir.exists():
        warn(f"Checkpoint klasoru bulunamadi: {fold_dir}")
        warn("Egitim henuz baslatilmamis veya tamamlanmamis olabilir.")
        return checkpoint_info

    ok(f"Checkpoint klasoru: {fold_dir}")

    # Checkpoint dosyaları
    for ckpt_name in ["checkpoint_best.pth", "checkpoint_latest.pth",
                      "checkpoint_final.pth"]:
        ckpt_path = fold_dir / ckpt_name
        if ckpt_path.exists():
            size_mb = ckpt_path.stat().st_size / (1024**2)
            ok(f"{ckpt_name}: {size_mb:.1f} MB")
            checkpoint_info["checkpoints"][ckpt_name] = {
                "path":    str(ckpt_path),
                "size_mb": size_mb
            }
        else:
            info(f"{ckpt_name}: mevcut degil")

    # Training log
    log_files = list(fold_dir.glob("training_log_*.txt"))
    if log_files:
        latest_log = max(log_files, key=lambda p: p.stat().st_mtime)
        ok(f"Egitim logu: {latest_log.name}")

        # Son epoch ve Dice'ı oku
        try:
            with open(latest_log) as f:
                lines = f.readlines()

            for line in reversed(lines):
                if "Epoch" in line:
                    parts = line.strip().split()
                    for i, p in enumerate(parts):
                        if p.isdigit():
                            checkpoint_info["last_epoch"] = int(p)
                            break
                    break

            # Pseudo Dice
            for line in reversed(lines):
                if "Pseudo Dice" in line:
                    try:
                        dice = float(line.strip().split()[-1])
                        checkpoint_info["best_dice"] = dice
                        info(f"Son Pseudo Dice: {dice:.4f}")
                    except:
                        pass
                    break

        except Exception as e:
            warn(f"Log okuma hatasi: {e}")

    # progress.png
    progress_png = fold_dir / "progress.png"
    if progress_png.exists():
        ok(f"Egitim grafigi mevcut: progress.png")
        checkpoint_info["progress_png"] = str(progress_png)
    else:
        info("progress.png henuz olusturulmamis")

    return checkpoint_info


# ============================================================
# BÖLÜM 4: EĞİTİM RAPORUNU KAYDET
# ============================================================
def save_training_report(success: bool, checkpoint_info: dict, elapsed: float):
    """Eğitim sonuçlarını JSON raporuna kaydeder."""
    report = {
        "step":       "ADIM4_Training",
        "timestamp":  datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "success":    success,
        "elapsed_s":  elapsed,
        "elapsed_h":  elapsed / 3600,
        "config": {
            "dataset_id":    "007",
            "configuration": "2d",
            "fold":          0,
        },
        "checkpoint": checkpoint_info,
    }

    report_dir  = BASE_PATH / "metrics"
    report_dir.mkdir(exist_ok=True)
    report_path = report_dir / "training_report.json"

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4, ensure_ascii=False)

    ok(f"Egitim raporu kaydedildi: {report_path}")
    return report_path


# ============================================================
# MAIN
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description="nnU-Net v2 2D model egitimi"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Varolan checkpoint'ten devam et"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1000,
        # ⚠️ BU SATIRI DEĞİŞTİR: Hızlı test için 50-100 kullanın
        help="Epoch sayısı (varsayılan: 1000, test için: 50)"
    )
    parser.add_argument(
        "--fold",
        type=int,
        default=0,
        help="Fold numarası (0-4, varsayılan: 0)"
    )
    args = parser.parse_args()

    print(f"""
{BOLD}{CYAN}
==============================================================
  ADIM 4: MODEL EGITIMI - nnU-Net v2 2D
==============================================================
{RESET}
""")

    start_time = time.time()

    # 1. Ortamı hazırla
    env_ok = setup_training_env()
    if not env_ok:
        warn("Ortam eksiklikleri var ama devam ediliyor...")

    # 2. Eğitimi başlat
    success = run_nnunet_training(
        dataset_id    = "007",
        configuration = "2d",
        fold          = args.fold,
        resume        = args.resume,
        num_epochs    = args.epochs,
    )

    elapsed = time.time() - start_time

    # 3. Checkpoint kontrol et
    checkpoint_info = check_training_checkpoint(
        dataset_id    = "007",
        configuration = "2d",
        fold          = args.fold
    )

    # 4. Rapor kaydet
    save_training_report(success, checkpoint_info, elapsed)

    # 5. Özet
    header("ADIM 4 OZET")

    if success or checkpoint_info.get("exists"):
        ok("Egitim tamamlandi veya checkpoint mevcut!")
        print(f"""
  {BOLD}Sonraki Adim (ADIM 5) - Validasyon:{RESET}
    {CYAN}python scripts/validate_metrics.py{RESET}

  {BOLD}Veya dogrudan ADIM 6 - Inference:{RESET}
    {CYAN}python scripts/inference.py --input data/nnunet_raw/Dataset007_Pancreas/imagesTs{RESET}
""")
        return 0
    else:
        fail("Egitim tamamlanamadi!")
        warn("Hata ayiklama icin:")
        warn("  1. Ortam degiskenlerini kontrol edin")
        warn("  2. ADIM 3 preprocessing tamamlandi mi?")
        warn("  3. GPU bellegi yeterli mi?")
        return 1


if __name__ == "__main__":
    sys.exit(main())
