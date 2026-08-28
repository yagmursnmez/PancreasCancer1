"""
============================================================
ADIM 10: MODEL KAYDETME & PRODUCTION INFERENCE SCRIPTI
============================================================
Bu script:
1. Eğitilmiş nnU-Net modelini export eder
2. Model metadata'sını kaydeder
3. Production-ready inference pipeline oluşturur
4. Command-line inference arayüzü sağlar

Kullanım:
    # Model bilgilerini göster
    python scripts/save_model.py --info

    # Modeli export et (zip/tar)
    python scripts/save_model.py --export

    # Tek CT dosyasını çalıştır
    python scripts/save_model.py --predict path/to/ct.nii.gz
============================================================
"""

import os
import sys
import json
import shutil
import hashlib
import argparse
import tarfile
import zipfile
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict

# ============================================================
# PATH AYARLARI
# ⚠️ BU SATIRI DEĞİŞTİR
# ============================================================
BASE_PATH = Path(__file__).parent.parent

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
# BÖLÜM 1: MODEL BİLGİLERİ
# ============================================================
def get_model_info() -> Dict:
    """
    Eğitilmiş model hakkında detaylı bilgi toplar.

    nnU-Net model yapısı:
    nnunet_results/
    └── Dataset007_Pancreas/
        └── nnUNetTrainer__nnUNetPlans__2d/
            ├── fold_0/
            │   ├── checkpoint_best.pth     (En iyi model ağırlıkları)
            │   ├── checkpoint_final.pth    (Son epoch ağırlıkları)
            │   ├── training_log_*.txt      (Eğitim kayıtları)
            │   └── progress.png           (Loss/Dice grafiği)
            └── plans.json                 (Model konfigürasyonu)
    """
    header("MODEL BİLGİLERİ")

    # Ortam değişkenleri
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

    results_dir = Path(os.environ.get(
        "nnUNet_results",
        str(BASE_PATH / "data" / "nnunet_results")
    ))

    model_dir = results_dir / "Dataset007_Pancreas" / "nnUNetTrainer__nnUNetPlans__2d"
    fold_dir  = model_dir / "fold_0"

    model_info = {
        "dataset":       "Dataset007_Pancreas",
        "configuration": "2d",
        "trainer":       "nnUNetTrainer",
        "fold":          0,
        "model_dir":     str(model_dir),
        "fold_dir":      str(fold_dir),
        "checkpoints":   {},
        "training_completed": False,
        "last_epoch":    None,
        "best_dice":     None,
        "model_size_mb": 0,
    }

    if not model_dir.exists():
        warn(f"Model klasoru bulunamadi: {model_dir}")
        warn("Once ADIM 4 (train_model.py) tamamlanmali!")
        return model_info

    ok(f"Model klasoru: {model_dir}")

    # Checkpoint dosyaları
    for ckpt_name in ["checkpoint_best.pth", "checkpoint_final.pth",
                      "checkpoint_latest.pth"]:
        ckpt_path = fold_dir / ckpt_name
        if ckpt_path.exists():
            size_mb = ckpt_path.stat().st_size / (1024**2)
            # MD5 hash
            md5 = _compute_md5(ckpt_path)
            model_info["checkpoints"][ckpt_name] = {
                "path":    str(ckpt_path),
                "size_mb": round(size_mb, 2),
                "md5":     md5,
            }
            model_info["model_size_mb"] += size_mb
            ok(f"{ckpt_name}: {size_mb:.1f} MB | MD5: {md5[:12]}...")

    if "checkpoint_final.pth" in model_info["checkpoints"]:
        model_info["training_completed"] = True

    # Plans dosyası
    plans_path = model_dir / "plans.json"
    if plans_path.exists():
        with open(plans_path) as f:
            plans = json.load(f)
        model_info["plans"] = plans
        ok(f"Plans dosyasi bulundu")

    # Eğitim logu
    log_files = list(fold_dir.glob("training_log_*.txt")) if fold_dir.exists() else []
    if log_files:
        latest_log = max(log_files, key=lambda p: p.stat().st_mtime)
        model_info["training_log"] = str(latest_log)
        ok(f"Egitim logu: {latest_log.name}")

        # Son Dice ve epoch
        try:
            with open(latest_log) as f:
                lines = f.readlines()

            for line in reversed(lines):
                if "Pseudo Dice" in line:
                    try:
                        dice = float(line.strip().split()[-1])
                        model_info["best_dice"] = dice
                    except:
                        pass
                    break
        except Exception as e:
            warn(f"Log okuma hatasi: {e}")

    # Validation metrics varsa oku
    metrics_dir = BASE_PATH / "metrics"
    val_reports = sorted(metrics_dir.glob("validation_report_*.json"), reverse=True)
    if val_reports:
        with open(val_reports[0]) as f:
            val = json.load(f)
        model_info["validation"] = {
            "pancreas_dice": val.get("segmentation", {}).get("pancreas", {}).get("dice_mean"),
            "tumor_dice":    val.get("segmentation", {}).get("tumor", {}).get("dice_mean"),
            "f1_score":      val.get("classification", {}).get("f1_score"),
            "accuracy":      val.get("classification", {}).get("accuracy"),
        }

    return model_info


def _compute_md5(path: Path) -> str:
    md5 = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            md5.update(chunk)
    return md5.hexdigest()


# ============================================================
# BÖLÜM 2: MODEL EXPORT
# ============================================================
def export_model(output_path: Optional[Path] = None) -> Path:
    """
    Modeli deployment için paketler.

    Export yapısı:
    pancreas_nnunet_model.tar.gz/
    ├── model/              (nnU-Net checkpoint'leri)
    ├── metadata.json       (Model bilgileri)
    ├── requirements.txt    (Bağımlılıklar)
    └── README.md           (Kullanım talimatları)
    """
    header("MODEL EXPORT EDİLİYOR")

    model_info = get_model_info()

    timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
    export_name = f"pancreas_nnunet_2d_{timestamp}"

    if output_path is None:
        output_path = BASE_PATH / "models" / "exported"
    output_path.mkdir(parents=True, exist_ok=True)

    # Geçici export klasörü
    tmp_dir = output_path / export_name
    tmp_dir.mkdir(exist_ok=True)

    # 1. Model dosyalarını kopyala
    model_dst = tmp_dir / "model"
    model_dir = Path(model_info["model_dir"])

    if model_dir.exists():
        shutil.copytree(str(model_dir), str(model_dst),
                       ignore=shutil.ignore_patterns("*.log"))
        ok(f"Model kopyalandi: {model_dst}")
    else:
        warn("Model dizini bulunamadi! Sadece metadata kaydedilecek.")
        model_dst.mkdir(exist_ok=True)

    # 2. Metadata
    metadata = {
        "export_timestamp":    timestamp,
        "model":               model_info,
        "dataset":             "MSD Task07 Pancreas",
        "labels":              {0: "background", 1: "pancreas", 2: "tumor"},
        "configuration":       "2d",
        "framework":           "nnU-Net v2",
        "usage": {
            "inference_cmd":   "nnUNetv2_predict -i INPUT -o OUTPUT -d 007 -c 2d -f 0",
            "python_api":      "python scripts/inference.py --input INPUT_DIR",
            "web_interface":   "python web/app.py → http://localhost:5000",
        },
        "classification": {
            "rule":    "tumor_voxels > 50 → 'Tumor Var'",
            "label":   2,
            "threshold_voxels": 50
        }
    }

    metadata_path = tmp_dir / "metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4, ensure_ascii=False)
    ok(f"Metadata kaydedildi: {metadata_path.name}")

    # 3. README
    readme_content = f"""# PancreasAI — nnU-Net v2 2D Model

## Model Bilgisi
- **Dataset**: MSD Task07 Pancreas
- **Konfigürasyon**: 2D
- **Framework**: nnU-Net v2
- **Export Tarihi**: {timestamp}

## Inference

### Komut Satırı
```bash
nnUNetv2_predict \\
    -i /path/to/input_cts/ \\
    -o /path/to/output_masks/ \\
    -d 007 -c 2d -f 0
```

### Python API
```python
from scripts.inference import process_single_ct
from pathlib import Path

result = process_single_ct(Path("ct.nii.gz"))
print(result["prediction"])  # "Tumor Var" veya "Tumor Yok"
```

### Web Arayüzü
```bash
python web/app.py
# Tarayıcı: http://localhost:5000
```

## Etiketler
- 0: Arka Plan
- 1: Pankreas
- 2: Tümör

## Sınıflandırma Kuralı
```python
if (mask == 2).sum() > 50:
    karar = "Tumor Var"
else:
    karar = "Tumor Yok"
```

## Uyarı
Bu sistem araştırma amaçlıdır. Klinik karar verme için kullanılamaz.
"""
    readme_path = tmp_dir / "README.md"
    readme_path.write_text(readme_content, encoding="utf-8")
    ok("README.md olusturuldu")

    # 4. requirements.txt
    req_path = BASE_PATH / "requirements.txt"
    if req_path.exists():
        shutil.copy2(str(req_path), str(tmp_dir / "requirements.txt"))

    # 5. Sıkıştır
    tar_path = output_path / f"{export_name}.tar.gz"
    with tarfile.open(str(tar_path), "w:gz") as tar:
        tar.add(str(tmp_dir), arcname=export_name)

    size_mb = tar_path.stat().st_size / (1024**2)
    ok(f"Export tamamlandi: {tar_path.name} ({size_mb:.1f} MB)")

    # Geçici klasörü temizle
    shutil.rmtree(str(tmp_dir), ignore_errors=True)

    return tar_path


# ============================================================
# BÖLÜM 3: PRODUCTION INFERENCE
# ============================================================
def production_predict(
    ct_path: Path,
    output_dir: Optional[Path] = None,
    verbose: bool = True
) -> Dict:
    """
    Production-ready inference pipeline.

    Args:
        ct_path:    CT NIfTI dosyası
        output_dir: Çıktı klasörü
        verbose:    Ayrıntılı çıktı

    Returns:
        Sonuç sözlüğü
    """
    if verbose:
        header(f"PRODUCTION INFERENCE: {ct_path.name}")

    if output_dir is None:
        output_dir = BASE_PATH / "data" / "inference_output" / "production"
    output_dir.mkdir(parents=True, exist_ok=True)

    # inference.py'daki fonksiyonu kullan
    try:
        from inference import process_single_ct

        # Maske path'i belirle
        case_id   = ct_path.stem.replace("_0000", "")
        mask_path = (BASE_PATH / "data" / "inference_output" /
                    "segmentation_masks" / f"{case_id}.nii.gz")

        result = process_single_ct(ct_path, mask_path)

        if verbose and result:
            color = RED if result.get("has_tumor") else GREEN
            print(f"""
  {BOLD}{'='*50}
  PREDICTION: {color}{result['prediction']}{RESET}
  {'='*50}
  Tumor voksel:    {result.get('tumor_voxels', 0):,}
  Pankreas voksel: {result.get('pancreas_voxels', 0):,}
  Guven:           {result.get('confidence_level', '?')}
  Gorsel:          {result.get('visualization', 'N/A')}
  {'='*50}
""")

        return result

    except ImportError:
        fail("inference.py modulu bulunamadi!")
        return {}
    except Exception as e:
        fail(f"Inference hatasi: {e}")
        import traceback
        traceback.print_exc()
        return {}


# ============================================================
# BÖLÜM 4: MODEL METADATA KAYDET
# ============================================================
def save_model_metadata(model_info: Dict) -> Path:
    """Model metadata'sını JSON olarak kaydeder."""
    metrics_dir = BASE_PATH / "metrics"
    metrics_dir.mkdir(exist_ok=True)

    metadata_path = metrics_dir / "model_metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(model_info, f, indent=4, ensure_ascii=False,
                  default=str)
    ok(f"Model metadata kaydedildi: {metadata_path}")
    return metadata_path


# ============================================================
# MAIN
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description="Model kaydetme, export ve production inference"
    )
    parser.add_argument("--info",   action="store_true", help="Model bilgilerini göster")
    parser.add_argument("--export", action="store_true", help="Modeli export et")
    parser.add_argument("--predict", type=str,           help="CT dosyasını tahmin et")
    args = parser.parse_args()

    print(f"""
{BOLD}{CYAN}
==============================================================
  ADIM 10: MODEL KAYDETME & PRODUCTION INFERENCE
==============================================================
{RESET}
""")

    if args.info or (not args.export and not args.predict):
        # Model bilgilerini göster
        model_info = get_model_info()
        metadata_path = save_model_metadata(model_info)

        # Özet tablo
        header("MODEL OZETI")
        print(f"""
  {'Model Klasoru':<30} {model_info['model_dir']}
  {'Egitim Tamamlandi':<30} {'Evet' if model_info['training_completed'] else 'Hayir (Devam Ediyor)'}
  {'Toplam Model Boyutu':<30} {model_info['model_size_mb']:.1f} MB
  {'En Iyi Dice':<30} {model_info.get('best_dice') or 'N/A'}
""")

        val = model_info.get("validation")
        if val:
            print(f"""
  Validasyon Sonuclari:
  {'Pankreas Dice':<30} {val.get('pancreas_dice') or 'N/A'}
  {'Tumor Dice':<30} {val.get('tumor_dice') or 'N/A'}
  {'F1 Score':<30} {val.get('f1_score') or 'N/A'}
  {'Accuracy':<30} {val.get('accuracy') or 'N/A'}
""")

        print(f"""
  {BOLD}Kullanim Komutlari:{RESET}
    {CYAN}# Web arayuzu baslat:{RESET}
    python web/app.py

    {CYAN}# Tek CT analiz et:{RESET}
    python scripts/save_model.py --predict path/to/ct.nii.gz

    {CYAN}# Toplu inference:{RESET}
    python scripts/inference.py --input data/nnunet_raw/Dataset007_Pancreas/imagesTs

    {CYAN}# Model export et:{RESET}
    python scripts/save_model.py --export

    {CYAN}# nnU-Net direkt:{RESET}
    nnUNetv2_predict -i INPUT -o OUTPUT -d 007 -c 2d -f 0
""")

    if args.export:
        export_path = export_model()
        ok(f"Export hazir: {export_path}")

    if args.predict:
        ct_path = Path(args.predict)
        if not ct_path.exists():
            fail(f"Dosya bulunamadi: {ct_path}")
            return 1
        result = production_predict(ct_path, verbose=True)

    header("ADIM 10 TAMAMLANDI")
    ok("Proje tum adimlari ile hazir!")
    print(f"""
  {BOLD}PROJE OZETI:{RESET}
  {'-'*50}
  ADIM 1  ✓ Proje kurulumu
  ADIM 2  ✓ Dataset hazırlama (MSD Task07 → nnU-Net format)
  ADIM 3  ✓ Preprocessing (nnU-Net plan & preprocess)
  ADIM 4  ✓ Model egitimi (nnU-Net v2 2D, fold 0)
  ADIM 5  ✓ Validasyon (Dice, IoU, F1, ROC-AUC)
  ADIM 6  ✓ Inference (segmentasyon maskesi olusturma)
  ADIM 7  ✓ Siniflandirma (Tumor Var / Tumor Yok)
  ADIM 8  ✓ 3D Rekonstruksiyon (STL, PNG, HTML)
  ADIM 9  ✓ Web arayuzu (Flask)
  ADIM 10 ✓ Model kaydetme & production inference
  {'-'*50}

  {BOLD}WEB ARAYUZUNU BASLATMAK ICIN:{RESET}
    {CYAN}python web/app.py{RESET}
    Tarayici: http://localhost:5000
""")
    return 0


if __name__ == "__main__":
    sys.exit(main())
