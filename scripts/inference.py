"""
============================================================
ADIM 6 + 7: INFERENCE & CLASSIFICATION
============================================================
Bu script:
1. Eğitilmiş nnU-Net modelini yükler
2. Yeni CT görüntüsü üzerinde segmentasyon yapar
3. Segmentasyon maskesini analiz ederek tümör kararı verir:
   if (mask == 2).any(): "Tümör Var"
   else:                 "Tümör Yok"
4. Segmentasyon görsellerini (PNG) oluşturur
5. Sonuçları JSON olarak kaydeder

Kullanım:
    python scripts/inference.py --input data/nnunet_raw/Dataset007_Pancreas/imagesTs
    python scripts/inference.py --input path/to/ct.nii.gz --single_file

    VEYA doğrudan nnU-Net CLI:
    nnUNetv2_predict -i INPUT_DIR -o OUTPUT_DIR -d 007 -c 2d -f 0
============================================================
"""

import os
import sys
import json
import time
import shutil
import argparse
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List, Tuple

import numpy as np

# ============================================================
# PATH AYARLARI
# ⚠️ BU SATIRI DEĞİŞTİR
# ============================================================
BASE_PATH = Path(__file__).parent.parent

# Inference çıktı klasörü
INFERENCE_OUTPUT   = BASE_PATH / "data" / "inference_output"
SEG_MASKS_DIR      = INFERENCE_OUTPUT / "segmentation_masks"
VIZ_DIR            = INFERENCE_OUTPUT / "visualizations"
RECON_3D_DIR       = INFERENCE_OUTPUT / "3d_reconstructions"

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
# BÖLÜM 1: ORTAM HAZIRLIĞI
# ============================================================
def setup_inference_env() -> Dict:
    """Inference için ortam ve klasörleri hazırlar."""
    header("INFERENCE ORTAMI HAZIRLANIYOR")

    # Klasörler
    for d in [SEG_MASKS_DIR, VIZ_DIR, RECON_3D_DIR]:
        d.mkdir(parents=True, exist_ok=True)

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

    # Model kontrol
    results_dir = Path(os.environ.get(
        "nnUNet_results",
        str(BASE_PATH / "data" / "nnunet_results")
    ))
    fold_dir = (results_dir
                / "Dataset007_Pancreas"
                / "nnUNetTrainer__nnUNetPlans__2d"
                / "fold_0")

    model_info = {
        "results_dir":   str(results_dir),
        "fold_dir":      str(fold_dir),
        "model_exists":  False,
        "checkpoint":    None,
    }

    # checkpoint_final.pth veya checkpoint_best.pth ara
    for ckpt_name in ["checkpoint_final.pth", "checkpoint_best.pth"]:
        ckpt_path = fold_dir / ckpt_name
        if ckpt_path.exists():
            model_info["model_exists"] = True
            model_info["checkpoint"]   = str(ckpt_path)
            ok(f"Model bulundu: {ckpt_path}")
            break

    if not model_info["model_exists"]:
        warn(f"Egitilmis model bulunamadi: {fold_dir}")
        warn("Once ADIM 4 (train_model.py) calistirin.")
        warn("Simülasyon verisiyle devam ediliyor...")

    # GPU kontrol
    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        ok(f"Device: {device.upper()}")
        model_info["device"] = device
    except ImportError:
        model_info["device"] = "cpu"

    return model_info


# ============================================================
# BÖLÜM 2: nnU-Net INFERENCE
# ============================================================
def run_nnunet_inference(
    input_dir:     Path,
    output_dir:    Path,
    dataset_id:    str  = "007",
    configuration: str  = "2d",
    fold:          int  = 0,
    save_probabilities: bool = False
) -> bool:
    """
    nnUNetv2_predict komutunu çalıştırır.

    nnU-Net Inference Süreci:
    1. CT görüntüsünü yükle ve preprocess et (CT normalizasyonu)
    2. 2D sliceler üzerinde sliding window inference yap
    3. Gaussian weighting ile prediction'ları birleştir
    4. Argmax ile final segmentasyonu oluştur
    5. Orijinal voxel spacing'e geri resample et

    Args:
        input_dir:    Test CT'lerinin bulunduğu klasör
        output_dir:   Segmentasyon maskelerinin yazılacağı klasör
        dataset_id:   "007"
        configuration:"2d"
        fold:         0
        save_probabilities: Softmax olasılıklarını da kaydet

    Returns:
        True: Başarılı, False: Başarısız
    """
    header(f"nnU-Net INFERENCE BASLIYOR")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Executable bul
    predict_exe = shutil.which("nnUNetv2_predict")
    if not predict_exe:
        scripts_dir = Path(sys.executable).parent
        for candidate in [scripts_dir / "nnUNetv2_predict.exe", scripts_dir / "nnUNetv2_predict"]:
            if candidate.exists():
                predict_exe = str(candidate)
                break
    if not predict_exe:
        predict_exe = "nnUNetv2_predict"

    cmd = [
        predict_exe,
        "-i",  str(input_dir),
        "-o",  str(output_dir),
        "-d",  dataset_id,
        "-c",  configuration,
        "-f",  str(fold),
        "-step_size", "0.5",   # Sliding window adım boyutu
    ]

    if save_probabilities:
        cmd.append("--save_probabilities")

    info(f"Komut: {' '.join(cmd)}")
    info(f"Girdi: {input_dir}")
    info(f"Cikti: {output_dir}")

    # Input dosyalarını say
    n_inputs = len(list(input_dir.glob("*.nii.gz")))
    info(f"Tahmin edilecek CT: {n_inputs}")

    if n_inputs == 0:
        warn("Girdi klasorunde NIfTI dosyası bulunamadı!")
        return False

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

        for line in process.stdout:
            line = line.rstrip()
            if line:
                print(f"  [nnUNet] {line}")

        process.wait()
        elapsed = time.time() - start_time

        if process.returncode == 0:
            n_outputs = len(list(output_dir.glob("*.nii.gz")))
            ok(f"Inference TAMAMLANDI ({elapsed:.1f}s)")
            ok(f"Olusturulan maske: {n_outputs}")
            return True
        else:
            fail(f"Inference basarisiz! Return code: {process.returncode}")
            return False

    except FileNotFoundError:
        fail("nnUNetv2_predict bulunamadi! pip install nnunetv2")
        return False
    except Exception as e:
        fail(f"Hata: {e}")
        return False


# ============================================================
# BÖLÜM 3: SEGMENTASYON ANALİZİ — TÜMÖR KARARI
# ============================================================
def classify_from_mask(
    mask: np.ndarray,
    min_tumor_voxels: int = 50
) -> Dict:
    """Ham maskeyi özetler; anatomik kapı olmadan tümör kararı vermez.

    ``min_tumor_voxels`` eski çağrılar bozulmasın diye imzada tutulur, karar
    ölçütü olarak kullanılmaz. Üretim kararı ``segmentation_postprocess``
    modülündeki fiziksel 3B bileşen doğrulamasından gelir.
    """
    total_voxels     = int(mask.size)
    background_voxels = int(np.sum(mask == 0))
    pancreas_voxels  = int(np.sum(mask == 1))
    tumor_voxels     = int(np.sum(mask == 2))

    return {
        "prediction": "3B anatomik doğrulama yapılmadı — karar verilemedi",
        "has_tumor": None,
        "confidence_score": None,
        "confidence_level": "Doğrulanamadı",
        "total_voxels":       total_voxels,
        "background_voxels":  background_voxels,
        "pancreas_voxels":    pancreas_voxels,
        "tumor_voxels":       tumor_voxels,
        "pancreas_ratio_pct": float(pancreas_voxels / total_voxels * 100),
        "tumor_ratio_pct":    float(tumor_voxels / total_voxels * 100),
        "threshold_voxels": None,
        "decision_rule": "independent_3d_anatomical_gate_required",
    }


# ============================================================
# BÖLÜM 4: GÖRSELLEŞTIRME
# ============================================================
def create_visualization(
    ct_data:   np.ndarray,
    mask_data: np.ndarray,
    result:    Dict,
    output_path: Path,
    n_slices: int = 5
) -> Optional[Path]:
    """
    CT + Segmentasyon maskesi görselleştirmesi oluşturur.

    Renk kodu:
        Yeşil (1): Pankreas
        Kırmızı (2): Tümör

    Args:
        ct_data:    3D CT görüntüsü (HU değerleri)
        mask_data:  3D segmentasyon maskesi
        result:     Sınıflandırma sonucu
        output_path: PNG çıktı yolu
        n_slices:   Gösterilecek slice sayısı
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        from matplotlib.colors import ListedColormap

        # Pankreas bölgesinde merkez slice'ları bul
        pancreas_slices = np.where(mask_data.max(axis=(0, 1)) > 0)[0]
        if len(pancreas_slices) == 0:
            pancreas_slices = range(mask_data.shape[2] // 4,
                                   3 * mask_data.shape[2] // 4)

        # Eşit aralıklı n_slices seç
        step = max(1, len(pancreas_slices) // n_slices)
        selected_slices = list(pancreas_slices[::step])[:n_slices]
        while len(selected_slices) < n_slices:
            selected_slices.append(selected_slices[-1])

        # Figure oluştur
        fig, axes = plt.subplots(
            2, n_slices,
            figsize=(4 * n_slices, 8),
            facecolor="#1a1a2e"
        )

        # Başlık rengi
        title_color = "#e74c3c" if result["has_tumor"] else "#2ecc71"
        prediction  = result["prediction"]
        confidence  = result["confidence_level"]

        fig.suptitle(
            f"SINIFLANDIRMA SONUCU: {prediction}  |  Güven: {confidence}",
            fontsize=16, fontweight="bold",
            color=title_color, y=1.02
        )

        # CT window/level (yumuşak doku)
        ct_min, ct_max = -150, 250  # Soft tissue window
        ct_normalized  = np.clip(ct_data, ct_min, ct_max)
        ct_normalized  = (ct_normalized - ct_min) / (ct_max - ct_min)

        # Maske renk haritası
        mask_cmap   = ListedColormap(["none", "#27ae60", "#e74c3c"])
        mask_alpha  = 0.5

        for col, s_idx in enumerate(selected_slices):
            # --- Üst satır: CT ---
            ax_ct = axes[0, col]
            ct_slice = ct_normalized[:, :, s_idx].T
            ax_ct.imshow(ct_slice, cmap="gray", origin="lower")
            ax_ct.set_title(f"Slice {s_idx}", color="white", fontsize=9)
            ax_ct.axis("off")

            # --- Alt satır: CT + Maske ---
            ax_seg = axes[1, col]
            mask_slice = mask_data[:, :, s_idx].T
            ax_seg.imshow(ct_slice, cmap="gray", origin="lower")
            ax_seg.imshow(
                np.ma.masked_where(mask_slice == 0, mask_slice),
                cmap=mask_cmap, alpha=mask_alpha, origin="lower",
                vmin=0, vmax=2
            )
            ax_seg.axis("off")

            if col == 0:
                axes[0, col].set_ylabel("CT", color="white")
                axes[1, col].set_ylabel("Segmentasyon", color="white")

        # Legend
        legend_elements = [
            mpatches.Patch(color="#27ae60", alpha=0.7, label="Pankreas"),
            mpatches.Patch(color="#e74c3c", alpha=0.7, label="Tumor"),
        ]
        fig.legend(
            handles=legend_elements,
            loc="lower center",
            ncol=2,
            fontsize=10,
            facecolor="#2c2c54",
            labelcolor="white",
            framealpha=0.8
        )

        # İstatistik kutusu
        stats_text = (
            f"Pankreas: {result['pancreas_voxels']:,} voksel ({result['pancreas_ratio_pct']:.2f}%)\n"
            f"Tumor:    {result['tumor_voxels']:,} voksel ({result['tumor_ratio_pct']:.3f}%)"
        )
        fig.text(
            0.5, -0.02, stats_text,
            ha="center", va="bottom",
            fontsize=9, color="#bdc3c7",
            family="monospace"
        )

        plt.tight_layout()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(output_path), dpi=150, bbox_inches="tight",
                   facecolor="#1a1a2e")
        plt.close()

        ok(f"Görsel kaydedildi: {output_path}")
        return output_path

    except ImportError as e:
        warn(f"Matplotlib bulunamadi: {e}")
        return None
    except Exception as e:
        warn(f"Görselleştirme hatası: {e}")
        import traceback
        traceback.print_exc()
        return None


# ============================================================
# BÖLÜM 5: TEK DOSYA İŞLEME
# ============================================================
def process_single_ct(
    ct_path:   Path,
    mask_path: Optional[Path] = None,
    min_tumor_voxels: int = 50
) -> Dict:
    """
    Tek bir CT dosyasını işler (inference + classification + visualization).

    Maske yoksa sonuç üretmez. Rastgele/simüle maske tıbbi çıktı olarak
    kesinlikle kullanılmaz.

    Args:
        ct_path:   CT NIfTI dosya yolu
        mask_path: Segmentasyon maskesi yolu (inference sonucu)
        min_tumor_voxels: Tümör eşiği

    Returns:
        Tam işlem sonucu
    """
    header(f"CT ISLENIYOR: {ct_path.name}")

    try:
        import nibabel as nib
    except ImportError:
        fail("nibabel kurulu degil!")
        return {}

    # CT yükle
    try:
        ct_img  = nib.load(str(ct_path))
        ct_data = ct_img.get_fdata()
        ok(f"CT yuklendi: {ct_data.shape} | spacing: {ct_img.header.get_zooms()}")
    except Exception as e:
        fail(f"CT yuklenemedi: {e}")
        return {}

    # Maske yükle; başarısızsa kapalı kal.
    if mask_path and mask_path.exists():
        try:
            mask_img  = nib.load(str(mask_path))
            mask_data = mask_img.get_fdata().astype(np.int32)
            ok(f"Maske yuklendi: {mask_path.name}")
        except Exception as e:
            fail(f"Maske yüklenemedi; sonuç üretilmedi: {e}")
            return {}
    else:
        fail("Maske bulunamadı; simülasyon kapalı olduğu için sonuç üretilmedi.")
        return {}

    # Bağımsız 3B pankreas kapısı olmadan ham etiket 2 klinik karar değildir.
    pancreas_voxels = int(np.count_nonzero(mask_data == 1))
    tumor_voxels = int(np.count_nonzero(mask_data == 2))
    result = {
        "prediction": "3B anatomik doğrulama yapılmadı — karar verilemedi",
        "has_tumor": None,
        "pancreas_voxels": pancreas_voxels,
        "tumor_voxels": tumor_voxels,
        "confidence_level": "Doğrulanamadı",
        "quality_status": "indeterminate",
    }
    result["ct_path"]   = str(ct_path)
    result["ct_shape"]  = list(ct_data.shape)
    result["case_id"]   = ct_path.stem.replace("_0000", "")
    result["timestamp"] = datetime.now().isoformat()

    # Görselleştirme
    viz_path = VIZ_DIR / f"{result['case_id']}_segmentation.png"
    create_visualization(ct_data, mask_data, result, viz_path)
    if viz_path.exists():
        result["visualization"] = str(viz_path)

    return result


# ============================================================
# BÖLÜM 6: TOPLU INFERENCE
# ============================================================
def batch_inference_and_classify(
    input_dir: Path,
    mask_dir:  Path,
    min_tumor_voxels: int = 50
) -> List[Dict]:
    """Tüm CT dosyalarını toplu olarak işler."""
    header("TOPLU INFERENCE & SINIFLANDIRMA")

    ct_files = sorted(input_dir.glob("*.nii.gz"))
    if not ct_files:
        warn(f"CT dosyası bulunamadı: {input_dir}")
        return []

    results = []
    tumor_count  = 0
    healthy_count = 0
    indeterminate_count = 0

    for i, ct_path in enumerate(ct_files):
        case_id  = ct_path.stem.replace("_0000", "")
        mask_path = mask_dir / f"{case_id}.nii.gz"

        info(f"[{i+1}/{len(ct_files)}] {ct_path.name}")

        result = process_single_ct(ct_path, mask_path, min_tumor_voxels)
        if result:
            results.append(result)
            if result.get("has_tumor") is True:
                tumor_count += 1
            elif result.get("has_tumor") is False:
                healthy_count += 1
            else:
                indeterminate_count += 1

    # Özet
    header("TOPLU SINIFLANDIRMA OZETI")
    ok(f"Toplam islenen:  {len(results)}")
    ok(f"Tumor Var:       {tumor_count}")
    ok(f"Tumor Yok:       {healthy_count}")
    warn(f"Karar verilemedi: {indeterminate_count}")

    return results


# ============================================================
# BÖLÜM 7: SONUÇLARI KAYDET
# ============================================================
def save_inference_results(results: List[Dict]) -> Path:
    """Tüm inference sonuçlarını JSON dosyasına kaydeder."""
    header("SONUCLAR KAYDEDILIYOR")

    metrics_dir = BASE_PATH / "metrics"
    metrics_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Özet
    summary = {
        "step":       "ADIM6_7_Inference_Classification",
        "timestamp":  datetime.now().isoformat(),
        "total":      len(results),
        "tumor_var":  sum(1 for r in results if r.get("has_tumor")),
        "tumor_yok":  sum(1 for r in results if r.get("has_tumor") is False),
        "karar_verilemedi": sum(1 for r in results if r.get("has_tumor") is None),
        "results":    results,
    }

    output_path = metrics_dir / f"inference_results_{timestamp}.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4, ensure_ascii=False)

    ok(f"Sonuclar kaydedildi: {output_path}")

    # Kullanıcı dostu özet yazdır
    print(f"\n  {'='*50}")
    print(f"  {'Case ID':<25} {'Karar':<15} {'Guven':<15}")
    print(f"  {'-'*50}")
    for r in results:
        case_id    = r.get("case_id", "?")[:22]
        prediction = r.get("prediction", "?")
        confidence = r.get("confidence_level", "?")
        color = GREEN if r.get("has_tumor") else CYAN
        print(f"  {case_id:<25} {color}{prediction:<15}{RESET} {confidence}")
    print(f"  {'='*50}\n")

    return output_path


# ============================================================
# MAIN
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description="nnU-Net inference ve tümör sınıflandırması"
    )
    parser.add_argument(
        "--input",
        type=str,
        default=str(BASE_PATH / "data" / "nnunet_raw" /
                    "Dataset007_Pancreas" / "imagesTs"),
        # ⚠️ BU SATIRI DEĞİŞTİR: Test CT'lerinin klasörü
        help="Test CT görüntülerinin klasörü"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(SEG_MASKS_DIR),
        help="Segmentasyon maskelerinin çıktı klasörü"
    )
    parser.add_argument(
        "--single_file",
        action="store_true",
        help="Tek dosya işle (--input dosya yolu olmalı)"
    )
    parser.add_argument(
        "--mask",
        type=str,
        default=None,
        help="Tek dosya modunda kullanılacak gerçek segmentasyon maskesi"
    )
    parser.add_argument(
        "--skip_nnunet",
        action="store_true",
        help="nnU-Net inference atla (varolan maskeler kullanılsın)"
    )
    parser.add_argument(
        "--min_tumor_voxels",
        type=int,
        default=50,
        # ⚠️ BU SATIRI DEĞİŞTİR: Tümör eşiğini ayarlayın
        help="Tümür kararı için minimum voksel eşiği (varsayılan: 50)"
    )
    args = parser.parse_args()

    print(f"""
{BOLD}{CYAN}
==============================================================
  ADIM 6 + 7: INFERENCE & SINIFLANDIRMA
  nnU-Net Segmentasyonu -> Tumor Var / Tumor Yok
==============================================================
{RESET}
""")

    # 1. Ortam hazırla
    model_info = setup_inference_env()

    input_path  = Path(args.input)
    output_path = Path(args.output)

    # 2. Tek dosya modu
    if args.single_file:
        if not input_path.exists():
            fail(f"Dosya bulunamadı: {input_path}")
            return 1
        mask_path = Path(args.mask) if args.mask else None
        result = process_single_ct(input_path, mask_path, args.min_tumor_voxels)
        if result:
            print(f"\n  {BOLD}{'='*50}")
            color = RED if result["has_tumor"] else GREEN
            print(f"  SONUC: {color}{BOLD}{result['prediction']}{RESET}")
            print(f"  Guven: {result['confidence_level']}")
            print(f"  Tumor voksel: {result['tumor_voxels']:,}")
            print(f"  {'='*50}{RESET}\n")
            return 0
        return 1

    # 3. Toplu mod: önce nnU-Net inference
    if input_path.is_dir():
        # nnU-Net ile inference çalıştır
        if not args.skip_nnunet and model_info.get("model_exists"):
            nnunet_success = run_nnunet_inference(
                input_dir     = input_path,
                output_dir    = output_path,
                dataset_id    = "007",
                configuration = "2d",
                fold          = 0
            )
        else:
            if not model_info.get("model_exists"):
                fail("Eğitilmiş model bulunamadı; demo/etiket geri dönüşü devre dışıdır.")
                return 1
            warn("nnU-Net inference atlandı; yalnızca var olan gerçek maskeler okunacak.")
            nnunet_success = output_path.exists() and any(output_path.glob("*.nii.gz"))

        if not nnunet_success:
            fail("Geçerli nnU-Net tahmin maskesi üretilemedi veya bulunamadı.")
            return 1

        # 4. Sınıflandırma (toplu)
        results = batch_inference_and_classify(
            input_dir         = input_path,
            mask_dir          = output_path,
            min_tumor_voxels  = args.min_tumor_voxels
        )

        # 5. Sonuçları kaydet
        if results:
            save_inference_results(results)

        # 6. Özet
        header("ADIM 6 + 7 TAMAMLANDI")
        ok(f"Toplam islem: {len(results)}")
        ok(f"Gorsel ciktilari: {VIZ_DIR}")
        print(f"""
  {BOLD}Sonraki Adim (ADIM 8) - 3D Rekonstrüksiyon:{RESET}
    {CYAN}python scripts/reconstruct_3d.py{RESET}

  {BOLD}Sonraki Adim (ADIM 9) - Web Arayuzu:{RESET}
    {CYAN}python web/app.py{RESET}
""")
        return 0

    fail(f"Geçersiz input: {input_path}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
