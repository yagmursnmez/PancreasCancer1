"""
============================================================
ADIM 5: VALIDATION — Dice Score & IoU Hesaplama
============================================================
Bu script:
1. Eğitilmiş nnU-Net modelinin tahminlerini değerlendirir
2. Segmentasyon metrikleri hesaplar:
   - Dice Score (pancreas + tumor ayrı ayrı)
   - IoU / Jaccard Index
3. Sınıflandırma metrikleri hesaplar:
   - Accuracy, Precision, Recall, F1-score
   - ROC-AUC
4. Sonuçları CSV ve JSON olarak kaydeder
5. Görselleştirme grafiklerini oluşturur

Kullanım:
    python scripts/validate_metrics.py
    python scripts/validate_metrics.py --pred_dir data/inference_output/segmentation_masks
============================================================
"""

import os
import sys
import json
import time
import argparse
import warnings
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np

warnings.filterwarnings("ignore")

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
# BÖLÜM 1: SEGMENTASYON METRİKLERİ
# ============================================================
def compute_dice_score(
    pred: np.ndarray,
    target: np.ndarray,
    label: int,
    smooth: float = 1e-6
) -> float:
    """
    Belirli bir sınıf için Dice Score hesaplar.

    Dice = 2 * |P ∩ T| / (|P| + |T|)

    Args:
        pred:   Tahmin maskesi (H x W x D)
        target: Gerçek maske (H x W x D)
        label:  Değerlendirilecek sınıf (1=pancreas, 2=tumor)
        smooth: Sıfıra bölme koruması

    Returns:
        Dice skoru [0.0 - 1.0]

    Klinik yorum:
        < 0.5  : Kötü
        0.5-0.7: Kabul edilebilir
        0.7-0.9: İyi
        > 0.9  : Mükemmel
    """
    pred_bin   = (pred == label).astype(np.float32)
    target_bin = (target == label).astype(np.float32)

    intersection = np.sum(pred_bin * target_bin)
    dice = (2. * intersection + smooth) / (
        np.sum(pred_bin) + np.sum(target_bin) + smooth
    )
    return float(dice)


def compute_iou(
    pred: np.ndarray,
    target: np.ndarray,
    label: int,
    smooth: float = 1e-6
) -> float:
    """
    Belirli bir sınıf için IoU (Jaccard Index) hesaplar.

    IoU = |P ∩ T| / |P ∪ T|
        = intersection / (|P| + |T| - intersection)

    Args:
        pred:   Tahmin maskesi
        target: Gerçek maske
        label:  Sınıf etiketi
        smooth: Sıfıra bölme koruması

    Returns:
        IoU skoru [0.0 - 1.0]
    """
    pred_bin   = (pred == label).astype(np.float32)
    target_bin = (target == label).astype(np.float32)

    intersection = np.sum(pred_bin * target_bin)
    union        = np.sum(pred_bin) + np.sum(target_bin) - intersection
    iou = (intersection + smooth) / (union + smooth)
    return float(iou)


def compute_hausdorff_distance(
    pred: np.ndarray,
    target: np.ndarray,
    label: int,
    percentile: float = 95.0
) -> float:
    """
    Hausdorff mesafesini hesaplar (yüzey hatası ölçütü).
    
    HD95: Gerçek ve tahmin yüzeyler arasındaki %95 yüzdelik Hausdorff mesafesi.
    Daha küçük = daha iyi.

    Not: Scipy gerektirir — scipy.ndimage.morphology
    """
    try:
        from scipy.ndimage import distance_transform_edt

        pred_bin   = (pred == label).astype(bool)
        target_bin = (target == label).astype(bool)

        if not pred_bin.any() or not target_bin.any():
            return float("inf")

        # Yüzey pikselleri
        pred_surface   = pred_bin ^ _erode(pred_bin)
        target_surface = target_bin ^ _erode(target_bin)

        # Mesafe dönüşümü
        dt_pred   = distance_transform_edt(~pred_bin)
        dt_target = distance_transform_edt(~target_bin)

        hd_p2t = dt_target[pred_surface]
        hd_t2p = dt_pred[target_surface]

        if len(hd_p2t) == 0 or len(hd_t2p) == 0:
            return float("inf")

        all_distances = np.concatenate([hd_p2t, hd_t2p])
        return float(np.percentile(all_distances, percentile))

    except Exception:
        return float("nan")


def _erode(mask: np.ndarray, iterations: int = 1) -> np.ndarray:
    from scipy.ndimage import binary_erosion
    return binary_erosion(mask, iterations=iterations)


# ============================================================
# BÖLÜM 2: SINIFLANDIRMA METRİKLERİ
# ============================================================
def compute_classification_metrics(
    pred_masks: List[np.ndarray],
    true_masks: List[np.ndarray],
    min_tumor_voxels: int = 50
) -> Dict:
    """
    Segmentasyon maskelerinden tümör var/yok sınıflandırma metriklerini hesaplar.

    Karar kuralı:
        if (mask == 2).sum() > min_tumor_voxels:
            sonuç = "Tümör Var" (1)
        else:
            sonuç = "Tümör Yok" (0)

    Args:
        pred_masks:       Tahmin maskelerinin listesi
        true_masks:       Gerçek maskelerin listesi
        min_tumor_voxels: Minimum tümör vokseli eşiği (gürültü filtresi)

    Returns:
        Metrik sözlüğü: accuracy, precision, recall, f1, roc_auc
    """
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score,
        f1_score, roc_auc_score, confusion_matrix,
        classification_report
    )

    y_true = []
    y_pred = []
    y_prob = []  # Tümör olasılığı (tümör voksel oranı)

    for pred_mask, true_mask in zip(pred_masks, true_masks):
        # Gerçek etiket
        true_tumor  = 1 if (true_mask == 2).sum() > min_tumor_voxels else 0

        # Tahmin
        pred_tumor_voxels = (pred_mask == 2).sum()
        pred_label = 1 if pred_tumor_voxels > min_tumor_voxels else 0

        # Olasılık (normalize edilmiş tümör hacmi)
        total_voxels = pred_mask.size
        prob = min(pred_tumor_voxels / total_voxels * 100, 1.0)

        y_true.append(true_tumor)
        y_pred.append(pred_label)
        y_prob.append(prob)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_prob = np.array(y_prob)

    # Metrikler
    metrics = {
        "accuracy":  float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall":    float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score":  float(f1_score(y_true, y_pred, zero_division=0)),
        "n_samples": len(y_true),
        "n_tumor":   int(y_true.sum()),
        "n_healthy": int((1 - y_true).sum()),
        "y_true":    y_true.tolist(),
        "y_pred":    y_pred.tolist(),
    }

    # ROC-AUC (en az 2 sınıf gerekir)
    try:
        if len(np.unique(y_true)) > 1:
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
        else:
            metrics["roc_auc"] = float("nan")
            warn("ROC-AUC hesaplanamadı: sadece 1 sınıf mevcut")
    except Exception as e:
        metrics["roc_auc"] = float("nan")
        warn(f"ROC-AUC hatasi: {e}")

    # Confusion matrix
    try:
        cm = confusion_matrix(y_true, y_pred)
        metrics["confusion_matrix"] = cm.tolist()
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            metrics["TP"] = int(tp)
            metrics["TN"] = int(tn)
            metrics["FP"] = int(fp)
            metrics["FN"] = int(fn)
    except Exception as e:
        warn(f"Confusion matrix hatasi: {e}")

    return metrics


# ============================================================
# BÖLÜM 3: DOSYALARI DEĞERLENDIR
# ============================================================
def evaluate_all_cases(
    pred_dir:  Path,
    label_dir: Path,
    max_cases: Optional[int] = None
) -> Tuple[List[Dict], List[np.ndarray], List[np.ndarray]]:
    """
    Tüm tahmin-gerçek çiftlerini değerlendirir.

    Args:
        pred_dir:  Tahmin maskeleri (.nii.gz)
        label_dir: Gerçek maskeler (.nii.gz)
        max_cases: Maksimum değerlendirilecek vaka sayısı

    Returns:
        (per_case_metrics, pred_masks, true_masks)
    """
    header("VAKA BAZINDA DEGERLENDIRME")

    try:
        import nibabel as nib
    except ImportError:
        fail("nibabel kurulu degil!")
        return [], [], []

    pred_files = sorted(pred_dir.glob("*.nii.gz"))
    if not pred_files:
        warn(f"Tahmin dosyası bulunamadı: {pred_dir}")
        warn("Önce ADIM 6 (inference) çalıştırın.")
        return [], [], []

    if max_cases:
        pred_files = pred_files[:max_cases]

    per_case_metrics = []
    all_pred_masks   = []
    all_true_masks   = []

    for pred_path in pred_files:
        # Karşılık gelen gerçek maskeyi bul
        case_id = pred_path.name.replace("_0000", "")  # imagesTr adı temizle
        true_path = label_dir / case_id
        if not true_path.exists():
            # Benzer isimli başka bir hastayı sessizce eşlemek metrikleri geçersiz
            # kılar. Yalnızca bire bir vaka adı kabul edilir.
            warn(f"Gerçek maske bulunamadı: {case_id}, atlanıyor.")
            continue

        try:
            # Yükle
            pred_img = nib.load(str(pred_path))
            true_img = nib.load(str(true_path))
            pred_data = pred_img.get_fdata().astype(np.int32)
            true_data = true_img.get_fdata().astype(np.int32)

            # Boyut eşleşmesi kontrolü
            if pred_data.shape != true_data.shape:
                warn(f"Boyut uyusmazligi {pred_path.name}: pred={pred_data.shape}, true={true_data.shape}")
                continue

            # Metrikler hesapla
            case_metrics = {
                "case_id":        pred_path.stem,
                "pred_path":      str(pred_path),
                "true_path":      str(true_path),
                "shape":          list(pred_data.shape),

                # Pancreas metrikleri
                "dice_pancreas":  compute_dice_score(pred_data, true_data, label=1),
                "iou_pancreas":   compute_iou(pred_data, true_data, label=1),

                # Tumor metrikleri
                "dice_tumor":     compute_dice_score(pred_data, true_data, label=2),
                "iou_tumor":      compute_iou(pred_data, true_data, label=2),

                # Hacim istatistikleri
                "pred_tumor_voxels": int((pred_data == 2).sum()),
                "true_tumor_voxels": int((true_data == 2).sum()),
                "has_tumor_true":    bool((true_data == 2).sum() > 50),
                "has_tumor_pred":    bool((pred_data == 2).sum() > 50),
            }

            # Hausdorff (pahalı — opsiyonel)
            # case_metrics["hd95_pancreas"] = compute_hausdorff_distance(pred_data, true_data, 1)
            # case_metrics["hd95_tumor"] = compute_hausdorff_distance(pred_data, true_data, 2)

            per_case_metrics.append(case_metrics)
            all_pred_masks.append(pred_data)
            all_true_masks.append(true_data)

            # Raporla
            tumor_status = "TUMOR" if case_metrics["has_tumor_true"] else "     "
            info(
                f"{pred_path.stem[:20]:<20} | {tumor_status} | "
                f"Dice(pan)={case_metrics['dice_pancreas']:.3f} "
                f"Dice(tum)={case_metrics['dice_tumor']:.3f} "
                f"IoU(pan)={case_metrics['iou_pancreas']:.3f}"
            )

        except Exception as e:
            warn(f"Hata [{pred_path.name}]: {e}")
            continue

    ok(f"Toplam degerlendirilen vaka: {len(per_case_metrics)}")
    return per_case_metrics, all_pred_masks, all_true_masks


# ============================================================
# BÖLÜM 4: ÖZET METRİKLER
# ============================================================
def compute_summary_metrics(per_case_metrics: List[Dict]) -> Dict:
    """Tüm vakaların ortalamasını alarak özet metrikler hesaplar."""
    header("OZET METRIKLER")

    if not per_case_metrics:
        warn("Hesaplanacak metrik yok!")
        return {}

    # Numpy dizilerine çevir
    dice_pan  = np.array([m["dice_pancreas"] for m in per_case_metrics])
    dice_tum  = np.array([m["dice_tumor"]    for m in per_case_metrics])
    iou_pan   = np.array([m["iou_pancreas"]  for m in per_case_metrics])
    iou_tum   = np.array([m["iou_tumor"]     for m in per_case_metrics])

    # Sadece tümör olan vakalar için tümör metriklerini hesapla
    tumor_cases = [m for m in per_case_metrics if m["has_tumor_true"]]
    if tumor_cases:
        dice_tum_nonzero = np.array([m["dice_tumor"] for m in tumor_cases])
        iou_tum_nonzero  = np.array([m["iou_tumor"]  for m in tumor_cases])
    else:
        dice_tum_nonzero = dice_tum
        iou_tum_nonzero  = iou_tum

    summary = {
        # Pancreas segmentasyon
        "pancreas": {
            "dice_mean": float(np.mean(dice_pan)),
            "dice_std":  float(np.std(dice_pan)),
            "dice_median": float(np.median(dice_pan)),
            "dice_min":  float(np.min(dice_pan)),
            "dice_max":  float(np.max(dice_pan)),
            "iou_mean":  float(np.mean(iou_pan)),
            "iou_std":   float(np.std(iou_pan)),
        },
        # Tumor segmentasyon
        "tumor": {
            "dice_mean": float(np.mean(dice_tum_nonzero)),
            "dice_std":  float(np.std(dice_tum_nonzero)),
            "dice_median": float(np.median(dice_tum_nonzero)),
            "dice_min":  float(np.min(dice_tum_nonzero)),
            "dice_max":  float(np.max(dice_tum_nonzero)),
            "iou_mean":  float(np.mean(iou_tum_nonzero)),
            "iou_std":   float(np.std(iou_tum_nonzero)),
            "n_cases":   len(tumor_cases),
        },
        "n_total":  len(per_case_metrics),
        "n_tumor":  len(tumor_cases),
    }

    # Tablo yazdır
    print(f"""
  {BOLD}{'='*56}
  SEGMENTASYON METRIKLERI{RESET}
  {'='*56}
  {'Metrik':<25} {'Pankreas':>12} {'Tumor':>12}
  {'-'*56}
  {'Dice (Ort ± Std)':<25} {summary['pancreas']['dice_mean']:>8.4f}±{summary['pancreas']['dice_std']:.3f} {summary['tumor']['dice_mean']:>8.4f}±{summary['tumor']['dice_std']:.3f}
  {'Dice (Medyan)':<25} {summary['pancreas']['dice_median']:>12.4f} {summary['tumor']['dice_median']:>12.4f}
  {'Dice (Min)':<25} {summary['pancreas']['dice_min']:>12.4f} {summary['tumor']['dice_min']:>12.4f}
  {'Dice (Max)':<25} {summary['pancreas']['dice_max']:>12.4f} {summary['tumor']['dice_max']:>12.4f}
  {'IoU (Ort ± Std)':<25} {summary['pancreas']['iou_mean']:>8.4f}±{summary['pancreas']['iou_std']:.3f} {summary['tumor']['iou_mean']:>8.4f}±{summary['tumor']['iou_std']:.3f}
  {'-'*56}
  {'Toplam Vaka':<25} {summary['n_total']:>12}
  {'Tumor Vakasi':<25} {summary['n_tumor']:>12}
  {BOLD}{'='*56}{RESET}
""")

    return summary


# ============================================================
# BÖLÜM 5: VİZUALİZASYON
# ============================================================
def plot_metrics(
    per_case_metrics: List[Dict],
    summary:          Dict,
    clf_metrics:      Dict,
    output_dir:       Path
) -> None:
    """
    Metrik görselleştirmeleri oluşturur:
    1. Dice/IoU box plot'ları
    2. ROC eğrisi
    3. Confusion matrix
    """
    try:
        import matplotlib
        matplotlib.use("Agg")  # GUI gerektirmez
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches

        output_dir.mkdir(parents=True, exist_ok=True)

        # ---- 1. Dice ve IoU Dağılımları ----
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle("nnU-Net 2D - Segmentasyon Metrikleri", fontsize=14, fontweight="bold")

        dice_pan = [m["dice_pancreas"] for m in per_case_metrics]
        dice_tum = [m["dice_tumor"]    for m in per_case_metrics]
        iou_pan  = [m["iou_pancreas"]  for m in per_case_metrics]
        iou_tum  = [m["iou_tumor"]     for m in per_case_metrics]

        # Box plot
        ax = axes[0]
        bp = ax.boxplot(
            [dice_pan, dice_tum, iou_pan, iou_tum],
            tick_labels=["Dice\nPankreas", "Dice\nTumor", "IoU\nPankreas", "IoU\nTumor"],
            patch_artist=True,
            notch=True
        )
        colors = ["#3498db", "#e74c3c", "#2ecc71", "#f39c12"]
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax.set_title("Metrik Dağılımı")
        ax.set_ylabel("Skor")
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0.7, color="green", linestyle="--", alpha=0.5, label="0.7 (İyi)")
        ax.legend(fontsize=8)

        # Histogram - Pancreas Dice
        ax = axes[1]
        ax.hist(dice_pan, bins=10, color="#3498db", alpha=0.7, edgecolor="black", label="Pankreas")
        ax.hist(dice_tum, bins=10, color="#e74c3c", alpha=0.7, edgecolor="black", label="Tumor")
        ax.set_title("Dice Score Histogramı")
        ax.set_xlabel("Dice Score")
        ax.set_ylabel("Vaka Sayısı")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Confusion Matrix
        ax = axes[2]
        if "confusion_matrix" in clf_metrics:
            cm = np.array(clf_metrics["confusion_matrix"])
            if cm.size == 4:
                im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
                plt.colorbar(im, ax=ax)
                ax.set_title("Confusion Matrix\n(Tumör Sınıflandırma)")
                ax.set_xlabel("Tahmin")
                ax.set_ylabel("Gerçek")
                ax.set_xticks([0, 1])
                ax.set_yticks([0, 1])
                ax.set_xticklabels(["Tümör Yok", "Tümör Var"])
                ax.set_yticklabels(["Tümör Yok", "Tümör Var"])
                thresh = cm.max() / 2.
                for i in range(cm.shape[0]):
                    for j in range(cm.shape[1]):
                        ax.text(j, i, format(cm[i, j], "d"),
                               ha="center", va="center",
                               color="white" if cm[i, j] > thresh else "black",
                               fontsize=14, fontweight="bold")
        else:
            ax.text(0.5, 0.5, "Veri Yetersiz",
                   ha="center", va="center", transform=ax.transAxes, fontsize=12)
            ax.set_title("Confusion Matrix")

        plt.tight_layout()
        fig_path = output_dir / "segmentation_metrics.png"
        plt.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close()
        ok(f"Grafik kaydedildi: {fig_path}")

        # ---- 2. ROC Eğrisi ----
        if ("y_true" in clf_metrics and
            len(np.unique(clf_metrics["y_true"])) > 1):

            fig, ax = plt.subplots(1, 1, figsize=(6, 6))

            from sklearn.metrics import roc_curve, auc
            y_true = np.array(clf_metrics["y_true"])
            y_pred = np.array(clf_metrics.get("y_pred", y_true))
            fpr, tpr, _ = roc_curve(y_true, y_pred)
            roc_auc = auc(fpr, tpr)

            ax.plot(fpr, tpr, color="#e74c3c", lw=2,
                   label=f"ROC (AUC = {roc_auc:.3f})")
            ax.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--", label="Rastgele")
            ax.fill_between(fpr, tpr, alpha=0.1, color="#e74c3c")
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel("Yanlış Pozitif Oranı (FPR)")
            ax.set_ylabel("Doğru Pozitif Oranı (TPR)")
            ax.set_title("ROC Eğrisi — Tümör Sınıflandırma")
            ax.legend(loc="lower right")
            ax.grid(True, alpha=0.3)

            roc_path = output_dir / "roc_curve.png"
            plt.savefig(roc_path, dpi=150, bbox_inches="tight")
            plt.close()
            ok(f"ROC egrisi kaydedildi: {roc_path}")

    except ImportError as e:
        warn(f"Matplotlib bulunamadi, grafikler atlanıyor: {e}")
    except Exception as e:
        warn(f"Grafik olusturma hatasi: {e}")
        import traceback
        traceback.print_exc()


# ============================================================
# BÖLÜM 6: RAPOR KAYDET
# ============================================================
def save_metrics_report(
    summary:          Dict,
    clf_metrics:      Dict,
    per_case_metrics: List[Dict]
) -> Path:
    """Tüm metrikleri dosyalara kaydeder."""
    import csv
    from datetime import datetime

    metrics_dir = BASE_PATH / "metrics"
    metrics_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # JSON rapor
    report = {
        "step":      "ADIM5_Validation",
        "timestamp": datetime.now().isoformat(),
        "segmentation": summary,
        "classification": {k: v for k, v in clf_metrics.items()
                          if k not in ["y_true", "y_pred", "confusion_matrix"]},
        "confusion_matrix": clf_metrics.get("confusion_matrix"),
    }
    json_path = metrics_dir / f"validation_report_{timestamp}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4, ensure_ascii=False)
    ok(f"JSON rapor: {json_path}")

    # CSV — vaka bazında
    if per_case_metrics:
        csv_path = metrics_dir / f"per_case_metrics_{timestamp}.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            fieldnames = [k for k in per_case_metrics[0].keys()
                         if k not in ["pred_path", "true_path"]]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for m in per_case_metrics:
                row = {k: v for k, v in m.items() if k in fieldnames}
                writer.writerow(row)
        ok(f"CSV rapor: {csv_path}")

    return json_path


# ============================================================
# MAIN
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Segmentasyon ve sınıflandırma metriklerini hesapla")
    parser.add_argument(
        "--pred_dir",
        type=str,
        default=str(BASE_PATH / "data" / "inference_output" / "segmentation_masks"),
        # ⚠️ BU SATIRI DEĞİŞTİR: Tahmin maskelerinin klasörü
        help="Tahmin maskelerinin bulunduğu klasör"
    )
    parser.add_argument(
        "--label_dir",
        type=str,
        default=str(BASE_PATH / "data" / "nnunet_raw" / "Dataset007_Pancreas" / "labelsTr"),
        help="Gerçek maskelerin klasörü"
    )
    parser.add_argument(
        "--max_cases",
        type=int,
        default=None,
        help="Değerlendirilecek maksimum vaka sayısı (test için)"
    )
    args = parser.parse_args()

    print(f"""
{BOLD}{CYAN}
==============================================================
  ADIM 5: VALIDATION - Segmentasyon & Sınıflandırma Metrikleri
==============================================================
{RESET}
""")

    pred_dir  = Path(args.pred_dir)
    label_dir = Path(args.label_dir)

    if not label_dir.exists():
        fail(f"Gercek maske klasoru bulunamadi: {label_dir}")
        fail("Once ADIM 2 (prepare_dataset.py) calistirin.")
        return 1

    # Etiketi kendisiyle karşılaştırıp sahte Dice=1.0 üretmek kesinlikle yasak.
    if not pred_dir.exists() or not any(pred_dir.glob("*.nii.gz")):
        warn(f"Tahmin klasörü bos: {pred_dir}")
        fail("Doğrulama iptal edildi: önce bağımsız model tahminleri üretin.")
        return 1
    if pred_dir.resolve() == label_dir.resolve():
        fail("Doğrulama iptal edildi: tahmin ve gerçek etiket klasörü aynı olamaz.")
        return 1

    # 1. Vaka bazında metrikler
    per_case_metrics, pred_masks, true_masks = evaluate_all_cases(
        pred_dir  = pred_dir,
        label_dir = label_dir,
        max_cases = args.max_cases
    )

    if not per_case_metrics:
        fail("Hicbir vaka degerlendirilemedi!")
        return 1

    # 2. Özet metrikler
    summary = compute_summary_metrics(per_case_metrics)

    # 3. Sınıflandırma metrikleri
    header("SINIFLANDIRMA METRIKLERI")
    clf_metrics = compute_classification_metrics(pred_masks, true_masks)

    print(f"""
  {BOLD}{'='*50}
  TUMOR SINIFLANDIRMA SONUCLARI{RESET}
  {'='*50}
  {'Metrik':<25} {'Deger':>12}
  {'-'*50}
  {'Accuracy':<25} {clf_metrics.get('accuracy', 0):>12.4f}
  {'Precision':<25} {clf_metrics.get('precision', 0):>12.4f}
  {'Recall':<25} {clf_metrics.get('recall', 0):>12.4f}
  {'F1 Score':<25} {clf_metrics.get('f1_score', 0):>12.4f}
  {'ROC-AUC':<25} {clf_metrics.get('roc_auc', float('nan')):>12.4f}
  {'-'*50}
  {'Toplam Vaka':<25} {clf_metrics.get('n_samples', 0):>12}
  {'Tumor Var':<25} {clf_metrics.get('n_tumor', 0):>12}
  {'Tumor Yok':<25} {clf_metrics.get('n_healthy', 0):>12}
  {BOLD}{'='*50}{RESET}
""")

    # Confusion matrix
    cm = clf_metrics.get("confusion_matrix")
    if cm and len(cm) == 2:
        print(f"  Confusion Matrix:")
        print(f"              Tahmin: TumYok  TumVar")
        print(f"  Gercek: TumYok   {cm[0][0]:6d}  {cm[0][1]:6d}")
        print(f"  Gercek: TumVar   {cm[1][0]:6d}  {cm[1][1]:6d}")

    # 4. Görselleştirme
    viz_dir = BASE_PATH / "metrics" / "plots"
    plot_metrics(per_case_metrics, summary, clf_metrics, viz_dir)

    # 5. Rapor kaydet
    save_metrics_report(summary, clf_metrics, per_case_metrics)

    header("ADIM 5 TAMAMLANDI")
    ok(f"Toplam {len(per_case_metrics)} vaka değerlendirildi")
    ok(f"Pancreas Dice: {summary.get('pancreas', {}).get('dice_mean', 0):.4f}")
    ok(f"Tumor Dice:    {summary.get('tumor', {}).get('dice_mean', 0):.4f}")
    ok(f"F1 Score:      {clf_metrics.get('f1_score', 0):.4f}")

    print(f"""
  {BOLD}Sonraki Adim (ADIM 6) - Inference:{RESET}
    {CYAN}python scripts/inference.py{RESET}
""")
    return 0


if __name__ == "__main__":
    sys.exit(main())
