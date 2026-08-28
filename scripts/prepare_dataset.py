"""
============================================================
ADIM 2: DATASET HAZIRLAMA - nnU-Net FORMAT
============================================================
Proje: CT Görüntülerinden Pankreas Tümörü Tespiti

Bu script:
1. Kaggle MSD Task07 verisetini nnU-Net v2 formatına çevirir
2. Dosya isimlerini düzenler:
   - CT:    case_XXXX_0000.nii.gz
   - Maske: case_XXXX.nii.gz
3. dataset.json oluşturur
4. Veri bütünlüğünü doğrular

Kullanım:
    python scripts/prepare_dataset.py

Gerekli Klasör Yapısı (çalıştırmadan önce):
    data/raw/
        └── Task07_Pancreas/
            ├── imagesTr/    ← Orijinal CT görüntüleri
            ├── labelsTr/    ← Orijinal segmentasyon maskeleri
            └── dataset.json ← Orijinal dataset tanımı
============================================================
"""

import os
import sys
import json
import shutil
import gzip
import time
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Dict, Optional

import numpy as np

# ============================================================
# PATH AYARLARI
# ⚠️ BU SATIRI DEĞİŞTİR: Proje kök dizinini ayarlayın
# ============================================================
BASE_PATH = Path(__file__).parent.parent
# Alternatif: BASE_PATH = Path("YOUR_PATH_HERE")

# Kaynak (Kaggle'dan indirilen ham veri)
RAW_DATA_PATH = BASE_PATH / "data" / "raw" / "Task07_Pancreas"

# Hedef (nnU-Net formatı)
NNUNET_RAW_PATH = BASE_PATH / "data" / "nnunet_raw" / "Dataset007_Pancreas"
IMAGES_TR_PATH  = NNUNET_RAW_PATH / "imagesTr"
LABELS_TR_PATH  = NNUNET_RAW_PATH / "labelsTr"
IMAGES_TS_PATH  = NNUNET_RAW_PATH / "imagesTs"

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
# BÖLÜM 1: KAYNAK VERİ DOĞRULAMA
# ============================================================
def validate_source_data(raw_path: Path) -> Tuple[List[Path], List[Path]]:
    """
    Kaynak verinin varlığını ve bütünlüğünü kontrol eder.

    Returns:
        (image_files, label_files): Sıralanmış dosya listesi
    Raises:
        FileNotFoundError: Kaynak klasör bulunamazsa
        ValueError: Görüntü/maske sayıları eşleşmezse
    """
    header("KAYNAK VERİ DOĞRULAMA")

    if not raw_path.exists():
        fail(f"Kaynak klasör bulunamadı: {raw_path}")
        print(f"""
  {YELLOW}Çözüm:{RESET}
  1. Kaggle'dan 'Medical Segmentation Decathlon' indirin
  2. Task07_Pancreas klasörünü şuraya çıkartın:
     {raw_path}
  
  İndirme linki:
  https://www.kaggle.com/datasets/andrewmvd/medical-segmentation-decathlon
        """)
        raise FileNotFoundError(f"Kaynak klasör bulunamadı: {raw_path}")

    # imagesTr klasörü
    images_dir = raw_path / "imagesTr"
    labels_dir = raw_path / "labelsTr"

    if not images_dir.exists():
        raise FileNotFoundError(f"imagesTr klasörü bulunamadı: {images_dir}")
    if not labels_dir.exists():
        raise FileNotFoundError(f"labelsTr klasörü bulunamadı: {labels_dir}")

    # Dosyaları listele (.nii ve .nii.gz)
    image_files = sorted([
        f for f in images_dir.iterdir()
        if f.name.endswith(".nii.gz") or f.name.endswith(".nii")
    ])
    label_files = sorted([
        f for f in labels_dir.iterdir()
        if f.name.endswith(".nii.gz") or f.name.endswith(".nii")
    ])

    if len(image_files) == 0:
        raise ValueError(f"imagesTr klasöründe NIfTI dosyası bulunamadı: {images_dir}")

    if len(image_files) != len(label_files):
        warn(f"Görüntü sayısı ({len(image_files)}) != Maske sayısı ({len(label_files)})")
        warn("Eşleşmeyen dosyalar atlanacak.")

    ok(f"Kaynak klasör bulundu: {raw_path}")
    ok(f"CT görüntüsü sayısı:   {len(image_files)}")
    ok(f"Maske sayısı:          {len(label_files)}")

    # Dosya boyutu tahmini
    total_size_gb = sum(f.stat().st_size for f in image_files + label_files) / (1024**3)
    info(f"Toplam veri boyutu:    {total_size_gb:.2f} GB")

    return image_files, label_files


# ============================================================
# BÖLÜM 2: EŞLEŞTIRME
# ============================================================
def match_images_and_labels(
    image_files: List[Path],
    label_files: List[Path]
) -> List[Tuple[Path, Path]]:
    """
    CT görüntülerini karşılık gelen maskelerle eşleştirir.

    MSD Task07 dosya formatı:
        imagesTr/pancreas_001.nii.gz
        labelsTr/pancreas_001.nii.gz

    Returns:
        [(image_path, label_path), ...] eşleşmiş çiftler
    """
    header("GORUNTU-MASKE ESLESTIRME")

    # Dosya adı bazında eşleştir (uzantı hariç)
    def stem(p: Path) -> str:
        name = p.name
        if name.endswith(".nii.gz"):
            return name[:-7]
        return p.stem

    image_dict = {stem(f): f for f in image_files}
    label_dict = {stem(f): f for f in label_files}

    common_keys = sorted(set(image_dict.keys()) & set(label_dict.keys()))
    only_images = set(image_dict.keys()) - set(label_dict.keys())
    only_labels = set(label_dict.keys()) - set(image_dict.keys())

    if only_images:
        warn(f"Maskesi olmayan {len(only_images)} görüntü atlanıyor: {list(only_images)[:3]}")
    if only_labels:
        warn(f"Görüntüsü olmayan {len(only_labels)} maske atlanıyor: {list(only_labels)[:3]}")

    pairs = [(image_dict[k], label_dict[k]) for k in common_keys]
    ok(f"Eslestirilen cift: {len(pairs)}")

    return pairs


# ============================================================
# BÖLÜM 3: DÖNÜŞÜM
# ============================================================
def convert_to_nnunet_format(
    pairs: List[Tuple[Path, Path]],
    images_tr_path: Path,
    labels_tr_path: Path,
    test_split: float = 0.2,
    images_ts_path: Optional[Path] = None,
    dry_run: bool = False
) -> Tuple[List[str], List[str]]:
    """
    Dosyaları nnU-Net formatına dönüştürür ve kopyalar.

    nnU-Net Dosya İsimlendirme Kuralı:
    -----------------------------------
    Eğitim CT:    case_XXXX_0000.nii.gz
                  └─────┘ └────┘ └────┘
                  case_id  mod  uzantı
    Eğitim maske: case_XXXX.nii.gz
    
    "_0000" suffix = modality 0 = CT
    MRI T1 olsaydı: _0000, T2: _0001 vb.

    Args:
        pairs:           (image, label) çiftleri
        images_tr_path:  imagesTr çıktı klasörü
        labels_tr_path:  labelsTr çıktı klasörü
        test_split:      Test için ayrılacak oran (0.2 = %20)
        images_ts_path:  imagesTs çıktı klasörü (test görüntüleri)
        dry_run:         True ise sadece simüle et, dosya kopyalama

    Returns:
        (training_cases, test_cases): Case ID listeleri
    """
    header("NNUNET FORMATINA DONUSUM")

    # Klasörleri oluştur
    images_tr_path.mkdir(parents=True, exist_ok=True)
    labels_tr_path.mkdir(parents=True, exist_ok=True)
    if images_ts_path:
        images_ts_path.mkdir(parents=True, exist_ok=True)

    total = len(pairs)
    n_test = int(total * test_split)
    n_train = total - n_test

    info(f"Toplam vaka:   {total}")
    info(f"Egitim:        {n_train} vaka (%{100-int(test_split*100)})")
    info(f"Test:          {n_test}  vaka (%{int(test_split*100)})")
    if dry_run:
        warn("DRY-RUN modu aktif: Dosyalar kopyalanmayacak.")

    training_cases = []
    test_cases = []

    for idx, (img_path, lbl_path) in enumerate(pairs):
        case_id = f"case_{idx+1:04d}"  # case_0001, case_0002, ...
        is_test = idx >= n_train

        # nnU-Net formatı dosya adları
        img_new_name = f"{case_id}_0000.nii.gz"  # modality 0 = CT
        lbl_new_name = f"{case_id}.nii.gz"

        if is_test:
            img_dest = images_ts_path / img_new_name if images_ts_path else None
            test_cases.append(case_id)
        else:
            img_dest = images_tr_path / img_new_name
            lbl_dest = labels_tr_path / lbl_new_name
            training_cases.append(case_id)

        # Dosyayı kopyala veya sembolik link oluştur
        if not dry_run:
            try:
                # --- CT görüntüsü ---
                if img_dest and not img_dest.exists():
                    src_name = img_path.name
                    if src_name.endswith(".nii.gz"):
                        shutil.copy2(img_path, img_dest)
                    elif src_name.endswith(".nii"):
                        # .nii → .nii.gz dönüşümü
                        _compress_nii(img_path, img_dest)
                    info(f"[{idx+1:3d}/{total}] {'[TEST] ' if is_test else '       '}{img_path.name} -> {img_new_name}")

                # --- Maske (sadece eğitim) ---
                if not is_test:
                    if not lbl_dest.exists():
                        src_name = lbl_path.name
                        if src_name.endswith(".nii.gz"):
                            shutil.copy2(lbl_path, lbl_dest)
                        elif src_name.endswith(".nii"):
                            _compress_nii(lbl_path, lbl_dest)

            except Exception as e:
                fail(f"Kopyalama hatasi [{case_id}]: {e}")
                continue
        else:
            print(f"  [SIM] {img_path.name} -> {img_new_name}")

    ok(f"Egitim vakalari: {len(training_cases)}")
    ok(f"Test vakalari:   {len(test_cases)}")

    return training_cases, test_cases


def _compress_nii(src: Path, dest: Path) -> None:
    """
    .nii dosyasını .nii.gz olarak sıkıştırır.
    Büyük 3D CT dosyaları için bellek verimli chunk okuma kullanır.
    """
    CHUNK = 1024 * 1024  # 1MB chunk
    with open(src, "rb") as f_in:
        with gzip.open(dest, "wb", compresslevel=6) as f_out:
            while True:
                chunk = f_in.read(CHUNK)
                if not chunk:
                    break
                f_out.write(chunk)


# ============================================================
# BÖLÜM 4: MASKE ETİKET DOĞRULAMA
# ============================================================
def validate_labels(labels_tr_path: Path, sample_n: int = 5) -> Dict:
    """
    Maske dosyalarındaki etiket değerlerini doğrular.
    Beklenen: 0 (background), 1 (pancreas), 2 (tumor)

    Args:
        labels_tr_path: labelsTr klasörü
        sample_n:       Kaç dosyayı örnekleyerek kontrol etsin

    Returns:
        Doğrulama istatistikleri sözlüğü
    """
    header("MASKE ETIKET DOGRULAMA")

    try:
        import nibabel as nib
    except ImportError:
        warn("nibabel kurulu degil, etiket doğrulama atlandı.")
        return {}

    label_files = sorted(labels_tr_path.glob("*.nii.gz"))
    if not label_files:
        warn("Kontrol edilecek maske bulunamadı.")
        return {}

    # Rastgele sample_n dosya seç
    import random
    sample_files = label_files[:sample_n]

    stats = {
        "total_files":    len(label_files),
        "sampled":        len(sample_files),
        "has_tumor":      0,
        "only_pancreas":  0,
        "empty":          0,
        "unique_labels":  set(),
        "tumor_ratio":    [],
    }

    for lbl_path in sample_files:
        try:
            img  = nib.load(str(lbl_path))
            data = img.get_fdata().astype(np.int32)

            unique = np.unique(data)
            stats["unique_labels"].update(unique.tolist())

            has_bg       = 0 in unique
            has_pancreas = 1 in unique
            has_tumor    = 2 in unique

            tumor_voxels = np.sum(data == 2)
            total_voxels = data.size

            if has_tumor:
                stats["has_tumor"] += 1
                stats["tumor_ratio"].append(tumor_voxels / total_voxels * 100)
            elif has_pancreas:
                stats["only_pancreas"] += 1
            else:
                stats["empty"] += 1

            status = (
                f"bg={'Y' if has_bg else 'N'} "
                f"pan={'Y' if has_pancreas else 'N'} "
                f"tumor={'Y' if has_tumor else 'N'} "
                f"vol={data.shape}"
            )
            info(f"  {lbl_path.name}: {status}")

        except Exception as e:
            warn(f"  {lbl_path.name} okunamadi: {e}")

    stats["unique_labels"] = sorted(stats["unique_labels"])

    print()
    ok(f"Toplam maske dosyasi: {stats['total_files']}")
    ok(f"Ornek alınan:         {stats['sampled']}")
    ok(f"Tumor iceren:         {stats['has_tumor']}")
    ok(f"Sadece pankreas:      {stats['only_pancreas']}")

    if stats["tumor_ratio"]:
        avg_tumor = sum(stats["tumor_ratio"]) / len(stats["tumor_ratio"])
        info(f"Ort. tumor hacmi:    %{avg_tumor:.3f}")

    ok(f"Bulunan etiketler:    {stats['unique_labels']}")

    expected = {0, 1, 2}
    found    = set(stats["unique_labels"])
    if not found.issubset({0, 1, 2}):
        warn(f"Beklenmeyen etiket değerleri bulundu: {found - expected}")
    else:
        ok("Etiket degerleri dogru: [0=background, 1=pancreas, 2=tumor]")

    return stats


# ============================================================
# BÖLÜM 5: DATASET.JSON OLUŞTURMA
# ============================================================
def create_dataset_json(
    output_path: Path,
    training_cases: List[str],
    test_cases: List[str],
    label_stats: Dict
) -> None:
    """
    nnU-Net v2 için zorunlu dataset.json dosyasını oluşturur.

    nnU-Net v2 dataset.json formatı:
    https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/dataset_format.md

    Kritik alanlar:
    - channel_names: Her modalite için isim (CT için normalizasyon önemli)
    - labels:        Segmentasyon sınıfları
    - numTraining:   Eğitim vaka sayısı
    - file_ending:   .nii.gz
    """
    header("DATASET.JSON OLUSTURULUYOR")

    dataset = {
        # nnU-Net v2 zorunlu alanlar
        "channel_names": {
            "0": "CT"
            # CT normalizasyonu: nnU-Net otomatik CT window/level yapar
            # MRI T1 olsaydı: "0": "T1"
            # MRI T2 olsaydı: "1": "T2"
        },
        "labels": {
            "background": 0,
            "pancreas":   1,
            "tumor":      2
        },
        "numTraining": len(training_cases),
        "file_ending": ".nii.gz",

        # Opsiyonel ama faydalı alanlar
        "name":            "Dataset007_Pancreas",
        "description":     "CT goruntulerinden pankreas ve tumor segmentasyonu",
        "reference":       "Medical Segmentation Decathlon - Task07",
        "licence":         "CC-BY-SA 4.0",
        "release":         "1.0",
        "created_at":      datetime.now().strftime("%Y-%m-%d %H:%M:%S"),

        # Eğitim vakalarının listesi
        "training": [
            {
                "image": f"./imagesTr/{case_id}_0000.nii.gz",
                "label": f"./labelsTr/{case_id}.nii.gz"
            }
            for case_id in training_cases
        ],

        # Test vakaları (etiket yok)
        "test": [
            f"./imagesTs/{case_id}_0000.nii.gz"
            for case_id in test_cases
        ],

        # İstatistikler (bilgi amaçlı)
        "_stats": {
            "n_training":       len(training_cases),
            "n_test":           len(test_cases),
            "label_stats":      {
                k: v for k, v in label_stats.items()
                if k not in ["unique_labels", "tumor_ratio"]
            }
        }
    }

    json_path = output_path / "dataset.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=4, ensure_ascii=False)

    ok(f"dataset.json olusturuldu: {json_path}")
    info(f"  numTraining: {dataset['numTraining']}")
    info(f"  Labels:      {dataset['labels']}")
    info(f"  file_ending: {dataset['file_ending']}")

    return json_path


# ============================================================
# BÖLÜM 6: ADIM 2 DOĞRULAMA
# ============================================================
def verify_nnunet_format(nnunet_raw_path: Path) -> bool:
    """
    Oluşturulan nnU-Net formatını doğrular.
    nnU-Net'in beklediği yapıyla karşılaştırır.
    """
    header("NNUNET FORMAT DOGRULAMA")

    errors = 0

    # 1. Zorunlu klasörler
    required_dirs = ["imagesTr", "labelsTr"]
    for d in required_dirs:
        path = nnunet_raw_path / d
        if path.exists():
            n = len(list(path.glob("*.nii.gz")))
            ok(f"{d}/ mevcut ({n} dosya)")
        else:
            fail(f"{d}/ EKSIK!")
            errors += 1

    # 2. dataset.json
    json_path = nnunet_raw_path / "dataset.json"
    if json_path.exists():
        with open(json_path) as f:
            ds = json.load(f)
        required_keys = ["channel_names", "labels", "numTraining", "file_ending"]
        for key in required_keys:
            if key in ds:
                ok(f"dataset.json['{key}'] = {ds[key]}")
            else:
                fail(f"dataset.json['{key}'] EKSIK!")
                errors += 1
    else:
        fail("dataset.json EKSIK!")
        errors += 1

    # 3. imagesTr dosya adı formatı kontrolü
    images = list((nnunet_raw_path / "imagesTr").glob("*.nii.gz"))
    if images:
        sample = images[0].name
        if "_0000.nii.gz" in sample:
            ok(f"Dosya isimlendirme dogru: {sample}")
        else:
            warn(f"Dosya ismi beklenmeyen formatta: {sample}")
            warn("Beklenen: case_XXXX_0000.nii.gz")

    # 4. imagesTr ve labelsTr sayı eşleşmesi
    n_img = len(list((nnunet_raw_path / "imagesTr").glob("*.nii.gz")))
    n_lbl = len(list((nnunet_raw_path / "labelsTr").glob("*.nii.gz")))
    if n_img == n_lbl:
        ok(f"Goruntu/maske sayisi esit: {n_img}")
    else:
        warn(f"Goruntu ({n_img}) != Maske ({n_lbl}) sayisi!")

    if errors == 0:
        ok("nnU-Net format doğrulaması BASARILI")
    else:
        fail(f"{errors} hata bulundu!")

    return errors == 0


# ============================================================
# SIMÜLASYON MODU: Gerçek veri yokken test için
# ============================================================
def create_dummy_data_for_testing(
    images_tr_path: Path,
    labels_tr_path: Path,
    n_cases: int = 5
) -> Tuple[List[str], List[str]]:
    """
    Gerçek veri olmadan pipeline'ı test etmek için sahte NIfTI dosyaları oluşturur.
    
    ⚠️ SADECE GELIŞTIRME/TEST AMAÇLIDIR.
    Gerçek veri mevcut olduğunda bu fonksiyon kullanılmaz.

    Boyutlar: (128, 128, 64) - küçük test boyutu
    """
    header("SIMÜLASYON: YAPAY TEST VERİSİ OLUŞTURULUYOR")
    warn("Bu mod sadece pipeline testı içindir!")
    warn("Gerçek Kaggle verisiyle değiştirin.")

    try:
        import nibabel as nib
    except ImportError:
        fail("nibabel kurulu degil!")
        return [], []

    images_tr_path.mkdir(parents=True, exist_ok=True)
    labels_tr_path.mkdir(parents=True, exist_ok=True)

    training_cases = []
    affine = np.eye(4)  # Birim dönüşüm matrisi

    for i in range(n_cases):
        case_id = f"case_{i+1:04d}"

        # Sahte CT görüntüsü: HU değerleri -1000 ile 1000 arasında
        ct_data = np.random.uniform(-1000, 1000, size=(128, 128, 64)).astype(np.float32)
        # Pankreas bölgesini simüle et (merkez küboid)
        ct_data[45:83, 45:83, 20:44] = np.random.uniform(50, 150, size=(38, 38, 24))

        ct_img = nib.Nifti1Image(ct_data, affine)
        ct_path = images_tr_path / f"{case_id}_0000.nii.gz"
        nib.save(ct_img, str(ct_path))

        # Sahte maske: 0=background, 1=pancreas, 2=tumor
        mask_data = np.zeros((128, 128, 64), dtype=np.uint8)
        mask_data[45:83, 45:83, 20:44] = 1  # Pankreas bölgesi

        # Bazı vakalarda tümör ekle
        if i % 2 == 0:  # Her ikinci vakada tümör var
            mask_data[55:65, 55:65, 25:35] = 2  # Tümör bölgesi

        mask_img = nib.Nifti1Image(mask_data, affine)
        mask_path = labels_tr_path / f"{case_id}.nii.gz"
        nib.save(mask_img, str(mask_path))

        tumor_str = "TUMOR" if (i % 2 == 0) else "     "
        ok(f"[{case_id}] {tumor_str} CT: {ct_data.shape} | Maske: {mask_data.shape}")
        training_cases.append(case_id)

    ok(f"{n_cases} yapay vaka olusturuldu")
    return training_cases, []


# ============================================================
# MAIN
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description="MSD Task07 verisetini nnU-Net v2 formatına dönüştür"
    )
    parser.add_argument(
        "--raw_path",
        type=str,
        default=str(RAW_DATA_PATH),
        help="Kaggle'dan indirilen Task07_Pancreas klasörü"
        # ⚠️ BU SATIRI DEĞİŞTİR: Kendi veri yolunuzu girin
    )
    parser.add_argument(
        "--test_split",
        type=float,
        default=0.2,
        help="Test için ayrılacak vaka oranı (varsayılan: 0.2)"
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Simüle et, gerçek kopyalama yapma"
    )
    parser.add_argument(
        "--simulate",
        action="store_true",
        help="Gerçek veri olmadan yapay veri oluştur (test amaçlı)"
    )
    parser.add_argument(
        "--n_dummy",
        type=int,
        default=5,
        help="Simülasyon modunda oluşturulacak vaka sayısı"
    )
    args = parser.parse_args()

    raw_path = Path(args.raw_path)

    print(f"""
{BOLD}{CYAN}
==============================================================
  ADIM 2: DATASET HAZIRLAMA - nnU-Net FORMAT DONUSUMU
==============================================================
{RESET}
""")

    start_time = time.time()

    try:
        # ---- SIMÜLASYON MODU ----
        if args.simulate or not raw_path.exists():
            if not args.simulate:
                warn(f"Kaynak veri bulunamadı: {raw_path}")
                warn("SIMULASYON modu ile devam ediliyor (test icin yapay veri).")
                warn("Gercek veri için: --raw_path YOLUNUZU_GIRIN")
            
            training_cases, test_cases = create_dummy_data_for_testing(
                IMAGES_TR_PATH,
                LABELS_TR_PATH,
                n_cases=args.n_dummy
            )

        # ---- GERÇEK VERİ MODU ----
        else:
            # 1. Kaynak veriyi doğrula
            image_files, label_files = validate_source_data(raw_path)

            # 2. Görüntü-maske eşleştirme
            pairs = match_images_and_labels(image_files, label_files)

            # 3. nnU-Net formatına dönüştür
            training_cases, test_cases = convert_to_nnunet_format(
                pairs=pairs,
                images_tr_path=IMAGES_TR_PATH,
                labels_tr_path=LABELS_TR_PATH,
                test_split=args.test_split,
                images_ts_path=IMAGES_TS_PATH,
                dry_run=args.dry_run
            )

        # 4. Maske etiketlerini doğrula
        label_stats = validate_labels(LABELS_TR_PATH, sample_n=5)

        # 5. dataset.json oluştur
        create_dataset_json(
            output_path=NNUNET_RAW_PATH,
            training_cases=training_cases,
            test_cases=test_cases,
            label_stats=label_stats
        )

        # 6. nnU-Net formatını doğrula
        is_valid = verify_nnunet_format(NNUNET_RAW_PATH)

        # ---- ÖZET ----
        elapsed = time.time() - start_time
        header("ADIM 2 TAMAMLANDI")

        ok(f"Egitim vakalari: {len(training_cases)}")
        ok(f"Test vakalari:   {len(test_cases)}")
        ok(f"Format gecerli:  {is_valid}")
        info(f"Gecen sure:      {elapsed:.1f} saniye")
        info(f"Cikti klasoru:   {NNUNET_RAW_PATH}")

        print(f"""
  {BOLD}Sonraki Adim (ADIM 3):{RESET}
  nnU-Net preprocessing:
  
    {CYAN}nnUNetv2_plan_and_preprocess -d 007 -c 2d --verify_dataset_integrity{RESET}

  {YELLOW}NOT: Once ortam degiskenlerini ayarlayin!{RESET}
    $env:nnUNet_raw          = "{NNUNET_RAW_PATH.parent}"
    $env:nnUNet_preprocessed = "{BASE_PATH / 'data' / 'nnunet_preprocessed'}"
    $env:nnUNet_results      = "{BASE_PATH / 'data' / 'nnunet_results'}"
""")

        return 0

    except FileNotFoundError as e:
        fail(str(e))
        return 1
    except Exception as e:
        fail(f"Beklenmeyen hata: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
