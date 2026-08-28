"""
============================================================
ADIM 3: PREPROCESSING — nnU-Net Plan & Preprocess
============================================================
Bu script nnU-Net'in otomatik preprocessing pipeline'ını çalıştırır.

nnU-Net preprocessing ne yapar?
1. Dataset'i analiz eder (voxel spacing, yoğunluk istatistikleri)
2. 2D için optimal patch size, batch size hesaplar
3. CT normalizasyon parametrelerini belirler (HU windowing)
4. Resampling planı oluşturur
5. Verileri ön işler ve kaydeder (RAM uyarısı!)

RAM UYARISI:
    Preprocessing tüm CT hacimlerini RAM'e yükler.
    MSD Task07 için ~16-32 GB RAM gerekebilir.
    Önerimiz: Başka programları kapatın.

Kullanım:
    python scripts/run_preprocessing.py

    VEYA doğrudan nnU-Net CLI ile:
    nnUNetv2_plan_and_preprocess -d 007 -c 2d --verify_dataset_integrity
============================================================
"""

import os
import sys
import subprocess
import time
import json
import psutil
from pathlib import Path

# ============================================================
# PATH AYARLARI
# ⚠️ BU SATIRI DEĞİŞTİR: Proje kök dizinini ayarlayın
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
# BÖLÜM 1: ORTAM DEĞİŞKENLERİNİ AYARLA
# ============================================================
def setup_nnunet_env() -> dict:
    """
    nnU-Net için zorunlu ortam değişkenlerini kontrol eder ve ayarlar.
    Mevcut değilse config.json'dan okur.

    Returns:
        nnU-Net ortam değişkenleri sözlüğü
    """
    header("ORTAM DEGISKENLERI KONTROL EDILIYOR")

    # config.json'dan oku
    config_path = BASE_PATH / "config.json"
    config = {}
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)

    paths = config.get("paths", {})

    env_map = {
        "nnUNet_raw":          paths.get("nnunet_raw",
                                str(BASE_PATH / "data" / "nnunet_raw")),
        "nnUNet_preprocessed": paths.get("nnunet_preprocessed",
                                str(BASE_PATH / "data" / "nnunet_preprocessed")),
        "nnUNet_results":      paths.get("nnunet_results",
                                str(BASE_PATH / "data" / "nnunet_results")),
    }

    for key, default_val in env_map.items():
        current = os.environ.get(key)
        if current:
            ok(f"{key} = {current}")
        else:
            os.environ[key] = default_val
            warn(f"{key} ayarlanmadı, varsayılan kullanıldı: {default_val}")

    return env_map


# ============================================================
# BÖLÜM 2: SİSTEM KAYNAK KONTROLÜ
# ============================================================
def check_system_resources():
    """
    RAM ve disk alanını kontrol eder.
    Preprocessing için minimum gereksinimler bildirilir.
    """
    header("SISTEM KAYNAKLARI KONTROL EDILIYOR")

    # RAM
    ram = psutil.virtual_memory()
    ram_total_gb  = ram.total / (1024**3)
    ram_avail_gb  = ram.available / (1024**3)
    ram_used_pct  = ram.percent

    info(f"RAM Toplam:    {ram_total_gb:.1f} GB")
    info(f"RAM Kullanılabilir: {ram_avail_gb:.1f} GB ({100-ram_used_pct:.0f}% bos)")

    if ram_avail_gb < 8:
        warn(f"Kullanılabilir RAM {ram_avail_gb:.1f} GB — preprocessing yavaş olabilir!")
        warn("Önerimiz: Diğer uygulamaları kapatın.")
    elif ram_avail_gb < 16:
        warn(f"RAM {ram_avail_gb:.1f} GB — preprocessing çalışır ama dikkatli olun.")
    else:
        ok(f"RAM yeterli: {ram_avail_gb:.1f} GB kullanılabilir")

    # Disk
    disk = psutil.disk_usage(str(BASE_PATH))
    disk_free_gb = disk.free / (1024**3)
    info(f"Disk Bos: {disk_free_gb:.1f} GB")
    if disk_free_gb < 50:
        warn(f"Disk alanı az ({disk_free_gb:.1f} GB). Preprocessing ~20-50 GB yer kaplar.")
    else:
        ok(f"Disk alani yeterli: {disk_free_gb:.1f} GB")

    # CPU
    cpu_count = psutil.cpu_count(logical=True)
    info(f"CPU çekirdeği: {cpu_count}")

    return {
        "ram_avail_gb": ram_avail_gb,
        "disk_free_gb": disk_free_gb,
        "cpu_count":    cpu_count
    }


# ============================================================
# BÖLÜM 3: nnU-Net PLAN & PREPROCESS
# ============================================================
def run_plan_and_preprocess(
    dataset_id: str = "007",
    configuration: str = "2d",
    num_processes: int = 4,
    verify_integrity: bool = True
) -> bool:
    """
    nnUNetv2_plan_and_preprocess komutunu çalıştırır.

    Bu komut şunları yapar:
    1. Dataset'i fingerprint çıkarır (spacing, intensity stats)
    2. 2D konfigürasyon için plan oluşturur:
       - Patch size (tipik: 512x512 veya 640x640)
       - Batch size (GPU belleğine göre)
       - Normalizasyon parametreleri
    3. Tüm vakaları preprocess eder (resample + normalize)

    Args:
        dataset_id:       Dataset numarası (007)
        configuration:    "2d" — GPU kısıtı nedeniyle 3d kullanmıyoruz
        num_processes:    Paralel işlem sayısı (RAM'e göre ayarlayın)
        verify_integrity: Dataset bütünlüğünü doğrula

    ⚠️ RAM UYARISI: num_processes arttıkça RAM kullanımı artar.
       8GB RAM için num_processes=2 öneririz.
       16GB RAM için num_processes=4 güvenli.
    """
    header("nnU-Net PLAN & PREPROCESS BASLIYOR")

    # Komut oluştur
    cmd = [
        "nnUNetv2_plan_and_preprocess",
        "-d", dataset_id,
        "-c", configuration,
        "-np", str(num_processes),
    ]

    if verify_integrity:
        cmd.append("--verify_dataset_integrity")

    info(f"Komut: {' '.join(cmd)}")
    info(f"Dataset ID:    {dataset_id}")
    info(f"Konfigürasyon: {configuration} (2D — GPU kisiti nedeniyle)")
    info(f"Paralel islem: {num_processes}")
    warn("Bu islem 5-60 dakika sürebilir (veri büyüklüğüne göre).")
    warn("RAM kullanımı yüksek olacak — diğer programları kapatın!")
    print()

    start_time = time.time()

    try:
        # nnU-Net CLI çalıştır (gerçek zamanlı çıktı ile)
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
            env=os.environ.copy()
        )

        # Gerçek zamanlı çıktı göster
        for line in process.stdout:
            line = line.rstrip()
            if line:
                print(f"  [nnUNet] {line}")

        process.wait()
        elapsed = time.time() - start_time

        if process.returncode == 0:
            ok(f"Plan & Preprocess TAMAMLANDI ({elapsed/60:.1f} dakika)")
            return True
        else:
            fail(f"Preprocessing basarisiz! Return code: {process.returncode}")
            return False

    except FileNotFoundError:
        fail("nnUNetv2_plan_and_preprocess komutu bulunamadi!")
        warn("Cozum: pip install nnunetv2")
        warn("Veya: .\\venv\\Scripts\\pip install nnunetv2")
        return False
    except Exception as e:
        fail(f"Hata: {e}")
        return False


# ============================================================
# BÖLÜM 4: PREPROCESSING ÇIKTISINI DOĞRULA
# ============================================================
def verify_preprocessing_output(
    dataset_id: str = "007",
    configuration: str = "2d"
) -> bool:
    """
    Preprocessing çıktısının oluşturulduğunu doğrular.

    Beklenen çıktı yapısı:
    nnunet_preprocessed/
    └── Dataset007_Pancreas/
        ├── dataset.json
        ├── dataset_fingerprint.json   ← voxel istatistikleri
        ├── nnUNetPlans.json           ← eğitim planı
        └── nnUNetPlans_2d/            ← preprocess edilmiş veriler
            ├── case_0001.npz
            ├── case_0001.pkl
            └── ...
    """
    header("PREPROCESSING CIKTISI DOGRULANIYOR")

    preprocessed_base = Path(os.environ.get(
        "nnUNet_preprocessed",
        str(BASE_PATH / "data" / "nnunet_preprocessed")
    ))

    dataset_name = f"Dataset{dataset_id}_Pancreas"
    dataset_dir  = preprocessed_base / dataset_name
    plans_2d_dir = dataset_dir / f"nnUNetPlans_{configuration}"

    checks = {
        "Dataset klasörü":         dataset_dir,
        "dataset_fingerprint.json": dataset_dir / "dataset_fingerprint.json",
        "nnUNetPlans.json":         dataset_dir / "nnUNetPlans.json",
        f"nnUNetPlans_{configuration}/": plans_2d_dir,
    }

    all_ok = True
    for name, path in checks.items():
        if path.exists():
            if path.is_dir():
                n = len(list(path.iterdir()))
                ok(f"{name} ({n} dosya/klasor)")
            else:
                size_kb = path.stat().st_size / 1024
                ok(f"{name} ({size_kb:.1f} KB)")
        else:
            warn(f"{name} henüz oluşturulmamış")
            all_ok = False

    # nnUNetPlans.json içeriğini göster
    plans_json = dataset_dir / "nnUNetPlans.json"
    if plans_json.exists():
        try:
            with open(plans_json) as f:
                plans = json.load(f)
            configs = plans.get("configurations", {})
            if "2d" in configs:
                cfg_2d = configs["2d"]
                print(f"\n  {BOLD}nnU-Net 2D Plan Ozeti:{RESET}")
                if "patch_size" in cfg_2d:
                    info(f"  Patch size:    {cfg_2d['patch_size']}")
                if "batch_size" in cfg_2d:
                    info(f"  Batch size:    {cfg_2d['batch_size']}")
                if "n_conv_per_stage_encoder" in cfg_2d:
                    info(f"  Encoder katmanları: {cfg_2d['n_conv_per_stage_encoder']}")
        except Exception as e:
            warn(f"nnUNetPlans.json okunamadi: {e}")

    return all_ok


# ============================================================
# BÖLÜM 5: PREPROCESSING RAPORUNU KAYDET
# ============================================================
def save_preprocessing_report(env_vars: dict, resources: dict, success: bool):
    """Preprocessing sonuçlarını JSON raporuna kaydeder."""
    report = {
        "step":         "ADIM3_Preprocessing",
        "timestamp":    time.strftime("%Y-%m-%d %H:%M:%S"),
        "success":      success,
        "env_vars":     env_vars,
        "resources":    resources,
        "config": {
            "dataset_id":    "007",
            "configuration": "2d",
        }
    }
    report_dir = BASE_PATH / "metrics"
    report_dir.mkdir(exist_ok=True)
    report_path = report_dir / "preprocessing_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4, ensure_ascii=False)
    ok(f"Rapor kaydedildi: {report_path}")


# ============================================================
# MAIN
# ============================================================
def main():
    print(f"""
{BOLD}{CYAN}
==============================================================
  ADIM 3: PREPROCESSING - nnU-Net Plan & Preprocess
==============================================================
{RESET}
""")

    # 1. Ortam değişkenlerini ayarla
    env_vars = setup_nnunet_env()

    # 2. Sistem kaynaklarını kontrol et
    resources = check_system_resources()

    # RAM'e göre paralel işlem sayısını otomatik ayarla
    ram_gb = resources["ram_avail_gb"]
    if ram_gb < 8:
        num_processes = 1
        warn(f"Dusuk RAM ({ram_gb:.1f}GB): num_processes=1 kullanılıyor")
    elif ram_gb < 16:
        num_processes = 2
        info(f"Orta RAM ({ram_gb:.1f}GB): num_processes=2 kullanılıyor")
    else:
        num_processes = 4
        info(f"Yeterli RAM ({ram_gb:.1f}GB): num_processes=4 kullanılıyor")

    # 3. Preprocessing'i çalıştır
    success = run_plan_and_preprocess(
        dataset_id="007",
        configuration="2d",
        num_processes=num_processes,
        verify_integrity=True
    )

    # 4. Çıktıyı doğrula
    verify_preprocessing_output(dataset_id="007", configuration="2d")

    # 5. Rapor kaydet
    save_preprocessing_report(env_vars, resources, success)

    # 6. Özet
    header("ADIM 3 OZET")
    if success:
        ok("Preprocessing basariyla tamamlandi!")
        print(f"""
  {BOLD}Sonraki Adim (ADIM 4) - Model Egitimi:{RESET}

    {CYAN}nnUNetv2_train 007 2d 0{RESET}

  Parametreler:
    007   = Dataset ID
    2d    = Konfigürasyon (GPU kisiti: 3d kullanmiyoruz)
    0     = Fold numarasi (sadece fold 0 ile basliyoruz)
""")
        return 0
    else:
        fail("Preprocessing basarisiz!")
        warn("Yukaridaki hata mesajlarini kontrol edin.")
        warn("Cozum onerileri:")
        warn("  1. nnU-Net ortam degiskenlerini ayarlayin")
        warn("  2. dataset.json dosyasinin dogru oldugunu kontrol edin")
        warn("  3. RAM yetersizse diger programlari kapatin")
        return 1


if __name__ == "__main__":
    sys.exit(main())
