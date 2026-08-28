# PancreasCancerDetection

CT görüntülerinde araştırma amaçlı pankreas ve tümör adayı segmentasyonu — nnU-Net 2D + DiNTS 3D + MedFormer PanTS + R-Super PanTS uzamsal konsensüsü ve TotalSegmentator 3D geometrik kalite kapısı. Çıktı uzman doğrulaması veya klinik tanı değildir.

## Proje Yapısı

```
PancreasCancer/
├── data/
│   ├── raw/                         # Kaggle'dan indirilen orijinal NIfTI dosyaları
│   ├── nnunet_raw/                  # nnU-Net formatına çevrilmiş veriler
│   │   └── Dataset007_Pancreas/
│   │       ├── imagesTr/            # Eğitim CT'leri (case_XXXX_0000.nii.gz)
│   │       ├── labelsTr/            # Eğitim maskeleri (case_XXXX.nii.gz)
│   │       ├── imagesTs/            # Test CT'leri
│   │       └── dataset.json
│   ├── nnunet_preprocessed/         # nnU-Net ön işleme (otomatik)
│   ├── nnunet_results/              # Eğitilmiş model ağırlıkları
│   └── inference_output/            # Segmentasyon sonuçları
├── scripts/
│   ├── utils.py                     # Ortak yardımcı fonksiyonlar
│   ├── prepare_dataset.py           # ADIM 2: Veri hazırlama
│   ├── validate_metrics.py          # ADIM 5: Doğrulama metrikleri
│   ├── inference.py                 # ADIM 6-7: Inference + Classification
│   ├── segmentation_postprocess.py  # 3B anatomik kapı ve fiziksel bileşen filtresi
│   ├── audit_tumor_ensemble.py      # Ensemble doğrulama denetimi
│   └── reconstruct_3d.py            # ADIM 8: 3D Rekonstrüksiyon
├── web/
│   ├── app.py                       # Flask web uygulaması
│   ├── templates/
│   └── static/
├── logs/
├── metrics/
├── notebooks/
├── setup_project.py                 # ADIM 1: Proje kurulumu
├── verify_step1.py                  # ADIM 1: Kurulum doğrulama
├── requirements.txt
├── requirements-3d.txt              # Ayrı 3B model ortamı
├── setup.bat                        # Windows kurulum
├── setup.sh                         # Linux/Mac kurulum
└── config.json                      # Merkezi konfigürasyon
```

## Hızlı Başlangıç

### 1. Kurulum (Windows)
```batch
setup.bat
```

### 2. Sanal Ortamı Aktif Et
```powershell
.\venv\Scripts\activate
```

### 3. nnU-Net Ortam Değişkenleri (PowerShell)
```powershell
$env:nnUNet_raw          = "C:\...\data\nnunet_raw"
$env:nnUNet_preprocessed = "C:\...\data\nnunet_preprocessed"
$env:nnUNet_results      = "C:\...\data\nnunet_results"
```

### 4. Veriyi İndir
Kaggle'dan [MSD Task07 Pancreas](https://www.kaggle.com/datasets/andrewmvd/medical-segmentation-decathlon) veri setini `data/raw/` klasörüne indirin.

### 5. Adımları Çalıştır
```bash
python scripts/prepare_dataset.py        # ADIM 2
nnUNetv2_plan_and_preprocess -d 007 -c 2d  # ADIM 3
nnUNetv2_train 007 2d 0                  # ADIM 4
python scripts/validate_metrics.py       # ADIM 5
python scripts/inference.py              # ADIM 6-7
python web/app.py                        # ADIM 9
```

## Etiketler

| Değer | Anlamı |
|-------|--------|
| 0 | Arka plan (Background) |
| 1 | Pankreas |
| 2 | Tümör |

## Metrikler

**Segmentasyon:** Dice Score, IoU
**Sınıflandırma:** Accuracy, Precision, Recall, F1, ROC-AUC

## Üretim karar mantığı

- DICOM yüklemesinde portal-venöz abdomen serisi seçilir; toraks/boyun reddedilir.
- Hasta geometrisi IOP/IPP/PixelSpacing ile NIfTI affine'e taşınır.
- nnU-Net 2D ve MONAI DiNTS 3D adayları birleştirilir.
- TotalSegmentator 3D full pankreası bağımsız olarak doğrular.
- MedFormer PanTS ve R-Super PanTS, 80 mm üç eksenli ROI'de bağımsız olasılık haritaları üretir.
- 46 uzman maskeli doğrulama olgusunda seçilen 0,45/0,45 eşikleriyle üçlü ana sınır; 0,60 çekirdek ve 0,30 duyarlı sınır kaydedilir.
- Tümör bileşeni en az 0,30 mL, en az 2 kesit ve pankreasın 5 mm bandında en az %10 destekli olmalıdır.
- 2B ve 3B tümör maskeleri en az 0,10 Dice ve 0,10 mL doğrudan ortak alan göstermelidir.
- Tam voxel kesişimi olmayan fakat aynı anatomik bölgede kalan adaylar ayrıca 5 mm fiziksel yakınlıkla ölçülür; yakınlık desteği yetersizse aday silinip "negatif" yapılmaz, belirsiz kontur olarak korunur.
- Anlamlı tek-model veya uzlaşmaz aday varsa sonuç üç-durumlu olarak `indeterminate` döner. Tam profilde MedFormer ve R-Super bu aday üzerinde hakem olarak gerçekten çalışır; iki hakem uzlaşırsa aday doğrulanır.
- Anatomik doğrulama başarısızsa ham maske gösterilmez ve karar verilmez.

Ayrıntılı ölçüm ve değişiklikler: [SEGMENTASYON_DUZELTME_RAPORU.md](SEGMENTASYON_DUZELTME_RAPORU.md)

## NVIDIA CUDA dogrulamasi

GPU secimi Windows Gorev Yoneticisi'ndeki degisken `GPU 0/GPU 1` etiketlerine
veya belirli bir kart modeline bagli degildir. `CUDA_VISIBLE_DEVICES` fiziksel
NVIDIA aygitini secer; model alt surecleri bu gorunur listenin mantiksal
`cuda:0` aygitini kullanir. Servis, PyTorch'un bildirdigi aygit UUID'sini NVML
ile eslestirerek telemetriyi ayni fiziksel karttan okur.

`verify_nvidia_gpu.bat` yaklasik 30 saniyelik gercek CUDA yuku olusturur.
Gorev Yoneticisi kullanilacaksa NVIDIA kartinin grafik basligindan **CUDA** veya
**Compute_0** motoru secilmelidir; varsayilan 3D grafiginin `%0` olmasi CUDA
hesabinin yapilmadigini kanitlamaz. Uygulama arayuzu anlik deger ile gecmis
asama tepesini ayirir; bitmis asama ozeti acikca `CANLI DEGIL` diye etiketlenir.

`run_web.bat`, calistiricilari Windows Grafik Ayarlari'nda yuksek performansli
GPU'ya yonlendirir. NVIDIA CUDA kullanilamazsa servis CPU'ya sessizce dusmez.
`cudaMallocAsync` ve `LAZY` yalniz yurutme/bellek mekanizmasidir; model
hassasiyetini veya segmentasyon kurallarini degistirmez.

Web arayuzunun varsayilani **Hizli dogrulanmis analiz** modudur. Ayni CT'nin
icerik karmasi, model/config imzasi ve geometrisi eslesirse birebir kayitli
maske kullanilir; DICOM->NIfTI donusumu de secilen serinin tam dosya icerigiyle
anahtarlanan ayri bir onbellekten yeniden kullanilir. Yeni CT'de tam model
zinciri yine calisir. **Tum modelleri zorla yeniden calistir** secenegi yalniz
tanilama/karsilastirma icin onbellegi atlar.

Her taze kosunun komutu, baslatici PID'i, saniyelik butun-aygit sayaclari,
model surec agacina ait CUDA PID'leri, CPU/RAM/disk-I/O ve ham stdout kaydi
`metrics/gpu_runs/<audit-id>/` altinda tutulur. Butun-aygit tepe degeri tek
basina kanit sayilmaz: bir asama ancak model surec agacina ait CUDA PID'i ile
GPU yuk/VRAM etkinligi ayni telemetri orneginde birlikte gorulurse dogrulanir. Gecersiz/sicrama yapan
surucu guc degerleri (or. bosta 752.7 W) rapor tepesine alinmaz.

Gercek model zincirini onbellegi atlayarak ve NVIDIA telemetrisi kaydederek
dogrulamak icin:

```powershell
.\venv\Scripts\python.exe scripts\verify_model_gpu.py web\static\uploads\denemetumorlu_0000.nii.gz --case-id gpu_model_test --bypass-cache --output metrics\gpu_model_test.json
```

14 Agustos 2026 son kontrollu 512x512x497 tam-hacim testinde bes modelin tamami
ayri CUDA PID'leriyle dogrulandi: tepe NVIDIA kullanimi `%100`, aktif CUDA
ortalamasi `%78.6`, tepe VRAM `5386 MiB`, tepe guc `115.3 W` ve tepe SM saati
`2002 MHz` oldu. Sorunlu web kosusunda `599.000 sn` olan bes model toplami,
PanTS bellek/sayfalama ve tekrar hesaplama hatalari giderildikten sonra
`440.547 sn` oldu (`%26.5` azalma); dogrudan taze zincir `518.235 sn` surdu.
R-Super sirasinda minimum bos RAM `17.6 MiB` degerinden `5994.8 MiB` degerine,
son birlestirme/yazma evresi `53.303 sn` degerinden `16.385 sn` degerine geldi.
Model, TTA/ROI, cozumurluk, hassasiyet, esik veya maske kurali azaltilmadi.

Yalniz TotalSegmentator'in iki GPU tahmini `27.97 + 25.65 = 53.62 sn` surer;
diger dort model bu noktada henuz baslamamistir. Yeni tam hacimleri bes modelle
bir dakikanin altina indirmek olcum yapilan 6 GB NVIDIA laptop GPU'da bu ayarlarla
fiziksel olarak mumkun degildir. Ayrintili karsilastirma ve kanitlar
`metrics/FRESH_GPU_BOTTLENECK_REPORT_20260814.md` dosyasindadir.

Kullanici logundaki sorunlu taze kosu 14:02:57-14:16:21 arasinda gerceklesti;
29 dakika degildi. Log daha sonra baslayan ikinci bir kosuyu da icerdigi icin
toplam zaman 29 dakika gibi gorundu. Ilk kosuda da 5/5 asama NVIDIA CUDA ile
dogrulanmisti; ekran goruntuleri modellerin CPU on-isleme, model yukleme veya
NIfTI diske yazma aralarinda alinmisti. Eski bes model asamasi toplami `678.736
sn`, yeni olcum `386.844 sn` oldu (`%43.0` azalma). Yeni kanitlar
`metrics/gpu_runs/20260814T153455_verify-1786710895_718430a3/` ve ozet
`metrics/gpu_bottleneck_fix_20260814.json` dosyasindadir. Yeni taze CUDA
kosusunda 130285568 voxel icinde yalniz 6 pankreas sinir voxeli sayisal
degiskenlik gostermis, tumor voxelleri ayni kalmistir; onbellek sonucu ise
referansla birebirdir.

18 Ağustos 2026 aday-hakem düzeltmesinde, daha önce iki ilk modelin tam voxel
kesişimi olmadığı için yanlış biçimde negatifleştirilen 512x512x294 çalışma
hızlı profilde `indeterminate` olarak 3,265 mL belirsiz adayla korundu. Tam
profilden geçen taze 5/5 CUDA koşusunda MedFormer/R-Super Dice uyumu 0,6521
oldu ve 3,196 mL ortak tümör segmentasyon adayı üretildi. Hızlı/tam süreler
aynı cihazda sırasıyla 171,1 ve 306,7 saniyeydi; kanıtlar
`metrics/longitudinal_2_hasta_fix_balanced.json` ve
`metrics/longitudinal_2_hasta_fix_full.json` dosyalarındadır.

DICOM seri secimindeki metadata taramasi piksel verisini okumadan 32 I/O
iscisine kadar paralel calisir. ZIP girisleri yol guvenligi korunarak 8 MB
tamponla acilir. Hacim verisi 16 GB sistem RAM'ini doldurmamak icin gecici
disk-eslemeli akista tutulur; sinir agi tensorleri model alt surecinde secili NVIDIA
VRAM'ine tasinir.

## Notlar

- GPU belleği 8GB altındaysa `--batch_size 2` kullanın
- Sadece fold 0 ile eğitim yapılmaktadır
- 3B modeller ayrı `.venv_totalseg` ortamında çalışır; CUDA uyumlu PyTorch gerekir.
- PanTS modelleri 6 GB GPU'da 96³, %50 örtüşmeli kayan pencereyle çalışır; yollar ve eşikler `config.json > pants_refinement` altındadır.
- Anatomik kapı, nnU-Net ve DiNTS artık geniş pankreas ROI'sinde çalışır; ilk analiz hacme göre birkaç dakika sürebilir, aynı doğrulanmış hacim sonraki çalışmalarda önbellekten alınır.
- Sonuç sayfasındaki **İnteraktif 3B Görüntüle** düğmesi doğrulanmış pankreas/tümör yüzeyini açar; HTML dosyaları `data/inference_output/3d_reconstructions/<vaka>/` altında tutulur.
- Maskeli DICOM ZIP sonuç gösterildikten sonra arka planda hazırlanır; indirme düğmesi paket tamamlanınca otomatik etkinleşir.
- Araştırma amaçlıdır; klinik tanı aracı değildir.
