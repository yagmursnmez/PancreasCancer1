# PancreasAI NVIDIA CUDA Yurutme ve Kanit Politikasi

## Aygit secimi

- Fiziksel NVIDIA aygiti `CUDA_VISIBLE_DEVICES` ile secilir (varsayilan `0`).
- Model surecleri gorunur listenin mantiksal `cuda:0` aygitini kullanir.
- PyTorch gercek aygit adini ve UUID'sini okur; NVML telemetrisi ayni UUID ile
  eslestirilir.
- Herhangi bir NVIDIA CUDA karti kabul edilir. Windows Gorev Yoneticisi sira
  numarasi ve belirli bir GeForce/RTX model adi konfigurasyonda kullanilmaz.
- CUDA yoksa CPU'ya sessiz gecis yapilmaz.

## Canli telemetri ile gecmis tepe ayrimi

Analiz penceresi uc durumu ayri gosterir:

1. `CANLI NVIDIA CUDA HESABI`: model surec agacina ait CUDA PID'i o anda
   gorulmustur; yuk/VRAM degeri anlik ornektir.
2. `ANLIK GPU ETKIN DEGIL`: model asamasi CPU on-isleme, agirlik yukleme,
   yeniden ornekleme veya I/O evresindedir. Baska bir surecin GPU yuku modele
   yazilmaz.
3. `SON ASAMA OZETI - CANLI DEGIL`: asama bitmistir; degerler yalnizca bitmis
   asamanin tarihsel tepesidir.

Gorev Yoneticisi varsayilan 3D motorunu gosterebilir. Karsilastirma icin NVIDIA
kartinin grafik basligindan `CUDA` veya `Compute_0` secilmelidir; uygulamanin
kanit kaynagi yine NVIDIA surucu API'sidir.

## Kanit kosulu

Butun-kart kullanimi tek basina kanit sayilmaz. Bir model asamasi ancak su iki
kosul birlikte saglanirsa dogrulanir:

- CUDA PID'i baslatilan modelin surec agacina aittir.
- Ayni telemetri orneginde secili NVIDIA aygitinda yuk veya anlamli VRAM
  artisi gorulur.

Ham ornekler, model stdout'u, sahipli/sahipsiz CUDA PID'leri ve ozetler
`metrics/gpu_runs/<audit-id>/` altinda tutulur. Surucunun gecis aninda uretebildigi
fiziksel olarak gecersiz guc sicramalari (ornegin kart bostayken 752.7 W)
gecersiz isaretlenir ve tepe guc hesabina katilmaz.

## Hiz politikasi

Onbellek farkli/yeni bir CT'nin suresini kisaltmaz. Bu nedenle model hesabi iki
acik profile ayrildi:

- `balanced` (varsayilan): TotalSegmentator + nnU-Net + DiNTS. nnU-Net TTA
  kapali, DiNTS pencere ortusmesi %50, iki ek PanTS modeli kapali. Bu ayar
  512x512x497 test hacminde asama olcumlerine gore yaklasik 4-6 dakika hedefler.
- `full_ensemble`: onceki bes modelin tamami, nnU-Net TTA ve DiNTS %62.5
  ortusme ile aynen korunur; ayni hacimde yaklasik 7-10 dakika beklenir.
- Profil, model/config imzasinin parcasidir; iki profil birbirinin onbellek
  maskesini kullanamaz.
- `fresh_gpu` yalniz birebir ayni CT icin var olabilecek kaydi zorla atlar.

Hizli profil sonuc-degistirmeyen bir optimizasyon gibi sunulmaz: arayuzde model
sayisi ve kalite/sure farki acik yazilir. Tek 497-kesitli vakadaki kontrollu
denemede tam ve hizli cekirdek fuzyon arasinda pankreas Dice 0.9137, tumor Dice
0.7931 ve tumor hacmi 10.151 mL / 10.860 mL olculmustur. Tam profil bu nedenle
ayri secenek olarak korunmustur.

## 17 Agustos taze istek kritik yol analizi

09:30:32.383 ile 09:39:58.866 arasindaki istek toplam 566.483 saniyedir:

- 1357 dosyanin alinmasi: 9.62 s
- DICOM baslik taramasi: 8.95 s
- DICOM piksel donusumu ve NIfTI kaydi: 28.72 s
- Bes modelin dis sureleri: 451.406 s
- Model gecisleri, yeniden ornekleme, gzip I/O ve sonuc hazirlama: yaklasik 67.8 s

Model toplami: 106.375 s TotalSegmentator, 142.297 s nnU-Net, 106.781 s DiNTS,
48.469 s MedFormer ve 47.484 s R-Super. Bu kosuda termal/guc kisma kaniti yoktur;
aktif orneklerde yaklasik 80-91% GPU, 97-105 W ve 1807-1899 MHz gorulmustur.

Eski 3-4 dakikalik `PATIENT975891` olcumu esit is yukune ait degildir: 157
kesit, 512x512x96 ROI ve uc model kullanmistir (239.58 s). Yeni CT 497 kesit,
512x512x222 ROI ve tam profilde bes modeldir. Giris kesit sayisi 3.16 kat,
PanTS ek yuku de 95.953 saniyedir.

Kontrollu tek-degisken deneyleri:

- TotalSegmentator `--fast`: 81.200 -> 74.661 s; pankreas maskesi Dice 0.8323.
  Yalniz 6.5 s kazandirdigi ve kapinin kendisini belirgin degistirdigi icin
  kullanilmadi.
- nnU-Net TTA kapali: 140.843 -> 88.684 s. Ham tumor Dice 0.7156.
- DiNTS ortusme %50: 93.046 -> 59.623 s. Ham tumor Dice 0.8444.

Son iki ayar sadece acikca etiketlenen `balanced` profile kondu; tam profilde
hicbir esik veya model ayari dusurulmedi.

Uygulama sonrasinda ayni 512x512x497 CT, onbellek zorla kapali olarak gercek
`balanced` zincirinden gecirildi. Model yolu toplam 262.041 s (4:22) surdu:
TotalSegmentator 66.813 s, nnU-Net 91.078 s, DiNTS 65.063 s; kalan 39.087 s
ROI/gzip, model gecisleri, fuzyon ve kayittir. Uc asamanin ucunde de modele ait
CUDA PID'i goruldu; tepe GPU %100, tepe VRAM 5374 MiB ve tepe guc 114.8 W'tir.
Kanıt: `metrics/gpu_runs/20260817T105342_balanced-benchmark-20260817_ca80297a/`.

Olculen 512x512x497 hacimde 17 Agustos taze kosusunun bes model toplami
451.406 saniyedir: 106.375 s anatomik model, 142.297 s 2B model, 106.781 s 3B
tumor modeli ve 95.953 s iki PanTS modeli. Modeller 6 GB kartta VRAM nedeniyle
paralel calistirilamaz. Ayni hacmin dogrulanmis model onbellegi daha once 0.75
saniyede yuklenmistir; DICOM yukleme/donusum suresi bundan ayridir.

## Dogrulama

- Hizli CUDA yuku: `verify_nvidia_gpu.bat`
- Gercek model zinciri: `scripts/verify_model_gpu.py`
- Politika/API: `http://127.0.0.1:5000/api/gpu_status`
- Kosu kaniti: `http://127.0.0.1:5000/api/gpu_audit/<audit-id>`

Bu proje arastirma amaclidir; GPU dogrulamasi klinik dogruluk veya uzman onayi
anlamina gelmez.
