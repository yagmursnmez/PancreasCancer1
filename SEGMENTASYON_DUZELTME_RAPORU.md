# Segmentasyon Düzeltme Raporu — 2026-08-06

## Son durum

Üretim web hattı artık üç aşamalıdır:

1. **nnU-Net v2 2D:** pankreas ve tümör adayı üretir.
2. **MONAI DiNTS 3D:** bağımsız hacimsel pankreas ve tümör adayı üretir.
3. **TotalSegmentator 3D full:** pankreasın anatomik konumunu bağımsız olarak doğrular.

2B ve 3B tümör adayları birleştirildikten sonra son maskeye ancak şu koşullarla girer:

- 3B pankreas kapısı 3–250 mL ve en az 3 kesit olmalı.
- Pankreas modelleri arası Dice en az 0,20 olmalı.
- Her tümör bileşeni en az 0,30 mL ve en az 2 kesit olmalı.
- Bileşen pankreasın 5 mm anatomik bandında en az %10 desteklenmeli.
- 2B/3B tümör modelleri en az 0,10 Dice ve 0,10 mL doğrudan ortak alan göstermeli.
- Anatomik model çalışmazsa veya pankreası doğrulayamazsa ham maske yayımlanmaz; sonuç **karar verilemedi** olur.

## Düzeltilen ana hatalar

- DICOM hacmi eskiden kalıcı olarak `[-125, 225]` aralığına kırpılıyordu. Artık özgün HU korunuyor; yalnızca FOV dolgu/metal uçları `[-1024, 3071]` aralığına sınırlandırılıyor.
- Eski NIfTI affine hasta yönelimi ve konumunu yok sayıyordu. Artık DICOM IOP/IPP/PixelSpacing kullanılarak LPS→RAS affine, qform ve sform yazılıyor.
- Toraks/boyun serileri pankreas modeline gidebiliyordu. Negatif puanlı abdomen-dışı seriler artık reddediliyor; en az 16 dosya ve 10 gerçek uzamsal kesit aranıyor.
- `tumor_voxels > 50` tek başına tümör kararı veriyordu. Bu kural kaldırıldı.
- Voksel oranından uydurulan “% doğruluk/güven” kaldırıldı.
- Rastgele simülasyon maskesi geri dönüşleri kaldırıldı; başarısızlıkta sistem kapalı kalıyor.
- Boş tahmin klasöründe etiketi kendisiyle karşılaştırıp Dice=1,00 üreten doğrulama geri dönüşü kaldırıldı.
- Tek GPU’daki eşzamanlı web istekleri model kilidiyle sıraya alındı.

## Ölçümler

### Gerçek nnU-Net fold-0 doğrulaması — 46 vaka

- Pankreas Dice: **0,7698**
- Tümör Dice: **0,3104**
- Tümör tespiti: **34/46**

Rapor: `metrics/validation_report_20260806_121041.json`

`metrics/validation_report_20260729_120712.json` geçersizdir; etiketi kendisiyle karşılaştıran eski geri dönüş nedeniyle Dice=1,00 göstermiştir.

### PanTS ile tam fold-0 kalibrasyonu — 46 vaka

Eski 8 pozitif vaka denetimindeki **0,8537** değeri seçilmiş örneklem yanlılığı taşıdığı için üretim performansı olarak kullanılmamaktadır.

- Eski nnU-Net tümör Dice ortalaması: **0,2984**
- nnU-Net + MedFormer PanTS (eşik 0,40): **0,5724**
- nnU-Net + R-Super PanTS (eşik 0,30): **0,5635**
- Üçlü model, MedFormer 0,45 + R-Super 0,45: **0,5832**
- Üçlü model medyan Dice: **0,6304**
- Pankreas Dice: nnU-Net **0,8017**, PanTS birleşimi **0,8312**

Raporlar: `metrics/medformer_msd_fold0_full.json`, `metrics/rsuper_msd_fold0_full.json`, `metrics/three_model_msd_fold0_full.json`

### PATIENT975891 uçtan uca sonuç

- Seçilen DICOM: 157 kesitlik portal-venöz abdomen serisi
- Spacing: 0,8079 × 0,8079 × 3,0 mm
- 3B model uyum Dice: **0,8097**
- Son pankreas hacmi: **74,498 mL**
- nnU-Net ham tümör: **1.230 voksel**
- DiNTS ham tümör: **0 voksel**
- Reddedilen ham tümör: **1.230 / 1.230 voksel**
- Son doğrulanmış tümör hacmi: **0,0 mL**
- Sonuç: **Doğrulanmış tümör adayı saptanmadı**
- Toplam süre: **342,55 saniye**

Son görsel: `web/static/results/PATIENT975891_ultra_seg.png`

### 6 Ağustos 2026 performans ve interaktif 3B doğrulaması

- TotalSegmentator anatomik kapısı artık ilk çalışır; fiziksel olarak geçersiz/boş pankreas kapısında nnU-Net ve DiNTS güvenlik için hiç çalıştırılmaz.
- Kapı geçerliyse X/Y çözünürlüğü değiştirilmeden, pankreasın axial aralığı iki yönde 80 mm payla kırpılır (en az 96 kesit).
- PATIENT975891 üzerinde model girdisi **157 → 96 kesit** oldu (`%38,85` hacim azalması).
- İlk üç-model çalışma süresi **342,55 → 239,58 saniye** oldu (`%30,1` azalma).
- Yeni ve önceki son pankreas maskesi Dice uyumu **0,9969**; tümör kararı iki çalışmada da negatiftir.
- Tek kesitlik **683** ham tümör vokseli fiziksel kesit eşiğiyle reddedildi.
- Aynı NIfTI + aynı checkpoint + aynı karar ayarlarının ikinci çalışması doğrulanmış maske önbelleğinden **0,24 saniyede** geldi.
- DICOM ZIP artık sonuç sayfasını bekletmeden arka planda hazırlanır.
- Doğrulanmış maskeden otomatik interaktif Plotly 3B HTML üretilir; sonuç ekranındaki **İnteraktif 3B Görüntüle** düğmesi yeni sekmede açar.

Doğrulama çıktısı:

`web/static/results/PATIENT975891_roi_validation_3d_interactive.html`

### Negatif kontrol

Eski `denemetumorsuz` maskesinde **136.998** yanlış pankreas vokseli vardı. Bağımsız 3B kapı 0 pankreas vokseli üretti; yeni filtre son maskeyi tamamen boşalttı ve sonucu **karar verilemedi** yaptı.

## Tıbbi sınırlar

- Bu yazılım klinik karar verme veya kesin tanı aracı değildir.
- “Doğrulanmış tümör adayı saptanmadı” ifadesi kanseri dışlamaz.
- MSD Task07 eğitim/doğrulama verileri pankreas tümörlü portal-venöz CT’lerden oluşur; sağlıklı negatif kohort yoktur. Bu nedenle gerçek negatif özgüllük, farklı cihaz/faz ve dış-merkez performansı ayrıca uzman etiketli veriyle ölçülmelidir.
- Kusursuz/hatasız çalışma garanti edilemez. Sistem belirsiz durumda pozitif/negatif uydurmak yerine karar vermemek üzere tasarlanmıştır.

## Geri dönüş

Başlangıç geri dönüş noktası:

`checkpoints/rollback_20260806_before_segmentation_fix/GERI_DONUS_NOKTASI.md`

Kaynak arşivi SHA-256:

`BBC1F2B2957658012B41E840903C95666FFF0F993B90C68533283B32FF20CAC6`

İnteraktif 3B ve hız değişikliklerinden hemen önceki ikinci geri dönüş noktası:

`checkpoints/rollback_20260806_before_3d_speed_fix/GERI_DONUS_NOKTASI.md`

Kaynak arşivi SHA-256:

`A21A6ECC98A1E2BBC8BEF7387C2FC7DA4188D202EE9841B4C90514FF882B2CC5`
