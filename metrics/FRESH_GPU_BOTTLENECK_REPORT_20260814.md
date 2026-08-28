# Taze RTX 3060 Darboğaz Raporu — 14 Ağustos 2026

## Kapsam

C:\Users\monster\Desktop\denemetümörlü dizininden üretilen
web/static/uploads/denemetumorlu_0000.nii.gz hacmi, sonuç önbelleği zorla
atlanarak beş modelin tamamında yeniden çalıştırıldı. Model ağırlıkları,
çözünürlük, ROI sınırları, kayan pencere/TTA, hassasiyet, eşikler ve maske
kuralları değiştirilmedi.

## Sonuç

| Ölçüm | Sorunlu web koşusu | Güncel taze koşu | Değişim |
|---|---:|---:|---:|
| Beş model aşaması | 599.000 sn | 440.547 sn | -%26,5 |
| Model audit toplamı | 734.340 sn | 517.519 sn | -%29,5 |
| TotalSegmentator | 115.000 sn | 90.328 sn | -%21,5 |
| nnU-Net | 203.344 sn | 151.797 sn | -%25,4 |
| DiNTS | 122.734 sn | 106.953 sn | -%12,9 |
| MedFormer PanTS | 79.735 sn | 45.937 sn | -%42,4 |
| R-Super PanTS | 78.187 sn | 45.532 sn | -%41,8 |
| Son PanTS sonrası birleştirme/yazma | 53.303 sn | 16.385 sn | -%69,3 |

Doğrudan taze doğrulamanın dış kronometresi 518.235 sn oldu. Güncel servis
başlangıcı ayrıca üç CUDA Python ortamını istek kabul edilmeden doğrular; ilk
isteğin kritik yolundan ölçülen yaklaşık 9,3 sn sabit CUDA başlatma işi
çıkarılmıştır. Bu model sonucu önbelleği değildir ve segmentasyon saklamaz.

## NVIDIA kanıtı

- Doğrulanan model aşaması: 5/5
- Ayrı CUDA model PID'i: 5
- Tepe RTX kullanımı: %100
- Aktif CUDA örnekleri ortalaması: %78,6
- Tepe VRAM: 5386 / 6144 MiB
- Tepe güç: 115,3 W
- Tepe SM saati: 1995 MHz (dış ölçümde 2002 MHz)
- CUDA aygıtı: cuda:0; aynı fiziksel kartın Windows adı: GPU 1

Bu veriler Intel GPU'ya/CPU'ya model düşmesi olmadığını gösterir. GPU'nun sıfır
göründüğü aralıklar model yükleme, CPU yeniden örnekleme, bağlı bileşen analizi
ve NIfTI okuma/yazma evreleridir.

## Bulunan gerçek regresyon

Sorunlu web koşusunda PanTS kodu, yalnız pankreasla ilgili beş kanal gerekirken
26 sınıfın tüm tam-ROI olasılık hacimlerini CPU RAM'inde biriktiriyordu. Ayrıca
iki PanTS çıktısı 512×512×497 tam hacme genişletiliyor, tam hacimdeki bağlı
bileşen işlemleri bir kez boşa hesaplanıp hemen ardından tekrar ediliyordu.

| PanTS bellek ölçümü | Sorunlu koşu | Güncel koşu |
|---|---:|---:|
| MedFormer tepe süreç RSS | 6336,3 MiB | 2209,4 MiB |
| MedFormer minimum boş RAM | 1070,6 MiB | 6255,2 MiB |
| R-Super tepe süreç RSS | 7253,2 MiB | 2339,5 MiB |
| R-Super minimum boş RAM | 17,6 MiB | 5994,8 MiB |

17,6 MiB boş RAM Windows sayfalamasını tetikliyor ve GPU'yu veri bekler halde
bırakıyordu. Güncel kod kullanılmayan çıkış kanallarını GPU hesabından sonra,
GPU→CPU aktarımından önce ayırır; ağı yine 26 kanalın tamamını hesaplar. PanTS
birleştirmesi aynı fiziksel ROI içinde tek kez yapılır ve sonuç tam hacimdeki
aynı koordinatlara yapıştırılır. Eski tam hacim dizileri de bir sonraki modelden
önce serbest bırakılır.

Sorunlu çalışmanın başladığı anda AnyDesk ekran yakalama süreçleri, DWM,
GPU-Z canlı grafik, Görev Yöneticisi, tarayıcılar ve masaüstü animasyonları
birlikte CPU/RAM baskısı oluşturuyordu. Aynı kodla daha önceki kontrollü taze
koşunun model toplamı 386.844 sn idi. AnyDesk önceliği High seviyesinden
Normal, masaüstü animasyonu süreci BelowNormal seviyesine indirildi.

## Çıktı bütünlüğü

Güncel ve referans NIfTI hacimleri aynı (512, 512, 497) şekle, aynı affine'e
ve aynı voxel aralığına sahiptir. 130.285.568 voxelin yalnızca 8 tanesi
farklıdır (%0,00000614): 2 × 1→0, 4 × 0→1, 2 × 1→2. Pankreas voxel
toplamı aynıdır; tümör farkı +2 voxel = +0,001497 mL'dir. Sekiz değişimin tümü
mevcut sınıf sınırına tam bir voxel uzaklıktadır. Tarihsel taze CUDA koşuları da
aynı 23 sınır koordinatında 4–16 voxel değişkenlik göstermiştir; bu desen CUDA
kayan nokta/kernel sırasının eşik sınırındaki doğal değişkenliğiyle uyumludur.

ROI-tam hacim birleştirme eşdeğerliği ve süreç önceliğinin geri yüklenmesi
kalıcı testlere eklendi. Derleme kontrolü ve 30/30 güvenlik testi geçti. Web
inference sırasında yalnız CPU ön/birleştirme işi geçici AboveNormal önceliğe
alınır ve sonunda eski seviyesine döner. Arka plan DICOM ZIP üretimi de yeni
bir model zinciriyle aynı anda RAM/disk tüketmeyecek şekilde serileştirildi.

## Bir dakika hedefinin fiziksel alt sınırı

Son TotalSegmentator stdout kaydında yalnız iki gerçek GPU tahmini 27,97 sn +
25,65 sn = 53,62 sn sürmektedir. Bu sürede nnU-Net, DiNTS, MedFormer ve
R-Super henüz çalışmamıştır. Güncel beş aşamada NVIDIA kullanımının sıfırdan
büyük olduğu saniyelik örnek sayısı bile 206'dır. Kart 6 GB olduğu ve ilk üç
model tek başına yaklaşık 5,3–5,4 GB VRAM kullandığı için bu modeller güvenli
biçimde paralel çalıştırılamaz.

Dolayısıyla yeni bir hacmin beş modeli aynı ayarlarla RTX 3060 6 GB üzerinde
60 saniyenin altında bitmesi mümkün değildir. Bu sınır bir CUDA seçim hatası
değil, gerçek model hesap miktarı ve tek kartın kapasitesidir. Bir dakikanın
altına inmek için model/TTA/ROI/çözünürlük/hassasiyet değişikliği veya belirgin
biçimde daha hızlı ve daha yüksek VRAM'li donanım gerekir; bu denetimde
kullanıcının yasakladığı analiz değişikliklerinin hiçbiri uygulanmamıştır.

## Kanıt dosyaları

- Güncel özet: metrics/gpu_fresh_memory_fix_20260814.json
- Güncel ayrıntılı audit: metrics/gpu_runs/20260814T164127_verify-1786714886_02334d51/gpu_run_report.json
- Sorunlu web audit'i: metrics/gpu_runs/20260814T160913_d56ee479-7103-4b75-b34f-74bfa13bc9f5_25c8ba15/gpu_run_report.json
- Önceki kontrollü taze audit: metrics/gpu_runs/20260814T153455_verify-1786710895_718430a3/gpu_run_report.json
