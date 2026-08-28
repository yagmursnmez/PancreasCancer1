# ESKI / TARIHSEL RAPOR - PancreasAI NVIDIA GPU Yapilandirmasi

> Bu belge 14 Agustos 2026'daki tek makineye ozgu eski yapilandirmayi anlatir.
> Guncel, donanimdan bagimsiz politika icin `NVIDIA_GPU_KULLANIM_RAPORU.md`
> kullanilmalidir. Uygulama artik kart modeli veya Windows GPU sira numarasi
> aramaz.

## Sonuc

Projenin butun model ortamlari Windows Gorev Yoneticisi'ndeki **GPU 1**, yani
**NVIDIA GeForce RTX 3060 Laptop GPU** uzerinde CUDA ile calisacak sekilde
sabitlenmistir.

Windows ve CUDA ayni karta farkli numara verir:

- Windows Gorev Yoneticisi: `GPU 1`
- NVIDIA/PyTorch CUDA: `cuda:0`

Bu bilgisayarda yalniz bir NVIDIA karti vardir. CUDA Intel GPU'yu listelemez.
Bu nedenle `CUDA_VISIBLE_DEVICES=0`, Windows GPU 1 olan RTX 3060'i secer.
Degeri `1` yapmak RTX 3060'i gorunmez hale getirir.

## Uygulanan ayarlar

1. `.env` icinde `CUDA_VISIBLE_DEVICES=0`, `CUDA_DEVICE_ORDER=PCI_BUS_ID` ve
   `CUDA_REQUIRED=True` ayarlanmistir.
2. nnU-Net komutuna acikca `-device cuda` verilmistir.
3. TotalSegmentator komutunda `--device gpu` kullanilmaktadir.
4. DiNTS modeli ve tum tensorleri CUDA aygitina tasinmaktadir.
5. PanTS modelleri `--gpu 0` ile tek NVIDIA kartina sabitlenmistir.
6. Web, DiNTS/TotalSegmentator ve PanTS alt sureclerinin ortamlari ayri ayri
   RTX 3060 testi yapar. Kart bulunamazsa CPU'ya sessiz gecis yapilmaz.
7. Windows `UserGpuPreferences` kaydinda proje calistiricilarinin tamami
   `GpuPreference=2` (Yuksek performans) olarak ayarlanmistir.
8. `run_web.bat` her baslangicta bu Windows kayitlarini yeniler ve gercek CUDA
   tensoru calistirmadan web servisini acmaz.
9. DiNTS kayan pencere batch degeri, 6 GB RTX kartta test edilerek 1'den 4'e
   cikarilmistir.
10. Windows Optimus uyumluluk bayragi `SHIM_MCCOMPAT_ENABLE_GPU=1` web ve tum
    model alt sureclerine aktarilmistir.
11. DICOM seri basliklari, piksel verisini okumadan, 32 eszamanli I/O iscisine
    kadar paralel taranir. Yalniz seri secimi icin gereken etiketler okunur.
12. ZIP cikarma akisi guvenli yol denetimini koruyarak 8 MB tamponla yazilir.
13. DICOM/NIfTI hacmi RAM'i tasirmamak icin disk-eslemeli `memmap` ile tutulur;
    model tensörleri model surecinde RTX 3060 VRAM'ine tasinir.

14. Bozuk ana ve 3B sanal ortam baslaticilari, mevcut Python 3.12 calisma
    zamanina yeniden baglanmistir; iki ortam da ayri CUDA tensor islemiyle
    dogrulanmistir.
15. `requirements.txt`, CPU PyTorch kurulmasini engellemek icin
    `torch==2.7.1+cu118` ve `torchvision==0.22.1+cu118` olarak sabitlenmistir.
16. `setup.bat` ve `run_all.bat` artik genel `python` komutuna veya sessiz CPU
    gecisine guvenmez; proje ortamini kullanir ve RTX 3060 yoksa durur.
17. Kasitli CUDA hazir-bekleme dongusu tamamen kaldirilmistir. Web servisi
    bosta beklerken GPU kullanimi ve NVIDIA bellek tahsisi yoktur. RTX 3060
    yalniz nnU-Net, TotalSegmentator, DiNTS ve PanTS model surecleri analiz
    yaparken kullanilir. Durum `/api/gpu_status` adresinden denetlenebilir.
18. Python 3.12 temel calisma zamani proje icindeki `.python_runtime` klasorune
    kalici olarak alinmistir. Ana ortam, 3B ortam ve PanTS artik Codex onbellegine
    bagli degildir; bilgisayar yeniden baslatilsa veya dis onbellek temizlense de
    proje kendi Python'u ve CUDA paketleriyle acilir.
19. Analiz ilerleme penceresine NVIDIA surucusunden okunan gercek GPU 1
    kullanimi, VRAM, sicaklik, guc ve performans durumu eklenmistir. nnU-Net'in
    CUDA tahmini ile tahmin sonrasindaki CPU yeniden ornekleme/dosya yazma evresi
    artik ayri metinlerle gosterilir. CPU disari aktarma evresinde GPU etiketi
    kasitli olarak kaldirilir; bosta sahte yuk olusturulmaz.
20. CUDA alt sureclerinde `CUDA_MODULE_LOADING=LAZY` ve
    `PYTORCH_CUDA_ALLOC_CONF=backend:cudaMallocAsync` kullanilir. Bu iki ayar
    yalniz modul/bellek yurutmesini degistirir; agirlik, sayisal hassasiyet,
    TTA, ROI, esik ve maske kurallarina dokunmaz.
21. Windows model alt surecleri `ABOVE_NORMAL_PRIORITY_CLASS` ile baslatilir.
    Bu, CPU yeniden ornekleme evrelerinde masaustu rekabetini azaltir; gercek
    zamanli/yuksek oncelik kullanilmadigi icin sistem yanit verebilir kalir.
22. `scripts/verify_model_gpu.py`, gercek bes-model zincirini istege bagli
    onbellek atlama, asama bazli NVIDIA telemetrisi ve voxel karsilastirmasiyla
    yeniden uretilebilir bicimde denetler.
23. Web formunun varsayilani `fresh_gpu` yapilmistir. Ayni CT olsa bile agir
    modeller yeniden calisir; onbellek ancak kullanici ayri `cache_allowed`
    secenegini secerse kullanilir.
24. Her taze kosu `metrics/gpu_runs/<audit-id>/` altinda model komutu,
    baslatici PID, NVIDIA CUDA PID, saniyelik yuk/VRAM/guc/P-state, ham stdout
    ve JSON ozetini kalici kaydeder. Sonuc sayfasi bu kanita dogrudan baglanir.
25. `GPU_EVIDENCE_REQUIRED=True` ile her agir model asamasi icin GPU yuku,
    en az 64 MiB VRAM artisi veya yeni NVIDIA CUDA PID kanitlarindan biri
    zorunludur. Ucu de sifirsa CPU sonucu gibi devam edilmez; analiz hata verir.
26. Web servisi debug, DICOM secimi, ilerleme ve model audit ozetlerini
    `logs/pancreas_web.log` dosyasina donusumlu olarak kalici yazar; model ham
    stdout ve her NVIDIA ornegi ilgili `metrics/gpu_runs/<audit-id>/` klasorundedir.

## Canli test sonucu

Projenin `scripts/verify_nvidia_gpu.py` testi sirasinda olculen degerler:

- NVIDIA kullanimi: `%100`
- GPU gucu: yaklasik `93 W`
- Performans durumu: `P0`
- SM saat hizi: yaklasik `1965-1972 MHz`
- Son yuk testi: `1515` CUDA matris islemi, `25.70` saniye
- PyTorch: `2.7.1+cu118`
- CUDA runtime: `11.8`
- Fiziksel kart: `NVIDIA GeForce RTX 3060 Laptop GPU`
- Bosta servis: `%0` NVIDIA kullanimi ve `0 MiB` model bellegi
- Analiz/model testi: `%100` NVIDIA kullanimi
- GPU calisma politikasi: `http://127.0.0.1:5000/api/gpu_status`

Gercek web analizi sirasinda ayni vaka icin NVIDIA surucusunden 20 ornek
alinmistir: `%63-%89` GPU kullanimi, `3345 MiB` VRAM, `106-109 W`, `P0` ve
`69-73 C`. CUDA sureci PID `24900` ile proje Python'u olarak gorulmustur.
nnU-Net gunlugu `perform_everything_on_device: True` kaydetmis; 888/888 GPU
tahmini yaklasik 56 saniyede tamamlandiktan sonra 18.79 saniyelik CPU yeniden
ornekleme ve disari aktarma evresine gecmistir.

Uc model ortami da gercek `cuda:0` tensor testiyle ayri ayri dogrulanmistir.

### 14 Agustos 2026 tam model dogrulamasi

512x512x497 boyutlu gercek hacimde onbellek atlanarak TotalSegmentator,
nnU-Net, DiNTS, MedFormer PanTS ve R-Super PanTS Merlin agirliklarinin tamami
calistirildi. Son yurutme ayarlarindan sonraki
`metrics/gpu_model_after_runtime_tuning_20260814.json` dosyasindaki dogrudan
NVIDIA surucu olcumu:

- Toplam soguk zincir: `482.844 sn`
- Onceki ayni test: `501.547 sn`
- Kazanc: `18.703 sn` (`%3.73`)
- Tepe NVIDIA kullanimi: `%100`
- Aktif orneklerde ortalama NVIDIA kullanimi: `%85.3`
- Tepe VRAM: `5259 / 6144 MiB` (oncekinden `660 MiB` daha dusuk)
- Tepe guc: `114.59 W`
- TotalSegmentator: `67.031 sn`
- nnU-Net: `127.953 sn`
- DiNTS: `95.046 sn`
- MedFormer PanTS: `53.844 sn`
- R-Super PanTS Merlin: `53.906 sn`

Ayni dogrulanmis hacmin uretim onbellegi testi `0.718 sn` surdu ve referans
maskeyle `0` farkli voxel verdi. Yeni bir tam hacmin bes buyuk modelle soguk
calismasi RTX 3060 6 GB kartta saniyeler degil dakikalar surer; bunu saniyelere
zorlamak ancak model/TTA/ROI/cozunurluk azaltmakla mumkun olurdu ve bu projede
ozellikle uygulanmamistir.

Soguk CUDA yeniden calistirmalari, ayarlar degismeden de sinir voxellerinde
cok kucuk sayisal degiskenlik gosterebilir. Ayar oncesi kosu referanstan 7,
ayar sonrasi kosu 8 voxel farkliydi; iki soguk kosunun tumor etiketi tamamen
ayniydi ve 130285568 voxel icinde yalniz 11 pankreas sinir voxeli farkliydi.
Referans maske uzerine yazilmadi. Onbellek testi ise referansla `0` farkli
voxel verdi.

### 14 Agustos 2026 kullanici logu ve kalici PID kaniti

Paylasilan `son log.txt` dosyasindaki 11:44:38 kaydi acikca
`Dogrulanmis maske onbellekten alindi (Agir modeller tekrar calistirilmadi)`
diyordu. GPU-Z'de 159 MiB/P8/%0 ve Gorev Yoneticisi'nde %0 gorulmesinin nedeni
bu istekte model alt sureclerinin hic baslatilmamis olmasiydi; Intel'e dusen
bir CUDA modeli yoktu.

Duzeltme sonrasi ayni 512x512x497 CT onbellek atlanarak yeniden calistirildi.
`metrics/gpu_runs/20260814T120336_verify-1786698215_bf830548/gpu_run_report.json`
kaydi:

- TotalSegmentator: PID `16872`, CUDA PID `2188`, tepe `%99`, `5235 MiB`
- nnU-Net: PID `21256`, CUDA PID `14728`, tepe `%100`, `5231 MiB`
- MONAI DiNTS: PID `17412`, CUDA PID `14404`, tepe `%97`, `5237 MiB`
- MedFormer PanTS: PID/CUDA PID `8348`, tepe `%100`, `2901 MiB`
- R-Super PanTS: PID/CUDA PID `21784`, tepe `%100`, `2933 MiB`
- Genel: 5/5 asama dogrulandi, `395` kalici ornek, tepe `115.1 W`

Tam ozet `metrics/gpu_fix_full_20260814.json` dosyasindadir. Tumor maskesi ve
tumor voxel sayisi referansla tamamen aynidir. 130285568 voxel icinde yalniz
9 pankreas sinir voxeli soguk CUDA kosularinin kucuk sayisal degiskenligini
gostermistir; model, esik, ROI, cozumurluk veya maske kurali degistirilmemistir.

Ayni anda alinan Windows WDDM sayacinda Python icin `3D=%0` gorunurken NVIDIA
surucu sayaci `%100`, `P0` ve yaklasik `92 W` gosterdi. Bu surucu, CUDA hesap
motorunu Gorev Yoneticisi'ne guvenilir bicimde aktarmadigi icin GPU 1 grafiginin
`%0` kalmasi CPU/Intel inference kaniti degildir.

## Bilerek uygulanmayan hatali/zararli zorlamalar

- `__NV_PRIME_RENDER_OFFLOAD` ve `__GLX_VENDOR_LIBRARY_NAME` Linux/GLX
  degiskenleridir; Windows'ta CUDA aygiti secmezler.
- `ctypes.windll.LoadLibrary('nvapi64.dll')` yalniz DLL'yi yukler. Bir Python
  degiskenini `NvOptimusEnablement` adiyla sonradan atamak EXE'den disari
  aktarilan Optimus bayragini olusturmaz. Windows `GpuPreference=2` kaydi
  calistirici bazinda uygulanmistir.
- `d3d11.dll` render/Direct3D motorudur; PyTorch CUDA cekirdeklerini hizlandiran
  alternatif bir hesap motoru degildir. CUDA surucusu PyTorch tarafindan
  dogrudan baslatilir ve gercek tensor testiyle denetlenir.
- DICOM etiketleri ilk 1 KB icinde bulunmak zorunda degildir. Ilk 1 KB'yi zorla
  kesmek gecerli tetkikleri bozabileceginden bunun yerine `specific_tags` ve
  `stop_before_pixels=True` kullanilir; piksel verisi baslik taramasinda okunmaz.
- Binlerce DICOM'u tamamen RAM'de tutmak 16 GB bellekte tasma ve kopyalama
  maliyeti yaratir. Bu nedenle gecici hacim sifir-kopyaya yakin `memmap` olarak
  tutulur ve is bitince silinir; kalici gereksiz ara dosya birakilmaz.

## Gorev Yoneticisi'nde dogru grafik

1. `Ctrl+Shift+Esc` ile Gorev Yoneticisi'ni acin.
2. `Performans > GPU 1` ekranina girin.
3. Grafiklerden birinin sol ustundeki grafik adina tiklayin; listede varsa
   `CUDA` veya `Compute_0` motorunu secin.
4. Bu bilgisayardaki NVIDIA WDDM surucusu CUDA PID'sinin hesap motoru sayacini
   Gorev Yoneticisi'ne her zaman dogru aktarmiyor. Bu nedenle `3D`, `Copy` veya
   diger grafikler model gercekte calisirken bile `%0` gosterebilir.
5. Proje klasorundeki `verify_gpu1.bat` dosyasina cift tiklayin.
6. Yaklasik 30 saniye boyunca RTX 3060 hesaplama grafigi yukselecektir.

Dogru ve dogrudan olcum, analiz ilerleme penceresine eklenen `GPU 1 | RTX 3060`
satiridir; bu veri NVIDIA surucusunun `nvidia-smi` sayacindan gelir. Ekran
goruntusundeki `1.4/6.0 GB` adanmis bellek ve `72 C` sicaklik da RTX 3060'in
modeli yukledigini gosterir. O anda yuzdenin sifira dusmesi, GPU tahmininden
sonraki CPU yeniden ornekleme/dosya yazma evresine denk gelebilir. Intel GPU
0'in yuksek gorunmesi Chrome, Windows masaustu yoneticisi ve arayuz ciziminden
kaynaklanabilir; model tensorlerinin Intel'de calistigi anlamina gelmez.

## NVIDIA surucu durumu

13 Agustos 2026 tarihli denetimde NVIDIA surucusu `610.88`, WDDM CUDA UMD
surumu `13.3` olarak goruldu. Kurulu PyTorch CUDA 11.8 calisma zamaniyla geriye
uyumlu calisti; surucu engeli veya surucu guncellemesi gereksinimi bulunmadi.

## Onbellek nedeniyle GPU'nun bos gorunebilecegi durum

Ayni CT hacmi daha once basariyla analiz edildiyse dogrulanmis maske
onbellekten okunabilir. Bu artik yalniz web formunda **Hizli onbellek sonucu**
secildiginde olur; varsayilan **RTX 3060 — modelleri yeniden calistir** secenegi
onbellegi atlar. Onbellek isteginde agir modeller calismadigi icin GPU kullanimi
sifir gorunur ve sonuc sayfasi bunu acikca yazar. DICOM okuma,
NIfTI donusumu, ZIP ve sonuc gorsellestirme adimlari CPU/disk isidir; bunlarin
GPU grafigini yukseltmesi beklenmez.

## Tekrar dogrulama

- Projeyi baslatmak: `run_web.bat`
- GPU 1'i gorunur bicimde test etmek: `verify_gpu1.bat`
- Ham Python testi: `scripts/verify_nvidia_gpu.py --seconds 30`
- Gercek model+telemetri testi: `scripts/verify_model_gpu.py <girdi.nii.gz> --bypass-cache`

Servis RTX 3060'i veya CUDA'yi bulamazsa baslamaz ve acik hata verir. Boylece
projenin fark edilmeden CPU/Intel uzerinde model calistirmasi engellenmistir.

### 14 Agustos 2026 son bellek ve darboğaz düzeltmesi

Kullanıcının `bela.txt` günlüğündeki tek taze web koşusu `13:01.27` sürdü;
önbellek açıkça atlanmıştı ve beş modelin beşi ayrı CUDA PID'iyle RTX 3060'ı
kullandı. Asıl regresyon, R-Super sırasında boş RAM'in `17.6 MiB` seviyesine
düşmesi ve Windows sayfalamasının GPU'yu aç bırakmasıydı.

PanTS ağları yine tüm 26 eğitilmiş kanalı hesaplamaktadır; yalnız kullanılmayan
21 kanal artık tam hacim halinde GPU'dan CPU'ya taşınıp RAM'de biriktirilmez.
Tam hacimde yinelenen ölü birleştirme kaldırıldı ve aynı hesap tek fiziksel ROI
içinde yapıldı. Güncel taze koşuda beş aşama `599.000 sn` değerinden `440.547
sn` değerine, R-Super `78.187 sn` değerinden `45.532 sn` değerine indi. Tepe
GPU `%100`, VRAM `5386 MiB`, güç `115.3 W`; `5/5` aşama doğrulandı.
R-Super minimum boş RAM'i `5994.8 MiB` oldu.

Web ana süreci, inference boyunca CPU tarafındaki yeniden örnekleme ve
birleştirme için geçici `AboveNormal` önceliğe alınır ve işlem bitince önceki
önceliğine geri döner. Önceki sonucun arka plan DICOM ZIP üretimi, yeni model
analiziyle RAM/disk için yarışmayacak şekilde model kilidiyle serileştirilir.

Referansla fark 130285568 voxelde yalnız sekiz sınır voxelidir; affine ve
pankreas voxel toplamı aynıdır. Ayrıntılı rapor:
`metrics/FRESH_GPU_BOTTLENECK_REPORT_20260814.md`.
