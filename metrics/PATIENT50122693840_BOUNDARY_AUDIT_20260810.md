# PATIENT50122693840 — sınır güveni denetimi

## Sonuç

Pozitif pankreas lezyonu adayı ve baş–gövde lokalizasyonu desteklenmektedir; otomatik dış sınır doğrulanamamıştır. Bu olgu için uygulama sonucu **“tümör adayı saptandı; sınır güveni çok düşük — uzman konturu gerekli”** olarak gösterilmelidir.

## Kanıtlar

- nnU-Net / DiNTS tümör Dice: **0,2931**.
- PanTS modellerinin seed-anchored Dice değeri: **0,2339**.
- Pankreas modelleri Dice: **0,3452**.
- Doğrulanmış çekirdek: **28.495 voksel, 15,708 mL, 46,1 × 32,6 × 38,8 mm**.
- Belirsiz olası zarf: **57.966 voksel, 31,954 mL, yaklaşık 49,9 × 41,1 × 46,2 mm**.
- Çekirdek yoğunluğu: medyan **44 HU** (IQR 33–59 HU).
- Pankreas maskesi yoğunluğu: medyan **69 HU** (IQR 48–86 HU).
- Kaba RAS lokalizasyonu: **baş–gövde**.
- Mevcut proje içi rapor referansı: baş–gövdede kistik alanlar içeren yaklaşık **66 × 55 mm** yer kaplayıcı lezyon.

Sağlanan `C:\Users\monster\Downloads\BUGGRA 1\BUGGRA\PATIENT50122693840` klasöründe ayrı bir yazılı rapor dosyası bulunmadı. 66 × 55 mm referansı daha önce kaydedilmiş proje denetim dosyasından alınmıştır; kaynak rapor yeniden eklenirse metin tekrar doğrulanmalıdır.

## Eski geniş maske neden kabul edilmedi?

Eski PanTS OR birleşimi **51,762 mL ve 56,7 × 55,5 × 71,0 mm** üreterek raporun iki çapına yaklaşmıştı. Ancak PanTS modelleri arası Dice yalnızca **0,1988** idi. Boyut yakınlığı piksel düzeyinde anatomik doğruluğu kanıtlamaz; rapora göre eşik ayarlayıp maskeyi büyütmek veri sızıntısı ve hasta-özel aşırı uyum olur.

## Doğruluk sınırı

Bu hastaya ait uzman çizimli referans kontur yoktur. Bu nedenle gerçek Dice, surface Dice, HD95, hacim hatası veya damar temas doğruluğu ölçülemez. Görüntü ve rapor yalnızca:

- pozitif adayın varlığı,
- kaba baş–gövde lokalizasyonu,
- mevcut çekirdeğin 66 × 55 mm rapor referansına göre muhtemel eksik kapsama riski

hakkında denetim sağlar.

## Güvenli ilerleme planı

1. Abdominal radyolog tüm olguda üç ayrı referans çizer: solid/viabl tümör, kistik-nekrotik bileşen ve infiltratif uzanım.
2. İkinci radyolog bağımsız kontur üretir; anlaşmazlık konsensüs oturumuyla çözülür.
3. Bu dış-merkez olgu eğitim verisine katılmadan önce ayrı test olgusu olarak kilitlenir.
4. Benzer kistik/heterojen, solid ve infiltratif örneklerden çok-merkezli uzman etiketli kohort oluşturulur.
5. 3B nnU-Net full-resolution/cascade ile bir transformer tabanlı 3B model bağımsız eğitilir; birleşim yalnız ayrılmış doğrulama setinde seçilir.
6. Hasta düzeyinde ayrılmış iç doğrulama ve hiç görülmemiş merkezden dış test yapılır.
7. Dice yanında surface Dice, HD95, lezyon duyarlılığı, hacim yanlılığı ve “karar verilemedi” oranı raporlanır.
8. Klinik toleransı karşılamayan veya modelleri ayrışan olgularda sistem otomatik olarak uzman konturu ister.

“Sıfır hata” veri üzerinde doğrulanabilir bir hedef değildir. Güvenli hedef; hatayı ölçmek, dış testte sınırlandırmak ve sistem emin olmadığında kesin sonuç üretmesini engellemektir.
