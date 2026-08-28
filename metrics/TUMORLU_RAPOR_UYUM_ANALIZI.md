# TÜMÖRLÜ olgu — rapor ve segmentasyon uyum analizi

## Rapor referansı

Radyoloji raporu pankreas başında yaklaşık 27 mm çaplı hipodens kitle tarif ediyor. Ayrıca portosplenik konfluens, SMA proksimali ve portal invazyon ile SMA'yı 360° çevreleyen infiltratif/perinöral yumuşak doku uzanımı bildiriliyor.

## İlk yayımlanan maskenin denetimi

- İlk ana maske: 7.076 voksel, 17,491 mL, 39,4 × 34,5 × 35,0 mm.
- Doğrulanmış nnU-Net + DiNTS tohumu: 2.391 voksel, 5,910 mL, 25,3 × 22,5 × 35,0 mm.
- PanTS aşaması tohumu 4.685 voksel / 11,581 mL büyütmüş; ilk hacmin yalnızca %33,8'i iki-model tohumundan geliyordu.
- PanTS modellerinin kendi Dice uyumu 0,4417 idi. Bu nedenle OR birleşimiyle oluşan 17,491 mL sınır güvenilir ana maske kabul edilmedi.

## Düzeltilmiş sonuç

- Karar: iki-model destekli tümör segmentasyon adayı.
- Konum: kaba RAS tabanlı tahminle pankreas başı; raporla uyumlu.
- Ana maske: 2.391 voksel, 5,910 mL.
- Boyut: R/L 25,3 mm, A/P 22,5 mm, S/I 35,0 mm.
- Ana in-plane boyut rapordaki 27 mm'den %6,3 küçüktür. S/I ölçüm %29,6 büyüktür; 5 mm kesit kalınlığı nedeniyle bu eksen yedi kesite kuantize edilmiştir.
- Hacim eşdeğeri küresel çap 22,4 mm'dir. Bu değer yalnızca hacimsel özet olup gerçek düzensiz lezyon çapının yerine geçmez.
- PanTS belirsizlik zarfı 13,240 mL'dir; bu alan ana tümör hacmine dahil edilmemiştir.
- Reddedilen uzak/tek-kesitli ham aday: 106 voksel.

## Doğruluk yorumu

- Olgu düzeyinde pozitiflik ve pankreas başı lokalizasyonu raporla uyumludur.
- İlk 17,491 mL maske rapordaki yaklaşık 27 mm primer kitleye göre aşırı genişleme riski taşıyordu; düzeltilmiş maske bu riski azaltır.
- Rapor, üç boyutlu piksel düzeyinde referans maske değildir. Bu nedenle bu hastaya ait gerçek Dice, IoU, Hausdorff veya sınır doğruluğu hesaplanamaz.
- Raporun tarif ettiği damar invazyonu ve infiltratif/perinöral uzanımın otomatik maskede eksik veya fazla çizilip çizilmediği, kontrastlı BT üzerinde abdominal radyolog tarafından doğrulanmalıdır.
- Sistem çıktısı tanısal değildir ve uzman onayı olmadan klinik kararda kullanılmamalıdır.

## Uygulanan güvenlik düzeltmesi

PanTS artık ana maskeyi yalnızca iki PanTS modeli aynı yeni vokselde uzlaşıyorsa, seed-anchored Dice en az 0,50 ise ve ek alan doğrulanmış maskeye en fazla 5 mm uzaklıktaysa genişletebilir. Tek modelin düşük eşikli uzanımı yalnızca belirsizlik katmanında tutulur. PanTS yapılandırması önbellek anahtarına eklenmiştir; eski hatalı maske yeni çalışmada geri gelmez.
