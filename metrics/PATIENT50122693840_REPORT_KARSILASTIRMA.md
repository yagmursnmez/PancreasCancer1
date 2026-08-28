# PATIENT50122693840 — nihai rapor/model karşılaştırması

## Sonuç

Üçlü kalibre model, lezyonu raporla aynı yerde **pankreas baş–gövde** bölgesinde gösterdi. Nihai ana sınır **56,7 × 55,5 × 71,0 mm** ve **51,762 mL** ölçüldü. Raporun 66 × 55 mm iki çapı, 3B maskenin en büyük iki RAS ekseniyle karşılaştırıldığında fark yaklaşık **+5,0 mm (%7,6)** ve **+1,7 mm (%3,1)** oldu. Ölçüm düzlemleri aynı olmadığından bu oranlar teknik tutarlılık denetimidir.

Önceki maske 46,1 × 32,6 × 38,8 mm ve 15,708 mL idi; rapor boyutunu belirgin eksik temsil ediyordu. Yeni sınır rapora çok daha yakındır ve görüntü üzerinde pankreas başındaki kistik/heterojen yer kaplayıcı alanı kapsamaktadır.

## Belirsizlik katmanları

- Yüksek güvenli çekirdek: **42,402 mL**
- Kalibre ana sınır: **51,762 mL**
- Duyarlı dış sınır: **62,755 mL**
- PanTS modellerinin hasta üzerindeki eşiklenmiş Dice’ı: **0,1988**

Model ayrışması nedeniyle tek yüzeyi kesin patoloji sınırı gibi sunmak yerine çekirdek, ana sınır ve duyarlı sınır 3B görünümde ayrı katmanlar olarak gösterildi.

## Bağımsız doğrulama

MSD Task07 fold-0’daki 46 uzman maskeli olguda:

- Eski nnU-Net tümör Dice ortalaması: **0,2984**
- nnU-Net + MedFormer PanTS (0,40): **0,5724**
- nnU-Net + R-Super PanTS (0,30): **0,5635**
- Üçlü model, MedFormer 0,45 + R-Super 0,45: **0,5832**; medyan Dice **0,6304**
- Pankreas Dice: eski nnU-Net **0,8017**, PanTS birleşimi **0,8312**

Eşikler hasta raporuna göre şişirilmedi; bu 46 olguluk uzman maskeli doğrulamada seçildi.

## Yazılım düzeltmeleri

- PanTS’in 36 bin üzeri CT ile geliştirilen MedFormer ve R-Super ağırlıkları eklendi.
- 6 GB GPU için doğruluğu koruyan 96³ örtüşmeli kayan pencere ve 80 mm üç eksenli ROI kullanıldı.
- Üçlü olasılık füzyonu, tohum destekli bağlı bileşen seçimi ve ayrı belirsizlik maskesi üretime alındı.
- Pankreas yüzeyi de PanTS pankreas sınıfıyla genişletildi.
- 2B görünüm hasta yönünde bırakıldı; R/L işaretleri ve çekirdek/dış sınır konturları eklendi.
- 3B yüzeyler RAS milimetre koordinatında ve gerçek fiziksel oranla çizildi.

## Klinik sınır

Bu olgu için radyolog çizimli piksel düzeyinde referans maske bulunmadığından hastaya özgü gerçek Dice veya yüzey mesafesi ölçülemez. Rapor ölçüsü ile otomatik 3B kutu yakın olsa da nihai tedavi/cerrahi sınırı olarak kullanılmadan önce abdominal radyolog tarafından kaynak kesitlerde onaylanmalıdır.
