# Geri Dönüş Noktası — Segmentasyon Düzeltmesi Öncesi

Oluşturulma tarihi: 6 Ağustos 2026  
Amaç: Pankreas/tümör maskeleme düzeltmelerine başlamadan önceki çalışan kaynak durumunu korumak.

## Korunan içerik

- `source_snapshot.zip`: Değiştirilebilecek Python kaynakları, web uygulaması, şablonlar, yapılandırma ve çalıştırma dosyaları.
- `baseline_results/`: İlgili vakalar için düzeltme öncesindeki PNG çıktı örnekleri.
- Veri kümeleri, model ağırlıkları ve `.env` değiştirilmedi; bu yüzden arşive alınmadı.

## Bütünlük doğrulaması

- Arşiv: `source_snapshot.zip`
- Boyut: `199821` bayt
- SHA-256: `BBC1F2B2957658012B41E840903C95666FFF0F993B90C68533283B32FF20CAC6`

Geri dönüşten önce arşivin SHA-256 değeri yukarıdaki değerle karşılaştırılmalıdır. Değer eşleşmiyorsa arşiv kullanılmamalıdır.

## Geri dönmek için

Codex'e şu talimatı verin:

> `checkpoints/rollback_20260806_before_segmentation_fix/GERI_DONUS_NOKTASI.md dosyasını oku, SHA-256'yı doğrula ve bu geri dönüş noktasına dön.`

Geri dönüş işlemi mevcut kaynakların üstüne yazacağı için yalnızca açık talep üzerine yapılmalıdır. Codex önce o anki durum için ayrıca bir güvenlik kopyası oluşturmalı, sonra bu ZIP'i proje köküne açmalıdır.

## Kapsam

Bu nokta aşağıdakileri korur: `config.json`, kök çalıştırma/kurulum dosyaları, `scripts/`, `web/app.py`, `web/templates/`, `web/static/css/` ve `web/static/js/`.

Bu geri dönüş noktası sonraki test çıktılarından ve model/veri dosyalarından bağımsız tutulmalıdır; düzeltme sırasında bu klasördeki ZIP veya başlangıç PNG'leri değiştirilmemelidir.
