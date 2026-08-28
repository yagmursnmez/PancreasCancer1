# PACS iş istasyonu bağlantısı — başlangıç kurulumu

Bu katman mevcut modeli, `web/app.py` dosyasını ve mevcut dosya-temelli DICOM
akışını değiştirmez. Ayrı çalışan `scripts/pacs_bridge.py`, PACS'in tanıdığı
bir DICOM cihazı olur:

```text
PACS -- C-ECHO / C-STORE (CT) --> PANC_AI (RAM cache) --> analiz adaptörü --> DICOM SEG/SR --> PACS
```

Alınan C-STORE nesneleri kalıcı klasöre veya veritabanına yazılmaz. Her nesne
yalnızca süreç RAM'inde tutulur; `max_cache_mb` dolunca yeni C-STORE isteği
kaynak yetersizliği ile reddedilir. Daha önce kabul edilmiş bir seri otomatik
silinmez. Bu, kabul edilmiş bir
görüntünün diskte kaybolmadan sonra yeniden işlenebileceği anlamına gelmez.
PACS, asıl arşiv olmaya devam etmelidir.

## 1. PACS yöneticisine verilecek kayıt bilgisi

`pacs_config.example.json` dosyasını kopyalayarak kurum değerleriyle doldurun
ve bu canlı yapılandırmayı kaynak koda eklemeyin. Örnek AE değerleri yer
tutucudur.

| PACS AE tablosu alanı | Girilecek değer |
| --- | --- |
| Device / Description | PancreasAI research workstation |
| AE Title | `local.ae_title` — ör. `PANC_AI` |
| IP address | İş istasyonunun **sabit gerçek IP adresi** |
| Port | `local.port` — ör. `11112` |
| Rol / servis | Verification SCP ve CT Image Storage SCP |
| Gönderen AE sınırı | Yalnız `security.allowed_calling_aes` içindeki AE'ler |

Güvenlik duvarında sadece PACS IP'sinden bu TCP portuna giriş açılmalıdır.
NAT, dinamik IP veya `0.0.0.0` PACS AE tablosuna yazılmaz; `0.0.0.0` sadece
sunucunun tüm yerel ağ arayüzlerinde dinlemesi için kullanılabilir.

## 2. Bağımlılık ve yapılandırma doğrulaması

```powershell
.\venv\Scripts\python.exe -m pip install -r requirements.txt
Copy-Item pacs_config.example.json pacs_config.json
.\venv\Scripts\python.exe scripts\pacs_bridge.py --config pacs_config.json --validate
```

Doğrulama komutu ağ portu açmaz; PACS ekibine verilecek AE bilgilerini yazdırır.
`enabled` varsayılan olarak `false` kalmalıdır. AE tablosu, IP ve firewall
onaylandığında bilinçli olarak `true` yapın.

## 3. Bağlantı testi ve dinleme

PACS yöneticisi önce `PANC_AI` hedefi için C-ECHO çalıştırır. Başarılıysa
yalnız küçük, test amaçlı bir CT serisini C-STORE ile yollar. Ardından:

```powershell
.\venv\Scripts\python.exe scripts\pacs_bridge.py --config pacs_config.json --serve
```

Dinleyici sadece CT Image Storage kabul eder. Yetkisiz AE Title, CT dışı SOP
sınıfı, seri başına örnek sınırı veya RAM sınırı aşılırsa C-STORE başarıyla
almış gibi davranmaz; PACS'e hata durumu döndürür. Bu özellikle hasta verisinin
sessizce kaybolmaması içindir.

Web ana sayfasındaki **PACS Bağlantısı** sekmesi yapılandırmanın güvenli biçimde
hazır olup olmadığını gösterir; dinleyiciyi web sürecinden başlatmaz. Windows
sunucuda worker'ı ayrı konsolda `run_pacs.bat` ile, web arayüzünü ise her zaman
`run_web.bat` ile başlatın. Böylece yoğun C-STORE alımı GPU/model zincirini
ve tarayıcı isteklerini bloke etmez.

## Sonuçları PACS'e gönderme sınırı

Köprüde bulunan `PacsResultSender`, kaynak CT kesitlerine bağlı standart
**DICOM SEG** nesnelerini üretip C-STORE ile hedef PACS'e gönderebilir. Pankreas
ve tümör adayı, ayrı ikili SEG nesneleri olarak gönderilir. Mevcut
`dicom_export.py` araştırma görüntülü ZIP üretir ve PACS'e sonuç olarak
gönderilmez.

PACS yöneticisinin Segmentation Storage SOP Class
`1.2.840.10008.5.1.4.1.1.66.4` kabulünü doğrulaması gerekir. DICOM SEG, kaynak
CT kesitleriyle SOP Instance ve hasta-geometrisi üzerinden ilişkilidir; normal
CT görüntüsü gibi sahte bir kopya değildir.
