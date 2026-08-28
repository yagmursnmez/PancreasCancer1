# 🚀 Yeni Bilgisayarda Proje Kurulum Rehberi

Proje klasörünü başka bir bilgisayara taşıdığınızda tüm bağımlılıkları ve Python ortamını tek tıkla kurabilirsiniz.

---

## 📋 Adım Adım Kurulum

### 1. Proje Klasörünü Yeni Bilgisayara Kopyalayın
Projeyi yeni bilgisayarınızda istediğiniz bir klasöre atın (Örn: `C:\PancreasCancer`).

> **ÖNEMLİ NOT:** Başka bilgisayardan kopyalanan varsayılan `venv` ve `.venv_totalseg` sanal ortam klasörlerindeki dosya yolları eski bilgisayara ait olacağı için `KURULUM_OTOMATIK.bat` bu eski sanal ortamları tespit edip sizin için otomatik olarak yenileyecektir.

---

### 2. Kurulumu Çalıştırın
1. Proje klasörünün içindeki **`KURULUM_OTOMATIK.bat`** (veya `setup.bat`) dosyasına **çift tıklayın**.
2. Script sırasıyla şunları yapacaktır:
   - **Python Kontrolü**: Yeni bilgisayarda Python 3.10+ kurulu mu bakacak. Eğer Python kurulu değilse, **PowerShell** aracılığıyla resmi **Python 3.12 64-bit** kurucusunu indirip **otomatik (sessiz) olarak kuracaktır**.
   - **Sanal Ortamlar (venv)**: `venv` ve `.venv_totalseg` sanal ortamlarını yeni bilgisayardaki klasör yoluna uygun olarak sıfırdan ve temiz bir şekilde oluşturacaktır.
   - **Kütüphanelerin Kurulumu**: `requirements.txt` ve `requirements-3d.txt` içerisinde yer alan PyTorch (CUDA destekli), nnU-Net v2, SimpleITK, NiBabel, Flask, OpenCV, ReportLab gibi tüm kütüphaneleri otomatik yükleyecektir.
   - **Ortam Değişkenleri**: `nnUNet_raw`, `nnUNet_preprocessed`, `nnUNet_results` değişkenlerini ve Windows GPU ayarlarını sisteme otomatik kaydedecektir.
   - **Doğrulama**: Tüm kütüphaneleri ve GPU kullanılabilirliğini test edecektir.

---

### 3. Uygulamayı Başlatın
Kurulum bittiğinde projeyi çalıştırmak için:
- **`run_web.bat`** dosyasına çift tıklamanız yeterlidir.
- Uygulama otomatik olarak `http://localhost:5000` adresinde başlayacaktır.

---

### 🛡️ Dosya Güvenliği Notu
> **UYARI:** Proje kodu, trained model ağırlıkları (checkpoints), segmentasyon algoritmaları ve veri dosyalarına kesinlikle dokunulmamıştır ve korunmaktadır.
