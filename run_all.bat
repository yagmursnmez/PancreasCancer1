@echo off
:: ============================================================
:: PancreasAI — Tum Adimlari Calistir
:: ============================================================
:: Bu script projenin tum adimlarini sirayla calistirir.
:: Veri hazir olmadan baslat: sadece ADIM 9-10 web icin.
:: ============================================================
title PancreasAI Pipeline
cd /d "%~dp0"

SET BASE=%~dp0
SET "PYTHON=%~dp0venv\Scripts\python.exe"
SET "CUDA_DEVICE_ORDER=PCI_BUS_ID"
IF NOT DEFINED CUDA_VISIBLE_DEVICES SET "CUDA_VISIBLE_DEVICES=0"
SET "CUDA_REQUIRED=True"
SET "CUDA_MODULE_LOADING=LAZY"
SET "PYTORCH_CUDA_ALLOC_CONF=backend:cudaMallocAsync"
SET "SHIM_MCCOMPAT_ENABLE_GPU=1"
SET "GPU_EVIDENCE_REQUIRED=True"
SET "GPU_TELEMETRY_INTERVAL_SECONDS=1.0"
SET "FLASK_DEBUG=False"
SET "PANCREAS_DEBUG=True"

IF NOT EXIST "%PYTHON%" (
    echo [HATA] Proje Python ortami bulunamadi: %PYTHON%
    echo Once setup.bat dosyasini calistirin.
    exit /b 1
)

"%PYTHON%" -c "import torch; assert torch.cuda.is_available() and torch.cuda.device_count()>0 and torch.version.cuda; p=torch.cuda.get_device_properties(0); print('[NVIDIA]',p.name,'CUDA',torch.version.cuda,'UUID',getattr(p,'uuid',''))"
IF %ERRORLEVEL% NEQ 0 (
    echo [HATA] NVIDIA CUDA dogrulanamadi. CPU modunda devam edilmeyecek.
    exit /b 1
)

echo.
echo ============================================================
echo  PANKREAS KANSERI TESPITI — PIPELINE
echo ============================================================
echo.

:: --- ADIM 2: Dataset Hazirla ---
echo [ADIM 2] Dataset hazirlaniyor...
"%PYTHON%" scripts\prepare_dataset.py
IF %ERRORLEVEL% NEQ 0 (
    echo [UYARI] ADIM 2 tamamlanamadi. Devam ediliyor...
)

:: --- ADIM 3: Preprocessing ---
echo.
echo [ADIM 3] Preprocessing...
"%PYTHON%" scripts\run_preprocessing.py
IF %ERRORLEVEL% NEQ 0 (
    echo [UYARI] ADIM 3 tamamlanamadi. Devam ediliyor...
)

:: --- ADIM 4: Egitim ---
echo.
echo [ADIM 4] Model egitimi (bu uzun surebilir)...
echo NOT: Egitimi atlamak icin bu satiri yorum satirina alin.
:: "%PYTHON%" scripts\train_model.py --epochs 1000

:: --- ADIM 5: Validasyon ---
echo.
echo [ADIM 5] Validasyon metrikleri...
"%PYTHON%" scripts\validate_metrics.py
IF %ERRORLEVEL% NEQ 0 (
    echo [UYARI] ADIM 5 tamamlanamadi.
)

:: --- ADIM 6+7: Inference ---
echo.
echo [ADIM 6+7] Inference ve siniflandirma...
"%PYTHON%" scripts\inference.py
IF %ERRORLEVEL% NEQ 0 (
    echo [UYARI] ADIM 6+7 tamamlanamadi.
)

:: --- ADIM 8: 3D Rekonstrüksiyon ---
echo.
echo [ADIM 8] 3D rekonstruksiyon...
"%PYTHON%" scripts\reconstruct_3d.py
IF %ERRORLEVEL% NEQ 0 (
    echo [UYARI] ADIM 8 tamamlanamadi.
)

:: --- ADIM 10: Model Bilgisi ---
echo.
echo [ADIM 10] Model bilgileri gosteriliyor...
"%PYTHON%" scripts\save_model.py --info

:: --- ADIM 9: Web Uygulamasi ---
echo.
echo ============================================================
echo  ADIM 9: WEB UYGULAMASI BASLATILIYOR
echo  Tarayici: http://localhost:5000
echo  Durdurmak icin: Ctrl+C
echo ============================================================
echo.
call "%BASE%run_web.bat"

pause
