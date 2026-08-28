@echo off
setlocal EnableExtensions EnableDelayedExpansion
cd /d "%~dp0"

title PancreasAI Web Servisi
set "OLD_PID="
for /f "tokens=5" %%P in ('netstat -ano ^| findstr /R /C:":5000 .*LISTENING"') do (
    if not defined SEEN_%%P (
        set "SEEN_%%P=1"
        set "OLD_PID=%%P"
        echo [BILGI] 5000 portundaki eski servis kapatiliyor. PID=%%P
        taskkill /PID %%P /T /F >nul 2>&1
        if errorlevel 1 (
            echo [HATA] Eski servis kapatilamadi. PID=%%P
            echo Bu dosyayi bir kez Yonetici olarak calistirin.
            pause
            exit /b 1
        )
    )
)

if defined OLD_PID timeout /t 1 /nobreak >nul
for /f "tokens=5" %%P in ('netstat -ano ^| findstr /R /C:":5000 .*LISTENING"') do (
    echo [HATA] 5000 portu serbest birakilamadi. PID=%%P
    pause
    exit /b 1
)

set "PYTHON_EXE="
set "PROJECT_SITE_PACKAGES=%~dp0venv\Lib\site-packages"
set "PROJECT_PYTHON=%~dp0.python_runtime\python.exe"
if exist "%~dp0venv\Scripts\python.exe" (
    "%~dp0venv\Scripts\python.exe" -c "import sys" >nul 2>&1
    if not errorlevel 1 set "PYTHON_EXE=%~dp0venv\Scripts\python.exe"
)

if not defined PYTHON_EXE (
    if exist "!PROJECT_PYTHON!" (
        set "PYTHON_EXE=!PROJECT_PYTHON!"
        set "PYTHONPATH=!PROJECT_SITE_PACKAGES!"
    )
)

if not defined PYTHON_EXE (
    echo [HATA] Calisabilir Python ortami bulunamadi.
    echo venv taban Python'i eksik ve paketli Python bulunamiyor.
    pause
    exit /b 1
)

:: CUDA aygit secimi Windows Gorev Yoneticisi sira numarasindan bagimsizdir.
set "CUDA_DEVICE_ORDER=PCI_BUS_ID"
if not defined CUDA_VISIBLE_DEVICES set "CUDA_VISIBLE_DEVICES=0"
set "CUDA_REQUIRED=True"
set "CUDA_MODULE_LOADING=LAZY"
set "PYTORCH_CUDA_ALLOC_CONF=backend:cudaMallocAsync"
set "SHIM_MCCOMPAT_ENABLE_GPU=1"
set "GPU_EVIDENCE_REQUIRED=True"
set "GPU_TELEMETRY_INTERVAL_SECONDS=1.0"
set "FLASK_DEBUG=False"
set "PANCREAS_DEBUG=True"

:: Windows Grafik Ayarlari: proje calistiricilarini yuksek performansli
:: yuksek performansli ayrik NVIDIA GPU'ya yonlendir.
reg add "HKCU\Software\Microsoft\DirectX\UserGpuPreferences" /v "%PYTHON_EXE%" /t REG_SZ /d "GpuPreference=2;" /f >nul 2>&1
reg add "HKCU\Software\Microsoft\DirectX\UserGpuPreferences" /v "%~dp0.python_runtime\python.exe" /t REG_SZ /d "GpuPreference=2;" /f >nul 2>&1
reg add "HKCU\Software\Microsoft\DirectX\UserGpuPreferences" /v "%~dp0venv\Scripts\python.exe" /t REG_SZ /d "GpuPreference=2;" /f >nul 2>&1
reg add "HKCU\Software\Microsoft\DirectX\UserGpuPreferences" /v "%~dp0.venv_totalseg\Scripts\python.exe" /t REG_SZ /d "GpuPreference=2;" /f >nul 2>&1
reg add "HKCU\Software\Microsoft\DirectX\UserGpuPreferences" /v "%~dp0.venv_totalseg\Scripts\TotalSegmentator.exe" /t REG_SZ /d "GpuPreference=2;" /f >nul 2>&1

"%PYTHON_EXE%" -c "import torch; assert torch.cuda.is_available() and torch.cuda.device_count()>0 and torch.version.cuda; p=torch.cuda.get_device_properties(0); x=torch.ones(1, device='cuda:0'); torch.cuda.synchronize(); print('[NVIDIA]', p.name, '| CUDA cuda:0 | UUID', getattr(p,'uuid',''), '| PyTorch', torch.__version__, '| CUDA', torch.version.cuda)"
if errorlevel 1 (
    echo [HATA] NVIDIA CUDA dogrulanamadi; model CPU uzerinde baslatilmadi.
    pause
    exit /b 1
)

echo ============================================================
echo  PancreasAI guncel web servisi baslatiliyor
echo  URL: http://localhost:5000
echo  Yukleme siniri: .env dosyasindaki MAX_UPLOAD_MB
echo  Durdurmak icin: Ctrl+C
echo  Python: %PYTHON_EXE%
echo  NVIDIA: CUDA_VISIBLE_DEVICES=%CUDA_VISIBLE_DEVICES% ^> mantiksal cuda:0
echo  GPU kullanimi: yalniz analiz/model calisirken etkin
echo ============================================================
"%PYTHON_EXE%" web\app.py

if errorlevel 1 (
    echo.
    echo [HATA] Web servisi baslatilamadi.
    pause
)

endlocal
