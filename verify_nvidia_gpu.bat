@echo off
setlocal EnableExtensions EnableDelayedExpansion
cd /d "%~dp0"
title PancreasAI - NVIDIA CUDA Dogrulama

set "PYTHON_EXE="
set "PROJECT_SITE_PACKAGES=%~dp0venv\Lib\site-packages"
set "PROJECT_PYTHON=%~dp0.python_runtime\python.exe"
if exist "%~dp0venv\Scripts\python.exe" (
    "%~dp0venv\Scripts\python.exe" -c "import torch" >nul 2>&1
    if not errorlevel 1 set "PYTHON_EXE=%~dp0venv\Scripts\python.exe"
)
if not defined PYTHON_EXE (
    if exist "!PROJECT_PYTHON!" (
        set "PYTHON_EXE=!PROJECT_PYTHON!"
        set "PYTHONPATH=!PROJECT_SITE_PACKAGES!"
    )
)
if not defined PYTHON_EXE (
    echo [HATA] Proje Python ortami bulunamadi.
    pause
    exit /b 1
)

set "CUDA_DEVICE_ORDER=PCI_BUS_ID"
if not defined CUDA_VISIBLE_DEVICES set "CUDA_VISIBLE_DEVICES=0"
set "CUDA_MODULE_LOADING=LAZY"
set "PYTORCH_CUDA_ALLOC_CONF=backend:cudaMallocAsync"
set "SHIM_MCCOMPAT_ENABLE_GPU=1"
reg add "HKCU\Software\Microsoft\DirectX\UserGpuPreferences" /v "%PYTHON_EXE%" /t REG_SZ /d "GpuPreference=2;" /f >nul 2>&1

echo ============================================================
echo  PANKREAS AI - NVIDIA CUDA DOGRULAMA
echo ============================================================
echo  Fiziksel secici : CUDA_VISIBLE_DEVICES=%CUDA_VISIBLE_DEVICES%
echo  Mantiksal aygit : cuda:0
echo.
echo  Gorev Yoneticisi ^> Performans ekraninda NVIDIA kartini acin.
echo  Grafik basligina tiklayip CUDA veya Compute_0 secin.
echo  Simdi yaklasik 30 saniye boyunca yuk olusturulacak.
echo ============================================================

start "" taskmgr.exe
timeout /t 3 /nobreak >nul

"%PYTHON_EXE%" scripts\verify_nvidia_gpu.py --seconds 30
if errorlevel 1 (
    echo.
    echo [HATA] NVIDIA CUDA testi basarisiz.
    pause
    exit /b 1
)

echo.
echo [OK] Bu proje secili NVIDIA CUDA aygiti uzerinde calisti.
pause
endlocal
