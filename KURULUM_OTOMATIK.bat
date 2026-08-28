@echo off
setlocal EnableExtensions EnableDelayedExpansion
cd /d "%~dp0"

title PancreasAI Otomatik Kurulum ve Bagimlilik Yukleyici

echo ============================================================
echo   PANCREAS AI - OTOMATIK BILGISAYAR KURULUM SERVISI
echo ============================================================
echo.
echo Bu script projenin baska bir bilgisayara tasindiginda
echo Python surumunu, sanal ortami, tum kutuphaneleri ve dosya
echo yollarini (.env, config.json) otomatik duzeltmek uzere tasarlanmistir.
echo.

:: ------------------------------------------------------------
:: 1. PYTHON KONTROLU VE INDIRILMESI
:: ------------------------------------------------------------
echo [1/6] Python calisma zamani kontrol ediliyor...

set "PYTHON_CMD="

:: 1a. Sistemde python komutu var mi?
where python >nul 2>&1
if not errorlevel 1 (
    python -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)" >nul 2>&1
    if not errorlevel 1 set "PYTHON_CMD=python"
)

:: 1b. Python py launcher var mi?
if not defined PYTHON_CMD (
    where py >nul 2>&1
    if not errorlevel 1 (
        py -3 -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)" >nul 2>&1
        if not errorlevel 1 set "PYTHON_CMD=py -3"
    )
)

:: 1c. Yerel .python_runtime klasoru var mi?
if not defined PYTHON_CMD (
    if exist "%~dp0.python_runtime\python.exe" (
        "%~dp0.python_runtime\python.exe" -c "import sys" >nul 2>&1
        if not errorlevel 1 set "PYTHON_CMD=%~dp0.python_runtime\python.exe"
    )
)

:: 1d. Python bulunamadiysa otomatik indir
if not defined PYTHON_CMD goto :install_python
goto :python_ready

:install_python
echo.
echo [BILGI] Bu bilgisayarda Python 3.10+ bulunamadi.
echo [BILGI] Python 3.12 64-bit yukleyicisi indiriliyor...

set "INSTALLER_PATH=%TEMP%\python-3.12.8-amd64.exe"
powershell -NoProfile -ExecutionPolicy Bypass -Command "[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; (New-Object System.Net.WebClient).DownloadFile('https://www.python.org/ftp/python/3.12.8/python-3.12.8-amd64.exe', '%INSTALLER_PATH%')"

if exist "%INSTALLER_PATH%" (
    echo [BILGI] Python 3.12 sessizce kuruluyor, lutfen bekleyin...
    "%INSTALLER_PATH%" /quiet InstallAllUsers=1 PrependPath=1 Include_test=0
    del /f /q "%INSTALLER_PATH%" >nul 2>&1
    
    set "PATH=%ProgramFiles%\Python312;%ProgramFiles%\Python312\Scripts;%PATH%"
    set "PYTHON_CMD=python"
) else (
    echo.
    echo [HATA] Python 3.12 otomatik indirilemedi.
    echo Lutfen https://www.python.org/downloads/ adresinden Python 3.12 indirip kurun.
    pause
    exit /b 1
)

:python_ready
echo [OK] Kullanilacak Python: %PYTHON_CMD%

:: ------------------------------------------------------------
:: 2. SANAL ORTAMLARIN KONTROLU VE YENIDEN OLUSTURULMASI
:: ------------------------------------------------------------
echo.
echo [2/6] Sanal ortamlar - venv kontrol ediliyor...

set "RECREATE_VENV=0"
if exist "%~dp0venv\Scripts\python.exe" (
    "%~dp0venv\Scripts\python.exe" -c "import sys" >nul 2>&1
    if errorlevel 1 (
        echo [BILGI] Eski bilgisayardan kopyalanan venv yollari gecersiz. Yeniden olusturulacak.
        set "RECREATE_VENV=1"
    )
) else (
    set "RECREATE_VENV=1"
)

if "%RECREATE_VENV%"=="1" (
    if exist "%~dp0venv" rmdir /s /q "%~dp0venv" >nul 2>&1
    echo [BILGI] Ana sanal ortam venv olusturuluyor...
    %PYTHON_CMD% -m venv "%~dp0venv"
    if errorlevel 1 (
        echo [HATA] venv sanal ortami olusturulamadi!
        pause
        exit /b 1
    )
)

set "RECREATE_3D_VENV=0"
if exist "%~dp0.venv_totalseg\Scripts\python.exe" (
    "%~dp0.venv_totalseg\Scripts\python.exe" -c "import sys" >nul 2>&1
    if errorlevel 1 (
        echo [BILGI] Eski bilgisayardan kopyalanan .venv_totalseg yollari gecersiz. Yeniden olusturulacak.
        set "RECREATE_3D_VENV=1"
    )
) else (
    set "RECREATE_3D_VENV=1"
)

if "%RECREATE_3D_VENV%"=="1" (
    if exist "%~dp0.venv_totalseg" rmdir /s /q "%~dp0.venv_totalseg" >nul 2>&1
    echo [BILGI] 3B Segmentasyon sanal ortami .venv_totalseg olusturuluyor...
    %PYTHON_CMD% -m venv "%~dp0.venv_totalseg"
    if errorlevel 1 (
        echo [HATA] .venv_totalseg sanal ortami olusturulamadi!
        pause
        exit /b 1
    )
)

set "MAIN_PYTHON=%~dp0venv\Scripts\python.exe"
set "THREE_D_PYTHON=%~dp0.venv_totalseg\Scripts\python.exe"

:: ------------------------------------------------------------
:: 3. ANA ORTAM KUTUPHANELERININ KURULUMU
:: ------------------------------------------------------------
echo.
echo [3/6] Ana ortam kutuphaneleri - PyTorch, Flask, nnU-Net vb. kuruluyor...
"%MAIN_PYTHON%" -m pip install --upgrade pip setuptools wheel
if errorlevel 1 goto :error

"%MAIN_PYTHON%" -m pip install -r "%~dp0requirements.txt"
if errorlevel 1 goto :error

echo [3/6] PACS ve DICOM SEG kutuphaneleri dogrulaniyor...
"%MAIN_PYTHON%" -c "import pynetdicom, highdicom; print('PACS:', pynetdicom.__version__, '| DICOM SEG:', highdicom.__version__)"
if errorlevel 1 goto :error

:: ------------------------------------------------------------
:: 4. 3B ORTAM KUTUPHANELERININ KURULUMU
:: ------------------------------------------------------------
echo.
echo [4/6] 3B segmentasyon ortam kutuphaneleri kuruluyor...
"%THREE_D_PYTHON%" -m pip install --upgrade pip setuptools wheel
if errorlevel 1 goto :error

"%THREE_D_PYTHON%" -m pip install torch==2.7.1+cu118 torchvision==0.22.1+cu118 --index-url https://download.pytorch.org/whl/cu118
if errorlevel 1 goto :error

"%THREE_D_PYTHON%" -m pip install -r "%~dp0requirements-3d.txt"
if errorlevel 1 goto :error

:: ------------------------------------------------------------
:: 5. DOSYA YOLLARI, ORTAM DEGISKENLERI & REGISTRY AYARLARI
:: ------------------------------------------------------------
echo.
echo [5/6] Proje dosya yollari (.env, config.json), nnU-Net yollari ve GPU ayarlari yapiliyor...

if not exist "%~dp0data\raw" mkdir "%~dp0data\raw"
if not exist "%~dp0data\nnunet_raw" mkdir "%~dp0data\nnunet_raw"
if not exist "%~dp0data\nnunet_preprocessed" mkdir "%~dp0data\nnunet_preprocessed"
if not exist "%~dp0data\nnunet_results" mkdir "%~dp0data\nnunet_results"
if not exist "%~dp0data\inference_output" mkdir "%~dp0data\inference_output"
if not exist "%~dp0logs" mkdir "%~dp0logs"
if not exist "%~dp0metrics" mkdir "%~dp0metrics"

:: .env ve config.json icerisindeki dosya yollarini yeni bilgisayarin klasorune gore otomatik duzelt
"%MAIN_PYTHON%" scripts\fix_project_paths.py
if errorlevel 1 goto :error

setx nnUNet_raw "%~dp0data\nnunet_raw" >nul 2>&1
setx nnUNet_preprocessed "%~dp0data\nnunet_preprocessed" >nul 2>&1
setx nnUNet_results "%~dp0data\nnunet_results" >nul 2>&1

reg add "HKCU\Software\Microsoft\DirectX\UserGpuPreferences" /v "%MAIN_PYTHON%" /t REG_SZ /d "GpuPreference=2;" /f >nul 2>&1
reg add "HKCU\Software\Microsoft\DirectX\UserGpuPreferences" /v "%THREE_D_PYTHON%" /t REG_SZ /d "GpuPreference=2;" /f >nul 2>&1

:: ------------------------------------------------------------
:: 6. KURULUM DOGRULAMA
:: ------------------------------------------------------------
echo.
echo [6/6] Kurulum ve bagimliliklar dogrulaniyor...
echo.
"%MAIN_PYTHON%" -c "import torch, nnunetv2, nibabel, SimpleITK, flask; print(' [OK] Ana paketler tamam. PyTorch:', torch.__version__, '| CUDA GPU Etkin:', torch.cuda.is_available())"
if errorlevel 1 goto :error

"%THREE_D_PYTHON%" -c "import torch; print(' [OK] 3B Segmentasyon paketleri tamam. PyTorch:', torch.__version__, '| CUDA GPU Etkin:', torch.cuda.is_available())"
if errorlevel 1 goto :error

echo.
echo ============================================================
echo [BASARILI] TEBRIKLER! KURULUM TAMAMLANDI!
echo ============================================================
echo Proje dosya yollari yeni bilgisayara gore otomatik guncellendi.
echo Web uygulamasini baslatmak icin: run_web.bat dosyasini calistirin.
echo GPU kontrolu icin: verify_nvidia_gpu.bat dosyasini calistirin.
echo ============================================================
echo.
pause
goto :end

:error
echo.
echo ============================================================
echo [HATA] Kurulum sirasinda bir hata olustu!
echo Lutfen internet baglantinizi ve sistem gereksinimlerinizi kontrol edin.
echo ============================================================
echo.
pause
exit /b 1

:end
endlocal
