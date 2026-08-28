@echo off
setlocal EnableExtensions
cd /d "%~dp0"

if not exist "pacs_config.json" (
    echo [HATA] pacs_config.json bulunamadi.
    echo once pacs_config.example.json dosyasini kopyalayip kurumun AE/IP/port bilgileriyle doldurun.
    exit /b 1
)

set "PYTHON_EXE=%~dp0venv\Scripts\python.exe"
if not exist "%PYTHON_EXE%" (
    echo [HATA] Proje Python ortami bulunamadi: %PYTHON_EXE%
    exit /b 1
)

echo ============================================================
echo  PancreasAI PACS worker baslatiliyor
echo  C-ECHO / CT C-STORE dinleyicisi ayri surecte calisir.
echo  Durdurmak icin: Ctrl+C
echo ============================================================
"%PYTHON_EXE%" scripts\pacs_bridge.py --config pacs_config.json --serve

endlocal
