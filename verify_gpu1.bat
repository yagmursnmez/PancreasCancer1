@echo off
setlocal EnableExtensions
cd /d "%~dp0"
echo [UYARI] verify_gpu1.bat eski adla uyumluluk icindir.
echo [UYARI] Genel NVIDIA dogrulayicisi baslatiliyor: verify_nvidia_gpu.bat
call "%~dp0verify_nvidia_gpu.bat"
endlocal
