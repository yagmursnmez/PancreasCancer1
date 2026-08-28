@echo off
setlocal EnableExtensions
cd /d "%~dp0"

call "%~dp0KURULUM_OTOMATIK.bat"
exit /b %errorlevel%

