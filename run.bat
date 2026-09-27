@echo off
title DepthWizard Server
echo ========================================================
echo   Starting DepthWizard (Single-View Height Estimation)
echo   Local URL: http://127.0.0.1:8000
echo ========================================================
cd /d "%~dp0"
call .venv\Scripts\activate.bat
python run_server.py
pause
