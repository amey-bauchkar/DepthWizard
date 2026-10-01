@echo off
title DepthWizard - Automated Azure Cloud Deployer
cd /d "%~dp0"
echo Starting Automated Cloud Deployer...
.venv\Scripts\python.exe scripts\deploy_to_vm.py
pause
