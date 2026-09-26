@echo off
rem DepthWizard one-click launcher (Windows). Serves the API + 3D viewer on http://127.0.0.1:8000 and opens the browser.
cd /d "%~dp0"
set "PY=python"
if exist ".venv\Scripts\python.exe" set "PY=.venv\Scripts\python.exe"
if not exist "frontend\dist\index.html" (
  echo [DepthWizard] frontend not built - building it now ^(needs Node.js 20+^)...
  pushd frontend
  call npm ci
  call npm run build
  popd
)
"%PY%" scripts\doctor.py >nul 2>&1 || echo [DepthWizard] environment check reported problems - run: "%PY%" scripts\doctor.py
"%PY%" run_server.py %*
