#!/usr/bin/env bash
# DepthWizard one-click launcher (Linux / macOS). Serves the API + 3D viewer on http://127.0.0.1:8000 and opens the browser.
set -e
cd "$(dirname "$0")"
PY=python3
[ -x .venv/bin/python ] && PY=.venv/bin/python
if [ ! -f frontend/dist/index.html ] || [ ! -f frontend/dist/standalone/standalone.js ]; then
  echo "[DepthWizard] frontend not built - building it now (needs Node.js 20+)..."
  (cd frontend && npm ci && npm run build)
fi
"$PY" scripts/doctor.py >/dev/null 2>&1 || echo "[DepthWizard] environment check reported problems - run: $PY scripts/doctor.py"
exec "$PY" run_server.py "$@"
