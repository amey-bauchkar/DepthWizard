"""One-command launcher: serves the API and the built 3D viewer, then opens the browser.

Usage:  python run_server.py [--host 127.0.0.1] [--port 8000] [--no-browser]
"""
import argparse
import sys
import threading
import webbrowser
from pathlib import Path

root = Path(__file__).resolve().parent
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

import uvicorn  # noqa: E402

from backend.main import app  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="DepthWizard server")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--no-browser", action="store_true")
    a = ap.parse_args()
    if not (root / "frontend" / "dist" / "index.html").exists():
        print("NOTE: frontend/dist not found - build it with: cd frontend && npm ci && npm run build")
    url = f"http://{a.host}:{a.port}/"
    print(f"DepthWizard running at {url}  (Ctrl+C to stop)")
    if not a.no_browser:
        threading.Timer(2.0, lambda: webbrowser.open(url)).start()
    uvicorn.run(app, host=a.host, port=a.port, log_level="warning")
