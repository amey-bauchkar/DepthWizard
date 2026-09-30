"""DepthWizard backend entrypoint (final prototype: Mode A + Mode B).

Run:  .venv/Scripts/python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
Serves the API under /api and the built frontend (frontend/dist) at / when present.
"""
from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from backend.api.routes import error_response, router
from backend.config.settings import REPO_ROOT, load_settings
from backend.errors import DepthWizardError
from backend.jobs.manager import JobManager
from backend.logging_setup import setup_logging
from core.geo.vertical import register_bundled_grids

__version__ = "1.0.0-prototype"


class SafeStaticFiles(StaticFiles):
    """StaticFiles that answers 404, not 500, for paths the OS cannot even stat (on Windows, e.g. characters such as
    { } ' < > : in a malformed URL raise OSError / WinError 123 inside os.stat)."""

    async def get_response(self, path, scope):
        from starlette.exceptions import HTTPException

        try:
            return await super().get_response(path, scope)
        except OSError:
            raise HTTPException(status_code=404) from None


def create_app(settings=None) -> FastAPI:
    setup_logging()
    register_bundled_grids()  # offline PROJ grids (C-1): bundled assets/proj or DW_PROJ_GRIDS; network stays disabled
    settings = settings or load_settings()

    @asynccontextmanager
    async def _lifespan(a: FastAPI):
        # server start (uvicorn): models load while the user picks a file. Test clients without a `with` block skip it.
        if a.state.settings.server.warm_up_models and os.environ.get("DW_NO_WARMUP") != "1":
            a.state.jobs.warm_up_in_background()
        yield

    app = FastAPI(title="DepthWizard", version=__version__, lifespan=_lifespan)
    app.state.settings = settings
    app.state.jobs = JobManager(settings)

    from fastapi.middleware.cors import CORSMiddleware

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(router)

    @app.middleware("http")
    async def _no_stale_html(request: Request, call_next):
        # index.html names the hashed JS/CSS bundles: a cached copy after an update loads the OLD app. HTML is always
        # revalidated; the hashed assets themselves may be cached forever.
        resp = await call_next(request)
        path = request.url.path
        if path == "/" or path.endswith(".html"):
            resp.headers["Cache-Control"] = "no-cache"
        elif path.startswith("/assets/"):
            resp.headers["Cache-Control"] = "public, max-age=31536000, immutable"
        return resp

    @app.exception_handler(DepthWizardError)
    async def _dw_error(_: Request, exc: DepthWizardError):
        logging.getLogger("depthwizard.api").warning("%s: %s", exc.code, exc.detail, extra={"code": exc.code})
        return error_response(exc)

    @app.exception_handler(Exception)
    async def _unhandled(_: Request, exc: Exception):
        logging.getLogger("depthwizard.api").exception("unhandled error")
        return JSONResponse(status_code=500, content={"error": {"code": "INTERNAL_ERROR", "message": "Unable to process request.", "detail": f"{type(exc).__name__}", "recoverable": False}})

    demo = REPO_ROOT / "assets" / "demo"
    if demo.exists():
        app.mount("/demo", SafeStaticFiles(directory=str(demo)), name="demo")

    dist = REPO_ROOT / settings.server.frontend_dist
    if dist.exists():
        app.mount("/", SafeStaticFiles(directory=str(dist), html=True), name="frontend")
    else:
        @app.get("/")
        def _root():
            return {"message": "DepthWizard API is running. Frontend not built: run `npm run build` in frontend/ or use the Vite dev server.", "docs": "/docs"}

    return app


app = create_app()

if __name__ == "__main__":  # pragma: no cover
    import uvicorn

    s = load_settings()
    uvicorn.run("backend.main:app", host=s.server.host, port=s.server.port, reload=False)
