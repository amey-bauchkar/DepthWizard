# Sprint 1 build notes (2026-09-21)

## Environment actually used
| Item | Value |
| --- | --- |
| OS | Windows 11 10.0.26200 |
| Python | 3.12.14 (uv-managed CPython in `.venv`; system Python is 3.14.0 — **not** used) |
| Node / npm | v25.6.1 / 11.9.0 (**deviation**: Phase 7 targets 24 LTS; Vite 8 / TS 5.9 / three 0.186 build fine) |
| torch / torchvision | 2.14.0+cpu / 0.29.0+cpu — `torch.cuda.is_available()` = False (no NVIDIA GPU on this machine) |
| rasterio / GDAL | 1.5.1 / 3.12.4 |
| pyproj / PROJ | 3.8.0 / 9.8.1 (geoid grids NOT bundled yet → Mode B absolute elevation would be refused, by design) |
| numpy / scipy / pillow | 2.5.2 / 1.18.1 / 12.3.0 |
| fastapi / uvicorn / pydantic | 0.141.1 / 0.53.0 / 2.13.5 |
| Model | Depth Anything V2 Small, `depth_anything_v2_vits.pth`, 99,218,434 bytes, sha256 `715fade13be8f229f8a70cc02066f656f2423a59effd0579197bbf57860e1378`, Apache-2.0; code vendored from upstream commit `a561b849ebae10a6f5ef49e26c83cbbcd36c71bf` (cv2 import made optional) |

## Measured timings (CPU, this machine; `data/perf_cpu_sprint1.json`)
| Input | Status | preprocessing | model forward | inference stage | raster+heightfield | total | mesh |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 500×500 demo (urban) | READY | 48 ms | 739 ms | 758 ms | 106 ms | 912 ms | 500² full |
| 500×500 demo (rural) | READY | 47 ms | 776 ms | 793 ms | 85 ms | 926 ms | 500² full |
| 512×512 | READY | 45 ms | 605 ms | 623 ms | 112 ms | 781 ms | 512² full |
| 1024×1024 | READY | 86 ms | 581 ms | 602 ms | 373 ms | 1063 ms | 512² (×2 area-average, residual RMSE 0.003 rel.) |
| 2048×2048 | READY | 426 ms | 542 ms | 594 ms | 1020 ms | 2041 ms | 683² (×3, residual 0.0024) |
| 4096×4096 | READY | 1500 ms | 727 ms | 1031 ms | 6741 ms | 9274 ms | 683² (×6, residual 0.0025) |

Cold start incl. model load: 1679 ms (model load 1579 ms). Model forward is ~0.55–0.78 s regardless of input size
because inference runs at the canonical 518-px lower-bound resolution; large inputs are dominated by writing the
full-resolution rDSM GeoTIFF/previews (4096²: 69 MB TIFF). Peak RAM/VRAM: **not measured** (no psutil; CPU only).
Browser: 500² mesh = 498,002 triangles renders and orbits smoothly in the desktop-app browser pane (frame rate not instrumented).

## Bugs found and fixed during the sprint
1. `HeightfieldMeta.nodata_value = NaN` broke JSON responses (FastAPI strict JSON) → sentinel string `"NaN"`.
2. Duplicate error log lines from `JobLogger.exception` → single structured event.
3. **Block-average downsampling cropped non-divisible rasters** (2048², 4096² failed with a broadcast error and the
   mesh footprint would have been ≤ f−1 px short) → padded, nodata-aware block sums covering the full footprint;
   regression test `test_heightfield_non_divisible_sizes_cover_full_footprint`.
4. Vendored DA-V2 code imports `cv2` at module level → made optional (we do our own preprocessing).

## Observations (not fixed — recorded for Sprint 2+)
* Zero-shot DA-V2 output on the urban tile shows a large-scale gradient (top-left brighter) — the low-frequency
  scale/shift drift documented in Phase 3 (E16). This is exactly what the fine-tuned nDSM head and the DEM terrain
  layer are meant to replace; nothing in Sprint 1 should be read as a height measurement.
* The 3D viewer uses a dense regular grid; for > 768 px inputs the backend area-averages and reports the factor and
  residual. RTIN (Phase 6 D-04) is deferred to Sprint 6.
* Peak memory not instrumented; `psutil` deliberately not added (not needed for the vertical slice).
