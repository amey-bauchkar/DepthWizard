"""L0-04 Ground-masked normalized-convolution terrain layer vs raw DEM add-back (synthetic, seeded).
Scene: 300x300 px @1 m. True terrain = smooth hills. Objects: buildings (10 m, 30 m), canopy region (15 m).
Coarse DEM (30 m cells) = block-mean of first-surface with partial canopy penetration + noise, i.e. biased upward over canopy/roofs.
Question: does NC on ground-weighted cells reduce terrain error over object cells vs raw upsampled DEM?"""
import json, numpy as np
from scipy import ndimage
rng = np.random.default_rng(42)
N, CELL = 300, 30
yy, xx = np.mgrid[0:N, 0:N].astype(float)
terrain = 100 + 6*np.sin(xx/60) + 4*np.cos(yy/45) + 0.02*xx          # true terrain (m)
ndsm = np.zeros((N,N))
ndsm[40:90, 40:120] = 10.0; ndsm[150:200, 60:100] = 30.0               # buildings
canopy = np.zeros((N,N), bool); canopy[180:290, 160:290] = True; ndsm[canopy] = 15.0   # forest block
dsm_true = terrain + ndsm
# DEM: radar/stereo surface sees canopy top partially (0.6) and roofs fully; 30 m block mean; noise 1 m
surface_seen = terrain + np.where(canopy, 0.6*ndsm, ndsm)
nc = N//CELL
dem_coarse = surface_seen.reshape(nc,CELL,nc,CELL).mean(axis=(1,3)) + rng.normal(0,1.0,(nc,nc))
# --- raw DEM add-back: bicubic upsample of dem_coarse
def up(a): return ndimage.zoom(a, CELL, order=3)[:N,:N]
terrain_raw = up(dem_coarse)
# --- NC terrain (design): ground mask from PREDICTED nDSM (use true ndsm + noise to emulate model output)
ndsm_pred = np.clip(ndsm + rng.normal(0,0.8,(N,N)), 0, None)
ground = ndsm_pred < 1.0
w = ground.reshape(nc,CELL,nc,CELL).mean(axis=(1,3))                    # ground support per DEM cell
sigma_cells, w_min = 1.5, 0.1
K = lambda a: ndimage.gaussian_filter(a, sigma_cells, mode="nearest")
num, den = K(w*dem_coarse), K(w)
nc_terrain = np.where(den >= w_min, num/np.maximum(den,1e-9), dem_coarse)
raw_fallback = den < w_min
terrain_nc = up(nc_terrain)
# --- errors vs TRUE terrain, evaluated on object cells (buildings/canopy) and on ground cells
obj = ndsm > 0
def stats(err, m):
    e = err[m]; return {"ME": round(float(e.mean()),3), "RMSE": round(float(np.sqrt((e**2).mean())),3), "NMAD": round(float(1.4826*np.median(np.abs(e-np.median(e)))),3), "n": int(m.sum())}
res = {"config": {"N":N,"cell_m":CELL,"sigma_cells":sigma_cells,"w_min":w_min,"h_ground":1.0,"dem_noise_sigma":1.0,"canopy_penetration":0.6},
       "ground_support_cells": {"min": round(float(w.min()),3), "mean": round(float(w.mean()),3), "raw_fallback_cells": int(raw_fallback.sum())},
       "raw_DEM_addback": {"canopy": stats(terrain_raw-terrain, canopy), "buildings": stats(terrain_raw-terrain, obj & ~canopy), "ground": stats(terrain_raw-terrain, ~obj)},
       "NC_terrain":      {"canopy": stats(terrain_nc-terrain, canopy),  "buildings": stats(terrain_nc-terrain, obj & ~canopy),  "ground": stats(terrain_nc-terrain, ~obj)}}
# composed DSM error (design: dsm = terrain_layer + ndsm_pred) vs true DSM
for name, T in (("raw", terrain_raw), ("NC", terrain_nc)):
    e = (T + ndsm_pred) - dsm_true
    res[f"composed_DSM_error_{name}"] = {"all": stats(e, np.ones_like(obj)), "canopy": stats(e, canopy), "buildings": stats(e, obj & ~canopy)}
# consistency check statistic as designed: lowpass(terrain+ndsm) - dem
lp = (terrain_nc + ndsm_pred).reshape(nc,CELL,nc,CELL).mean(axis=(1,3)) - dem_coarse
res["consistency_check_ME_NMAD"] = [round(float(lp.mean()),3), round(float(1.4826*np.median(np.abs(lp-np.median(lp)))),3)]
res["bias_reduction_canopy_pct"] = round(100*(1 - abs(res["NC_terrain"]["canopy"]["ME"])/max(abs(res["raw_DEM_addback"]["canopy"]["ME"]),1e-9)),1)
res["bias_reduction_buildings_pct"] = round(100*(1 - abs(res["NC_terrain"]["buildings"]["ME"])/max(abs(res["raw_DEM_addback"]["buildings"]["ME"]),1e-9)),1)
json.dump(res, open("results/l0_04_nc_terrain.json","w"), indent=2); print(json.dumps(res, indent=1))
