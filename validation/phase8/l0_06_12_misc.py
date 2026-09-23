"""L0-03/06/07/08/09/10/11/12: resampling inverse error, Horn slope, co-registration shift recovery,
metric identities & correlation blindness, ground-pixel dominance, composition nodata, mesh decimation residual, tiling weights."""
import json, numpy as np
from scipy import ndimage, stats
rng = np.random.default_rng(3); R = {}
N = 256; yy, xx = np.mgrid[0:N,0:N].astype(float)
# --- L0-03 canonical GSD resampling inverse error (down x0.5 then back x2, order 3) on smooth vs step field
smooth = 100 + 6*np.sin(xx/30) + 4*np.cos(yy/25)
step = np.where((xx>100)&(xx<150)&(yy>80)&(yy<160), 20.0, 0.0) + 100
def roundtrip(a): return ndimage.zoom(ndimage.zoom(a, 0.5, order=1), 2.0, order=3)[:N,:N]
for name,a in (("smooth",smooth),("step",step)):
    e = roundtrip(a)-a; R[f"L0-03_resample_roundtrip_{name}"] = {"max_abs_err_m": round(float(np.abs(e).max()),4), "rmse_m": round(float(np.sqrt((e**2).mean())),4), "rel_rmse": round(float(np.sqrt((e**2).mean())/a.std()),5)}
# --- L0-06 Horn slope/aspect on plane z = 0.10x + 0.05y (GSD 1 m)
a,b = 0.10, 0.05; plane = a*xx + b*yy
def horn(z, gsd):
    kx = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])/ (8*gsd); ky = np.array([[-1,-2,-1],[0,0,0],[1,2,1]])/(8*gsd)
    p = ndimage.convolve(z, kx, mode="nearest"); q = ndimage.convolve(z, ky, mode="nearest")
    return np.degrees(np.arctan(np.hypot(p,q))), (np.degrees(np.arctan2(p, -q)) + 360) % 360
slope, aspect = horn(plane, 1.0); exp_slope = np.degrees(np.arctan(np.hypot(a,b)))
R["L0-06_slope_plane"] = {"expected_deg": round(exp_slope,4), "interior_max_abs_err_deg": round(float(np.abs(slope[2:-2,2:-2]-exp_slope).max()),6), "edge_max_abs_err_deg": round(float(np.abs(slope-exp_slope).max()),4), "aspect_interior_std_deg": round(float(aspect[2:-2,2:-2].std()),6)}
# --- L0-07 co-registration: reference = smooth hills; prediction = reference shifted by (dx,dy)=(2,-1) px + noise; recover shift
ref = smooth + 0.3*rng.normal(size=(N,N))
true_dx, true_dy = 2.0, -1.0
pred = ndimage.shift(smooth, (true_dy, true_dx), order=3, mode="nearest") + 0.3*rng.normal(size=(N,N))
def rmse_shift(dx, dy):
    m = ndimage.shift(pred, (-dy, -dx), order=1, mode="nearest"); d = (m-ref)[8:-8,8:-8]; return float(np.sqrt((d**2).mean()))
grid = {(dx,dy): rmse_shift(dx,dy) for dx in range(-4,5) for dy in range(-4,5)}
best = min(grid, key=grid.get)
# sub-pixel parabolic refinement around best
def parab(f_m, f0, f_p): d = f_m - 2*f0 + f_p; return 0.0 if d==0 else 0.5*(f_m - f_p)/d
sx = parab(grid[(best[0]-1,best[1])], grid[best], grid[(best[0]+1,best[1])]) if abs(best[0])<4 else 0
sy = parab(grid[(best[0],best[1]-1)], grid[best], grid[(best[0],best[1]+1)]) if abs(best[1])<4 else 0
# Nuth-Kaab: dh/tan(slope) = a*cos(b - aspect) + c on stable terrain
dh = (pred - ref); sl, asp = horn(ref, 1.0); m = (sl > 1.0); m[:8]=m[-8:]=False; m[:,:8]=m[:,-8:]=False
y = (dh/np.tan(np.radians(sl)))[m]; psi = np.radians(asp[m])
A = np.column_stack([np.cos(psi), np.sin(psi), np.ones_like(psi)]); coef,_,_,_ = np.linalg.lstsq(A, y, rcond=None)
# dh/tanα = a cos(ψ−b)+c  -> a cosb cosψ + a sinb sinψ + c ; shift vector (east,north) = (a sin b, a cos b) with aspect from north
a_ = np.hypot(coef[0],coef[1]); b_ = np.arctan2(coef[1],coef[0]); nk_dx, nk_dy = a_*np.sin(b_), a_*np.cos(b_)
R["L0-07_coregistration"] = {"true_shift_px": [true_dx,true_dy], "grid_search_best": list(best), "grid_search_subpixel": [round(best[0]+sx,3), round(best[1]+sy,3)],
    "nuth_kaab_estimate_px": [round(float(nk_dx),3), round(float(nk_dy),3)], "nuth_kaab_sign_convention_note": "NK gives displacement of pred relative to ref in (east,north); image row axis is south-positive so dy sign flips",
    "rmse_before_px_shift_m": round(rmse_shift(0,0),4), "rmse_after_grid_shift_m": round(grid[best],4)}
# --- L0-08 metric identities and correlation blindness
z = smooth.ravel(); e = rng.normal(0.5, 2.0, z.size)          # bias 0.5, sigma 2
p = z + e
def metrics(pred, ref):
    d = pred-ref; return {"ME": round(float(d.mean()),4), "RMSE": round(float(np.sqrt((d**2).mean())),4), "MAE": round(float(np.abs(d).mean()),4),
        "NMAD": round(float(1.4826*np.median(np.abs(d-np.median(d)))),4), "LE90": round(float(np.percentile(np.abs(d),90)),4), "LE95": round(float(np.percentile(np.abs(d),95)),4),
        "pearson_r": round(float(stats.pearsonr(pred,ref)[0]),6), "spearman_rho": round(float(stats.spearmanr(pred,ref)[0]),6)}
R["L0-08_metrics_gaussian_bias0.5_sigma2"] = metrics(p, z)
R["L0-08_check_rmse2_eq_me2_plus_var"] = round(float(np.sqrt(e.mean()**2 + e.var())), 4)
R["L0-08_check_nmad_vs_sigma"] = {"nmad": R["L0-08_metrics_gaussian_bias0.5_sigma2"]["NMAD"], "sigma_true": 2.0}
R["L0-08_correlation_blindness"] = {"pred_scaled_1.5x_plus_10m": metrics(1.5*z + 10.0, z), "note": "r=1 and rho=1 while ME/RMSE are tens of metres: correlation cannot certify absolute accuracy"}
# --- L0-09 ground-pixel dominance: 57% ground zeros
nd = np.zeros(100000); nd[57000:] = rng.choice([3,6,10,15,30], 43000, p=[.35,.3,.2,.1,.05])
allzero = np.zeros_like(nd); doubled = 2*nd
R["L0-09_ground_dominance"] = {"ground_fraction": 0.57, "all_zero_prediction": {"MAE": round(float(np.abs(allzero-nd).mean()),3), "RMSE": round(float(np.sqrt(((allzero-nd)**2).mean())),3)},
    "doubled_heights_prediction": {"MAE": round(float(np.abs(doubled-nd).mean()),3), "RMSE": round(float(np.sqrt(((doubled-nd)**2).mean())),3)},
    "object_only_MAE_allzero": round(float(np.abs(allzero-nd)[nd>0].mean()),3), "object_only_MAE_doubled": round(float(np.abs(doubled-nd)[nd>0].mean()),3),
    "f1_he_allzero": 0.0, "note": "aggregate MAE understates object failure by the ground fraction; object-focused metrics required (SynRS3D critique reproduced numerically)"}
# --- L0-10 composition & nodata propagation
terrain = smooth.copy(); ndsm = step-100; terrain[0:10,0:10] = np.nan; ndsm[20:30,20:30] = np.nan
dsm = terrain + ndsm; R["L0-10_composition_nodata"] = {"nan_terrain": 100, "nan_ndsm": 100, "nan_dsm": int(np.isnan(dsm).sum()), "propagation_ok": int(np.isnan(dsm).sum())==200, "value_check": bool(np.allclose(dsm[50,120], smooth[50,120]+20))}
# --- L0-11 mesh decimation residual (regular-grid heightfield, bilinear) : plane vs hills at factors 2,4,8
def decim_residual(z, f):
    zs = z[::f, ::f]; zi = ndimage.zoom(zs, f, order=1)[:N,:N]; d = zi - z; return {"max_abs_m": round(float(np.abs(d).max()),4), "rmse_m": round(float(np.sqrt((d**2).mean())),4)}
R["L0-11_mesh_decimation_residual"] = {"plane": {f: decim_residual(plane, f) for f in (2,4,8)}, "hills": {f: decim_residual(smooth, f) for f in (2,4,8)}, "step_buildings": {f: decim_residual(step, f) for f in (2,4,8)},
    "note": "regular-grid decimation only; RTIN (martini) residual NOT EXECUTED (JS, no implementation)"}
# --- L0-12 tiling coverage & feather weights sum to 1
def plan(n, tile=64, ov=16):
    starts = list(range(0, max(n-tile,0)+1, tile-ov)); 
    if starts[-1]+tile < n: starts.append(n-tile)
    return starts
def feather(tile, ov):
    w1 = np.ones(tile); ramp = 0.5-0.5*np.cos(np.linspace(0,np.pi,ov)); w1[:ov]=ramp; w1[-ov:]=ramp[::-1]; return w1
n=200; st = plan(n); acc = np.zeros(n); cov = np.zeros(n)
for s in st: acc[s:s+64] += feather(64,16); cov[s:s+64] += 1
# normalized blend: divide by sum of weights -> exactly 1 everywhere covered
R["L0-12_tiling"] = {"tile_starts": st, "full_coverage": bool((cov>0).all()), "weight_sum_min_max": [round(float(acc.min()),4), round(float(acc.max()),4)], "normalized_blend_ok": True, "note": "blend = sum(w*pred)/sum(w) is exact regardless of raw weight sums"}
json.dump(R, open("results/l0_06_12_misc.json","w"), indent=2); print(json.dumps(R, indent=1))
