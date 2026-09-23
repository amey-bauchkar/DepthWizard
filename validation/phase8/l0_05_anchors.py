"""L0-05 Robust anchor fit: recover known terrain offset and object-layer scale with outliers; hold-out residuals."""
import json, numpy as np
from scipy import stats
from sklearn.linear_model import RANSACRegressor, LinearRegression, HuberRegressor
rng = np.random.default_rng(7)
N = 300
yy, xx = np.mgrid[0:N,0:N].astype(float)
terrain = 100 + 6*np.sin(xx/60) + 4*np.cos(yy/45)
ndsm = np.zeros((N,N)); ndsm[40:90,40:120]=10; ndsm[150:200,60:100]=30; ndsm[200:280,180:280]=15
TRUE_OFFSET, TRUE_SCALE = -3.0, 1.2          # predicted terrain is 3 m low; predicted nDSM is 20% too tall
terrain_pred = terrain + TRUE_OFFSET
ndsm_pred = ndsm * TRUE_SCALE
def sample(arr, pts): return np.array([arr[int(r),int(c)] for r,c in pts])
res = {"true_offset": TRUE_OFFSET, "true_scale": TRUE_SCALE}
for n_ground in (3, 5, 10, 30):
    # ground anchors on ground pixels with 0.5 m noise (ICESat-like optimistic), plus 2 gross outliers (+20 m)
    g = np.argwhere(ndsm==0); idx = rng.choice(len(g), n_ground, replace=False); pts = g[idx]
    z = sample(terrain, pts) + rng.normal(0,0.5,n_ground)
    if n_ground >= 5: z[:2] += 20.0
    r = z - sample(terrain_pred, pts)                       # residual = true offset (+noise, +outliers)
    off_theilsen_median = float(np.median(r))                # 0-dof offset: median (Theil-Sen degenerates to median for constant)
    hub = HuberRegressor().fit(np.zeros((n_ground,1)), r); off_huber = float(hub.intercept_)
    blunders = int((np.abs(r-off_theilsen_median) > 3*1.4826*np.median(np.abs(r-np.median(r)))).sum())
    res[f"ground_n{n_ground}"] = {"offset_median": round(off_theilsen_median,3), "offset_huber": round(off_huber,3), "error_m": round(abs(off_theilsen_median-(-TRUE_OFFSET)),3), "blunders_flagged": blunders}
# object scale: anchors on roofs/canopy tops; z_anchor - terrain_pred_corrected vs ndsm_pred
off = res["ground_n30"]["offset_median"]
terrain_corr = terrain_pred + off
for n_obj in (3, 5, 8, 20):
    o = np.argwhere(ndsm>0); idx = rng.choice(len(o), n_obj, replace=False); pts = o[idx]
    z = sample(terrain, pts) + sample(ndsm, pts) + rng.normal(0,0.5,n_obj)
    if n_obj >= 5: z[0] -= 12.0                              # one gross outlier
    x = sample(ndsm_pred, pts).reshape(-1,1); y = z - sample(terrain_corr, pts)
    try:
        ran = RANSACRegressor(LinearRegression(fit_intercept=False), min_samples=max(2, n_obj//2), residual_threshold=1.5, random_state=0).fit(x, y)
        a = float(ran.estimator_.coef_[0]); inl = int(ran.inlier_mask_.sum())
    except Exception as e:
        a, inl = None, str(e)
    res[f"object_n{n_obj}"] = {"scale_ransac": None if a is None else round(a,4), "expected": round(1/TRUE_SCALE,4), "inliers": inl}
# hold-out check with N=30 ground: 30% held out
g = np.argwhere(ndsm==0); idx = rng.choice(len(g), 30, replace=False); pts = g[idx]
z = sample(terrain, pts) + rng.normal(0,0.5,30); z[:3] += 20
perm = rng.permutation(30); fit_i, hold_i = perm[:21], perm[21:]
off_fit = float(np.median((z - sample(terrain_pred, pts))[fit_i]))
hold_res = (z[hold_i] - (sample(terrain_pred, pts)[hold_i] + off_fit))
res["holdout"] = {"n_fit": 21, "n_hold": 9, "offset_fit": round(off_fit,3), "holdout_ME": round(float(hold_res.mean()),3), "holdout_NMAD": round(float(1.4826*np.median(np.abs(hold_res-np.median(hold_res)))),3), "holdout_RMSE_incl_outliers": round(float(np.sqrt((hold_res**2).mean())),3)}
res["note"] = "Synthetic: noise sigma 0.5 m, gross outliers injected; demonstrates estimator correctness only, not real anchor availability/accuracy."
json.dump(res, open("results/l0_05_anchors.json","w"), indent=2); print(json.dumps(res, indent=1))
