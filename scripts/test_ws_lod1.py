import rasterio
import numpy as np
import time, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scipy.ndimage import binary_opening, binary_closing, distance_transform_edt, find_objects, sobel
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
import rasterio.features

job = "data/jobs/3dc0f73c9959"
with rasterio.open(f"{job}/ndsm.tif") as src:
    ndsm = src.read(1)
with rasterio.open(f"{job}/terrain.tif") as src:
    terr = src.read(1)
with rasterio.open(f"{job}/input.tif") as src:
    rgb = src.read()[:3].transpose(1, 2, 0)

from core.terrain.building_segmentation import extract_building_mask_ai
mask = extract_building_mask_ai(rgb, ndsm, min_height_m=2.2, veg_threshold=0.05, building_score_threshold=0.3, use_unet=False)
mask = binary_opening(mask, structure=np.ones((2, 2), dtype=bool))
mask = binary_closing(mask, structure=np.ones((3, 3), dtype=bool))

H, W = mask.shape
gsd_m = 0.5
pixel_area_m2 = gsd_m * gsd_m

dist = distance_transform_edt(mask)
ndsm_clean = np.where(mask & np.isfinite(ndsm), ndsm, 0.0)
gx = sobel(ndsm_clean, axis=1)
gy = sobel(ndsm_clean, axis=0)
grad = np.hypot(gx, gy)
ws_surface = -dist + np.clip(grad * 1.5, 0, 15)

coords = peak_local_max(dist, min_distance=9, threshold_abs=2.5)
markers = np.zeros_like(mask, dtype=np.int32)
for idx, (r, c) in enumerate(coords, start=1):
    markers[r, c] = idx

labels = watershed(ws_surface, markers, mask=mask)
n_labels = int(labels.max())

slices = find_objects(labels)

shapes_gen = rasterio.features.shapes(
    labels.astype(np.int32),
    mask=(labels > 0),
    transform=rasterio.Affine.identity(),
)

best_per_label = {}
for geom, val in shapes_gen:
    val = int(val)
    if val == 0:
        continue
    coords_poly = geom['coordinates'][0]
    if len(coords_poly) < 4:
        continue
    pts_col = np.array([p[0] for p in coords_poly])
    pts_row = np.array([p[1] for p in coords_poly])
    area_px = 0.5 * np.abs(np.dot(pts_col[:-1], pts_row[1:]) - np.dot(pts_col[1:], pts_row[:-1]))
    if val not in best_per_label or area_px > best_per_label[val][1]:
        best_per_label[val] = (geom, area_px)

from core.terrain.lod1 import rdp_simplify

buildings = []
for val, (geom, area_px) in best_per_label.items():
    area_m2 = area_px * pixel_area_m2
    if area_m2 < 15.0 or area_m2 > 50000.0:
        continue
    sl = slices[val - 1]
    if sl is None:
        continue
    sub_l = labels[sl]
    sub_n = ndsm[sl]
    sub_t = terr[sl]
    sub_mask = (sub_l == val)
    valid_h = sub_n[sub_mask & np.isfinite(sub_n)]
    if valid_h.size == 0:
        continue
    height_m = float(np.percentile(valid_h, 85))
    if height_m < 2.2:
        continue
    valid_t = sub_t[sub_mask & np.isfinite(sub_t)]
    base_elev_m = float(np.median(valid_t)) if valid_t.size else 0.0

    raw_coords = geom['coordinates'][0]
    simplified = rdp_simplify([[float(p[0]), float(p[1])] for p in raw_coords], epsilon=1.0)
    if len(simplified) < 4:
        continue

    buildings.append({
        'id': len(buildings) + 1,
        'height_m': round(height_m, 2),
        'base_elev_m': round(base_elev_m, 2),
        'area_m2': round(float(area_m2), 1),
        'n_verts': len(simplified),
    })

print(f"Successfully built {len(buildings)} buildings!")
heights = [b['height_m'] for b in buildings]
print(f"Min height: {min(heights):.1f}m, Max height: {max(heights):.1f}m, Mean height: {np.mean(heights):.1f}m")
n_verts = [b['n_verts'] for b in buildings]
print(f"Average vertices per building: {np.mean(n_verts):.1f}")
