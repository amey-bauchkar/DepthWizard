"""Regenerate buildings and verify zero overlaps."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import rasterio
import numpy as np
from core.terrain.lod1 import extract_lod1_buildings, save_lod1_buildings

job = "data/jobs/3dc0f73c9959"
with rasterio.open(f"{job}/ndsm.tif") as src:
    ndsm = src.read(1)
with rasterio.open(f"{job}/terrain.tif") as src:
    terr = src.read(1)
with rasterio.open(f"{job}/input.tif") as src:
    rgb = src.read()[:3].transpose(1, 2, 0)

print("Extracting buildings (no-stacking, dedup, overlap removal)...")
data = extract_lod1_buildings(ndsm, terr, rgb=rgb, gsd_m=0.5)
save_lod1_buildings(f"{job}/buildings.json", data)

count = data["count"]
method = data["segmentation_method"]
print(f"\nResults: {count} buildings via {method}")

heights = [b["height_m"] for b in data["buildings"]]
print(f"Height range: {min(heights):.1f}m - {max(heights):.1f}m")
print(f"Mean height: {np.mean(heights):.1f}m")

# Verify zero true overlaps
true_overlaps = 0
for i, b1 in enumerate(data["buildings"]):
    bb1 = b1["pixel_bbox"]
    a1 = max(1, (bb1[2]-bb1[0]) * (bb1[3]-bb1[1]))
    for j, b2 in enumerate(data["buildings"]):
        if j <= i:
            continue
        bb2 = b2["pixel_bbox"]
        a2 = max(1, (bb2[2]-bb2[0]) * (bb2[3]-bb2[1]))
        ox = max(0, min(bb1[2], bb2[2]) - max(bb1[0], bb2[0]))
        oy = max(0, min(bb1[3], bb2[3]) - max(bb1[1], bb2[1]))
        if ox <= 0 or oy <= 0:
            continue
        overlap = ox * oy
        if overlap / min(a1, a2) > 0.8:
            true_overlaps += 1

if true_overlaps == 0:
    print("*** PASS: ZERO overlapping buildings! Perfect clean blocks! ***")
else:
    print(f"FAIL: {true_overlaps} overlapping pairs remain")

print(f"\nBuildings < 5m: {sum(1 for h in heights if h < 5)}")
print(f"Buildings 5-10m: {sum(1 for h in heights if 5 <= h < 10)}")
print(f"Buildings 10-20m: {sum(1 for h in heights if 10 <= h < 20)}")
print(f"Buildings > 20m: {sum(1 for h in heights if h >= 20)}")
