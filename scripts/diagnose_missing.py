"""Diagnose why buildings are missing — find where they're being filtered out."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import rasterio
import numpy as np
from scipy.ndimage import label as nd_label, binary_opening, binary_closing

job = "data/jobs/3dc0f73c9959"
with rasterio.open(f"{job}/ndsm.tif") as src:
    ndsm = src.read(1)
with rasterio.open(f"{job}/terrain.tif") as src:
    terr = src.read(1)
with rasterio.open(f"{job}/input.tif") as src:
    rgb = src.read()[:3].transpose(1, 2, 0)

H, W = ndsm.shape
gsd_m = 0.5
pixel_area_m2 = gsd_m * gsd_m
print(f"Image size: {W}x{H}, GSD: {gsd_m}m")

# Step 1: Check raw height mask
raw_mask = (ndsm >= 2.2) & np.isfinite(ndsm) & np.isfinite(terr) & (ndsm < 200.0)
raw_px = int(raw_mask.sum())
print(f"\n=== RAW HEIGHT MASK (ndsm >= 2.2m) ===")
print(f"  Pixels: {raw_px} ({raw_px/(H*W)*100:.1f}%)")

# Step 2: AI segmentation mask
from core.terrain.building_segmentation import extract_building_mask_ai
ai_mask = extract_building_mask_ai(rgb, ndsm, min_height_m=2.2, veg_threshold=0.05,
                                    building_score_threshold=0.3, use_unet=True)
ai_px = int(ai_mask.sum())
print(f"\n=== AI SEGMENTATION MASK ===")
print(f"  Pixels: {ai_px} ({ai_px/(H*W)*100:.1f}%)")
print(f"  Lost from raw: {raw_px - ai_px} pixels ({(raw_px - ai_px)/max(raw_px,1)*100:.1f}%)")

# Step 3: After morphological cleanup
clean = binary_opening(ai_mask, structure=np.ones((2, 2), dtype=bool))
clean = binary_closing(clean, structure=np.ones((3, 3), dtype=bool))
clean_px = int(clean.sum())
print(f"\n=== AFTER MORPH CLEANUP ===")
print(f"  Pixels: {clean_px} ({clean_px/(H*W)*100:.1f}%)")
print(f"  Lost from AI: {ai_px - clean_px} pixels")

# Step 4: After gradient boundary carving
from core.terrain.lod1 import _carve_height_boundaries
carved = _carve_height_boundaries(clean, ndsm)
carved_px = int(carved.sum())
print(f"\n=== AFTER GRADIENT CARVING ===")
print(f"  Pixels: {carved_px} ({carved_px/(H*W)*100:.1f}%)")
print(f"  Lost from clean: {clean_px - carved_px} pixels ({(clean_px - carved_px)/max(clean_px,1)*100:.1f}%)")

# Step 5: Connected components
labeled, n_comp = nd_label(carved)
print(f"\n=== CONNECTED COMPONENTS ===")
print(f"  Total components: {n_comp}")

# Count by size
min_area_px = max(4, int(round(15.0 / pixel_area_m2)))
sizes = []
for i in range(1, n_comp + 1):
    sizes.append(int((labeled == i).sum()))
sizes = sorted(sizes, reverse=True)

too_small = sum(1 for s in sizes if s < min_area_px)
valid_size = sum(1 for s in sizes if s >= min_area_px)
print(f"  Min area threshold: {min_area_px} px ({15.0} m2)")
print(f"  Components >= min_area: {valid_size}")
print(f"  Components < min_area (filtered out): {too_small}")
print(f"  Largest 10 components (px): {sizes[:10]}")
print(f"  Sizes at 50th/90th/99th percentile: {sizes[len(sizes)//2] if sizes else 0}, "
      f"{sizes[len(sizes)//10] if len(sizes)>10 else 0}, {sizes[len(sizes)//100] if len(sizes)>100 else 0}")

# Step 6: Check spatial distribution - left half vs right half
left_mask = np.zeros_like(ai_mask)
left_mask[:, :W//2] = ai_mask[:, :W//2]
right_mask = np.zeros_like(ai_mask)
right_mask[:, W//2:] = ai_mask[:, W//2:]
print(f"\n=== SPATIAL DISTRIBUTION ===")
print(f"  Left half building pixels: {int(left_mask.sum())}")
print(f"  Right half building pixels: {int(right_mask.sum())}")

# Check nDSM values in right half
right_ndsm = ndsm[:, W//2:]
right_above_2m = (right_ndsm >= 2.2) & np.isfinite(right_ndsm)
print(f"  Right half pixels with ndsm >= 2.2m: {int(right_above_2m.sum())}")
right_above_1m = (right_ndsm >= 1.0) & np.isfinite(right_ndsm)
print(f"  Right half pixels with ndsm >= 1.0m: {int(right_above_1m.sum())}")

# Step 7: What if we lower thresholds?
low_mask = (ndsm >= 1.5) & np.isfinite(ndsm) & np.isfinite(terr) & (ndsm < 200.0)
low_px = int(low_mask.sum())
print(f"\n=== WITH LOWER HEIGHT THRESHOLD (1.5m) ===")
print(f"  Pixels: {low_px} ({low_px/(H*W)*100:.1f}%)")
print(f"  Additional pixels vs 2.2m: {low_px - raw_px}")
