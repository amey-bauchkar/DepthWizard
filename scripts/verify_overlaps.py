"""Verify building overlap status and check for stacking issues."""
import json
import numpy as np

with open("data/jobs/3dc0f73c9959/buildings.json") as f:
    data = json.load(f)

count = data["count"]
method = data.get("segmentation_method", "unknown")
print(f"Total buildings: {count}")
print(f"Method: {method}")

buildings = data["buildings"]
heights = [b["height_m"] for b in buildings]
print(f"Height range: {min(heights):.1f}m - {max(heights):.1f}m")
print(f"Mean height: {np.mean(heights):.1f}m")

# Check TRUE overlap (>80% bbox overlap = actually stacked)
true_overlaps = 0
adjacent_only = 0
for i, b1 in enumerate(buildings):
    bb1 = b1["pixel_bbox"]
    for j, b2 in enumerate(buildings):
        if j <= i:
            continue
        bb2 = b2["pixel_bbox"]
        ox = max(0, min(bb1[2], bb2[2]) - max(bb1[0], bb2[0]))
        oy = max(0, min(bb1[3], bb2[3]) - max(bb1[1], bb2[1]))
        if ox <= 0 or oy <= 0:
            continue
        overlap = ox * oy
        area1 = max(1, (bb1[2] - bb1[0]) * (bb1[3] - bb1[1]))
        area2 = max(1, (bb2[2] - bb2[0]) * (bb2[3] - bb2[1]))
        ratio = overlap / min(area1, area2)
        if ratio > 0.8:
            true_overlaps += 1
            print(f"  TRUE OVERLAP: B{b1['id']}({b1['height_m']}m, {area1}px) "
                  f"vs B{b2['id']}({b2['height_m']}m, {area2}px) = {ratio*100:.0f}%")
        elif ratio > 0.5:
            adjacent_only += 1

print(f"\nTrue overlaps (>80%): {true_overlaps}")
print(f"Adjacent buildings (50-80%): {adjacent_only}")
if true_overlaps == 0:
    print("*** PASS: No actual stacking detected! ***")
