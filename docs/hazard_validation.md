# Hazard screening: checks against real-world data and the fixes they led to

The scenes were processed with the current pipeline (`python scripts/hazard_study_prepare.py`).

## Helicopter landing zones

**Real helipad test.** OpenStreetMap maps **Namchi Helipad** (27.1567 N, 88.3220 E) inside the Teesta west demo. The screening found:

| Size | Sites in the whole 1.44 km² scene | Where they are |
|---|---|---|
| 1 (25 m) | 6 | all within 49 m of the helipad |
| 2 (35 m) | 2 | 23 m from the helipad |
| 3 (50 m) | 1 | 25 m from the helipad |

That was before the obstacle-margin change below. The screening therefore picked out the real helipad on its own, with nothing else proposed across the scene.

**Empty results are now explained.** Namchi town gave 0 sites at every size. The explanation is now shown with the result:
* 100 % of pad centres have a building or tree within the pad plus a 10 m buffer.
* 84–98 % are too steep.

ICESat-2 confirms this is real forest: 113 of 117 laser segments there show canopy of 1 m or more, median 14.5 m. So the zero is correct. The app now says why, and suggests a smaller pad size or a hover or winch operation.

**Obstacle margin from measurements.** Against ICESat-2, the model reads the top of trees and roofs in Sikkim **2.7–6.7 m too low** (mean error on 21–408 checkpoints per scene). When a job has a point validation with at least 20 checkpoints, the approach check now adds that measured under-reading to its 1-sigma margin.
* Example, Teesta west: 5.5 → 8.8 m.
* The 50 m pad at the helipad then has no clear approach, which is the cautious answer. The 25 m size still finds 4 sites there.
* Without a validation, the result warns that obstacles may be under-read.

## Flood screening

**Problem.** A single still water level is meaningless in a 400 m-relief Himalayan valley. At Chungthang, a level 10 m above the lowest point flooded 2.7 % of the scene and **0 buildings**, although the valley flats sit just above the river.

**New default in hilly scenes: river rise above the channel (HAND, height above nearest drainage).**
* Nobre et al. 2011; the approach behind NOAA's HAND flood-inundation maps.
* The water surface follows the valley slope. On an analytic sloping valley HAND is within 0.45–1 m of the exact value (`tests/unit/test_hand_flood.py`).
* Mapped rivers from OpenStreetMap are **burned into** the flow routing, using an AGREE trench of 20 m depth and 90 m half-width. The channels therefore follow the real Lachen Chu, Lachung Chu and Teesta rather than the 30 m DEM's approximate valley lines.
* Unmapped channels start at 2 ha of contributing area.

**Chungthang with the river model:**

| River rise | Area flooded | Buildings exposed | What floods |
|---|---|---|---|
| 3 m | 11 % | 38 | the river gravel beds |
| 10 m | 20 % | 77 | the riverside flats |
| 20 m | 30 % | 112 | the lower town |

The still-water model remains for lakes and flat plains. The app chooses the default from the terrain relief: river model if P95 − P5 > 40 m.

**Limits** (stated with every result):
* no discharge, hydraulics, embankments or timing;
* terrain from the 30 m DEM;
* cells that drain out of the scene before reaching a channel are never shown wet.


## Flood vs the REAL 4 Oct 2023 Teesta GLOF at Chungthang (measured)

**Observed extent.** Sentinel-2 L2A, same season: 25 Nov 2022 before and 30 Dec 2023 after (`scripts/glof_observed_extent.py`). The flood scar is fresh sediment or water where there was vegetation before, plus the pre-flood river channel: **23.8 ha** in the 1.44 km² scene. Visually it follows both river banks, the lower town at the confluence, and the Teesta down to the dam.

**Score.** `scripts/validate_flood.py` → `docs/flood_validation_glof.json`. The model is compared at a 10 m grid, taking each model's best water height, because the GLOF's true stage at Chungthang is not known here:

| Model / surface | Best height | Precision | Recall | F1 | IoU |
|---|---|---|---|---|---|
| River rise (HAND), terrain layer + OSM rivers (app default) | 18 m | 0.45 | 0.76 | **0.57** | 0.40 |
| River rise (HAND), raw CartoDEM | 22 m | 0.48 | 0.74 | 0.59 | 0.41 |
| River rise (HAND), raw Copernicus | 32 m | 0.41 | 0.86 | 0.55 | 0.38 |
| River rise, mapped rivers only (no side gullies) | 22 m | 0.51 | 0.58 | 0.54 | 0.37 |
| Still water level | 1593 m | 0.48 | 0.80 | 0.60 | 0.43 |

**Reading.**
* Every variant finds three-quarters to four-fifths of the area the GLOF really hit, and floods about 1.5–2× that area.
* The error map shows two causes:
  * the 30 m DEM smooths the steep gorge walls, so scoured banks look too high (misses);
  * low terraces of the town fan look floodable but were not scarred (false alarms). Some of these may have been wet without a visible scar, so part of this apparent over-flooding may be the reference's limitation rather than the model's.
* Swapping DEMs or channel rules moves F1 only within 0.54–0.60. **The ceiling is the 30 m DEM, not the flood method.**
* At this compact confluence a single still level scores as well as HAND. HAND remains the default in hills because a still level cannot follow a river's gradient over longer reaches. That advantage is not measured here (the scene is only 1.2 km).
* **What would raise the score: a finer terrain model.** For example Cartosat-1 stereo 10 m, or ISRO/NRSC high-resolution DEMs uploaded as the job's DEM. Those are used automatically.

## Second real flood: Sunkoshi / Roshi Khola, Nepal, 27–28 Sep 2024 (measured)

One event is not proof, so the same test was repeated on a different flood in a different country.

* **Input image:** Maxar Open Data, event *Nepal-Floods-Sept-2024*, acquisition 10300100FC189500 (29 May 2024, before the flood, 25° off-nadir). The 1.2 km chip is centred at 27.434 N, 85.836 E (`assets/demo/nepal/sunkoshi_rgb_0.5m.tif`, CC BY-NC 4.0).
* **Elevation:** Copernicus GLO-30 (CartoDEM does not cover Nepal). OSM rivers: `assets/waterways/nepal_sunkoshi.geojson`.
* **Why this site:** a scan of the whole Maxar pre-flood image for fresh Sentinel-2 flood scars (3 Dec 2023 vs 27 Dec 2024) put this 1.2 km window first: **48.7 ha** of fields buried under new river sediment at the confluence. The scar was checked by eye on the before/after images.
* **Reproduce:** `python scripts/glof_observed_extent.py nepal_sunkoshi && python scripts/hazard_study_prepare.py nepal_sunkoshi && python scripts/validate_flood.py nepal_sunkoshi` → `docs/flood_validation_nepal_sunkoshi.json`.

| Model | Best height | Precision | Recall | F1 | IoU |
|---|---|---|---|---|---|
| River rise (HAND), app default | 9 m | 0.56 | 0.92 | **0.70** | 0.54 |
| Still water level | 523 m | 0.63 | 0.88 | 0.73 | 0.58 |

## Fair comparison across both floods

"Best height" is chosen after seeing the answer, which flatters every model. Two stricter checks:

| Check | Teesta GLOF (Chungthang) | Sunkoshi (Nepal) |
|---|---|---|
| River model, best height | 0.57 (18 m) | 0.70 (9 m) |
| **River model, height taken from the OTHER flood** | **0.51** (9 m) | **0.69** (18 m) |
| Still level, best height | 0.60 | 0.73 |
| Still level, height from the other flood | not possible: the level is an absolute elevation (1593 m vs 523 m) | same |
| Baseline: "everything within D m of a mapped river", best D | 0.59 (D = 70 m) | 0.57 (D = 590 m) |

**Reading.**
* The river model's water height carries over from one flood to the other with little loss: F1 drops by 0.05 at Chungthang and 0.01 in Nepal. A height in "metres above the river" means the same thing in any valley. A still level and a river buffer do not transfer: their best settings differ by 1,070 m of elevation and 520 m of distance between the two sites.
* With the height fixed in advance, the river model beats the river-buffer baseline in Nepal (0.69 vs 0.57) and loses to it at Chungthang (0.51 vs 0.59). At that narrow gorge site the 30 m DEM, not the method, sets the limit (section above).
* In both floods the model finds 76–92 % of the flooded area and floods about 1.6× too much.

## Flood confidence map: is it honest? (measured)

Every flood result now carries `flood_probability.tif`: P(wet) = Φ((W − S) / σ), where σ is the job's measured terrain error (7.6 m CartoDEM, 8.5 m Copernicus, against ICESat-2). The result also reports "likely" (P ≥ 0.9) and "possible" (P ≥ 0.1) areas and building counts. At the transferred heights:

| Band | Predicted | Observed flooded, Chungthang | Observed flooded, Nepal |
|---|---|---|---|
| likely (P ≥ 0.9) | 0.97 | none at 9 m | 0.57 |
| possible (0.1–0.9) | 0.57–0.59 | 0.44 | 0.23 |
| unlikely (P < 0.1) | 0.00 | 0.05 | 0.06 |

**The probabilities are over-confident in the wet bands.** The DEM's systematic over-flooding is not random noise, so Φ(·) overstates certainty. The "unlikely" band is reliable, however: 94–97 % of it stayed dry. The app therefore shows the **measured** rates next to the range ("of the 'likely' area 49–57 % really flooded…", `FLOOD_BAND_OBSERVED` in `core/disaster/flood.py`) rather than the raw probability. The main use of the map is ruling ground out as safe, not predicting exact flood edges.

## Finer DEM: how to test one

A job uploaded with a DEM uses that DEM automatically. To measure whether it helps, run one command against the same observed flood:

```
python scripts/hazard_study_prepare.py chungthang --dem <cartosat_10m.tif> --dem-vcrs EGM2008 --tag cartosat
python scripts/validate_flood.py chungthang__cartosat
```

* **Tested with Copernicus as the uploaded DEM:** river F1 0.555, still level 0.601, so no gain over the bundled CartoDEM (0.57 / 0.60). Both are 30 m.
* **DepthWizard's own terrain layer** (image model + DEM fusion) is effectively such a "finer DEM", and it did not raise F1 either (0.57 vs 0.59 on raw CartoDEM): its fine detail comes from the image, which sees roofs and canopy, not the ground under them.
* **Not yet tested:** no free elevation model finer than 30 m covers Sikkim or Nepal. Cartosat-1 stereo DEMs (10 m or better) are distributed by NRSC (Bhoonidhi) on request. This is the single most useful data request to make through the SIH nodal contacts.
