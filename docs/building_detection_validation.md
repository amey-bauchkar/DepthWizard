# Are the LoD-1 buildings real? Measured, and fixed

**The question.** On the rural Sikkim demos the 3D city showed thousands of white blocks: 2,968 "buildings" on Teesta east, a 1.2 × 1.2 km hillside of forest and farmland.

**The reference.** [Microsoft Global ML Building Footprints](https://github.com/microsoft/GlobalMLBuildingFootprints) (ODbL) inside each demo scene. It is an independent detector working on other imagery and dates, not a survey, so these numbers measure *agreement*.

**Metrics.**
* **Area precision:** the share of our building area that lies on a reference footprint.
* **Area recall:** the share of reference footprint area we cover.
* Both allow 2 m of misalignment.

The numbers are reproduced by `python scripts/building_study_prepare.py && python scripts/validate_buildings.py`. The raw output is in `docs/building_detection_validation.json`.

## 1. What the old rule-based detector produced

It kept every pixel that was elevated in the nDSM and not clearly green, then split the result into ~4.5 m pieces.

| Scene | Our "buildings" | Reference footprints | Area precision | Area recall |
|---|---|---|---|---|
| Teesta east (rural, forest) | 2,968 | 313 | **0.02** | 0.28 |
| Teesta west (rural) | 2,926 | 304 | **0.03** | 0.47 |
| Chungthang west (forest) | 2,857 | 69 | **0.02** | 0.51 |
| Chungthang (valley town) | 1,657 | 238 | 0.13 | 0.66 |
| Namchi (hill town) | 2,933 | 1,039 | 0.34 | 0.79 |
| North Sikkim alpine (glacier, rock) | 194 | 0 | 0 | – |
| Zürich (city) | 1,961 | 289 | 0.80 | 0.91 |
| Islahiye (town) | 2,429 | 1,435 | 0.63 | 0.85 |

**Verdict: in forested hills 97–98 % of the block area was not buildings.** The blocks were tree crowns, forest edges and rock, and many real small houses were missed. In cities the detector was acceptable.

The cause is the building mask (`core/terrain/building_segmentation.py`). It uses a greenness test that shaded Himalayan canopy fails, and it never checks whether an *object* looks like a building.

## 2. Fix A: footprints from open data where they exist (now the default)

Footprint datasets cover all of India (Microsoft, Google Open Buildings, OpenStreetMap, Bhuvan). DepthWizard now takes the **outlines from a footprint dataset and only the heights from its own nDSM**. The source is chosen in this order:
1. a GeoJSON uploaded with the job;
2. bundled Microsoft footprints covering ≥ 90 % of the scene (`scripts/fetch_building_footprints.py` → `assets/footprints/`);
3. detection (Fix B).

* **Co-registration.** Outlines from other imagery can sit metres off the roofs. They are shifted to the image by FFT cross-correlation with roof evidence (nDSM on non-green pixels), within ±10 m. The shift is applied only if it raises the evidence by ≥ 10 % and does not sit on the search limit.
  * Islahiye: 4.0 m found and applied.
  * Zürich, Namchi, Chungthang: already aligned, so no shift.
* **Result counts** (the reference by construction):
  * Teesta east 305 (was 2,968);
  * North Sikkim alpine 0 (was 194);
  * Islahiye 1,432 (was 2,429);
  * Zürich 315 (Microsoft merges some city blocks).
* **Labelling.** The source and licence are shown in the building panel and in `buildings.json` (`footprints`).

## 3. Fix B: an object filter when no footprints exist (the fallback)

Each detected object is scored by a logistic model on 16 explainable features (`core/terrain/building_filter.py`, weights in `building_filter_model.json`):
* colour: greenness, saturation, brightness;
* texture;
* roof flatness and height gradient;
* size and shape: solidity, rectangle fill, elongation.

It was trained against the reference footprints. **Leave-one-scene-out**: every number below comes from a scene the model did not see.

| Scene | Objects before → after | Area precision before → after | Area recall before → after |
|---|---|---|---|
| Namchi | 2,933 → 915 | 0.34 → **0.60** | 0.79 → 0.75 |
| Chungthang | 1,657 → 273 | 0.13 → **0.29** | 0.66 → 0.47 |
| Teesta west | 2,926 → 82 | 0.03 → 0.16 | 0.47 → 0.24 |
| Teesta east | 2,968 → 77 | 0.02 → 0.11 | 0.28 → 0.07 |
| Chungthang west | 2,857 → 163 | 0.02 → 0.12 | 0.51 → 0.21 |
| Zürich | 1,961 → 1,050 | 0.80 → 0.85 | 0.91 → 0.70 |
| Islahiye | 2,429 → 1,973 | 0.63 → 0.66 | 0.85 → 0.84 |
| North Sikkim alpine | 194 → 188 | 0 → 0 | – |

**Honest reading.**
* The filter removes most false blocks: object counts fall 3–35× in the hills.
* Detection from a single RGB image is still weak in rural forested terrain: precision stays at 11–16 %, and small houses are lost.
* It learned nothing about rock: the alpine scene has no positive examples.

That is why footprints are the default and detection is labelled "approximate" in the app.

## 4. Are the heights right? (independent of the outlines)

**Zürich**, held out from training. For 287 reference footprints, the median model nDSM was compared with the median swisstopo LiDAR nDSM:

| | Value |
|---|---|
| Bias | −0.64 m |
| RMSE | 3.27 m |
| MAE | 2.29 m |
| NMAD | 2.45 m |
| Correlation r | 0.87 |

By height class (RMSE, and bias where notable):

| LiDAR height | RMSE | Bias |
|---|---|---|
| < 10 m | 3.4 m | |
| 10–20 m | 2.9 m | |
| 20–40 m | 3.2 m | −2.3 m |

One tower over 40 m was read 19 m too low. **Building heights are reliable where the model knows the building type.**

**Rural Sikkim is a real limitation.** With correct outlines, the model reads **66 % of the known buildings on Teesta east below 2 m** (median 0.8 m); single-storey houses and greenhouses are ~3–5 m. The model was trained on Swiss (v1) and Swiss + US (v2) buildings and under-reads small South Asian rural houses. The app flags every such building `LOW_PREDICTED_HEIGHT`, and says so in the building panel when ≥ 25 % are affected.

The towns read plausibly (median height):

| Town | Median building height |
|---|---|
| Namchi | 7.7 m |
| Islahiye | 8.1 m |
| Chungthang | 4.1 m |

**The real fix is Indian training data (roadmap).**

## 5. Effect on the change screening (Islahiye)

With footprints, the building-level screening flags **37 "major height loss"** of 1,432 buildings.

A random visual audit of 24 flags found:
* 17 collapsed;
* 3–4 standing, i.e. false alarms;
* 3–4 uncertain.

The old detector flagged 69. Mapping them onto the new run:
* 37 are the same buildings;
* 11 are now below the threshold, because medians over merged footprints are steadier;
* 21 fall where the Microsoft dataset has no footprint. About 12 of those were trees, correctly dropped; about 8 were real collapsed buildings missing from the dataset.

Those buildings still appear in the pixel-level height-loss map, which does not depend on footprints. The app states this in the caveats.
