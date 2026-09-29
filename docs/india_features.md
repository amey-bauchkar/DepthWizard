# Features for Indian disaster management

The features added for ISRO / NDMA use: what each does, its data source, and what is **not** yet proven.

## 1. Indian data sources

| Source | How DepthWizard uses it | Status |
|---|---|---|
| **Bhuvan WMS** (NRSC / ISRO) | *Bhuvan layer* picker in the hazard map. It shows state road network, land use / land cover, slope and drainage layers aligned to the scene (`core/geo/bhuvan.py`, `GET /api/jobs/{id}/bhuvan`). | Works without login. Bhuvan's vector service (WFS) is disabled, so these layers are display only. Road analysis uses OpenStreetMap vectors. The national layers `mmi:india_roads` and LULC 1:250k do not render at scene scale; state layers do (Sikkim wired; other states follow the same layer names). |
| **Cartosat-2/2E/3, Resourcesat-2/2A LISS-IV** (NRSC Bhoonidhi) | Upload the product **zip** directly (`core/ingest/isro.py`), or convert with `python -c "from core.ingest.isro import convert; convert('product.zip', 'rgb.tif')"`. MX bands are re-ordered to RGB. LISS-IV (no blue band) gets simulated natural colour. PAN + MX are Brovey pan-sharpened on the PAN grid. | Tested on synthetic products that follow the NRSC band-file layout (`tests/unit/test_india_features.py`). **Not yet tested on a real Bhoonidhi product**: none is freely downloadable. |
| **CartoDEM** (Bhuvan) and **any user DEM** | Used automatically. `scripts/hazard_study_prepare.py --dem` measures whether a DEM improves flood accuracy (`docs/hazard_validation.md`). | Works. |
| **NDEM** (ndem.nrsc.gov.in) | Not integrated. Its hazard layers need an NDEM account, which is issued to government users. | To do with an account from the SIH nodal contacts. |

## 2. Landslide hazard screening

`POST /api/jobs/{id}/disaster/landslide` · `core/disaster/landslide.py`

**Susceptibility** uses **BIS IS 14496 (Part 2): 1998** (Landslide Hazard Evaluation Factor), the Indian standard for hazard zonation.

| Factor | Rating range | Source |
|---|---|---|
| slope morphometry | 0.5–2.0 | DepthWizard terrain |
| relative relief | 0.3–1.0 | DepthWizard terrain |
| land use / cover | 0.65–2.0 | image greenness + canopy height + buildings |
| lithology | 0.2–2.0 | user (e.g. GSI Bhukosh maps) |
| structure | 0.3–2.0 | user |
| ground water | 0–1.0 | user, or taken from the rainfall trigger |

Classes: very low < 3.5, low 3.5–5, moderate 5–6, high 6–7.5, very high > 7.5.

**Unknown geology is shown, not hidden.** With lithology and structure unknown, the result also gives the class shares for the best and worst geology. At Chungthang this is 0 % vs 97 % high hazard, so the geology inputs decide the answer.

**Rainfall trigger** uses the Nepal Himalaya threshold of Dahal & Hasegawa (2008), I = 73.90 D^-0.79 (mm/h, hours): about 144 mm in 24 h. Rain is either entered by the user or read from Open-Meteo for the scene centre (last 5 days + 3-day forecast, free, no key).

**New slope scars** are found as new bare ground on slopes of 20° or more in Sentinel-2 (last 60 days vs the same period a year earlier). This uses the same rule as the flood-scar mapping validated in `docs/hazard_validation.md`.

**Not validated yet.** The classes have not been compared with a landslide inventory. Next step: the GSI National Landslide Susceptibility Mapping inventory or NASA's Global Landslide Catalog over the Sikkim scenes.

## 3. Road access: which settlements are cut off

`POST /api/jobs/{id}/disaster/roads` · `core/disaster/roads.py` · data: `python scripts/fetch_roads.py` → `assets/roads/`

* **Network.** OpenStreetMap roads, tracks and footpaths, fetched 400 m beyond the scene. A node outside the scene counts as an exit to the outside world.
* **Cuts.**
  * A road link is cut where the last flood result is deeper than 0.3 m, the depth that can float a car.
  * Bridges are cut only above 5 m of water, an assumption because deck heights are not known.
  * Links in high or very high landslide hazard can also be cut, as an option.
* **Settlements.** Settlements are building clusters (buildings within 25 m of each other, at least 3 buildings), named after the nearest OSM place. A settlement is **cut off** when it had a route to an exit before the hazard and has none after. This is checked for vehicles and on foot.
* **Example, Chungthang.**
  * With the river 18 m above normal, 1.8 km of 6.3 km of road is cut and all 5 settlements (about 1,660 people, estimated) lose their road out.
  * At 3 m, 0.6 km is cut and 3 of the 5 are cut off.
  * The Oct 2023 GLOF did isolate Chungthang for weeks. That is consistent with this result but is not a measured validation.
* **Population** is estimated as 4.9 persons (Census 2011 mean household size) per floor of each building. It counts non-residential buildings too and is labelled as rough everywhere.

## 4. One-click damage report (PDF)

The *Damage report (PDF)* button calls `GET /api/jobs/{id}/report.pdf` (numbers as JSON at `/report`) · `core/export/report.py`

The report has four A4 pages, built from the **latest** result of each screening:
1. **Situation summary:** flood area and range, buildings and people exposed, roads cut, settlements cut off, landslide hazard and rainfall trigger, and landing sites.
2. **Map:** flood, landslide, roads, settlements and landing sites.
3. **Action lists:** settlements to reach, roads to clear, landing sites with coordinates and approach bearings, and the most exposed buildings.
4. **Limits:** every warning from every screening.

Screenings that were not run are listed as "not run", never filled with defaults the user did not choose. Landing sites that lie inside the current flood are removed from the report with a note. Found while testing: 20 of 22 Chungthang sites were on river gravel that the same flood covers.
