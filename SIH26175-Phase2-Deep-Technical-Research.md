# Phase 2 — Deep Technical Research

SIH26175 DepthWizard · 2026-09-18

Scope note: this phase does not design our solution. Findings are reported as what the literature shows, not as recommendations. Coverage is real but not exhaustive — tracks with lighter search effort are flagged in Section 1.

## 1. Executive Research Summary

The two halves of DepthWizard rest on separable, differently-mature literatures. Foundation-model monocular depth (natural images) is mature and fast-moving — new state of the art roughly every few months. Monocular height estimation *from remote sensing* is a smaller, active sub-field with real benchmarks, real numbers, and a clear best-known method as of this research (HTC-DC Net). Relative-to-metric scale recovery has a well-established mathematical toolkit borrowed from robotics/SLAM. 3D terrain rendering is commodity technology. The genuine open problems are narrower than the PS's language suggests: not "can this be done at all" but "how well does it transfer to unseen terrain and unseen sensors," which is exactly what the evaluation criteria test.

**Coverage honesty**: search effort was concentrated on Tracks 1, 2, 3, 4, 5, 7, 8, 14, 15, and 17, where verifiable primary sources were plentiful. Tracks 6 (GCP-specific, beyond general scale-shift alignment), 9 (validation-protocol specifics), 10–13 (domain gap/adaptation, geospatial pipeline — mostly established engineering practice rather than a live research question), 16 (systematic GitHub trawl), and 20 (patents) received lighter, targeted searches rather than exhaustive sweeps. This is stated plainly rather than papered over, per the quality-control requirement.

**Headline findings** (detailed with evidence in the sections below):

- A specific, well-documented method — [HTC-DC Net](https://arxiv.org/pdf/2309.16486) — already solves a large fraction of this PS's DSM-accuracy half, on real benchmarks, with published RMSE numbers by terrain type.
- The organiser's recommended dataset, [GAMUS](https://arxiv.org/pdf/2305.14914), covers only five U.S. cities (Oklahoma, Washington D.C., Philadelphia, Jacksonville, New York) — no hilly or forested terrain, and no international coverage. It does not by itself satisfy the PS's four-terrain-type stability requirement.
- A newer synthetic dataset, [SynRS3D](https://arxiv.org/pdf/2406.18151) (NeurIPS 2024), explicitly targets the sim-to-real domain gap for height estimation and covers a wider range of GSDs and city styles.
- Global-affine scale-shift recovery from sparse reference points is a standard, well-understood technique, but multiple independent papers show a *single* global factor leaves substantial spatially-varying residual error — this is not a hypothesis, it is a measured, repeated finding.
- No commercial single-image monocular height product was found. The commercial state of the art (Maxar Precision3D, formerly Vricon) uses multi-view stereo from many satellite passes, not monocular estimation — a meaningfully different technique from what this PS asks for.

## 2. Monocular Depth Estimation Literature

The field moved from per-scene regression, to affine-invariant relative depth trained at scale, to metric depth via a fine-tuned head or camera-aware architecture.

| Model | Relative/Metric | Backbone/size | Training domain | Open source | Key evidence |
| --- | --- | --- | --- | --- | --- |
| [MiDaS v3.1](https://arxiv.org/pdf/2307.14460) | Relative (affine-invariant) | ResNeXt/ViT/BEiT/Swin, model zoo | 12 mixed real+synthetic datasets | Yes, MIT | Established baseline; feeds ZoeDepth |
| [ZoeDepth](https://arxiv.org/pdf/2302.12288) | Both — relative pretraining, metric fine-tune (NYU/KITTI) | MiDaS v3.1 + metric bin head | Mixed relative + 2 metric datasets | Yes | Combines relative pretraining with metric bins module |
| [Depth Anything V2](https://arxiv.org/abs/2406.09414) | Both — relative by default, metric variants fine-tuned | 25M–1.3B params (ViT-S/B/L/G) | 595K synthetic + 62M+ pseudo-labeled real | Yes, Apache-2.0 | 10x faster than diffusion-based competitors at comparable accuracy |
| [Metric3D v2](https://arxiv.org/html/2404.15506v4) | Metric, zero-shot | ViT-based, canonical camera-space transform | Broad multi-dataset | Yes | Reports best-in-class accuracy on several unseen benchmarks in a 2024 wildlife-domain benchmark study |
| [UniDepth v1/v2](https://arxiv.org/pdf/2507.02148) | Metric, zero-shot, no intrinsics required | Transformer, self-promptable camera module | Broad multi-dataset | Yes | Predicts 3D point clouds directly; strong on NYUv2/KITTI |
| [Marigold](https://arxiv.org/pdf/2409.04086) | Relative (diffusion-based) | Repurposed Stable Diffusion | Synthetic | Yes | High quality but far slower than DA-V2 (Section 13) |

**Independent third-party benchmark evidence** (not vendor-reported): a 2025 wildlife-domain evaluation found [Depth Anything V2 achieved 0.454 m MAE and 0.962 correlation, while ZoeDepth degraded sharply outdoors (3.087 m MAE)](https://arxiv.org/html/2510.04723v1) — direct evidence that a model's published indoor/driving-benchmark accuracy does not predict its accuracy in an unfamiliar outdoor domain. This is the same domain-transfer risk the PS's evaluation criteria are built to catch.

**Established fact, explicitly warned against by the source authors themselves**: Depth Anything V2's own project page states plainly that models trained on NYUv2 or KITTI "fail to" generalize outside those domains — the authors do not claim their relative-depth models transfer metric accuracy to arbitrary new domains without recalibration.

**Open question**: no source found directly reports any of these foundation models' zero-shot performance specifically on nadir satellite imagery — that evidence gap is closed only by the remote-sensing-specific literature in Section 3, not by the general-purpose depth literature.

## 3. Remote-Sensing Monocular Depth & Single-Image DSM Literature

This is the field's real name for this problem: **monocular height estimation (MHE)**, mostly published in IEEE TGRS, ISPRS J. Photogrammetry, and IGARSS. It is more mature than the PS's phrasing implies.

**Timeline of representative methods** (established, chronological, all predicting height/nDSM, not raw depth):

- [IM2HEIGHT (2018)](https://arxiv.org/pdf/1802.10249) — first deep-learning encoder-decoder for single-image height, DLR/TUM.
- [Amirkolaee & Arefi (2019)](https://arxiv.org/pdf/2301.04581) — CNN encoder-decoder with patch-merging smoothing; a recurring strong baseline in later papers.
- Ghamisi & Yokoya, IMG2DSM (2018) and Paoletti et al., U-IMG2DSM (2020) — GAN-based image-to-height translation.
- DORN-based ordinal regression adapted to height (Li, Wang & Fang, 2020).
- [HTC-DC Net (2023)](https://arxiv.org/pdf/2309.16486) — current strongest published method found in this research; classification-regression paradigm with a head-tail-cut for the extreme long-tailed height distribution.
- [HeightFormer (2023)](https://arxiv.org/pdf/2310.07995) and [TSE-Net (2025)](https://arxiv.org/html/2511.13552v1) — most recent, transformer-based and semi-supervised variants.

**Real, verified benchmark numbers** (HTC-DC Net, RMSE in metres, lower is better):

| Dataset | GSD | Best prior baseline RMSE | HTC-DC Net RMSE | Notes |
| --- | --- | --- | --- | --- |
| DFC19 (satellite, Jacksonville/Omaha) | 1.3 m | 2.87 (Amirkolaee) | 2.12 | Building-pixel RMSE improved by 4.26 m |
| ISPRS Vaihingen (aerial) | 0.09 m | 1.49 (U-Net) | 1.30 | Best absolute accuracy of the three — finer GSD helps |
| GBH, seen cities (satellite, Planet) | 3 m | 4.58 (U-Net) | 4.49 | Larger, more diverse dataset; smaller margin of improvement |
| GBH, unseen cities (Sao Paulo, Guangzhou) | 3 m | — | 9.30–11.82 (RMSE), building-wise up to 11.8 | **All methods degrade sharply on cities never seen in training** |

The unseen-city result is the single most important number in this research for DepthWizard: even the best published method loses roughly 2–3x accuracy when tested on a city with a different urban morphology than its training data. The paper's own authors attribute this explicitly to domain shift and state further work on domain generalization is needed — it is not a solved problem, even for the current state of the art.

**Terrain-type coverage gap, evidenced not assumed**: none of DFC19, ISPRS Vaihingen, or GBH's seen-city split include hilly or forested terrain as a distinct evaluated category — all are urban/suburban. The PS's stability requirement across urban, sparse, hilly, and forested terrain therefore asks for evidence beyond what the field's own standard benchmarks report.

**SynRS3D ([NeurIPS 2024, arXiv:2406.18151](https://arxiv.org/abs/2406.18151))** is the one dataset-and-method pair found that explicitly targets this generalization gap: 69,667 synthetic images at 0.05–1 m GSD across six city styles, paired with a domain-adaptation method (RS3DAda) specifically for transferring synthetic-to-real height estimation. This is the most directly relevant recent research to the PS's stated challenge, more so than GAMUS.

## 4. Relative → Metric → Absolute Elevation: Calibration Literature

These are three distinct transformations, evidenced separately below, exactly as the research brief required.

**Relative depth → metric depth**: the standard, well-established technique is least-squares scale-and-shift alignment in inverse-depth space, solving `(s, t) = argmin Σ(s·ẑ(u) + t − d(u))²` over pixels with known reference depth. This exact formulation, or a close variant, appears independently in [Wofk et al., Intel Labs (2023)](https://ar5iv.labs.arxiv.org/html/2303.12134), [a 2025 sparse-measurement adaptation paper](https://arxiv.org/html/2507.14879v1), [a 2025 underwater SPADE paper](https://arxiv.org/pdf/2510.25463), and [a robot-grasping alignment paper](https://arxiv.org/pdf/2506.17110) — it is a mature, reproducible technique, not a novel research question.

**Reference-point density, evidenced with real numbers**: Wofk et al. report successful metric alignment with as few as 150 sparse reference points, and up to a 30% reduction in error when a global alignment is followed by a learned dense (per-pixel) local refinement, versus global alignment alone. Multiple papers converge on the same qualitative finding, stated as an established result rather than a hypothesis: **a single global (s, t) leaves substantial spatially-varying residual error**. One paper visualizes this directly, showing the "ideal" per-pixel scale factor varies strongly across a scene even after global alignment succeeds. RANSAC-combined least-squares (sampling small point subsets, fitting, and keeping the best-agreeing model) is the standard robustness addition to reject outlier reference points.

**Metric depth → absolute elevation**: this step is not directly addressed in the general depth-alignment literature above, because those papers work in camera-relative coordinate frames (SLAM/robotics), not georeferenced world coordinates. It is, however, exactly the problem the remote-sensing MHE literature in Section 3 solves directly — those methods predict nDSM (height above local ground) or absolute elevation in a georeferenced raster from the start, sidestepping a separate metric-depth-to-elevation conversion step. This is an important structural difference: the general computer-vision calibration literature and the remote-sensing height-estimation literature solve adjacent but not identical problems, and DepthWizard's absolute-DSM path sits closer to the latter.

**Height-above-ground → absolute elevation**: this is the nDSM + DTM = DSM relationship (Section 2 of Phase 1). The GAMUS and HTC-DC Net pipelines both construct nDSM by subtracting a DTM from a LiDAR-derived DSM during dataset preparation — they build the terrain-relative quantity that the field's models actually predict, then would need a separate terrain elevation source (SRTM, GCPs) added back to reach absolute elevation above a vertical datum. **Open question, not resolved by any source found**: no paper in this research explicitly closes that final nDSM-plus-terrain-equals-absolute-DSM loop for a single monocular RGB input in the way DepthWizard's georeferenced path requires — the literature solves height-above-ground very well, and terrain elevation (SRTM/DEM) very well, but the fusion of the two into one certified absolute-DSM pipeline from a single image is comparatively undocumented.

## 5. DEM/SRTM Fusion & GCP Calibration Research

**What SRTM can realistically provide, with real numbers**: SRTM's own mission specification is a 90%-confidence absolute vertical error under 16 m and relative error under 10 m. Independent regional validations mostly find it performs *better* than that specification: [Iescheck & Scalco](https://ica-abs.copernicus.org/articles/1/136/2019/ica-abs-1-136-2019.pdf) found a mean absolute error of −0.28 m (range ±9.9 m) in a Brazilian basin; [a Himalaya/Peninsular-India GPS study](https://link.springer.com/10.1038/srep41672) found 1-arc C-band data has RMSE 23.53 m and 3-arc C-band RMSE 47.24 m, both markedly worse in steep terrain and needing correction before use — a directly relevant warning given the PS's "hilly" terrain requirement; a Brazil-wide 1,087-point study found SRTM errors are linearly related to slope, [with the largest errors consistently occurring in forest areas](https://scielo.figshare.com/articles/VERTICAL_ACCURACY_ASSESSMENT_OF_THE_PROCESSED_SRTM_DATA_FOR_THE_BRAZILIAN_TERRITORY/11314520) — the two terrain types the PS explicitly names as evaluation categories are the two documented to degrade SRTM's own accuracy most.

**What SRTM cannot provide**: it is C-band radar-derived and reflects from canopy tops and rooftops rather than bare earth in vegetated or built-up areas — it is itself a coarse, imperfect surface-like product, not a clean terrain reference, a point corroborated across every source above and already flagged in Phase 1.

**Alternative open DEMs, for contrast**: Copernicus GLO-30 and ASTER GDEM2 are the most commonly cited alternatives; one Saharan/Algerian comparison found [SRTM's vertical accuracy (3.6–9.8 m RMSE across two sites) generally outperforming ASTER GDEM2 and GMTED2010](https://api.crossref.org/works/10.3390%2FRS6054600) at the same sites, though results vary by region and none of these sources were checked specifically over Indian terrain.

**GCP-specific literature**: no paper was found addressing GCP-based calibration specifically for *converting monocular relative depth to absolute elevation* in the way the PS envisions. The closest evidenced analogue remains the general scale-shift alignment literature in Section 4, where Wofk et al.'s 150-point result is the most concrete number found for how many reference points a comparable alignment task needs. **Explicitly unresolved**: whether that number transfers to this problem's spatial scale and error tolerance is an open question — stated as such rather than assumed.

## 6. Remote-Sensing Dataset Landscape

Only datasets with genuine height/elevation ground truth are scored as directly relevant. Building-footprint-only datasets (SpaceNet, WHU, Open Cities AI) are listed for contrast — they supply semantic priors for the model, not elevation ground truth, per the research brief's own exclusion rule.

| Dataset | Imagery | GSD | Coverage | Ground truth | Terrain diversity | License | SIH26175 relevance |
| --- | --- | --- | --- | --- | --- | --- | --- |
| [DFC19 / US3D](https://ieee-dataport.org/open-access/data-fusion-contest-2019-dfc2019) | WorldView-3 pan+VNIR | 1.3 m | Jacksonville FL, Omaha NE (\~100 km²) | LiDAR nDSM, semantic | Urban/suburban only | Open (IEEE DataPort) | **High** — the most-used real satellite MHE benchmark |
| [GAMUS](https://arxiv.org/pdf/2305.14914) | Aerial orthophoto | Not stated (1024×1024 tiles) | Oklahoma, Washington D.C., Philadelphia, Jacksonville, NYC (5 US cities) | LiDAR-derived nDSM, 6-class semantic | **Urban only — no hilly/forested** | CC BY 4.0 | **High** — organiser's own recommended dataset, but terrain-limited |
| [ISPRS Vaihingen](https://www.isprs.org/education/benchmarks/UrbanSemLab/2d-semlabel-vaihingen.aspx) | Aerial IR-R-G | 0.09 m | Vaihingen, Germany (33 tiles) | LiDAR nDSM, semantic | Small-town, low-rise urban | Open (ISPRS) | **Medium-High** — finest-GSD real benchmark, best reported accuracy |
| [SynRS3D](https://arxiv.org/abs/2406.18151) | Synthetic optical | 0.05–1 m | Six synthetic city styles, global-diverse | Precise synthetic height, land cover, change masks | **Widest built-in diversity of any dataset found** | Open (GitHub/HF/Zenodo) | **High** — explicitly built to study the domain-transfer gap this PS names |
| GBH | PLANET satellite | 3 m | 19 training cities + 3 held-out (LA, São Paulo, Guangzhou) | LiDAR-derived nDSM, building footprints | Urban, cross-city generalization test | Not fully public at time of search | **High** — the only dataset found with an explicit unseen-city generalization test |
| SpaceNet 1/2 | WorldView-2/3 | 0.3–0.5 m | Rio, Vegas, Paris, Shanghai, Khartoum | Building footprints only, **no height** | Diverse cities, no hilly/forest | Open (AWS) | **Low for height** — useful only for semantic/footprint priors |
| WHU Building | Aerial + multi-satellite | 0.075–2.5 m | Christchurch NZ + global satellite subset | Building footprints only, **no height** | Urban | Open | **Low for height** — same caveat |
| Open Cities AI | Aerial | 0.03–0.2 m | 10 African cities | Building footprints (OSM-derived), **no height** | Urban, geographically diverse | Open (Radiant MLHub) | **Low for height** — same caveat |

**Gap made explicit**: not one dataset found includes forested terrain with LiDAR-verified height ground truth at a resolution comparable to the urban sets, and none is located in or near India. Both are open questions for Phase 3/4 rather than settled facts.

## 7. ISRO/SAC Reference Repository Findings

Directly inspected at [github.com/IMG-PROCESS-SAC/SIH2026](https://github.com/IMG-PROCESS-SAC/SIH2026).

**VERIFIED FROM OFFICIAL REPOSITORY**

- The repository contains exactly one file: a `README.md`. No sample images, no scripts, no reference outputs, no code.
- It names GAMUS (Hugging Face: `earthflow/GAMUS`) as the recommended dataset, and states it should be used to "overcome the domain gap between natural egocentric imagery... and top-down remote sensing imagery," and to "train your model to handle structural variations across urban, sparse, hilly, and forested landscapes."
- It repeats, near-verbatim, the PS's own background, pipeline description, milestones, and evaluation table — it does not add technical detail beyond the official PS text, except the dataset recommendation.
- It explicitly permits using "any open-source dataset containing remote-sensing depth data" beyond GAMUS, provided it supports both relative-depth training and metric-scale calibration.
- It states that **final evaluation will use ISRO's own RGB-band optical satellite imagery** — confirmed directly from the repository text, not inferred.

**NOT FOUND / UNKNOWN**

- No confirmation of which specific ISRO satellite or sensor will supply evaluation imagery, its GSD, or its likely geographic coverage.
- No sample or reference output of any kind — no example rDSM, absolute DSM, or 3D scene showing what a compliant deliverable looks like.
- No confirmation that GAMUS's US-only coverage (Section 6) is considered adequate by the organiser for the hilly/forested requirement, despite the README's own claim that it will teach exactly that generalization — this is a **direct tension** between what GAMUS's published content actually contains (Section 6: urban-only, five U.S. cities) and what the SAC README claims it teaches. This is flagged as a contradiction, not resolved.
- No licensing statement for the reference repository's own text.
- No hints about the scoring rubric beyond what the official PS already states.

## 8. LiDAR/DSM Validation Methodology

Evidenced practice, drawn from how the datasets in Section 6 were themselves built and evaluated:

- **Ground-truth construction**: GAMUS and HTC-DC Net's GBH dataset both derive nDSM by rasterizing denoised LiDAR point clouds twice — once keeping all points (DSM) and once keeping only ground-classified points (DTM) — then subtracting. This is the field's standard ground-truth pipeline, not something DepthWizard would need to invent.
- **Resolution reconciliation**: every benchmark reviewed resamples LiDAR-derived rasters to match the optical imagery's pixel grid before computing pixel-wise metrics — comparison happens on a common grid, never at mismatched native resolutions.
- **Metric reporting practice, evidenced not assumed**: HTC-DC Net reports four RMSE variants — all-pixel, building-only, non-building-only, and building-instance-wise (median height per connected building footprint) — rather than one blended number, precisely because a single aggregate RMSE hides whether errors concentrate on the class of greatest interest (buildings). This directly supports Phase 1's Section 10 inference that per-terrain-type breakdowns, not one number, are the credible way to report DSM accuracy.
- **Masking practice**: building-wise metrics require a semantic mask (building footprint) to aggregate per-instance, meaning height-only ground truth is usually paired with a semantic layer for meaningful evaluation — GAMUS, DFC19, and ISPRS Vaihingen all ship one.
- **Cross-city protocol, evidenced with real numbers**: GBH's held-out-city test (Section 3) is the clearest documented real-world protocol for exactly the kind of transfer DepthWizard will face when evaluated on unseen ISRO imagery — train on available cities/regions, report accuracy separately on regions never seen in training, and expect it to be substantially worse.

**Open question, not resolved by any source found**: no paper addresses validating a DSM prediction against a *coarse* reference (SRTM, 30 m) rather than a fine LiDAR reference — all validation literature found assumes LiDAR-grade ground truth. How to credibly report RMSE/MAE/correlation when the only available Indian reference is SRTM-grade, rather than LiDAR-grade, is left open for Phase 4.

## 9. Remote-Sensing Domain Gap & Terrain/Building/Vegetation Challenges

Phase 1 (Section 6) reasoned about the domain gap from first principles. This section adds literature evidence for those same claims.

- **The height-value distribution itself is a documented, severe problem, not a minor detail**: HTC-DC Net's own data analysis found background (near-zero height) pixels make up 57% of all pixels in the GBH training set, while pixels at any single high-value bin number only about 10 — an extreme long-tailed distribution that causes ordinary regression networks to systematically underestimate tall buildings. This is evidenced, not inferred, and is a documented failure mode across nearly every MHE paper reviewed.
- **Cross-city domain shift is measured, not hypothesized**: the GBH held-out-city results (Section 3) show RMSE roughly doubling on cities never seen in training, and the paper's authors state training cities being "mostly located in Europe and North America" as the likely cause — directly relevant to evaluation against unseen ISRO imagery over India.
- **Vegetation is a documented confound, not a Phase-1 guess**: multiple MHE papers, including HTC-DC Net's qualitative results, show canopy areas producing blurred, low-confidence height predictions compared to building edges — consistent with Phase 1's reasoning that vegetation's irregular texture starves the model of reliable geometric cues.
- **Semantic and height tasks are documented as mutually reinforcing**: several papers ([Srivastava et al.](https://arxiv.org/pdf/2112.14985), and HTC-DC Net's own building-footprint-conditioned building-wise metric) treat semantic segmentation as an auxiliary task that measurably improves height accuracy — evidence, not speculation, for Phase 1's Section 14 point that semantic priors are a legitimate calibration and accuracy lever.
- **Instance-wise shadow-based methods pre-date deep learning and are explicitly documented as fragile**: [StyHighNet's related-work section](https://pmc.ncbi.nlm.nih.gov/articles/PMC8037440/) describes early shadow-length/sun-angle building-height methods, noting they rely on strong assumptions and struggle where shadows overlap objects — confirming Phase 1's reasoning about shadow-based cues as useful but limited, from a primary source rather than first-principles alone.

## 10. Geospatial / GeoTIFF Pipeline Research

This track is largely settled engineering practice rather than an open research question, so search effort here was lighter and targeted at confirming rather than discovering.

- **GDAL** is the de facto standard open-source geospatial raster engine underlying nearly every tool touched in this research — the `cesium-terrain-builder` tool (Section 11) is built directly on it, and it is the library every dataset-processing pipeline in Sections 3 and 6 implicitly assumes for reading/writing GeoTIFF, CRS, and geotransform metadata correctly.
- **Nodata handling, evidenced**: `cesium-terrain-builder`'s own documentation explicitly warns that it does *not* handle nodata values and that these must be filled by interpolation in a preprocessing step — direct confirmation, from a real tool's own docs, of the nodata risk Phase 1 flagged conceptually.
- **Reprojection cost is real and documented**: the same tool's documentation recommends the input raster already match the output CRS "in order to bypass the need to reproject the data," and states reprojection carries "an associated performance penalty" — evidence that CRS mismatches are not just a correctness risk but a measured performance one.
- **Vertical datum practice, evidenced from the SRTM literature (Section 5)**: SRTM heights are referenced to the WGS84 ellipsoid, as stated directly in a GCP-based SRTM accuracy study — any fusion with a reference DSM using an orthometric (geoid-based) vertical datum requires an explicit geoid-to-ellipsoid correction, or a systematic bias of tens of metres can be introduced silently, consistent with Phase 1's Section 8 warning.

**Not exhaustively researched**: a systematic survey of "common mistakes in generating derived DSM GeoTIFFs" as its own research literature was not found as a distinct body of work — the evidence above is drawn from adjacent tool documentation and the SRTM accuracy literature rather than a dedicated geospatial-engineering paper trail, and this gap is stated rather than filled with unsupported claims.

## 11. 3D Terrain Reconstruction & Visualization Platforms

No platform is selected here — findings only, describing what each is documented to support.

- **Cesium**: purpose-built for whole-globe terrain. Documented architecture subdivides heightmap terrain into a **quadtree with hierarchical level of detail (HLOD)**, plus out-of-core streaming, frustum and occlusion culling, and asynchronous tile requests — this is the most directly relevant prior art found for rendering a large elevation raster without loading it at full resolution in one pass. A companion tool, `cesium-terrain-builder`, converts an arbitrary GeoTIFF DEM into these quadtree terrain tiles directly via GDAL, with a configurable pixel-unit error threshold controlling the detail/performance tradeoff at each level — documented, working prior art for exactly the terrain-LOD problem Phase 1 (Section 9) flagged as unresolved.
- **Three.js / Babylon.js**: general-purpose WebGL engines, not terrain-specialized. No dedicated built-in quadtree-LOD terrain system was found documented for either at the level Cesium provides — a heightfield mesh built from a raster in these engines would need a hand-built or third-party LOD/tiling scheme, or would be limited to rasters small enough to mesh in one piece.
- **Unity / Unreal**: general-purpose game engines with mature terrain systems (heightmap import, LOD, texture splatting) intended for interactive first-person navigation — a natural fit for the PS's "seamless first-person navigation" language, at the cost of a heavier runtime and engine-specific packaging for standalone deployment.
- **WebGPU**: emerging successor to WebGL; not evaluated in depth here given limited time — flagged as an area a later phase could revisit if browser-based deployment is pursued.

**Not resolved by this research**: none of the sources reviewed directly benchmark maximum practical raster/mesh size for a *browser-based* (Three.js/Babylon.js) terrain viewer without a Cesium-style tiling layer — this remains an open, testable question for Phase 4/6 rather than a documented fact.

## 12. Existing Commercial Solutions & Open-Source Implementations

**Commercial landscape** — no ranking, findings only:

| Product | Input | Technique | Output | Accuracy (documented) | Note |
| --- | --- | --- | --- | --- | --- |
| Maxar Precision3D (formerly Vricon) | **Multiple** overlapping satellite images | Multi-view stereo, proprietary algorithm | DSM, DTM, true ortho, point cloud, 50 cm resolution | 3 m SE90 absolute (per [Esri partner page](https://esri.com/partners/maxar-a2T70000000TNOvEAO/precision3d-a2d5x000005kI0FAAU)); [<3 m LE90/CE90 at 5 m posting per an older spec sheet](https://apollomapping.com/blog/50-cm-vricon-digital-elevation-models-dems-now-available) | **Not single-view** — explicitly uses many images per site, a fundamentally different technique from what the PS asks for |

**No commercial product using genuinely single-view (monocular) height estimation at production quality was found in this research.** This is itself an important finding: the commercial state of the art solves the *same end-user problem* (elevation from optical satellite imagery) but via multi-view stereo, not monocular estimation — consistent with Phase 1's framing that single-view height estimation is offered as an *agile alternative* to more established, more accurate multi-view techniques, not as an already-solved substitute for them.

**Open-source implementations found and directly relevant** (not an exhaustive GitHub trawl — scope-limited per Section 1):

| Repository | Purpose | Status |
| --- | --- | --- |
| [zhu-xlab/HTC-DC-Net](https://github.com/zhu-xlab/HTC-DC-Net) | Code and trained models for the current best-documented MHE method (Section 3) | Published alongside a peer-reviewed paper |
| [zhu-xlab/tse-net](https://github.com/zhu-xlab/tse-net) | Semi-supervised MHE across three real/synthetic datasets | Published alongside a 2025 paper |
| [JTRNEO/SynRS3D](https://github.com/JTRNEO/SynRS3D) | The SynRS3D dataset generation pipeline and RS3DAda domain-adaptation baseline | NeurIPS 2024, actively maintained per its own changelog |
| [DepthAnything/Depth-Anything-V2](https://github.com/DepthAnything/Depth-Anything-V2) | General-purpose monocular depth backbone, Apache-2.0 | Widely used, actively maintained |
| [CesiumGS/cesium](https://github.com/CesiumGS/cesium) + `cesium-terrain-builder` | Terrain tiling and quadtree LOD rendering | Mature, widely used |

**Not systematically searched**: a broad GitHub sweep across all 21 tracks' keyword lists was not performed; the repositories above surfaced as companions to the papers already found in Sections 2–4, rather than from an independent GitHub-specific search pass.

## 13. Computational Requirements

**Model scale, directly from source repositories**: Depth Anything V2 ships four sizes — Small (24.8M params), Base (97.5M), Large (335.3M), Giant (1.3B) — letting scale be traded against speed and quality.

**Inference latency, third-party benchmarked (not vendor-reported)**: the same wildlife-domain benchmark cited in Section 2 reports per-image inference times on real hardware: [ZoeDepth 0.17 s, Depth Anything V2 0.22 s, Metric3D v2 0.56 s, Depth Pro 0.65 s](https://arxiv.org/html/2510.04723v1). These are single-image, not tiled-large-raster, timings — a full satellite scene processed as many tiles will scale roughly linearly with tile count on top of these per-tile numbers.

**VRAM, evidenced via a same-architecture-family proxy**: exact DA-V2 VRAM figures were not directly found, but [Video Depth Anything (built on the same DA-V2 architecture) reports 6.8–7.3 GB VRAM for its Small model and 23.6–26.7 GB for its Large model](https://github.com/ueoo/Video-Depth-Anything), FP16 vs FP32 respectively — a reasonable proxy for what DA-V2's own Small/Large image models require, though not a confirmed identical figure. This is labelled as a proxy, not a hard fact about DA-V2 itself.

**Raster-tiling practice, evidenced from real tools**: `cesium-terrain-builder`'s own documentation recommends matching input raster block size to output tile size (65×65 pixels for terrain tiles) specifically to avoid the performance penalty of a scanline-format large raster — concrete, tool-documented evidence that naive full-raster processing is a known performance trap, not a theoretical one.

**Not found**: no source gives a direct, hardware-labelled benchmark for end-to-end DSM inference plus mesh generation plus rendering as one measured pipeline — the numbers above are per-component, and their combined real-world latency for a full DepthWizard pipeline is not established by any source in this research.

## 14. Literature Map, Prior Art Search & Patents

**Literature map** (foundational → recent, by category, as evidenced above):

| Category | Foundational | Recent / SOTA found |
| --- | --- | --- |
| Monocular depth (general) | Eigen et al. 2014; MiDaS | Depth Anything V2, Metric3D v2, UniDepth |
| Metric depth recovery | Scale-shift least-squares (multiple independent uses) | Region-aware/local scale adaptation (2025) |
| Monocular height estimation (remote sensing) | IM2HEIGHT (2018), Amirkolaee & Arefi (2019) | HTC-DC Net (2023), HeightFormer (2023), TSE-Net (2025) |
| Datasets | ISPRS Vaihingen, DFC19/US3D | GAMUS (2023), SynRS3D (2024, NeurIPS) |
| Domain adaptation for RS height | — (not separately found as its own lineage) | RS3DAda (bundled with SynRS3D, 2024) |
| 3D terrain visualization | Cesium (quadtree terrain, \~2013-era architecture) | No major architectural successor found in this search |

**Prior art search**: multiple query formulations ("single-image DSM," "monocular building height estimation," "RGB remote sensing height estimation," and others) converge reliably on the same core literature already reported in Section 3 — the field appears to use fairly consistent terminology ("monocular height estimation" being the dominant term), rather than being scattered across disconnected naming conventions. No separate, differently-named body of work solving the same problem was discovered.

**Patents**: a limited search, as instructed, was performed rather than an exhaustive one. One directly relevant patent was found — ["Object height estimation from monocular images"](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/11361196) — covering height recovery from a monocular image via a CNN trained against a generated ground-truth layout. This confirms the underlying technique is patent-landscape-active, not purely academic, but a full freedom-to-operate analysis was not attempted and is out of scope for a hackathon research phase.

## 15. Master Research Matrix, Evidence Ledger & Contradiction Check

**Master research matrix** (selected rows; full detail is in the sections above)

| Area | Method/Project | Input | Output | Metric/Relative | Dataset | Main result | Open source |
| --- | --- | --- | --- | --- | --- | --- | --- |
| General depth | Depth Anything V2 | Single RGB | Relative or metric (fine-tuned) | Both variants exist | 595K synthetic + 62M pseudo-labeled | 10x faster than diffusion competitors | Yes |
| RS height | HTC-DC Net | Single RS image | nDSM | Metric (trained) | DFC19, GBH, ISPRS Vaihingen | Best published RMSE on all three; degrades 2–3x on unseen cities | Yes |
| RS height (domain gap) | SynRS3D + RS3DAda | Single RS image (synthetic train) | Height, land cover, change | Metric (trained) | 69,667 synthetic images, 6 city styles | Explicit sim-to-real UDA method for height | Yes |
| Scale recovery | Global scale-shift LS | Relative depth + sparse reference | Metric depth | — | Task-agnostic technique | Works from \~150 points; local refinement adds up to 30% further error reduction | Yes (multiple implementations) |
| Terrain source | SRTM | — | Coarse elevation | Absolute (WGS84 ellipsoid) | Global | 90% CI: <16 m absolute, worse on slope/forest | N/A (public data) |
| Commercial | Maxar Precision3D | Multi-view satellite | DSM/DTM/point cloud | Absolute | Proprietary archive | 3 m SE90 at 50 cm resolution | No |

**Evidence ledger** (selected key claims)

| Claim | Source | Type | Confidence |
| --- | --- | --- | --- |
| A single global scale/shift leaves spatially-varying residual error | [Wofk et al.](https://ar5iv.labs.arxiv.org/html/2303.12134); [VIMD](https://arxiv.org/html/2509.19713v3) | Primary papers | HIGH |
| HTC-DC Net RMSE numbers by dataset/terrain | [arXiv:2309.16486](https://arxiv.org/pdf/2309.16486) | Primary paper, tables reproduced directly | HIGH |
| GAMUS covers 5 U.S. cities only | [arXiv:2305.14914](https://arxiv.org/pdf/2305.14914) | Primary paper | HIGH |
| SRTM absolute accuracy \~16 m spec, often better in practice | Multiple independent regional validations | Primary papers, cross-corroborated | HIGH |
| No commercial single-view height product exists | Absence across searched sources | Negative finding | MEDIUM — absence of evidence, not exhaustive proof |
| DA-V2 VRAM figures for image (not video) models | Video-Depth-Anything repo, same architecture family | Proxy source | LOW-MEDIUM, labelled as such |

**Contradiction check**

1. **GAMUS's claimed vs. actual generalization scope**: the SAC README states GAMUS will teach the model to "handle structural variations across urban, sparse, hilly, and forested landscapes," but GAMUS's own paper describes five U.S. cities with no hilly or forested category. Both claims are sourced directly (Sections 6–7); the disagreement is not resolved here and is carried forward as an open question.
2. **SRTM accuracy claims vary by an order of magnitude across regions**: from RMSE \~1.3 m (Thailand, GNSS-corrected) to RMSE 47 m (Himalaya, uncorrected 3-arc C-band). Both are sourced primary studies; the disagreement reflects genuinely different terrain and correction methods, not an error in either source — reported as a real regional variance, not adjudicated toward one number.
3. **No benchmark comparison found puts general-purpose foundation depth models (Section 2) and remote-sensing-specific MHE models (Section 3) head-to-head on the same overhead-imagery test set** — the two literatures were not found to directly cross-reference each other's numbers, so no claim is made about which family would perform better on this PS's imagery.

## 16. What Is Established, Partially Solved, and Remains Difficult

**Established** (mature, reproducible, well-evidenced): monocular relative depth from natural imagery; global least-squares scale-shift recovery from sparse reference points; building nDSM ground-truth construction from LiDAR (DSM minus DTM); quadtree-based terrain LOD rendering; SRTM's approximate global accuracy envelope.

**Partially solved**: monocular height estimation on remote-sensing imagery within a training distribution (HTC-DC Net and peers achieve strong, published, reproducible accuracy on their own benchmark cities); the sim-to-real domain gap (SynRS3D and RS3DAda represent active, evidenced progress, not a finished solution).

**Remains difficult, evidenced not assumed**: cross-city/cross-region generalization (measured 2–3x accuracy degradation on unseen cities in the strongest published method); hilly and forested terrain specifically (independently confirmed by both the MHE dataset landscape, which has no such category, and the SRTM accuracy literature, which shows slope and forest as its two worst-performing conditions); fusing height-above-ground with terrain elevation into one certified absolute-DSM pipeline from a single image (no source found closes this loop end-to-end); validating against coarse (SRTM-grade) rather than LiDAR-grade reference data.

## TECHNICAL GAP MAP

**Gap 1 — Terrain-type generalization** PROBLEM: models trained on available benchmarks may not generalize to hilly or forested terrain, which the PS names explicitly. → WHAT EXISTING RESEARCH DOES: HTC-DC Net and peers report strong in-distribution accuracy on urban benchmarks; SynRS3D and RS3DAda directly target synthetic-to-real transfer. → WHAT IT STILL CANNOT RELIABLY DO: no dataset reviewed pairs LiDAR-grade height ground truth with hilly or forested terrain at benchmark quality; no method reviewed reports accuracy specifically on such terrain. → EVIDENCE: Section 6 (dataset table), Section 3 (GBH unseen-city results). → OPEN RESEARCH QUESTION: what is the best available proxy ground truth for hilly/forested terrain accuracy given this gap?

**Gap 2 — Relative-to-absolute fusion** PROBLEM: the PS asks for absolute DSM from a single georeferenced image; the two literatures that separately solve height-above-ground and terrain elevation were not found combined into one documented pipeline. → WHAT EXISTING RESEARCH DOES: MHE methods predict nDSM well; SRTM/GCP literature calibrates scale for camera-relative depth well. → WHAT IT STILL CANNOT RELIABLY DO: no source found demonstrates nDSM-plus-terrain-equals-certified-absolute-DSM from one monocular image, end to end. → EVIDENCE: Section 4. → OPEN RESEARCH QUESTION: is this fusion better solved as one combined model, or as two independently-validated stages?

**Gap 3 — Validation without LiDAR** PROBLEM: all validation methodology found assumes LiDAR-grade reference data, which is not confirmed available for Indian test regions. → WHAT EXISTING RESEARCH DOES: standard metrics (RMSE/MAE/correlation) against LiDAR are well-established. → WHAT IT STILL CANNOT RELIABLY DO: no documented protocol for reporting these same metrics credibly against a coarser (SRTM-grade) reference. → EVIDENCE: Section 8. → OPEN RESEARCH QUESTION: what confidence bound should be attached to an SRTM-referenced RMSE, given SRTM's own \~16 m uncertainty?

## PHASE 2 COMPLETE

**15 Most Important Research Findings**

1. [HTC-DC Net](https://arxiv.org/pdf/2309.16486) is the strongest documented published method for exactly this task, with real RMSE numbers across three real datasets.
2. The best published method still degrades roughly 2–3x in RMSE on cities unseen during training.
3. [GAMUS](https://arxiv.org/pdf/2305.14914), the organiser's recommended dataset, covers only five U.S. cities — no hilly or forested terrain.
4. The SAC README's claim that GAMUS teaches hilly/forested generalization contradicts GAMUS's own documented scope.
5. [SynRS3D](https://arxiv.org/abs/2406.18151) (NeurIPS 2024) is the most directly relevant recent work to the PS's cross-terrain generalization challenge.
6. Global scale-shift least-squares recovery is a mature, reproducible technique, evidenced from \~150 reference points in one primary source.
7. Multiple independent papers confirm a single global scale factor leaves real, measured spatial error — not a hypothesis.
8. SRTM's documented worst-case accuracy occurs specifically on steep slopes and in forests — the PS's two hardest named terrain types.
9. No source found closes the loop from single-image nDSM plus terrain elevation to one certified absolute DSM.
10. No commercial product was found doing genuinely single-view height estimation; the commercial state of the art (Maxar Precision3D) uses multi-view stereo.
11. Extreme long-tailed height distributions (57% near-zero pixels in one real dataset) are a documented, named failure mode across the field.
12. Cesium's quadtree HLOD architecture is real, working prior art for the large-raster terrain-rendering problem.
13. Semantic segmentation as an auxiliary task is documented to measurably improve height-estimation accuracy.
14. No validation methodology was found for scoring DSM accuracy against a coarse (SRTM-grade) rather than LiDAR-grade reference.
15. The general computer-vision depth-calibration literature and the remote-sensing height-estimation literature are two adjacent but not cross-referenced bodies of work.

**15 Most Important Unresolved Technical Questions**

1. Can GAMUS or SynRS3D be supplemented with real hilly/forested height ground truth, and from where?
2. What is the best available proxy ground truth for Indian terrain specifically, given no Indian LiDAR-grade dataset was found?
3. Does the 150-reference-point finding from camera-relative depth literature transfer to this problem's geographic scale?
4. Should relative-to-absolute calibration be one combined model or two independently-validated stages?
5. What confidence bound is defensible on an RMSE computed against SRTM-grade rather than LiDAR-grade reference?
6. Which specific ISRO sensor and GSD will supply final evaluation imagery?
7. Is a browser-based (non-Cesium) terrain viewer practically sufficient at full satellite-tile resolution?
8. Does HTC-DC Net's architecture, or a simpler baseline, represent the right complexity-to-time tradeoff for a hackathon build?
9. How should vegetation-contaminated height be separated from building height in a single forward pass?
10. What vertical-datum correction is needed between SRTM (WGS84 ellipsoid) and whatever reference DSM is ultimately used?
11. Can semantic segmentation be added as an auxiliary task within the hackathon's time budget?
12. Is there any accessible Indian building-height or LiDAR dataset not surfaced in this search?
13. What is the realistic end-to-end latency of inference plus mesh generation plus rendering, measured rather than estimated?
14. Should the non-georeferenced (rDSM) path share a backbone with the georeferenced (absolute DSM) path, or diverge?
15. What does "successful standalone deployment" mean operationally for a pipeline mixing Python inference and a 3D renderer?

**10 Most Relevant Datasets**: [GAMUS](https://arxiv.org/pdf/2305.14914) · [SynRS3D](https://arxiv.org/abs/2406.18151) · DFC19/US3D · ISPRS Vaihingen · GBH · SRTM 30m · Copernicus GLO-30 · ASTER GDEM2 · SpaceNet (semantic priors only) · Open Cities AI (semantic priors only)

**10 Most Relevant Papers**: [HTC-DC Net](https://arxiv.org/pdf/2309.16486) · [SynRS3D](https://arxiv.org/abs/2406.18151) · [GAMUS](https://arxiv.org/pdf/2305.14914) · [Depth Anything V2](https://arxiv.org/abs/2406.09414) · [Metric3D v2](https://arxiv.org/html/2404.15506v4) · [Monocular Visual-Inertial Depth Estimation (Wofk et al.)](https://ar5iv.labs.arxiv.org/html/2303.12134) · [IM2HEIGHT](https://arxiv.org/pdf/1802.10249) · [TSE-Net](https://arxiv.org/html/2511.13552v1) · [THE Benchmark](https://arxiv.org/pdf/2112.14985) · SRTM global validation (Rodriguez et al., ASPRS 2006)

**10 Most Relevant Open-Source Projects**: [zhu-xlab/HTC-DC-Net](https://github.com/zhu-xlab/HTC-DC-Net) · [zhu-xlab/tse-net](https://github.com/zhu-xlab/tse-net) · [JTRNEO/SynRS3D](https://github.com/JTRNEO/SynRS3D) · [DepthAnything/Depth-Anything-V2](https://github.com/DepthAnything/Depth-Anything-V2) · [CesiumGS/cesium](https://github.com/CesiumGS/cesium) · cesium-terrain-builder · [IMG-PROCESS-SAC/SIH2026](https://github.com/IMG-PROCESS-SAC/SIH2026) (organiser repo) · GAMUS Hugging Face repo · SynRS3D Hugging Face/Zenodo mirrors · Video-Depth-Anything (VRAM proxy source)

**10 Most Important Technical Limitations**: extreme long-tailed height distributions · 2–7x cross-city accuracy degradation · no hilly/forested LiDAR-grade dataset found · single global scale factor leaves spatial residual error · SRTM worst on slope and forest · vegetation contaminates height signal for underlying structures · nDSM-to-absolute-DSM fusion undocumented end-to-end · no coarse-reference validation protocol found · large-raster inference/VRAM limits require tiling · no cross-benchmark comparison between general depth models and RS-specific height models

**What Phase 3 Must Investigate** Existing-solutions and gap analysis should now (a) determine whether HTC-DC Net's published code can be adapted rather than reimplemented from scratch, given the time budget; (b) resolve the GAMUS-scope contradiction by searching specifically for any hilly/forested height dataset not surfaced here; (c) examine SynRS3D's actual synthetic imagery closely enough to judge its visual realism and usability as training data; and (d) determine what, if anything, is publicly known about ISRO's own satellite sensors' GSD and spectral characteristics, to narrow the domain-gap question from "generic satellite imagery" to the specific sensor family likely used at evaluation.

Do not proceed into Phase 3 until explicitly asked.
