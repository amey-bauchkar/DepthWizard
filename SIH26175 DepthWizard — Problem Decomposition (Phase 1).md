# SIH26175 DepthWizard — Problem Understanding & Requirement Decomposition

2026-09-18 · @Someone

Phase 1 only: decomposition and requirement analysis. No architecture, model, or stack selection below.

## 1. Executive Interpretation

The PS asks for a single-image-to-navigable-terrain pipeline: ingest one RGB image, produce a height/elevation raster, and render that raster as an explorable 3D scene — deployable without a live backend.

**A. Explicit requirements**

- Accept PNG/JPG (non-georeferenced) and GeoTIFF (georeferenced) as input.
- Non-georeferenced path outputs a relative DSM (rDSM) — no metric claim required.
- Georeferenced path outputs an absolute DSM with metric heights, calibrated via SRTM or GCPs.
- Use a pre-trained monocular depth backbone for initial extraction.
- Project the RGB image onto a 3D terrain mesh, rendered with a first-person, freely navigable camera.
- Package as a standalone deployable application.
- Validate the DSM against LiDAR/reference data using RMSE, MAE, and correlation, tested across urban, sparse, hilly, and forested terrain.
- Deliver full source code and technical documentation.

**B. Implicit expectations** *(reasonable inference)*

- A working end-to-end pipeline, not a research notebook: "software stability" and "standalone deployment" imply production-shaped software.
- Self-supplied validation data. The PS names no official test set beyond pointing to a reference dataset, so satisfying the 50% accuracy criterion requires the team to obtain or construct its own reference comparison.
- Correct geospatial metadata handling (CRS, geotransform) is implied by "GeoTIFF" and "standard geospatial format" even though no specific file format or standard is named.

**C. What would NOT satisfy the problem**

- A stereo or multi-view pipeline — the PS specifies single-view.
- A viewer that only visualizes a manually supplied DEM, without deriving elevation from RGB via the model.
- Presenting uncalibrated relative depth as absolute elevation for georeferenced input.
- Either half alone: a validated DSM with no 3D viewer, or a polished viewer with no accuracy measurement, given the 50/50 evaluation split.
- A calibration hardcoded to one test image rather than a generalizable module.

**D. Likely real-world operational objective** *(reasonable inference, not stated)* Provisional elevation/terrain products from a single optical pass, for use when stereo pairs, LiDAR, or InSAR coverage is unavailable or too slow to acquire — plausible in disaster response or reconnaissance contexts. This is inferred from the background's framing (stereo/LiDAR/InSAR as "cost-prohibitive... dependent on sensor availability") and the Disaster Management theme tag, not stated directly as the objective.

## 2. The Core Technical Problem

The PS reduces to two coupled but distinct subproblems: (1) monocular relative depth estimation from RGB — well studied, produces depth known only up to an unknown scale and shift — and (2) relative-to-absolute calibration, converting that scale-ambiguous depth into elevation tied to a real vertical datum.

**Why these are not equivalent** *(established technical fact)*: a single image is geometrically consistent with infinitely many 3D scenes that differ by an unknown similarity transform (the classical scale ambiguity of monocular vision). Depth networks are typically trained with scale-invariant losses precisely because ground-truth scale varies across training sensors and enforcing an absolute prediction under this ambiguity is ill-posed. The learned output therefore relates to true depth by an unknown affine transform (predicted ≈ a·true + b), with a and b unresolved until fixed by an external absolute reference. Elevation, in contrast, is a specific physical quantity referenced to a defined datum and must hold consistently across the whole scene, not just be locally ordinally correct.

**Precise terminology**

- **Depth**: distance from a specific sensor position to a surface point along the viewing ray. Viewpoint-dependent.
- **Relative depth**: depth known only up to an unknown, often per-image, affine transform — correct ordering, no fixed unit or origin.
- **Height** (as used in this PS): vertical distance of a surface point above a reference surface, independent of viewpoint.
- **Elevation**: height above a defined vertical datum (e.g. an ellipsoid or a geoid) — an absolute geospatial quantity.
- **DEM** (Digital Elevation Model): bare-earth terrain elevation — ground only, excluding buildings and vegetation.
- **DSM** (Digital Surface Model): elevation of the highest reflective surface at each location — ground plus buildings, canopy, and any object on it.
- **DTM** (Digital Terrain Model): commonly synonymous with DEM; a bare-earth model, sometimes including terrain breaklines.
- **nDSM**: DSM minus DEM — the height of objects above local ground. Much of the CV literature this problem draws on actually predicts nDSM-like quantities rather than raw DSM.
- **rDSM** (as this PS defines it): a DSM-shaped output derived from relative depth, structurally correct but without absolute-scale calibration — no verified metric elevation.
- **Absolute DSM**: a DSM where each pixel expresses a real elevation value in metres against a stated vertical datum.

**Central bottlenecks**: (a) a depth backbone trained on a different visual domain, (b) an unresolved scale/shift ambiguity standing between relative depth and absolute output, (c) scarcity of reference data to compute the PS's own accuracy metrics, and (d) rendering a real-time 3D scene from per-pixel estimates that are inherently noisy, without visible geometric artifacts.

## 3. Input → Process → Output Decomposition

Conceptual stages only; no technology selected.

**Path A — Non-georeferenced PNG/JPG**

| Stage | Input | Transformation | Output | Purpose | Dependencies | Risks |
| --- | --- | --- | --- | --- | --- | --- |
| Ingestion | Raw PNG/JPG | Validate format, decode pixels | Decoded RGB array | Entry point | None | Unknown/absent metadata must not be assumed present |
| Relative depth inference | RGB array | Monocular depth backbone forward pass | Per-pixel relative depth map | Extract geometric structure | Pre-trained backbone | Domain gap (Section 6) degrades quality silently |
| rDSM construction | Relative depth map | Reshape/normalize into a DSM-like raster | rDSM (structurally correct, no metric claim) | Deliverable elevation product for this path | None beyond depth output | Presenting this as if it were metric would violate PS intent |
| Mesh generation | rDSM | Build heightfield/terrain geometry | 3D mesh | Base geometry for visualization | rDSM | Noisy depth propagates directly into mesh artifacts |
| Texture projection | Original RGB + mesh | Map image pixels onto mesh surface | Textured mesh | Visual realism | Mesh, RGB | Misalignment causes visible "sliding" texture |
| Interactive rendering | Textured mesh | Real-time render loop, camera control | Navigable 3D scene | Satisfies visualization criterion | Mesh, texture | Performance limits at full image resolution |
| Validation | rDSM | Comparison against any available reference (best-effort, since no absolute claim exists) | Qualitative/relative accuracy notes | Partial evidence for accuracy criterion | External reference data | Relative output cannot be scored the same way as absolute DSM |

**Path B — Georeferenced GeoTIFF**

| Stage | Input | Transformation | Output | Purpose | Dependencies | Risks |
| --- | --- | --- | --- | --- | --- | --- |
| Ingestion | GeoTIFF | Parse pixels plus CRS, geotransform, bounds | RGB array + geospatial metadata | Entry point, preserves georeferencing | Metadata must be intact | Malformed or missing metadata silently mishandled |
| Relative depth inference | RGB array | Monocular depth backbone forward pass | Per-pixel relative depth map | Extract geometric structure | Pre-trained backbone | Same domain gap as Path A |
| Scale calibration | Relative depth map + geotransform + SRTM/GCPs | Solve for scale/shift or region-wise correction (Section 5) | Depth map mapped toward metric units | Bridge relative depth to absolute elevation | External DEM or GCP source, correct vertical datum | Wrong or insufficient reference points bias the whole output |
| Absolute DSM construction | Calibrated depth | Assemble as a georeferenced elevation raster | Absolute DSM with embedded CRS/geotransform/datum | Deliverable elevation product for this path | Calibration output | Losing or mismatching metadata invalidates downstream comparison |
| Mesh generation | Absolute DSM | Build heightfield/terrain geometry in real-world scale | 3D mesh | Base geometry for visualization | Absolute DSM | Same artifact risk as Path A, now at real-world scale |
| Texture projection | Original RGB + mesh | Map image pixels onto mesh surface | Textured mesh | Visual realism | Mesh, RGB | Same as Path A |
| Interactive rendering | Textured mesh | Real-time render loop, camera control | Navigable 3D scene, slope/height queries | Satisfies visualization criterion | Mesh, texture | Same as Path A, plus needing to expose real units |
| Validation | Absolute DSM | RMSE/MAE/correlation against LiDAR or reference DSM | Quantitative accuracy metrics | Satisfies the 50% accuracy criterion | A reference dataset the team must source | No official test set is provided (Section 1B) |

## 4. Non-Georeferenced vs Georeferenced Mode

| Aspect | PNG/JPG (non-georeferenced) | GeoTIFF (georeferenced) |
| --- | --- | --- |
| CRS present | No | Yes — declares the horizontal coordinate system |
| Geotransform (pixel→ground mapping) | No | Yes — origin, pixel size, rotation |
| Geographic location known | No | Yes — the scene's real-world position |
| Scale / ground sample distance known | No | Yes, derivable from the geotransform |
| Elevation reference (vertical datum) available | No, not inherently | Not automatically embedded either, but can be reconciled against an external source (SRTM, GCPs) because location is known |
| Absolute height feasible | No | Yes, conditional on external calibration data |
| What is reliably inferable | Ordinal depth structure only (what's relatively taller/closer) | Ordinal structure plus, after calibration, metric elevation tied to a real location |
| What is NOT inferable | Any real-world unit, scale, or geographic location | Metric elevation without external calibration — georeferencing alone supplies location and scale, not height |

Without a CRS, geotransform, and known ground sample distance, there is no principled way to convert pixel-space relative depth into metres — any such conversion would require an arbitrary, unverifiable assumption. This is why the PS restricts absolute-DSM output to the georeferenced case, and why the non-georeferenced path is only ever asked to produce a relative surface.

## 5. Relative Depth → Absolute Height Problem

**Why models produce relative depth** *(established technical fact)*: monocular depth networks are commonly trained with scale-invariant loss formulations, because ground-truth depth scale varies by sensor and training set, and directly enforcing absolute scale under monocular geometric ambiguity is ill-posed — a single image is consistent with infinitely many 3D scenes differing by an unknown similarity transform. The practical consequence is a predicted depth related to true depth by an unknown, often per-image, affine transform: predicted ≈ a·true + b, with scale (a) and shift (b) unresolved until fixed against an external absolute reference.

**The calibration problem restated**: given a relative depth map with unknown (a, b), and access to some absolute reference (SRTM, GCPs), solve for a transform — global or local — that maps relative depth into elevation consistent with that reference, then apply it across the full scene.

**Classes of calibration method** *(conceptual survey — no method selected)*

- **Global affine fit against sparse absolute references**: solve for (a, b) by least-squares between predicted depth and known elevation at reference locations (GCPs, or SRTM cells), then apply the same (a, b) everywhere.
- **Low-resolution DEM as a coarse anchor surface**: align the relative depth map's large-scale statistics (mean, range, low-frequency structure) to SRTM's coarse terrain trend, retaining the model's finer local structure. Caveat *(established technical fact)*: SRTM is itself a radar-derived surface that reflects off canopy and rooftops, not strictly bare earth — it is already an imperfect, coarse DSM-like proxy at roughly 30 m resolution, and cannot resolve building-scale detail.
- **Scene-level statistical priors**: heuristics using typical real-world extents implied by the imagery — for example inferring scale from a known road width, or inferring height from shadow length combined with sun angle.
- **Semantic priors**: recognizing object classes (buildings, roads, trees) and anchoring scale locally using typical or known height ranges per class, rather than one global scale factor.
- **Known object heights**: identifying one or more reference structures of known height, used as single- or multi-point calibration anchors.

**Open tension** *(open question)*: a single global (a, b) may be wrong locally, since scale is rarely spatially uniform across real terrain — particularly over hilly relief, where depth compression varies with distance and slope. Per-region calibration would correct for this but requires more reference points than may realistically be available from sparse GCPs or coarse SRTM cells. How this tension resolves is left to Phase 2/5, not decided here.

## 6. Remote-Sensing Domain Gap

Foundational monocular depth models (the class the PS calls for) are trained largely on street scenes, indoor rooms, and driving datasets: near-horizontal viewpoints, close range, strong perspective cues (vanishing lines, a visible horizon, relative object size along a ground plane), human- and vehicle-scale objects.

Nadir and near-nadir remote-sensing imagery differs on several axes at once:

| Axis | Training-domain imagery | Remote-sensing imagery |
| --- | --- | --- |
| Viewpoint | Near-horizontal, egocentric | Near-vertical (nadir) — removes standard perspective depth cues |
| Scale | Close range, fine detail | Coarse ground sample distance; a small pixel shift implies a large real-world distance |
| Texture / spectral | Consumer RGB camera statistics | Satellite sensor statistics, atmospheric haze, different contrast profile |
| Dominant objects | Facades, street furniture, pedestrians | Roof polygons, roads, tree canopy |
| Building geometry cue | Visible vertical facades | Height inferable mainly from shadow and slight off-nadir parallax, not facades |
| Shadows | Minor cue | One of the only strong monocular height cues from nadir — shadow length plus sun angle behaves like a sundial |
| Occlusion pattern | Street-level occlusion by nearer objects | Canopy over structures, dense urban canyons, overhangs |
| Terrain appearance | Rare/absent | Kilometre-scale relief structure, a spatial frequency largely unseen in training |

**Consequence** *(reasonable inference)*: zero-shot relative depth from a natural-image backbone likely preserves coarse ordinal correctness — taller objects still tend to read as "higher" — but is unreliable at fine structural detail, and its inherent scale/shift carries essentially no relationship to real-world units in this new domain. This reinforces why calibration (Section 5) cannot be skipped, and why the PS's explicit requirement to remain stable across urban, sparse, hilly, and forested terrain (Section 10) is a genuine test of generalization rather than a formality.

## 7. DSM-Specific Challenges

| Surface class | Why it's difficult |
| --- | --- |
| Buildings | Sharp height discontinuities at edges; roof shapes (flat, pitched, complex) create ambiguous within-footprint height gradients; small structures near the backbone's effective resolution limit risk being smoothed away |
| Trees / vegetation canopy | Irregular, semi-transparent-looking texture; height-to-radius ratio varies by species and season; canopy occludes true ground or building height beneath it, contaminating the signal for whatever lies underneath |
| Roads | Expected near-zero height above local ground, but must be distinguished from bare earth elsewhere; homogeneous low-texture surfaces give the depth backbone little reliable signal |
| Bare earth | Low-texture regions (fields, deserts) are hard to differentiate: brightness and texture cues a learned model implicitly relies on may not correlate with actual elevation gradient |
| Bridges | Height should reflect the deck, not the ground beneath it; a common failure mode is modelling the bridge as if it were at grade |
| Shadows | Strong cast shadows can be misread as low-albedo ground rather than an occlusion artifact, contaminating both semantic and depth cues |
| Roofs | Material variation (metal, shingle, green roof) introduces spectral variance decoupled from true height, risking depth predictions driven by texture rather than geometry |
| Repeated structures | Dense uniform housing or industrial rooftops give the network few distinguishing local cues, risking a spuriously flattened, uniform output |
| Occlusions / image borders | Tile-edge receptive-field truncation degrades quality; systematic edge artefacts are a common failure in tiled inference pipelines |
| Water | Near-zero elevation but visually similar in texture-poor scenes to shadow or bare ground; specular reflections can produce spurious holes or spikes |
| Homogeneous surfaces | Parking lots, large rooftops, playing fields — low texture reduces reliable monocular cues of any kind |
| Steep terrain | Illumination-dependent shading from relief can be conflated with material or texture differences by the model |

## 8. Georeferenced Imagery Technical Requirements

What a georeferenced processing pipeline must preserve, conceptually:

- **CRS**: every georeferenced raster carries (or must be assigned) a coordinate reference system mapping pixel coordinates to real-world coordinates. Any output claiming absolute height must retain and declare a consistent horizontal CRS.
- **Affine transform (geotransform)**: the mapping from pixel row/column to CRS coordinates — origin, pixel size, rotation or skew. Required to know real-world ground sample distance per pixel and to project outputs back to real coordinates.
- **Pixel resolution**: ground sample distance per pixel, in CRS units. Determines what a one-pixel error physically represents, and is needed to relate pixel-space depth gradients to real slope.
- **Bounds / extent**: the geographic footprint the raster covers — needed to align with external sources such as SRTM tiles or a reference LiDAR DSM for validation.
- **Raster dimensions**: width and height in pixels, tied to bounds and resolution.
- **Nodata**: invalid pixels must be explicitly flagged rather than silently defaulting to zero, since zero could be misread as sea-level or otherwise valid elevation.
- **Vertical reference (datum)**: elevation values need a defined vertical datum — for example ellipsoidal height versus orthometric/geoid height. SRTM, GCP surveys, and any reference LiDAR dataset may each use a different vertical datum; mixing them without reconciliation introduces systematic bias.
- **Output geospatial metadata**: an elevation product intended to be compared against LiDAR or SRTM must carry CRS, geotransform, and a stated vertical datum. Without these, "RMSE against reference" is not well-defined, because it isn't certain the same physical location or the same vertical reference is being compared.

"Standard geospatial format," as the PS uses the phrase, conceptually means a raster format that embeds this metadata so any GIS tool can correctly interpret, reproject, and overlay the elevation values without needing out-of-band information supplied separately.

## 9. 3D Visualization Requirement

The chain: 2D RGB image + elevation map → terrain mesh → textured 3D scene → interactive navigation.

- **Terrain mesh**: a geometric surface, typically a heightfield or triangulated grid, whose vertex heights come from the elevation map. A mesh coarser than the raster loses detail; a 1:1 mesh at full satellite resolution may be far too dense for real-time rendering — this tradeoff is not resolved here.
- **Texture projection**: mapping the original RGB pixels onto mesh UV coordinates so visual features (roads, roofs) align with their corresponding geometry. Misalignment produces visibly "sliding" texture over the mesh.
- **Height exaggeration**: real terrain relief is often visually subtle relative to horizontal extent, so some vertical exaggeration is typically needed for a compelling flythrough. This must be an exposed display parameter, not conflated with the actual measured elevation used for the accuracy criterion.
- **Camera / first-person navigation**: requires the camera to follow or collide with the mesh so movement feels grounded rather than clipping through terrain.
- **Arbitrary aerial perspectives**: the camera is not confined to a scripted flythrough path but can be freely positioned, which raises performance requirements across the whole scene rather than only along a pre-baked path.
- **Slope analysis**: computing local slope (gradient magnitude and direction) from the elevation surface, as a derived analytic layer distinct from raw height.
- **Structural height analysis**: extracting height-above-ground for a selected feature. If the underlying data is a DSM rather than an nDSM, this requires isolating a local ground reference to compute the visible feature's height above it.
- **Interaction**: some mechanism for the user to select a point or region and read back numeric height or slope values, tying the visualization back to the quantitative elevation data.
- **Performance**: real-time frame rates for a "seamless," "navigable" experience across whatever imagery size is supplied — raising open questions of level of detail, tiling, or mesh simplification not yet specified by the PS.
- **Standalone deployment**: implies the interactive session must run without a live server dependency, though preprocessing could plausibly happen beforehand — left as an open question for a later phase.

**Correct DSM vs. convincing flythrough**: these are independent axes of merit, which the 50/50 evaluation split treats as such. A technically correct DSM is judged by numeric agreement with reference data at the raster level. A convincing flythrough is judged by subjective and interaction qualities — fidelity, smoothness, intuitiveness — that can look impressive even over a geometrically imperfect DSM (via exaggeration, smoothing, or texture tricks), and conversely a numerically accurate DSM can still render as an unconvincing scene if the rendering pipeline is naive.

## 10. Evaluation Criteria — Reverse Engineering

**DSM Estimation — Accuracy and Validation (50%)**

| Metric | What it technically measures |
| --- | --- |
| RMSE | Penalizes larger errors disproportionately; sensitive to outliers — one badly predicted tall building can dominate RMSE over an entire tile |
| MAE | A more robust, less outlier-sensitive companion; comparing it against RMSE reveals whether error is broadly distributed or dominated by a few large failures |
| Correlation | Whether the predicted surface co-varies spatially with the reference, independent of absolute bias or scale. High correlation with poor RMSE points to a scale/offset calibration problem (Section 5), not a structural modeling failure |
| Stability across urban / sparse / hilly / forested | The PS expects consistent, not merely best-case, performance; a model tuned only on urban scenes is directly exposed by this requirement |

Evidence a team would need: per-terrain-type quantitative tables, not a single aggregate number; an explanation of how the reference "ground truth" was obtained, since the PS supplies none directly; ideally an error map rather than only scalar summary statistics.

**Visualization — Rendering Quality and User Experience (50%)**

| Criterion | What it technically measures |
| --- | --- |
| Projection accuracy | Geometric alignment between textured imagery and the derived elevation surface — evaluable by visual inspection for misregistration |
| Visual fidelity | General rendering quality: texture resolution, lighting, absence of obvious mesh or texture defects |
| Navigability | Whether first-person movement feels controllable, free of clipping or broken collision |
| Interface intuitiveness | Whether a first-time user can upload imagery and explore without instructions |
| Software stability | Absence of crashes across varied input sizes and types over a demo session |
| Standalone deployment | A runnable packaged application, not only a locally run dev script or a page requiring evaluators to stand up a backend themselves |

Evidence needed: a live, repeatable demo across at least the imagery types the PS names (PNG/JPG and GeoTIFF), and ideally a packaged installable artifact rather than source code alone.

## 11. Hidden Failure Modes

**A. ML failure modes**

- Uncorrected scale ambiguity presented as if it were metric.
- Unrealistic terrain: implausible slopes or discontinuities from noisy per-pixel predictions.
- Domain shift causing systematically biased predictions on unfamiliar terrain types.
- Scale calibration overfit to the specific reference points used, not generalizing beyond them.
- Poor generalization across the four named terrain types, masked by only demonstrating one.

**B. Geospatial failure modes**

- CRS mismatch: comparing input imagery and reference data in different projections without reprojecting first.
- Vertical datum mismatch: comparing ellipsoidal to orthometric heights without correction, which can introduce tens of metres of systematic bias in some regions.
- Invalid or missing georeferencing metadata silently defaulting to an incorrect assumption rather than failing visibly.
- Coordinate precision loss, for example float32 rounding at large UTM coordinate magnitudes.
- Output metadata corruption: writing a raster without embedding correct CRS/geotransform, making the file geometrically meaningless to downstream GIS tools despite looking correct visually.

**C. Visualization failure modes**

- Mesh artifacts (spikes, holes) where noisy depth propagates directly into geometry.
- Texture stretching at steep slopes or geometric discontinuities.
- Seams or holes at image or raster-tile borders.
- Terrain popping or visible level-of-detail transitions during navigation.
- Vertical exaggeration presented to the user as if it were true, unscaled elevation.

**D. Software / deployment failure modes**

- GPU or memory limits when processing full-resolution large rasters, which can be far larger than the image sizes typical depth-inference pipelines expect.
- Inference latency incompatible with a "seamless" live demo.
- Inconsistent tiling and stitching of large rasters, producing visible seams at tile boundaries.
- Standalone-packaging failures on the evaluator's machine due to missing runtime dependencies.

## 12. What the Evaluators Are Probably Looking For

**Explicitly stated**: numeric accuracy against reference data across four terrain types; correct handling of both non-georeferenced and georeferenced input; a real interactive 3D deployment.

**Strongly implied engineering expectations** *(reasonable inference)*: correct, consistent geospatial metadata handling — since "GeoTIFF" and "standard geospatial format" are named explicitly, sloppy CRS or datum handling is plausibly noticeable to domain-literate evaluators, particularly from an ISRO panel; some explanation of how absolute scale is derived, since the background text singles this out as "a critical challenge"; genuine handling of a range of terrain types rather than one showcase example.

**Areas where a convincing demonstration would matter** *(reasonable inference)*: side-by-side numeric error breakdowns per terrain type rather than one blended figure; visible acknowledgment of known hard cases (forested areas, steep terrain) rather than only showing best-case results; a live, responsive flythrough rather than a pre-rendered video, given the PS's repeated use of "interactive" and "navigable."

## 13. Minimum Viable Compliance

The smallest conceptual system satisfying every explicit requirement — a compliance baseline, not the final solution.

- **Minimum input support**: accept PNG/JPG and GeoTIFF, even via two separate code paths.
- **Minimum elevation output**: a per-pixel height raster — relative for PNG/JPG; nominally scale-adjusted, even via one crude global calibration, for GeoTIFF.
- **Minimum calibration capability**: at least one working method from Section 5 (for example, a single global affine fit against a handful of SRTM points), applied consistently. Presence and consistency matter here, not sophistication.
- **Minimum visualization**: a 3D scene projecting the RGB texture onto a heightfield mesh derived from the elevation output, with basic free camera movement — polish is not required at this baseline.
- **Minimum validation**: some quantitative comparison (RMSE, MAE, or correlation) against at least one reference source, even for a subset of terrain types — the accuracy criterion cannot be satisfied with zero measurement.
- **Minimum deployment**: a runnable packaged build, even a simple executable or self-contained web bundle, rather than source code that requires manual environment setup.

## 14. Stronger-Than-Minimum Capability Areas

Categories where additional technical capability could differentiate a solution — not a design, not a feature list.

- **Better scale calibration**: methods that adapt per-region rather than applying one global factor, or that fuse multiple reference cues (DEM plus semantic priors) instead of one.
- **Stronger remote-sensing generalization**: demonstrated stability specifically across all four named terrain categories, genuinely addressing the domain gap in Section 6 rather than showcasing one favorable category.
- **Uncertainty estimation**: surfacing where the model is confident versus uncertain — for example a per-pixel confidence layer — as an explicit mitigation for the accuracy criterion and an honest acknowledgment of the failure regions in Section 11, rather than presenting all output with equal implied confidence.
- **Validation depth**: comparing against more than one reference source, or reporting metrics broken out by terrain category rather than one blended number.
- **Geospatial robustness**: correctly handling and being explicit about vertical-datum reconciliation, nodata, and varying input resolutions, rather than assuming a single well-behaved case.
- **Interactive analysis**: exposing the slope and height-query interactivity named in Section 9 as an analytic tool rather than a purely visual showpiece — this ties the visualization half of the score back to the accuracy half instead of treating them as unrelated deliverables.

## 15. Research Questions We Must Answer Next

**A. Monocular depth estimation** — What are the leading pre-trained backbones (architecture, training data, licensing) most likely to generalize? What loss functions and training regimes explain their scale/shift behavior? What is publicly documented about their zero-shot performance outside their training domain?

**B. Remote-sensing depth estimation** — What prior work targets monocular height or depth estimation from overhead imagery specifically? What accuracy has been reported, and on which terrain types?

**C. Single-image DSM generation** — What published methods generate a full DSM, not just depth, from a single overhead image? How do they handle the bare-earth versus surface distinction?

**D. Relative-to-metric scale calibration** — What calibration methods are documented in prior remote-sensing depth work? What reference-point density do they typically require for acceptable accuracy?

**E. DEM/SRTM fusion** — What is SRTM's actual vertical accuracy and its documented canopy/building bias? What fusion techniques exist for combining a coarse global DEM with a fine local depth estimate?

**F. Ground Control Points** — What is a hackathon-feasible GCP acquisition method? Can GCPs be derived automatically from a public reference layer, or do they require manual identification?

**G. Remote-sensing datasets** — What open datasets pair single-view RGB imagery with reference height/DSM data? The organiser's own linked repository recommends [GAMUS](https://huggingface.co/datasets/earthflow/GAMUS) — Phase 2 should determine its coverage, resolution, terrain-type diversity, and licence.

**H. LiDAR/reference DSM datasets** — What openly available LiDAR-derived DSMs exist, and do any overlap geographically with a candidate RGB dataset, to permit genuine independent validation?

**I. Existing open-source implementations** — Are there open-source repositories implementing single-view satellite depth/height estimation end-to-end, not just research code for a metric?

**J. Existing research papers** — What does the current state of the art (recent survey or benchmark work) say about single-view building/terrain height estimation from optical remote sensing?

**K. Existing commercial solutions** — Do any commercial or governmental tools already offer single-view height estimation, and how do they publicly position accuracy and limitations?

**L. 3D terrain visualization** — What existing engines or libraries are commonly used for heightfield rendering with texture draping, and what are known performance limits for large raster inputs?

**M. GeoTIFF/geospatial processing** — What are standard practices for reading and writing GeoTIFF metadata (CRS, geotransform, nodata, vertical-datum tagging) that the pipeline must respect?

**N. Benchmarking** — What is a defensible way to construct our own held-out validation split, given the PS provides no official test set and the true evaluation imagery may be unseen ISRO data?

**O. Deployment** — What standalone-packaging approaches exist for a pipeline combining a Python/ML inference component with a real-time 3D rendering component?

## 16. Final Problem Statement Model

**INPUT**: a single RGB image, optionally georeferenced (GeoTIFF) or not (PNG/JPG)

→ **CORE COMPUTATIONAL CHALLENGE**: monocular relative depth/height estimation across a remote-sensing domain gap (Section 6)

→ **CALIBRATION CHALLENGE**: converting scale/shift-ambiguous relative depth into metric elevation via DEM, GCPs, or semantic priors — only when georeferencing exists (Section 5)

→ **GEOSPATIAL OUTPUT**: a CRS- and datum-correct raster surface — rDSM for non-georeferenced input, absolute DSM for georeferenced input — in a standard geospatial format (Section 8)

→ **3D REPRESENTATION**: a textured, navigable terrain mesh built from that elevation surface (Section 9)

→ **INTERACTIVE ANALYSIS**: first-person flythrough plus queryable height and slope information tied back to the numeric elevation data

→ **VALIDATION**: RMSE, MAE, and correlation against LiDAR or reference data, stability-tested across urban, sparse, hilly, and forested terrain (Section 10)

## PHASE 1 COMPLETE

**10 most important things we now understand**

1. The PS splits into two independently scored halves — accuracy (50%) and visualization (50%) — that must both be genuinely addressed.
2. rDSM and absolute DSM are explicitly different deliverables, gated by whether the input is georeferenced.
3. Relative depth and absolute elevation are fundamentally different quantities, linked by an unknown affine transform rather than a simple unit conversion.
4. SRTM is itself an imperfect surface-like proxy — canopy- and roof-reflective, 30 m, roughly year-2000 — not a clean bare-earth ground truth, and it cannot resolve building-level detail alone.
5. The PS names no official validation dataset; the team must source or construct its own reference data to satisfy the 50% accuracy criterion.
6. The organiser's own linked reference repository recommends [GAMUS](https://huggingface.co/datasets/earthflow/GAMUS) as a training/validation dataset.
7. Final evaluation imagery is stated to be ISRO's own optical satellite data, distinct from whatever dataset the team trains and validates on — a generalization requirement, not a development convenience.
8. A monocular depth backbone trained on natural, egocentric imagery faces a genuine domain gap on nadir remote-sensing imagery, across viewpoint, scale, texture, and object-distribution axes.
9. Explicit multi-terrain stability testing (urban, sparse, hilly, forested) means a single best-case demo scene is insufficient evidence of compliance.
10. "Standard geospatial format" and correct CRS/geotransform/vertical-datum handling are implicit engineering requirements, even though the PS names no specific file format or standard.

**10 biggest unresolved technical questions**

1. What reference DSM or LiDAR data can we obtain that overlaps geographically with a depth-estimation training dataset, to allow genuine self-validation?
2. What is GAMUS's actual coverage, resolution, and terrain-type diversity, and does it include hilly and forested examples as the evaluation criteria require?
3. Which pre-trained monocular depth backbone, if any, has documented reasonable transfer to overhead imagery, versus one requiring substantial fine-tuning?
4. How many GCPs or SRTM anchor points are realistically needed for acceptable calibration accuracy, and is that number obtainable per image within a hackathon timeline?
5. What vertical datum do SRTM, our chosen reference DSM, and any GCP source each use, and what correction reconciles them?
6. How should scale calibration be validated as generalizing beyond the specific calibration points used, rather than overfitting to them?
7. What terrain-mesh resolution and rendering approach can stay real-time and navigable for a full-resolution satellite tile, without unacceptable simplification?
8. What published accuracy numbers exist for comparable single-view height-estimation tasks, to know what a credible target result looks like?
9. What does "successful standalone deployment" concretely require — is a self-contained desktop build sufficient, or is a specific packaging expectation implied?
10. How should the pipeline behave, and what should it report, when given imagery for which no reliable reference data exists at evaluation time?

**What Phase 2 must investigate** Answer the research questions grouped A–O in Section 15, prioritizing in this order: locating a usable reference DSM/LiDAR dataset for self-validation; characterizing GAMUS's actual content; surveying prior art on remote-sensing monocular height estimation and its reported accuracy; and identifying viable relative-to-absolute calibration techniques with realistic reference-point requirements.

Do not proceed into Phase 2 until explicitly asked.
