# Phase 10 — Final SIH PPT, Storytelling, Technical Narrative & Evaluator Pitch

SIH26175 DepthWizard · 2026-09-21

**Authorities for this phase.** Phase 8 (evidence): the only measured results are Level-0 design-verification checks on synthetic data; every system-level accuracy, latency, terrain, generalization or visualization result is NOT EXECUTED. Phase 9 (product): NO-GO — no application code, weights, outputs, screenshots or packaging exist (re-confirmed by directory inspection at the start of this phase: 0 code files outside `validation/phase8/`, 0 images, 0 decks).

**Consequence for the deck.** The presentation can honestly show: (1) the problem and its hidden structure; (2) the research and prior-art gaps; (3) the design (TL-CSM) and why it follows from evidence; (4) the validation *protocol*; (5) the Level-0 design-verification results, including two defects the team found and fixed in its own design; (6) the implementation plan and current status. It cannot show results, screenshots or a product. Every number below carries a status tag: **[M]** measured (Phase 8 L0, synthetic), **[C]** directly calculated (PROJ grids), **[X]** supported external fact (Phase 3/4 ledgers), **[D]** design specification, **[P]** planned/future. No projected or expected value appears anywhere.

---

## Part 1 — The Complete Story (reconstructed from Phases 1–9)

**A. Problem in one sentence.** Turn a single nadir RGB satellite/aerial image into a surface-elevation model — relative when the image has no coordinates, metric and datum-referenced when it does — and let a user fly through and measure it.

**B. Hidden technical problem.** A single image is scale-ambiguous; georeferencing gives horizontal position and pixel size but *no vertical information*; a free 30 m DEM gives terrain trend but no buildings or trees and is itself a few metres too high under canopy; and "elevation" only means something on a declared vertical datum — in India the ellipsoid–geoid gap is 24–99 m [C]. So the real problem is not "predict depth" but "assign each part of the surface to the source that can actually measure it, and prove the result against independent reference data across terrain types."

**C. Why naive solutions fail.** Zero-shot foundation depth at nadir shows δ₁ ≈ 0 % and metre-scale drift [X: AerialMetric 2026; Sat3R 2026]. Fitting a single scale/offset to a coarse DEM inherits the DEM's canopy/roof bias — on a synthetic scene, naive add-back left +7.8 m (canopy) and +12.2 m (buildings) terrain bias [M]. Random patch splits inflate accuracy: the same DFC19 data scores 2.1 m RMSE randomly split and 5.5–9.3 m on the official spatial test set [X]. Correlation certifies nothing about scale: r = 1.000 with a 60 m bias [M]. Aggregate MAE hides object failure: all-zero and doubled predictions score identically when 57 % of pixels are ground [M].

**D. Key insight.** A DSM is two physically different layers with two different competent sources: the *image* (after LiDAR-supervised adaptation) can measure height above local ground; only an *external absolute source* (DEM, anchors) can supply low-frequency terrain and a datum — and only at its resolution. Calibrate each layer with its own source, compose explicitly, declare the datum and tier, expose uncertainty, and downgrade honestly when a source is missing.

**E. Solution in one sentence.** TL-CSM — the Two-Layer Calibrated Surface Model: adapted monocular backbone → metric height above ground; datum-transformed, ground-masked coarse DEM → terrain layer; optional sparse anchors → refinement; DSM = terrain + height; tiers R/H/T/A label every output; validated on spatially blocked LiDAR references; rendered as exactly that surface in 3D.

**F. What has actually been built.** Research (Phases 1–4), the concept (5), the architecture (6), the implementation plan (7), and a reproducible Level-0 design-verification suite (Phase 8: 8 scripts, seeded, deterministic). **No application, model, dataset, or package exists.** [Phase 9 NO-GO]

**G. What has actually been validated.** Only design mathematics on synthetic data: geospatial round-trips; datum values; the ground-masked normalized-convolution terrain (canopy bias −74 %, buildings −96 %, composed-DSM RMSE 5.69 → 2.00 m — synthetic) [M]; robust anchor recovery (median offset error ≤ 0.13 m for N ≥ 5 with 40 % outliers; RANSAC scale ≤ 1 %) [M]; corrected co-registration converging to (2.000, −1.012) px for a true (2, −1) [M]; metric identities. Two defects found and fixed in the design: PROJ silently returns a 0 m "ballpark" vertical transform when geoid grids are absent [M]; a gradient-kernel orientation error made co-registration diverge [M].

**H. What remains limited.** Everything system-level is unmeasured. Known bounds from evidence: closed canopy leaves ~+2 m terrain residual in the synthetic test [M]; terrain accuracy in relief is capped by DEM slope error (SRTM 5.6 ± 15.7 m in high relief) [X]; no Indian LiDAR reference exists and sub-5 m ISRO imagery is priced for non-government entities [X]; the fine-tuned model cannot be trained on the current CPU-only machine [Phase 8 §3].

**I. Why it matters operationally.** When stereo, LiDAR or InSAR are unavailable or slow (the PS's disaster-management framing), a single optical pass could yield a *provisional*, honestly labelled surface model for terrain understanding, damage context and reconnaissance — the value lies in the labels and validation as much as in the numbers.

Consistency check: A→I contain no claim that the system works; they claim the design is evidence-driven and partially verified.

---

## Part 2 — Central Story

ONE IMAGE → UNDERSTAND SURFACE STRUCTURE (adapted backbone) → ESTIMATE HEIGHT ABOVE GROUND (metric nDSM, tier H) → RECOVER TERRAIN / ABSOLUTE REFERENCE (datum-transformed, ground-masked DEM; anchors; tier T/A) → COMPOSE DSM (terrain + nDSM) → VALIDATE (blocked LiDAR blocks; ME/RMSE/MAE/NMAD/r per terrain) → TURN INTO 3D (RTIN mesh with disclosed residual) → ANALYZE + FLY THROUGH (raster-backed, tier-labelled measurements). Reinforced on every slide: *this is a measurable surface model with labelled provenance, not a depth picture.*

## Part 3 — Core Insight (one sentence)

**"Let each source measure only what it can: the image measures how tall things are, the DEM measures where the ground roughly is, anchors fix the offset — and every metre we output is labelled with which of these produced it."**

---

## Part 4 & 5 — Final Slide Architecture (12 core + 4 appendix)

| # | Title | Purpose | Main message | Required content | Visual | Layout | Evaluator understands | NOT on slide | Evidence |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | DepthWizard — Single-View Height Estimation and 3D Flythrough | Identity | Calibrated, labelled surface models from one image | Team, PS ID 26175, ISRO, one-line definition, status badge "Design & verification stage" | Single clean diagram: image → layered surface → 3D | Title left, diagram right, whitespace | What it is; that the team is candid about stage | Logos clutter, buzzwords | — |
| 2 | The Problem | Requirement | One image → relative or metric DSM + 3D, scored 50/50 on accuracy and visualization | PS modes A/B; evaluation criteria; four terrains | 2D image with "unknown scale / unknown terrain / unknown datum" callouts → "metric DSM + 3D" | Left–right | Exactly what ISRO asked | Literature | PS text [X] |
| 3 | Why This Is Hard | Obstacles | Five independent hard problems | monocular scale ambiguity; nadir domain gap; relative→metric; terrain vs object; datum (24–99 m in India) [C]; multi-terrain validation | Five icons on a horizontal strip, one number each | Strip | Difficulty is structural, not tuning | Model names | Phase 1 §2, Phase 3 GAPs, [C] |
| 4 | What Existing Approaches Miss | Gap | Good components, missing integration | Natural-image depth ✔ structure ✕ nadir scale (δ₁≈0 %) [X]; DEM ✔ trend ✕ detail, +1.6/+5.2 m bias [X]; 3D engines ✔ display ✕ elevation; nDSM literature ✔ heights ✕ absolute DSM | Three-column "✔ / ✕" graphic | Columns | The gap is nDSM → certified absolute DSM | 20 citations | Phase 3 §3, §17 |
| 5 | Our Key Insight — TL-CSM | Core idea | Two layers, two competent sources, labelled | IMAGE→HEIGHT ABOVE GROUND + DEM/ANCHORS→TERRAIN ⇒ DSM; tiers R/H/T/A | Layer stack diagram; tier ladder | Center | Why each source does one job | "Novel invention" | Phase 5 §6–§10 [D] |
| 6 | End-to-End Solution | The brain | Pipeline from input to flythrough | Input/metadata/GSD → preprocessing → adapted model → nDSM → DEM/anchor calibration → terrain → DSM → validation/quality → 3D → flythrough | Horizontal pipeline with tier badges at the right stages | Full width | Where metres enter and how they are labelled | Code, APIs | Phase 6 §2 [D] |
| 7 | Architecture (evaluator view) | Structure | Seven layers, one offline app | Input · Intelligence · Geospatial · Reconstruction · Validation · Visualization · Deployment | Seven-row layered diagram | Rows | It is a coherent system, not a notebook | 13-layer engineering diagram | Phase 6 §1 [D] |
| 8 | Data & Training Strategy | Data discipline | The DEM is *not* ground truth; training truth is LiDAR nDSM | GAMUS (3 cities, nDSM, non-georef) [X]; swisstopo/LINZ/3DEP DSM−DTM across four terrains [X]; blocked splits with buffers; footprint-overlap exclusion; DEM = calibration input; anchors ≠ checkpoints | Data-flow map with "truth / calibration / validation" lanes | Three lanes | Separation of roles prevents leakage | Dataset logos | Phase 4 §4–§5, §10 |
| 9 | Validation Protocol | Rigour | How truth will be measured | blocked test → co-registration → independent reference → ME/RMSE/MAE/NMAD/r → per terrain; ASPRS ≥ 30 checkpoints [X]; "correlation ≠ accuracy: r = 1.000 at 60 m bias" [M] | Vertical protocol chain + one small chart | Chain left, chart right | Numbers will be honest | Any system RMSE | Phase 4 §30, Phase 8 L0-08 |
| 10 | Design Verification — What We Have Proven So Far | Evidence | Mathematics verified; two defects found and fixed | NC terrain: canopy bias 7.79→2.01 m, buildings 12.16→−0.48 m, DSM RMSE 5.69→2.00 m (synthetic) [M]; anchors: offset error 0.03–0.13 m, N≥5 [M]; shift recovery 2.000/−1.012 px [M]; **defect 1:** offline PROJ returns 0 m "ballpark" [M]; **defect 2:** kernel orientation → divergence, fixed [M]; suite deterministic [M] | Two bar pairs (raw vs NC) + two "defect found → fixed" cards | Grid 2×2 | The team tests its own design and reports failures | Labelling synthetic results as accuracy | Phase 8 §8, §24 |
| 11 | From DSM to Interactive 3D (specification) | Product concept | 3D is the DSM, provably | DSM → heightfield tiles → RTIN mesh with disclosed residual → RGB texture → walk/fly; raster-backed labelled measurements; uniform decimation loses 20 m at building edges [M] | Wireframe-over-texture mock diagram (clearly labelled "specification") | Left pipeline, right diagram | Visual fidelity will not be allowed to hide geometry error | Fake screenshots | Phase 6 §11–12, Phase 8 L0-11 |
| 12 | Status, Limitations & Roadmap | Maturity | Where we are, what is bounded, what is next | Status: research + design + verified maths complete; implementation not started (NO-GO for demo) [Phase 9]; bounded: closed canopy +2 m residual (synthetic) [M], DEM slope error in relief [X], no Indian LiDAR [X], no GPU locally; roadmap: MVP B01–B30 → blocked validation → offline package | "Validated / Bounded / Next" three columns | Columns | Engineering maturity and candour | "Coming soon" as capability | Phase 8 §30–33, Phase 9 §25 |
| A1 | Prior-art map | Appendix | Where the field is | HTC-DC Net, GlobalBuildingAtlas (Asia 5.9 m) [X], DFC19 protocol numbers [X], canopy models | Table | — | — | — | Phase 3 |
| A2 | Datum table for India | Appendix | Why datum is first-order | 8 cities EGM96/EGM2008 undulations [C] | Table/map | — | — | — | Phase 8 L0-01 |
| A3 | Calibration tiers & failure behaviour | Appendix | Honest degradation | R/H/T/A entry conditions, failure detection, downgrade | Ladder | — | — | — | Phase 5 §10 |
| A4 | Implementation plan & team | Appendix | Feasibility | Backlog B01–B30, critical path, roles | Gantt-like strip | — | — | — | Phase 7 |

Slides intentionally removed versus the brief: "Quantitative Results" and "Visual Validation" (no measured results or real outputs exist — Phase 8), "Where This Helps" merged into Slide 12 as *potential* use, "Productization" merged into Slide 12 as roadmap (nothing packaged — Phase 9).

---

## Part 6–10 — Results, 3D, Operational, Deployment, Limitations (what the deck can honestly do)

- **Results:** Slide 10 presents only Level-0 synthetic verification, labelled as such on the slide ("design verification on synthetic scenes — not system accuracy"). No RMSE/MAE/r against real references exists; the deck says so.
- **Visual validation:** no real INPUT → nDSM → DSM → REFERENCE → ERROR chain exists; Slide 11 shows a specification diagram, not screenshots.
- **3D/product:** specification only (Slide 11); the demo cannot be shown (Phase 9 NO-GO).
- **Operational value:** framed on Slide 12 as "potential operational use enabled by a validated system": provisional terrain understanding and damage context when stereo/LiDAR/InSAR are unavailable — conditional on the validation still to be run.
- **Deployment:** Slide 12 states "offline single-executable package — planned; PyInstaller not yet attempted" [Phase 8 §3].
- **Limitations table (Slide 12):**

| Validated (design level, synthetic) | Still to be tested (system) |
| --- | --- |
| Geo round-trips, datum values, ballpark guard requirement [M][C] | Any real accuracy; all four terrains |
| NC terrain reduces DEM object bias [M] | Real DEM + real imagery; closed-canopy behaviour |
| Median/RANSAC anchor recovery [M] | Real anchor availability (ICESat-2/GCP) |
| Shift recovery after kernel fix [M] | Aerial→satellite transfer; GSD robustness |
| Metric identities; correlation blindness [M] | Fine-tuned vs zero-shot ablation; Indian imagery |
| Suite determinism [M] | Packaging on clean machine; performance |

---

## Part 11 — Innovation Audit

| Component | Status | May be called a contribution? |
| --- | --- | --- |
| Foundation depth backbone (DA-V2-S) | Existing | No |
| Fine-tuning a foundation encoder for nadir height | Existing lineage (Depth Any Canopy, Sat3R, Tolan) [X] | No |
| DEM fusion / de-biasing | Existing concept (FABDEM; add-back) [X] | No |
| **TL-CSM two-layer composition with ground-masked NC terrain** | Proposed system integration; mechanism verified synthetically [M] | **Yes** (as integration, once validated on real data) |
| **Tiered calibration semantics R/H/T/A with failure detection and honest downgrade** | Proposed system design [D] | **Yes** (design contribution) |
| **Validation protocol** (blocked, co-registered, datum-consistent, object-aware, per terrain) | Assembled methodology from DEM/RS standards [X][D] | Yes, as discipline, not invention |
| Measurement-faithful visualization with disclosed mesh residual | Engineering integration [D] | Secondary |
| Datum-safety guard (reject "ballpark" transforms) | Engineering finding from Phase 8 [M] | Secondary, memorable |

Framing to use: *"Our contribution is the integration, the calibration discipline, the validation methodology and the honest labelling of these components into one defensible end-to-end workflow — not any single algorithm."* This reflects the evidence.

## Part 12 & 13 — Design Language & Visual Priority

Navy (#1B2A41) / slate blue / white system, one accent (amber for tier badges and warnings); Inter or Source Sans; 12-column grid; ≥ 40 % whitespace; diagrams drawn as flat vector, no glow/gradients/stock art/fake satellite imagery. Visual priority: real outputs (none available → omitted), real benchmark results (none → omitted; synthetic results shown as labelled bar pairs), real architecture diagrams (Slides 6–7), real data examples (Slide 8 sources), explanatory diagrams (3–5, 11), decoration (none).

## Part 14 — Slide Density Audit

| Slide | Words | Visual elements | Messages | Verdict |
| --- | --- | --- | --- | --- |
| 1 | ≤ 15 | 1 | 1 | OK |
| 2 | ≤ 40 | 1 | 1 | OK |
| 3 | ≤ 45 | 5 icons | 1 (five obstacles, one message: structural difficulty) | OK |
| 4 | ≤ 50 | 3 columns | 1 | OK |
| 5 | ≤ 35 | 2 | 1 | OK |
| 6 | ≤ 45 | 1 pipeline | 1 | OK (keep labels short) |
| 7 | ≤ 40 | 7 rows | 1 | OK |
| 8 | ≤ 60 | 3 lanes | 1 ("DEM ≠ truth") | Watch density |
| 9 | ≤ 60 | chain + chart | 1 | OK |
| 10 | ≤ 70 | 4 cards | 2 (verified maths; defects fixed) | Acceptable — both serve "we test ourselves"; otherwise split |
| 11 | ≤ 45 | 2 | 1 | OK |
| 12 | ≤ 80 | 3 columns | 1 (maturity) | Watch density; move roadmap detail to A4 |

## Part 15 — Evaluator Attention Model (per slide: 5 s / 20 s / 60 s)

1: "DepthWizard, single image → labelled surface model" / "design & verification stage" / "candid team". 2: "one image → DSM + 3D" / "two modes, 50/50 scoring" / "why the modes differ". 3: "five hard problems" / "scale, domain, datum, terrain, validation" / "datum alone is 24–99 m in India". 4: "components exist, integration doesn't" / "depth ✔ structure ✕ scale; DEM ✔ trend ✕ detail" / "the undocumented step is nDSM → absolute DSM". 5: "two layers" / "image measures heights, DEM measures ground" / "tiers label every metre". 6: "a pipeline" / "where calibration happens" / "how honesty is enforced by design". 7: "seven layers, offline" / "geospatial layer is first-class" / "why a monolith". 8: "DEM is not truth" / "LiDAR nDSM trains, DEM calibrates, LiDAR validates" / "blocked splits prevent 28 % inflation". 9: "how we measure" / "co-register, then ME/RMSE/MAE/NMAD/r per terrain" / "why correlation alone fails". 10: "verified maths, found bugs" / "NC cuts DEM object bias; ballpark datum bug" / "team tests its own design". 11: "3D is the DSM" / "residual on screen; raster-backed numbers" / "why decimation is banned". 12: "not built yet; here is the plan" / "what is bounded" / "what MVP proves first".

---

## Part 16 — Pitch Scripts

**60-second pitch.** "ISRO asked for a surface model and a 3D flythrough from a single satellite image. The hard part isn't predicting depth — it's that one image has no scale, coordinates give no vertical information, and free elevation maps are 30-metre blobs that sit a few metres too high under trees. DepthWizard splits the job: an adapted depth model measures how tall things are above the ground; a coarse DEM, read only where the ground is visible and converted to one vertical datum, supplies the terrain; anchors fix the offset when available. We add the layers into a DSM and label every metre with the tier that produced it — relative, height-above-ground, or absolute. Our validation protocol uses spatially blocked LiDAR references across urban, sparse, hilly and forested terrain. So far we have verified the mathematics on synthetic scenes — the terrain step removed 74–96 % of the DEM's object bias — and found and fixed two defects in our own design, including a silent datum error. The application itself is our next build."

**2-minute pitch.** Adds: why zero-shot depth fails at nadir (δ₁ ≈ 0 %) [X]; why the DEM cannot resolve buildings (30 m posting, +1.6/+5.2 m bias) [X]; the tier ladder R/H/T/A and honest downgrade; the datum point (Kanyakumari −98.6 m vs Leh −23.6 m) [C]; the DFC19 split discrepancy (2.1 vs 5.5–9.3 m) as the reason for blocked splits [X]; a clear statement that no accuracy number exists yet and what the first validation run will report.

**5-minute pitch.** Slides 1–12 at ~25 s each: problem → five difficulties → gap → insight → pipeline → architecture → data discipline → validation protocol → design verification with both defects → 3D specification → status/limitations/roadmap; closes on the five must-understand points (Final Question).

**10-minute deep technical pitch.** Adds appendix A1–A4: prior-art numbers (HTC-DC Net 2.12 m random split vs 5.46 m official; GlobalBuildingAtlas Asia 5.9 m) [X]; the L0 results in detail (anchor N-curves, Huber failure at 40 % outliers, hold-out RMSE 6.81 vs NMAD 0.83 with one blunder, kernel-orientation divergence and correction) [M]; datum table [C]; the calibration tier entry conditions and failure detectors [D]; the implementation critical path and the Python 3.12/Node 24/GPU constraints [Phase 7/8].

---

## Part 17 — Slide-by-Slide Speaker Notes

| # | What I say (essence) | Point at | Evidence mentioned | Do NOT say | Likely interruption | One-sentence recovery |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | "DepthWizard turns one image into a labelled surface model; we are at design-and-verification stage." | status badge | — | "working system" | "Is it built?" | "Not yet — Phase 8 verified the mathematics; the MVP is the next build, and this deck shows exactly what it will prove." |
| 2 | "Two modes: relative for PNG, metric for GeoTIFF; scored half on accuracy, half on visualization." | mode arrows | PS text | "high precision" | "Which terrains?" | "All four named terrains; our references for them are Swiss, NZ and US LiDAR." |
| 3 | "Five independent difficulties; the datum alone is 24 to 99 metres across India." | datum icon | [C] PROJ grids | "solved" | "Why does datum matter for a DSM?" | "Because mixing ellipsoidal and geoid heights would dwarf every building." |
| 4 | "Each existing piece does one thing well and misses the others; the undocumented step is nDSM to certified absolute DSM." | ✕ marks | [X] δ₁≈0 %; FABDEM bias | "nobody has done this" | "Isn't this just HTC-DC Net plus SRTM?" | "HTC-DC Net stops at height above ground; the absolute step and its validation are what we add — and what we must still prove." |
| 5 | "Let each source measure what it can; label every metre." | tier ladder | [D] | "novel algorithm" | "Why not learn everything end-to-end?" | "An end-to-end model can copy the DEM and look right at 30 m while failing at buildings; explicit layers stay inspectable." |
| 6 | Walk the pipeline left to right, pausing at the tier badges. | badges | [D] | code names | "Where does scale come from?" | "From LiDAR-supervised fine-tuning plus true pixel size — tier H — and optionally anchors — tier A." |
| 7 | "Seven layers in one offline application; geospatial correctness is a layer, not an afterthought." | geospatial row | [D] | microservices | "Why Three.js not Unity?" | "One local scene, no globe, no engine runtime to package; commodity either way." |
| 8 | "LiDAR nDSM trains; the DEM only calibrates; LiDAR validates — never the same data twice." | lanes | [X] GAMUS 3 cities; Kattenborn 28 % | "GAMUS covers all terrains" | "GAMUS is recommended by ISRO — why not enough?" | "It is three US cities, height-above-ground only, without coordinates — good for training, useless for absolute calibration or hills and forests." |
| 9 | "Blocked splits, co-registration, then bias and robust spread with RMSE, per terrain — and never correlation alone." | r = 1.0 chart | [M] L0-08 | any system RMSE | "What RMSE do you get?" | "None yet — the protocol is fixed before training so the first number we report is honest." |
| 10 | "On synthetic scenes the terrain step cut DEM object bias by 74–96 %, anchors recovered offsets to 0.1 m, and we found two bugs in our own design." | defect cards | [M] | "accuracy" | "Synthetic isn't real." | "Correct — it proves the mathematics, not the accuracy; real blocked validation is the first MVP milestone." |
| 11 | "The 3D scene is the DSM; the mesh error is on screen; measurements come from the raster." | residual badge | [M] decimation 20 m loss | "photorealistic" | "Can I trust what I see?" | "Only as far as the residual and the validation table beside it — by design." |
| 12 | "Research, design and verified mathematics are done; implementation is next; here is what is bounded and why." | three columns | Phase 8/9 | "coming soon" as done | "When will it run?" | "The MVP critical path is defined to the task level; the first milestone is a blocked validation table on real LiDAR." |

---

## Part 18 — Thirty Hard Evaluator Questions

| # | Group | Question | 15-s answer | 60-s answer (core) | Evidence | Do not claim |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | Problem | Why is single-view DSM even plausible? | Heights above ground are learnable from nadir images with LiDAR supervision; absolute terrain must come from outside. | Field results (nDSM RMSE 1.3–5.5 m in-distribution) show the object layer is learnable; absolute elevation is not in the image, hence the two-layer design. | Phase 3 E4/E12 [X] | parity with stereo |
| 2 | Problem | What does "relative DSM" mean in your system? | Structure only, unitless, watermarked RELATIVE. | Without GSD the metric scale is unknowable; we normalise per image and refuse metres unless the user supplies a scale hint, which is then labelled unverified. | Phase 5 §8 [D] | metres for PNG |
| 3 | Problem | Isn't georeferencing enough for elevation? | No — it gives x, y and pixel size, not z. | The vertical reference must come from a DEM or anchors and be transformed to one datum; India's geoid separation is −24 to −99 m. | [C] L0-01 | "GeoTIFF gives height" |
| 4 | AI/CV | Why Depth Anything V2? | PS mandates a pretrained monocular backbone; Small is Apache-2.0 with training code. | Fine-tuning such encoders for nadir height is demonstrated (canopy MAE 2.8 m; DFC2019 MAE 4.59→2.82 after fine-tune). | Phase 3 E13/E14/E16; Phase 5 G2 [X] | zero-shot metric depth works |
| 5 | AI/CV | Why not use it zero-shot? | At nadir, zero-shot metric models show δ₁ ≈ 0 % and scale drift. | AerialMetric and Sat3R both report catastrophic zero-shot scale failure; structure survives, scale does not, so adaptation is mandatory. | [X] | "works out of the box" |
| 6 | AI/CV | How do you handle tall buildings? | Long-tail-aware loss; report height-bin errors and building-instance RMSE. | 57 % of pixels are near-zero; HTC-DC's head-tail cut and F1-HE address it; we will report bins rather than hide them. | Phase 3 E4/E28 [X] | high-rise accuracy |
| 7 | AI/CV | What about GSD differences? | Canonical GSD bands with true GSD from the transform; out-of-range flagged. | Downsampling softens edges (13 m max on a synthetic step); we prefer canonical bands ≤ input GSD and document blur. | L0-03 [M] | GSD invariance |
| 8 | RS | Aerial-trained, satellite-tested — will it transfer? | Unknown until tested; planned sensor-shift experiment. | Training mixes satellite-like DFC19 and aerial LiDAR sets; the NAIP→LISS-IV test is defined. | Phase 4 §21 | transfer to Cartosat |
| 9 | RS | Forests? | Worst case by physics; flagged and reported separately. | First-surface sensing hides ground; where no ground support exists the DEM is used raw and flagged; synthetic residual was +2 m. | L0-04 [M]; Phase 4 F7 | forest accuracy |
| 10 | RS | Hilly terrain? | Terrain layer inherits DEM slope error; co-registration mandatory. | SRTM error is 5.6 ± 15.7 m in high relief; our validation reports error by slope class after shift removal. | [X] E21; L0-07 [M] | hilly accuracy |
| 11 | RS | Off-nadir images? | Flagged; heights may be biased. | Ortho near-nadir assumed; facade heuristic planned; geopose-style methods need RPCs the PNG path lacks. | Phase 3 E11 | correctness off-nadir |
| 12 | Calibration | How exactly do you get metres? | Learned metric head + true pixel size (tier H); anchors refine (tier A). | Scale is supervised in metres during fine-tuning; anchors correct offset/scale robustly; synthetic recovery ≤ 0.13 m / ≤ 1 %. | L0-05 [M] | real-data scale accuracy |
| 13 | Calibration | Why not just fit the prediction to SRTM? | That copies SRTM's canopy/roof bias into the result. | Naive add-back left +7.8 m (canopy) and +12.2 m (buildings) synthetic bias; ground-masked NC cut those to +2.0 and −0.5 m. | L0-04 [M] | DEM as truth |
| 14 | Calibration | What if no DEM? | Tier H: metric heights above ground, no elevation, said so. | Monotone downgrade R←H←T←A is designed in; nothing is fabricated. | Phase 5 §10 [D] | metres regardless |
| 15 | Calibration | How many GCPs? | Unknown for real scenes; N ≥ 5 recovered offset/scale synthetically. | The anchor-count curve is a planned experiment; ASPRS wants ≥ 30 checkpoints for a full assessment. | L0-05 [M]; F7 [X] | "two points suffice" |
| 16 | Calibration | Robust estimator? | Median/Theil–Sen and RANSAC; not Huber. | Huber failed at 40 % gross outliers (10.99 m vs 3.0); median held to 0.05 m. | L0-05 [M] | — |
| 17 | DEM/Geo | Which vertical datum? | EGM2008 by default; everything transformed via PROJ grids. | SRTM is EGM96, Copernicus EGM2008, ICESat-2 ellipsoidal; the app refuses tier T if grids are missing because PROJ otherwise silently returns 0 m. | L0-01/01b [M] | "WGS84 heights" |
| 18 | DEM/Geo | Is the terrain layer a DTM? | No — a low-frequency DEM-derived surface, labelled as such. | It carries DEM accuracy and NC smoothing; certified DTM is a non-claim. | Phase 5 §11 [D] | DTM |
| 19 | DEM/Geo | Pixel alignment errors? | One grid per job; co-registration reported. | Δz ≈ Δx·tan θ; a 1-px shift equals full building height at edges; grid search recovered a synthetic shift to 0.02 px. | L0-07 [M] | — |
| 20 | Validation | What accuracy do you have? | None yet — no system run. | The protocol is fixed: blocked splits, co-registration, ME/RMSE/MAE/NMAD/r, per terrain and class; first run is the MVP milestone. | Phase 8 §14 | any number |
| 21 | Validation | Why not RMSE alone? | Blunders dominate RMSE. | One +20 m point gave hold-out RMSE 6.81 m against NMAD 0.83 m; ASPRS requires blunder investigation and mean-error reporting. | L0-05 [M]; F7 [X] | — |
| 22 | Validation | Correlation looks great — enough? | No: r = 1.000 with a 60 m bias. | Correlation is scale/offset-blind; we always pair it with ME and RMSE. | L0-08 [M] | r as accuracy |
| 23 | Validation | Leakage? | Blocked splits with buffers; footprint exclusion; anchors ≠ checkpoints. | Random splits inflate up to 28 %; DFC19 shows 2.1 vs 5.5–9.3 m; our manifests will record splits and overlap checks. | [X] E22/E4/E11 | "no leakage" before manifests exist |
| 24 | Dataset | Why not GAMUS alone? | Three US cities, nDSM, no coordinates. | It trains heights above ground but cannot support absolute calibration or hilly/forested tests; Swiss/NZ/US LiDAR fill that. | E6 [X] | GAMUS covers four terrains |
| 25 | Dataset | Indian data? | No open Indian LiDAR; sub-5 m ISRO imagery priced for NGEs. | India evidence will be bounded by ICESat-2 (0.7–4 m) and CartoDEM (8 m LE90). | F6/F10/E23 [X] | Indian metre-level accuracy |
| 26 | 3D | Does the 3D prove accuracy? | No — it proves faithfulness to the DSM via the residual. | Error-bounded RTIN with displayed residual; uniform decimation loses 20 m at edges; measurements read the raster. | L0-11 [M] | fidelity = accuracy |
| 27 | Deployment | Standalone? | Planned offline bundle with grids, DEM tiles, weights; not yet built. | Clean-machine test is a P0 gate; the datum self-test runs at start. | Phase 9 §10 | "runs anywhere" |
| 28 | Novelty | What is new? | Integration, calibration discipline, validation and labelling — not components. | Backbone, fine-tuning, DEM fusion, viewers all exist; the documented nDSM→absolute step with tiers and per-layer validation is the contribution to prove. | Phase 3 §16–17 | novel algorithm |
| 29 | Limitations | Biggest risk? | Indian morphology and terrain transfer. | Best global systems lose 2–3× on unseen morphology (Asia 5.9 m); we can only bound India with sparse anchors. | E4/E12 [X] | — |
| 30 | Limitations | Why should we trust a design with no results? | Because it is built from measured failure modes and we already found and fixed two of our own defects. | Silent PROJ ballpark; kernel-orientation divergence — both caught by synthetic tests we ship in CI. | Phase 8 §8, §27 | trust beyond evidence |

---

## Part 19 — "Why Should I Believe You?" (Claim → Evidence Matrix)

| Claim in deck | Status | Evidence | Rewrite if weak |
| --- | --- | --- | --- |
| One image cannot yield absolute elevation without an external vertical source | Established | Phase 1 §2; [C] datum values | — |
| Zero-shot foundation depth fails metrically at nadir | [X] | AerialMetric δ₁≈0 %; Sat3R MAE 4.59 m drift | — |
| Coarse DEMs carry +1.6/+5.2 m object bias and no sub-60 m detail | [X] | FABDEM; Nyquist | — |
| Ground-masked NC terrain reduces that bias | [M] synthetic | L0-04: −74 % / −96 % | Must say "on synthetic scenes" |
| Anchors recover offset/scale robustly | [M] synthetic | L0-05 | "with N ≥ 5, median/RANSAC" |
| Co-registration recovers shifts | [M] synthetic | L0-07d | "after kernel fix; iterative or grid search" |
| Correlation cannot certify accuracy | [M] | L0-08 r = 1.0 at 60 m | — |
| Blocked splits are necessary | [X] | Kattenborn 28 %; DFC19 2.1 vs 5.5–9.3 | — |
| Datum handling is first-order in India | [C] | −98.6…−23.6 m | — |
| Offline datum transforms can fail silently | [M] | L0-01b "ballpark" 0 m | — |
| The system achieves X m RMSE | **none** | — | **Removed from deck** |
| The 3D scene is faithful | [D] + [M] decimation evidence | residual design; L0-11 | "by design; residual displayed" |
| Standalone offline deployment | [P] | none | "planned; clean-machine test pending" |
| Works across four terrains | [P] | none | "designed and to be tested; references identified" |

## Part 20 — "So What?" Test

Kept because they affect accuracy/correctness/compliance: adaptation, two-layer composition, datum guard, tiers, blocked validation, residual-disclosed 3D, raster-backed measurements, offline packaging. De-emphasised: framework names, API design, job system, uncertainty raster (planned), semantic head, glTF export, contours (cut in Phase 9).

## Part 21 — Defensible Distinctions (demonstrated vs designed)

Demonstrated (synthetic): explicit nDSM/terrain separation reduces DEM bias; datum-awareness incl. failure detection; robust anchor statistics; correlation-blindness awareness. Designed, not yet demonstrated: layer-specific calibration on real data; blocked multi-terrain validation; honest degradation modes in software; measurement-faithful 3D; standalone deployment; provenance panel. The deck labels each accordingly.

## Part 22 — The One Slide That Sells the Project (Slide 5 + proof strip)

INPUT: one nadir RGB image (± GeoTIFF metadata, ± coarse DEM, ± anchors). INTELLIGENCE: an adapted depth model that measures height above local ground. CALIBRATION: terrain from a datum-transformed, ground-masked DEM; offset from anchors; every output tagged R/H/T/A. OUTPUT: metric nDSM + EGM2008 DSM as GeoTIFF + faithful 3D scene. PROOF: blocked LiDAR validation with ME/RMSE/MAE/NMAD/r per terrain — *protocol fixed; first run pending* — and design mathematics already verified (NC −74/−96 % synthetic bias; datum guard). One visual: a layered surface (terrain sheet + object layer) with a tier ladder on the right and a proof strip beneath.

---

## Part 23 — Final PPT Red Team

| Problem | Severity | Slide | Fix |
| --- | --- | --- | --- |
| Evaluator expects a demo; there is none | Critical | 1, 12 | State stage on Slide 1; put the honest status and critical path on 12; never imply screenshots |
| Synthetic numbers mistaken for accuracy | High | 10 | Header "design verification — synthetic"; grey background; no "RMSE" without the word synthetic |
| "TL-CSM" sounds like a claimed invention | Medium | 5 | Subtitle "our proposed integrated framework"; innovation table in A1 |
| Too many obstacles on Slide 3 | Medium | 3 | Five icons, one number each, no sentences |
| Data slide dense | Medium | 8 | Three lanes, ≤ 60 words; details to A4 |
| Terminology (nDSM, NC, tier) unexplained | Medium | 5–6 | Footer glossary line on 5 and 6 |
| No operational value slide | Low | 12 | One line "potential use, conditional on validation" |
| Weak final message | High | 12 | Close on the five points (Final Question) |
| Mismatch with software | Critical if ignored | all | Every capability verb in future/conditional tense until Phase 9 GO |

## Part 24 — Presentation Readiness Score

| Dimension | /10 | Why |
| --- | --- | --- |
| Problem clarity | 9 | Phase 1 decomposition is precise and visual |
| Technical clarity | 8 | Two-layer insight is explainable in one sentence |
| Research credibility | 8 | Verified external facts with ledgers; corrected inherited errors |
| Evidence quality | 4 | Only synthetic design verification; no system results |
| Solution storytelling | 7 | Coherent chain; weakened by absent product |
| Architecture communication | 7 | Seven-layer view is evaluator-readable |
| Validation communication | 6 | Protocol strong; no numbers |
| Product demonstration | 1 | Nothing to show |
| Visual quality | 6 (design spec) | Depends on execution; no real imagery available |
| Innovation communication | 7 | Honest integration framing |
| Honesty / claim discipline | 10 | Every number tagged; non-claims explicit |
| Evaluator memorability | 6 | Datum-bug story and two-layer idea are memorable; lack of demo hurts |
| **Overall PPT quality** | **6** | A credible research-and-design deck; not yet a results deck |

---

## Part 25 — Final Output Package

**A. Final story** — Part 1. **B. Slide list** — Part 4 (12 core + A1–A4). **C. Slide content** — Part 4 table. **D. Visual design** — Parts 12–13 and layouts in Part 4. **E. Speaker notes** — Part 17. **F–I. Pitches** — Part 16. **J. 30 questions** — Part 18. **K. Claim → evidence matrix** — Part 19. **L. PPT ↔ product consistency** — every capability stated in future/conditional tense; no screenshots; results slides removed; status badge on Slide 1; matches Phase 9 NO-GO. **M. Red-team findings** — Part 23. **N. Readiness score** — Part 24.

---

## Final Question — The five things an expert evaluator must understand in five minutes

1. **The problem is a calibration-and-reference problem, not a depth problem:** one image has no scale, coordinates have no height, and India's datum gap alone is 24–99 m.
2. **Each source measures only what it can:** the adapted image model gives height above ground; the coarse DEM gives low-frequency terrain read only where ground is visible; anchors fix the offset — and every metre is labelled R/H/T/A.
3. **The DEM is a calibration input, never ground truth:** naive add-back inherits +8–12 m object bias; our ground-masked terrain step removed 74–96 % of it in synthetic tests.
4. **Validation is fixed before results:** spatially blocked LiDAR references, co-registration, ME/RMSE/MAE/NMAD/r per terrain — because random splits inflate and correlation alone is blind to a 60 m bias.
5. **We are at design-and-verification stage, and we test our own design:** the mathematics is verified and two real defects (a silent datum fallback and a kernel-orientation bug) were found and fixed; the application, its measured accuracy and its offline package are the next milestones — not claims we make today.
