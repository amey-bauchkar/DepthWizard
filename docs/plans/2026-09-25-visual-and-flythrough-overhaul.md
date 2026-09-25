# 3D Visualization, Cinematic Flythrough & Command Center UI Overhaul — Implementation Plan

> **For Claude / Antigravity:** REQUIRED SUB-SKILL: Use `executing-plans` to implement this plan task-by-task.

**Goal:** Transform DepthWizard from a developer-diagnostic utility into a visually stunning, production-grade 3D geospatial intelligence platform tailored specifically to win SIH Problem Statement 26175 (ISRO).

**Architecture:** 
1. Three.js custom GLSL shader pipeline supporting seamless toggling between Aerial Photometric Texture, Scientific Elevation Heatmap (Turbo/Viridis color ramp with dynamic topographic contour lines), and Cyber Blueprint.
2. Parametric spline-based cinematic drone flythrough camera system enabling one-click automated aerial tours.
3. Architectural base pedestal skirt on the terrain mesh to eliminate raw polygon edge cuts.
4. Aerospace Command Center dark-mode UI redesign with visual demo cards and modern HUD telemetry.

**Tech Stack:** Three.js (WebGL2, GLSL Shaders, OrbitControls, Tweening), TypeScript, Vite, CSS3 Design Tokens, FastAPI.

---

### Task 1: Scientific Elevation Heatmap & Topographic Contour GLSL Shader

**Goal:** Provide an instant 1-click shader toggle between `🛰️ Aerial RGB` and `🌈 Elevation Heatmap` in the 3D viewer. The heatmap maps vertex altitude directly to a vibrant, scientific color gradient (Deep Blue $\to$ Cyan $\to$ Emerald $\to$ Amber $\to$ Neon Red) with subtle 5m/10m topographic contour rings, completely eliminating texture smearing while visually proving single-view height estimation.

**Files:**
- Create: `frontend/src/shaders/heatmap.ts`
- Modify: `frontend/src/viewer.ts`
- Modify: `frontend/src/main.ts`
- Modify: `frontend/index.html`

**Step 1: Write `frontend/src/shaders/heatmap.ts`**
- Implement GLSL Turbo colormap function:
  ```glsl
  vec3 turboColormap(float t) { ... }
  ```
- Implement contour lines calculation using screen-space derivatives `fwidth`:
  ```glsl
  float contourLine(float val, float interval, float thickness) { ... }
  ```
- Export shader chunk injectors for `onBeforeCompile`.

**Step 2: Add Shader Toggle to `HeightfieldViewer` in `frontend/src/viewer.ts`**
- Add state: `shaderMode: "aerial" | "heatmap" | "cyber"`.
- Pass uniforms `uShaderMode`, `uZMin`, `uZMax`, `uContourInterval`.
- Provide method `setShaderMode(mode: "aerial" | "heatmap" | "cyber")`.

**Step 3: Add UI Controls to `frontend/index.html` & `frontend/src/main.ts`**
- Add segmented button group in 3D toolbar:
  `[ 🛰️ Aerial Photo | 🌈 Elevation Heatmap | ⚡ Blueprint ]`
- Wire events in `main.ts`.

**Step 4: Build & Test**
- Run `npm run build` in `frontend/`.
- Verify in browser that toggling Heatmap renders glowing, sharp elevation colors with zero texture stretching.

---

### Task 2: 🎬 Automated Cinematic Drone Flythrough Engine

**Goal:** Satisfy the core SIH deliverable *"3D Flythrough"* by adding a 1-click automated cinematic camera flight. The camera glides along a smooth, curved drone path at a dramatic 35° oblique angle orbiting the terrain center, providing a broadcast-quality visual tour without requiring manual mouse maneuvering.

**Files:**
- Modify: `frontend/src/viewer.ts`
- Modify: `frontend/src/main.ts`
- Modify: `frontend/index.html`

**Step 1: Implement Flythrough Path in `frontend/src/viewer.ts`**
- Add state: `flythroughActive: boolean`, `flythroughAngle: number`, `flythroughSpeed: number`.
- In `animate()` loop, when `flythroughActive === true`:
  - Calculate camera position on an elliptical orbit with subtle vertical altitude oscillation:
    ```ts
    const radiusX = Math.max(this.extentX, this.extentY) * 0.75;
    const radiusY = Math.max(this.extentX, this.extentY) * 0.65;
    const height = Math.max(this.extentX, this.extentY) * 0.45 + Math.sin(this.flythroughAngle * 2) * 20;
    this.camera.position.set(
      Math.sin(this.flythroughAngle) * radiusX,
      Math.cos(this.flythroughAngle) * radiusY,
      height
    );
    this.controls.target.set(0, 0, (this.zMax - this.zMin) * 0.3 * this.exaggeration);
    this.camera.lookAt(this.controls.target);
    ```
  - Increment `flythroughAngle += 0.003`.

**Step 2: Add Camera Control Buttons in `frontend/index.html`**
- Add buttons: `[ 🛰️ Orbit | 🎬 Cinematic Flythrough | 🚶 Walk ]`.
- Add quick camera presets: `[ Nadir (Top) | 45° Oblique | Horizon ]`.

**Step 3: Build & Test**
- Run `npm run build` in `frontend/`.
- Test in browser: clicking "🎬 Cinematic Flythrough" smoothly animates the camera over the terrain.

---

### Task 3: Architectural Ground Pedestal & Atmospheric Lighting

**Goal:** Eliminate the harsh, cut-off floating mesh look where terrain edges drop off into empty space. Add a clean, beveled ground skirt/pedestal so the terrain looks like an architectural physical model sitting in a high-tech holographic chamber, with soft rim lighting and atmospheric horizon.

**Files:**
- Modify: `frontend/src/viewer.ts`

**Step 1: Add Pedestal Skirt Geometry**
- Along the 4 boundary edges of the terrain (North, South, East, West), extrude vertices down to `z = -15` scene units into a clean dark bevel material (`#0f172a`).
- Closes the visual volume cleanly.

**Step 2: Polish Atmospheric Lighting & Ground Grid**
- Soften hemisphere light and add subtle rim light (`0x38bdf8`, cyan glow) to pick out mountain and rooftop silhouettes.
- Add subtle, clean coordinate grid lines on the zero floor.

**Step 3: Set Default Vertical Exaggeration to 1.0×**
- Ensure default exaggeration is $1.0\times$ (true metric scale) instead of $2.5\times$ or $3.0\times$, preventing spiky stalagmite distortion.

---

### Task 4: Command Center Dark Theme & Visual Demo Cards

**Goal:** Replace the dated grey developer form aesthetic with an **Aerospace Command Center Dark Theme** (`#090d16` obsidian, glassmorphic cards, glowing borders, crisp typography) and replace raw text links with **Visual Demo Cards** showing satellite thumbnails and terrain tags.

**Files:**
- Modify: `frontend/index.html`
- Modify: `frontend/src/style.css`
- Modify: `frontend/src/main.ts`

**Step 1: CSS Design System Overhaul in `frontend/src/style.css`**
- Update color tokens:
  - Background: `#080c14` (Deep space obsidian)
  - Card Surface: `rgba(15, 23, 42, 0.75)` with `backdrop-filter: blur(12px)`
  - Primary Accent: `#38bdf8` (Electric Cyan) / `#6366f1` (Indigo)
  - Text: `#f8fafc` primary, `#94a3b8` secondary
- Modernize buttons, inputs, tabs, and status badges with sleek pill designs and subtle glow transitions.

**Step 2: Visual Demo Cards in `frontend/index.html` & `frontend/src/main.ts`**
- In Section 1 (Input), render demo tiles as clickable preview cards:
  - **Tile 1:** 🏢 Zürich Urban (0.5m GSD · High-Density Architecture · SwissSURFACE3D LiDAR reference)
  - **Tile 2:** 🌲 Emmental Hilly Forest (2.0m GSD · Rural Mountainous Relief · SwissALTI3D reference)
  - **Tile 3:** 🏔️ Mountain Steep Relief (Extreme elevation gradient)
- Include thumbnail images, badge tags, and one-click loading.

**Step 3: HUD Telemetry Redesign**
- Style the 3D HUD with glassmorphism badges:
  - Calibration Tier & Vertical Reference (`Tier A · EGM2008`)
  - Real-time Cursor Readout (`Elevation: 412.5 m AMSL · Slope: 8.2°`)
  - Compass Needle & Scale Ruler.

---

### Task 5: End-to-End Verification & Browser Demonstration

**Goal:** Verify that the entire application builds cleanly, all 72 automated tests pass, and conduct a full live browser session demonstrating the new visual quality.

**Steps:**
1. Run `python -m pytest -q` $\to$ verify all 72 tests pass.
2. Run `npm run build` in `frontend/` $\to$ verify TypeScript compiles with 0 errors.
3. Open browser via `browser_subagent` and test:
   - Click Zürich Urban demo card.
   - Click Generate Calibrated Surface (Mode B).
   - Open 3D View.
   - Toggle `🌈 Elevation Heatmap` $\to$ verify vivid, smooth colormap and contours.
   - Click `🎬 Cinematic Flythrough` $\to$ verify smooth camera glide.
   - Capture high-resolution screenshots for review.
