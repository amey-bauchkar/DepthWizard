/**
 * HeightfieldViewer (Three.js) — DepthWizard 3D Surface Explorer.
 *
 * Architecture:
 *  - The mesh is a DSM-derived heightfield. No fake geometry, no procedural structures.
 *  - The viewer is a visual representation ONLY. Measurements are never read from the mesh.
 *  - Picks are converted to raster pixel coordinates; the caller samples server rasters.
 *  - Vertical exaggeration is a display-only parameter. It never modifies the underlying DSM.
 *
 * Two display modes:
 *  - relative (Mode A / tier R): unitless; display height scale is a pure visual setting.
 *  - metric (Mode B): 1 scene unit = 1 metre. Vertical = metres above layer minimum ×
 *    exaggeration (default 1.0 = true scale). Exaggeration is permanently shown in HUD.
 */
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import type { HeightfieldMeta } from "./api";
import { buildRTINGeometry, type RTINMeshResult } from "./martini";
import { buildLoD1BuildingGroup, createFacadeTexture, type LoD1Data } from "./lod1";
import { TURBO_GLSL, type ShaderVisualMode } from "./shaders/heatmap";
import { buildPedestalGeometry } from "./pedestal";

export interface ViewerInfo { vertices: number; triangles: number; nodataDropped: number; extent: [number, number] }
export interface LoadOptions {
  metric: boolean;
  spacing: number;
  exaggeration?: number;
  hud: {
    state: string;      // e.g. "TERRAIN ELEVATION ONLY" or "RELATIVE SURFACE"
    tier: string;       // e.g. "Tier T · EGM2008" or "Tier R · non-metric"
    quality: string;    // e.g. "LIMITED"
    layer: string;      // e.g. "DSM" or "Relative structure"
    showNorth: boolean; // true for Mode B (georeferenced), false for Mode A
  };
}
export type PickHandler = (colSource: number, rowSource: number) => void;

export class HeightfieldViewer {
  private renderer: THREE.WebGLRenderer;
  private scene = new THREE.Scene();
  private camera: THREE.PerspectiveCamera;
  private controls: OrbitControls;
  private mesh: THREE.Mesh | null = null;
  private regularGeometry: THREE.BufferGeometry | null = null;
  private rtinGeometry: THREE.BufferGeometry | null = null;
  private gridHelper: THREE.GridHelper | null = null;
  private marker: THREE.Mesh | null = null;
  private baseZ: Float32Array | null = null;
  private zScale = 0.2;
  private exaggeration = 1.0;
  private metric = false;
  private spacing = 1;
  private zMin = 0;
  private zMax = 0;
  private W = 0; private H = 0; private factor = 1;
  private extentX = 1; private extentY = 1;

  // Anti-Smear Facade Shader & LoD-1 Buildings state
  private antiSmearEnabled = true;
  private facadeTex: THREE.CanvasTexture | null = null;
  private shaderUniforms: Record<string, { value: any }> | null = null;
  private buildingsData: LoD1Data | null = null;
  private buildingsGroup: THREE.Group | null = null;
  private lod1Visible = true;
  private aerialTex: THREE.Texture | null = null;
  private cityMode = false;

  // Visual Shaders & Pedestal
  private visualMode: ShaderVisualMode = "aerial";
  private pedestalMesh: THREE.Mesh | null = null;

  // Cinematic Flythrough & Camera modes
  private cameraMode: "orbit" | "walk" = "orbit";
  private flythroughActive = false;
  private flythroughAngle = 0;
  private flythroughSpeed = 0.0032;
  private onFlythroughToggleCb: ((active: boolean) => void) | null = null;

  private meshMode: "regular" | "rtin" = "regular";
  private rtinTolerance = 0.5;
  private onCameraModeChangeCb: ((mode: "orbit" | "walk") => void) | null = null;
  private onWalkStatsCb: ((eyeZ: number, speedKmH: number) => void) | null = null;

  // First-Person Walk state (Z-up coordinate system)
  private yaw = 0;     // radians, 0 = facing North (+Y)
  private pitch = 0;   // radians, 0 = horizontal
  private keys: Record<string, boolean> = {};
  private walkVelocity = new THREE.Vector3();
  private clock = new THREE.Clock();

  // HUD elements
  private hudTL: HTMLDivElement;
  private hudTR: HTMLDivElement;
  private hudBR: HTMLDivElement;
  private hudBL: HTMLDivElement;

  private raf = 0;
  private raycaster = new THREE.Raycaster();
  private pickHandler: PickHandler | null = null;
  private downAt: [number, number] | null = null;

  constructor(private container: HTMLElement) {
    if (!HeightfieldViewer.webglAvailable()) throw new Error("WEBGL_UNAVAILABLE");

    // Renderer
    this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.setSize(container.clientWidth, container.clientHeight);
    this.renderer.shadowMap.enabled = false;
    container.appendChild(this.renderer.domElement);

    // Camera
    this.camera = new THREE.PerspectiveCamera(45, container.clientWidth / container.clientHeight, 0.1, 200000);

    // Controls
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.08;
    this.controls.screenSpacePanning = true;
    this.controls.minDistance = 0.5;
    this.controls.maxDistance = 150000;
    this.controls.maxPolarAngle = Math.PI * 0.88;
    this.controls.addEventListener("start", () => {
      if (this.flythroughActive) {
        this.setFlythrough(false);
      }
    });

    // Scene background — sleek dark obsidian slate
    this.scene.background = new THREE.Color(0x0b1120);
    this.scene.fog = new THREE.FogExp2(0x0b1120, 0.00006);

    // 1. Ambient baseline illumination — guarantees all building facades are clean, bright, and legible
    const amb = new THREE.AmbientLight(0xffffff, 0.90);
    this.scene.add(amb);

    // 2. Z-UP Hemisphere light — bright sky from above (+Z), soft fill from below (-Z)
    const hemi = new THREE.HemisphereLight(0xffffff, 0xdbeafe, 0.70);
    hemi.position.set(0, 0, 1);
    this.scene.add(hemi);

    // 3. Primary warm sun — casts crisp architectural relief across roofs and walls
    const sun = new THREE.DirectionalLight(0xfffaea, 1.30);
    sun.position.set(600, -800, 1200);
    this.scene.add(sun);

    // 4. Soft cool sky fill — illuminates opposing facades
    const fill = new THREE.DirectionalLight(0xe0f2fe, 0.50);
    fill.position.set(-600, 800, 900);
    this.scene.add(fill);

    // HUD containers
    this.hudTL = this._hud("hud-tl");
    this.hudTR = this._hud("hud-tr");
    this.hudBR = this._hud("hud-br");
    this.hudBL = this._hud("hud-bl");

    // Events
    window.addEventListener("resize", () => this.resize());
    const el = this.renderer.domElement;
    el.addEventListener("pointerdown", (e) => { this.downAt = [e.clientX, e.clientY]; });
    el.addEventListener("pointerup", (e) => {
      if (!this.downAt) return;
      const moved = Math.hypot(e.clientX - this.downAt[0], e.clientY - this.downAt[1]);
      this.downAt = null;
      if (moved < 4) this.pick(e);
    });

    // First-person pointer lock & mouse look
    el.addEventListener("click", () => {
      if (this.cameraMode === "walk" && document.pointerLockElement !== el) {
        el.requestPointerLock();
      }
    });
    document.addEventListener("mousemove", (e) => {
      if (this.cameraMode !== "walk" || document.pointerLockElement !== el) return;
      const sens = 0.0022;
      this.yaw += e.movementX * sens;
      this.pitch -= e.movementY * sens;
      const maxP = Math.PI / 2 - 0.05;
      this.pitch = Math.max(-maxP, Math.min(maxP, this.pitch));
    });
    document.addEventListener("pointerlockchange", () => {
      const isLocked = document.pointerLockElement === el;
      if (!isLocked && this.cameraMode === "walk") {
        this.setCameraMode("orbit");
      }
    });
    window.addEventListener("keydown", (e) => {
      if (this.cameraMode === "walk") {
        this.keys[e.code] = true;
        if (["KeyW", "KeyA", "KeyS", "KeyD", "Space", "ShiftLeft", "ShiftRight"].includes(e.code)) {
          e.preventDefault();
        }
      }
    });
    window.addEventListener("keyup", (e) => {
      if (this.cameraMode === "walk") {
        this.keys[e.code] = false;
      }
    });

    this.animate();
  }

  private _hud(cls: string): HTMLDivElement {
    const d = document.createElement("div");
    d.className = cls;
    this.container.appendChild(d);
    return d;
  }

  static webglAvailable(): boolean {
    try { const c = document.createElement("canvas"); return !!(c.getContext("webgl2") || c.getContext("webgl")); } catch { return false; }
  }

  onPick(h: PickHandler | null) { this.pickHandler = h; }

  private pick(e: PointerEvent) {
    if (this.cameraMode === "walk" || !this.mesh || !this.pickHandler) return;
    const r = this.renderer.domElement.getBoundingClientRect();
    const nd = new THREE.Vector2(((e.clientX - r.left) / r.width) * 2 - 1, -((e.clientY - r.top) / r.height) * 2 + 1);
    this.raycaster.setFromCamera(nd, this.camera);
    const hit = this.raycaster.intersectObject(this.mesh, false)[0];
    if (!hit) return;
    const col = (hit.point.x + this.extentX / 2) / this.spacing;
    const row = (this.extentY / 2 - hit.point.y) / this.spacing;
    this.placeMarker(hit.point);
    this.pickHandler(col * this.factor, row * this.factor);
  }

  private placeMarker(p: THREE.Vector3) {
    const size = Math.max(this.extentX, this.extentY) * 0.007;
    if (!this.marker) {
      const geo = new THREE.SphereGeometry(1, 12, 12);
      const mat = new THREE.MeshBasicMaterial({ color: 0xf59e0b });
      this.marker = new THREE.Mesh(geo, mat);
      // Ring around marker
      const ring = new THREE.Mesh(
        new THREE.TorusGeometry(1.6, 0.2, 8, 24),
        new THREE.MeshBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.5 })
      );
      this.marker.add(ring);
      this.scene.add(this.marker);
    }
    this.marker.scale.setScalar(size);
    this.marker.position.copy(p);
    this.marker.visible = true;
  }

  private resize() {
    const w = this.container.clientWidth, h = this.container.clientHeight;
    if (!w || !h) return;
    this.renderer.setSize(w, h);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
  }

  private animate = () => {
    this.raf = requestAnimationFrame(this.animate);
    if (this.cameraMode === "walk") {
      this.updateWalk();
    } else if (this.flythroughActive) {
      this.updateFlythrough();
    } else {
      this.controls.update();
    }
    this.renderer.render(this.scene, this.camera);
  };

  private updateFlythrough() {
    this.flythroughAngle += this.flythroughSpeed;
    const d = Math.max(this.extentX, this.extentY);
    const radX = d * 0.70;
    const radY = d * 0.60;
    const zBase = Math.max(40, d * 0.38);
    const zWave = Math.sin(this.flythroughAngle * 2.0) * (d * 0.06);
    this.camera.position.set(
      Math.sin(this.flythroughAngle) * radX,
      Math.cos(this.flythroughAngle) * radY,
      zBase + zWave
    );
    const targetZ = this.metric ? (this.zMax - this.zMin) * this.exaggeration * 0.25 : 0;
    this.controls.target.set(0, 0, targetZ);
    this.camera.lookAt(this.controls.target);
    this.controls.update();
  }

  private getGroundZ(x: number, y: number): number {
    if (!this.baseZ || this.W === 0 || this.H === 0) return 0;
    const col = Math.max(0, Math.min(this.W - 1, (x + this.extentX / 2) / this.spacing));
    const row = Math.max(0, Math.min(this.H - 1, (this.extentY / 2 - y) / this.spacing));

    const c0 = Math.floor(col);
    const r0 = Math.floor(row);
    const c1 = Math.min(this.W - 1, c0 + 1);
    const r1 = Math.min(this.H - 1, r0 + 1);

    const fc = col - c0;
    const fr = row - r0;

    const h00 = this.baseZ[r0 * this.W + c0];
    const h10 = this.baseZ[r0 * this.W + c1];
    const h01 = this.baseZ[r1 * this.W + c0];
    const h11 = this.baseZ[r1 * this.W + c1];

    const top = h00 * (1 - fc) + h10 * fc;
    const btm = h01 * (1 - fc) + h11 * fc;
    const rawZ = top * (1 - fr) + btm * fr;

    if (this.metric) {
      return (rawZ - this.zMin) * this.exaggeration;
    } else {
      const k = this.zScale * Math.max(this.extentX, this.extentY);
      return rawZ * k;
    }
  }

  private updateWalk() {
    const dt = Math.min(0.08, this.clock.getDelta());

    // Input direction
    const moveX = (this.keys["KeyD"] ? 1 : 0) - (this.keys["KeyA"] ? 1 : 0);
    const moveY = (this.keys["KeyW"] ? 1 : 0) - (this.keys["KeyS"] ? 1 : 0);
    const isSprint = !!(this.keys["ShiftLeft"] || this.keys["ShiftRight"]);

    // Movement speed (metres/sec in metric mode, proportional in relative mode)
    const baseSpeed = this.metric
      ? Math.max(3.0, Math.min(30.0, this.extentX * 0.03))
      : Math.max(0.2, Math.min(5.0, this.extentX * 0.03));
    const currentSpeed = baseSpeed * (isSprint ? 2.5 : 1.0);

    // Direction vectors on horizontal plane (yaw = 0 points +Y North, +pi/2 points +X East)
    const fwdX = Math.sin(this.yaw);
    const fwdY = Math.cos(this.yaw);
    const rightX = Math.cos(this.yaw);
    const rightY = -Math.sin(this.yaw);

    let vx = 0;
    let vy = 0;
    if (moveX !== 0 || moveY !== 0) {
      const len = Math.hypot(moveX, moveY);
      const nx = moveX / len;
      const ny = moveY / len;
      vx = (fwdX * ny + rightX * nx) * currentSpeed;
      vy = (fwdY * ny + rightY * nx) * currentSpeed;
    }

    // Velocity damping
    const damp = Math.min(1, dt * 10);
    this.walkVelocity.x += (vx - this.walkVelocity.x) * damp;
    this.walkVelocity.y += (vy - this.walkVelocity.y) * damp;

    this.camera.position.x += this.walkVelocity.x * dt;
    this.camera.position.y += this.walkVelocity.y * dt;

    // Bounds clamping
    const pad = Math.max(2, this.spacing);
    const halfX = Math.max(1, this.extentX / 2 - pad);
    const halfY = Math.max(1, this.extentY / 2 - pad);
    this.camera.position.x = Math.max(-halfX, Math.min(halfX, this.camera.position.x));
    this.camera.position.y = Math.max(-halfY, Math.min(halfY, this.camera.position.y));

    // Ground collision clamping (1.7m eye-height)
    const groundZ = this.getGroundZ(this.camera.position.x, this.camera.position.y);
    const eyeOffset = this.metric ? 1.70 * this.exaggeration : 0.05 * Math.max(this.extentX, this.extentY);
    const targetZ = groundZ + eyeOffset;

    // Smooth head vertical tracking, but hard clamp to never penetrate terrain
    this.camera.position.z += (targetZ - this.camera.position.z) * Math.min(1, dt * 14);
    if (this.camera.position.z < targetZ) {
      this.camera.position.z = targetZ;
    }

    // Look direction
    const lookX = Math.sin(this.yaw) * Math.cos(this.pitch);
    const lookY = Math.cos(this.yaw) * Math.cos(this.pitch);
    const lookZ = Math.sin(this.pitch);

    this.camera.lookAt(
      this.camera.position.x + lookX,
      this.camera.position.y + lookY,
      this.camera.position.z + lookZ
    );
    this.camera.up.set(0, 0, 1);

    if (this.onWalkStatsCb) {
      const speedKmH = Math.hypot(this.walkVelocity.x, this.walkVelocity.y) * 3.6;
      this.onWalkStatsCb(this.camera.position.z, speedKmH);
    }
  }

  setCameraMode(mode: "orbit" | "walk") {
    if (this.cameraMode === mode) return;
    this.cameraMode = mode;
    const el = this.renderer.domElement;
    if (mode === "walk") {
      this.controls.enabled = false;
      this.clock.start();

      // Compute initial yaw and pitch from current camera direction
      const dir = new THREE.Vector3();
      this.camera.getWorldDirection(dir);
      this.pitch = Math.asin(Math.max(-0.99, Math.min(0.99, dir.z)));
      this.yaw = Math.atan2(dir.x, dir.y);

      // Position camera within bounds
      const pad = Math.max(2, this.spacing);
      const halfX = Math.max(1, this.extentX / 2 - pad);
      const halfY = Math.max(1, this.extentY / 2 - pad);
      let cx = this.camera.position.x;
      let cy = this.camera.position.y;
      if (Math.abs(cx) > halfX || Math.abs(cy) > halfY) {
        cx = 0;
        cy = 0;
      }
      this.camera.position.x = cx;
      this.camera.position.y = cy;
      const groundZ = this.getGroundZ(cx, cy);
      const eyeOffset = this.metric ? 1.70 * this.exaggeration : 0.05 * Math.max(this.extentX, this.extentY);
      this.camera.position.z = groundZ + eyeOffset;

      if (document.pointerLockElement !== el) {
        el.requestPointerLock();
      }
    } else {
      if (document.pointerLockElement === el) {
        document.exitPointerLock();
      }
      const fwdX = Math.sin(this.yaw) * Math.cos(this.pitch);
      const fwdY = Math.cos(this.yaw) * Math.cos(this.pitch);
      const fwdZ = Math.sin(this.pitch);
      const lookDist = Math.max(10, Math.min(100, this.extentX * 0.1));
      this.controls.target.set(
        this.camera.position.x + fwdX * lookDist,
        this.camera.position.y + fwdY * lookDist,
        this.camera.position.z + fwdZ * lookDist
      );
      this.controls.enabled = true;
      this.controls.update();
    }
    if (this.onCameraModeChangeCb) {
      this.onCameraModeChangeCb(this.cameraMode);
    }
  }

  onCameraModeChange(fn: (mode: "orbit" | "walk") => void) {
    this.onCameraModeChangeCb = fn;
  }

  onWalkStats(fn: (eyeZ: number, speedKmH: number) => void) {
    this.onWalkStatsCb = fn;
  }

  setMeshMode(mode: "regular" | "rtin", tolerance?: number): { vertices: number; triangles: number; reductionPct: number } {
    this.meshMode = mode;
    if (tolerance !== undefined) {
      this.rtinTolerance = tolerance;
    }
    if (!this.mesh || !this.baseZ || this.W === 0 || this.H === 0) {
      return { vertices: 0, triangles: 0, reductionPct: 0 };
    }

    if (mode === "regular") {
      if (this.regularGeometry) {
        this.mesh.geometry = this.regularGeometry;
      }
      const tri = (this.regularGeometry?.getIndex()?.count ?? 0) / 3;
      return {
        vertices: this.W * this.H,
        triangles: tri,
        reductionPct: 0,
      };
    } else {
      // RTIN mode
      const rtinResult = buildRTINGeometry(
        this.baseZ,
        this.W,
        this.H,
        this.extentX,
        this.extentY,
        this.rtinTolerance,
        this.exaggeration,
        this.metric,
        this.zMin,
        this.zScale
      );
      if (this.rtinGeometry) {
        this.rtinGeometry.dispose();
      }
      this.rtinGeometry = rtinResult.geometry;
      this.mesh.geometry = this.rtinGeometry;

      return {
        vertices: rtinResult.numVertices,
        triangles: rtinResult.numTriangles,
        reductionPct: rtinResult.reductionPct,
      };
    }
  }

  getMeshStats() {
    if (!this.mesh) return null;
    const geom = this.mesh.geometry;
    const vCount = geom.attributes.position ? geom.attributes.position.count : 0;
    const tCount = geom.index ? geom.index.count / 3 : 0;
    const regTri = this.regularGeometry?.index ? this.regularGeometry.index.count / 3 : (this.W - 1) * (this.H - 1) * 2;
    const redPct = this.meshMode === "rtin" && regTri > 0
      ? Math.max(0, Math.round((1 - tCount / regTri) * 100))
      : 0;
    return {
      mode: this.meshMode,
      tolerance: this.rtinTolerance,
      vertices: vCount,
      triangles: tCount,
      reductionPct: redPct,
    };
  }

  get currentCameraMode() { return this.cameraMode; }
  get currentMeshMode() { return this.meshMode; }

  /** Build the mesh from DSM-derived heightfield. heights: row-major HxW float32 (NaN=nodata). */
  load(heights: Float32Array, meta: HeightfieldMeta, textureUrl: string, opts: LoadOptions): Promise<ViewerInfo> {
    const W = meta.width, H = meta.height;
    if (heights.length !== W * H) throw new Error(`heightfield size mismatch: ${heights.length} != ${W}×${H}`);
    this.clear();
    this.W = W; this.H = H; this.factor = meta.downsample_factor || 1;
    this.metric = opts.metric;
    this.spacing = opts.metric ? opts.spacing : 1;
    if (opts.exaggeration !== undefined) this.exaggeration = opts.exaggeration;
    this.extentX = (W - 1) * this.spacing;
    this.extentY = (H - 1) * this.spacing;

    // Build geometry
    const geom = new THREE.PlaneGeometry(this.extentX, this.extentY, W - 1, H - 1);
    const pos = geom.attributes.position as THREE.BufferAttribute;
    const base = new Float32Array(W * H);
    let dropped = 0; let mn = Infinity; let mx = -Infinity;
    for (let i = 0; i < W * H; i++) {
      const v = heights[i];
      if (Number.isFinite(v)) { base[i] = v; if (v < mn) mn = v; if (v > mx) mx = v; }
      else { base[i] = NaN; dropped++; }
    }
    this.zMin = Number.isFinite(mn) ? mn : 0;
    this.zMax = Number.isFinite(mx) ? mx : 1;
    for (let i = 0; i < W * H; i++) if (!Number.isFinite(base[i])) base[i] = this.zMin;
    this.baseZ = base;
    this.applyZ(pos);
    geom.computeVertexNormals();
    geom.computeBoundingBox();

    // Remove triangles touching nodata pixels
    if (dropped > 0) {
      const idx = geom.getIndex()!; const keep: number[] = []; const arr = idx.array as ArrayLike<number>;
      for (let t = 0; t < arr.length; t += 3) {
        const a = arr[t], b = arr[t + 1], c = arr[t + 2];
        if (Number.isFinite(heights[a]) && Number.isFinite(heights[b]) && Number.isFinite(heights[c])) keep.push(a, b, c);
      }
      geom.setIndex(keep);
    }
    // Initialize procedural facade texture for vertical building faces
    if (!this.facadeTex) {
      this.facadeTex = createFacadeTexture();
      this.facadeTex.colorSpace = THREE.SRGBColorSpace;
    }

    // Material with aerial texture + slope-aware triplanar anti-smear shader + Turbo elevation heatmap
    const tex = new THREE.TextureLoader().load(textureUrl);
    tex.colorSpace = THREE.SRGBColorSpace;
    tex.anisotropy = Math.min(16, this.renderer.capabilities.getMaxAnisotropy());
    this.aerialTex = tex;
    if (this.buildingsGroup) {
      this.buildingsGroup.traverse((child) => {
        if (child instanceof THREE.Mesh && Array.isArray(child.material)) {
          const roof = child.material[0];
          if (roof instanceof THREE.MeshStandardMaterial) {
            roof.map = tex;
            roof.needsUpdate = true;
          }
        }
      });
    }
    const mat = new THREE.MeshStandardMaterial({ map: tex, side: THREE.DoubleSide, roughness: 0.72, metalness: 0.0 });

    const facade = this.facadeTex;
    mat.onBeforeCompile = (shader) => {
      shader.uniforms.uFacadeTex = { value: facade };
      shader.uniforms.uAntiSmear = { value: this.antiSmearEnabled ? 1.0 : 0.0 };
      shader.uniforms.uVisualMode = { value: this.visualMode === "heatmap" ? 1.0 : this.visualMode === "cyber" ? 2.0 : 0.0 };
      shader.uniforms.uZMin = { value: 0.0 };
      shader.uniforms.uZMax = { value: Math.max(1.0, (this.zMax - this.zMin) * this.exaggeration) };
      shader.uniforms.uContourInterval = { value: Math.max(2.0, ((this.zMax - this.zMin) * this.exaggeration) / 25.0) };
      this.shaderUniforms = shader.uniforms;

      shader.vertexShader = shader.vertexShader.replace(
        "#include <common>",
        `#include <common>
        varying vec3 vDWWorldPos;
        varying vec3 vDWNormal;`
      );

      shader.vertexShader = shader.vertexShader.replace(
        "#include <worldpos_vertex>",
        `#include <worldpos_vertex>
        vDWWorldPos = (modelMatrix * vec4(transformed, 1.0)).xyz;
        vDWNormal = normalize((modelMatrix * vec4(normal, 0.0)).xyz);`
      );

      shader.fragmentShader = shader.fragmentShader.replace(
        "#include <common>",
        `#include <common>
        uniform sampler2D uFacadeTex;
        uniform float uAntiSmear;
        uniform float uVisualMode;
        uniform float uZMin;
        uniform float uZMax;
        uniform float uContourInterval;
        varying vec3 vDWWorldPos;
        varying vec3 vDWNormal;

        ${TURBO_GLSL}`
      );

      shader.fragmentShader = shader.fragmentShader.replace(
        "#include <map_fragment>",
        `#ifdef USE_MAP
          vec4 texColor = texture2D(map, vMapUv);

          float altSpan = max(0.01, uZMax - uZMin);
          float tNorm = clamp((vDWWorldPos.z - uZMin) / altSpan, 0.0, 1.0);

          if (uVisualMode > 1.5) {
            // 2.0 = Cyber Blueprint Mode
            vec3 cyberBase = vec3(0.03, 0.07, 0.14);
            float cMajor = computeContour(vDWWorldPos.z, uContourInterval * 2.0, 1.8);
            float cMinor = computeContour(vDWWorldPos.z, uContourInterval, 1.0);
            vec3 contourGlow = mix(vec3(0.0, 0.72, 1.0) * cMinor, vec3(0.2, 0.95, 1.0) * cMajor, 0.6);
            texColor = vec4(cyberBase + contourGlow * 0.85, 1.0);
          } else if (uVisualMode > 0.5) {
            // 1.0 = Scientific Elevation Heatmap Mode
            vec3 heat = turboColormap(tNorm);
            float contour = computeContour(vDWWorldPos.z, uContourInterval, 1.2);
            heat = mix(heat, vec3(0.06, 0.08, 0.12), contour * 0.40);
            texColor = vec4(heat, 1.0);
          } else {
            // 0.0 = Photometric Aerial Map Mode
            if (uAntiSmear > 0.5) {
              float slopeCos = abs(vDWNormal.z);
              float wallWeight = smoothstep(0.72, 0.48, slopeCos);
              if (wallWeight > 0.01) {
                vec2 fUv = vec2(vDWWorldPos.x * 0.12 + vDWWorldPos.y * 0.12, vDWWorldPos.z * 0.12);
                vec4 fColor = texture2D(uFacadeTex, fUv);
                texColor = mix(texColor, fColor, wallWeight * 0.65);
              }
            }
          }
          diffuseColor *= texColor;
        #endif`
      );
    };

    this.mesh = new THREE.Mesh(geom, mat);
    this.regularGeometry = geom;
    this.meshMode = "regular";
    this.cameraMode = "orbit";
    this.controls.enabled = true;
    this.scene.add(this.mesh);

    // Build architectural ground pedestal
    if (this.pedestalMesh) {
      this.scene.remove(this.pedestalMesh);
      this.pedestalMesh.geometry.dispose();
      (this.pedestalMesh.material as THREE.Material).dispose();
      this.pedestalMesh = null;
    }
    const pedestalGeom = buildPedestalGeometry(
      this.baseZ,
      W,
      H,
      this.extentX,
      this.extentY,
      this.spacing,
      this.zMin,
      this.exaggeration,
      this.metric,
      this.zScale,
      { zBottom: -12.0 }
    );
    const pedestalMat = new THREE.MeshStandardMaterial({
      color: 0x0f172a,
      roughness: 0.85,
      metalness: 0.15,
      side: THREE.DoubleSide,
    });
    this.pedestalMesh = new THREE.Mesh(pedestalGeom, pedestalMat);
    this.scene.add(this.pedestalMesh);

    // Subtle grid at ground level (metric mode only)
    if (opts.metric) {
      const gridSize = Math.max(this.extentX, this.extentY) * 1.2;
      const divisions = Math.min(20, Math.floor(gridSize / (opts.spacing * 10)));
      this.gridHelper = new THREE.GridHelper(gridSize, Math.max(4, divisions), 0x1e3a5f, 0x1e3a5f);
      (this.gridHelper.material as THREE.Material).opacity = 0.25;
      (this.gridHelper.material as THREE.Material).transparent = true;
      this.gridHelper.rotation.x = Math.PI / 2; // align to XY plane
      this.gridHelper.position.z = -0.5;
      this.scene.add(this.gridHelper);
    }

    this.updateHUD(opts.hud, meta);
    this.resetCamera();
    const tri = (geom.getIndex()!.count / 3) | 0;
    return Promise.resolve({ vertices: W * H, triangles: tri, nodataDropped: dropped, extent: [this.extentX, this.extentY] });
  }

  updateHUD(hud: LoadOptions["hud"], meta?: HeightfieldMeta) {
    // Top-left: state pill + tier pill
    this.hudTL.innerHTML = `
      <div class="hud-pill hud-state">${hud.state}</div>
      <div class="hud-pill hud-tier">${hud.tier}</div>
    `;

    // Top-right: quality + exaggeration (metric only)
    const exagLine = this.metric
      ? `<div class="hud-pill hud-exag">⚡ VERT. EXAG ${this.exaggeration.toFixed(1)}×${Math.abs(this.exaggeration - 1) < 1e-6 ? " (true scale)" : " — VISUAL ONLY"}</div>`
      : `<div class="hud-pill hud-tier">display scale ${this.zScale.toFixed(2)} — not a measurement</div>`;
    this.hudTR.innerHTML = `
      <div class="hud-pill hud-quality-${hud.quality}">QUALITY: ${hud.quality}</div>
      ${exagLine}
    `;

    // Bottom-left: layer + z-range
    const zRangeLine = meta && this.metric
      ? `<div class="hud-pill hud-zrange">z ${meta.min.toFixed(1)} – ${meta.max.toFixed(1)} m</div>`
      : "";
    this.hudBL.innerHTML = `
      <div class="hud-pill hud-layer">${hud.layer}</div>
      ${zRangeLine}
    `;

    // Bottom-right: scale bar + north indicator
    this.hudBR.innerHTML = "";
    if (this.metric && this.extentX > 0) {
      // Approximate scale bar: target ~15% of viewport width → compute real distance
      const targetFrac = 0.15;
      const sceneWidth = this.extentX;
      const rawDist = sceneWidth * targetFrac;
      const niceValues = [0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000];
      const niceDist = niceValues.find(v => v >= rawDist) ?? rawDist;
      const barWidthPx = Math.round((niceDist / sceneWidth) * 200);
      const label = niceDist >= 1000 ? `${(niceDist / 1000).toFixed(niceDist >= 1000 ? 0 : 1)} km` : `${niceDist} m`;
      this.hudBR.innerHTML += `
        <div class="hud-scalebar">
          <div class="hud-scalebar-label">${label}</div>
          <div class="hud-scalebar-bar" style="width:${barWidthPx}px"></div>
        </div>
      `;
    }
    if (hud.showNorth) {
      this.hudBR.innerHTML += `
        <div class="hud-north">
          <div class="hud-north-arrow">↑</div>
          <div class="hud-north-label">N</div>
        </div>
      `;
    }
  }

  private applyZ(pos?: THREE.BufferAttribute) {
    if (this.meshMode === "rtin" && this.baseZ) {
      this.setMeshMode("rtin", this.rtinTolerance);
      if (this.marker) this.marker.visible = false;
      return;
    }
    if (!this.mesh && !pos) return;
    const p = pos ?? (this.mesh!.geometry.attributes.position as THREE.BufferAttribute);
    const z = this.baseZ!;
    if (this.metric) {
      for (let i = 0; i < z.length; i++) p.setZ(i, (z[i] - this.zMin) * this.exaggeration);
    } else {
      const k = this.zScale * Math.max(this.extentX, this.extentY);
      for (let i = 0; i < z.length; i++) p.setZ(i, z[i] * k);
    }
    p.needsUpdate = true;
    if (this.mesh) { this.mesh.geometry.computeVertexNormals(); this.mesh.geometry.computeBoundingBox(); }
    if (this.marker) this.marker.visible = false;
  }

  setZScale(s: number) {
    this.zScale = s;
    if (!this.metric) {
      this.applyZ();
      // Update exag pill in TR
      const pill = this.hudTR.querySelector(".hud-tier") as HTMLElement;
      if (pill) pill.textContent = `display scale ${this.zScale.toFixed(2)} — not a measurement`;
    }
  }

  setExaggeration(e: number) {
    this.exaggeration = e;
    if (this.metric) {
      this.applyZ();
      const pill = this.hudTR.querySelector(".hud-exag") as HTMLElement;
      if (pill) pill.textContent = `⚡ VERT. EXAG ${e.toFixed(1)}×${Math.abs(e - 1) < 1e-6 ? " (true scale)" : " — VISUAL ONLY"}`;
    }
    if (this.shaderUniforms) {
      if (this.shaderUniforms.uZMax) {
        this.shaderUniforms.uZMax.value = Math.max(1.0, (this.zMax - this.zMin) * this.exaggeration);
      }
      if (this.shaderUniforms.uContourInterval) {
        this.shaderUniforms.uContourInterval.value = Math.max(2.0, ((this.zMax - this.zMin) * this.exaggeration) / 25.0);
      }
    }
    if (this.pedestalMesh && this.baseZ) {
      this.pedestalMesh.geometry.dispose();
      this.pedestalMesh.geometry = buildPedestalGeometry(
        this.baseZ,
        this.W,
        this.H,
        this.extentX,
        this.extentY,
        this.spacing,
        this.zMin,
        this.exaggeration,
        this.metric,
        this.zScale,
        { zBottom: -12.0 }
      );
    }
    if (this.buildingsData) {
      this.loadBuildings(this.buildingsData, this.cityMode);
    }
  }

  setFlythrough(active: boolean) {
    this.flythroughActive = active;
    if (active) {
      if (this.cameraMode === "walk") {
        this.setCameraMode("orbit");
      }
      this.flythroughAngle = 0;
    }
    if (this.onFlythroughToggleCb) {
      this.onFlythroughToggleCb(active);
    }
  }

  onFlythroughToggle(cb: (active: boolean) => void) {
    this.onFlythroughToggleCb = cb;
  }

  get isFlythrough() { return this.flythroughActive; }

  setShaderMode(mode: ShaderVisualMode) {
    this.visualMode = mode;
    if (this.shaderUniforms && this.shaderUniforms.uVisualMode) {
      this.shaderUniforms.uVisualMode.value = mode === "heatmap" ? 1.0 : mode === "cyber" ? 2.0 : 0.0;
    }
  }

  get currentShaderMode() { return this.visualMode; }

  setPresetView(preset: "nadir" | "oblique" | "horizon") {
    if (this.flythroughActive) this.setFlythrough(false);
    if (this.cameraMode === "walk") this.setCameraMode("orbit");
    const d = Math.max(this.extentX, this.extentY);
    const targetZ = this.metric ? (this.zMax - this.zMin) * this.exaggeration * 0.25 : 0;
    this.controls.target.set(0, 0, targetZ);

    if (preset === "nadir") {
      this.camera.position.set(0, 0, d * 1.35);
      this.camera.up.set(0, 1, 0);
    } else if (preset === "oblique") {
      this.camera.position.set(0, -d * 0.65, d * 0.70);
      this.camera.up.set(0, 0, 1);
    } else if (preset === "horizon") {
      this.camera.position.set(-d * 0.60, -d * 0.60, Math.max(30, d * 0.22));
      this.camera.up.set(0, 0, 1);
    }
    this.controls.update();
  }

  setAntiSmear(enabled: boolean) {
    this.antiSmearEnabled = enabled;
    if (this.shaderUniforms && this.shaderUniforms.uAntiSmear) {
      this.shaderUniforms.uAntiSmear.value = enabled ? 1.0 : 0.0;
    }
  }

  loadBuildings(data: LoD1Data | null, cityMode = false) {
    this.buildingsData = data;
    this.cityMode = cityMode;
    if (this.buildingsGroup) {
      this.scene.remove(this.buildingsGroup);
      this.buildingsGroup = null;
    }
    if (!data) return;
    this.buildingsGroup = buildLoD1BuildingGroup(data, {
      exaggeration: this.exaggeration,
      metric: this.metric,
      zMin: this.zMin,
      extentX: this.extentX,
      extentY: this.extentY,
      dataExtentX: data.extent?.[0],
      dataExtentY: data.extent?.[1],
      cityMode,
      facadeTexture: this.facadeTex || undefined,
      aerialTexture: this.aerialTex || undefined,
    });
    this.buildingsGroup.visible = this.lod1Visible;
    this.scene.add(this.buildingsGroup);
  }

  setBuildingsVisible(visible: boolean) {
    this.lod1Visible = visible;
    if (this.buildingsGroup) {
      this.buildingsGroup.visible = visible;
    }
  }

  get isAntiSmear() { return this.antiSmearEnabled; }
  get isBuildingsVisible() { return this.lod1Visible; }
  get buildingCount() { return this.buildingsData ? this.buildingsData.count : 0; }

  setWireframe(on: boolean) {
    if (this.mesh) (this.mesh.material as THREE.MeshStandardMaterial).wireframe = on;
  }

  get isMetric() { return this.metric; }

  resetCamera() {
    this.setPresetView("oblique");
  }

  clear() {
    if (this.mesh) {
      this.scene.remove(this.mesh);
      this.mesh.geometry.dispose();
      ((this.mesh.material as THREE.MeshStandardMaterial).map)?.dispose();
      (this.mesh.material as THREE.Material).dispose();
      this.mesh = null;
    }
    if (this.rtinGeometry) {
      this.rtinGeometry.dispose();
      this.rtinGeometry = null;
    }
    this.regularGeometry = null;
    if (this.pedestalMesh) {
      this.scene.remove(this.pedestalMesh);
      this.pedestalMesh.geometry.dispose();
      (this.pedestalMesh.material as THREE.Material).dispose();
      this.pedestalMesh = null;
    }
    if (this.buildingsGroup) {
      this.scene.remove(this.buildingsGroup);
      this.buildingsGroup = null;
    }
    this.buildingsData = null;
    if (this.gridHelper) { this.scene.remove(this.gridHelper); this.gridHelper = null; }
    if (this.marker) this.marker.visible = false;
    this.hudTL.innerHTML = "";
    this.hudTR.innerHTML = "";
    this.hudBL.innerHTML = "";
    this.hudBR.innerHTML = "";
  }

  dispose() { cancelAnimationFrame(this.raf); this.clear(); this.renderer.dispose(); }
}
