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

  // Smooth Camera Flight Transition
  private cameraTransition: {
    startPos: THREE.Vector3;
    endPos: THREE.Vector3;
    startTarget: THREE.Vector3;
    endTarget: THREE.Vector3;
    startTime: number;
    duration: number;
    onComplete?: () => void;
  } | null = null;

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

    // Camera — standard Z-up cartographic coordinate system
    this.camera = new THREE.PerspectiveCamera(45, container.clientWidth / container.clientHeight, 0.1, 200000);
    this.camera.up.set(0, 0, 1);

    // Controls: intuitive standard GIS / 3D navigation
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.08;
    this.controls.screenSpacePanning = true;
    this.controls.minDistance = 0.5;
    this.controls.maxDistance = 150000;
    this.controls.minPolarAngle = 0.01;
    // Constrain camera strictly to horizon and above (never tilt below ground plane / horizon)
    this.controls.maxPolarAngle = Math.PI / 2 - 0.02; // ~88.8 deg, prevents going below horizon
    this.controls.mouseButtons = {
      LEFT: THREE.MOUSE.ROTATE,
      MIDDLE: THREE.MOUSE.DOLLY,
      RIGHT: THREE.MOUSE.PAN,
    };
    this.controls.touches = {
      ONE: THREE.TOUCH.ROTATE,
      TWO: THREE.TOUCH.DOLLY_PAN,
    };
    this.controls.addEventListener("start", () => {
      if (this.flythroughActive) {
        this.setFlythrough(false);
      }
      if (this.cameraTransition) {
        this.cameraTransition = null;
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
      if (moved < 4 && e.button === 0) this.pick(e);
    });
    el.addEventListener("dblclick", (e) => {
      this.focusOnPoint(e);
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
      } else {
        // Intuitive keyboard shortcuts in Orbit mode
        if (e.code === "KeyN") {
          this.alignNorth();
        } else if (e.code === "Equal" || e.code === "NumpadAdd") {
          this.zoomBy(0.75);
        } else if (e.code === "Minus" || e.code === "NumpadSubtract") {
          this.zoomBy(1.33);
        } else if (e.code === "KeyF" && !e.ctrlKey && !e.metaKey) {
          this.resetCamera();
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

  private buildingPickHandler: ((id: number) => void) | null = null;
  private highlighted: { mesh: THREE.Mesh; mats: THREE.Material | THREE.Material[] } | null = null;
  onBuildingPick(h: ((id: number) => void) | null) { this.buildingPickHandler = h; }

  private buildingMesh(id: number): THREE.Mesh | null {
    let found: THREE.Mesh | null = null;
    this.buildingsGroup?.traverse((o) => { if (!found && o instanceof THREE.Mesh && o.userData.buildingId === id) found = o; });
    return found;
  }

  /** Highlight one LoD-1 block (amber) and restore the previous one. */
  highlightBuilding(id: number | null) {
    if (this.highlighted) { this.highlighted.mesh.material = this.highlighted.mats; this.highlighted = null; }
    if (id === null) return;
    const m = this.buildingMesh(id);
    if (!m) return;
    const hi = new THREE.MeshStandardMaterial({ color: 0xf59e0b, emissive: 0x7c2d12, emissiveIntensity: 0.6, roughness: 0.5 });
    this.highlighted = { mesh: m, mats: m.material };
    m.material = [hi, hi];
  }

  /** Orbit camera to a building (scene coordinates are metres, centred on the scene). */
  focusBuilding(id: number) {
    const m = this.buildingMesh(id);
    if (!m) return;
    if (this.flythroughActive) this.setFlythrough(false);
    if (this.cameraMode === "walk") this.setCameraMode("orbit");
    const box = new THREE.Box3().setFromObject(m);
    const c = box.getCenter(new THREE.Vector3());
    const size = Math.max(40, box.getSize(new THREE.Vector3()).length() * 3);
    this.controls.target.copy(c);
    this.camera.up.set(0, 0, 1);
    this.camera.position.set(c.x + size * 0.6, c.y - size * 0.8, c.z + size * 0.7);
    this.controls.update();
    this.highlightBuilding(id);
  }

  private pick(e: PointerEvent) {
    if (this.cameraMode === "walk" || !this.mesh || !this.pickHandler) return;
    const r = this.renderer.domElement.getBoundingClientRect();
    const nd = new THREE.Vector2(((e.clientX - r.left) / r.width) * 2 - 1, -((e.clientY - r.top) / r.height) * 2 + 1);
    this.raycaster.setFromCamera(nd, this.camera);
    if (this.buildingsGroup?.visible && this.buildingPickHandler) {
      const bh = this.raycaster.intersectObject(this.buildingsGroup, true)[0];
      if (bh && bh.object.userData.buildingId !== undefined) {
        this.highlightBuilding(bh.object.userData.buildingId as number);
        this.buildingPickHandler(bh.object.userData.buildingId as number);
      }
    }
    const hit = this.raycaster.intersectObject(this.mesh, false)[0];
    if (!hit) return;
    // vertex i sits at the centre of source block i -> continuous source coordinate (i + 0.5) * factor
    const col = (hit.point.x + this.extentX / 2) / this.spacing;
    const row = (this.extentY / 2 - hit.point.y) / this.spacing;
    this.placeMarker(hit.point);
    this.pickHandler((col + 0.5) * this.factor, (row + 0.5) * this.factor);
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
    if (this.cameraTransition) {
      this.updateCameraTransition();
    } else if (this.cameraMode === "walk") {
      this.updateWalk();
    } else if (this.flythroughActive) {
      this.updateFlythrough();
    } else {
      this.controls.update();
      // Enforce camera stays above ground and never drops below the horizon
      if (this.mesh && this.baseZ) {
        const minZ = Math.max(
          this.controls.target.z,
          this.getGroundZ(this.camera.position.x, this.camera.position.y) + 0.8
        );
        if (this.camera.position.z < minZ) {
          this.camera.position.z = minZ;
        }
      }
    }
    if ((this.frame++ & 7) === 0) this.updateDynamicHud();
    this.renderer.render(this.scene, this.camera);
  };

  /**
   * Smoothly animates camera position and orbit target to a new location.
   */
  flyTo(
    endPos: THREE.Vector3,
    endTarget: THREE.Vector3,
    durationMs = 600,
    onComplete?: () => void
  ) {
    if (this.cameraMode === "walk") {
      this.setCameraMode("orbit");
    }
    if (this.flythroughActive) {
      this.setFlythrough(false);
    }
    this.cameraTransition = {
      startPos: this.camera.position.clone(),
      endPos: endPos.clone(),
      startTarget: this.controls.target.clone(),
      endTarget: endTarget.clone(),
      startTime: performance.now(),
      duration: Math.max(150, durationMs),
      onComplete,
    };
  }

  private updateCameraTransition() {
    if (!this.cameraTransition) return;
    const now = performance.now();
    const elapsed = now - this.cameraTransition.startTime;
    const progress = Math.min(1.0, elapsed / this.cameraTransition.duration);
    // Smooth cubic ease-in-out curve
    const t = progress < 0.5
      ? 4 * progress * progress * progress
      : 1 - Math.pow(-2 * progress + 2, 3) / 2;

    this.camera.position.lerpVectors(this.cameraTransition.startPos, this.cameraTransition.endPos, t);
    this.controls.target.lerpVectors(this.cameraTransition.startTarget, this.cameraTransition.endTarget, t);

    // Keep camera above horizon of target and above terrain surface during flight
    if (this.mesh && this.baseZ) {
      const minZ = Math.max(
        this.controls.target.z + 0.5,
        this.getGroundZ(this.camera.position.x, this.camera.position.y) + 1.0
      );
      if (this.camera.position.z < minZ) {
        this.camera.position.z = minZ;
      }
    }

    this.controls.update();

    if (progress >= 1.0) {
      const cb = this.cameraTransition.onComplete;
      this.cameraTransition = null;
      if (cb) cb();
    }
  }

  /**
   * Focuses and centers the camera orbit pivot directly on the clicked building or terrain point.
   */
  focusOnPoint(e: MouseEvent) {
    if (this.cameraMode === "walk" || !this.mesh) return;
    const r = this.renderer.domElement.getBoundingClientRect();
    const nd = new THREE.Vector2(((e.clientX - r.left) / r.width) * 2 - 1, -((e.clientY - r.top) / r.height) * 2 + 1);
    this.raycaster.setFromCamera(nd, this.camera);
    const objectsToIntersect: THREE.Object3D[] = [this.mesh];
    if (this.buildingsGroup && this.lod1Visible) {
      objectsToIntersect.push(this.buildingsGroup);
    }
    const hits = this.raycaster.intersectObjects(objectsToIntersect, true);
    if (!hits || hits.length === 0) return;
    const hitPoint = hits[0].point;

    // Glide target to hit point, preserve current viewing angle
    const currentOffset = this.camera.position.clone().sub(this.controls.target);
    const currentDist = currentOffset.length();
    const maxDim = Math.max(this.extentX, this.extentY);
    const comfortableDist = Math.max(25, maxDim * 0.30);
    if (currentDist > comfortableDist * 1.5) {
      currentOffset.setLength(comfortableDist);
    }
    // Prevent camera offset from dipping below horizon of the focus target
    if (currentOffset.z < 2.0) {
      currentOffset.z = 2.0;
    }
    const newPos = hitPoint.clone().add(currentOffset);
    const groundZ = this.getGroundZ(newPos.x, newPos.y);
    if (newPos.z < groundZ + 2) {
      newPos.z = groundZ + 2;
    }

    this.placeMarker(hitPoint);
    this.flyTo(newPos, hitPoint, 550);
  }

  /**
   * Smoothly zooms camera in or out by a scaling factor.
   */
  zoomBy(factor: number) {
    if (this.cameraMode === "walk") return;
    const offset = this.camera.position.clone().sub(this.controls.target);
    const currentDist = offset.length();
    const targetDist = Math.max(this.controls.minDistance, Math.min(this.controls.maxDistance, currentDist * factor));
    offset.setLength(targetDist);
    const newPos = this.controls.target.clone().add(offset);
    this.flyTo(newPos, this.controls.target, 280);
  }

  /**
   * Smoothly aligns camera heading to due North (+Y) while maintaining current tilt and distance.
   */
  alignNorth() {
    if (this.cameraMode === "walk") {
      this.yaw = 0;
      return;
    }
    const offset = this.camera.position.clone().sub(this.controls.target);
    const horizDist = Math.max(1.0, Math.hypot(offset.x, offset.y));
    const newPos = new THREE.Vector3(
      this.controls.target.x,
      this.controls.target.y - horizDist,
      Math.max(this.controls.target.z + 1.0, this.camera.position.z)
    );
    this.camera.up.set(0, 0, 1);
    this.flyTo(newPos, this.controls.target, 500);
  }

  private frame = 0;

  /** North arrow follows the camera heading; the scale bar is measured at the orbit target (metric mode only). */
  private updateDynamicHud() {
    const dir = new THREE.Vector3();
    this.camera.getWorldDirection(dir);
    const horiz = Math.hypot(dir.x, dir.y) > 0.15 ? dir : new THREE.Vector3(0, 1, 0).applyQuaternion(this.camera.quaternion);
    const heading = Math.atan2(horiz.x, horiz.y); // 0 = looking north (+Y)

    const arrow = this.hudBR.querySelector(".hud-north") as HTMLElement | null;
    if (arrow) {
      arrow.style.transform = `rotate(${(-heading * 180) / Math.PI}deg)`;
    }
    const navNeedle = document.getElementById("nav-compass-needle") as HTMLElement | null;
    if (navNeedle) {
      navNeedle.style.transform = `rotate(${(-heading * 180) / Math.PI}deg)`;
    }

    const bar = this.hudBR.querySelector(".hud-scalebar-bar") as HTMLElement | null;
    const label = this.hudBR.querySelector(".hud-scalebar-label") as HTMLElement | null;
    if (!bar || !label || !this.metric) return;
    const h = this.renderer.domElement.clientHeight || 1;
    const dist = this.cameraMode === "walk" ? 20 : Math.max(0.5, this.camera.position.distanceTo(this.controls.target));
    const mPerPx = (2 * dist * Math.tan(THREE.MathUtils.degToRad(this.camera.fov) / 2)) / h;
    const target = mPerPx * 110; // aim for a ~110 px bar
    const nice = [0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000];
    const d = nice.find((v) => v >= target * 0.6) ?? nice[nice.length - 1];
    bar.style.width = `${Math.max(12, Math.min(220, d / mPerPx)).toFixed(0)}px`;
    label.textContent = d >= 1000 ? `${d / 1000} km` : `${d} m`;
  }

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

    // move axis by axis so the walker slides along building walls instead of passing through them
    // (a walker that starts inside a footprint may always move, so it can never get stuck)
    const free = this.insideBuilding(this.camera.position.x, this.camera.position.y);
    const nx = this.camera.position.x + this.walkVelocity.x * dt;
    if (free || !this.insideBuilding(nx, this.camera.position.y)) this.camera.position.x = nx;
    else this.walkVelocity.x = 0;
    const ny = this.camera.position.y + this.walkVelocity.y * dt;
    if (free || !this.insideBuilding(this.camera.position.x, ny)) this.camera.position.y = ny;
    else this.walkVelocity.y = 0;

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
      // tolerance is given in displayed units (metres, or scene units for relative layers) -> raw height units
      const rawTol = this.metric ? this.rtinTolerance : this.rtinTolerance / Math.max(1e-6, this.zScale * Math.max(this.extentX, this.extentY));
      const rtinResult = buildRTINGeometry(
        this.baseZ,
        this.W,
        this.H,
        this.extentX,
        this.extentY,
        rawTol,
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
      // sized every few frames by updateDynamicHud() from the camera distance (valid at the view centre)
      this.hudBR.innerHTML += `
        <div class="hud-scalebar" title="Horizontal scale at the view centre (perspective view: approximate elsewhere)">
          <div class="hud-scalebar-label">—</div>
          <div class="hud-scalebar-bar" style="width:0px"></div>
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
      this.rebuildPedestal();
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
    this.rebuildPedestal();
    if (this.buildingsData) {
      this.loadBuildings(this.buildingsData, this.cityMode);
    }
  }

  private rebuildPedestal() {
    if (!this.pedestalMesh || !this.baseZ) return;
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
    const target = new THREE.Vector3(0, 0, targetZ);
    const endPos = new THREE.Vector3();

    this.camera.up.set(0, 0, 1);

    if (preset === "nadir") {
      // Offset slightly south (-Y) so North (+Y) is straight up on screen, avoiding zenith singularity
      endPos.set(0, -Math.max(1, d * 0.001), d * 1.35);
    } else if (preset === "oblique") {
      endPos.set(0, -d * 0.65, d * 0.70);
    } else if (preset === "horizon") {
      // Perspective positioned comfortably above the horizon line
      endPos.set(-d * 0.60, -d * 0.60, Math.max(targetZ + 15, d * 0.22));
    }
    this.flyTo(endPos, target, 650);
  }

  setAntiSmear(enabled: boolean) {
    this.antiSmearEnabled = enabled;
    if (this.shaderUniforms && this.shaderUniforms.uAntiSmear) {
      this.shaderUniforms.uAntiSmear.value = enabled ? 1.0 : 0.0;
    }
  }

  loadBuildings(data: LoD1Data | null, cityMode = false) {
    this.highlighted = null;
    this.buildingsData = data;
    this.footprints = [];
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
    this.buildFootprints(data);
  }

  /** True when (x, y) is inside a visible LoD-1 footprint whose roof is above the walker's eye. */
  private insideBuilding(x: number, y: number): boolean {
    if (!this.lod1Visible || !this.buildingsGroup || !this.footprints.length) return false;
    const r = 0.4; // walker radius (m)
    const eye = this.camera.position.z;
    for (const f of this.footprints) {
      if (x < f.minX - r || x > f.maxX + r || y < f.minY - r || y > f.maxY + r || eye > f.top) continue;
      let inside = false;
      const p = f.pts;
      for (let i = 0, j = p.length - 1; i < p.length; j = i++) {
        if ((p[i][1] > y) !== (p[j][1] > y) && x < ((p[j][0] - p[i][0]) * (y - p[i][1])) / (p[j][1] - p[i][1] + 1e-12) + p[i][0]) inside = !inside;
      }
      if (inside) return true;
    }
    return false;
  }

  private footprints: { minX: number; maxX: number; minY: number; maxY: number; top: number; pts: [number, number][] }[] = [];

  private buildFootprints(data: LoD1Data | null) {
    this.footprints = [];
    if (!data?.buildings || !this.metric) return;
    for (const b of data.buildings) {
      if (!b.coords || b.coords.length < 3) continue;
      const xs = b.coords.map((c) => c[0]), ys = b.coords.map((c) => c[1]);
      const base = Math.max(0, (b.base_elev_m - this.zMin) * this.exaggeration);
      this.footprints.push({ minX: Math.min(...xs), maxX: Math.max(...xs), minY: Math.min(...ys), maxY: Math.max(...ys), top: base + Math.max(3, b.height_m * this.exaggeration), pts: b.coords });
    }
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
    this.footprints = [];
    if (this.gridHelper) { this.scene.remove(this.gridHelper); this.gridHelper = null; }
    if (this.marker) this.marker.visible = false;
    this.hudTL.innerHTML = "";
    this.hudTR.innerHTML = "";
    this.hudBL.innerHTML = "";
    this.hudBR.innerHTML = "";
  }

  dispose() { cancelAnimationFrame(this.raf); this.clear(); this.renderer.dispose(); }
}
