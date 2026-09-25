/**
 * LoD-1 (Level of Detail 1) 3D Vector Building Renderer.
 *
 * Renders extruded 3D polygon prisms with clean 90-degree vertical walls
 * and flat roofs. Each building is ONE solid vertical block from ground
 * to roof — no stacking, no tier layers, no merged blobs.
 *
 * Visual design: buildings are height-categorised (low/mid/high-rise) and
 * colour-coded with distinct facade textures, roof materials, and subtle
 * edge-line overlays so they read clearly against the aerial texture ground.
 *
 * Performance: materials are shared across buildings of the same class.
 */
import * as THREE from "three";

export interface BuildingInstance {
  id: number;
  height_m: number;
  base_elev_m: number;
  area_m2: number;
  coords: [number, number][];
}

export interface LoD1Data {
  buildings: BuildingInstance[];
  count: number;
  min_height_m: number;
  gsd_m: number;
  extent: [number, number];
}

// ─── Facade texture factory ──────────────────────────────────────────────────

/**
 * Creates a clean, minimalist architectural facade texture (Apple Maps / ArchViz exhibition style).
 * Smooth, elegant matte architectural surfaces with subtle horizontal floor joints and soft ambient shading.
 */
export function createFacadeTexture(
  variant: "warm" | "cool" | "steel" = "cool"
): THREE.CanvasTexture {
  const W = 512, H = 512;
  const canvas = document.createElement("canvas");
  canvas.width = W; canvas.height = H;
  const ctx = canvas.getContext("2d")!;

  // Clean, bright, museum-grade architectural palettes (Apple Maps 3D / ArchViz style)
  const palettes = {
    warm:  {
      base: "#faf8f4",        // Clean warm architectural limestone
      slab: "#e8e2d5",        // Delicate storey joint line
      bayJoint: "rgba(0, 0, 0, 0.025)",
      ambientBot: "rgba(30, 20, 10, 0.06)",
      ambientTop: "rgba(255, 255, 255, 0.08)",
      windowTint: "rgba(90, 110, 130, 0.06)",
    },
    cool:  {
      base: "#f3f6f9",        // Clean modern precast alabaster
      slab: "#dbe3ec",        // Delicate storey joint line
      bayJoint: "rgba(0, 0, 0, 0.025)",
      ambientBot: "rgba(15, 23, 42, 0.06)",
      ambientTop: "rgba(255, 255, 255, 0.08)",
      windowTint: "rgba(70, 100, 135, 0.06)",
    },
    steel: {
      base: "#e9edf2",        // Contemporary architectural zinc / light slate
      slab: "#cbd5e1",        // Delicate storey joint line
      bayJoint: "rgba(0, 0, 0, 0.03)",
      ambientBot: "rgba(20, 30, 45, 0.07)",
      ambientTop: "rgba(255, 255, 255, 0.07)",
      windowTint: "rgba(60, 90, 120, 0.07)",
    },
  };
  const p = palettes[variant];

  // Base clean architectural stone finish
  ctx.fillStyle = p.base;
  ctx.fillRect(0, 0, W, H);

  // Soft vertical ambient gradient: grounding at foundation, luminous at roofline
  const grad = ctx.createLinearGradient(0, H, 0, 0);
  grad.addColorStop(0, p.ambientBot);
  grad.addColorStop(0.5, "transparent");
  grad.addColorStop(1, p.ambientTop);
  ctx.fillStyle = grad;
  ctx.fillRect(0, 0, W, H);

  // 1 UV repeat = 1 storey (3.5 m).
  // Single subtle horizontal storey slab reveal at the bottom (y = H - 3)
  ctx.fillStyle = p.slab;
  ctx.fillRect(0, H - 3, W, 2);
  ctx.fillStyle = "rgba(0, 0, 0, 0.03)";
  ctx.fillRect(0, H - 1, W, 1);

  // 4 clean architectural bays across width (every 128px = ~1m)
  for (let x = 0; x < W; x += 128) {
    ctx.fillStyle = p.bayJoint;
    ctx.fillRect(x, 0, 1, H);

    // Subtle, clean window opening in each bay
    ctx.fillStyle = p.windowTint;
    ctx.fillRect(x + 16, 120, 96, 260);
    // Micro-frame outline
    ctx.strokeStyle = "rgba(0, 0, 0, 0.04)";
    ctx.lineWidth = 1;
    ctx.strokeRect(x + 16, 120, 96, 260);
  }

  const tex = new THREE.CanvasTexture(canvas);
  tex.wrapS = THREE.RepeatWrapping;
  tex.wrapT = THREE.RepeatWrapping;
  tex.colorSpace = THREE.SRGBColorSpace;
  tex.anisotropy = 16;
  tex.generateMipmaps = true;
  tex.minFilter = THREE.LinearMipmapLinearFilter;
  tex.magFilter = THREE.LinearFilter;
  return tex;
}

// Pre-build three facade variants (shared across all buildings for performance)
let _facadeWarm: THREE.CanvasTexture | null = null;
let _facadeCool: THREE.CanvasTexture | null = null;
let _facadeSteel: THREE.CanvasTexture | null = null;

function getFacade(variant: "warm" | "cool" | "steel"): THREE.CanvasTexture {
  if (variant === "warm")  { if (!_facadeWarm)  _facadeWarm  = createFacadeTexture("warm");  return _facadeWarm;  }
  if (variant === "cool")  { if (!_facadeCool)  _facadeCool  = createFacadeTexture("cool");  return _facadeCool;  }
  if (!_facadeSteel) _facadeSteel = createFacadeTexture("steel"); return _facadeSteel;
}

// ─── Public API ──────────────────────────────────────────────────────────────

export interface LoD1Options {
  exaggeration: number;
  metric: boolean;
  zMin: number;
  extentX?: number;
  extentY?: number;
  dataExtentX?: number;  // from buildings JSON extent[0]
  dataExtentY?: number;  // from buildings JSON extent[1]
  cityMode?: boolean;    // if true, buildings sit at Z=0 (nDSM ground)
  facadeTexture?: THREE.Texture;
  aerialTexture?: THREE.Texture;
}

/**
 * Builds a THREE.Group of extruded 3D building blocks from LoD-1 JSON data.
 *
 * Each building is rendered as ONE clean vertical block:
 *   - Flat roof with projected aerial satellite texture or architectural gravel (material 0)
 *   - Clean vertical walls with realistic procedural architectural facades (material 1)
 *   - No stacking, no layers, no merged blobs
 */
export function buildLoD1BuildingGroup(
  data: LoD1Data,
  options: LoD1Options
): THREE.Group {
  const group = new THREE.Group();
  group.name = "lod1-buildings";

  if (!data.buildings || !data.buildings.length) return group;

  // UV extent for roof aerial texture projection
  const uvExtentX = Math.max(1, options.dataExtentX || options.extentX || 1000);
  const uvExtentY = Math.max(1, options.dataExtentY || options.extentY || 1000);

  // Roof material (Material Index 0 in Three.js ExtrudeGeometry) —
  // uses real aerial satellite photograph when available, else elegant roof tone
  const roofMat = options.aerialTexture
    ? new THREE.MeshStandardMaterial({
        map: options.aerialTexture,
        roughness: 0.60,
        metalness: 0.02,
        color: 0xffffff,
      })
    : new THREE.MeshStandardMaterial({
        color: 0xd8e2eb,
        roughness: 0.65,
        metalness: 0.05,
      });

  // Shared wall materials (Material Index 1 in Three.js ExtrudeGeometry)
  const wallMats: Record<string, THREE.MeshStandardMaterial> = {
    warm: new THREE.MeshStandardMaterial({
      map: getFacade("warm"),
      roughness: 0.50,
      metalness: 0.04,
      color: 0xffffff,
    }),
    cool: new THREE.MeshStandardMaterial({
      map: getFacade("cool"),
      roughness: 0.45,
      metalness: 0.05,
      color: 0xffffff,
    }),
    steel: new THREE.MeshStandardMaterial({
      map: getFacade("steel"),
      roughness: 0.40,
      metalness: 0.08,
      color: 0xffffff,
    }),
  };

  // Build UV generator for correct aerial-texture roof projection
  const customUVGenerator: THREE.UVGenerator = {
    generateTopUV(_geom, verts, iA, iB, iC) {
      const ax = verts[iA * 3], ay = verts[iA * 3 + 1];
      const bx = verts[iB * 3], by = verts[iB * 3 + 1];
      const cx = verts[iC * 3], cy = verts[iC * 3 + 1];
      return [
        new THREE.Vector2((ax + uvExtentX * 0.5) / uvExtentX, (ay + uvExtentY * 0.5) / uvExtentY),
        new THREE.Vector2((bx + uvExtentX * 0.5) / uvExtentX, (by + uvExtentY * 0.5) / uvExtentY),
        new THREE.Vector2((cx + uvExtentX * 0.5) / uvExtentX, (cy + uvExtentY * 0.5) / uvExtentY),
      ];
    },
    generateSideWallUV(_geom, verts, iA, iB, iC, iD) {
      const ax = verts[iA * 3], ay = verts[iA * 3 + 1], az = verts[iA * 3 + 2];
      const bx = verts[iB * 3], by = verts[iB * 3 + 1], bz = verts[iB * 3 + 2];
      const cz = verts[iC * 3 + 2];
      const dz = verts[iD * 3 + 2];
      const wallLen = Math.hypot(bx - ax, by - ay);
      // 1 tile = 4 m wide × 3.5 m tall (storey module)
      const uRepeat = wallLen / 4.0;
      const vBot = Math.min(az, dz) / 3.5;
      const vTop = Math.max(bz, cz) / 3.5;
      return [
        new THREE.Vector2(0,       vBot),
        new THREE.Vector2(uRepeat, vBot),
        new THREE.Vector2(uRepeat, vTop),
        new THREE.Vector2(0,       vTop),
      ];
    },
  };

  for (const b of data.buildings) {
    if (!b.coords || b.coords.length < 3) continue;

    // Height classification → facade variant
    const h = b.height_m;
    const variant: "warm" | "cool" | "steel" =
      h < 8 ? "warm" : h < 20 ? "cool" : "steel";
    const wallMat = wallMats[variant];

    // Build 2D footprint shape
    const shape = new THREE.Shape();
    shape.moveTo(b.coords[0][0], b.coords[0][1]);
    for (let i = 1; i < b.coords.length; i++) shape.lineTo(b.coords[i][0], b.coords[i][1]);
    shape.closePath();

    // Height in scene units with 2.0m subterranean foundation skirt to anchor cleanly into sloping terrain
    const skirt = 2.0 * options.exaggeration;
    const baseZ = Math.max(0, (b.base_elev_m - options.zMin) * options.exaggeration);
    const depth = Math.max(3.0, h * options.exaggeration) + skirt;

    const geom = new THREE.ExtrudeGeometry(shape, {
      depth,
      bevelEnabled: false,
      UVGenerator: customUVGenerator,
    });
    geom.computeVertexNormals();

    // In Three.js ExtrudeGeometry:
    // Index 0 = Front & Back Caps (The Roof) -> roofMat
    // Index 1 = Sides (Vertical Walls) -> wallMat
    const mesh = new THREE.Mesh(geom, [roofMat, wallMat]);

    // ── Align building base to terrain mesh ──────────────────────────────
    // Seated at ground elevation with skirt sunk into terrain so no gaps or clipping occur on slopes
    mesh.position.set(0, 0, Math.max(0, baseZ - skirt));
    mesh.castShadow = false;
    mesh.receiveShadow = false;

    group.add(mesh);
  }

  return group;
}
