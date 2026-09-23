/**
 * LoD-1 (Level of Detail 1) 3D Vector Building Renderer.
 *
 * Renders extruded 3D polygon prisms with clean 90-degree vertical walls
 * and flat roofs, eliminating the "pyramid tent" and "texture smearing"
 * inherent to 2.5D raster heightfield meshes.
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

/**
 * Creates a procedural architectural facade texture for building walls.
 */
export function createFacadeTexture(): THREE.CanvasTexture {
  const canvas = document.createElement("canvas");
  canvas.width = 256;
  canvas.height = 256;
  const ctx = canvas.getContext("2d")!;

  // Base concrete/stone facade tone
  ctx.fillStyle = "#8a94a6";
  ctx.fillRect(0, 0, 256, 256);

  // Subtle floor dividers (horizontal cornices)
  ctx.fillStyle = "#6e7889";
  for (let y = 0; y < 256; y += 32) {
    ctx.fillRect(0, y, 256, 3);
  }

  // Window grid
  ctx.fillStyle = "#2a364a";
  for (let y = 6; y < 256; y += 32) {
    for (let x = 8; x < 256; x += 24) {
      // Window pane
      ctx.fillRect(x, y, 14, 20);
      // Window sill highlight
      ctx.fillStyle = "#a0abbd";
      ctx.fillRect(x - 1, y + 20, 16, 2);
      ctx.fillStyle = "#2a364a";
    }
  }

  const tex = new THREE.CanvasTexture(canvas);
  tex.wrapS = THREE.RepeatWrapping;
  tex.wrapT = THREE.RepeatWrapping;
  tex.repeat.set(1, 1);
  return tex;
}

/**
 * Builds a THREE.Group of extruded 3D building blocks from LoD-1 JSON data.
 */
export function buildLoD1BuildingGroup(
  data: LoD1Data,
  options: {
    exaggeration: number;
    metric: boolean;
    zMin: number;
    facadeTexture?: THREE.Texture;
  }
): THREE.Group {
  const group = new THREE.Group();
  group.name = "lod1-buildings";

  if (!data.buildings || !data.buildings.length) {
    return group;
  }

  const facadeTex = options.facadeTexture || createFacadeTexture();
  const wallMat = new THREE.MeshStandardMaterial({
    map: facadeTex,
    roughness: 0.65,
    metalness: 0.1,
    color: 0xdde3ed,
  });

  const roofMat = new THREE.MeshStandardMaterial({
    color: 0x5a6578,
    roughness: 0.8,
    metalness: 0.05,
  });

  const materials = [wallMat, roofMat];

  for (const b of data.buildings) {
    if (!b.coords || b.coords.length < 3) continue;

    const shape = new THREE.Shape();
    shape.moveTo(b.coords[0][0], b.coords[0][1]);
    for (let i = 1; i < b.coords.length; i++) {
      shape.lineTo(b.coords[i][0], b.coords[i][1]);
    }
    shape.closePath();

    const depth = Math.max(1.0, b.height_m * options.exaggeration);
    const geom = new THREE.ExtrudeGeometry(shape, {
      depth,
      bevelEnabled: false,
    });
    geom.computeVertexNormals();

    const mesh = new THREE.Mesh(geom, materials);

    // In DepthWizard, Z is up. THREE.ExtrudeGeometry extrudes along +Z by default.
    // Base elevation position:
    const baseZ = options.metric
      ? (b.base_elev_m - options.zMin) * options.exaggeration
      : b.base_elev_m * options.exaggeration;

    mesh.position.set(0, 0, baseZ);
    mesh.castShadow = false;
    mesh.receiveShadow = false;

    group.add(mesh);
  }

  return group;
}
