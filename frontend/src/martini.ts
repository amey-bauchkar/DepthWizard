/**
 * Client-Side Right-Triangulated Irregular Network (RTIN) mesh builder.
 * Powered by Mapbox Martini with error-bounded simplification.
 *
 * Flattens coplanar regions (water, roads, flat fields) while preserving
 * sharp vertical building edges and ridge lines.
 */
import * as THREE from "three";
// @ts-expect-error - @mapbox/martini does not ship types by default
import Martini from "@mapbox/martini";

export interface RTINMeshResult {
  geometry: THREE.BufferGeometry;
  numVertices: number;
  numTriangles: number;
  reductionPct: number;
  maxError: number;
}

/**
 * Bilinearly samples a height from a 1D row-major height array of shape W x H.
 */
function sampleBilinear(arr: Float32Array, W: number, H: number, u: number, v: number): number {
  const col = Math.max(0, Math.min(W - 1, u * (W - 1)));
  const row = Math.max(0, Math.min(H - 1, v * (H - 1)));

  const c0 = Math.floor(col);
  const r0 = Math.floor(row);
  const c1 = Math.min(W - 1, c0 + 1);
  const r1 = Math.min(H - 1, r0 + 1);

  const fc = col - c0;
  const fr = row - r0;

  const h00 = arr[r0 * W + c0];
  const h10 = arr[r0 * W + c1];
  const h01 = arr[r1 * W + c0];
  const h11 = arr[r1 * W + c1];

  const top = h00 * (1 - fc) + h10 * fc;
  const btm = h01 * (1 - fc) + h11 * fc;
  return top * (1 - fr) + btm * fr;
}

/**
 * Builds an adaptive Three.js BufferGeometry using RTIN / Mapbox Martini.
 */
export function buildRTINGeometry(
  baseZ: Float32Array,
  W: number,
  H: number,
  extentX: number,
  extentY: number,
  maxError: number,
  exaggeration: number,
  metric: boolean,
  zMin: number,
  zScale: number
): RTINMeshResult {
  // Determine canonical grid size 2^k + 1 (e.g. 257 or 513)
  const maxDim = Math.max(W, H);
  const size = maxDim <= 257 ? 257 : 513;

  // Sample heightfield onto canonical Martini grid
  const terrain = new Float32Array(size * size);
  for (let r = 0; r < size; r++) {
    const v = r / (size - 1);
    for (let c = 0; c < size; c++) {
      const u = c / (size - 1);
      terrain[r * size + c] = sampleBilinear(baseZ, W, H, u, v);
    }
  }

  // Instantiate Martini RTIN hierarchy
  const martini = new Martini(size);
  const tile = martini.createTile(terrain);

  // maxError is given in elevation units (metres in metric mode, unitless in relative mode)
  const meshData = tile.getMesh(maxError);
  const { vertices, triangles } = meshData;

  const numVertices = vertices.length / 2;
  const numTriangles = triangles.length / 3;

  const positions = new Float32Array(numVertices * 3);
  const uvs = new Float32Array(numVertices * 2);

  const zCoeff = metric ? exaggeration : zScale * Math.max(extentX, extentY);

  for (let i = 0; i < numVertices; i++) {
    const gx = vertices[2 * i];
    const gy = vertices[2 * i + 1];

    const u = gx / (size - 1);
    const v = gy / (size - 1);

    // Coordinate mapping: origin centered at (0, 0)
    const x = (u - 0.5) * extentX;
    const y = (0.5 - v) * extentY;

    const rawZ = terrain[gy * size + gx];
    const z = metric ? (rawZ - zMin) * zCoeff : rawZ * zCoeff;

    positions[i * 3 + 0] = x;
    positions[i * 3 + 1] = y;
    positions[i * 3 + 2] = z;

    uvs[i * 2 + 0] = u;
    uvs[i * 2 + 1] = 1 - v;
  }

  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
  geometry.setAttribute("uv", new THREE.BufferAttribute(uvs, 2));
  geometry.setIndex(new THREE.BufferAttribute(triangles, 1));
  geometry.computeVertexNormals();

  const regularGridTriangles = (W - 1) * (H - 1) * 2;
  const reductionPct = Math.max(0, Math.round((1 - numTriangles / regularGridTriangles) * 100));

  return {
    geometry,
    numVertices,
    numTriangles,
    reductionPct,
    maxError,
  };
}
