/**
 * Architectural Ground Pedestal (Skirt + Baseplate) for Heightfield Meshes.
 *
 * Drops a sleek vertical skirt along the terrain perimeter down to zBottom
 * and adds a baseplate, transforming the raw floating heightfield into a
 * clean, solid architectural 3D relief model (exhibition pedestal style).
 */
import * as THREE from "three";

export interface PedestalOptions {
  zBottom?: number;        // scene Z coordinate for pedestal bottom (default -12)
  materialColor?: number;  // default dark slate 0x0f172a
}

export function buildPedestalGeometry(
  heights: Float32Array,
  W: number,
  H: number,
  extentX: number,
  extentY: number,
  spacing: number,
  zMin: number,
  exaggeration: number,
  metric: boolean,
  zScale: number,
  opts?: PedestalOptions
): THREE.BufferGeometry {
  const zBottom = opts?.zBottom ?? -15.0;

  function getZ(col: number, row: number): number {
    const raw = heights[row * W + col];
    const val = Number.isFinite(raw) ? raw : zMin;
    if (metric) {
      return (val - zMin) * exaggeration;
    } else {
      return val * (zScale * Math.max(extentX, extentY));
    }
  }

  function getPos(col: number, row: number): [number, number, number] {
    const x = -extentX / 2 + col * spacing;
    const y = extentY / 2 - row * spacing;
    const z = getZ(col, row);
    return [x, y, z];
  }

  // Collect boundary points clockwise:
  // 1. North edge: col 0..W-1, row 0
  // 2. East edge:  col W-1, row 1..H-1
  // 3. South edge: col W-2..0, row H-1
  // 4. West edge:  col 0, row H-2..1
  const boundary: [number, number, number][] = [];

  for (let c = 0; c < W; c++) boundary.push(getPos(c, 0));
  for (let r = 1; r < H; r++) boundary.push(getPos(W - 1, r));
  for (let c = W - 2; c >= 0; c--) boundary.push(getPos(c, H - 1));
  for (let r = H - 2; r >= 1; r--) boundary.push(getPos(0, r));

  const N = boundary.length;
  // Each segment has 4 vertices for the wall (topA, topB, botB, botA) -> 2 triangles
  const positions: number[] = [];
  const normals: number[] = [];

  for (let i = 0; i < N; i++) {
    const next = (i + 1) % N;
    const [ax, ay, az] = boundary[i];
    const [bx, by, bz] = boundary[next];

    // Outward normal in XY plane
    const dx = bx - ax;
    const dy = by - ay;
    const len = Math.hypot(dx, dy) || 1;
    // Clockwise boundary -> outward normal is (dy/len, -dx/len, 0)
    const nx = dy / len;
    const ny = -dx / len;
    const nz = 0.0;

    // Triangle 1: topA -> topB -> botB
    positions.push(
      ax, ay, az,
      bx, by, bz,
      bx, by, zBottom
    );
    normals.push(
      nx, ny, nz,
      nx, ny, nz,
      nx, ny, nz
    );

    // Triangle 2: topA -> botB -> botA
    positions.push(
      ax, ay, az,
      bx, by, zBottom,
      ax, ay, zBottom
    );
    normals.push(
      nx, ny, nz,
      nx, ny, nz,
      nx, ny, nz
    );
  }

  // Baseplate (bottom cap): simple quad at zBottom
  const hx = extentX / 2;
  const hy = extentY / 2;
  // Quad covering [-hx, -hy] to [hx, hy] facing down (-Z)
  positions.push(
    -hx, -hy, zBottom,
     hx, -hy, zBottom,
     hx,  hy, zBottom,

    -hx, -hy, zBottom,
     hx,  hy, zBottom,
    -hx,  hy, zBottom
  );
  for (let k = 0; k < 6; k++) {
    normals.push(0, 0, -1);
  }

  const geom = new THREE.BufferGeometry();
  geom.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3));
  geom.setAttribute("normal", new THREE.Float32BufferAttribute(normals, 3));
  return geom;
}
