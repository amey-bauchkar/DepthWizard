"""Grid: the single raster-metadata object that travels with every job (Phase 6 §9.3, Phase 7 §7).

Vocabulary is fixed here and must be used everywhere:
  * relative height  -> unitless, Mode A (tier R)
  * nDSM             -> metres above local ground (tier H)
  * terrain / DSM    -> metres on a declared vertical CRS (tier T/A)
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

from affine import Affine

Tier = Literal["R", "H", "T", "A"]
Units = Literal["relative", "metres"]


@dataclass
class Grid:
    width: int
    height: int
    transform: Affine | None = None  # pixel -> CRS (GDAL convention, pixel corner origin)
    crs: str | None = None  # EPSG:xxxx or WKT; None => pixel space only
    dtype: str = "float32"
    nodata: float | None = -9999.0
    units: Units = "relative"
    metric: bool = False
    vertical_reference: str | None = None  # e.g. "EPSG:3855" (EGM2008); None for relative outputs
    tier: Tier = "R"
    extra: dict[str, Any] = field(default_factory=dict)

    # ---- geometry helpers ------------------------------------------------
    @property
    def has_georeferencing(self) -> bool:
        return self.crs is not None and self.transform is not None

    @property
    def bounds(self) -> tuple[float, float, float, float] | None:
        """(left, bottom, right, top) in CRS units, or None in pixel space."""
        if self.transform is None:
            return None
        t = self.transform
        xs, ys = [], []
        for col, row in ((0, 0), (self.width, 0), (0, self.height), (self.width, self.height)):
            x, y = t * (col, row)
            xs.append(x)
            ys.append(y)
        return (min(xs), min(ys), max(xs), max(ys))

    @property
    def pixel_size(self) -> tuple[float, float] | None:
        if self.transform is None:
            return None
        t = self.transform
        return ((t.a**2 + t.d**2) ** 0.5, (t.b**2 + t.e**2) ** 0.5)

    def pixel_center_to_crs(self, col: float, row: float) -> tuple[float, float]:
        """Centre-of-pixel convention: pixel (col,row) centre is at (col+0.5, row+0.5)."""
        if self.transform is None:
            return (col + 0.5, row + 0.5)
        return self.transform * (col + 0.5, row + 0.5)

    def crs_to_pixel(self, x: float, y: float) -> tuple[float, float]:
        if self.transform is None:
            return (x - 0.5, y - 0.5)
        col, row = ~self.transform * (x, y)
        return (col - 0.5, row - 0.5)

    def assert_same(self, other: "Grid") -> None:
        if (self.width, self.height) != (other.width, other.height):
            raise ValueError(f"Grid size mismatch {self.width}x{self.height} vs {other.width}x{other.height}")
        if (self.transform is None) != (other.transform is None) or (
            self.transform is not None and not _affine_close(self.transform, other.transform)  # type: ignore[arg-type]
        ):
            raise ValueError("Grid transform mismatch")
        if (self.crs or None) != (other.crs or None):
            raise ValueError(f"Grid CRS mismatch {self.crs} vs {other.crs}")

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["transform"] = list(self.transform.to_gdal()) if self.transform is not None else None
        d["bounds"] = self.bounds
        d["pixel_size"] = self.pixel_size
        d["has_georeferencing"] = self.has_georeferencing
        return d

    @classmethod
    def pixel_space(cls, width: int, height: int, *, nodata: float | None = -9999.0) -> "Grid":
        """Mode A grid: no CRS, no transform, relative units, tier R."""
        return cls(width=width, height=height, transform=None, crs=None, nodata=nodata, units="relative", metric=False, vertical_reference=None, tier="R")


def _affine_close(a: Affine, b: Affine, tol: float = 1e-9) -> bool:
    return all(abs(x - y) <= tol * max(1.0, abs(x), abs(y)) for x, y in zip(a[:6], b[:6]))
