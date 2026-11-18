"""Geometry helpers.

Lon/lat are angles, so we project each trip to a local metric plane
(equirectangular around the trip centroid). Each trip spans at most a few tens
of km, where this approximation has a distance error well under 1 %.
"""
import numpy as np

EARTH_R = 6_371_008.8


class LocalProjection:
    def __init__(self, lon0: float, lat0: float):
        self.lon0, self.lat0 = lon0, lat0
        self.kx = EARTH_R * np.cos(np.radians(lat0)) * np.pi / 180.0
        self.ky = EARTH_R * np.pi / 180.0

    def to_xy(self, lon, lat):
        lon = np.asarray(lon, float)
        lat = np.asarray(lat, float)
        return np.stack([(lon - self.lon0) * self.kx, (lat - self.lat0) * self.ky], axis=-1)

    def to_lonlat(self, xy):
        xy = np.asarray(xy, float)
        return np.stack([xy[..., 0] / self.kx + self.lon0, xy[..., 1] / self.ky + self.lat0], axis=-1)


def point_segment(p, a, b):
    """Distance from points p to segments a->b (all (n,2), broadcast).

    Returns (dist, frac) where frac in [0,1] is the position of the projection
    along the segment.
    """
    ab = b - a
    ap = p - a
    L2 = np.einsum("ij,ij->i", ab, ab)
    frac = np.where(L2 > 0, np.einsum("ij,ij->i", ap, ab) / np.where(L2 > 0, L2, 1), 0.0)
    frac = np.clip(frac, 0.0, 1.0)
    proj = a + frac[:, None] * ab
    d = np.linalg.norm(p - proj, axis=1)
    return d, frac


def bearing(dx, dy):
    """Compass-style bearing in degrees [0, 360) of a vector in the local plane."""
    return (np.degrees(np.arctan2(dx, dy)) + 360.0) % 360.0


def angle_diff(a, b):
    """Smallest absolute difference between two angles in degrees, in [0, 180]."""
    d = np.abs(a - b) % 360.0
    return np.minimum(d, 360.0 - d)
