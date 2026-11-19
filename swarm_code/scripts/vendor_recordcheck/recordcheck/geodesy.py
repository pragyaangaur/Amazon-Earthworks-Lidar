"""Geodesy for offline known-record checks.  Pure Python + numpy, no network.

Primary distance service is Vincenty's inverse formula on the ellipsoid, in a
vectorised form for bulk candidate-to-record matrices.  It is NOT trusted on
its own: ``tests/test_geodesy.py`` validates it against

  * PROJ's geodesic (Karney) via ``pyproj.Geod`` when pyproj is importable,
  * independent numerical integration of the meridional arc, i.e. the distance
    along a meridian between two latitudes,
  * the exact equator identity s = a * dlon,
  * direct/inverse round trips and symmetry.

Documented limit: Vincenty's inverse does not converge for (near-)antipodal
pairs (roughly > 179.5 deg apart) and is singular exactly antipodal.  Amazon
work never sees such pairs; the vectorised routine returns NaN there instead of
a wrong number, and callers must treat NaN as "unreliable", never as a distance.

Latitudes/longitudes are decimal degrees on WGS84 unless another ellipsoid is
passed explicitly.  SIRGAS 2000 vs WGS84 changes such distances by well under
1 mm, and both ellipsoids are provided.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

__all__ = [
    "Ellipsoid",
    "WGS84",
    "GRS80",
    "haversine",
    "meridian_arc",
    "vincenty_inverse",
    "vincenty_direct",
    "pairwise_distance",
    "nearest",
]


@dataclass(frozen=True)
class Ellipsoid:
    """Reference ellipsoid defined by semi-major axis and inverse flattening."""

    name: str
    a: float
    rf: float

    @property
    def f(self) -> float:
        return 1.0 / self.rf

    @property
    def b(self) -> float:
        return self.a * (1.0 - self.f)

    @property
    def e2(self) -> float:
        f = self.f
        return 2.0 * f - f * f


WGS84 = Ellipsoid("WGS84", 6378137.0, 298.257223563)
GRS80 = Ellipsoid("GRS80", 6378137.0, 298.257222101)


# --------------------------------------------------------------------------
# spherical helpers (cheap prefilter / cross-check only)
# --------------------------------------------------------------------------
def haversine(lat1, lon1, lat2, lon2, radius: float = WGS84.a):
    """Great-circle distance on a sphere of ``radius`` metres.  Degrees in."""
    la1, lo1, la2, lo2 = (np.radians(np.asarray(v, dtype=float)) for v in (lat1, lon1, lat2, lon2))
    dla = la2 - la1
    dlo = lo2 - lo1
    h = np.sin(dla / 2.0) ** 2 + np.cos(la1) * np.cos(la2) * np.sin(dlo / 2.0) ** 2
    return 2.0 * radius * np.arcsin(np.sqrt(np.clip(h, 0.0, 1.0)))


def _norm_lon(dlon):
    """Wrap a longitude difference (radians) into (-pi, pi]."""
    return np.arctan2(np.sin(dlon), np.cos(dlon))


# --------------------------------------------------------------------------
# meridional arc M(phi): distance from the equator to latitude phi
# --------------------------------------------------------------------------
def meridian_arc(phi_deg, ell: Ellipsoid = WGS84):
    """Distance (m) along the meridian from the equator to latitude phi.

    Termwise integration of a(1-e^2)(1-e^2 sin^2 phi)^(-3/2), series in e^2 up
    to e^10.  ``tests/test_geodesy.py`` checks it against Gauss-Legendre
    quadrature and requires < 1e-6 m difference.
    """
    phi = np.radians(np.asarray(phi_deg, dtype=float))
    e2 = ell.e2
    e4 = e2 * e2
    e6 = e4 * e2
    e8 = e4 * e4
    e10 = e8 * e2
    c0 = 1 - e2 / 4 - 3 * e4 / 64 - 5 * e6 / 256 - 175 * e8 / 16384 - 441 * e10 / 65536
    c2 = 3 * e2 / 8 + 3 * e4 / 32 + 45 * e6 / 1024 + 105 * e8 / 4096 + 2205 * e10 / 131072
    c4 = 15 * e4 / 256 + 45 * e6 / 1024 + 525 * e8 / 16384 + 1575 * e10 / 65536
    c6 = 35 * e6 / 3072 + 175 * e8 / 12288 + 2205 * e10 / 262144
    c8 = 315 * e8 / 131072 + 2205 * e10 / 1048576
    c10 = 693 * e10 / 8388608
    return ell.a * (
        c0 * phi
        - c2 * np.sin(2 * phi)
        + c4 * np.sin(4 * phi)
        - c6 * np.sin(6 * phi)
        + c8 * np.sin(8 * phi)
        - c10 * np.sin(10 * phi)
    )


# --------------------------------------------------------------------------
# Vincenty inverse (scalar reference implementation)
# --------------------------------------------------------------------------
def vincenty_inverse(lat1: float, lon1: float, lat2: float, lon2: float,
                     ell: Ellipsoid = WGS84, tol: float = 1e-12, max_iter: int = 200):
    """Ellipsoidal inverse geodesic problem.

    Returns (s_metres, azimuth1_deg, azimuth2_deg).  ``s`` is NaN where the
    iteration cannot converge (near-antipodal) or the input is invalid; callers
    must treat that as "unreliable", not as a distance.
    """
    if not (all(map(math.isfinite, (lat1, lon1, lat2, lon2)))):
        return math.nan, math.nan, math.nan
    if abs(lat1) > 90.0 or abs(lat2) > 90.0:
        return math.nan, math.nan, math.nan
    a, b, f = ell.a, ell.b, ell.f
    if lat1 == lat2 and ((lon2 - lon1 + 180.0) % 360.0) - 180.0 == 0.0:
        return 0.0, math.nan, math.nan
    L = math.radians(((lon2 - lon1 + 180.0) % 360.0) - 180.0)
    U1 = math.atan((1.0 - f) * math.tan(math.radians(lat1)))
    U2 = math.atan((1.0 - f) * math.tan(math.radians(lat2)))
    sinU1, cosU1 = math.sin(U1), math.cos(U1)
    sinU2, cosU2 = math.sin(U2), math.cos(U2)
    lam = L
    for _ in range(max_iter):
        sin_lam, cos_lam = math.sin(lam), math.cos(lam)
        sin_sigma = math.hypot(cosU2 * sin_lam, cosU1 * sinU2 - sinU1 * cosU2 * cos_lam)
        cos_sigma = sinU1 * sinU2 + cosU1 * cosU2 * cos_lam
        if sin_sigma == 0.0:
            # coincident (cos_sigma > 0) or exactly antipodal (cos_sigma < 0)
            return (0.0, math.nan, math.nan) if cos_sigma > 0 else (math.nan, math.nan, math.nan)
        sigma = math.atan2(sin_sigma, cos_sigma)
        sin_alpha = cosU1 * cosU2 * sin_lam / sin_sigma
        cos2_alpha = 1.0 - sin_alpha * sin_alpha
        cos2_sigma_m = (cos_sigma - 2.0 * sinU1 * sinU2 / cos2_alpha) if cos2_alpha != 0.0 else 0.0
        C = f / 16.0 * cos2_alpha * (4.0 + f * (4.0 - 3.0 * cos2_alpha))
        lam_prev = lam
        lam = L + (1.0 - C) * f * sin_alpha * (
            sigma + C * sin_sigma * (cos2_sigma_m + C * cos_sigma * (-1.0 + 2.0 * cos2_sigma_m ** 2))
        )
        if abs(lam - lam_prev) < tol:
            break
    else:
        return math.nan, math.nan, math.nan

    u2 = cos2_alpha * (a * a - b * b) / (b * b)
    A = 1.0 + u2 / 16384.0 * (4096.0 + u2 * (-768.0 + u2 * (320.0 - 175.0 * u2)))
    B = u2 / 1024.0 * (256.0 + u2 * (-128.0 + u2 * (74.0 - 47.0 * u2)))
    d_sigma = B * sin_sigma * (
        cos2_sigma_m
        + B / 4.0 * (cos_sigma * (-1.0 + 2.0 * cos2_sigma_m ** 2)
                     - B / 6.0 * cos2_sigma_m * (-3.0 + 4.0 * sin_sigma ** 2)
                     * (-3.0 + 4.0 * cos2_sigma_m ** 2))
    )
    s = b * A * (sigma - d_sigma)
    az1 = math.degrees(math.atan2(cosU2 * math.sin(lam), cosU1 * sinU2 - sinU1 * cosU2 * math.cos(lam)))
    az2 = math.degrees(math.atan2(cosU1 * math.sin(lam), -sinU1 * cosU2 + cosU1 * sinU2 * math.cos(lam)))
    return s, az1, az2


def vincenty_direct(lat1: float, lon1: float, az1_deg: float, s: float,
                    ell: Ellipsoid = WGS84, tol: float = 1e-12, max_iter: int = 200):
    """Ellipsoidal direct geodesic problem.  Returns (lat2, lon2, az2) degrees."""
    a, b, f = ell.a, ell.b, ell.f
    if s == 0.0:
        return lat1, lon1, az1_deg
    alpha1 = math.radians(az1_deg)
    sin_alpha1, cos_alpha1 = math.sin(alpha1), math.cos(alpha1)
    tanU1 = (1.0 - f) * math.tan(math.radians(lat1))
    cosU1 = 1.0 / math.sqrt(1.0 + tanU1 * tanU1)
    sinU1 = tanU1 * cosU1
    sigma1 = math.atan2(tanU1, cos_alpha1)
    sin_alpha = cosU1 * sin_alpha1
    cos2_alpha = 1.0 - sin_alpha * sin_alpha
    u2 = cos2_alpha * (a * a - b * b) / (b * b)
    A = 1.0 + u2 / 16384.0 * (4096.0 + u2 * (-768.0 + u2 * (320.0 - 175.0 * u2)))
    B = u2 / 1024.0 * (256.0 + u2 * (-128.0 + u2 * (74.0 - 47.0 * u2)))
    sigma = s / (b * A)
    for _ in range(max_iter):
        cos2_sigma_m = math.cos(2.0 * sigma1 + sigma)
        sin_sigma, cos_sigma = math.sin(sigma), math.cos(sigma)
        d_sigma = B * sin_sigma * (
            cos2_sigma_m
            + B / 4.0 * (cos_sigma * (-1.0 + 2.0 * cos2_sigma_m ** 2)
                         - B / 6.0 * cos2_sigma_m * (-3.0 + 4.0 * sin_sigma ** 2)
                         * (-3.0 + 4.0 * cos2_sigma_m ** 2))
        )
        sigma_prev = sigma
        sigma = s / (b * A) + d_sigma
        if abs(sigma - sigma_prev) < tol:
            break
    else:
        return math.nan, math.nan, math.nan
    tmp = sinU1 * math.sin(sigma) - cosU1 * math.cos(sigma) * cos_alpha1
    lat2 = math.degrees(math.atan2(
        sinU1 * math.cos(sigma) + cosU1 * math.sin(sigma) * cos_alpha1,
        (1.0 - f) * math.hypot(sin_alpha, tmp)))
    lam = math.atan2(math.sin(sigma) * sin_alpha1,
                     cosU1 * math.cos(sigma) - sinU1 * math.sin(sigma) * cos_alpha1)
    C = f / 16.0 * cos2_alpha * (4.0 + f * (4.0 - 3.0 * cos2_alpha))
    cos2m = math.cos(2.0 * sigma1 + sigma)
    L = lam - (1.0 - C) * f * sin_alpha * (
        sigma + C * math.sin(sigma) * (cos2m + C * math.cos(sigma) * (-1.0 + 2.0 * cos2m ** 2)))
    lon2 = lon1 + math.degrees(L)
    lon2 = ((lon2 + 180.0) % 360.0) - 180.0
    az2 = math.degrees(math.atan2(sin_alpha, -tmp))
    return lat2, lon2, az2


# --------------------------------------------------------------------------
# Vectorised pairwise distances (candidate rows x record columns)
# --------------------------------------------------------------------------
def pairwise_distance(lat_a, lon_a, lat_b, lon_b, ell: Ellipsoid = WGS84, chunk: int = 128):
    """Full distance matrix, shape (len(lat_a), len(lat_b)), metres.

    Vincenty's inverse vectorised over the broadcast pair matrix.  NaN marks a
    pair where the series does not converge (near-antipodal) or bad input; NaNs
    are never silently replaced by a number.

    ``chunk`` bounds memory: the iteration holds ~12 float64 arrays of shape
    (chunk, len(lat_b)); 128 x 961 is ~12 MB.
    """
    la1 = np.atleast_1d(np.asarray(lat_a, dtype=float)).ravel()
    lo1 = np.atleast_1d(np.asarray(lon_a, dtype=float)).ravel()
    la2 = np.atleast_1d(np.asarray(lat_b, dtype=float)).ravel()
    lo2 = np.atleast_1d(np.asarray(lon_b, dtype=float)).ravel()
    if not (la1.shape == lo1.shape and la2.shape == lo2.shape):
        raise ValueError("lat/lon length mismatch")
    n, m = la1.size, la2.size
    out = np.full((n, m), np.nan, dtype=float)
    if n == 0 or m == 0:
        return out

    a, b, f = ell.a, ell.b, ell.f
    valid_a = np.isfinite(la1) & np.isfinite(lo1) & (np.abs(la1) <= 90.0)
    valid_b = np.isfinite(la2) & np.isfinite(lo2) & (np.abs(la2) <= 90.0)

    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        rows = np.where(valid_a[start:stop])[0]
        if rows.size == 0:
            continue
        A1 = la1[start:stop][rows][:, None]
        O1 = lo1[start:stop][rows][:, None]
        A2 = la2[None, :]
        O2 = lo2[None, :]
        ok = valid_b[None, :] & np.ones_like(A1, dtype=bool)

        L = _norm_lon(np.radians(O2 - O1))
        U1 = np.arctan((1.0 - f) * np.tan(np.radians(A1)))
        U2 = np.arctan((1.0 - f) * np.tan(np.radians(A2)))
        sinU1, cosU1 = np.sin(U1), np.cos(U1)
        sinU2, cosU2 = np.sin(U2), np.cos(U2)
        lam = L.copy()
        delta = np.full_like(lam, np.inf)
        sin_sigma = cos_sigma = None
        for _ in range(200):
            sin_lam, cos_lam = np.sin(lam), np.cos(lam)
            sin_sigma = np.hypot(cosU2 * sin_lam, cosU1 * sinU2 - sinU1 * cosU2 * cos_lam)
            cos_sigma = sinU1 * sinU2 + cosU1 * cosU2 * cos_lam
            sigma = np.arctan2(sin_sigma, cos_sigma)
            safe = np.where(sin_sigma == 0.0, 1.0, sin_sigma)
            sin_alpha = np.where(sin_sigma == 0.0, 0.0, cosU1 * cosU2 * sin_lam / safe)
            cos2_alpha = 1.0 - sin_alpha * sin_alpha
            denom = np.where(cos2_alpha != 0.0, cos2_alpha, 1.0)
            cos2_sigma_m = np.where(cos2_alpha != 0.0, cos_sigma - 2.0 * sinU1 * sinU2 / denom, 0.0)
            C = f / 16.0 * cos2_alpha * (4.0 + f * (4.0 - 3.0 * cos2_alpha))
            lam_new = L + (1.0 - C) * sin_alpha * f * (
                sigma + C * sin_sigma * (cos2_sigma_m + C * cos_sigma * (-1.0 + 2.0 * cos2_sigma_m ** 2))
            )
            delta = np.abs(lam_new - lam)
            lam = lam_new
            if ok.any() and np.all(delta[ok] < 1e-12):
                break

        # recompute everything from the final lambda (keeps stored values consistent)
        sin_lam, cos_lam = np.sin(lam), np.cos(lam)
        sin_sigma = np.hypot(cosU2 * sin_lam, cosU1 * sinU2 - sinU1 * cosU2 * cos_lam)
        cos_sigma = sinU1 * sinU2 + cosU1 * cosU2 * cos_lam
        sigma = np.arctan2(sin_sigma, cos_sigma)
        safe = np.where(sin_sigma == 0.0, 1.0, sin_sigma)
        sin_alpha = np.where(sin_sigma == 0.0, 0.0, cosU1 * cosU2 * sin_lam / safe)
        cos2_alpha = 1.0 - sin_alpha * sin_alpha
        denom = np.where(cos2_alpha != 0.0, cos2_alpha, 1.0)
        cos2_sigma_m = np.where(cos2_alpha != 0.0, cos_sigma - 2.0 * sinU1 * sinU2 / denom, 0.0)
        u2 = cos2_alpha * (a * a - b * b) / (b * b)
        A = 1.0 + u2 / 16384.0 * (4096.0 + u2 * (-768.0 + u2 * (320.0 - 175.0 * u2)))
        B = u2 / 1024.0 * (256.0 + u2 * (-128.0 + u2 * (74.0 - 47.0 * u2)))
        d_sigma = B * sin_sigma * (
            cos2_sigma_m
            + B / 4.0 * (cos_sigma * (-1.0 + 2.0 * cos2_sigma_m ** 2)
                         - B / 6.0 * cos2_sigma_m * (-3.0 + 4.0 * sin_sigma ** 2)
                         * (-3.0 + 4.0 * cos2_sigma_m ** 2))
        )
        s = b * A * (sigma - d_sigma)
        coincident = (sin_sigma == 0.0) & (cos_sigma > 0.0)
        antipodal = (sin_sigma == 0.0) & (cos_sigma <= 0.0)
        s = np.where(coincident, 0.0, s)
        good = (delta < 1e-12) & ok & ~antipodal
        block = out[start:stop, :]
        block[rows, :] = np.where(good, s, np.nan)
        out[start:stop, :] = block
    return out


def nearest(lat, lon, rec_lat, rec_lon, ell: Ellipsoid = WGS84, chunk: int = 128):
    """Index and ellipsoidal distance (m) of the nearest record for each point.

    Returns (idx int array, dist float array); idx is -1 and dist NaN where no
    finite distance exists (bad input or non-converging pair).
    """
    pts = np.atleast_1d(np.asarray(lat, dtype=float)).ravel()
    d = pairwise_distance(lat, lon, rec_lat, rec_lon, ell=ell, chunk=chunk)
    if d.size == 0:
        return np.full(pts.size, -1, dtype=int), np.full(pts.size, np.nan)
    allnan = np.all(np.isnan(d), axis=1)
    idx = np.where(allnan, -1, np.nanargmin(d, axis=1)).astype(int)
    dist = np.where(allnan, np.nan, d[np.arange(d.shape[0]), np.maximum(idx, 0)])
    return idx, dist
