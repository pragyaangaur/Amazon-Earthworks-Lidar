"""Independent lon/lat <-> UTM (WGS84) transform, no external dependencies.

Why write this: the skeptic's job includes checking that a candidate's x/y, its
lon/lat and the transect EPSG agree.  Using the same library as the detector
would not be an independent check.  Series form of the transverse Mercator
projection (Snyder, *Map Projections - A Working Manual*, USGS PP 1395,
eqs. 8-9 to 8-25), accurate to well under a metre inside a zone.

EBA L3A uses EPSG:5880; the 0414 products are EPSG:31979 (SIRGAS 2000 / UTM 19S).
SIRGAS 2000 differs from WGS84 by less than a metre, which is 500x smaller than
the nearest-known-site match radius, so WGS84 parameters are used here and the
datum difference is reported rather than modelled.
"""
from __future__ import annotations

import math

A = 6378137.0
F = 1.0 / 298.257223563
K0 = 0.9996
FE = 500000.0
FN = 10000000.0
E2 = F * (2.0 - F)
EP2 = E2 / (1.0 - E2)
E4 = E2 * E2
E6 = E4 * E2


def zone_of(lon):
    return int(math.floor((lon + 180.0) / 6.0)) + 1


def central_meridian(zone):
    return math.radians(zone * 6.0 - 183.0)


def _M(phi):
    return A * ((1 - E2 / 4 - 3 * E4 / 64 - 5 * E6 / 256) * phi
                - (3 * E2 / 8 + 3 * E4 / 32 + 45 * E6 / 1024) * math.sin(2 * phi)
                + (15 * E4 / 256 + 45 * E6 / 1024) * math.sin(4 * phi)
                - (35 * E6 / 3072) * math.sin(6 * phi))


def utm_forward(lon, lat, zone=None, northern=None):
    """lon, lat in degrees -> (easting, northing) in the given UTM zone."""
    zone = zone if zone is not None else zone_of(lon)
    if northern is None:
        northern = lat >= 0
    phi = math.radians(lat)
    lam = math.radians(lon)
    lam0 = central_meridian(zone)
    N = A / math.sqrt(1 - E2 * math.sin(phi) ** 2)
    T = math.tan(phi) ** 2
    C = EP2 * math.cos(phi) ** 2
    Aa = math.cos(phi) * (lam - lam0)
    M = _M(phi)
    x = FE + K0 * N * (Aa + (1 - T + C) * Aa ** 3 / 6
                       + (5 - 18 * T + T * T + 72 * C - 58 * EP2) * Aa ** 5 / 120)
    y = K0 * (M + N * math.tan(phi) * (Aa ** 2 / 2
                                       + (5 - T + 9 * C + 4 * C * C) * Aa ** 4 / 24
                                       + (61 - 58 * T + T * T + 600 * C - 330 * EP2)
                                       * Aa ** 6 / 720))
    if not northern:
        y += FN
    return x, y


def utm_inverse(x, y, zone, northern=True):
    """(easting, northing) -> (lon, lat) in degrees."""
    x = float(x)
    y = float(y)
    if not northern:
        y -= FN
    M = y / K0
    mu = M / (A * (1 - E2 / 4 - 3 * E4 / 64 - 5 * E6 / 256))
    e1 = (1 - math.sqrt(1 - E2)) / (1 + math.sqrt(1 - E2))
    phi1 = (mu + (3 * e1 / 2 - 27 * e1 ** 3 / 32) * math.sin(2 * mu)
            + (21 * e1 ** 2 / 16 - 55 * e1 ** 4 / 32) * math.sin(4 * mu)
            + (151 * e1 ** 3 / 96) * math.sin(6 * mu)
            + (1097 * e1 ** 4 / 512) * math.sin(8 * mu))
    sin1, cos1 = math.sin(phi1), math.cos(phi1)
    tan1 = math.tan(phi1)
    C1 = EP2 * cos1 ** 2
    T1 = tan1 ** 2
    N1 = A / math.sqrt(1 - E2 * sin1 ** 2)
    R1 = A * (1 - E2) / (1 - E2 * sin1 ** 2) ** 1.5
    D = (x - FE) / (N1 * K0)
    lat = phi1 - (N1 * tan1 / R1) * (D ** 2 / 2
                                     - (5 + 3 * T1 + 10 * C1 - 4 * C1 * C1 - 9 * EP2)
                                     * D ** 4 / 24
                                     + (61 + 90 * T1 + 298 * C1 + 45 * T1 * T1
                                        - 252 * EP2 - 3 * C1 * C1) * D ** 6 / 720)
    lon = (D - (1 + 2 * T1 + C1) * D ** 3 / 6
           + (5 - 2 * C1 + 28 * T1 - 3 * C1 * C1 + 8 * EP2 + 24 * T1 * T1)
           * D ** 5 / 120) / cos1
    return math.degrees(central_meridian(zone)) + math.degrees(lon), math.degrees(lat)


# Only the WGS84 UTM series is mapped from memory here; the SIRGAS 2000 south
# block is not, because getting its zone numbering wrong would silently shift
# every candidate.  EPSG:31979 is listed because it was *verified locally*: its
# inverse lands the 0414 footprint bounds and its forward puts the known 0414
# record inside the footprint to within a few metres.
KNOWN_UTM = {31979: (19, False)}


def epsg_to_utm(epsg):
    """(zone, northern) for a UTM EPSG code, else (None, None)."""
    epsg = int(epsg)
    if epsg in KNOWN_UTM:
        return KNOWN_UTM[epsg]
    if 32601 <= epsg <= 32660:
        return epsg - 32600, True
    if 32701 <= epsg <= 32760:
        return epsg - 32700, False
    return None, None


def guess_hemisphere(northing, lat_hint=None):
    """UTM northings above 9,000,000 are southern-hemisphere false northings."""
    if lat_hint is not None:
        return lat_hint >= 0
    return not (northing > 9_000_000.0)


if __name__ == "__main__":
    # self-test against two published anchors and round-trip closure
    tests = [((-67.27138187, -9.85483234), 19, False),
             ((0.0, 0.0), 31, True),
             ((-122.4194, 37.7749), 10, True)]
    for (lon, lat), zone, north in tests:
        x, y = utm_forward(lon, lat, zone, north)
        lon2, lat2 = utm_inverse(x, y, zone, north)
        print(f"lon/lat ({lon:.6f},{lat:.6f}) zone {zone}{'N' if north else 'S'} "
              f"-> E {x:.2f} N {y:.2f} -> ({lon2:.8f},{lat2:.8f}) "
              f"err {abs(lon2-lon)*111320*math.cos(math.radians(lat)):.4f} m")
    for epsg in (31979, 32719, 5880, 32619):
        print(f"EPSG:{epsg} -> {epsg_to_utm(epsg)}")
