"""List known earthworks that fall inside or near a processed transect grid.

Usage: python -I src/known_in_grid.py <transect.laz> <outdir>

Prints the source database, the grid row and column at the DTM resolution, and the distance in metres to the nearest lidar cell. Positions stay on screen and are never written to tracked files.
"""
import sys
from pathlib import Path

import geopandas as gpd
import laspy
import numpy as np
import pandas as pd
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).parent))
from dtm import RES

REF = Path(__file__).resolve().parent.parent / "data" / "ref" / "earthworks_known.csv"

laz, out = sys.argv[1], Path(sys.argv[2])
dtm = np.load(out / "dtm.npy")
x0, y0 = np.load(out / "origin.npy")
crs = laspy.open(laz).header.parse_crs()
k = pd.read_csv(REF)
g = gpd.GeoDataFrame(k, geometry=gpd.points_from_xy(k.Longitude, k.Latitude), crs=4326).to_crs(crs)
dist = ndimage.distance_transform_edt(np.isnan(dtm)) * RES
pad = int(2000 / RES)
for db, p in zip(g.Database, g.geometry):
    r, c = int((y0 - p.y) / RES), int((p.x - x0) / RES)
    if -pad < r < dtm.shape[0] + pad and -pad < c < dtm.shape[1] + pad:
        inside = 0 <= r < dtm.shape[0] and 0 <= c < dtm.shape[1]
        d = dist[r, c] if inside else float("nan")
        print(f"{db:40s} row {r:6d} col {c:6d} off-data {d:.0f} m")
