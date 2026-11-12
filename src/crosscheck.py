"""Compare a candidate position with the known earthworks in the Peripato et al. 2023 data release.

Usage: python -I src/crosscheck.py <transect.laz> <outdir> <row> <col>

The row and column are pixel positions in the DTM from dtm.py. The script prints the distance to the nearest known earthwork and its source database. It never writes the candidate position to a tracked file.
"""
import sys
from pathlib import Path

import geopandas as gpd
import laspy
import numpy as np
import pandas as pd

REF = Path(__file__).resolve().parent.parent / "data" / "ref" / "earthworks_known.csv"


def pixel_to_point(laz, outdir, row, col):
    crs = laspy.open(laz).header.parse_crs()
    x0, y0 = np.load(Path(outdir) / "origin.npy")
    return gpd.GeoSeries(gpd.points_from_xy([x0 + col + 0.5], [y0 - row - 0.5]), crs=crs)


def nearest(pt, k=3):
    known = pd.read_csv(REF)
    g = gpd.GeoDataFrame(known, geometry=gpd.points_from_xy(known.Longitude, known.Latitude), crs=4326)
    g = g.to_crs(pt.crs)
    g["dist_m"] = g.distance(pt.iloc[0])
    return g.nsmallest(k, "dist_m")[["Database", "dist_m"]]


if __name__ == "__main__":
    laz, outdir, row, col = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
    print(nearest(pixel_to_point(laz, outdir, row, col)).to_string(index=False))
