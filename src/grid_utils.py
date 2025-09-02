import numpy as np
from typing import Tuple, List

def infer_grid(points: list):
    """
    Infer unique sorted lat/lon arrays and grid shape (n_lat, n_lon) from raw points.
    Round to 4 decimal places to avoid floating precision mismatches.
    """
    lats = sorted({round(float(p["lat"]), 4) for p in points})
    lons = sorted({round(float(p["lon"]), 4) for p in points})
    lat_grid = np.array(lats, dtype=float)
    lon_grid = np.array(lons, dtype=float)
    n_lat, n_lon = len(lat_grid), len(lon_grid)
    return lat_grid, lon_grid, n_lat, n_lon

def col_label(lat: float, lon: float) -> str:
    """Standardized column label for wide CSV."""
    return f"{lat:.4f},{lon:.4f}"
