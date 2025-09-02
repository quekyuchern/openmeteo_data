import numpy as np
import pandas as pd
from typing import Tuple, List
from .time_utils import build_time_index_from_point, choose_reference_point
from .grid_utils import infer_grid, col_label

def raw_to_time_and_cube(raw: dict, tz: str = "Asia/Singapore"):
    """
    Convert raw JSON rainfall data into:
      - times: DatetimeIndex (hourly timestamps, localized)
      - cube : 3D numpy array (T, n_lat, n_lon) of rainfall (mm)
      - lat_grid, lon_grid: sorted coordinate arrays
    """ 
    points = raw["points"]
    ref = choose_reference_point(points)
    times = build_time_index_from_point(ref["t_start"], ref["t_end"], ref["dt"], tz=tz)
    T = len(times)

    lat_grid, lon_grid, n_lat, n_lon = infer_grid(points)
    cube = np.full((T, n_lat, n_lon), np.nan, dtype=np.float32)

    # Build lookup indices with rounding
    lat_index = {round(float(lat), 4): i for i, lat in enumerate(lat_grid)}
    lon_index = {round(float(lon), 4): j for j, lon in enumerate(lon_grid)}

    for p in points:
        i = lat_index[round(float(p["lat"]), 4)]
        j = lon_index[round(float(p["lon"]), 4)]
        vals = np.asarray(p["precip"], dtype=np.float32)
        tmin = min(T, len(vals))
        cube[:tmin, i, j] = vals[:tmin]

    return times, cube, lat_grid, lon_grid


def cube_to_wide_df(times, lat_grid, lon_grid, cube) -> pd.DataFrame:
    """
    Flatten a (T, n_lat, n_lon) cube into a wide table:
      - 'timestamp' + one column per (lat,lon)
    """
    df = pd.DataFrame({"timestamp": times})
    for i, lat in enumerate(lat_grid):
        for j, lon in enumerate(lon_grid):
            df[col_label(lat, lon)] = cube[:, i, j]
    return df

def build_folium_frames(times, lat_grid, lon_grid, cube) -> Tuple[List[List[List[float]]], List[str]]:
    """
    Prepare frames and labels for Folium HeatMapWithTime.
    frames[k] = [[lat, lon, value], ...]
    index[k]  = time label string 'YYYY-MM-DD HH:MM'
    """
    frames = []
    for k in range(len(times)):
        frame = []
        for i, lat in enumerate(lat_grid):
            for j, lon in enumerate(lon_grid):
                val = float(cube[k, i, j])
                if np.isfinite(val):
                    frame.append([float(lat), float(lon), val])
        frames.append(frame)
    index = [t.strftime('%Y-%m-%d %H:%M') for t in times]
    return frames, index
