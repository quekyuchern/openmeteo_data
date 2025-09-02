"""
CLI entry-point: read raw -> process -> write CSV + HTML.
Usage:
    python scripts/process_rainfall.py
"""
import os
from src.io_utils import load_raw_json, save_csv, save_html
from src.transform import raw_to_time_and_cube, cube_to_wide_df
from src.viz_folium import make_folium_map

RAW_PATH  = "data/raw/sg_rainfall_raw.json"
CSV_PATH  = "data/processed/sg_rainfall_processed.csv"
HTML_PATH = "data/processed/sg_rainfall_processed.html"

# SG bounds used for folium map framing
LAT_MIN, LON_MIN, LAT_MAX, LON_MAX = 1.22, 103.60, 1.47, 104.05

def main():
    raw = load_raw_json(RAW_PATH)
    times, cube, lat_grid, lon_grid = raw_to_time_and_cube(raw, tz="Asia/Singapore")

    # Save processed CSV (wide format)
    df = cube_to_wide_df(times, lat_grid, lon_grid, cube)
    save_csv(df, CSV_PATH)
    print(f"Processed CSV saved to: {os.path.abspath(CSV_PATH)}")

    # Save processed Folium map
    fmap = make_folium_map(times, lat_grid, lon_grid, cube, LAT_MIN, LON_MIN, LAT_MAX, LON_MAX)
    save_html(fmap, HTML_PATH)
    print(f"Processed HTML saved to: {os.path.abspath(HTML_PATH)}")

if __name__ == "__main__":
    main()
