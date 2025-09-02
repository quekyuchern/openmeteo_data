import json
import os
import pandas as pd

def load_raw_json(path: str) -> dict:
    """Load the raw Open-Meteo JSON structure from disk."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def ensure_dir(path: str) -> None:
    """Create parent directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)

def save_csv(df: pd.DataFrame, path: str) -> None:
    """Save a DataFrame to CSV (no index)."""
    ensure_dir(os.path.dirname(path) or ".")
    df.to_csv(path, index=False)

def save_html(html_map, path: str) -> None:
    """Save a Folium map object to HTML."""
    ensure_dir(os.path.dirname(path) or ".")
    html_map.save(path)
