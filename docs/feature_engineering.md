# Feature Engineering: Temporal Rainfall (Per‑Cell, No Aggregation)

This page documents `src/feature_engin.py`, which generates a **starter set of temporal rainfall features per grid cell** (no spatial aggregation). It operates on hourly rainfall (mm) where each data column is named like `LAT,LON` (e.g., `1.2200,103.6000`) and the index (or `timestamp` column) is hourly.

---

## At a glance

- **Per‑cell features only** (you will get many columns: ~30–35 per cell × 80 cells ≈ 2.4–2.8k total).
- **Past‑only features** (no leakage): lags, rolling sums/max, APIs, dry‑spell, rainnow, delta.
- **Forecast‑aware features** (look‑ahead): next‑H sum/max, time‑to‑peak (TTP), frontshare.
- **Column naming**: `<feature>_<LAT>_<LON>` → e.g., `lag1h_1.2200_103.6000`.

> **Leakage note**  
> Forecast features are computed by looking forward in the **same historical series**. In production, replace them with **actual forecast inputs** available at time *t* up to your prediction horizon.

---

## Input expectations

- Columns: `timestamp` (used as index) + many rainfall columns named `LAT,LON` (mm/h).  
- Frequency: hourly.  
- Timezone: Asia/Singapore (informational only; functions do not enforce tz).

---

## Quick start

```python
import pandas as pd
from src.feature_engin import engineer_temporal_features_all_cells

df = pd.read_csv("../data/sg_rainfall_processed.csv", parse_dates=["timestamp"]).set_index("timestamp")
feats = engineer_temporal_features_all_cells(df)   # uses sensible defaults
feats.to_csv("../data/rainfall_temporal_features.csv")

print(feats.shape)  # (n_rows, n_features)
```

**Typical size**: ~30–35 features × 80 cells ≈ **2,400–2,800** columns.

---

## Feature families

### 1) Past‑only (no leakage)

- **Lags**: `lag{h}h` for `h ∈ {1,2,3,4,5,6,12,24}`  
  *Rain at t−h hours.*
- **Rolling sums**: `sum{w}h` for `w ∈ {3,6,12,24}`  
  *Total rain in the last w hours (including t).*  
- **Rolling max**: `max{w}h` for `w ∈ {3,6}`  
  *Peak 1‑h intensity in the last w hours (including t).*  
- **API (wetness memory)**: `api_hl{hl}h` for half‑lives `{6,24,168}`  
  *Exponential index with decay k = 0.5 ** (1/hl).*  
- **Dry‑spell**: `dryspell`  
  *Consecutive dry hours **before** t (does not count t). “Dry” means ≤ 0.2 mm by default.*  
- **Instantaneous**: `rainnow`, `delta1h`  
  *Is raining now? (`> 0.2 mm` → 1.0) and 1‑hour change (`t − (t−1)`).*

### 2) Forecast‑aware (look‑ahead)

- **Next‑H sums**: `next{H}h_sum` for `H ∈ {1,3,6,12}`  
  *Forecasted total rain in t+1..t+H.*
- **Next‑H max**: `next{H}h_max` for `H ∈ {3,6,12}`  
  *Max forecast intensity in t+1..t+H.*
- **Time‑to‑peak**: `ttp_next{H}h` for `H ∈ {6,12}`  
  *How many hours until the peak occurs in t+1..t+H (1..H).*
- **Frontshare**: `frontshare_next{H}h` (default `H=12`)  
  *Fraction of forecast rain that falls in the first half of the horizon.*

> Use forecast features **only up to** your prediction horizon H to avoid leakage.

---

## Naming convention

Every feature is named `<feature>_<LAT>_<LON>`, preserving 4 decimal places from the input headers.  
Example: `sum6h_1.2914_103.8500`.

---

## Function reference (summary)

### `parse_value_columns(df) -> list[str]`
Return rainfall value columns that look like `"lat,lon"`.

### `latlon_from_col(col) -> (str, str)`
Split `"lat,lon"` into `(lat, lon)` strings for naming.

### `feature_name(prefix, col) -> str`
Compose `"prefix_LAT_LON"` (e.g., `lag1h_1.2200_103.6000`).

### `compute_lags(s, hours) -> dict[str, Series]`
Lag features `s(t − h)` for each `h`.

### `compute_rolling_sums(s, windows) -> dict[str, Series]`
Rolling sums over the past `w` hours (including t).

### `compute_rolling_max(s, windows) -> dict[str, Series]`
Rolling max over the past `w` hours (including t).

### `compute_api(s, halflife_hours) -> Series`
Exponential Antecedent Precipitation Index with half‑life `hl` hours.  
`API_t = k*API_{t-1} + R_t`, where `k = 0.5 ** (1/hl)`.

### `compute_dryspell_hours(s, threshold=0.2) -> Series`
Consecutive dry hours **before** t (does not count t). Dry means `≤ threshold` mm.

### `compute_now_and_delta(s, wet_threshold=0.2) -> dict[str, Series]`
`rainnow` (1.0 if `> wet_threshold` else 0.0), and `delta1h` (`t − (t−1)`).

### `compute_future_sums(s, horizons) -> dict[str, Series]`
Sum of future rain over `t+1..t+H` for each H.

### `compute_future_max(s, horizons) -> dict[str, Series]`
Max future intensity over `t+1..t+H` for each H.

### `compute_time_to_peak(s, horizon) -> Series`
Time (hours) until the peak occurs within `t+1..t+H` (1..H).

### `compute_frontshare(s, horizon=12) -> Series`
Share of total future rain that falls in the **first half** of the horizon.

### `engineer_temporal_features_per_cell(df, col, **kwargs) -> dict[str, Series]`
Build the starter set for **one** coordinate column (no aggregation).  
Key knobs: `lag_hours`, `sum_windows`, `max_windows`, `api_halflives`, thresholds, horizons.

### `engineer_temporal_features_all_cells(df, **kwargs) -> DataFrame`
Build the starter set for **all** `LAT,LON` columns and return a feature table.

---

## Edge cases & NaNs

- Start/end of series create `NaN`s (lags/rollings at the start; look‑ahead at the end).  
- `dryspell` counts **before** t; `frontshare` is `NaN` if total future rain is zero.  
- Handle missing values via model tolerance or imputation downstream.

---

## Tips

- Consider `float32` downstream if memory is tight.  
- Save to Parquet for faster I/O when iterating.  
- Use **blocked / rolling‑origin** CV (no shuffling) for time‑series.

---

## Example: tweak windows & horizons

```python
from src.feature_engin import engineer_temporal_features_all_cells

feats = engineer_temporal_features_all_cells(
    df,
    lag_hours=(1,2,3,6,12,24),
    sum_windows=(3,6,12,24),
    max_windows=(3,6),
    api_halflives=(6,24,168),
    dry_threshold=0.2,
    wet_threshold=0.2,
    future_sum_horizons=(1,3,6,12),
    future_max_horizons=(3,6,12),
    ttp_horizons=(6,12),
    frontshare_h=12,
)
```

---

## Optional polish (MkDocs)

If you use **MkDocs Material**, enable these to improve styling:

```yaml
theme:
  name: material
markdown_extensions:
  - admonition
  - pymdownx.details
  - pymdownx.superfences
  - tables
```

This enables callouts, collapsible sections, and better fenced code blocks.
