# src/feature_engin.py
# ------------------------------------------------------------
# Per-cell temporal rainfall feature engineering (no aggregation).
# Produces lag/rolling/API/dryspell/current-step and forecast-lookahead
# features for each coordinate column in the processed rainfall CSV.
#
# Naming convention:  <feature>_<LAT>_<LON>
# Example: lag1h_1.2200_103.6000
#
# NOTE ON LEAKAGE:
# - "Forecast" features here are computed by looking forward in the same
#   historical time series (shift(-1), rolling on future steps). In live
#   production, replace these with actual forecast inputs available at t.
# ------------------------------------------------------------

from __future__ import annotations
from typing import Dict, Iterable, List, Tuple
import numpy as np
import pandas as pd


# ======================================================================
# Column / coordinate helpers
# ======================================================================

def parse_value_columns(df: pd.DataFrame) -> List[str]:
    """Return the rainfall value columns that look like 'lat,lon'.

    Args:
        df: DataFrame with a 'timestamp' column (or index) plus many 'lat,lon' columns.

    Returns:
        A list of column names whose header contains a comma (assumed to be 'lat,lon').
    """
    return [c for c in df.columns if "," in c]


def latlon_from_col(col: str) -> Tuple[str, str]:
    """Split a 'lat,lon' header into ('LAT', 'LON') strings.

    Args:
        col: Column name formatted as "lat,lon" (e.g., "1.2200,103.6000").

    Returns:
        Tuple of (lat_str, lon_str) with any whitespace trimmed.
    """
    lat, lon = col.split(",")
    return lat.strip(), lon.strip()


def feature_name(prefix: str, col: str) -> str:
    """Compose a per-cell feature name using 'prefix_LAT_LON'.

    Args:
        prefix: Feature prefix (e.g., "lag1h", "sum6h").
        col:    Original column name in "lat,lon" format.

    Returns:
        The feature name "prefix_LAT_LON", e.g. "lag1h_1.2200_103.6000".
    """
    lat, lon = latlon_from_col(col)
    return f"{prefix}_{lat}_{lon}"


# ======================================================================
# Core past-only computations (no leakage)
# ======================================================================

def compute_lags(s: pd.Series, hours: Iterable[int]) -> Dict[str, pd.Series]:
    """Compute lag features s(t - h) for each hour h.

    Args:
        s:     Rainfall series (mm/h), indexed by timestamp.
        hours: Iterable of integer lags in hours (e.g., (1,2,3,6,12,24)).

    Returns:
        Dict mapping "lag{h}h" -> Series lagged by h hours.
    """
    feats = {}
    for h in hours:
        feats[f"lag{h}h"] = s.shift(h)
    return feats


def compute_rolling_sums(s: pd.Series, windows: Iterable[int]) -> Dict[str, pd.Series]:
    """Compute rolling sums over past windows (including current t).

    Args:
        s:       Rainfall series (mm/h).
        windows: Iterable of window sizes in hours (e.g., (3,6,12,24)).

    Returns:
        Dict mapping "sum{w}h" -> rolling sum Series with window w.
        Uses min_periods=1 so early rows are populated (may be partial windows).
    """
    feats = {}
    for w in windows:
        feats[f"sum{w}h"] = s.rolling(window=w, min_periods=1).sum()
    return feats


def compute_rolling_max(s: pd.Series, windows: Iterable[int]) -> Dict[str, pd.Series]:
    """Compute rolling max intensities over past windows (including current t).

    Args:
        s:       Rainfall series (mm/h).
        windows: Iterable of window sizes in hours (e.g., (3,6)).

    Returns:
        Dict mapping "max{w}h" -> rolling max Series with window w.
    """
    feats = {}
    for w in windows:
        feats[f"max{w}h"] = s.rolling(window=w, min_periods=1).max()
    return feats


def compute_api(s: pd.Series, halflife_hours: int) -> pd.Series:
    """Compute an Antecedent Precipitation Index (API) with exponential decay.

    API_t = k * API_{t-1} + R_t,  where  k = 0.5 ** (1 / halflife_hours)

    Args:
        s:              Rainfall series (mm/h).
        halflife_hours: Exponential half-life in hours (e.g., 6, 24, 168).

    Returns:
        A Series of API values (same index as s).

    Notes:
        - API is a simple wetness memory; larger halflife => slower decay.
        - Implemented via lightweight recursion; fast for hourly series.
    """
    k = 0.5 ** (1.0 / float(halflife_hours))
    out = np.empty(len(s), dtype=float)
    api_prev = 0.0
    vals = s.values.astype(float)
    for i, x in enumerate(vals):
        api_prev = k * api_prev + (0.0 if np.isnan(x) else x)
        out[i] = api_prev
    return pd.Series(out, index=s.index)


def compute_dryspell_hours(s: pd.Series, threshold: float = 0.2) -> pd.Series:
    """Count consecutive dry hours BEFORE t (excludes current hour).

    Args:
        s:         Rainfall series (mm/h).
        threshold: Rainfall <= threshold is considered "dry" (default 0.2 mm).

    Returns:
        Series with the length (in hours) of the current dry spell before t.

    Example:
        If t-1..t-3 were dry and t is wet, value at t is 3. If t-1 is wet, value is 0.
    """
    dry_prev = s.shift(1).fillna(0) <= threshold
    grp = (~dry_prev).cumsum()      # new group whenever dry_prev switches to False
    streak = dry_prev.groupby(grp).cumsum()
    return streak.astype("float")


def compute_now_and_delta(s: pd.Series, wet_threshold: float = 0.2) -> Dict[str, pd.Series]:
    """Compute 'is raining now' and 1-hour delta.

    Args:
        s:             Rainfall series (mm/h).
        wet_threshold: Threshold above which the hour is considered "wet" (default 0.2 mm).

    Returns:
        Dict with:
          - "rainnow": 1.0 if s(t) > wet_threshold else 0.0
          - "delta1h": s(t) - s(t-1)
    """
    return {
        "rainnow": (s > wet_threshold).astype(float),
        "delta1h": s.diff(1),
    }


# ======================================================================
# Forecast-lookahead computations (use with care re: leakage)
# ======================================================================

def compute_future_sums(s: pd.Series, horizons: Iterable[int]) -> Dict[str, pd.Series]:
    """Sum of NEXT H hours (t+1..t+H) for each horizon H.

    Args:
        s:         Rainfall series (mm/h).
        horizons:  Iterable of horizons in hours (e.g., (1,3,6,12)).

    Returns:
        Dict mapping "next{H}h_sum" -> Series of forward sums.

    Notes:
        - Implemented as rolling on s.shift(-1), right-aligned.
        - In production, replace with *actual* forecast inputs available at time t.
    """
    shifted = s.shift(-1)
    feats = {}
    for H in horizons:
        feats[f"next{H}h_sum"] = shifted.rolling(window=H, min_periods=1).sum()
    return feats


def compute_future_max(s: pd.Series, horizons: Iterable[int]) -> Dict[str, pd.Series]:
    """Max of NEXT H hours (t+1..t+H) for each horizon H.

    Args:
        s:         Rainfall series (mm/h).
        horizons:  Iterable of horizons in hours (e.g., (3,6,12)).

    Returns:
        Dict mapping "next{H}h_max" -> Series of forward max values.

    Notes:
        See leakage note in compute_future_sums().
    """
    shifted = s.shift(-1)
    feats = {}
    for H in horizons:
        feats[f"next{H}h_max"] = shifted.rolling(window=H, min_periods=1).max()
    return feats


def compute_time_to_peak(s: pd.Series, horizon: int) -> pd.Series:
    """Time-to-peak (in hours) within NEXT `horizon` hours (exclude current t).

    Args:
        s:        Rainfall series (mm/h).
        horizon:  Lookahead window size in hours (e.g., 6 or 12).

    Returns:
        Series where each value is the number of hours until the peak
        intensity occurs within (t+1 .. t+horizon).
        NaN if no future window or all-NaN window.

    Notes:
        - If the first future hour is the max, value is 1. If the max is at t+H, value is H.
        - For model safety, treat NaNs as missing rather than 0.
    """
    n = len(s)
    out = np.full(n, np.nan, dtype=float)
    values = s.values.astype(float)
    for i in range(n):
        start, end = i + 1, min(i + 1 + horizon, n)
        if start >= end:
            continue
        window = values[start:end]
        mask = ~np.isnan(window)
        if not np.any(mask):
            continue
        # argmax over the masked subset
        idx_in_masked = np.argmax(window[mask])
        true_positions = np.flatnonzero(mask)
        out[i] = float(true_positions[idx_in_masked] + 1)  # +1 because t+1..t+H
    return pd.Series(out, index=s.index)


def compute_frontshare(s: pd.Series, horizon: int = 12) -> pd.Series:
    """Fraction of future rain in the FIRST half of the next `horizon` hours.

    Args:
        s:       Rainfall series (mm/h).
        horizon: Lookahead horizon in hours (default 12). Uses t+1..t+horizon.

    Returns:
        Series of shares in [0,1] when total future rain > 0; NaN when total == 0.

    Notes:
        - Computed as rolling sums on s.shift(-1); both first-half and total are aligned.
        - Choose horizon to match your prediction horizon H to avoid leakage.
    """
    H = int(horizon)
    H1 = H // 2
    shifted = s.shift(-1)
    sum_first = shifted.rolling(window=H1, min_periods=1).sum()
    sum_total = shifted.rolling(window=H,  min_periods=1).sum()
    share = (sum_first / sum_total).where(sum_total > 0.0)
    return share


# ======================================================================
# Master per-cell feature builders (no aggregation)
# ======================================================================

def engineer_temporal_features_per_cell(
    df: pd.DataFrame,
    col: str,
    *,
    lag_hours: Iterable[int] = (1, 2, 3, 4, 5, 6, 12, 24),
    sum_windows: Iterable[int] = (3, 6, 12, 24),
    max_windows: Iterable[int] = (3, 6),
    api_halflives: Iterable[int] = (6, 24, 168),   # 6h (fast), 24h (medium), 168h (~7d)
    dry_threshold: float = 0.2,
    wet_threshold: float = 0.2,
    future_sum_horizons: Iterable[int] = (1, 3, 6, 12),
    future_max_horizons: Iterable[int] = (3, 6, 12),
    ttp_horizons: Iterable[int] = (6, 12),
    frontshare_h: int = 12,
) -> Dict[str, pd.Series]:
    """Build the *starter* per-cell temporal features for one coordinate column.

    Args:
        df:                    DataFrame indexed by timestamp; must contain the column `col`.
        col:                   Column name in "lat,lon" format (e.g., "1.2200,103.6000").
        lag_hours:             Hours for lag features s(t - h).
        sum_windows:           Past rolling sum windows (hours).
        max_windows:           Past rolling max windows (hours).
        api_halflives:         Half-lives (hours) for exponential API (e.g., 6, 24, 168).
        dry_threshold:         Threshold (mm) for "dry" in dryspell computation.
        wet_threshold:         Threshold (mm) for "rainnow" indicator.
        future_sum_horizons:   Horizons (hours) for forward sum features (t+1..t+H).
        future_max_horizons:   Horizons (hours) for forward max features (t+1..t+H).
        ttp_horizons:          Horizons (hours) for time-to-peak features.
        frontshare_h:          Horizon (hours) for "frontshare" feature (default 12).

    Returns:
        Dict mapping feature_name -> Series, with names like:
        - lag1h_1.2200_103.6000
        - sum6h_1.2200_103.6000
        - api_hl24h_1.2200_103.6000
        - dryspell_1.2200_103.6000
        - rainnow_1.2200_103.6000
        - next12h_sum_1.2200_103.6000
        - ttp_next6h_1.2200_103.6000
        - frontshare_next12h_1.2200_103.6000

    Notes:
        - All features are computed per cell; no spatial aggregation is performed.
        - Forward-looking features use future values in the same series and must be
          replaced by true forecasts at runtime to avoid information leakage.
    """
    s = df[col].astype(float)

    feats: Dict[str, pd.Series] = {}

    # Past-only features
    feats.update({feature_name(k, col): v for k, v in compute_lags(s, lag_hours).items()})
    feats.update({feature_name(k, col): v for k, v in compute_rolling_sums(s, sum_windows).items()})
    feats.update({feature_name(k, col): v for k, v in compute_rolling_max(s, max_windows).items()})

    # APIs (wetness memory)
    for hl in api_halflives:
        feats[feature_name(f"api_hl{hl}h", col)] = compute_api(s, halflife_hours=hl)

    # Dryspell + now/delta
    feats[feature_name("dryspell", col)] = compute_dryspell_hours(s, threshold=dry_threshold)
    nd = compute_now_and_delta(s, wet_threshold=wet_threshold)
    for k, v in nd.items():
        feats[feature_name(k, col)] = v

    # Forecast-aware (lookahead) features
    feats.update({feature_name(k, col): v for k, v in compute_future_sums(s, future_sum_horizons).items()})
    feats.update({feature_name(k, col): v for k, v in compute_future_max(s, future_max_horizons).items()})
    for H in ttp_horizons:
        feats[feature_name(f"ttp_next{H}h", col)] = compute_time_to_peak(s, horizon=H)
    feats[feature_name(f"frontshare_next{frontshare_h}h", col)] = compute_frontshare(s, horizon=frontshare_h)

    return feats


def engineer_temporal_features_all_cells(
    df: pd.DataFrame,
    **kwargs
) -> pd.DataFrame:
    """Build the *starter* temporal features for all coordinate columns in `df`.

    Args:
        df:     DataFrame indexed by timestamp; columns: "lat,lon" rainfall series.
        **kwargs: Keyword arguments passed through to `engineer_temporal_features_per_cell`,
                 e.g., lag_hours, sum_windows, api_halflives, thresholds, horizons.

    Returns:
        A DataFrame indexed by timestamp with only engineered feature columns,
        concatenating all per-cell outputs. Columns are sorted alphabetically.

    Example:
        >>> feats = engineer_temporal_features_all_cells(df,
        ...     lag_hours=(1,2,3,6,12,24),
        ...     future_sum_horizons=(1,3,6,12))
        >>> feats.shape
        (n_rows, n_features)
    """
    value_cols = parse_value_columns(df)
    all_feats: Dict[str, pd.Series] = {}
    for col in value_cols:
        per_cell = engineer_temporal_features_per_cell(df, col, **kwargs)
        all_feats.update(per_cell)
    out = pd.DataFrame(all_feats, index=df.index).sort_index(axis=1)
    return out
