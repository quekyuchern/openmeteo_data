import pandas as pd
from typing import Sequence

def build_time_index_from_point(t_start: int, t_end: int, dt_sec: int, tz: str = "Asia/Singapore") -> pd.DatetimeIndex:
    """
    Construct a localized DatetimeIndex inclusive of start and exclusive of end.
    """
    t0 = pd.to_datetime(t_start, unit="s", utc=True).tz_convert(tz)
    t1 = pd.to_datetime(t_end,   unit="s", utc=True).tz_convert(tz)
    step = pd.Timedelta(seconds=int(dt_sec))
    return pd.date_range(start=t0, end=t1, freq=step, inclusive="left")

def choose_reference_point(points: Sequence[dict]) -> dict:
    """
    Choose a representative point to derive the global time axis.
    Assumes all points share the same timing metadata.
    """
    return points[0]
