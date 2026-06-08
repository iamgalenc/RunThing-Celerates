"""
feature_engineering.py
=======================
Derives all running metrics from raw GPX DataFrames.

Features computed:
  - distance (Haversine, metres & km)
  - speed (km/h) and pace (min/km)
  - elevation gain / loss
  - rolling pace (smoothed)
  - segment pace variability
  - run-level summary stats used as ML features
"""

import math
import numpy as np
import pandas as pd
from typing import Optional


# ---------------------------------------------------------------------------
# Haversine Distance
# ---------------------------------------------------------------------------

def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Return distance in metres between two WGS-84 coordinates."""
    R = 6_371_000  # Earth radius in metres
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))


# ---------------------------------------------------------------------------
# Point-level Feature Engineering
# ---------------------------------------------------------------------------

def compute_point_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add point-by-point derived columns to a GPX DataFrame.

    Input columns required: lat, lon, elevation, timestamp
    Output adds:
        segment_dist_m   – distance from previous point (m)
        cumulative_dist_m – cumulative distance (m)
        dist_km          – cumulative distance (km)
        elapsed_s        – seconds since run start
        speed_kmh        – instantaneous speed (km/h)
        pace_min_km      – instantaneous pace (min/km); clipped to [2, 20]
        elev_diff        – elevation change from previous point (m)
        elev_gain        – cumulative positive elevation (m)
        elev_loss        – cumulative negative elevation (m)
        rolling_pace     – 10-point rolling mean of pace
        pace_variability – rolling std of pace (10-point window)
    """
    df = df.copy().reset_index(drop=True)

    # ---- segment distances ------------------------------------------------
    lats = df["lat"].values
    lons = df["lon"].values
    seg_dist = np.zeros(len(df))
    for i in range(1, len(df)):
        seg_dist[i] = haversine_m(lats[i - 1], lons[i - 1], lats[i], lons[i])

    df["segment_dist_m"] = seg_dist
    df["cumulative_dist_m"] = seg_dist.cumsum()
    df["dist_km"] = df["cumulative_dist_m"] / 1000.0

    # ---- time features ----------------------------------------------------
    ts = pd.to_datetime(df["timestamp"])
    t0 = ts.iloc[0]
    df["elapsed_s"] = (ts - t0).dt.total_seconds()

    # ---- speed & pace (per segment) ---------------------------------------
    dt = df["elapsed_s"].diff().fillna(1).clip(lower=0.1)  # seconds, avoid /0
    ds_km = df["segment_dist_m"] / 1000.0

    df["speed_kmh"] = (ds_km / (dt / 3600)).clip(0, 40)
    # pace = minutes per km; avoid division by zero
    with np.errstate(divide="ignore", invalid="ignore"):
        raw_pace = np.where(df["speed_kmh"] > 0.5, 60.0 / df["speed_kmh"], np.nan)
    df["pace_min_km"] = pd.Series(raw_pace).clip(2, 20)

    # ---- elevation features ----------------------------------------------
    elev = df["elevation"].values
    elev_diff = np.diff(elev, prepend=elev[0])
    df["elev_diff"] = elev_diff
    df["elev_gain"] = np.where(elev_diff > 0, elev_diff, 0).cumsum()
    df["elev_loss"] = np.where(elev_diff < 0, -elev_diff, 0).cumsum()

    # ---- rolling stats ----------------------------------------------------
    df["rolling_pace"] = (
        df["pace_min_km"]
        .rolling(window=10, min_periods=1, center=True)
        .mean()
    )
    df["pace_variability"] = (
        df["pace_min_km"]
        .rolling(window=10, min_periods=2, center=True)
        .std()
        .fillna(0)
    )

    return df


# ---------------------------------------------------------------------------
# Run-level Summary
# ---------------------------------------------------------------------------

RUN_TYPE_MAP = {
    "easy": 0,
    "tempo": 1,
    "interval": 2,
    "long": 3,
}
RUN_TYPE_INV = {v: k for k, v in RUN_TYPE_MAP.items()}


def infer_run_type_from_filename(fname: str) -> str:
    """Guess run type from filename (works with synthetic names)."""
    fname_lower = fname.lower()
    for rtype in RUN_TYPE_MAP:
        if rtype in fname_lower:
            return rtype
    return "easy"


def compute_run_summary(
    df: pd.DataFrame,
    run_name: str = "unknown",
    run_type: Optional[str] = None,
) -> dict:
    """
    Collapse a point-level DataFrame into a single-row summary dict.

    Used to build the per-run feature matrix for ML models.
    """
    if "dist_km" not in df.columns:
        df = compute_point_features(df)

    total_dist_km = df["dist_km"].iloc[-1] if len(df) > 1 else 0.0
    total_elevation_gain = df["elev_gain"].iloc[-1] if "elev_gain" in df.columns else 0.0
    total_elevation_loss = df["elev_loss"].iloc[-1] if "elev_loss" in df.columns else 0.0
    total_duration_s = df["elapsed_s"].iloc[-1] if "elapsed_s" in df.columns else 0.0
    total_duration_min = total_duration_s / 60.0

    avg_pace = df["pace_min_km"].dropna().mean()
    median_pace = df["pace_min_km"].dropna().median()
    std_pace = df["pace_min_km"].dropna().std()
    avg_speed = df["speed_kmh"].mean()
    avg_rolling_pace = df["rolling_pace"].dropna().mean() if "rolling_pace" in df.columns else avg_pace
    avg_pace_variability = df["pace_variability"].dropna().mean() if "pace_variability" in df.columns else 0.0

    # Infer run type if not provided
    if run_type is None:
        run_type = infer_run_type_from_filename(run_name)

    run_type_code = RUN_TYPE_MAP.get(run_type, 0)

    # Fatigue proxy: compare avg pace in second half vs first half
    midpoint = len(df) // 2
    pace_first = df["pace_min_km"].iloc[:midpoint].mean()
    pace_second = df["pace_min_km"].iloc[midpoint:].mean()
    fatigue_index = pace_second - pace_first  # positive = slowing down

    return {
        "run_name": run_name,
        "run_type": run_type,
        "run_type_code": run_type_code,
        "total_dist_km": round(total_dist_km, 3),
        "total_elevation_gain_m": round(total_elevation_gain, 1),
        "total_elevation_loss_m": round(total_elevation_loss, 1),
        "total_duration_min": round(total_duration_min, 2),
        "avg_pace_min_km": round(avg_pace, 3),
        "median_pace_min_km": round(median_pace, 3),
        "std_pace_min_km": round(std_pace, 3),
        "avg_speed_kmh": round(avg_speed, 3),
        "avg_rolling_pace": round(avg_rolling_pace, 3),
        "avg_pace_variability": round(avg_pace_variability, 3),
        "fatigue_index": round(fatigue_index, 3),
        "elev_gain_per_km": round(total_elevation_gain / max(total_dist_km, 0.01), 2),
        "num_points": len(df),
    }


# ---------------------------------------------------------------------------
# Build Feature Matrix from Multiple Runs
# ---------------------------------------------------------------------------

def build_feature_matrix(runs_raw: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Given a dict of {filename: raw_gpx_df}, return a single DataFrame where
    each row is a run with all engineered summary features.
    """
    summaries = []
    for fname, raw_df in runs_raw.items():
        run_type = infer_run_type_from_filename(fname)
        enriched = compute_point_features(raw_df)
        summary = compute_run_summary(enriched, run_name=fname, run_type=run_type)
        summaries.append(summary)

    return pd.DataFrame(summaries)
