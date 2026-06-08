"""
data_loader.py
==============
Handles loading GPX files from upload or local folder,
parsing coordinates, elevation, and timestamps into DataFrames.
Also generates synthetic GPX data for demo purposes.
"""

import os
import io
import math
import random
import numpy as np
import pandas as pd
import gpxpy
import gpxpy.gpx
from datetime import datetime, timedelta
from typing import Optional


# ---------------------------------------------------------------------------
# GPX Parsing
# ---------------------------------------------------------------------------

def parse_gpx_file(file_obj) -> Optional[pd.DataFrame]:
    """
    Parse a GPX file object (BytesIO or file path) into a DataFrame.

    Returns a DataFrame with columns:
        lat, lon, elevation, timestamp
    Returns None if parsing fails.
    """
    try:
        if isinstance(file_obj, (str, os.PathLike)):
            with open(file_obj, "r") as f:
                gpx = gpxpy.parse(f)
        else:
            content = file_obj.read()
            gpx = gpxpy.parse(io.BytesIO(content))

        rows = []
        for track in gpx.tracks:
            for segment in track.segments:
                for point in segment.points:
                    rows.append({
                        "lat": point.latitude,
                        "lon": point.longitude,
                        "elevation": point.elevation if point.elevation is not None else 0.0,
                        "timestamp": point.time,
                    })

        if not rows:
            return None

        df = pd.DataFrame(rows)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        return df

    except Exception as e:
        print(f"[data_loader] Error parsing GPX: {e}")
        return None


def load_gpx_folder(folder_path: str, max_files: int = 100) -> dict[str, pd.DataFrame]:
    """
    Load all .gpx files from a local folder.

    Returns a dict mapping filename → parsed DataFrame.
    """
    runs = {}
    if not os.path.isdir(folder_path):
        return runs

    gpx_files = [f for f in os.listdir(folder_path) if f.lower().endswith(".gpx")]
    gpx_files = sorted(gpx_files)[:max_files]

    for fname in gpx_files:
        fpath = os.path.join(folder_path, fname)
        df = parse_gpx_file(fpath)
        if df is not None and len(df) > 0:
            runs[fname] = df

    return runs



# ---------------------------------------------------------------------------
# Synthetic GPX Generator
# ---------------------------------------------------------------------------

# Run type profiles: (avg_pace_min_km, pace_variability, elevation_per_km, typical_distance_km)
RUN_PROFILES = {
    "easy":     (6.5, 0.4, 8,  6.0),
    "tempo":    (4.8, 0.2, 5,  8.0),
    "interval": (4.2, 1.2, 3,  5.0),
    "long":     (6.8, 0.5, 12, 18.0),
}


def _simulate_route(
    start_lat: float,
    start_lon: float,
    num_points: int,
    total_distance_km: float,
    elevation_gain: float,
) -> list[tuple]:
    """
    Simulate a realistic running route as a list of (lat, lon, elevation) tuples.
    Uses a random-walk with drift to create a looped course.
    """
    lats, lons, elevs = [start_lat], [start_lon], [50.0]

    # Metres per degree of latitude ≈ 111_320
    metres_per_deg_lat = 111_320
    step_m = (total_distance_km * 1000) / num_points

    # Random bearing drift — creates realistic curved routes
    bearing = random.uniform(0, 360)
    angle_rad = math.radians(bearing)

    half = num_points // 2
    for i in range(1, num_points):
        # Gradually return toward start in second half for a looped course
        if i == half:
            back_lat = start_lat - lats[-1]
            back_lon = start_lon - lons[-1]
            angle_rad = math.atan2(back_lon, back_lat)

        angle_rad += random.gauss(0, 0.08)  # gentle bearing drift
        d_lat = step_m * math.cos(angle_rad) / metres_per_deg_lat
        metres_per_deg_lon = metres_per_deg_lat * math.cos(math.radians(lats[-1]))
        d_lon = step_m * math.sin(angle_rad) / metres_per_deg_lon

        lats.append(lats[-1] + d_lat)
        lons.append(lons[-1] + d_lon)

        # Elevation: rises in first half, falls in second
        if i < half:
            elev_change = (elevation_gain / half) + random.gauss(0, 1.5)
        else:
            elev_change = -(elevation_gain / (num_points - half)) + random.gauss(0, 1.5)
        elevs.append(max(0.0, elevs[-1] + elev_change))

    return list(zip(lats, lons, elevs))


def generate_synthetic_runs(
    n_runs: int = 50,
    seed: int = 42,
    start_lat: float = -7.966620,   # Malang, East Java
    start_lon: float = 112.632632,
) -> dict[str, pd.DataFrame]:
    """
    Generate n_runs synthetic GPX DataFrames covering all four run types.

    Returns dict: filename → DataFrame with columns (lat, lon, elevation, timestamp)
    """
    random.seed(seed)
    np.random.seed(seed)

    run_types = list(RUN_PROFILES.keys())
    runs = {}
    base_time = datetime(2024, 1, 1, 6, 0, 0)

    for i in range(n_runs):
        rtype = run_types[i % len(run_types)]
        avg_pace, pace_var, elev_per_km, base_dist = RUN_PROFILES[rtype]

        # Randomise distance within ±30 % of profile baseline
        distance_km = base_dist * random.uniform(0.7, 1.3)
        elevation_gain = elev_per_km * distance_km * random.uniform(0.6, 1.4)

        # 1 GPS point roughly every 5 seconds
        num_points = max(20, int(distance_km * 1000 / 25))

        route = _simulate_route(
            start_lat + random.uniform(-0.02, 0.02),
            start_lon + random.uniform(-0.02, 0.02),
            num_points,
            distance_km,
            elevation_gain,
        )

        # Build timestamps from paces (with random variability)
        timestamps = [base_time]
        for _ in range(1, num_points):
            pace = max(3.0, np.random.normal(avg_pace, pace_var))
            # pace in min/km -> seconds per segment
            seconds_per_segment = (pace * 60) * (distance_km / num_points)
            timestamps.append(timestamps[-1] + timedelta(seconds=seconds_per_segment))

        df = pd.DataFrame(route, columns=["lat", "lon", "elevation"])
        df["timestamp"] = pd.to_datetime(timestamps, utc=True)

        # Advance base_time by 2-4 days for next run
        base_time += timedelta(days=random.randint(2, 4), hours=random.randint(-1, 1))

        fname = f"run_{i+1:02d}_{rtype}.gpx"
        runs[fname] = df

    return runs
