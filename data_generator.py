"""
generate_gpx_to_csv.py
Generates synthetic GPX run data and exports each run to a CSV file.
"""

import os
import math
import random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

OUTPUT_DIR = "run_csvs"
N_RUNS = 20
SEED = 42
START_LAT = -7.966620   # Malang, East Java
START_LON = 112.632632

RUN_PROFILES = {
    "easy":     (6.5, 0.4, 8,  6.0),
    "tempo":    (4.8, 0.2, 5,  8.0),
    "interval": (4.2, 1.2, 3,  5.0),
    "long":     (6.8, 0.5, 12, 18.0),
}

# ---------------------------------------------------------------------------
# Route simulation
# ---------------------------------------------------------------------------

def simulate_route(start_lat, start_lon, num_points, total_distance_km, elevation_gain):
    lats, lons, elevs = [start_lat], [start_lon], [50.0]
    metres_per_deg_lat = 111_320
    step_m = (total_distance_km * 1000) / num_points
    bearing = random.uniform(0, 360)
    angle_rad = math.radians(bearing)
    half = num_points // 2

    for i in range(1, num_points):
        if i == half:
            back_lat = start_lat - lats[-1]
            back_lon = start_lon - lons[-1]
            angle_rad = math.atan2(back_lon, back_lat)

        angle_rad += random.gauss(0, 0.08)
        d_lat = step_m * math.cos(angle_rad) / metres_per_deg_lat
        metres_per_deg_lon = metres_per_deg_lat * math.cos(math.radians(lats[-1]))
        d_lon = step_m * math.sin(angle_rad) / metres_per_deg_lon

        lats.append(lats[-1] + d_lat)
        lons.append(lons[-1] + d_lon)

        if i < half:
            elev_change = (elevation_gain / half) + random.gauss(0, 1.5)
        else:
            elev_change = -(elevation_gain / (num_points - half)) + random.gauss(0, 1.5)
        elevs.append(max(0.0, elevs[-1] + elev_change))

    return list(zip(lats, lons, elevs))

# ---------------------------------------------------------------------------
# Derived metrics
# ---------------------------------------------------------------------------

def compute_metrics(df):
    """Add distance_m, speed_kmh, and pace_min_km columns to a run DataFrame."""
    R = 6371000  # Earth radius in metres

    lats = np.radians(df["lat"].values)
    lons = np.radians(df["lon"].values)

    dlat = np.diff(lats, prepend=lats[0])
    dlon = np.diff(lons, prepend=lons[0])

    a = np.sin(dlat / 2) ** 2 + np.cos(lats) * np.cos(np.roll(lats, 1)) * np.sin(dlon / 2) ** 2
    a[0] = 0
    dist_m = 2 * R * np.arcsin(np.sqrt(a))

    elapsed_s = df["timestamp"].diff().dt.total_seconds().fillna(0).values
    elapsed_s[elapsed_s == 0] = 1e-6  # avoid division by zero

    speed_ms = dist_m / elapsed_s
    speed_kmh = speed_ms * 3.6
    pace_min_km = np.where(speed_ms > 0.1, (1 / (speed_ms * 60 / 1000)), np.nan)

    df = df.copy()
    df["distance_m"]    = dist_m.round(2)
    df["cum_distance_m"] = dist_m.cumsum().round(2)
    df["speed_kmh"]     = speed_kmh.round(3)
    df["pace_min_km"]   = np.round(pace_min_km, 3)
    return df

# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

def generate_and_export(
    n_runs=N_RUNS,
    seed=SEED,
    start_lat=START_LAT,
    start_lon=START_LON,
    output_dir=OUTPUT_DIR,
):
    random.seed(seed)
    np.random.seed(seed)
    os.makedirs(output_dir, exist_ok=True)

    run_types = list(RUN_PROFILES.keys())
    base_time = datetime(2024, 1, 1, 6, 0, 0)
    summary_rows = []

    for i in range(n_runs):
        rtype = run_types[i % len(run_types)]
        avg_pace, pace_var, elev_per_km, base_dist = RUN_PROFILES[rtype]

        distance_km    = base_dist * random.uniform(0.7, 1.3)
        elevation_gain = elev_per_km * distance_km * random.uniform(0.6, 1.4)
        num_points     = max(20, int(distance_km * 1000 / 25))

        route = simulate_route(
            start_lat + random.uniform(-0.02, 0.02),
            start_lon + random.uniform(-0.02, 0.02),
            num_points,
            distance_km,
            elevation_gain,
        )

        timestamps = [base_time]
        for _ in range(1, num_points):
            pace = max(3.0, np.random.normal(avg_pace, pace_var))
            # pace in min/km -> seconds per segment
            seconds_per_segment = (pace * 60) * (distance_km / num_points)
            timestamps.append(timestamps[-1] + timedelta(seconds=seconds_per_segment))

        df = pd.DataFrame(route, columns=["lat", "lon", "elevation"])
        df["timestamp"] = pd.to_datetime(timestamps, utc=True)
        df["run_type"]  = rtype
        df["run_id"]    = i + 1
        df = compute_metrics(df)

        fname = f"run_{i+1:02d}_{rtype}.csv"
        fpath = os.path.join(output_dir, fname)
        df.to_csv(fpath, index=False)
        print(f"  Saved {fname}  ({len(df)} points, {distance_km:.1f} km)")

        # Collect summary stats
        summary_rows.append({
            "run_id":           i + 1,
            "filename":         fname,
            "run_type":         rtype,
            "date":             base_time.strftime("%Y-%m-%d"),
            "distance_km":      round(distance_km, 2),
            "elevation_gain_m": round(elevation_gain, 1),
            "duration_min":     round((timestamps[-1] - timestamps[0]).total_seconds() / 60, 1),
            "avg_pace_min_km":  round(df["pace_min_km"].median(), 2),
            "avg_speed_kmh":    round(df["speed_kmh"].median(), 2),
            "num_points":       len(df),
        })

        base_time += timedelta(days=random.randint(2, 4), hours=random.randint(-1, 1))

    # Save summary
    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(output_dir, "_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"\n Summary saved → {summary_path}")
    print(summary_df.to_string(index=False))

# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"Generating {N_RUNS} synthetic runs → '{OUTPUT_DIR}/'...\n")
    generate_and_export()