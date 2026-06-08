"""
gpx_exporter.py
===============
Converts a parsed GPX DataFrame (lat, lon, elevation, timestamp)
back into a standards-compliant GPX 1.1 XML string ready for
download via Streamlit's st.download_button.

Also provides a helper to zip multiple runs into a single archive.
"""

import io
import zipfile
from datetime import timezone
import pandas as pd
import gpxpy
import gpxpy.gpx


# ---------------------------------------------------------------------------
# Single-run export
# ---------------------------------------------------------------------------

def dataframe_to_gpx_bytes(
    df: pd.DataFrame,
    track_name: str = "Synthetic Run",
    track_type: str = "running",
) -> bytes:
    """
    Convert a GPX DataFrame back to GPX 1.1 XML bytes.

    Parameters
    ----------
    df          : DataFrame with columns lat, lon, elevation, timestamp
    track_name  : Name embedded in the <name> tag of the GPX track
    track_type  : Activity type string (e.g. 'running', 'cycling')

    Returns
    -------
    UTF-8 encoded GPX XML bytes suitable for st.download_button
    """
    gpx = gpxpy.gpx.GPX()
    gpx.creator = "GPX Running Analytics — Streamlit App"

    track = gpxpy.gpx.GPXTrack()
    track.name = track_name
    track.type = track_type
    gpx.tracks.append(track)

    segment = gpxpy.gpx.GPXTrackSegment()
    track.segments.append(segment)

    for _, row in df.iterrows():
        ts = row.get("timestamp", None)

        if ts is not None:
            ts = pd.to_datetime(ts)
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            else:
                ts = ts.astimezone(timezone.utc)
            ts = ts.to_pydatetime()

        point = gpxpy.gpx.GPXTrackPoint(
            latitude=float(row["lat"]),
            longitude=float(row["lon"]),
            elevation=float(row.get("elevation", 0.0)),
            time=ts,
        )
        segment.points.append(point)

    xml_str = gpx.to_xml(version="1.1")
    return xml_str.encode("utf-8")


# ---------------------------------------------------------------------------
# Multi-run ZIP export
# ---------------------------------------------------------------------------

def runs_to_zip_bytes(
    runs: dict,
    name_map: dict = None,
) -> bytes:
    """
    Pack multiple GPX DataFrames into a single in-memory ZIP archive.

    Parameters
    ----------
    runs     : dict mapping filename -> DataFrame
    name_map : optional dict mapping filename -> track display name

    Returns
    -------
    ZIP file bytes suitable for st.download_button
    """
    buf = io.BytesIO()

    with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for fname, df in runs.items():
            if name_map and fname in name_map:
                track_name = name_map[fname]
            else:
                track_name = fname.replace(".gpx", "").replace("_", " ").title()

            track_type = "running"
            for rtype in ("easy", "tempo", "interval", "long"):
                if rtype in fname.lower():
                    track_type = f"running:{rtype}"
                    break

            gpx_bytes = dataframe_to_gpx_bytes(df, track_name=track_name, track_type=track_type)
            zip_entry = fname if fname.endswith(".gpx") else fname + ".gpx"
            zf.writestr(zip_entry, gpx_bytes)

    buf.seek(0)
    return buf.read()


# ---------------------------------------------------------------------------
# Convenience: single-run bytes with auto track name from filename
# ---------------------------------------------------------------------------

def run_to_gpx_bytes(fname: str, df: pd.DataFrame) -> bytes:
    """Derives track_name and track_type automatically from the filename."""
    track_name = fname.replace(".gpx", "").replace("_", " ").title()
    track_type = "running"
    for rtype in ("easy", "tempo", "interval", "long"):
        if rtype in fname.lower():
            track_type = f"running:{rtype}"
            break
    return dataframe_to_gpx_bytes(df, track_name=track_name, track_type=track_type)
