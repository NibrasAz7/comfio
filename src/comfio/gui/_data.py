"""Data loading helpers for the comfio Streamlit app.

Reads CSV/Excel uploads into DataFrames, provides packaged demo data,
and generates downloadable CSV column templates.
"""

from __future__ import annotations

import io
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Human-readable parameter descriptions for the column mapper and templates.
COLUMN_DESCRIPTIONS: dict[str, str] = {
    "air_temp_c": "Air temperature [°C]",
    "radiant_temp_c": "Mean radiant temperature [°C]",
    "relative_humidity_pct": "Relative humidity [%]",
    "air_velocity_ms": "Air speed [m/s]",
    "illuminance_lux": "Illuminance [lux]",
    "co2_ppm": "CO₂ concentration [ppm]",
    "noise_laeq_db": "Sound level LAeq [dB]",
    "metabolic_rate_met": "Metabolic rate [met]",
    "clothing_insulation_clo": "Clothing insulation [clo]",
    "pm25_ugm3": "PM2.5 [µg/m³]",
    "pm10_ugm3": "PM10 [µg/m³]",
    "tvoc_ugm3": "TVOC [µg/m³]",
    "formaldehyde_ppb": "Formaldehyde [ppb]",
    "co_ppm": "Carbon monoxide [ppm]",
    "outdoor_temp_c": "Outdoor air temperature [°C]",
    "prevailing_mean_outdoor_c": "Prevailing mean outdoor temp [°C]",
    "running_mean_outdoor_c": "Running mean outdoor temp [°C]",
    "ugr": "Unified Glare Rating [-]",
    "tsv": "Thermal sensation vote [-3..+3]",
}


def is_excel(name: str) -> bool:
    """True if the filename looks like an Excel workbook."""
    return name.lower().endswith((".xlsx", ".xls"))


def excel_sheet_names(file: Any) -> list[str]:
    """Return the sheet names of an uploaded Excel workbook."""
    return list(pd.ExcelFile(file).sheet_names)


def read_uploaded(file: Any, sheet: str | int = 0) -> pd.DataFrame:
    """Read an uploaded CSV/Excel file into a DataFrame.

    Parameters
    ----------
    sheet : str or int
        Excel sheet to read (name or index); ignored for CSV.
    """
    name = str(file.name).lower()
    if is_excel(name):
        return pd.read_excel(file, sheet_name=sheet)
    if name.endswith(".csv"):
        return pd.read_csv(file)
    raise ValueError(f"Unsupported file type: {file.name} (use .csv, .xlsx or .xls)")


# Canonical temperature / velocity columns converted when the user
# declares the file is in IP units.
_IP_TEMP_COLS = (
    "air_temp_c",
    "radiant_temp_c",
    "outdoor_temp_c",
    "prevailing_mean_outdoor_c",
    "running_mean_outdoor_c",
)
_IP_VEL_COLS = ("air_velocity_ms",)


def convert_ip_to_si(df: pd.DataFrame, mapping: dict[str, str]) -> pd.DataFrame:
    """Convert mapped IP-unit columns (°F, fpm) to SI (°C, m/s) in a copy."""
    out = df.copy()
    for canonical in _IP_TEMP_COLS:
        col = mapping.get(canonical)
        if col is not None and col in out.columns:
            out[col] = (pd.to_numeric(out[col], errors="coerce") - 32.0) * 5.0 / 9.0
    for canonical in _IP_VEL_COLS:
        col = mapping.get(canonical)
        if col is not None and col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce") * 0.00508
    return out


def load_preloaded() -> pd.DataFrame | None:
    """Load a DataFrame passed via ``COMFIO_GUI_DATA`` env var (comfio.gui(df))."""
    path = os.environ.get("COMFIO_GUI_DATA")
    if path and Path(path).exists():
        return pd.read_pickle(path)
    return None


def demo_data(n: int = 168, seed: int = 7) -> pd.DataFrame:
    """Generate a realistic demo dataset (1 week of hourly data).

    Deterministic so the app behaves identically for every user.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    hours = pd.date_range("2026-03-01", periods=n, freq="h")
    diurnal = 3.0 * np.sin(2 * np.pi * t / 24 - 1.2)
    occupied = ((hours.hour >= 8) & (hours.hour <= 18)).astype(float)

    df = pd.DataFrame(
        {
            "timestamp": hours,
            "air_temp_c": 21.5 + diurnal + rng.normal(0, 0.4, n),
            "radiant_temp_c": 21.8 + diurnal + rng.normal(0, 0.4, n),
            "relative_humidity_pct": np.clip(48 - 3 * diurnal + rng.normal(0, 3, n), 20, 90),
            "air_velocity_ms": np.clip(0.08 + rng.normal(0, 0.03, n), 0.01, None),
            "co2_ppm": np.clip(450 + 500 * occupied + rng.normal(0, 60, n), 400, None),
            "illuminance_lux": np.clip(450 * occupied + 30 + rng.normal(0, 40, n), 0, None),
            "noise_laeq_db": np.clip(36 + 8 * occupied + rng.normal(0, 3, n), 20, None),
            "pm25_ugm3": np.clip(8 + rng.normal(0, 3, n), 0, None),
            "outdoor_temp_c": 10 + 6 * np.sin(2 * np.pi * t / 168) + diurnal * 0.5,
            "metabolic_rate_met": 1.2,
            "clothing_insulation_clo": 0.8,
        }
    )
    # Sparse occupant votes — a realistic ~1-in-4 sampling of occupied hours.
    tsv = np.full(n, np.nan)
    vote_idx = np.where((occupied == 1) & (t % 4 == 0))[0]
    tsv[vote_idx] = np.clip(
        0.35 * (df["air_temp_c"].to_numpy()[vote_idx] - 22.0) + rng.normal(0, 0.4, len(vote_idx)),
        -3,
        3,
    ).round()
    df["tsv"] = tsv
    return df


def template_csv(units: str = "si") -> str:
    """Return a downloadable column-template CSV as a string.

    Parameters
    ----------
    units : {"si", "ip"}
        Header unit labels; data values are illustrative only.
    """
    if units == "ip":
        header = (
            "timestamp,Air temperature [F],MRT [F],Air speed [fpm],"
            "Relative humidity [%],CO2 [ppm],Illuminance [lux],"
            "Sound level LAeq [dB],Outdoor temperature [F]\n"
        )
        row = "2026-01-01 09:00,72,72,20,45,700,500,38,50\n"
    else:
        header = (
            "timestamp,air_temp_c,radiant_temp_c,air_velocity_ms,"
            "relative_humidity_pct,co2_ppm,illuminance_lux,"
            "noise_laeq_db,outdoor_temp_c\n"
        )
        row = "2026-01-01 09:00,22,22,0.10,45,700,500,38,10\n"
    return header + row


def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue().encode("utf-8")


def prepare(
    df: pd.DataFrame,
    mapping: dict[str, str],
    config: dict,
    nan_strategy: str = "interpolate",
    ip_units: bool = False,
) -> tuple[pd.DataFrame, dict[str, str], list[str]]:
    """Inject assumption columns + apply NaN strategy before SensorData.

    Returns ``(df, mapping, notes)`` where ``notes`` lists every
    assumption injected — the UI must display these, never hide them.
    """
    df = df.copy()
    mapping = dict(mapping)
    notes: list[str] = []

    if ip_units:
        df = convert_ip_to_si(df, mapping)
        notes.append("Input declared in IP units — converted °F→°C and fpm→m/s.")

    if "radiant_temp_c" not in mapping and "air_temp_c" in mapping:
        if config.get("_mrt_equals_tdb", True):
            syn = "__comfio_radiant_temp_c"
            df[syn] = df[mapping["air_temp_c"]]
            mapping["radiant_temp_c"] = syn
            notes.append("Radiant temperature assumed equal to air temperature.")
        else:
            notes.append("No radiant temperature — thermal PMV unavailable.")

    if "air_velocity_ms" not in mapping:
        syn = "__comfio_air_velocity_ms"
        df[syn] = float(config.get("vr", 0.1))
        mapping["air_velocity_ms"] = syn
        notes.append(f"Air speed set to constant {config.get('vr', 0.1)} m/s.")

    if "outdoor_temp_c" not in mapping and "prevailing_mean_outdoor_c" not in mapping:
        notes.append("No outdoor temperature — adaptive thermal model skipped.")

    mapped_cols = [c for c in mapping.values() if c in df.columns]
    num = df[mapped_cols].apply(pd.to_numeric, errors="coerce")
    if nan_strategy == "drop":
        df = df.loc[num.dropna().index].reset_index(drop=True)
    elif nan_strategy == "fill_zero":
        df[mapped_cols] = num.fillna(0.0)
    else:  # interpolate
        df[mapped_cols] = num.interpolate().ffill().bfill()
    return df, mapping, notes


def attach_outdoor_temperature(
    df: pd.DataFrame, ts_col: str, lat: float, lon: float
) -> pd.DataFrame:
    """Fetch meteostat hourly outdoor temp and merge onto the file's timestamps.

    Nearest-hour match with 1 h tolerance; unmatched rows get NaN.
    Adds an ``outdoor_temp_c`` column (canonical name, auto-detected).
    """
    from comfio.integration.weather import fetch_outdoor_temperature

    ts = pd.to_datetime(df[ts_col])
    if ts.dt.tz is not None:
        ts = ts.dt.tz_convert("UTC").dt.tz_localize(None)
    wx = fetch_outdoor_temperature(lat, lon, ts.min().date(), ts.max().date())
    wx_idx = wx.index.tz_localize(None) if wx.index.tz is not None else wx.index
    wx_df = pd.DataFrame({"ts": wx_idx, "outdoor_temp_c": wx["temp"].to_numpy()})

    left = pd.DataFrame({"ts": ts, "_orig": np.arange(len(df))}).sort_values("ts")
    merged = pd.merge_asof(
        left,
        wx_df.sort_values("ts"),
        on="ts",
        direction="nearest",
        tolerance=pd.Timedelta("1h"),
    ).sort_values("_orig")
    out = df.copy()
    out["outdoor_temp_c"] = merged["outdoor_temp_c"].to_numpy()
    return out
