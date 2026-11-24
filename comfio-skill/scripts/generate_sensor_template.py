#!/usr/bin/env python3
"""Generate a synthetic sensor CSV for testing comfio pipelines.

Usage:
    python generate_sensor_template.py --output sensor.csv --rows 168
"""

import argparse

import numpy as np
import pandas as pd


def generate_sensor_data(rows: int = 168) -> pd.DataFrame:
    rng = np.random.default_rng(seed=42)
    hours = pd.date_range("2025-01-01", periods=rows, freq="h")

    base_temp = 23.0
    diurnal = 2.0 * np.sin(2 * np.pi * np.arange(rows) / 24.0)

    tdb = base_temp + diurnal + rng.normal(0, 0.5, rows)
    tr = tdb + rng.normal(-0.5, 0.3, rows)
    rh = np.clip(50.0 + rng.normal(0, 5, rows), 20, 80)
    vr = np.clip(0.1 + rng.normal(0, 0.03, rows), 0.05, 0.3)
    illuminance = np.clip(
        500.0 + 200.0 * np.sin(2 * np.pi * np.arange(rows) / 24.0)
        + rng.normal(0, 30, rows),
        0,
        1200,
    )
    co2 = np.clip(
        700.0
        + 300.0 * np.sin(2 * np.pi * np.arange(rows) / 24.0 + np.pi)
        + rng.normal(0, 50, rows),
        400,
        2000,
    )
    laeq = np.clip(35.0 + rng.normal(0, 5, rows), 25, 65)

    pm25 = np.clip(10.0 + rng.normal(0, 3, rows), 0, 75)
    tvoc = np.clip(200.0 + rng.normal(0, 50, rows), 0, 1000)
    formaldehyde = np.clip(25.0 + rng.normal(0, 5, rows), 0, 100)
    co = np.clip(2.0 + rng.normal(0, 0.5, rows), 0, 15)

    df = pd.DataFrame(
        {
            "timestamp": hours,
            "tdb": np.round(tdb, 2),
            "tr": np.round(tr, 2),
            "rh": np.round(rh, 1),
            "vr": np.round(vr, 3),
            "illuminance": np.round(illuminance, 1),
            "co2": np.round(co2, 1),
            "laeq": np.round(laeq, 1),
            "pm25": np.round(pm25, 2),
            "tvoc": np.round(tvoc, 1),
            "formaldehyde": np.round(formaldehyde, 1),
            "co": np.round(co, 2),
        }
    )
    return df


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate synthetic sensor CSV")
    parser.add_argument("--output", "-o", default="sensor_data.csv", help="Output CSV path")
    parser.add_argument("--rows", "-n", type=int, default=168, help="Number of rows (default: 168 = 1 week hourly)")
    args = parser.parse_args()

    df = generate_sensor_data(args.rows)
    df.to_csv(args.output, index=False)
    print(f"Generated {len(df)} rows → {args.output}")
    print(f"Columns: {list(df.columns)}")
    print(f"Time range: {df['timestamp'].iloc[0]} → {df['timestamp'].iloc[-1]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
