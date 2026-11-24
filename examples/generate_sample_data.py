"""Generate examples/sample_sensor_data.csv — a small, realistic multi-domain
IEQ dataset for quickly trying out comfio (SensorData, run_pipeline, the GUI, etc.)
without waiting for the full 6-month walkthrough dataset to be generated.

Uses the same additive seasonal + diurnal + Gaussian-noise decomposition as
docs/tutorials/06_walkthrough.md (see build_walkthrough_nb.py, Section 1),
but scoped to 7 days at 10-minute sampling (1,008 rows) with a fixed seed
(42) for reproducibility.

Usage
-----
    python examples/generate_sample_data.py

Regenerate whenever the canonical column set changes.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

rng = np.random.default_rng(42)

# 10-minute sampling for 1 week (Jan 1-7, 2025) -> 1,008 rows
start = datetime(2025, 1, 1, 0, 0, 0)
end = datetime(2025, 1, 7, 23, 50, 0)
ts_10min = pd.date_range(start, end, freq="10min")
n = len(ts_10min)

# Day-of-year fraction (0..1) for seasonal envelope (kept for formula parity
# with the 6-month walkthrough dataset; over 7 days this stays near 0)
doy = ts_10min.dayofyear.to_numpy()
seasonal = (doy - 1) / 365.0
hour = ts_10min.hour.to_numpy() + ts_10min.minute.to_numpy() / 60.0
diurnal = np.sin(2 * np.pi * (hour - 9.0) / 24.0)  # peak ~15:00, trough ~03:00
is_day = ((hour > 7) & (hour < 18)).astype(float)

# --- Indoor air temperature (degC): winter baseline + diurnal ---
air_temp = 20.0 + 7.0 * seasonal + 1.5 * diurnal + rng.normal(0, 0.4, n)

# --- Radiant temperature: ~air_temp + small solar offset during day ---
radiant = air_temp + 0.5 * is_day * diurnal + rng.normal(0, 0.3, n)

# --- Relative humidity (%): anti-correlated with temp ---
rh = 55.0 - 15.0 * seasonal - 5.0 * diurnal + rng.normal(0, 2.0, n)
rh = np.clip(rh, 15, 85)

# --- Air velocity (m/s): mostly stagnant ---
vr = 0.10 + rng.normal(0, 0.02, n)
vr = np.clip(vr, 0.02, 0.5)

# --- Illuminance (lux): daylight sinusoid + seasonal ---
daylight = np.maximum(0, np.sin(np.pi * (hour - 6) / 12))  # 0 at 06/18, peak 12
lux = daylight * (400 + 400 * seasonal) + rng.normal(0, 30, n)
lux = np.clip(lux, 0, 1200)

# --- Noise L_Aeq (dBA): occupancy-driven ---
noise = 38.0 + 10.0 * is_day + 4.0 * diurnal * is_day + rng.normal(0, 1.5, n)

# --- CO2 (ppm): baseline + occupancy ---
co2 = 420 + 600 * is_day + 150 * diurnal * is_day + rng.normal(0, 40, n)
co2 = np.clip(co2, 380, 1800)

# --- Pollutants ---
pm25 = 5 + 13 * seasonal + 3 * is_day + rng.normal(0, 1.0, n)
pm10 = pm25 + 4 + rng.normal(0, 1.5, n)
tvoc = 100 + 200 * seasonal + 50 * is_day + rng.normal(0, 15, n)
hcho = 15 + 20 * seasonal + 5 * is_day + rng.normal(0, 2.0, n)
co = 0.8 + 1.7 * seasonal + 0.4 * is_day + rng.normal(0, 0.15, n)

# --- Outdoor temperature (degC): for adaptive models ---
outdoor = 5.0 + 25.0 * seasonal + 5.0 * diurnal + rng.normal(0, 1.5, n)

# --- Clothing insulation: seasonal ---
clo = 1.0 - 0.5 * seasonal + rng.normal(0, 0.03, n)
clo = np.clip(clo, 0.3, 1.5)

# --- Metabolic rate: constant sedentary ---
met = np.full(n, 1.2)

df = pd.DataFrame(
    {
        "timestamp": ts_10min,
        "air_temp_c": air_temp.round(2),
        "radiant_temp_c": radiant.round(2),
        "relative_humidity_pct": rh.round(1),
        "air_velocity_ms": vr.round(3),
        "illuminance_lux": lux.round(0),
        "noise_laeq_db": noise.round(1),
        "co2_ppm": co2.round(0),
        "pm25_ugm3": pm25.round(2),
        "pm10_ugm3": pm10.round(2),
        "tvoc_ugm3": tvoc.round(1),
        "formaldehyde_ppb": hcho.round(2),
        "co_ppm": co.round(3),
        "outdoor_temp_c": outdoor.round(2),
        "metabolic_rate_met": met,
        "clothing_insulation_clo": clo.round(3),
    }
)

out_path = Path(__file__).parent / "sample_sensor_data.csv"
df.to_csv(out_path, index=False)
print(f"Wrote {len(df):,} rows x {len(df.columns)} cols to {out_path}")
print(df.head())
