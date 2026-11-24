"""Long-term IEQ time-in-category — full comfio example.

EN 16798-1 Annex B style: for every domain, classify each timestamp into
categories I-IV and report the percentage (and occupied hours) of time
spent in each, plus cumulative I / I+II / I+II+III / I+II+III+IV shares.

Domains covered:
    Thermal   — PMV (ISO 7730), sPMV (Buratti), TSV occupant votes
    Adaptive  — EN 16798-1 |t_op - t_comf| offsets
    IAQ       — CO2 above outdoor (EN categories)
    Pollutants— PM2.5, PM10, TVOC (WHO/EPA/WELL tiers)
    Visual    — maintained illuminance (EN 12464-1 style limits)
    Acoustic  — LAeq (office limits)

Run:  py examples/temporal_ieq_example.py
"""

import numpy as np
import pandas as pd

import comfio

rng = np.random.default_rng(42)

# ===========================================================================
# 1. SYNTHETIC SENSOR DATA — one year of hourly readings
# ===========================================================================
n = 8760
ts = pd.date_range("2024-01-01", periods=n, freq="h")
doy = ts.dayofyear.to_numpy()
hour = ts.hour.to_numpy()
weekday = (ts.dayofweek.to_numpy() < 5)
work_hours = weekday & (hour >= 8) & (hour <= 18)

season = 24 + 6 * np.sin(2 * np.pi * (doy - 100) / 365)
tdb = np.clip(season + rng.normal(0, 1.2, n), 15, 32)
rh = np.clip(45 + 15 * np.sin(2 * np.pi * (doy - 200) / 365) + rng.normal(0, 5, n), 30, 70)

co2 = np.where(work_hours, 920 + rng.normal(0, 60, n), 425 + rng.normal(0, 15, n))
pm25 = np.clip(8 + 6 * np.sin(2 * np.pi * (doy - 60) / 365) + rng.normal(0, 4, n), 1, 60)
pm10 = np.clip(pm25 * 1.8 + rng.normal(0, 5, n), 2, 120)
tvoc = np.clip(120 + 80 * rng.random(n) + rng.normal(0, 30, n), 20, 900)
illuminance = np.where(work_hours, 480 + rng.normal(0, 60, n), 120 + rng.normal(0, 20, n))
laeq = np.where(work_hours, 42 + rng.normal(0, 4, n), 30 + rng.normal(0, 3, n))
tsv_votes = np.clip(rng.normal(0, 0.8, n), -3, 3)          # occupant votes
delta_t_op = np.abs(rng.normal(1.5, 1.0, n))               # |t_op - t_comf| in K

summaries: dict[str, comfio.TimeInCategoryResult] = {}


def report(name: str, summary, **kwargs) -> None:
    """Evaluate-temporal wrapper + collect for the final report."""
    summaries[name] = summary
    print(f"== {name} ==")
    print("  percentages:", {k: round(v, 1) for k, v in summary.percentages.items()})
    print("  cumulative: ", {k: round(v, 1) for k, v in summary.cumulative.items()})
    if summary.hours:
        print("  hours:      ", {k: round(v) for k, v in summary.hours.items()})
    print()


# ===========================================================================
# 2. THERMAL DOMAIN
# ===========================================================================

# 2a. PMV — full heat-balance model (ISO 7730)
pmv_res = comfio.evaluate_thermal(
    tdb=tdb, tr=tdb, vr=np.full(n, 0.1), rh=rh, met=1.2, clo=0.5
)
report("thermal_pmv", comfio.evaluate_temporal_thermal(pmv_res, timestamps=ts))

# 2b. sPMV — simplified model (Buratti), only needs air temp + RH
spmv_res = comfio.evaluate_spmv(indoor_temp=tdb, indoor_rh=rh)
report("thermal_spmv", comfio.evaluate_temporal_thermal(spmv_res, timestamps=ts))

# 2c. TSV — occupant votes (bands are PMV-defined; heuristic on votes)
tsv_res = comfio.evaluate_tsv(tsv_votes)
report("thermal_tsv", comfio.evaluate_temporal_tsv(tsv_res, timestamps=ts))

# 2d. Adaptive comfort — raw |t_op - t_comf| offsets in K
report("adaptive", comfio.evaluate_temporal_adaptive(delta_t_op, timestamps=ts))

# ===========================================================================
# 3. INDOOR AIR QUALITY
# ===========================================================================

# 3a. CO2 — EN categories are ppm ABOVE outdoor air
iq_res = comfio.evaluate_iaq(co2=co2)
report("iaq_co2", comfio.evaluate_temporal_iaq(iq_res, outdoor=420.0, timestamps=ts))

# 3b. Particulates / gases — WHO/EPA/WELL tiers mapped to I-IV
poll_res = comfio.evaluate_iaq_pollutants(pm25=pm25, pm10=pm10, tvoc=tvoc)
report("pm25", comfio.evaluate_temporal_pollutant(poll_res, pollutant="pm25", timestamps=ts))
report("pm10", comfio.evaluate_temporal_pollutant(poll_res, pollutant="pm10", timestamps=ts))
report("tvoc", comfio.evaluate_temporal_pollutant(poll_res, pollutant="tvoc", timestamps=ts))

# ===========================================================================
# 4. VISUAL & ACOUSTIC
# ===========================================================================

vis_res = comfio.evaluate_visual(illuminance=illuminance, task_type="office")
report("visual", comfio.evaluate_temporal_visual(vis_res, timestamps=ts))

ac_res = comfio.evaluate_acoustic(laeq=laeq, nc_level="NC-35")
report("acoustic", comfio.evaluate_temporal_acoustic(ac_res, timestamps=ts))

# ===========================================================================
# 5. REPORTS
# ===========================================================================

# LLM-ready markdown block
print(comfio.temporal_to_markdown(summaries))

# Daily breakdown example (thermal only)
daily = comfio.evaluate_temporal_thermal(pmv_res, timestamps=ts, by="D")
print("Daily thermal (first 5 rows):")
print(daily.head())

# Occupancy-mask example: only work hours count as occupied
report(
    "iaq_co2_workhours_only",
    comfio.evaluate_temporal_iaq(
        co2, outdoor=420.0, timestamps=ts, occupied=work_hours
    ),
)
