"""Build the comfio walkthrough notebook (examples/walkthrough_executed.ipynb).

Constructs the full cell list with nbformat, then the notebook is executed
separately via nbconvert under Python 3.11.
"""
from __future__ import annotations

import nbformat as nbf

nb = nbf.v4.new_notebook()
nb.metadata["kernelspec"] = {
    "display_name": "Python 3.11",
    "language": "python",
    "name": "python3",
}
nb.metadata["language_info"] = {"name": "python", "version": "3.11"}

cells = []


def md(src: str) -> None:
    cells.append(nbf.v4.new_markdown_cell(src))


def code(src: str) -> None:
    cells.append(nbf.v4.new_code_cell(src))


# ===========================================================================
# TITLE
# ===========================================================================
md("""# comfio — Complete Walkthrough

**A living document**: this walkthrough grows with the package. Every public
API is exercised here on a single synthetic dataset so the behaviour is
end-to-end reproducible.

| Section | Topic | Extras required |
|---------|-------|-----------------|
| 1 | Synthetic data generation (6 months, 10-min + 1-h) | — |
| 2 | Data ingestion (`SensorData`) | — |
| 3 | Core domains: thermal / visual / acoustic / IAQ | — |
| 4 | Advanced thermal: sPMV, adaptive, TSV, personalisation | — |
| 5 | Pollutant IAQ (PM2.5, TVOC, HCHO, CO) | — |
| 6 | Advanced domains: daylighting, colour, RT60, STI, ventilation, psychrometrics | `[daylighting]` `[color]` `[acoustics]` `[psychrometrics]` |
| 7 | Global IEQ Index & weight presets | — |
| 8 | Compliance & performance contracts (Solidity ABI) | — |
| 9 | ML integration: sklearn / PyTorch / Keras — next-day forecast | `[ml]` `[torch]` `[keras]` |
| 10 | LLM integration: interpreters, prompts, tool schemas | `[agent]` |
| 11 | Smart-contract export (web3.py) | `[agent]` |
| 12 | Reports: CSV / PDF / DOCX / intelligent pipeline | — |

> **Run this notebook yourself** with `pip install comfio[ml,torch,keras,agent,acoustics,color,psychrometrics]`
> plus `plotly tensorflow langchain web3 reportlab python-docx`.
> On Windows, `[daylighting]` (Radiance) needs WSL.""")

# ===========================================================================
# 0. IMPORTS
# ===========================================================================
code("""import sys, warnings, json, time, io, textwrap
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio

pio.templates.default = "plotly_white"
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import comfio
print(f"comfio {comfio.__version__}  |  Python {sys.version.split()[0]}  |  NumPy {np.__version__}")

from comfio import (
    SensorData,
    # core domains
    evaluate_thermal, evaluate_visual, evaluate_acoustic, evaluate_iaq,
    # advanced thermal
    evaluate_spmv, evaluate_adaptive_ashrae, evaluate_adaptive_en,
    augment_tsv_cdf, evaluate_tsv,
    train_personalisation, train_seasonal_personalisation,
    evaluate_personalised_pmv, evaluate_seasonal_personalised_pmv,
    # pollutant IAQ
    evaluate_iaq_pollutants,
    # integration
    calculate_global_ieq,
    # compliance / contracts
    calculate_compliance,
    # reports
    run_pipeline, detect_capabilities, ieq_to_csv,
)
from comfio.integration.weights import preset_weights, custom_weights, default_weights
from comfio.domains.thermal import thermal_score
print("Core imports OK.")""")

# ===========================================================================
# 1. DATA GENERATION
# ===========================================================================
md("""## 1. Synthetic Data Generation

We generate **6 months** (Jan–Jun 2025) of indoor environmental data at
**10-minute sampling** (~25 920 rows) plus **Thermal Sensation Votes** at
**1-hour sampling** (~4 320 rows). Each channel combines:

* a **seasonal** envelope (winter → summer warming),
* a **diurnal** cycle (day/night, occupancy),
* **Gaussian noise** calibrated to realistic sensor spread.

| Channel | Seasonal range | Diurnal amplitude | σ |
|---------|---------------|-------------------|---|
| `air_temp_c` | 20 → 27 °C | ±1.5 °C | 0.4 |
| `radiant_temp_c` | air_temp + 0–1 °C | ±1.0 °C | 0.3 |
| `relative_humidity_pct` | 55 → 40 % | ±5 % | 2 |
| `air_velocity_ms` | 0.10 m/s | — | 0.02 |
| `illuminance_lux` | 0–800 lux (daylight) | sinusoidal | 30 |
| `noise_laeq_db` | 38–48 dBA (occupancy) | ±4 dBA | 1.5 |
| `co2_ppm` | 600–1100 ppm | ±150 ppm | 40 |
| `pm25_ugm3` | 5–18 µg/m³ | ±3 | 1.0 |
| `tvoc_ugm3` | 100–300 µg/m³ | ±50 | 15 |
| `formaldehyde_ppb` | 15–35 ppb | ±5 | 2 |
| `co_ppm` | 0.8–2.5 ppm | ±0.4 | 0.15 |
| `outdoor_temp_c` | 5 → 30 °C | ±5 °C | 1.5 |
| `clothing_insulation_clo` | 1.0 → 0.5 | — | — |
| `tsv` (1-h) | derived from PMV + noise | — | 0.5 |""")

code("""rng = np.random.default_rng(42)

# 10-minute sampling for 6 months (Jan 1 - Jun 30, 2025)
start = datetime(2025, 1, 1, 0, 0, 0)
end   = datetime(2025, 6, 30, 23, 50, 0)
ts_10min = pd.date_range(start, end, freq="10min")
n = len(ts_10min)
print(f"10-min rows: {n:,}  ({ts_10min[0]} → {ts_10min[-1]})")

# Day-of-year fraction (0..1) for seasonal envelope
doy = ts_10min.dayofyear.to_numpy()
seasonal = (doy - 1) / 365.0                     # 0..1 across the year
hour = ts_10min.hour.to_numpy() + ts_10min.minute.to_numpy()/60.0
diurnal = np.sin(2*np.pi*(hour - 9.0)/24.0)      # peak ~15:00, trough ~03:00

# --- Indoor air temperature (°C): 20 winter -> 27 summer + diurnal ---
air_temp = 20.0 + 7.0*seasonal + 1.5*diurnal + rng.normal(0, 0.4, n)

# --- Radiant temperature: ~air_temp + small solar offset during day ---
is_day = ((hour > 7) & (hour < 18)).astype(float)
radiant = air_temp + 0.5*is_day*diurnal + rng.normal(0, 0.3, n)

# --- Relative humidity (%): higher in winter, anti-correlated with temp ---
rh = 55.0 - 15.0*seasonal - 5.0*diurnal + rng.normal(0, 2.0, n)
rh = np.clip(rh, 15, 85)

# --- Air velocity (m/s): mostly stagnant ---
vr = 0.10 + rng.normal(0, 0.02, n)
vr = np.clip(vr, 0.02, 0.5)

# --- Illuminance (lux): daylight sinusoid + seasonal ---
daylight = np.maximum(0, np.sin(np.pi*(hour - 6)/12))   # 0 at 06/18, peak 12
lux = daylight * (400 + 400*seasonal) + rng.normal(0, 30, n)
lux = np.clip(lux, 0, 1200)

# --- Noise L_Aeq (dBA): occupancy-driven 38 night -> 48 day ---
noise = 38.0 + 10.0*is_day + 4.0*diurnal*is_day + rng.normal(0, 1.5, n)

# --- CO2 (ppm): 420 baseline + occupancy 600 + diurnal ---
co2 = 420 + 600*is_day + 150*diurnal*is_day + rng.normal(0, 40, n)
co2 = np.clip(co2, 380, 1800)

# --- Pollutants ---
pm25  = 5 + 13*seasonal + 3*is_day + rng.normal(0, 1.0, n)
pm10  = pm25 + 4 + rng.normal(0, 1.5, n)
tvoc  = 100 + 200*seasonal + 50*is_day + rng.normal(0, 15, n)
hcho  = 15 + 20*seasonal + 5*is_day + rng.normal(0, 2.0, n)
co    = 0.8 + 1.7*seasonal + 0.4*is_day + rng.normal(0, 0.15, n)

# --- Outdoor temperature (°C): for adaptive models ---
outdoor = 5.0 + 25.0*seasonal + 5.0*diurnal + rng.normal(0, 1.5, n)

# --- Clothing insulation: seasonal (1.0 winter -> 0.5 summer) ---
clo = 1.0 - 0.5*seasonal + rng.normal(0, 0.03, n)
clo = np.clip(clo, 0.3, 1.5)

# --- Metabolic rate: constant sedentary ---
met = np.full(n, 1.2)

df = pd.DataFrame({
    "timestamp": ts_10min,
    "air_temp_c": air_temp,
    "radiant_temp_c": radiant,
    "relative_humidity_pct": rh,
    "air_velocity_ms": vr,
    "illuminance_lux": lux,
    "noise_laeq_db": noise,
    "co2_ppm": co2,
    "pm25_ugm3": pm25,
    "pm10_ugm3": pm10,
    "tvoc_ugm3": tvoc,
    "formaldehyde_ppb": hcho,
    "co_ppm": co,
    "outdoor_temp_c": outdoor,
    "metabolic_rate_met": met,
    "clothing_insulation_clo": clo,
})
print(f"DataFrame shape: {df.shape}")
df.head()""")

code("""# --- TSV at 1-hour sampling ---
ts_1h = pd.date_range(start, end, freq="1h")
n1 = len(ts_1h)
h1 = ts_1h.hour.to_numpy() + 0.0
is_day1 = ((h1 > 7) & (h1 < 18)).astype(float)
doy1 = ts_1h.dayofyear.to_numpy()
seasonal1 = (doy1 - 1)/365.0
diurnal1 = np.sin(2*np.pi*(h1 - 9)/24)

# Indoor temp at 1-h resolution
temp_1h = 20 + 7*seasonal1 + 1.5*diurnal1
rh_1h   = 55 - 15*seasonal1 - 5*diurnal1
# Approximate PMV from temp (warmer -> positive PMV)
approx_pmv = (temp_1h - 23.5) * 0.35
tsv_raw = approx_pmv + rng.normal(0, 0.5, n1)
tsv_votes = np.clip(np.round(tsv_raw), -3, 3)

df_tsv = pd.DataFrame({
    "timestamp": ts_1h,
    "tsv": tsv_votes,
    "air_temp_c": temp_1h + rng.normal(0, 0.4, n1),
    "relative_humidity_pct": rh_1h + rng.normal(0, 2, n1),
})
print(f"TSV rows: {len(df_tsv):,}  |  TSV range: [{tsv_votes.min():.0f}, {tsv_votes.max():.0f}]")
print(f"TSV distribution:\\n{pd.Series(tsv_votes).value_counts().sort_index()}")""")

code("""# --- Plotly: overview of the 6-month dataset ---
fig = make_subplots(rows=4, cols=1, shared_xaxes=True,
                    subplot_titles=("Air temperature & RH", "Illuminance & CO₂",
                                    "PM2.5 & TVOC", "Noise & Outdoor temp"))
fig.add_trace(go.Scatter(x=df.timestamp, y=df.air_temp_c, name="T_air", line=dict(width=0.5)), row=1, col=1)
fig.add_trace(go.Scatter(x=df.timestamp, y=df.relative_humidity_pct, name="RH", yaxis="y2", line=dict(width=0.5)), row=1, col=1)
fig.add_trace(go.Scatter(x=df.timestamp, y=df.illuminance_lux, name="Lux", line=dict(width=0.5)), row=2, col=1)
fig.add_trace(go.Scatter(x=df.timestamp, y=df.co2_ppm, name="CO₂", line=dict(width=0.5)), row=2, col=1)
fig.add_trace(go.Scatter(x=df.timestamp, y=df.pm25_ugm3, name="PM2.5", line=dict(width=0.5)), row=3, col=1)
fig.add_trace(go.Scatter(x=df.timestamp, y=df.tvoc_ugm3, name="TVOC", line=dict(width=0.5)), row=3, col=1)
fig.add_trace(go.Scatter(x=df.timestamp, y=df.noise_laeq_db, name="L_Aeq", line=dict(width=0.5)), row=4, col=1)
fig.add_trace(go.Scatter(x=df.timestamp, y=df.outdoor_temp_c, name="T_out", line=dict(width=0.5)), row=4, col=1)
fig.update_layout(height=900, title_text="6-Month Synthetic IEQ Dataset (10-min sampling)", showlegend=True)
fig.show()""")

code("""# --- TSV time series ---
fig = go.Figure()
fig.add_trace(go.Scatter(x=df_tsv.timestamp, y=df_tsv.tsv, mode="markers",
                         marker=dict(size=3, opacity=0.4), name="TSV"))
fig.update_layout(title="Thermal Sensation Votes (1-h sampling, 6 months)",
                  xaxis_title="Date", yaxis_title="TSV (-3 cold .. +3 hot)",
                  height=350)
fig.show()""")

# ===========================================================================
# 2. DATA INGESTION
# ===========================================================================
md("""## 2. Data Ingestion with `SensorData`

`SensorData` wraps the DataFrame, auto-maps common column aliases to canonical
names, validates physical bounds, and reports which IEQ domains can be
evaluated from the available columns.""")

code("""sensor = SensorData(df=df, timestamp_col="timestamp")
sensor.validate()
print(sensor)
print(f"\\nMapped columns ({len(sensor.column_map)}):")
for canon, actual in sensor.column_map.items():
    print(f"  {canon:30s} -> {actual}")
print(f"\\nAvailable domains:        {sensor.available_domains()}")
print(f"Available advanced domains: {sensor.available_advanced_domains()}")""")

code("""# Capabilities detected by the intelligent pipeline
caps = detect_capabilities(sensor)
print("Detected capabilities:")
for k, v in caps.items():
    print(f"  {k:30s} {v}")""")

# ===========================================================================
# 3. CORE DOMAINS
# ===========================================================================
md("""## 3. Core Domain Evaluations

### 3a. Thermal Comfort — PMV / PPD (ISO 7730)

Fanger's Predicted Mean Vote (PMV) is the most widely used thermal-comfort
index. It combines six variables: air temperature, radiant temperature, air
velocity, relative humidity, metabolic rate, and clothing insulation.

$$\\text{PMV} = f(t_a, \\bar{t}_r, v, RH, M, I_{cl}) \\in [-3, +3]$$

The Predicted Percentage Dissatisfied (PPD) is a non-linear function of PMV:

$$\\text{PPD} = 100 - 95\\,\\exp\\!\\left(-0.03353\\,\\text{PMV}^4 - 0.2179\\,\\text{PMV}^2\\right)$$

ISO 7730 Category B targets |PMV| ≤ 0.5 (PPD ≤ 10 %).""")

code("""thermal = evaluate_thermal(
    tdb=sensor.get_validated("air_temp_c"),
    tr=sensor.get_validated("radiant_temp_c"),
    vr=sensor.get_validated("air_velocity_ms"),
    rh=sensor.get_validated("relative_humidity_pct"),
    met=sensor.get_validated("metabolic_rate_met"),
    clo=sensor.get_validated("clothing_insulation_clo"),
    standard="7730-2005", category="B",
)
print(f"PMV  mean={np.mean(thermal.pmv):+.2f}  std={np.std(thermal.pmv):.2f}  range=[{thermal.pmv.min():+.2f}, {thermal.pmv.max():+.2f}]")
print(f"PPD  mean={np.mean(thermal.ppd):.1f}%  max={thermal.ppd.max():.1f}%")
print(f"Thermal score  mean={np.mean(thermal_score(thermal.pmv, thermal.ppd)):.1f}/100")
print(f"Category-B compliant (|PMV|<=0.5): {np.mean(np.abs(thermal.pmv)<=0.5)*100:.1f}%")""")

code("""fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                    subplot_titles=("PMV (ISO 7730)", "PPD % + thermal score"))
fig.add_trace(go.Scatter(x=df.timestamp, y=thermal.pmv, name="PMV", line=dict(width=0.5)), row=1, col=1)
fig.add_hline(y=0.5, line_dash="dash", line_color="red", row=1, col=1)
fig.add_hline(y=-0.5, line_dash="dash", line_color="blue", row=1, col=1)
fig.add_trace(go.Scatter(x=df.timestamp, y=thermal.ppd, name="PPD %", line=dict(width=0.5)), row=2, col=1)
fig.add_trace(go.Scatter(x=df.timestamp, y=thermal_score(thermal.pmv, thermal.ppd), name="score", line=dict(width=0.5, color="green")), row=2, col=1)
fig.update_layout(height=600, title="Thermal comfort over 6 months")
fig.show()""")

md("""### 3b. Visual Comfort — EN 12464-1:2021

Evaluates maintained illuminance against task-specific targets. The default
`"general"` task requires **500 lux**. UGR (Unified Glare Rating) limits are
also defined per task type.""")

code("""visual = evaluate_visual(
    illuminance=sensor.get_validated("illuminance_lux"),
    task_type="general",
)
print(f"Target illuminance: {visual.target_lux:.0f} lux")
print(f"Visual score  mean={np.mean(visual.score):.1f}/100")
print(f"Compliant (>= target): {np.mean(visual.compliant)*100:.1f}%")
print(f"Mean illuminance: {np.mean(visual.illuminance):.0f} lux (daytime only meaningful)")""")

code("""fig = go.Figure()
fig.add_trace(go.Scatter(x=df.timestamp, y=df.illuminance_lux, name="Illuminance", line=dict(width=0.5)))
fig.add_hline(y=visual.target_lux, line_dash="dash", line_color="red", annotation_text=f"target {visual.target_lux:.0f} lux")
fig.add_trace(go.Scatter(x=df.timestamp, y=visual.score, name="Visual score", yaxis="y2", line=dict(width=0.5, color="green")))
fig.update_layout(title="Visual comfort (EN 12464-1)", yaxis=dict(title="lux"), yaxis2=dict(title="score", overlaying="y", side="right"), height=400)
fig.show()""")

md("""### 3c. Acoustic Comfort — Noise Criteria (NC)

A-weighted equivalent sound levels ($L_{Aeq}$) are compared against NC curves
(ASHRAE Handbook — HVAC Applications, Ch. 49). `NC-35` (41 dBA) is the default
for general offices.""")

code("""acoustic = evaluate_acoustic(
    laeq=sensor.get_validated("noise_laeq_db"),
    nc_level="NC-35",
)
print(f"NC threshold: {acoustic.threshold_db:.0f} dBA")
print(f"Acoustic score  mean={np.mean(acoustic.score):.1f}/100")
print(f"Compliant (<= threshold): {np.mean(acoustic.compliant)*100:.1f}%")""")

code("""fig = go.Figure()
fig.add_trace(go.Scatter(x=df.timestamp, y=df.noise_laeq_db, name="L_Aeq", line=dict(width=0.5)))
fig.add_hline(y=acoustic.threshold_db, line_dash="dash", line_color="red", annotation_text=f"NC-35 = {acoustic.threshold_db:.0f} dBA")
fig.add_trace(go.Scatter(x=df.timestamp, y=acoustic.score, name="score", yaxis="y2", line=dict(width=0.5, color="green")))
fig.update_layout(title="Acoustic comfort (NC-35)", yaxis=dict(title="dBA"), yaxis2=dict(title="score", overlaying="y", side="right"), height=400)
fig.show()""")

md("""### 3d. Indoor Air Quality — CO₂ (ASHRAE 62.1 indicators)

ASHRAE 62.1 uses ventilation-rate procedures, not CO₂ limits. However, CO₂ is
the standard ventilation-adequacy indicator. Practical thresholds:

| Level | CO₂ (ppm) | Interpretation |
|-------|-----------|----------------|
| excellent | ≤ 800 | ~10 L/s per person |
| good | ≤ 1000 | commonly cited benchmark |
| moderate | ≤ 1200 | marginal |
| poor | ≤ 1500 | inadequate |""")

code("""iaq = evaluate_iaq(
    co2=sensor.get_validated("co2_ppm"),
    threshold_level="good",
)
print(f"CO₂ threshold: {iaq.threshold_ppm:.0f} ppm  (level='{iaq.threshold_level}')")
print(f"IAQ score  mean={np.mean(iaq.score):.1f}/100")
print(f"Compliant (<= threshold): {np.mean(iaq.compliant)*100:.1f}%")
print(f"CO₂  mean={np.mean(iaq.co2):.0f}  peak={iaq.co2.max():.0f} ppm")""")

code("""fig = go.Figure()
fig.add_trace(go.Scatter(x=df.timestamp, y=df.co2_ppm, name="CO₂", line=dict(width=0.5)))
fig.add_hline(y=iaq.threshold_ppm, line_dash="dash", line_color="red", annotation_text=f"good = {iaq.threshold_ppm:.0f} ppm")
fig.add_trace(go.Scatter(x=df.timestamp, y=iaq.score, name="score", yaxis="y2", line=dict(width=0.5, color="green")))
fig.update_layout(title="IAQ — CO₂ ventilation indicator", yaxis=dict(title="ppm"), yaxis2=dict(title="score", overlaying="y", side="right"), height=400)
fig.show()""")

# ===========================================================================
# 4. ADVANCED THERMAL
# ===========================================================================
md("""## 4. Advanced Thermal Models

### 4a. Simplified PMV — Buratti et al. (2009)

When only indoor temperature and humidity are available, the Buratti seasonal
model gives a reduced-form PMV:

$$\\text{sPMV} = a \\cdot t_{in} + b \\cdot RH + c$$

with season-specific coefficients $(a, b, c)$. Internal vapor pressure uses the
Magnus formula. This is ideal for BMS deployments with limited sensor coverage.

**Reference**: Buratti, C., Ricciardi, P., & Vergoni, M. (2009). *Simplified
PMV model for HVAC systems control.* Building and Environment, 44(3), 441–449.""")

code("""spmv = evaluate_spmv(
    indoor_temp=sensor.get_validated("air_temp_c"),
    indoor_rh=sensor.get_validated("relative_humidity_pct"),
    date_ref=start,
)
print(f"Season determined: {spmv.season}")
print(f"sPMV  mean={np.mean(spmv.spmv):+.2f}  range=[{spmv.spmv.min():+.2f}, {spmv.spmv.max():+.2f}]")
print(f"sPMV score  mean={np.mean(spmv.score):.1f}/100")

# Compare sPMV vs full PMV
corr = np.corrcoef(spmv.spmv, thermal.pmv)[0,1]
print(f"\\nCorrelation sPMV vs full PMV: {corr:.3f}")""")

code("""fig = go.Figure()
fig.add_trace(go.Scatter(x=df.timestamp, y=thermal.pmv, name="Full PMV", line=dict(width=0.5)))
fig.add_trace(go.Scatter(x=df.timestamp, y=spmv.spmv, name="sPMV (Buratti)", line=dict(width=0.5, color="orange")))
fig.update_layout(title="Full PMV vs simplified PMV (Buratti)", yaxis_title="PMV", height=400)
fig.show()""")

md("""### 4b. Adaptive Thermal Comfort

For naturally ventilated buildings, adaptive models relate the **comfort
temperature** to the running mean outdoor temperature. Two standards are
implemented:

* **ASHRAE 55-2023**: $t_{comf} = 0.31 \\, t_{out} + 17.8$ (80 % acceptability band ±3.5 °C)
* **EN 16798-1:2019**: $t_{comf} = 0.33 \\, t_{rm} + 18.8$ (Category II band ±3 °C)

where $t_{rm}$ is the exponentially-weighted running mean of outdoor
temperature.""")

code("""# Compute a simple running mean of outdoor temperature (alpha=0.8)
outdoor_arr = sensor.get_validated("outdoor_temp_c")
rm = np.zeros(n)
rm[0] = outdoor_arr[0]
for i in range(1, n):
    rm[i] = 0.8*rm[i-1] + 0.2*outdoor_arr[i]

# Use daily prevailing mean for ASHRAE (simpler: rolling daily mean)
daily_out = pd.Series(outdoor_arr, index=df.timestamp).resample("1D").mean()
prevailing = daily_out.rolling(7, min_periods=1).mean().reindex(df.timestamp, method="ffill").values

ashrae = evaluate_adaptive_ashrae(
    tdb=sensor.get_validated("air_temp_c"),
    tr=sensor.get_validated("radiant_temp_c"),
    t_prevail=float(np.mean(prevailing)),
    acceptability=80,
)
en = evaluate_adaptive_en(
    tdb=sensor.get_validated("air_temp_c"),
    tr=sensor.get_validated("radiant_temp_c"),
    t_running_mean=float(np.mean(rm)),
    category="ii",
)
print(f"ASHRAE 55  t_comf mean={np.mean(ashrae.t_comf):.1f}°C  compliance={np.mean(ashrae.compliant)*100:.1f}%")
print(f"EN 16798   t_comf mean={np.mean(en.t_comf):.1f}°C  compliance={np.mean(en.compliant)*100:.1f}%")""")

code("""fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                    subplot_titles=("ASHRAE 55 — operative vs comfort temp", "EN 16798 — operative vs comfort temp"))
fig.add_trace(go.Scatter(x=df.timestamp, y=ashrae.t_op, name="t_op", line=dict(width=0.5)), row=1, col=1)
fig.add_hline(y=ashrae.t_comf, line_dash="solid", line_color="red", line_width=1, row=1, col=1, annotation_text=f"t_comf={ashrae.t_comf:.1f}")
fig.add_hline(y=ashrae.t_comf_upper, line_dash="dot", line_color="red", line_width=0.8, row=1, col=1)
fig.add_hline(y=ashrae.t_comf_lower, line_dash="dot", line_color="red", line_width=0.8, row=1, col=1)
fig.add_trace(go.Scatter(x=df.timestamp, y=en.t_op, name="t_op EN", line=dict(width=0.5)), row=2, col=1)
fig.add_hline(y=en.t_comf, line_dash="solid", line_color="purple", line_width=1, row=2, col=1, annotation_text=f"t_comf={en.t_comf:.1f}")
fig.add_hline(y=en.t_comf_upper, line_dash="dot", line_color="purple", line_width=0.8, row=2, col=1)
fig.add_hline(y=en.t_comf_lower, line_dash="dot", line_color="purple", line_width=0.8, row=2, col=1)
fig.update_layout(height=700, title="Adaptive comfort bands (80% / Category II)")
fig.show()""")

md("""### 4c. TSV Augmentation & Evaluation

Occupant Thermal Sensation Votes are sparse (1-h). `augment_tsv_cdf` remaps
them to the 10-min sensor grid via CDF matching. `evaluate_tsv` checks
compliance per ASHRAE 55-2023 Appendix L (|TSV| ≤ 1.5 for 80 % acceptability).""")

code("""tsv_aug = augment_tsv_cdf(
    sparse_votes=df_tsv.tsv.values.astype(float),
    vote_timestamps=df_tsv.timestamp.values.astype("datetime64[ns]").astype(float),
    target_timestamps=df.timestamp.values.astype("datetime64[ns]").astype(float),
)
tsv_result = evaluate_tsv(tsv_aug)
print(f"Augmented TSV length: {len(tsv_aug)}  (matches sensor grid)")
print(f"TSV mean={np.mean(tsv_aug):+.2f}  std={np.std(tsv_aug):.2f}")
print(f"Compliance rate (|TSV|<=1.5): {tsv_result.compliance_rate*100:.1f}%")
print(f"TSV score mean={np.mean(tsv_result.score):.1f}/100")""")

code("""fig = go.Figure()
fig.add_trace(go.Scatter(x=df.timestamp, y=tsv_aug, name="Augmented TSV", line=dict(width=0.5, color="teal")))
fig.add_hline(y=1.5, line_dash="dash", line_color="red")
fig.add_hline(y=-1.5, line_dash="dash", line_color="blue")
fig.update_layout(title="Augmented TSV on 10-min grid (ASHRAE 55 Appendix L)", yaxis_title="TSV", height=400)
fig.show()""")

md("""### 4d. Personalised Comfort (OLS Regression)

When both PMV (model) and TSV (occupant) data are available, comfio fits a
linear personalisation index:

$$\\text{TSV} \\approx \\alpha \\cdot \\text{PMV}_{model} + \\beta$$

The personalised PMV is then $\\alpha \\cdot \\text{PMV} + \\beta$, and PPD is
recomputed. Seasonal indices allow different personalisation per season.""")

code("""# Align TSV (1-h) to PMV (10-min) by nearest timestamp
pmv_1h = pd.Series(thermal.pmv, index=df.timestamp).reindex(df_tsv.timestamp, method="nearest").values
tsv_1h = df_tsv.tsv.values.astype(float)

idx = train_personalisation(pmv=pmv_1h, tsv=tsv_1h)
print(f"Personalisation: alpha={idx.alpha:.3f}  beta={idx.beta:.3f}  R²={idx.r_squared:.3f}  n={idx.n_samples}")

# Seasonal personalisation
dates_1h = df_tsv.timestamp.dt.date.tolist()
seasonal_idx = train_seasonal_personalisation(pmv=pmv_1h, tsv=tsv_1h, dates=dates_1h)
print(f"\\nSeasonal indices:")
for s, si in seasonal_idx.indices.items():
    print(f"  {s:8s} alpha={si.alpha:.3f} beta={si.beta:.3f} R²={si.r_squared:.3f} n={si.n_samples}")""")

code("""# Apply personalisation to the full 10-min grid
personalised = evaluate_personalised_pmv(
    tdb=sensor.get_validated("air_temp_c"),
    tr=sensor.get_validated("radiant_temp_c"),
    vr=sensor.get_validated("air_velocity_ms"),
    rh=sensor.get_validated("relative_humidity_pct"),
    met=1.2, clo=0.5,
    personalisation_index=idx,
)
print(f"Base PMV mean={np.mean(personalised.base_pmv):+.2f}  ->  Personalised PMV mean={np.mean(personalised.personalised_pmv):+.2f}")
print(f"Base PPD mean={np.mean(personalised.base_ppd):.1f}%  ->  Personalised PPD mean={np.mean(personalised.personalised_ppd):.1f}%")
print(f"Personalised score mean={np.mean(personalised.score):.1f}/100")""")

code("""fig = go.Figure()
fig.add_trace(go.Scatter(x=df.timestamp, y=personalised.base_pmv, name="Base PMV", line=dict(width=0.5)))
fig.add_trace(go.Scatter(x=df.timestamp, y=personalised.personalised_pmv, name="Personalised PMV", line=dict(width=0.5, color="orange")))
fig.update_layout(title="Base vs personalised PMV", yaxis_title="PMV", height=400)
fig.show()""")

# ===========================================================================
# 5. POLLUTANT IAQ
# ===========================================================================
md("""## 5. Pollutant IAQ

PM2.5, PM10, TVOC, formaldehyde, and CO are evaluated against WHO, EPA, and
WELL Building Standard thresholds. The pollutant IAQ score blends 50/50 with
the CO₂-based IAQ score in the Global IEQ Index.""")

code("""pollutant = evaluate_iaq_pollutants(
    pm25=sensor.get_validated("pm25_ugm3"),
    pm10=sensor.get_validated("pm10_ugm3"),
    tvoc=sensor.get_validated("tvoc_ugm3"),
    formaldehyde=sensor.get_validated("formaldehyde_ppb"),
    co=sensor.get_validated("co_ppm"),
    threshold_level="good",
)
print(f"Pollutant IAQ score  mean={np.mean(pollutant.score):.1f}/100")
print(f"PM2.5 compliant:  {np.mean(pollutant.compliant_pm25)*100:.1f}%   (WHO 24h guideline: 15 µg/m³)")
print(f"PM10 compliant:   {np.mean(pollutant.compliant_pm10)*100:.1f}%")
print(f"TVOC compliant:   {np.mean(pollutant.compliant_tvoc)*100:.1f}%")
print(f"HCHO compliant:   {np.mean(pollutant.compliant_formaldehyde)*100:.1f}%")
print(f"CO compliant:     {np.mean(pollutant.compliant_co)*100:.1f}%")""")

code("""fig = make_subplots(rows=3, cols=2, shared_xaxes=True,
                    subplot_titles=("PM2.5 µg/m³", "PM10 µg/m³", "TVOC µg/m³", "HCHO ppb", "CO ppm", "Pollutant score"))
fig.add_trace(go.Scatter(x=df.timestamp, y=df.pm25_ugm3, name="PM2.5", line=dict(width=0.5)), row=1, col=1)
fig.add_trace(go.Scatter(x=df.timestamp, y=df.pm10_ugm3, name="PM10", line=dict(width=0.5)), row=1, col=2)
fig.add_trace(go.Scatter(x=df.timestamp, y=df.tvoc_ugm3, name="TVOC", line=dict(width=0.5)), row=2, col=1)
fig.add_trace(go.Scatter(x=df.timestamp, y=df.formaldehyde_ppb, name="HCHO", line=dict(width=0.5)), row=2, col=2)
fig.add_trace(go.Scatter(x=df.timestamp, y=df.co_ppm, name="CO", line=dict(width=0.5)), row=3, col=1)
fig.add_trace(go.Scatter(x=df.timestamp, y=pollutant.score, name="score", line=dict(width=0.5, color="green")), row=3, col=2)
fig.update_layout(height=700, title="Pollutant IAQ (WHO / EPA / WELL thresholds)", showlegend=False)
fig.show()""")

# ===========================================================================
# 6. ADVANCED DOMAINS
# ===========================================================================
md("""## 6. Advanced Domains (optional extras)

### 6a. Daylighting (Radiance) — `pip install comfio[daylighting]`

> **Note**: `pyradiance` requires Radiance binaries which on Windows need WSL.
> The code is shown for completeness; this cell is skipped if the extra is
> not installed.""")

code("""try:
    from comfio import evaluate_daylighting
    # evaluate_daylighting needs an octree file + sensor points — omitted here
    # because it requires a Radiance scene. See the daylighting tutorial.
    print("evaluate_daylighting is available. Provide an .oct scene to run.")
except ImportError as e:
    print(f"Daylighting extra not installed: {e}")""")

md("""### 6b. Colour Quality (CRI / CCT) — `pip install comfio[color]`

Evaluates a light source's Colour Rendering Index (CIE 1995 / 2024),
Correlated Colour Temperature, and $D_{uv}$ from its spectral power
distribution.

> **Theory — CRI (Ra).** The Colour Rendering Index, defined by CIE 13.3
> (1995), quantifies how faithfully a light source renders object colours
> compared to a reference illuminant (a Planckian radiator for CCT < 5000 K,
> or CIE D-series daylight for CCT >= 5000 K). The test source and reference
> are each used to illuminate 8 (or 14) standard test colour samples
> (TCS); the special CRI $R_i$ for each sample is computed from the
> colour-difference $\\Delta E_i$ in the 1964 U*V*W* colour space:
>
> $$R_i = 100 - 4.6 \\, \\Delta E_i$$
>
> The **general CRI** $R_a$ is the arithmetic mean of $R_1 \\dots R_8$.
> $R_a = 100$ means perfect colour rendering; values >= 80 are considered
> acceptable for indoor work (EN 12464-1:2021).
>
> **Correlated Colour Temperature (CCT)** is the temperature of the Planckian
> radiator whose perceived colour most closely matches the source. **D_uv**
> is the distance from the Planckian locus in the CIE 1960 UCS diagram;
> $|D_{uv}| < 0.006$ is considered "white".
>
> **References:**
> - CIE 13.3-1995, *Method of measuring and specifying colour rendering
>   properties of light sources*.
> - CIE 015:2018, *Colorimetry*, 4th ed. (CCT and D_uv formulas).
> - EN 12464-1:2021, §4.2 — minimum CRI requirements per task type.
> - Royer, M.P. et al. (2017), "Color rendering of light sources: a
>   review", *LEUKOS* 13(4), 187–209. — peer-reviewed critique of CRI and
>   the case for TM-30.""")

code("""from comfio import evaluate_color_quality
# Synthetic SPD: warm white LED (~3000 K) with a smooth blackbody-like curve
wavelengths = np.linspace(380, 780, 81)
# Simple Planckian-ish SPD peaking in the red-yellow
spd = np.exp(-((wavelengths - 590)**2) / (2 * 80**2)) * 0.8 + 0.2
try:
    color = evaluate_color_quality(
        spectral_distribution=spd,
        wavelengths=wavelengths,
        method="CIE 1995",
        min_cri=80,
    )
    print(f"CRI (Ra):  {color.cri:.1f}")
    print(f"CCT:       {color.cct:.0f} K")
    print(f"D_uv:      {color.duv:.4f}")
    print(f"Compliant: {color.compliant}")
    print(f"Score:     {color.score:.1f}/100")
except Exception as e:
    print(f"Colour quality evaluation failed (colour-science API compatibility): {e}")
    print("This is a known issue with newer colour-science versions.")""")

md("""### 6c. Reverberation Time (RT60) — `pip install comfio[acoustics]`

Sabine / Eyring RT60 calculation from surface areas and absorption
coefficients, compared against room-type targets.

> **Theory — Reverberation and the Sabine formula.**
>
> Reverberation time (RT60) is the time required for the steady-state sound
> pressure level in a room to decay by 60 dB after the source stops. Wallace
> Sabine (1898) derived the first and most widely used formula:
>
> $$T_{60} = 0.161 \\, \\frac{V}{\\sum_i S_i \\, \\alpha_i}$$
>
> where $V$ is the room volume (m³), $S_i$ is the area of surface $i$ (m²),
> and $\\alpha_i$ is the sound absorption coefficient of surface $i$
> (dimensionless, 0 = perfectly reflective, 1 = perfectly absorptive).
> The denominator $A = \\sum S_i \\alpha_i$ is the total room absorption
> (m² Sabins).
>
> The **Eyring** (1930) formula refines Sabine for more absorptive rooms:
>
> $$T_{60} = -0.161 \\, \\frac{V}{S \\, \\ln(1 - \\bar{\\alpha})}$$
>
> where $\\bar{\\alpha} = A / S$ is the mean absorption coefficient and $S$
> is the total surface area. Eyring is preferred when $\\bar{\\alpha} > 0.3$.
>
> **Noise Reduction Coefficient (NRC)** is the arithmetic mean of
> absorption coefficients at 250, 500, 1000, and 2000 Hz, rounded to the
> nearest 0.05. It is a single-number rating used in architectural acoustics.
>
> **Target RT60 values** (mid-frequency, 500–1000 Hz) per room type:
>
> | Room type | Target RT60 (s) | Standard |
> |-----------|-----------------|----------|
> | Office | 0.6–0.8 | ISO 3382-2 |
> | Meeting room | 0.6–1.0 | ISO 3382-2 |
> | Classroom | 0.6–0.8 | ANSI S12.60 |
> | Concert hall | 1.8–2.2 | ISO 3382-1 |
>
> **References:**
> - Sabine, W.C. (1922), *Collected Papers on Acoustics*, Harvard University
>   Press. — the foundational derivation.
> - Eyring, C.F. (1930), "Reverberation time in 'dead' rooms",
>   *Journal of the Acoustical Society of America* 1(2A), 217–241.
> - ISO 3382-2:2008, *Acoustics — Measurement of room acoustic parameters
>   — Part 2: Reverberation time in ordinary rooms*.
> - ANSI/ASA S12.60-2019, *Acoustical Performance Criteria, Design
>   Requirements, and Guidelines for Schools*.
> - Beranek, L.L. (1954), *Acoustics*, McGraw-Hill, Ch. 13 — textbook
>   treatment of Sabine and Eyring.""")

code("""from comfio import evaluate_reverberation
# A 50 m³ meeting room with 6 octave-band absorption coefficients
surfaces = np.array([20, 20, 25, 25, 10, 10], dtype=float)  # m² per surface
abs_coeffs = np.array([
    [0.10, 0.15, 0.20, 0.30, 0.40, 0.45],  # walls
    [0.10, 0.15, 0.20, 0.30, 0.40, 0.45],  # walls
    [0.05, 0.05, 0.10, 0.10, 0.15, 0.20],  # floor
    [0.30, 0.40, 0.50, 0.60, 0.70, 0.80],  # ceiling (absorptive)
    [0.05, 0.05, 0.05, 0.05, 0.05, 0.05],  # window
    [0.10, 0.10, 0.15, 0.20, 0.25, 0.30],  # door
])
try:
    reverb = evaluate_reverberation(
        surfaces=surfaces, absorption_coeffs=abs_coeffs, volume=50.0,
        method="sabine", room_type="meeting_room",
    )
    print(f"RT60 per band: {np.round(reverb.rt60, 3)}")
    print(f"Mean RT60:     {np.mean(reverb.rt60):.2f} s")
    print(f"NRC:           {reverb.nrc:.2f}" if reverb.nrc is not None else "NRC: n/a")
    print(f"Compliant:     {reverb.compliant}  (target {reverb.room_type})")
    print(f"Score:         {reverb.score:.1f}/100")
except Exception as e:
    print(f"Reverberation evaluation failed: {e}")
    print("Note: python-acoustics may have scipy compatibility issues (sph_harm removed in scipy >= 1.15).")""")

md("""### 6d. Speech Intelligibility (STI) — `pip install comfio[acoustics]`

Speech Transmission Index from a room impulse response, rated per
IEC 60268-16.

> **Theory — Speech Transmission Index (STI).**
>
> The STI, defined by Houtgast & Steeneken (1971) and standardised in
> IEC 60268-16, quantifies the intelligibility of speech transmitted through
> a room or communication system. It is based on the **Modulation Transfer
> Function (MTF)**, which describes how much the intensity envelope of a
> speech signal is preserved after passing through the acoustic channel:
>
> $$m(f) = \\frac{\\int_0^\\infty I(t) \\, I(t + 1/f) \\, dt}{\\int_0^\\infty I^2(t) \\, dt}$$
>
> where $m(f)$ is the modulation reduction factor at modulation frequency $f$.
> The MTF is measured at 14 modulation frequencies (0.63–12.5 Hz) across
> 7 octave bands (125 Hz–8 kHz). Each octave band contributes a weighted
> **apparent signal-to-noise ratio**:
>
> $$\\bar{X}_k = -10 \\log_{10}\\left(\\frac{1}{14}\\sum_f \\frac{1}{m_k(f)} - 1\\right)$$
>
> The STI is the weighted average of the transmission indices $T_k$ derived
> from $\\bar{X}_k$ across the 7 octave bands, with weights reflecting the
> contribution of each band to speech intelligibility.
>
> **STI rating scale** (IEC 60268-16:2020):
>
> | STI range | Rating | Typical application |
> |-----------|--------|---------------------|
> | < 0.30 | Bad | — |
> | 0.30–0.45 | Poor | Public address, low-quality |
> | 0.45–0.60 | Fair | Classrooms, meeting rooms |
> | 0.60–0.75 | Good | Theatres, lecture halls |
> | > 0.75 | Excellent | Broadcast studios |
>
> A value of STI >= 0.60 is generally required for "good" speech
> intelligibility in workplaces (EN ISO 9921:2003).
>
> **References:**
> - Houtgast, T. & Steeneken, H.J.M. (1971), "Evaluation of speech
>   transmission channels by using artificial signals", *Acustica* 25,
>   355–367. — original MTF/STI concept.
> - IEC 60268-16:2020, *Sound system equipment — Part 16: Objective rating
>   of speech intelligibility by speech transmission index*.
> - EN ISO 9921:2003, *Ergonomics — Assessment of speech communication*.
> - Steeneken, H.J.M. & Houtgast, T. (1980), "A physical method for
>   measuring speech-transmission quality", *Journal of the Acoustical
>   Society of America* 67(1), 318–326. — peer-reviewed validation.
> - Rindel, J.H. (2018), "Computer simulation techniques for acoustical
>   design of rooms", *Acoustics Australia* 46, 67–75. — modern review.""")

code("""from comfio import evaluate_speech_intelligibility
# Synthetic impulse response: decaying exponential + noise (1 s at 16 kHz)
sr = 16000
ir = np.exp(-np.linspace(0, 4, sr)) + rng.normal(0, 0.01, sr)
try:
    sti = evaluate_speech_intelligibility(
        impulse_response=ir, sample_rate=sr,
    )
    print(f"STI:       {sti.sti:.3f}")
    print(f"Rating:    {sti.rating}")
    print(f"RT60:      {sti.rt60_measured:.2f} s")
    print(f"Compliant: {sti.compliant}  (STI >= 0.60)")
    print(f"Score:     {sti.score:.1f}/100")
except Exception as e:
    print(f"Speech intelligibility evaluation failed: {e}")
    print("Note: pyroomacoustics may have scipy compatibility issues.")""")

md("""### 6e. Ventilation Rate (CO₂ decay) — `pip install comfio[psychrometrics]`

Estimates the air-change rate (ACH) from CO₂ decay or steady-state, and scores
against ASHRAE 62.1 minimum ACH targets.

> **Theory — CO₂-based ventilation estimation.**
>
> CO₂ is a metabolically inert tracer gas emitted by occupants at a rate
> proportional to their metabolic activity. In a well-mixed room with
> outdoor-air ventilation rate $Q$ (m³/s) and occupant CO₂ generation rate
> $G$ (L/s), the steady-state indoor concentration is:
>
> $$C_{ss} = C_{out} + \\frac{G}{Q} \\times 10^6$$
>
> where $C_{out}$ is the outdoor CO₂ concentration (~420 ppm). When
> occupancy ceases, CO₂ decays exponentially toward $C_{out}$:
>
> $$C(t) = C_{out} + (C_0 - C_{out}) \\, e^{-N t}$$
>
> where $N = Q / V$ is the **air-change rate** (ACH, 1/h) and $V$ is the
> room volume (m³). Fitting the exponential decay to measured CO₂ data
> yields $N$ directly — this is the **CO₂ decay method** (ASTM D7297).
>
> The **steady-state method** inverts the first equation using measured
> $C_{ss}$ and known $G$:
>
> $$N = \\frac{G \\times 10^6}{V \\, (C_{ss} - C_{out})}$$
>
> **Ventilation efficiency** $\\epsilon_v$ compares the actual ACH to the
> ASHRAE 62.1 minimum required ACH for the occupancy type:
>
> $$\\epsilon_v = \\frac{N_{measured}}{N_{required}}$$
>
> ASHRAE 62.1-2022 minimum outdoor air rates (office, per person):
> ~5 L/s per person + 0.3 L/s per m² floor area.
>
> **References:**
> - ASHRAE Standard 62.1-2022, *Ventilation for Acceptable Indoor Air
>   Quality*, Table 6.1 — minimum ventilation rates.
> - ASTM D7297-14, *Standard Practice for Determining Ventilation
>   Effectiveness of Residential and Commercial Buildings* — CO₂ decay
>   method.
> - Persily, A. (2015), "Challenges in developing ventilation and indoor
>   air quality standards", *Building and Environment* 91, 61–69.
>   — peer-reviewed discussion of CO₂ as a ventilation indicator.
> - ASTM E741-11, *Standard Test Method for Determining Air Change in a
>   Single Zone by Means of a Tracer Gas Dilution* — general tracer-gas
>   decay methodology.
> - Carrer, P. et al. (2018), "What does the scientific literature tell us
>   about the ventilation–health relationship in public and residential
>   buildings?", *Building and Environment* 133, 267–286. — WHO/EU review.""")

code("""from comfio import evaluate_ventilation
vent = evaluate_ventilation(
    co2=sensor.get_validated("co2_ppm"),
    outdoor_co2=420.0,
    room_volume=50.0,
    n_occupants=5,
)
print(f"ACH:                 {vent.ach:.2f} /h  (method: {vent.ach_method})")
print(f"Ventilation eff.:    {vent.ventilation_efficiency:.2f}")
print(f"CO₂ peak:            {vent.co2_peak:.0f} ppm")
print(f"CO₂ steady-state:    {vent.co2_steady_state:.0f} ppm")
print(f"Compliant:           {vent.compliant}")
print(f"Score:               {vent.score:.1f}/100")""")

md("""### 6f. Psychrometrics — `pip install comfio[psychrometrics]`

Full moist-air properties (wet bulb, dew point, enthalpy, humidity ratio,
vapour pressure) via PsychroLib (ASHRAE Handbook — Fundamentals, 2017).

> **Theory — Psychrometrics.**
>
> Psychrometrics describes the thermodynamic properties of moist air (a
> mixture of dry air and water vapour). The fundamental relations are
> defined in ASHRAE Handbook — Fundamentals (2017), Chapter 1:
>
> - **Saturation vapour pressure** (Hyland & Wexler 1983 equations):
>   $p_{ws}(T) = f(T)$ — the partial pressure of water vapour in saturated
>   air at temperature $T$.
> - **Humidity ratio** (kg water / kg dry air):
>   $W = 0.622 \\, p_w / (p - p_w)$, where $p_w$ is the actual vapour
>   pressure and $p$ is total atmospheric pressure.
> - **Relative humidity**: $\\phi = p_w / p_{ws}(T)$.
> - **Enthalpy** of moist air (J/kg dry air):
>   $h = 1.006 \\, T + W \\, (2501 + 1.86 \\, T)$, where $T$ is dry-bulb
>   temperature (°C).
> - **Wet-bulb temperature** ($T_{wb}$): the temperature read by a
>   thermometer covered in water-soaked cloth over which air flows. It is
>   the lowest temperature achievable by evaporative cooling.
> - **Dew-point temperature** ($T_{dp}$): the temperature at which water
>   vapour begins to condense ($\\phi = 100\\%$).
> - **Specific volume** (m³/kg dry air): $v = R_{da} T (1 + 1.6078 W) / p$.
>
> These properties are essential for HVAC load calculations, condensation
> risk assessment, and thermal comfort analysis.
>
> **References:**
> - ASHRAE Handbook — Fundamentals (2017), Ch. 1: *Psychrometrics*.
> - Hyland, R.W. & Wexler, A. (1983), "Formulations for the thermodynamic
>   properties of dry air from 173.15 K to 473.15 K, and of saturated moist
>   air from 173.15 K to 372.15 K", *ASHRAE Transactions* 89(2A).
>   — peer-reviewed source of the ASHRAE psychrometric equations.
> - PsychroLib: https://github.com/psychrometrics/psychrolib — open-source
>   implementation of ASHRAE equations.
> - Wilhelm, L.R. (1976), "Numerical calculation of psychrometric
>   properties", *Transactions of the ASAE* 19(2), 318–325.""")

code("""from comfio import get_psychrometrics
# Sample 500 points to keep output manageable
idx_sample = np.linspace(0, n-1, 500, dtype=int)
psych_results = []
for i in idx_sample:
    p = get_psychrometrics(
        tdb=float(df.air_temp_c.iloc[i]),
        rh=float(df.relative_humidity_pct.iloc[i])/100.0,
        pressure=101325.0,
    )
    psych_results.append((df.timestamp.iloc[i], p.twb, p.tdew, p.enthalpy, p.hum_ratio, p.vapor_pressure, p.moist_air_volume))

psych_df = pd.DataFrame(psych_results, columns=["timestamp","twb","tdew","enthalpy","hum_ratio","vap_pressure","moist_vol"])
print(psych_df.describe().round(2))""")

code("""fig = make_subplots(rows=2, cols=2, shared_xaxes=True,
                    subplot_titles=("Wet bulb & dew point (°C)", "Enthalpy (kJ/kg)",
                                    "Humidity ratio (kg/kg)", "Vapour pressure (Pa)"))
fig.add_trace(go.Scatter(x=psych_df.timestamp, y=psych_df.twb, name="T_wb", line=dict(width=0.5)), row=1, col=1)
fig.add_trace(go.Scatter(x=psych_df.timestamp, y=psych_df.tdew, name="T_dew", line=dict(width=0.5)), row=1, col=1)
fig.add_trace(go.Scatter(x=psych_df.timestamp, y=psych_df.enthalpy, name="h", line=dict(width=0.5)), row=1, col=2)
fig.add_trace(go.Scatter(x=psych_df.timestamp, y=psych_df.hum_ratio, name="W", line=dict(width=0.5)), row=2, col=1)
fig.add_trace(go.Scatter(x=psych_df.timestamp, y=psych_df.vap_pressure, name="p_v", line=dict(width=0.5)), row=2, col=2)
fig.update_layout(height=600, title="Psychrometric properties (500-point sample)", showlegend=True)
fig.show()""")

# ===========================================================================
# 7. GLOBAL IEQ INDEX
# ===========================================================================
md("""## 7. Global IEQ Index & Weight Presets

The Global IEQ Index is a weighted average of domain scores (0–100). Weights
are renormalised when domains are missing. Five presets are available:

| Preset | thermal | iaq | visual | acoustic | Source |
|--------|---------|-----|--------|----------|--------|
| default | 0.40 | 0.25 | 0.20 | 0.15 | Pierson et al. (2019) |
| equal | 0.25 | 0.25 | 0.25 | 0.25 | — |
| school | 0.27 | 0.26 | 0.24 | 0.23 | Yang et al. (2020) |
| office | 0.45 | 0.30 | 0.15 | 0.10 | office emphasis |
| healthcare | 0.25 | 0.40 | 0.15 | 0.20 | IAQ emphasis |""")

code("""# Baseline: 4 core domains
ieq_base = calculate_global_ieq(thermal=thermal, visual=visual, acoustic=acoustic, iaq=iaq)
print(f"Baseline (4 domains)  IEQ mean={np.mean(ieq_base.index):.1f}  domains={ieq_base.domains}")

# With pollutant IAQ (blends 50/50 with CO2)
ieq_poll = calculate_global_ieq(thermal=thermal, visual=visual, acoustic=acoustic, iaq=iaq, pollutant_iaq=pollutant)
print(f"With pollutant IAQ     IEQ mean={np.mean(ieq_poll.index):.1f}  domains={ieq_poll.domains}")

# With TSV (overrides thermal)
ieq_tsv = calculate_global_ieq(thermal=thermal, visual=visual, acoustic=acoustic, iaq=iaq, pollutant_iaq=pollutant, tsv=tsv_result)
print(f"With pollutant + TSV   IEQ mean={np.mean(ieq_tsv.index):.1f}  domains={ieq_tsv.domains}")

# With ventilation + adaptive
ieq_full = calculate_global_ieq(
    thermal=thermal, visual=visual, acoustic=acoustic, iaq=iaq,
    pollutant_iaq=pollutant, tsv=tsv_result, ventilation=vent,
)
print(f"Full integration       IEQ mean={np.mean(ieq_full.index):.1f}  domains={ieq_full.domains}")""")

code("""# Compare weight presets on the full integration
print("Weight preset comparison (full integration):")
for preset in ["default", "equal", "school", "office", "healthcare"]:
    w = preset_weights(preset)
    ieq_w = calculate_global_ieq(
        thermal=thermal, visual=visual, acoustic=acoustic, iaq=iaq,
        pollutant_iaq=pollutant, tsv=tsv_result, ventilation=vent,
        weights=w,
    )
    print(f"  {preset:12s}  IEQ={np.mean(ieq_w.index):.1f}  weights={ieq_w.weights_used}")""")

code("""# Custom weights
custom = custom_weights(thermal=0.5, visual=0.1, acoustic=0.1, iaq=0.3)
ieq_custom = calculate_global_ieq(thermal=thermal, visual=visual, acoustic=acoustic, iaq=iaq, weights=custom)
print(f"Custom weights  IEQ={np.mean(ieq_custom.index):.1f}  weights={ieq_custom.weights_used}")""")

code("""fig = go.Figure()
fig.add_trace(go.Scatter(x=df.timestamp, y=ieq_base.index, name="4 domains", line=dict(width=0.5)))
fig.add_trace(go.Scatter(x=df.timestamp, y=ieq_poll.index, name="+ pollutant", line=dict(width=0.5)))
fig.add_trace(go.Scatter(x=df.timestamp, y=ieq_tsv.index, name="+ TSV", line=dict(width=0.5)))
fig.add_trace(go.Scatter(x=df.timestamp, y=ieq_full.index, name="full", line=dict(width=0.5, color="black")))
fig.add_hline(y=80, line_dash="dash", line_color="green", annotation_text="compliance threshold 80")
fig.update_layout(title="Global IEQ Index — progressive domain integration", yaxis_title="IEQ (0-100)", height=450)
fig.show()""")

code("""# Stacked domain scores for the full integration
fig = go.Figure()
for domain, scores in ieq_full.domain_scores.items():
    fig.add_trace(go.Scatter(x=df.timestamp, y=scores, name=domain, mode="lines", line=dict(width=0.5), stackgroup="d"))
fig.update_layout(title="Domain score breakdown (full integration)", yaxis_title="score", height=450)
fig.show()""")

# ===========================================================================
# 8. COMPLIANCE & CONTRACTS
# ===========================================================================
md("""## 8. Compliance & Performance Contracts

`calculate_compliance` converts the IEQ Index array into time-based compliance
metrics. The `ComplianceReport` maps to a Solidity ABI for blockchain oracle
integration in performance-based smart contracts.""")

code("""report = calculate_compliance(ieq_full, threshold=80.0)
print(f"IEQ avg:         {report.ieq_index_avg:.1f}")
print(f"IEQ min/max:     {report.ieq_index_min:.1f} / {report.ieq_index_max:.1f}")
print(f"Compliance rate: {report.compliance_rate_pct:.1f}%  (threshold {report.threshold:.0f})")
print(f"Compliant hours: {report.compliant_hours:.0f} / {report.total_hours:.0f}")
print(f"\\nPer-domain compliance (% with score >= 80):")
for d, rate in report.domain_compliance.items():
    print(f"  {d:14s}: {rate:.1f}%  (avg score {report.domain_scores_avg[d]:.1f})")""")

code("""from comfio.performance.contract_schema import default_compliance_schema
schema = default_compliance_schema()
print(f"Contract: {schema.contract_name}  Function: {schema.function_name}")
print(f"\\nFields ({len(schema.fields)}):")
for f in schema.fields:
    print(f"  {f.name:25s} {f.solidity_type:10s} <- {f.source}")""")

code("""payload = report.to_contract_payload()
print("Solidity-ready payload:")
for k, v in payload.items():
    print(f"  {k:25s} {str(v):<20s} ({type(v).__name__})")""")

code("""abi = schema.to_abi()
print("ABI fragment:")
print(json.dumps(abi, indent=2))""")

# ===========================================================================
# 9. ML INTEGRATION
# ===========================================================================
md("""## 9. ML Integration — Next-Day IEQ Forecast

We frame a **next-day forecast** problem: given today's IEQ features, predict
tomorrow's mean IEQ Index. Three backends are demonstrated, each using the
corresponding comfio ML adapter:

1. **scikit-learn** — `IEQFeatureExtractor` transforms daily sensor DataFrames
   into IEQ features, then RandomForest predicts next-day IEQ.
2. **PyTorch** — `IEQTimeSeriesDataset` wraps the 10-min sensor DataFrame into
   1-day windows with IEQ scores; an LSTM predicts the next day's mean.
3. **Keras/TensorFlow** — `IEQPreprocessingLayer` computes IEQ features from
   daily sensor DataFrames; a dense model predicts next-day IEQ.

First, build daily-mean sensor DataFrames.""")

code("""# Daily-mean sensor DataFrame (columns match comfio canonical names)
daily_df = df.set_index("timestamp").resample("1D").mean().reset_index()
print(f"Daily rows: {len(daily_df)}")
print(f"Columns: {list(daily_df.columns)}")
daily_df.head()""")

md("""### 9a. scikit-learn — `IEQFeatureExtractor` + RandomForest

`IEQFeatureExtractor` is an sklearn-compatible transformer that wraps comfio
domain evaluations. Given a DataFrame with sensor columns, it computes the
Global IEQ Index and per-domain scores per row.""")

code("""from comfio.ml.sklearn_transformers import IEQFeatureExtractor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# IEQFeatureExtractor computes IEQ features per day from sensor columns
extractor = IEQFeatureExtractor()
extractor.fit(daily_df)
ieq_features = extractor.transform(daily_df)
print(f"IEQ features shape: {ieq_features.shape}  names: {extractor._feature_names}")

# ieq_index is the first column; build supervised pairs (day i -> day i+1)
ieq_daily = ieq_features[:, 0]  # IEQ index per day
X_sk = ieq_features[:-1]         # today's features
y_sk = ieq_daily[1:]             # tomorrow's IEQ
split = int(0.8 * len(X_sk))
X_sk_tr, X_sk_te = X_sk[:split], X_sk[split:]
y_sk_tr, y_sk_te = y_sk[:split], y_sk[split:]

rf = RandomForestRegressor(n_estimators=200, random_state=42)
rf.fit(X_sk_tr, y_sk_tr)
pred_sk = rf.predict(X_sk_te)
mse_sk = mean_squared_error(y_sk_te, pred_sk)
mae_sk = mean_absolute_error(y_sk_te, pred_sk)
r2_sk = r2_score(y_sk_te, pred_sk)
print(f"sklearn RF  MSE={mse_sk:.2f}  MAE={mae_sk:.2f}  R²={r2_sk:.3f}")
print(f"Feature importances: {dict(zip(extractor._feature_names, rf.feature_importances_.round(3)))}")""")

code("""days_test = daily_df["timestamp"].iloc[split+1:]
fig = go.Figure()
fig.add_trace(go.Scatter(x=days_test, y=y_sk_te, mode="lines+markers", name="Actual IEQ"))
fig.add_trace(go.Scatter(x=days_test, y=pred_sk, mode="lines+markers", name="sklearn RF"))
fig.update_layout(title="Next-day IEQ forecast — sklearn RandomForest", yaxis_title="IEQ", height=400)
fig.show()""")

md("""### 9b. PyTorch — `IEQTimeSeriesDataset` + LSTM

`IEQTimeSeriesDataset` wraps the 10-min sensor DataFrame into windowed samples
with computed IEQ scores. We use 1-day windows (144 timesteps at 10-min
sampling) and predict the next day's mean IEQ Index.""")

code("""import torch
from comfio.ml.torch_dataset import IEQTimeSeriesDataset

# IEQTimeSeriesDataset on 10-min data: 1-day windows, non-overlapping
dataset = IEQTimeSeriesDataset(df, window_size=144, stride=144)
print(f"PyTorch dataset windows: {len(dataset)}")
sample = dataset[0]
print(f"Sample keys: {list(sample.keys())}")
print(f"  ieq_index shape: {sample['ieq_index'].shape}")
print(f"  raw shape: {sample['raw'].shape}")

# Build supervised pairs: window i -> mean IEQ of window i+1
ieq_windows = np.array([dataset[i]["ieq_index"] for i in range(len(dataset))])
X_seq = ieq_windows[:-1]               # (n-1, 144) — today's IEQ series
y_seq = ieq_windows[1:].mean(axis=1)   # (n-1,) — tomorrow's mean IEQ

X_t = torch.tensor(X_seq, dtype=torch.float32).unsqueeze(-1)  # (n-1, 144, 1)
y_t = torch.tensor(y_seq, dtype=torch.float32)
split_t = int(0.8 * len(X_t))
print(f"Train: {split_t}  Test: {len(X_t)-split_t}")""")

code("""class IEQLSTM(torch.nn.Module):
    def __init__(self, hidden=32):
        super().__init__()
        self.lstm = torch.nn.LSTM(1, hidden, num_layers=2, batch_first=True, dropout=0.1)
        self.head = torch.nn.Sequential(torch.nn.Linear(hidden, 16), torch.nn.ReLU(), torch.nn.Linear(16, 1))
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :]).squeeze(-1)

torch.manual_seed(42)
model = IEQLSTM()
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = torch.nn.MSELoss()

epochs = 80
train_losses, test_losses = [], []
for ep in range(epochs):
    model.train()
    pred = model(X_t[:split_t])
    loss = loss_fn(pred, y_t[:split_t])
    opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        vp = model(X_t[split_t:])
        vl = loss_fn(vp, y_t[split_t:]).item()
    train_losses.append(loss.item())
    test_losses.append(vl)

pred_torch = vp.numpy()
y_test_t = y_t[split_t:].numpy()
mse_t = mean_squared_error(y_test_t, pred_torch)
mae_t = mean_absolute_error(y_test_t, pred_torch)
r2_t = r2_score(y_test_t, pred_torch)
print(f"PyTorch LSTM  MSE={mse_t:.2f}  MAE={mae_t:.2f}  R²={r2_t:.3f}")""")

code("""days_lstm = daily_df["timestamp"].iloc[split_t+1:]
fig = make_subplots(rows=2, cols=1, subplot_titles=("Training loss", "Next-day forecast"))
fig.add_trace(go.Scatter(y=train_losses, name="train loss"), row=1, col=1)
fig.add_trace(go.Scatter(y=test_losses, name="test loss"), row=1, col=1)
fig.add_trace(go.Scatter(x=days_lstm, y=y_test_t, mode="lines+markers", name="Actual"), row=2, col=1)
fig.add_trace(go.Scatter(x=days_lstm, y=pred_torch, mode="lines+markers", name="LSTM"), row=2, col=1)
fig.update_layout(height=600, title="PyTorch LSTM — training & forecast")
fig.show()""")

md("""### 9c. Keras / TensorFlow — `IEQPreprocessingLayer`

`IEQPreprocessingLayer` wraps comfio domain evaluations as a callable Keras
preprocessing layer. It accepts a pandas DataFrame and returns IEQ features
as a TensorFlow tensor.""")

code("""import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
from comfio.ml.keras_adapter import IEQPreprocessingLayer

# IEQPreprocessingLayer computes IEQ features from the daily DataFrame
layer = IEQPreprocessingLayer()
ieq_features_k = layer(daily_df).numpy()
print(f"Keras IEQ features shape: {ieq_features_k.shape}")

# Supervised pairs: day i -> day i+1
ieq_daily_k = ieq_features_k[:, 0]
X_k = ieq_features_k[:-1]
y_k = ieq_daily_k[1:]
split_k = int(0.8 * len(X_k))

# Dense Keras model
keras_model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation="relu", input_shape=(X_k.shape[1],)),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Dense(32, activation="relu"),
    tf.keras.layers.Dense(1, name="ieq_pred"),
])
keras_model.compile(optimizer="adam", loss="mse", metrics=["mae"])
keras_model.summary()""")

code("""history = keras_model.fit(X_k[:split_k], y_k[:split_k], epochs=80, batch_size=8,
                        validation_split=0.15, verbose=0)
pred_keras = keras_model.predict(X_k[split_k:], verbose=0).flatten()
y_test_k = y_k[split_k:]
mse_k = mean_squared_error(y_test_k, pred_keras)
mae_k = mean_absolute_error(y_test_k, pred_keras)
r2_k = r2_score(y_test_k, pred_keras)
print(f"Keras model  MSE={mse_k:.2f}  MAE={mae_k:.2f}  R²={r2_k:.3f}")""")

code("""days_keras = daily_df["timestamp"].iloc[split_k+1:]
fig = make_subplots(rows=2, cols=1, subplot_titles=("Keras training history", "Forecast comparison"))
fig.add_trace(go.Scatter(y=history.history["loss"], name="train loss"), row=1, col=1)
fig.add_trace(go.Scatter(y=history.history["val_loss"], name="val loss"), row=1, col=1)
fig.add_trace(go.Scatter(x=days_keras, y=y_test_k, mode="lines+markers", name="Actual"), row=2, col=1)
fig.add_trace(go.Scatter(x=days_keras, y=pred_keras, mode="lines+markers", name="Keras"), row=2, col=1)
fig.update_layout(height=600, title="Keras/TensorFlow — training & forecast")
fig.show()""")

code("""# --- Model comparison ---
print("=" * 60)
print(f"{'Model':<25s} {'MSE':>8s} {'MAE':>8s} {'R²':>8s}")
print("-" * 60)
for name, yt, pt in [("sklearn RandomForest", y_sk_te, pred_sk),
                      ("PyTorch LSTM", y_test_t, pred_torch),
                      ("Keras/TensorFlow", y_test_k, pred_keras)]:
    print(f"{name:<25s} {mean_squared_error(yt, pt):8.2f} {mean_absolute_error(yt, pt):8.2f} {r2_score(yt, pt):8.3f}")
print("=" * 60)""")

# ===========================================================================
# 10. LLM INTEGRATION
# ===========================================================================
md("""## 10. LLM Integration

comfio provides three layers for LLM integration:

1. **Interpreters** — `ieq_to_markdown`, `ieq_to_summary_dict`,
   `generate_markdown_summary` convert results into token-efficient text.
2. **Prompts** — `EDGE_SYSTEM_PROMPT`, `DIAGNOSTIC_PROMPT_TEMPLATE` for
   guarded building-diagnostic agents.
3. **Tool schemas** — `to_openai_tools()`, `to_langchain_tools()` expose
   comfio evaluations as function-calling tools.""")

code("""from comfio import (
    ieq_to_markdown, ieq_to_summary_dict, generate_markdown_summary,
    EDGE_SYSTEM_PROMPT, DIAGNOSTIC_PROMPT_TEMPLATE, format_prompt,
)
md_report = ieq_to_markdown(ieq_full, compliance_report=report, zone_id="A-101")
print(md_report)""")

code("""summary = ieq_to_summary_dict(ieq_full, compliance_report=report)
print(json.dumps(summary, indent=2))""")

code("""# One-shot markdown summary from the raw DataFrame (first 7 days)
md_auto = generate_markdown_summary(df.iloc[:7*144], window_hours=24, threshold=80.0, zone_id="A-101")
print(md_auto[:1500])""")

code("""from comfio.llm.tools import to_openai_tools, to_langchain_tools
tools = to_openai_tools()
print(f"OpenAI tool schemas: {len(tools)}")
for t in tools:
    print(f"  - {t['function']['name']}: {t['function']['description'][:70]}...")""")

code("""try:
    lc_tools = to_langchain_tools()
    print(f"LangChain tools: {len(lc_tools)}")
    for t in lc_tools:
        print(f"  - {t.name}: {t.description[:70]}...")
except Exception as e:
    print(f"LangChain tools not available: {e}")
    print("Install with: pip install langchain")""")

code("# Example diagnostic prompt\nprompt = format_prompt(\n    DIAGNOSTIC_PROMPT_TEMPLATE,\n    ieq_report=md_report,\n    complaint=\"Occupants report feeling warm in the afternoons.\",\n    pollutant_status=\"PM2.5 within WHO 24h limit\",\n    adaptive_status=\"Within ASHRAE 80% band\",\n    tsv_status=f\"Mean TSV={np.mean(tsv_aug):+.2f}, compliance={tsv_result.compliance_rate*100:.0f}%\",\n)\nprint(f\"Diagnostic prompt ({len(prompt)} chars):\\n\")\nprint(prompt[:800])")

# ===========================================================================
# 11. SMART CONTRACT EXPORT
# ===========================================================================
md("""## 11. Smart-Contract Export (web3.py)

The compliance payload maps directly to a Solidity `submitCompliance` function.
Below we show how to serialise it for an oracle and (optionally) sign a
transaction with web3.py.""")

code("""contract_json = report.to_contract_json()
print(f"Contract JSON ({len(contract_json)} chars):")
print(contract_json)""")

code("""# Example: how this would be sent to a deployed IEQComplianceOracle
# (We do NOT send a real transaction — just demonstrate the ABI call encoding.)
try:
    from web3 import Web3
    abi = [schema.to_abi()]
    # In production: w3 = Web3(Web3.HTTPProvider(INFURA_URL))
    # contract = w3.eth.contract(address=ORACLE_ADDR, abi=abi)
    # tx = contract.functions.submitCompliance(**payload).build_transaction({...})
    print("web3.py available. ABI call encoding would produce:")
    # Encode the function call (without sending)
    from eth_abi import encode
    encoded = encode(
        ["uint256","uint256","uint8","uint8","bool","bool","bool","bool","uint32","uint32"],
        [payload["periodStart"], payload["periodEnd"], payload["ieqIndexAvg"],
         payload["complianceRatePct"], payload["thermalCompliant"], payload["visualCompliant"],
         payload["acousticCompliant"], payload["iaqCompliant"],
         payload["totalOccupiedHours"], payload["compliantHours"]],
    )
    print(f"  Encoded calldata: 0x{encoded.hex()[:80]}...  ({len(encoded)} bytes)")
except ImportError as e:
    print(f"web3/eth-abi not available for encoding demo: {e}")
""")

# ===========================================================================
# 12. REPORTS
# ===========================================================================
md("""## 12. Reports — CSV / PDF / DOCX / Intelligent Pipeline

comfio can export IEQ results to CSV, PDF (reportlab), DOCX (python-docx),
and run an intelligent pipeline that auto-detects and evaluates all available
domains.""")

code("""# CSV export
csv_str = ieq_to_csv(ieq_full, compliance_report=report)
print(f"CSV export: {len(csv_str)} chars, {csv_str.count(chr(10))} lines")
print(csv_str[:300])""")

code("""# Intelligent pipeline (auto-detects and runs everything)
pipe_result = run_pipeline(sensor, config={"threshold": 80.0})
print(f"Pipeline capabilities detected: {sum(pipe_result.capabilities.values())}/{len(pipe_result.capabilities)}")
print(f"Domains evaluated: {list(pipe_result.domain_results.keys())}")
print(f"IEQ mean: {np.mean(pipe_result.ieq_result.index):.1f}" if pipe_result.ieq_result else "No IEQ result")
print(f"Warnings: {len(pipe_result.warnings)}")
for w in pipe_result.warnings[:5]:
    print(f"  - {w}")""")

code("""# PDF export (requires reportlab)
from comfio.reports import ieq_to_pdf
try:
    import tempfile, os
    pdf_bytes = ieq_to_pdf(ieq_full, compliance_report=report, zone_id="A-101")
    pdf_path = os.path.join(tempfile.gettempdir(), "comfio_walkthrough.pdf")
    with open(pdf_path, "wb") as f:
        f.write(pdf_bytes)
    print(f"PDF written: {pdf_path}  ({len(pdf_bytes):,} bytes)")
except Exception as e:
    print(f"PDF export: {e}")""")

code("""# DOCX export (requires python-docx)
from comfio.reports import ieq_to_docx
try:
    import tempfile, os
    docx_bytes = ieq_to_docx(ieq_full, compliance_report=report, zone_id="A-101")
    docx_path = os.path.join(tempfile.gettempdir(), "comfio_walkthrough.docx")
    with open(docx_path, "wb") as f:
        f.write(docx_bytes)
    print(f"DOCX written: {docx_path}  ({len(docx_bytes):,} bytes)")
except Exception as e:
    print(f"DOCX export: {e}")""")

code("""# Reproducible script export
from comfio.reports import generate_pipeline_script
try:
    script = generate_pipeline_script(pipe_result.config)
    print(f"Generated reproducible script: {len(script)} chars")
    print(script[:500])
except Exception as e:
    print(f"Script export: {e}")""")

# ===========================================================================
# REFERENCES & CHANGELOG
# ===========================================================================
md("""## References

1. **ISO 7730:2005** — Ergonomics of the thermal environment — Analytical determination and interpretation of thermal comfort using PMV and PPD indices.
2. **ASHRAE 55-2023** — Thermal Environmental Conditions for Human Occupancy (incl. Appendix L — TSV analysis).
3. **EN 16798-1:2019** — Energy performance of buildings — Ventilation for buildings (adaptive comfort categories I–III).
4. **EN 12464-1:2021** — Light and lighting — Lighting of work places (illuminance targets, UGR limits).
5. **ASHRAE 62.1-2022** — Ventilation for Acceptable Indoor Air Quality.
6. **Buratti, C., Ricciardi, P., & Vergoni, M. (2009)**. Simplified PMV model for HVAC systems control. *Building and Environment*, 44(3), 441–449.
7. **Pierson, A., et al. (2019)**. Multi-domain IEQ weighting study.
8. **Cao, B., et al. (2012)**. Individual weight differences in IEQ assessment.
9. **Frontczak, M., & Wargocki, P. (2011)**. Literature survey on how different factors influence human comfort in indoor environments.
10. **Yang, W., et al. (2020)**. IEQ preferences of school children.
11. **WHO Global Air Quality Guidelines (2021)** — PM2.5, PM10, CO thresholds.
12. **WELL Building Standard v2** — TVOC, formaldehyde, CO limits.
13. **ASHRAE Handbook — HVAC Applications, Ch. 49** — Noise Criteria (NC) curves.
14. **IEC 60268-16** — Speech Transmission Index.
15. **PsychroLib** (ASHRAE Handbook — Fundamentals, 2017) — psychrometric properties.
16. **Wienold & Christoffersen** — Daylight Glare Probability (DGP).

## Changelog (living document)

| Date | Version | Change |
|------|---------|--------|
| 2025-01 | comfio 0.1.5 | Initial walkthrough covering all public APIs, ML (sklearn/PyTorch/Keras), LLM, smart-contract, and report integrations. |

> **To extend this walkthrough when new comfio functionality is added**: add a
> new section above the References, update the overview table at the top, and
> add a changelog row. The companion notebook
> (`examples/walkthrough_executed.ipynb`) should be re-executed with
> `jupyter nbconvert --execute --to notebook --inplace` after any change.""")

# ===========================================================================
# WRITE
# ===========================================================================
nb.cells = cells
out = r"C:\Users\utente\Desktop\Projects\ComfortPy\examples\walkthrough_executed.ipynb"
with open(out, "w", encoding="utf-8") as f:
    nbf.write(nb, f)
print(f"Wrote {len(cells)} cells to {out}")
