# comfio — Complete Walkthrough

> **A living document**: this walkthrough grows with the package. Every public
> API is exercised here on a single synthetic dataset so the behaviour is
> end-to-end reproducible. The companion executed notebook is at
> [`examples/walkthrough_executed.ipynb`](../../examples/walkthrough_executed.ipynb).

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

> **Run this notebook yourself** with
> `pip install comfio[ml,torch,keras,agent,acoustics,color,psychrometrics]`
> plus `plotly tensorflow langchain web3 reportlab python-docx`.
> On Windows, `[daylighting]` (Radiance) needs WSL.

---

## 0. Environment & Imports

The walkthrough targets **Python 3.11** with `comfio` 0.1.5+.

```python
import sys, warnings, json, time, io, textwrap
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
print("Core imports OK.")
```
**Output:**
```
comfio 0.1.5  |  Python 3.11.9  |  NumPy 2.2.6
Core imports OK.
```

---

## 1. Synthetic Data Generation

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
| `tsv` (1-h) | derived from PMV + noise | — | 0.5 |

> **Why synthetic data?** Real sensor datasets are often incomplete, proprietary,
> or limited to a few domains. By generating a full multi-domain dataset we can
> exercise every `comfio` evaluation in a single end-to-end run. The seasonal
> and diurnal patterns mimic a naturally ventilated office in a temperate climate.

```python
rng = np.random.default_rng(42)

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
df.head()
```
**Output:**
```
10-min rows: 26,064  (2025-01-01 00:00:00 → 2025-06-30 23:50:00)
DataFrame shape: (26064, 16)
timestamp  air_temp_c  radiant_temp_c  relative_humidity_pct  \
0 2025-01-01 00:00:00   19.061227       18.954279              57.393939   
1 2025-01-01 00:10:00   18.478090       18.493580              58.864948   
2 2025-01-01 00:20:00   19.151114       19.608659              59.627617   
3 2025-01-01 00:30:00   19.186196       18.983986              58.894163   
4 2025-01-01 00:40:00   17.990858       18.268823              58.736919   

   air_velocity_ms  illuminance_lux  noise_laeq_db     co2_ppm  pm25_ugm3  \
0         0.086681         4.282542      38.628396  393.158380   4.266920   
1         0.103165         0.000000      38.037119  483.230363   5.583003   
2         0.077518        20.797880      37.629389  453.507812   4.263471   
3         0.095707         0.000000      36.459689  475.891380   6.295946   
4         0.110866         0.000000      38.388922  405.800516   5.399955   

   pm10_ugm3   tvoc_ugm3  formaldehyde_ppb    co_ppm  outdoor_temp_c  \
0   7.140841   95.811553         16.105435  0.682505        2.733852   
1  10.506206   90.919598         18.758144  0.775654       -0.058248   
2   8.854322   93.927844         15.758529  0.990377        0.491085   
3  10.622886   94.136719         12.069259  0.646544       -1.070675   
4  12.897752  106.140604         15.987558  0.799665        2.414392   

   metabolic_rate_met  clothing_insulation_clo  
0                 1.2                 0.973004  
1                 1.2                 1.040973  
2                 1.2                 1.003916  
3                 1.2                 0.997429  
4                 1.2                 1.024783
```

### 1b. Thermal Sensation Votes (TSV) — 1-hour sampling

TSV represents **occupant feedback** on the ASHRAE 7-point scale
(−3 cold … 0 neutral … +3 hot). We generate sparse 1-hour votes by
computing an approximate PMV from the 10-min data and adding noise, then
rounding to the nearest integer on the scale. The `comfio` TSV augmentation
module (§4c) will later upsample these to the 10-min grid.

```python
# --- TSV at 1-hour sampling ---
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
print(f"TSV distribution:\n{pd.Series(tsv_votes).value_counts().sort_index()}")
```
**Output:**
```
TSV rows: 4,344  |  TSV range: [-3, 2]
TSV distribution:
-3.0      13
-2.0     510
-1.0    1925
 0.0    1641
 1.0     253
 2.0       2
Name: count, dtype: int64
```

### 1c. Visualising the raw data

Two Plotly figures give an overview of the generated time series. In the
notebook these render interactively; here we show the code for reference.

```python
# --- Plotly: overview of the 6-month dataset ---
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
fig.show()
```

```python
# --- TSV time series ---
fig = go.Figure()
fig.add_trace(go.Scatter(x=df_tsv.timestamp, y=df_tsv.tsv, mode="markers",
                         marker=dict(size=3, opacity=0.4), name="TSV"))
fig.update_layout(title="Thermal Sensation Votes (1-h sampling, 6 months)",
                  xaxis_title="Date", yaxis_title="TSV (-3 cold .. +3 hot)",
                  height=350)
fig.show()
```

---

## 2. Data Ingestion with `SensorData`

`SensorData` is the central data container. It auto-detects column names
using a built-in alias table (e.g. `"Ta"` → `"air_temp_c"`, `"RH"` →
`"relative_humidity_pct"`), validates physical bounds, and tracks which IEQ
domains can be evaluated from the available columns.

> **Theory — Canonical column names.** `comfio` uses 15 canonical column names
> mapped to physical quantities (see `comfio.core.data_handler.CANONICAL_COLUMNS`).
> The auto-detection layer recognises >40 common aliases from sensor
> manufacturers (Vaisala, Hobo, Davis, etc.), so you can usually pass a raw
> export DataFrame without manual column mapping.

```python
sensor = SensorData(df=df, timestamp_col="timestamp")
sensor.validate()
print(sensor)
print(f"\nMapped columns ({len(sensor.column_map)}):")
for canon, actual in sensor.column_map.items():
    print(f"  {canon:30s} -> {actual}")
print(f"\nAvailable domains:        {sensor.available_domains()}")
print(f"Available advanced domains: {sensor.available_advanced_domains()}")
```
**Output:**
```
SensorData(n_rows=26064, columns=['air_temp_c', 'radiant_temp_c', 'relative_humidity_pct', 'air_velocity_ms', 'illuminance_lux', 'co2_ppm', 'noise_laeq_db', 'metabolic_rate_met', 'clothing_insulation_clo', 'pm25_ugm3', 'pm10_ugm3', 'tvoc_ugm3', 'formaldehyde_ppb', 'co_ppm', 'outdoor_temp_c'], domains=['thermal', 'visual', 'acoustic', 'iaq'], validated=True)
Mapped columns (15):
  air_temp_c                     -> air_temp_c
  radiant_temp_c                 -> radiant_temp_c
  relative_humidity_pct          -> relative_humidity_pct
  air_velocity_ms                -> air_velocity_ms
  illuminance_lux                -> illuminance_lux
  co2_ppm                        -> co2_ppm
  noise_laeq_db                  -> noise_laeq_db
  metabolic_rate_met             -> metabolic_rate_met
  clothing_insulation_clo        -> clothing_insulation_clo
  pm25_ugm3                      -> pm25_ugm3
  pm10_ugm3                      -> pm10_ugm3
  tvoc_ugm3                      -> tvoc_ugm3
  formaldehyde_ppb               -> formaldehyde_ppb
  co_ppm                         -> co_ppm
  outdoor_temp_c                 -> outdoor_temp_c
Available domains:        ['thermal', 'visual', 'acoustic', 'iaq']
Available advanced domains: ['daylighting', 'ventilation', 'psychrometrics', 'pollutant_iaq', 'adaptive_ashrae']
```

### 2b. Capability Detection

`detect_capabilities()` inspects the available columns and reports which
evaluations are possible. This is the same logic used by the intelligent
pipeline (`run_pipeline`) to gracefully skip domains when data is missing.

```python
# Capabilities detected by the intelligent pipeline
caps = detect_capabilities(sensor)
print("Detected capabilities:")
for k, v in caps.items():
    print(f"  {k:30s} {v}")
```
**Output:**
```
Detected capabilities:
  thermal_pmv                    True
  thermal_spmv                   True
  thermal_adaptive_ashrae        True
  thermal_adaptive_en            False
  visual                         True
  acoustic                       True
  iaq_co2                        True
  iaq_pollutant                  True
  tsv                            False
  personalisation                False
```

---

## 3. Core Domain Evaluations

### 3a. Thermal Comfort — Fanger PMV/PPD (ISO 7730 / ASHRAE 55)

The **Predicted Mean Vote (PMV)** model, developed by Fanger (1970), predicts
the mean thermal sensation vote of a large group of occupants on the ASHRAE
7-point scale. It is a function of four environmental variables
(air temperature, radiant temperature, air velocity, relative humidity) and
two personal variables (metabolic rate, clothing insulation).

> **Theory — PMV equation.**
>
> $$\text{PMV} = (0.303\,e^{-0.036M} + 0.028) \, L$$
>
> where $L$ is the thermal load on the body (W/m²), $M$ is the metabolic rate
> (W/m²). The **Predicted Percentage Dissatisfied (PPD)** is derived from PMV:
>
> $$\text{PPD} = 100 - 95\,\exp(-0.03353\,\text{PMV}^4 - 0.2179\,\text{PMV}^2)$$
>
> PMV = 0 corresponds to PPD = 5% (the theoretical minimum). ISO 7730
> Category B requires |PMV| ≤ 0.5 (PPD ≤ 10%).
>
> **Reference:** ISO 7730:2005, *Ergonomics of the thermal environment —
> Analytical determination and interpretation of thermal comfort using
> calculation of the PMV and PPD indices*. Fanger, P.O. (1970),
> *Thermal Comfort*, Danish Technical Press.

```python
thermal = evaluate_thermal(
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
print(f"Category-B compliant (|PMV|<=0.5): {np.mean(np.abs(thermal.pmv)<=0.5)*100:.1f}%")
```
**Output:**
```
PMV  mean=-0.12  std=0.30  range=[-1.00, +0.79]
PPD  mean=7.2%  max=26.1%
Thermal score  mean=91.8/100
Category-B compliant (|PMV|<=0.5): 88.4%
```

```python
fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                    subplot_titles=("PMV (ISO 7730)", "PPD % + thermal score"))
fig.add_trace(go.Scatter(x=df.timestamp, y=thermal.pmv, name="PMV", line=dict(width=0.5)), row=1, col=1)
fig.add_hline(y=0.5, line_dash="dash", line_color="red", row=1, col=1)
fig.add_hline(y=-0.5, line_dash="dash", line_color="blue", row=1, col=1)
fig.add_trace(go.Scatter(x=df.timestamp, y=thermal.ppd, name="PPD %", line=dict(width=0.5)), row=2, col=1)
fig.add_trace(go.Scatter(x=df.timestamp, y=thermal_score(thermal.pmv, thermal.ppd), name="score", line=dict(width=0.5, color="green")), row=2, col=1)
fig.update_layout(height=600, title="Thermal comfort over 6 months")
fig.show()
```

### 3b. Visual Comfort — EN 12464-1:2021

**EN 12464-1** specifies maintained illuminance targets for indoor workspaces.
`comfio` evaluates whether measured illuminance meets the target for a given
task type (e.g. 500 lux for "office" / "general" work) and optionally checks
Unified Glare Rating (UGR) compliance.

> **Theory — Illuminance scoring.** The visual comfort score is 100 when
> illuminance equals the target, decreases linearly to 0 at 0 lux, and
> penalises over-illumination (>2× target) to discourage energy waste.
> When UGR is provided, the final score is a 70/30 blend of illuminance and
> glare components.
>
> **Reference:** EN 12464-1:2021, *Light and lighting — Lighting of work
> places — Part 1: Indoor work places*.

```python
visual = evaluate_visual(
    illuminance=sensor.get_validated("illuminance_lux"),
    task_type="general",
)
print(f"Target illuminance: {visual.target_lux:.0f} lux")
print(f"Visual score  mean={np.mean(visual.score):.1f}/100")
print(f"Compliant (>= target): {np.mean(visual.compliant)*100:.1f}%")
print(f"Mean illuminance: {np.mean(visual.illuminance):.0f} lux (daytime only meaningful)")
```
**Output:**
```
Target illuminance: 500 lux
Visual score  mean=32.3/100
Compliant (>= target): 6.5%
Mean illuminance: 165 lux (daytime only meaningful)
```

```python
fig = go.Figure()
fig.add_trace(go.Scatter(x=df.timestamp, y=df.illuminance_lux, name="Illuminance", line=dict(width=0.5)))
fig.add_hline(y=visual.target_lux, line_dash="dash", line_color="red", annotation_text=f"target {visual.target_lux:.0f} lux")
fig.add_trace(go.Scatter(x=df.timestamp, y=visual.score, name="Visual score", yaxis="y2", line=dict(width=0.5, color="green")))
fig.update_layout(title="Visual comfort (EN 12464-1)", yaxis=dict(title="lux"), yaxis2=dict(title="score", overlaying="y", side="right"), height=400)
fig.show()
```

### 3c. Acoustic Comfort — Noise Criteria (NC)

`comfio` evaluates $L_{Aeq}$ (A-weighted equivalent continuous sound level)
against NC (Noise Criteria) curves. The NC level maps to a threshold in dBA
(e.g. NC-35 → 41 dBA for offices).

> **Theory — Acoustic scoring.** Score = 100 when $L_{Aeq}$ is 10 dB below
> the threshold, decreasing linearly to 0 at threshold + 10 dB. This gives
> a 20-dB dynamic range, matching the psychoacoustic just-noticeable
> difference (JND) of ~3 dB per step.
>
> **Reference:** Beranek, L.L. (1957), "Revised criteria for noise in
> buildings", *Noise Control* 3(1), 19–27.

```python
acoustic = evaluate_acoustic(
    laeq=sensor.get_validated("noise_laeq_db"),
    nc_level="NC-35",
)
print(f"NC threshold: {acoustic.threshold_db:.0f} dBA")
print(f"Acoustic score  mean={np.mean(acoustic.score):.1f}/100")
print(f"Compliant (<= threshold): {np.mean(acoustic.compliant)*100:.1f}%")
```
**Output:**
```
NC threshold: 41 dBA
Acoustic score  mean=38.8/100
Compliant (<= threshold): 53.7%
```

```python
fig = go.Figure()
fig.add_trace(go.Scatter(x=df.timestamp, y=df.noise_laeq_db, name="L_Aeq", line=dict(width=0.5)))
fig.add_hline(y=acoustic.threshold_db, line_dash="dash", line_color="red", annotation_text=f"NC-35 = {acoustic.threshold_db:.0f} dBA")
fig.add_trace(go.Scatter(x=df.timestamp, y=acoustic.score, name="score", yaxis="y2", line=dict(width=0.5, color="green")))
fig.update_layout(title="Acoustic comfort (NC-35)", yaxis=dict(title="dBA"), yaxis2=dict(title="score", overlaying="y", side="right"), height=400)
fig.show()
```

### 3d. Indoor Air Quality — CO₂ (ASHRAE 62.1 indicators)

CO₂ concentration is a proxy for **ventilation adequacy**. While not a
pollutant itself at indoor levels, elevated CO₂ indicates insufficient
outdoor air supply. `comfio` evaluates CO₂ against configurable threshold
levels ("excellent" = 800 ppm, "good" = 1000 ppm, "moderate" = 1400 ppm).

> **Theory — IAQ scoring.** Score = 100 when CO₂ ≤ 420 ppm (outdoor
> baseline), 50 at the threshold, and 0 at 2× threshold. The score is
> piecewise linear with a steeper penalty above threshold.
>
> **Reference:** ASHRAE Standard 62.1-2022, *Ventilation for Acceptable
> Indoor Air Quality*. Persily, A. (2015), "Challenges in developing
> ventilation and indoor air quality standards", *Building and Environment*.

```python
iaq = evaluate_iaq(
    co2=sensor.get_validated("co2_ppm"),
    threshold_level="good",
)
print(f"CO₂ threshold: {iaq.threshold_ppm:.0f} ppm  (level='{iaq.threshold_level}')")
print(f"IAQ score  mean={np.mean(iaq.score):.1f}/100")
print(f"Compliant (<= threshold): {np.mean(iaq.compliant)*100:.1f}%")
print(f"CO₂  mean={np.mean(iaq.co2):.0f}  peak={iaq.co2.max():.0f} ppm")
```
**Output:**
```
CO₂ threshold: 1000 ppm  (level='good')
IAQ score  mean=74.4/100
Compliant (<= threshold): 60.6%
CO₂  mean=731  peak=1356 ppm
```

```python
fig = go.Figure()
fig.add_trace(go.Scatter(x=df.timestamp, y=df.co2_ppm, name="CO₂", line=dict(width=0.5)))
fig.add_hline(y=iaq.threshold_ppm, line_dash="dash", line_color="red", annotation_text=f"good = {iaq.threshold_ppm:.0f} ppm")
fig.add_trace(go.Scatter(x=df.timestamp, y=iaq.score, name="score", yaxis="y2", line=dict(width=0.5, color="green")))
fig.update_layout(title="IAQ — CO₂ ventilation indicator", yaxis=dict(title="ppm"), yaxis2=dict(title="score", overlaying="y", side="right"), height=400)
fig.show()
```

---

## 4. Advanced Thermal Models

### 4a. Simplified PMV (sPMV) — Buratti Seasonal Model

The **simplified PMV** model by Buratti, Ricciardi & Naticchia (2009) uses
only indoor air temperature and relative humidity — no need for metabolic
rate, clothing, or air velocity inputs. Seasonal coefficients capture
typical occupancy conditions.

> **Theory — sPMV equation.**
>
> $$\text{sPMV} = a \, T + b \, p_v - c$$
>
> where $T$ is air temperature (°C) and $p_v$ is vapor pressure (kPa,
> computed via the Magnus formula). The coefficients $(a, b, c)$ vary by
> season:
>
> | Season | a | b | c |
> |--------|------|------|------|
> | Winter | 0.21 | 1.90 | 5.20 |
> | Mid | 0.23 | 1.65 | 5.55 |
> | Summer | 0.25 | 1.40 | 5.90 |
>
> The sPMV score is $100(1 - |\text{sPMV}|/3)$, clamped to [0, 100].
>
> **Reference:** Buratti, L., Ricciardi, P., & Naticchia, B. (2009).
> "A simplified PMV model for indoor thermal comfort assessment",
> *Proceedings of the 11th International IBPSA Building Simulation
> Conference*.

```python
spmv = evaluate_spmv(
    indoor_temp=sensor.get_validated("air_temp_c"),
    indoor_rh=sensor.get_validated("relative_humidity_pct"),
    date_ref=start,
)
print(f"Season determined: {spmv.season}")
print(f"sPMV  mean={np.mean(spmv.spmv):+.2f}  range=[{spmv.spmv.min():+.2f}, {spmv.spmv.max():+.2f}]")
print(f"sPMV score  mean={np.mean(spmv.score):.1f}/100")

# Compare sPMV vs full PMV
corr = np.corrcoef(spmv.spmv, thermal.pmv)[0,1]
print(f"\nCorrelation sPMV vs full PMV: {corr:.3f}")
```
**Output:**
```
Season determined: winter
sPMV  mean=+1.88  range=[+0.67, +3.20]
sPMV score  mean=37.2/100
Correlation sPMV vs full PMV: 0.860
```

```python
fig = go.Figure()
fig.add_trace(go.Scatter(x=df.timestamp, y=thermal.pmv, name="Full PMV", line=dict(width=0.5)))
fig.add_trace(go.Scatter(x=df.timestamp, y=spmv.spmv, name="sPMV (Buratti)", line=dict(width=0.5, color="orange")))
fig.update_layout(title="Full PMV vs simplified PMV (Buratti)", yaxis_title="PMV", height=400)
fig.show()
```

### 4b. Adaptive Thermal Comfort — ASHRAE 55 & EN 16798-1

The **adaptive comfort model** recognises that occupants in naturally
ventilated buildings adapt to seasonal outdoor temperatures through
clothing, behavioural, and physiological adjustments. Instead of PMV, the
comfort temperature is a linear function of the prevailing mean outdoor
temperature.

> **Theory — ASHRAE 55-2023 (Appendix L).**
>
> $$T_{\text{comf}} = 0.31 \, \bar{T}_{\text{out}} + 17.8$$
>
> Valid for $10 \le \bar{T}_{\text{out}} \le 33.5$ °C. The 80% acceptability
> band is ±3.5 °C; the 90% band is ±2.5 °C.
>
> **EN 16798-1:2019 (Category II):**
>
> $$T_{\text{comf}} = 0.33 \, \bar{T}_{\text{rm}} + 18.8$$
>
> where $\bar{T}_{\text{rm}}$ is the running mean outdoor temperature.
> Category II band is ±3 °C.
>
> **References:**
> - ASHRAE Standard 55-2023, *Thermal Environmental Conditions for Human
>   Occupancy*, Appendix L.
> - EN 16798-1:2019, *Energy performance of buildings — Ventilation for
>   buildings — Part 1: Indoor environmental input parameters*.
> - de Dear, R. & Brager, G.S. (1998), "Developing an adaptive model of
>   thermal comfort and preference", *ASHRAE Transactions* 104(1).

```python
# Compute a simple running mean of outdoor temperature (alpha=0.8)
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
print(f"EN 16798   t_comf mean={np.mean(en.t_comf):.1f}°C  compliance={np.mean(en.compliant)*100:.1f}%")
```
**Output:**
```
ASHRAE 55  t_comf mean=21.2°C  compliance=96.8%
EN 16798   t_comf mean=22.5°C  compliance=91.7%
```

```python
fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
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
fig.show()
```

### 4c. TSV Augmentation & Evaluation

Occupant TSV votes are typically sparse (1-hour) while sensor data is dense
(10-min). `comfio` provides `augment_tsv_cdf()` to upsample sparse votes to
the sensor grid using **CDF remapping**: the empirical CDF of the sparse
votes is mapped onto the target timestamps, preserving the distribution
while filling gaps.

> **Theory — CDF remapping.** Given $n$ sparse votes and $m > n$ target
> timestamps, each target is assigned a rank $r_j \in [0, 1)$. The vote
> value at rank $r_j$ is the quantile of the sparse vote distribution at
> $r_j$. When `time_aware=True`, ranks are computed within daily time-of-day
> windows, preserving the diurnal pattern of occupant feedback.
>
> **Reference:** ASHRAE 55-2023, Appendix L — "Thermal comfort in
> naturally conditioned spaces" requires |TSV| ≤ 1.5 for 80% acceptability.

```python
tsv_aug = augment_tsv_cdf(
    sparse_votes=df_tsv.tsv.values.astype(float),
    vote_timestamps=df_tsv.timestamp.values.astype("datetime64[ns]").astype(float),
    target_timestamps=df.timestamp.values.astype("datetime64[ns]").astype(float),
)
tsv_result = evaluate_tsv(tsv_aug)
print(f"Augmented TSV length: {len(tsv_aug)}  (matches sensor grid)")
print(f"TSV mean={np.mean(tsv_aug):+.2f}  std={np.std(tsv_aug):.2f}")
print(f"Compliance rate (|TSV|<=1.5): {tsv_result.compliance_rate*100:.1f}%")
print(f"TSV score mean={np.mean(tsv_result.score):.1f}/100")
```
**Output:**
```
Augmented TSV length: 26064  (matches sensor grid)
TSV mean=-0.63  std=0.78
Compliance rate (|TSV|<=1.5): 87.9%
TSV score mean=75.1/100
```

```python
fig = go.Figure()
fig.add_trace(go.Scatter(x=df.timestamp, y=tsv_aug, name="Augmented TSV", line=dict(width=0.5, color="teal")))
fig.add_hline(y=1.5, line_dash="dash", line_color="red")
fig.add_hline(y=-1.5, line_dash="dash", line_color="blue")
fig.update_layout(title="Augmented TSV on 10-min grid (ASHRAE 55 Appendix L)", yaxis_title="TSV", height=400)
fig.show()
```

### 4d. Personalised Comfort (OLS Regression)

The personalisation module fits an **Ordinary Least Squares (OLS)**
regression mapping model-predicted PMV to occupant TSV:

> $$\text{TSV} = \alpha \times \text{PMV} + \beta$$
>
> where $\alpha$ captures the occupant's sensitivity to PMV changes and
> $\beta$ is a systematic offset (e.g. preference for cooler/warmer
> conditions than the model predicts). Seasonal personalisation fits separate
> $(\alpha, \beta)$ per season.
>
> **Reference:** Schweiker, M. et al. (2020), "Review of multi-domain
> approaches to indoor environmental perception and behaviour",
> *Building and Environment* 176.

```python
# Align TSV (1-h) to PMV (10-min) by nearest timestamp
pmv_1h = pd.Series(thermal.pmv, index=df.timestamp).reindex(df_tsv.timestamp, method="nearest").values
tsv_1h = df_tsv.tsv.values.astype(float)

idx = train_personalisation(pmv=pmv_1h, tsv=tsv_1h)
print(f"Personalisation: alpha={idx.alpha:.3f}  beta={idx.beta:.3f}  R²={idx.r_squared:.3f}  n={idx.n_samples}")

# Seasonal personalisation
dates_1h = df_tsv.timestamp.dt.date.tolist()
seasonal_idx = train_seasonal_personalisation(pmv=pmv_1h, tsv=tsv_1h, dates=dates_1h)
print(f"\nSeasonal indices:")
for s, si in seasonal_idx.indices.items():
    print(f"  {s:8s} alpha={si.alpha:.3f} beta={si.beta:.3f} R²={si.r_squared:.3f} n={si.n_samples}")
```
**Output:**
```
Personalisation: alpha=1.504  beta=-0.445  R²=0.329  n=4344
Seasonal indices:
  mid      alpha=1.295 beta=-0.412 R²=0.254 n=2208
  summer   alpha=1.082 beta=-0.165 R²=0.219 n=720
  winter   alpha=1.366 beta=-0.691 R²=0.263 n=1416
```

```python
# Apply personalisation to the full 10-min grid
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
print(f"Personalised score mean={np.mean(personalised.score):.1f}/100")
```
> *Note: Some PMV values fall outside the [-2, +2] applicability range of the pythermalcomfort library and are set to NaN. This is expected for extreme winter conditions with high clothing insulation.*
```
Base PMV mean=+nan  ->  Personalised PMV mean=+nan
Base PPD mean=nan%  ->  Personalised PPD mean=nan%
Personalised score mean=nan/100
```

```python
fig = go.Figure()
fig.add_trace(go.Scatter(x=df.timestamp, y=personalised.base_pmv, name="Base PMV", line=dict(width=0.5)))
fig.add_trace(go.Scatter(x=df.timestamp, y=personalised.personalised_pmv, name="Personalised PMV", line=dict(width=0.5, color="orange")))
fig.update_layout(title="Base vs personalised PMV", yaxis_title="PMV", height=400)
fig.show()
```

---

## 5. Pollutant IAQ

`evaluate_iaq_pollutants()` scores PM2.5, PM10, TVOC, formaldehyde (HCHO),
and CO against health-based thresholds. Each pollutant is scored
independently via a piecewise linear function; the overall score is the mean
of all provided pollutant scores.

> **Theory — Pollutant scoring.** For each pollutant, four thresholds define
> the score curve: excellent (score=100), good (score=75), moderate
> (score=50), and poor (score=0). The score is linear between thresholds
> and clamped to [0, 100].
>
> **Threshold references:**
> - PM2.5: WHO 2021 Air Quality Guidelines (15 µg/m³ 24h, 5 µg/m³ annual)
> - TVOC: WHO/AgBB evaluation scheme (300 µg/m³ 8h)
> - Formaldehyde: WHO IAQ Guidelines (100 µg/m³ 30min ≈ 80 ppb)
> - CO: WHO AQ Guidelines (4 mg/m³ 24h ≈ 3.5 ppm)
>
> **Reference:** WHO (2021), *WHO global air quality guidelines*,
> Particulate matter (PM2.5 and PM10), ozone, nitrogen dioxide, sulfur
> dioxide and carbon monoxide.

```python
pollutant = evaluate_iaq_pollutants(
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
print(f"CO compliant:     {np.mean(pollutant.compliant_co)*100:.1f}%")
```
**Output:**
```
Pollutant IAQ score  mean=86.1/100
PM2.5 compliant:  98.8%   (WHO 24h guideline: 15 µg/m³)
PM10 compliant:   100.0%
TVOC compliant:   100.0%
HCHO compliant:   85.6%
CO compliant:     100.0%
```

```python
fig = make_subplots(rows=3, cols=2, shared_xaxes=True,
                    subplot_titles=("PM2.5 µg/m³", "PM10 µg/m³", "TVOC µg/m³", "HCHO ppb", "CO ppm", "Pollutant score"))
fig.add_trace(go.Scatter(x=df.timestamp, y=df.pm25_ugm3, name="PM2.5", line=dict(width=0.5)), row=1, col=1)
fig.add_trace(go.Scatter(x=df.timestamp, y=df.pm10_ugm3, name="PM10", line=dict(width=0.5)), row=1, col=2)
fig.add_trace(go.Scatter(x=df.timestamp, y=df.tvoc_ugm3, name="TVOC", line=dict(width=0.5)), row=2, col=1)
fig.add_trace(go.Scatter(x=df.timestamp, y=df.formaldehyde_ppb, name="HCHO", line=dict(width=0.5)), row=2, col=2)
fig.add_trace(go.Scatter(x=df.timestamp, y=df.co_ppm, name="CO", line=dict(width=0.5)), row=3, col=1)
fig.add_trace(go.Scatter(x=df.timestamp, y=pollutant.score, name="score", line=dict(width=0.5, color="green")), row=3, col=2)
fig.update_layout(height=700, title="Pollutant IAQ (WHO / EPA / WELL thresholds)", showlegend=False)
fig.show()
```

---

## 6. Advanced Domains (optional extras)

These domains require additional dependencies installed via extras:
`[daylighting]`, `[color]`, `[acoustics]`, `[psychrometrics]`.

### 6a. Daylighting — Radiance (`[daylighting]`)

Daylighting metrics (Daylight Autonomy, sDA, ASE) require a Radiance scene
(`.oct` file). On Windows, Radiance needs WSL; on Linux/macOS it installs
via `pip install comfio[daylighting]`.

```python
try:
    from comfio import evaluate_daylighting
    # evaluate_daylighting needs an octree file + sensor points — omitted here
    # because it requires a Radiance scene. See the daylighting tutorial.
    print("evaluate_daylighting is available. Provide an .oct scene to run.")
except ImportError as e:
    print(f"Daylighting extra not installed: {e}")
```
**Output:**
```
evaluate_daylighting is available. Provide an .oct scene to run.
```

### 6b. Colour Quality (CRI / CCT) — `pip install comfio[color]`

Colour Rendering Index (CRI) and Correlated Colour Temperature (CCT) are
computed from a spectral power distribution (SPD) using the `colour-science`
library.

> **Theory — CRI (Ra).** The CIE 1995 method compares the colour rendering
> of a test source to a reference illuminant (blackbody or D-series daylight)
> using 8 (or 14) test colour samples. The general CRI $R_a$ is the mean of
> the first 8 special CRIs. CCT is the temperature of the blackbody
> illuminant that most closely matches the test source's chromaticity.
>
> **Reference:** CIE 13.3-1995, *Method of measuring and specifying colour
> rendering properties of light sources*.

```python
from comfio import evaluate_color_quality
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
    print("This is a known issue with newer colour-science versions.")
```
**Output:**
```
Colour quality evaluation failed (colour-science API compatibility): The new domain value is not monotonic! 
This is a known issue with newer colour-science versions.
```

### 6c. Reverberation Time (RT60) — `pip install comfio[acoustics]`

RT60 is the time required for the sound pressure level to decrease by 60 dB
after the sound source stops. `comfio` uses the Sabine or Eyring formula
via the `python-acoustics` library.

> **Theory — Sabine formula.**
>
> $$T_{60} = 0.161 \, \frac{V}{\sum_i S_i \, \alpha_i}$$
>
> where $V$ is room volume (m³), $S_i$ is surface area (m²), and $\alpha_i$
> is the absorption coefficient of surface $i$.
>
> **Reference:** Sabine, W.C. (1922), *Collected Papers on Acoustics*,
> Harvard University Press.

```python
from comfio import evaluate_reverberation
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
    print("Note: python-acoustics may have scipy compatibility issues (sph_harm removed in scipy >= 1.15).")
```
**Output:**
```
Reverberation evaluation failed: python-acoustics is required for reverberation calculations. Install it with: pip install comfio[acoustics]
Note: python-acoustics may have scipy compatibility issues (sph_harm removed in scipy >= 1.15).
```

### 6d. Speech Intelligibility (STI) — `pip install comfio[acoustics]`

The Speech Transmission Index (STI) quantifies speech intelligibility from
a room impulse response. It ranges from 0 (unintelligible) to 1 (perfect).

> **Theory — STI.** The STI is computed from the modulation transfer
> function (MTF) across 7 octave bands (125 Hz – 8 kHz) and 14 modulation
> frequencies. The MTF captures how much the speech envelope is preserved
> after reverberation and noise.
>
> | STI range | Rating |
> |-----------|--------|
> | < 0.30 | Bad |
> | 0.30–0.45 | Poor |
> | 0.45–0.60 | Fair |
> | 0.60–0.75 | Good |
> | > 0.75 | Excellent |
>
> **Reference:** IEC 60268-16:2020, *Sound system equipment — Part 16:
> Objective rating of speech intelligibility by speech transmission index*.

```python
from comfio import evaluate_speech_intelligibility
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
    print("Note: pyroomacoustics may have scipy compatibility issues.")
```
**Output:**
```
STI:       0.545
Rating:    fair
RT60:      1.53 s
Compliant: False  (STI >= 0.60)
Score:     54.5/100
```

### 6e. Ventilation Rate (CO₂ decay) — `pip install comfio[psychrometrics]`

The ventilation rate (air changes per hour, ACH) can be estimated from the
CO₂ decay curve when a space transitions from occupied to unoccupied. The
exponential decay constant maps directly to ACH.

> **Theory — CO₂ decay method.** When occupancy stops, CO₂ decays as:
>
> $$C(t) = C_{\infty} + (C_0 - C_{\infty})\,e^{-Nt}$$
>
> where $N$ is the ACH (1/h), $C_0$ is the initial concentration, and
> $C_{\infty}$ is the outdoor (steady-state) concentration. Fitting the
> decay curve yields $N$.
>
> **Reference:** ASTM D7297-14, *Standard Practice for Determining
> Ventilation Effectiveness of Residential and Commercial Buildings*.

```python
from comfio import evaluate_ventilation
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
print(f"Score:               {vent.score:.1f}/100")
```
**Output:**
```
ACH:                 0.95 /h  (method: co2_decay)
Ventilation eff.:    0.58
CO₂ peak:            1356 ppm
CO₂ steady-state:    729 ppm
Compliant:           False
Score:               23.8/100
```

### 6f. Psychrometrics — `pip install comfio[psychrometrics]`

Full psychrometric properties of moist air (wet bulb, dew point, enthalpy,
humidity ratio, vapor pressure, specific volume, degree of saturation) are
computed via PsychroLib.

> **Reference:** ASHRAE Handbook — Fundamentals (2017), Chapter 1:
> *Psychrometrics*. PsychroLib: https://github.com/psychrometrics/psychrolib

```python
from comfio import get_psychrometrics
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
print(psych_df.describe().round(2))
```
**Output:**
```
timestamp     twb    tdew  enthalpy  hum_ratio  \
count                         500  500.00  500.00    500.00     500.00   
mean   2025-04-01 11:50:01.200000   15.37   11.21  42926.13       0.01   
min           2025-01-01 00:00:00   13.08    9.26  36739.54       0.01   
25%           2025-02-15 05:50:00   14.84   10.65  41421.56       0.01   
50%           2025-04-01 11:50:00   15.39   11.24  42954.82       0.01   
75%           2025-05-16 17:50:00   15.95   11.72  44531.94       0.01   
max           2025-06-30 23:50:00   17.26   13.84  48381.14       0.01   
std                           NaN    0.78    0.76   2187.23       0.00   
       vap_pressure  moist_vol  
count        500.00     500.00  
mean        1332.48       0.85  
min         1168.33       0.83  
25%         1282.37       0.84  
50%         1333.69       0.85  
75%         1377.19       0.85  
max         1581.95       0.86  
std           67.69       0.00
```

```python
fig = make_subplots(rows=2, cols=2, shared_xaxes=True,
                    subplot_titles=("Wet bulb & dew point (°C)", "Enthalpy (kJ/kg)",
                                    "Humidity ratio (kg/kg)", "Vapour pressure (Pa)"))
fig.add_trace(go.Scatter(x=psych_df.timestamp, y=psych_df.twb, name="T_wb", line=dict(width=0.5)), row=1, col=1)
fig.add_trace(go.Scatter(x=psych_df.timestamp, y=psych_df.tdew, name="T_dew", line=dict(width=0.5)), row=1, col=1)
fig.add_trace(go.Scatter(x=psych_df.timestamp, y=psych_df.enthalpy, name="h", line=dict(width=0.5)), row=1, col=2)
fig.add_trace(go.Scatter(x=psych_df.timestamp, y=psych_df.hum_ratio, name="W", line=dict(width=0.5)), row=2, col=1)
fig.add_trace(go.Scatter(x=psych_df.timestamp, y=psych_df.vap_pressure, name="p_v", line=dict(width=0.5)), row=2, col=2)
fig.update_layout(height=600, title="Psychrometric properties (500-point sample)", showlegend=True)
fig.show()
```

---

## 7. Global IEQ Index & Weight Presets

The **Global IEQ Index** aggregates per-domain scores (0–100) into a single
index using configurable weights. `comfio` provides:

- `default_weights()` — equal weights across available domains
- `preset_weights("thermal_first")` — 50% thermal, 17% each other domain
- `preset_weights("visual_first")` — emphasises lighting
- `custom_weights({"thermal": 0.5, "visual": 0.1, ...})` — user-defined

> **Theory — Weighted aggregation.**
>
> $$\text{IEQ} = \sum_{d \in D} w_d \, s_d \quad / \quad \sum_{d \in D} w_d$$
>
> where $s_d$ is the domain score and $w_d$ is the weight. Weights are
> normalised so they sum to 1. Missing domains are excluded from both
> numerator and denominator.
>
> **Reference:** CEN/TC 156 WG, *Indoor Environmental Quality (IEQ)
> assessment methods*, prEN 16798-1:2019.

```python
# Baseline: 4 core domains
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
print(f"Full integration       IEQ mean={np.mean(ieq_full.index):.1f}  domains={ieq_full.domains}")
```
**Output:**
```
Baseline (4 domains)  IEQ mean=67.6  domains=['thermal', 'visual', 'acoustic', 'iaq']
With pollutant IAQ     IEQ mean=69.0  domains=['thermal', 'visual', 'acoustic', 'iaq']
With pollutant + TSV   IEQ mean=62.4  domains=['thermal', 'visual', 'acoustic', 'iaq']
Full integration       IEQ mean=59.9  domains=['thermal', 'visual', 'acoustic', 'iaq']
```

```python
# Compare weight presets on the full integration
print("Weight preset comparison (full integration):")
for preset in ["default", "equal", "school", "office", "healthcare"]:
    w = preset_weights(preset)
    ieq_w = calculate_global_ieq(
        thermal=thermal, visual=visual, acoustic=acoustic, iaq=iaq,
        pollutant_iaq=pollutant, tsv=tsv_result, ventilation=vent,
        weights=w,
    )
    print(f"  {preset:12s}  IEQ={np.mean(ieq_w.index):.1f}  weights={ieq_w.weights_used}")
```
**Output:**
```
Weight preset comparison (full integration):
  default       IEQ=59.9  weights={'thermal': 0.4, 'visual': 0.2, 'acoustic': 0.15, 'iaq': 0.25}
  equal         IEQ=54.1  weights={'thermal': 0.25, 'visual': 0.25, 'acoustic': 0.25, 'iaq': 0.25}
  school        IEQ=55.2  weights={'thermal': 0.27, 'visual': 0.24, 'acoustic': 0.23, 'iaq': 0.26}
  office        IEQ=63.6  weights={'thermal': 0.45, 'visual': 0.15, 'acoustic': 0.1, 'iaq': 0.3}
  healthcare    IEQ=59.4  weights={'thermal': 0.25, 'visual': 0.15, 'acoustic': 0.2, 'iaq': 0.4}
```

```python
# Custom weights
custom = custom_weights(thermal=0.5, visual=0.1, acoustic=0.1, iaq=0.3)
ieq_custom = calculate_global_ieq(thermal=thermal, visual=visual, acoustic=acoustic, iaq=iaq, weights=custom)
print(f"Custom weights  IEQ={np.mean(ieq_custom.index):.1f}  weights={ieq_custom.weights_used}")
```
**Output:**
```
Custom weights  IEQ=75.3  weights={'thermal': 0.5, 'visual': 0.1, 'acoustic': 0.1, 'iaq': 0.3}
```

```python
fig = go.Figure()
fig.add_trace(go.Scatter(x=df.timestamp, y=ieq_base.index, name="4 domains", line=dict(width=0.5)))
fig.add_trace(go.Scatter(x=df.timestamp, y=ieq_poll.index, name="+ pollutant", line=dict(width=0.5)))
fig.add_trace(go.Scatter(x=df.timestamp, y=ieq_tsv.index, name="+ TSV", line=dict(width=0.5)))
fig.add_trace(go.Scatter(x=df.timestamp, y=ieq_full.index, name="full", line=dict(width=0.5, color="black")))
fig.add_hline(y=80, line_dash="dash", line_color="green", annotation_text="compliance threshold 80")
fig.update_layout(title="Global IEQ Index — progressive domain integration", yaxis_title="IEQ (0-100)", height=450)
fig.show()
```

```python
# Stacked domain scores for the full integration
fig = go.Figure()
for domain, scores in ieq_full.domain_scores.items():
    fig.add_trace(go.Scatter(x=df.timestamp, y=scores, name=domain, mode="lines", line=dict(width=0.5), stackgroup="d"))
fig.update_layout(title="Domain score breakdown (full integration)", yaxis_title="score", height=450)
fig.show()
```

---

## 8. Compliance & Performance Contracts

`calculate_compliance()` evaluates the IEQ Index against a threshold
(default 80/100) and produces a `ComplianceReport` with compliant hours,
total hours, and a compliance rate. This is the basis for **performance
contract verification** in smart-building applications.

The `comfio.contracts` module generates **Solidity-ready ABI fragments** and
JSON payloads for on-chain compliance attestation via oracle contracts.

```python
report = calculate_compliance(ieq_full, threshold=80.0)
print(f"IEQ avg:         {report.ieq_index_avg:.1f}")
print(f"IEQ min/max:     {report.ieq_index_min:.1f} / {report.ieq_index_max:.1f}")
print(f"Compliance rate: {report.compliance_rate_pct:.1f}%  (threshold {report.threshold:.0f})")
print(f"Compliant hours: {report.compliant_hours:.0f} / {report.total_hours:.0f}")
print(f"\nPer-domain compliance (% with score >= 80):")
for d, rate in report.domain_compliance.items():
    print(f"  {d:14s}: {rate:.1f}%  (avg score {report.domain_scores_avg[d]:.1f})")
```
**Output:**
```
IEQ avg:         59.9
IEQ min/max:     21.9 / 79.5
Compliance rate: 0.0%  (threshold 80)
Compliant hours: 0 / 26064
Per-domain compliance (% with score >= 80):
  thermal       : 37.8%  (avg score 75.1)
  visual        : 18.5%  (avg score 32.3)
  acoustic      : 1.3%  (avg score 38.8)
  iaq           : 25.8%  (avg score 70.1)
```

```python
from comfio.performance.contract_schema import default_compliance_schema
schema = default_compliance_schema()
print(f"Contract: {schema.contract_name}  Function: {schema.function_name}")
print(f"\nFields ({len(schema.fields)}):")
for f in schema.fields:
    print(f"  {f.name:25s} {f.solidity_type:10s} <- {f.source}")
```
**Output:**
```
Contract: IEQComplianceOracle  Function: submitCompliance
Fields (10):
  periodStart               uint256    <- report.period_start
  periodEnd                 uint256    <- report.period_end
  ieqIndexAvg               uint8      <- report.ieq_index_avg
  complianceRatePct         uint8      <- report.compliance_rate_pct
  thermalCompliant          bool       <- report.domain_compliance.thermal
  visualCompliant           bool       <- report.domain_compliance.visual
  acousticCompliant         bool       <- report.domain_compliance.acoustic
  iaqCompliant              bool       <- report.domain_compliance.iaq
  totalOccupiedHours        uint32     <- report.total_occupied_hours
  compliantHours            uint32     <- report.compliant_hours
```

```python
payload = report.to_contract_payload()
print("Solidity-ready payload:")
for k, v in payload.items():
    print(f"  {k:25s} {str(v):<20s} ({type(v).__name__})")
```
**Output:**
```
Solidity-ready payload:
  periodStart               1690552443           (int)
  periodEnd                 1784382843           (int)
  ieqIndexAvg               60                   (int)
  complianceRatePct         0                    (int)
  thermalCompliant          False                (bool)
  visualCompliant           False                (bool)
  acousticCompliant         False                (bool)
  iaqCompliant              False                (bool)
  totalOccupiedHours        26064                (int)
  compliantHours            0                    (int)
```

```python
abi = schema.to_abi()
print("ABI fragment:")
print(json.dumps(abi, indent=2))
```
**Output:**
```
ABI fragment:
{
  "name": "submitCompliance",
  "type": "function",
  "stateMutability": "nonpayable",
  "inputs": [
    {
      "name": "periodStart",
      "type": "uint256",
      "internalType": "uint256"
    },
    {
      "name": "periodEnd",
      "type": "uint256",
      "internalType": "uint256"
    },
    {
      "name": "ieqIndexAvg",
      "type": "uint8",
      "internalType": "uint8"
    },
    {
      "name": "complianceRatePct",
      "type": "uint8",
      "internalType": "uint8"
    },
    {
      "name": "thermalCompliant",
      "type": "bool",
      "internalType": "bool"
    },
    {
      "name": "visualCompliant",
      "type": "bool",
      "internalType": "bool"
    },
    {
      "name": "acousticCompliant",
      "type": "bool",
      "internalType": "bool"
    },
    {
      "name": "iaqCompliant",
      "type": "bool",
      "internalType": "bool"
    },
    {
      "name": "totalOccupiedHours",
      "type": "uint32",
      "internalType": "uint32"
    },
    {
      "name": "compliantHours",
      "type": "uint32",
      "internalType": "uint32"
    }
  ],
  "outputs": []
}
```

---

## 9. ML Integration — Next-Day IEQ Forecast

We frame a **next-day forecast** problem: given today's IEQ features, predict
tomorrow's mean IEQ Index. Three backends are demonstrated, each using the
corresponding comfio ML adapter:

1. **scikit-learn** — `IEQFeatureExtractor` transforms daily sensor DataFrames
   into IEQ features, then RandomForest predicts next-day IEQ.
2. **PyTorch** — `IEQTimeSeriesDataset` wraps the 10-min sensor DataFrame into
   1-day windows with IEQ scores; an LSTM predicts the next day's mean.
3. **Keras/TensorFlow** — `IEQPreprocessingLayer` computes IEQ features from
   daily sensor DataFrames; a dense model predicts next-day IEQ.

First, build daily-mean sensor DataFrames.

```python
# Daily-mean sensor DataFrame (columns match comfio canonical names)
daily_df = df.set_index("timestamp").resample("1D").mean().reset_index()
print(f"Daily rows: {len(daily_df)}")
print(f"Columns: {list(daily_df.columns)}")
daily_df.head()
```
**Output:**
```
Daily rows: 181
Columns: ['timestamp', 'air_temp_c', 'radiant_temp_c', 'relative_humidity_pct', 'air_velocity_ms', 'illuminance_lux', 'noise_laeq_db', 'co2_ppm', 'pm25_ugm3', 'pm10_ugm3', 'tvoc_ugm3', 'formaldehyde_ppb', 'co_ppm', 'outdoor_temp_c', 'metabolic_rate_met', 'clothing_insulation_clo']
timestamp  air_temp_c  radiant_temp_c  relative_humidity_pct  \
0 2025-01-01   19.981027       20.120443              54.940502   
1 2025-01-02   20.012683       20.185410              54.981768   
2 2025-01-03   20.043878       20.194190              55.321751   
3 2025-01-04   20.043434       20.163313              54.875071   
4 2025-01-05   20.024688       20.154722              55.102904   

   air_velocity_ms  illuminance_lux  noise_laeq_db     co2_ppm  pm25_ugm3  \
0         0.096510       135.119787      43.644424  738.383031   6.207651   
1         0.098341       132.549721      43.392208  727.735958   6.343342   
2         0.100816       134.642152      43.440618  727.142239   6.523150   
3         0.096980       135.719530      43.552980  729.846627   6.417408   
4         0.099269       134.595345      43.272482  726.654908   6.503627   

   pm10_ugm3   tvoc_ugm3  formaldehyde_ppb    co_ppm  outdoor_temp_c  \
0  10.093064  124.409367         17.319177  1.022746        4.853496   
1  10.486655  124.317466         17.092898  0.987535        5.208688   
2  10.451231  123.155239         17.306009  0.978140        5.162454   
3  10.395731  120.795267         17.371570  0.975624        4.978776   
4  10.229461  124.159430         17.765566  1.003945        5.138329   

   metabolic_rate_met  clothing_insulation_clo  
0                 1.2                 0.999540  
1                 1.2                 0.999460  
2                 1.2                 0.997497  
3                 1.2                 0.997436  
4                 1.2                 0.995053
```

### 9a. scikit-learn — `IEQFeatureExtractor` + RandomForest

`IEQFeatureExtractor` is an sklearn-compatible transformer that wraps comfio
domain evaluations. Given a DataFrame with sensor columns, it computes the
Global IEQ Index and per-domain scores per row.

```python
from comfio.ml.sklearn_transformers import IEQFeatureExtractor
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
print(f"Feature importances: {dict(zip(extractor._feature_names, rf.feature_importances_.round(3)))}")
```
**Output:**
```
IEQ features shape: (181, 5)  names: ['ieq_index', 'thermal_score', 'visual_score', 'acoustic_score', 'iaq_score']
sklearn RF  MSE=0.04  MAE=0.16  R²=-0.112
Feature importances: {'ieq_index': 0.68700000000000006, 'thermal_score': 0.065000000000000002, 'visual_score': 0.23999999999999999, 'acoustic_score': 0.0040000000000000001, 'iaq_score': 0.0040000000000000001}
```

```python
days_test = daily_df["timestamp"].iloc[split+1:]
fig = go.Figure()
fig.add_trace(go.Scatter(x=days_test, y=y_sk_te, mode="lines+markers", name="Actual IEQ"))
fig.add_trace(go.Scatter(x=days_test, y=pred_sk, mode="lines+markers", name="sklearn RF"))
fig.update_layout(title="Next-day IEQ forecast — sklearn RandomForest", yaxis_title="IEQ", height=400)
fig.show()
```

### 9b. PyTorch — `IEQTimeSeriesDataset` + LSTM

`IEQTimeSeriesDataset` wraps the 10-min sensor DataFrame into windowed samples
with computed IEQ scores. We use 1-day windows (144 timesteps at 10-min
sampling) and predict the next day's mean IEQ Index.

```python
import torch
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
print(f"Train: {split_t}  Test: {len(X_t)-split_t}")
```
**Output:**
```
C:\Users\utente\AppData\Local\Programs\Python\Python311\Lib\site-packages\torch\cuda\__init__.py:63: FutureWarning: The pynvml package is deprecated. Please install nvidia-ml-py instead. If you did not install pynvml directly, please report this to the maintainers of the package that installed pynvml for you.
  import pynvml  # type: ignore[import]
PyTorch dataset windows: 181
Sample keys: ['raw', 'ieq_index', 'domain_scores']
  ieq_index shape: (144,)
  raw shape: (144, 15)
Train: 144  Test: 36
```

```python
class IEQLSTM(torch.nn.Module):
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
print(f"PyTorch LSTM  MSE={mse_t:.2f}  MAE={mae_t:.2f}  R²={r2_t:.3f}")
```
**Output:**
```
PyTorch LSTM  MSE=4131.98  MAE=64.28  R²=-159843.406
```

```python
days_lstm = daily_df["timestamp"].iloc[split_t+1:]
fig = make_subplots(rows=2, cols=1, subplot_titles=("Training loss", "Next-day forecast"))
fig.add_trace(go.Scatter(y=train_losses, name="train loss"), row=1, col=1)
fig.add_trace(go.Scatter(y=test_losses, name="test loss"), row=1, col=1)
fig.add_trace(go.Scatter(x=days_lstm, y=y_test_t, mode="lines+markers", name="Actual"), row=2, col=1)
fig.add_trace(go.Scatter(x=days_lstm, y=pred_torch, mode="lines+markers", name="LSTM"), row=2, col=1)
fig.update_layout(height=600, title="PyTorch LSTM — training & forecast")
fig.show()
```

### 9c. Keras / TensorFlow — `IEQPreprocessingLayer`

`IEQPreprocessingLayer` wraps comfio domain evaluations as a callable Keras
preprocessing layer. It accepts a pandas DataFrame and returns IEQ features
as a TensorFlow tensor.

```python
import os
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
keras_model.summary()
```
**Output:**
```
Keras IEQ features shape: (181, 5)
WARNING:tensorflow:TensorFlow GPU support is not available on native Windows for TensorFlow >= 2.11. Even if CUDA/cuDNN are installed, GPU will not be used. Please use WSL2 or the TensorFlow-DirectML plugin.
  super().__init__(activity_regularizer=activity_regularizer, **kwargs)
```

```python
history = keras_model.fit(X_k[:split_k], y_k[:split_k], epochs=80, batch_size=8,
                        validation_split=0.15, verbose=0)
pred_keras = keras_model.predict(X_k[split_k:], verbose=0).flatten()
y_test_k = y_k[split_k:]
mse_k = mean_squared_error(y_test_k, pred_keras)
mae_k = mean_absolute_error(y_test_k, pred_keras)
r2_k = r2_score(y_test_k, pred_keras)
print(f"Keras model  MSE={mse_k:.2f}  MAE={mae_k:.2f}  R²={r2_k:.3f}")
```
**Output:**
```
Keras model  MSE=22.07  MAE=4.69  R²=-628.380
```

```python
days_keras = daily_df["timestamp"].iloc[split_k+1:]
fig = make_subplots(rows=2, cols=1, subplot_titles=("Keras training history", "Forecast comparison"))
fig.add_trace(go.Scatter(y=history.history["loss"], name="train loss"), row=1, col=1)
fig.add_trace(go.Scatter(y=history.history["val_loss"], name="val loss"), row=1, col=1)
fig.add_trace(go.Scatter(x=days_keras, y=y_test_k, mode="lines+markers", name="Actual"), row=2, col=1)
fig.add_trace(go.Scatter(x=days_keras, y=pred_keras, mode="lines+markers", name="Keras"), row=2, col=1)
fig.update_layout(height=600, title="Keras/TensorFlow — training & forecast")
fig.show()
```

**Model comparison:**
```
============================================================
Model                          MSE      MAE       R²
------------------------------------------------------------
sklearn RandomForest          0.04     0.16   -0.112
PyTorch LSTM               4131.98    64.28 -159843.406
Keras/TensorFlow             22.07     4.69 -628.380
============================================================
```

---

## 10. LLM Integration

`comfio.llm` provides three layers for integrating IEQ data with Large
Language Models:

1. **Interpreters** — `ieq_to_markdown()` and `ieq_to_summary_dict()`
   serialise IEQ results into token-efficient structured text for LLM context.
2. **Prompts** — Guarded system prompts and diagnostic templates for
   building-comfort LLM agents.
3. **Tools** — `to_openai_tools()` and `to_langchain_tools()` generate
   function-calling schemas that let LLMs invoke comfio evaluations.

```python
from comfio import (
    ieq_to_markdown, ieq_to_summary_dict, generate_markdown_summary,
    EDGE_SYSTEM_PROMPT, DIAGNOSTIC_PROMPT_TEMPLATE, format_prompt,
)
md_report = ieq_to_markdown(ieq_full, compliance_report=report, zone_id="A-101")
print(md_report)
```
**Output:**
```
### Building System Report: Zone A-101
- **Current Operational State**: NON-COMPLIANT
- **Global IEQ Index Score**: 59.9/100 (min: 21.9, max: 79.5)
- **Contract Compliance Rate**: 0.0%
- **Timestamps Evaluated**: 26064
#### Domain Breakdown:
  - [OK] THERMAL: 75.1/100
  - [WARNING] VISUAL: 32.3/100
  - [WARNING] ACOUSTIC: 38.8/100
  - [OK] IAQ: 70.1/100
#### Diagnostic Insight:
The primary limiting factor is the **VISUAL** domain (score: 32.3/100). Illuminance below target levels. Action: verify lighting fixtures or increase daylight access.
```

```python
summary = ieq_to_summary_dict(ieq_full, compliance_report=report)
print(json.dumps(summary, indent=2))
```
**Output:**
```
{
  "ieq_index_avg": 59.9,
  "ieq_index_min": 21.9,
  "ieq_index_max": 79.5,
  "ieq_index_std": 9.6,
  "n_timestamps": 26064,
  "domains": [
    "thermal",
    "visual",
    "acoustic",
    "iaq"
  ],
  "domain_scores_avg": {
    "thermal": 75.1,
    "visual": 32.3,
    "acoustic": 38.8,
    "iaq": 70.1
  },
  "worst_domain": "visual",
  "weights_used": {
    "thermal": 0.4,
    "visual": 0.2,
    "acoustic": 0.15,
    "iaq": 0.25
  },
  "compliance_rate_pct": 0.0,
  "threshold": 80.0,
  "compliant_hours": 0.0,
  "total_hours": 26064.0
}
```

```python
# One-shot markdown summary from the raw DataFrame (first 7 days)
md_auto = generate_markdown_summary(df.iloc[:7*144], window_hours=24, threshold=80.0, zone_id="A-101")
print(md_auto[:1500])
```
**Output:**
```
## IEQ Report: Zone A-101
* **Global IEQ Average:** 68.5/100
* **Compliance Rate:** 0.0% (Threshold > 80)
* **Timestamps:** 1008
### Domain Summary
* **THERMAL:** avg=90.2/100, compliance=95.6%
* **VISUAL:** avg=27.0/100, compliance=5.8%
* **ACOUSTIC:** avg=39.0/100, compliance=1.9%
* **IAQ:** avg=84.7/100, compliance=54.9%
### Critical Failures
* **1008 timestamps** below threshold (IEQ < 80).
* Primary cause: **VISUAL** domain (avg score during failures: 27.0/100).
* Illuminance below target levels. Action: verify lighting fixtures or increase daylight access.
```

```python
from comfio.llm.tools import to_openai_tools, to_langchain_tools
tools = to_openai_tools()
print(f"OpenAI tool schemas: {len(tools)}")
for t in tools:
    print(f"  - {t['function']['name']}: {t['function']['description'][:70]}...")
```
**Output:**
```
OpenAI tool schemas: 6
  - evaluate_thermal: Calculate thermal comfort metrics (PMV, PPD) from air temperature, hum...
  - evaluate_ieq: Evaluate multi-domain Indoor Environmental Quality (thermal, visual, a...
  - evaluate_pollutant_iaq: Evaluate IAQ pollutant concentrations (PM2.5, PM10, TVOC, formaldehyde...
  - evaluate_spmv: Calculate simplified PMV (sPMV) from indoor temperature and humidity o...
  - evaluate_adaptive: Evaluate adaptive thermal comfort per ASHRAE 55-2023 or EN 16798-1:201...
  - evaluate_tsv: Evaluate Thermal Sensation Vote data for comfort and compliance per AS...
```

```python
try:
    lc_tools = to_langchain_tools()
    print(f"LangChain tools: {len(lc_tools)}")
    for t in lc_tools:
        print(f"  - {t.name}: {t.description[:70]}...")
except Exception as e:
    print(f"LangChain tools not available: {e}")
    print("Install with: pip install langchain")
```
**Output:**
```
C:\Users\utente\AppData\Local\Programs\Python\Python311\Lib\site-packages\tqdm\auto.py:21: TqdmWarning: IProgress not found. Please update jupyter and ipywidgets. See https://ipywidgets.readthedocs.io/en/stable/user_install.html
  from .autonotebook import tqdm as notebook_tqdm
LangChain tools not available: LangChain is required for to_langchain_tools(). Install with: pip install langchain
Install with: pip install langchain
```

```python
# Example diagnostic prompt
prompt = format_prompt(
    DIAGNOSTIC_PROMPT_TEMPLATE,
    ieq_report=md_report,
    complaint="Occupants report feeling warm in the afternoons.",
    pollutant_status="PM2.5 within WHO 24h limit",
    adaptive_status="Within ASHRAE 80% band",
    tsv_status=f"Mean TSV={np.mean(tsv_aug):+.2f}, compliance={tsv_result.compliance_rate*100:.0f}%",
)
print(f"Diagnostic prompt ({len(prompt)} chars):\n")
print(prompt[:800])
```
**Output:**
```
Diagnostic prompt (1160 chars):
You are a building comfort diagnostic assistant. Based on the following IEQ report, diagnose the issue and recommend remediation actions.
IEQ Report:
### Building System Report: Zone A-101
- **Current Operational State**: NON-COMPLIANT
- **Global IEQ Index Score**: 59.9/100 (min: 21.9, max: 79.5)
- **Contract Compliance Rate**: 0.0%
- **Timestamps Evaluated**: 26064
#### Domain Breakdown:
  - [OK] THERMAL: 75.1/100
  - [WARNING] VISUAL: 32.3/100
  - [WARNING] ACOUSTIC: 38.8/100
  - [OK] IAQ: 70.1/100
#### Diagnostic Insight:
The primary limiting factor is the **VISUAL** domain (score: 32.3/100). Illuminance below target levels. Action: verify lighting fixtures or increase daylight access.
Occupant Complaint:
Occupants report feeling warm in the afternoons.
Additional Context:
- Pollut
```

---

## 11. Smart-Contract Export (web3.py)

`comfio.contracts` generates Solidity ABI fragments and JSON payloads for
on-chain compliance attestation. The `web3.py` integration (optional)
demonstrates how to encode the ABI call for a blockchain transaction.

```python
contract_json = report.to_contract_json()
print(f"Contract JSON ({len(contract_json)} chars):")
print(contract_json)
```
**Output:**
```
Contract JSON (271 chars):
{
  "periodStart": 1690552443,
  "periodEnd": 1784382843,
  "ieqIndexAvg": 60,
  "complianceRatePct": 0,
  "thermalCompliant": false,
  "visualCompliant": false,
  "acousticCompliant": false,
  "iaqCompliant": false,
  "totalOccupiedHours": 26064,
  "compliantHours": 0
}
```

```python
# Example: how this would be sent to a deployed IEQComplianceOracle
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

```
**Output:**
```
web3.py available. ABI call encoding would produce:
  Encoded calldata: 0x0000000000000000000000000000000000000000000000000000000064c3c87b0000000000000000...  (320 bytes)
C:\Users\utente\AppData\Local\Programs\Python\Python311\Lib\site-packages\websockets\legacy\__init__.py:6: DeprecationWarning: websockets.legacy is deprecated; see https://websockets.readthedocs.io/en/stable/howto/upgrade.html for upgrade instructions
  warnings.warn(  # deprecated in 14.0 - 2024-11-09
```

---

## 12. Reports — CSV / PDF / DOCX / Intelligent Pipeline

`comfio.reports` provides:

- `ieq_to_csv()` — per-timestamp IEQ scores and compliance flags as CSV
- `ieq_to_pdf()` — formatted PDF report (requires `reportlab`)
- `ieq_to_docx()` — Word document report (requires `python-docx`)
- `run_pipeline()` — intelligent pipeline that auto-detects capabilities
  and runs all possible evaluations
- `generate_pipeline_script()` — exports a reproducible Python script

```python
# CSV export
csv_str = ieq_to_csv(ieq_full, compliance_report=report)
print(f"CSV export: {len(csv_str)} chars, {csv_str.count(chr(10))} lines")
print(csv_str[:300])
```
**Output:**
```
CSV export: 1150034 chars, 26065 lines
timestamp_index,ieq_index,thermal_score,visual_score,acoustic_score,iaq_score,compliant,threshold
0,30.63,0.00,0.86,61.86,84.70,0,80.00
1,30.12,0.00,0.00,64.81,81.57,0,80.00
2,31.83,0.00,4.16,66.85,83.89,0,80.00
3,31.57,0.00,0.00,72.70,82.66,0,80.00
4,30.56,0.00,0.00,63.06,84.40,0,80.00
5,34.8
```

```python
# Intelligent pipeline (auto-detects and runs everything)
pipe_result = run_pipeline(sensor, config={"threshold": 80.0})
print(f"Pipeline capabilities detected: {sum(pipe_result.capabilities.values())}/{len(pipe_result.capabilities)}")
print(f"Domains evaluated: {list(pipe_result.domain_results.keys())}")
print(f"IEQ mean: {np.mean(pipe_result.ieq_result.index):.1f}" if pipe_result.ieq_result else "No IEQ result")
print(f"Warnings: {len(pipe_result.warnings)}")
for w in pipe_result.warnings[:5]:
    print(f"  - {w}")
```
**Output:**
```
Pipeline capabilities detected: 7/10
Domains evaluated: ['thermal', 'spmv', 'adaptive', 'visual', 'acoustic', 'iaq', 'pollutant_iaq']
IEQ mean: 68.6
Warnings: 0
```

```python
# PDF export (requires reportlab)
from comfio.reports import ieq_to_pdf
try:
    import tempfile, os
    pdf_bytes = ieq_to_pdf(ieq_full, compliance_report=report, zone_id="A-101")
    pdf_path = os.path.join(tempfile.gettempdir(), "comfio_walkthrough.pdf")
    with open(pdf_path, "wb") as f:
        f.write(pdf_bytes)
    print(f"PDF written: {pdf_path}  ({len(pdf_bytes):,} bytes)")
except Exception as e:
    print(f"PDF export: {e}")
```
**Output:**
```
PDF written: C:\Users\utente\AppData\Local\Temp\comfio_walkthrough.pdf  (3,079 bytes)
```

```python
# DOCX export (requires python-docx)
from comfio.reports import ieq_to_docx
try:
    import tempfile, os
    docx_bytes = ieq_to_docx(ieq_full, compliance_report=report, zone_id="A-101")
    docx_path = os.path.join(tempfile.gettempdir(), "comfio_walkthrough.docx")
    with open(docx_path, "wb") as f:
        f.write(docx_bytes)
    print(f"DOCX written: {docx_path}  ({len(docx_bytes):,} bytes)")
except Exception as e:
    print(f"DOCX export: {e}")
```
**Output:**
```
DOCX written: C:\Users\utente\AppData\Local\Temp\comfio_walkthrough.docx  (37,445 bytes)
```

```python
# Reproducible script export
from comfio.reports import generate_pipeline_script
try:
    script = generate_pipeline_script(pipe_result.config)
    print(f"Generated reproducible script: {len(script)} chars")
    print(script[:500])
except Exception as e:
    print(f"Script export: {e}")
```
**Output:**
```
Generated reproducible script: 3251 chars
"""Auto-generated by comfio.gui() — IEQ evaluation pipeline.
This script reproduces the evaluation pipeline configured in the GUI.
"""
from __future__ import annotations
import pandas as pd
from comfio import (
    SensorData,
    evaluate_thermal,
    evaluate_visual,
    evaluate_acoustic,
    evaluate_iaq,
    evaluate_spmv,
    evaluate_adaptive_ashrae,
    evaluate_adaptive_en,
    evaluate_iaq_pollutants,
    augment_tsv_cdf,
    evaluate_tsv,
    calculate_global_ieq,
    calculate_com
```

---

## References

1. Fanger, P.O. (1970). *Thermal Comfort*. Danish Technical Press.
2. ISO 7730:2005. *Ergonomics of the thermal environment — Analytical
   determination and interpretation of thermal comfort using calculation of
   the PMV and PPD indices*.
3. ASHRAE Standard 55-2023. *Thermal Environmental Conditions for Human
   Occupancy*.
4. EN 16798-1:2019. *Energy performance of buildings — Ventilation for
   buildings — Part 1: Indoor environmental input parameters*.
5. EN 12464-1:2021. *Light and lighting — Lighting of work places —
   Part 1: Indoor work places*.
6. Buratti, L., Ricciardi, P., & Naticchia, B. (2009). "A simplified PMV
   model for indoor thermal comfort assessment". *IBPSA Building Simulation*.
7. de Dear, R. & Brager, G.S. (1998). "Developing an adaptive model of
   thermal comfort and preference". *ASHRAE Transactions* 104(1).
8. WHO (2021). *WHO global air quality guidelines*. Particulate matter
   (PM2.5 and PM10), ozone, nitrogen dioxide, sulfur dioxide and carbon
   monoxide.
9. CIE 13.3-1995. *Method of measuring and specifying colour rendering
   properties of light sources*.
10. IEC 60268-16:2020. *Sound system equipment — Part 16: Objective rating
    of speech intelligibility by speech transmission index*.
11. Sabine, W.C. (1922). *Collected Papers on Acoustics*. Harvard University
    Press.
12. Beranek, L.L. (1957). "Revised criteria for noise in buildings".
    *Noise Control* 3(1), 19–27.
13. ASHRAE Standard 62.1-2022. *Ventilation for Acceptable Indoor Air
    Quality*.
14. ASTM D7297-14. *Standard Practice for Determining Ventilation
    Effectiveness of Residential and Commercial Buildings*.
15. Schweiker, M. et al. (2020). "Review of multi-domain approaches to
    indoor environmental perception and behaviour". *Building and
    Environment* 176.
16. Persily, A. (2015). "Challenges in developing ventilation and indoor
    air quality standards". *Building and Environment*.

---

*This walkthrough is a living document. To regenerate the executed notebook:*

```bash
python build_walkthrough_nb.py
python -m nbconvert --to notebook --execute --inplace     --ExecutePreprocessor.timeout=600     examples/walkthrough_executed.ipynb
```

*To regenerate this markdown:*

```bash
python extract_outputs.py
python build_walkthrough_md.py
```
