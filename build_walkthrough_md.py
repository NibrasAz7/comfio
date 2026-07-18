"""Build docs/tutorials/06_walkthrough.md from the executed notebook outputs.

Reads notebook_outputs.json (produced by extract_outputs.py) and constructs
a comprehensive markdown walkthrough with theory boxes, references, and
embedded executed outputs.
"""
from __future__ import annotations

import json
import textwrap
from pathlib import Path

with open("notebook_outputs.json", encoding="utf-8") as f:
    cells = json.load(f)

# Index cells by index for easy lookup
cell_map = {c["index"]: c for c in cells}


def get_stream_outputs(cell: dict) -> str:
    """Concatenate all stream outputs from a code cell, filtering warnings."""
    parts = []
    for o in cell.get("outputs", []):
        if o["type"] == "stream":
            text = o["text"]
            # Filter out pythermalcomfort warnings and tqdm progress bars
            lines = text.split("\n")
            filtered = [
                l for l in lines
                if "UserWarning" not in l
                and "pythermalcomfort" not in l
                and "valid_range" not in l
                and "pmv_valid" not in l
                and "outside the applicability" not in l
                and "it/s]" not in l
                and "it/s]" not in l
                and "s/it]" not in l
                and l.strip() != ""
            ]
            if filtered:
                parts.append("\n".join(filtered))
    return "\n".join(parts)


def get_result_output(cell: dict) -> str:
    """Get the execute_result text output (e.g. DataFrame repr)."""
    for o in cell.get("outputs", []):
        if o["type"] == "result":
            return o["text"]
    return ""


def format_output(cell: dict) -> str:
    """Format all text outputs as a markdown code block."""
    stream = get_stream_outputs(cell)
    result = get_result_output(cell)
    parts = []
    if stream.strip():
        parts.append(stream.strip())
    if result.strip():
        parts.append(result.strip())
    if not parts:
        return ""
    combined = "\n".join(parts)
    # Truncate very long outputs
    if len(combined) > 3000:
        combined = combined[:3000] + "\n... (truncated)"
    return f"```\n{combined}\n```"


def code_block(cell: dict) -> str:
    """Format the code source as a python code block."""
    return f"```python\n{cell['source']}\n```"


# ===========================================================================
# Build the markdown document
# ===========================================================================

md_parts = []

def w(s: str) -> None:
    md_parts.append(s)

# --- Title ---
w("""# comfio — Complete Walkthrough

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
""")

# Cell 1 — imports
w(code_block(cell_map[1]))
w("**Output:**")
w(format_output(cell_map[1]))
w("")

# --- Section 1: Data Generation ---
w("""---

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
""")

w(code_block(cell_map[3]))
w("**Output:**")
w(format_output(cell_map[3]))
w("")

w("""### 1b. Thermal Sensation Votes (TSV) — 1-hour sampling

TSV represents **occupant feedback** on the ASHRAE 7-point scale
(−3 cold … 0 neutral … +3 hot). We generate sparse 1-hour votes by
computing an approximate PMV from the 10-min data and adding noise, then
rounding to the nearest integer on the scale. The `comfio` TSV augmentation
module (§4c) will later upsample these to the 10-min grid.
""")

w(code_block(cell_map[4]))
w("**Output:**")
w(format_output(cell_map[4]))
w("")

# Cells 5, 6 — Plotly figures (code-only, no text output of interest)
w("""### 1c. Visualising the raw data

Two Plotly figures give an overview of the generated time series. In the
notebook these render interactively; here we show the code for reference.
""")

w(code_block(cell_map[5]))
w("")
w(code_block(cell_map[6]))
w("")

# --- Section 2: Data Ingestion ---
w("""---

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
""")

w(code_block(cell_map[8]))
w("**Output:**")
w(format_output(cell_map[8]))
w("")

w("""### 2b. Capability Detection

`detect_capabilities()` inspects the available columns and reports which
evaluations are possible. This is the same logic used by the intelligent
pipeline (`run_pipeline`) to gracefully skip domains when data is missing.
""")

w(code_block(cell_map[9]))
w("**Output:**")
w(format_output(cell_map[9]))
w("")

# --- Section 3: Core Domain Evaluations ---
w("""---

## 3. Core Domain Evaluations

### 3a. Thermal Comfort — Fanger PMV/PPD (ISO 7730 / ASHRAE 55)

The **Predicted Mean Vote (PMV)** model, developed by Fanger (1970), predicts
the mean thermal sensation vote of a large group of occupants on the ASHRAE
7-point scale. It is a function of four environmental variables
(air temperature, radiant temperature, air velocity, relative humidity) and
two personal variables (metabolic rate, clothing insulation).

> **Theory — PMV equation.**
>
> $$\\text{PMV} = (0.303\\,e^{-0.036M} + 0.028) \\, L$$
>
> where $L$ is the thermal load on the body (W/m²), $M$ is the metabolic rate
> (W/m²). The **Predicted Percentage Dissatisfied (PPD)** is derived from PMV:
>
> $$\\text{PPD} = 100 - 95\\,\\exp(-0.03353\\,\\text{PMV}^4 - 0.2179\\,\\text{PMV}^2)$$
>
> PMV = 0 corresponds to PPD = 5% (the theoretical minimum). ISO 7730
> Category B requires |PMV| ≤ 0.5 (PPD ≤ 10%).
>
> **Reference:** ISO 7730:2005, *Ergonomics of the thermal environment —
> Analytical determination and interpretation of thermal comfort using
> calculation of the PMV and PPD indices*. Fanger, P.O. (1970),
> *Thermal Comfort*, Danish Technical Press.
""")

w(code_block(cell_map[11]))
w("**Output:**")
w(format_output(cell_map[11]))
w("")

w(code_block(cell_map[12]))
w("")

w("""### 3b. Visual Comfort — EN 12464-1:2021

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
""")

w(code_block(cell_map[14]))
w("**Output:**")
w(format_output(cell_map[14]))
w("")

w(code_block(cell_map[15]))
w("")

w("""### 3c. Acoustic Comfort — Noise Criteria (NC)

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
""")

w(code_block(cell_map[17]))
w("**Output:**")
w(format_output(cell_map[17]))
w("")

w(code_block(cell_map[18]))
w("")

w("""### 3d. Indoor Air Quality — CO₂ (ASHRAE 62.1 indicators)

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
""")

w(code_block(cell_map[20]))
w("**Output:**")
w(format_output(cell_map[20]))
w("")

w(code_block(cell_map[21]))
w("")

# --- Section 4: Advanced Thermal Models ---
w("""---

## 4. Advanced Thermal Models

### 4a. Simplified PMV (sPMV) — Buratti Seasonal Model

The **simplified PMV** model by Buratti, Ricciardi & Naticchia (2009) uses
only indoor air temperature and relative humidity — no need for metabolic
rate, clothing, or air velocity inputs. Seasonal coefficients capture
typical occupancy conditions.

> **Theory — sPMV equation.**
>
> $$\\text{sPMV} = a \\, T + b \\, p_v - c$$
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
> The sPMV score is $100(1 - |\\text{sPMV}|/3)$, clamped to [0, 100].
>
> **Reference:** Buratti, L., Ricciardi, P., & Naticchia, B. (2009).
> "A simplified PMV model for indoor thermal comfort assessment",
> *Proceedings of the 11th International IBPSA Building Simulation
> Conference*.
""")

w(code_block(cell_map[23]))
w("**Output:**")
w(format_output(cell_map[23]))
w("")

w(code_block(cell_map[24]))
w("")

w("""### 4b. Adaptive Thermal Comfort — ASHRAE 55 & EN 16798-1

The **adaptive comfort model** recognises that occupants in naturally
ventilated buildings adapt to seasonal outdoor temperatures through
clothing, behavioural, and physiological adjustments. Instead of PMV, the
comfort temperature is a linear function of the prevailing mean outdoor
temperature.

> **Theory — ASHRAE 55-2023 (Appendix L).**
>
> $$T_{\\text{comf}} = 0.31 \\, \\bar{T}_{\\text{out}} + 17.8$$
>
> Valid for $10 \\le \\bar{T}_{\\text{out}} \\le 33.5$ °C. The 80% acceptability
> band is ±3.5 °C; the 90% band is ±2.5 °C.
>
> **EN 16798-1:2019 (Category II):**
>
> $$T_{\\text{comf}} = 0.33 \\, \\bar{T}_{\\text{rm}} + 18.8$$
>
> where $\\bar{T}_{\\text{rm}}$ is the running mean outdoor temperature.
> Category II band is ±3 °C.
>
> **References:**
> - ASHRAE Standard 55-2023, *Thermal Environmental Conditions for Human
>   Occupancy*, Appendix L.
> - EN 16798-1:2019, *Energy performance of buildings — Ventilation for
>   buildings — Part 1: Indoor environmental input parameters*.
> - de Dear, R. & Brager, G.S. (1998), "Developing an adaptive model of
>   thermal comfort and preference", *ASHRAE Transactions* 104(1).
""")

w(code_block(cell_map[26]))
w("**Output:**")
w(format_output(cell_map[26]))
w("")

w(code_block(cell_map[27]))
w("")

w("""### 4c. TSV Augmentation & Evaluation

Occupant TSV votes are typically sparse (1-hour) while sensor data is dense
(10-min). `comfio` provides `augment_tsv_cdf()` to upsample sparse votes to
the sensor grid using **CDF remapping**: the empirical CDF of the sparse
votes is mapped onto the target timestamps, preserving the distribution
while filling gaps.

> **Theory — CDF remapping.** Given $n$ sparse votes and $m > n$ target
> timestamps, each target is assigned a rank $r_j \\in [0, 1)$. The vote
> value at rank $r_j$ is the quantile of the sparse vote distribution at
> $r_j$. When `time_aware=True`, ranks are computed within daily time-of-day
> windows, preserving the diurnal pattern of occupant feedback.
>
> **Reference:** ASHRAE 55-2023, Appendix L — "Thermal comfort in
> naturally conditioned spaces" requires |TSV| ≤ 1.5 for 80% acceptability.
""")

w(code_block(cell_map[29]))
w("**Output:**")
w(format_output(cell_map[29]))
w("")

w(code_block(cell_map[30]))
w("")

w("""### 4d. Personalised Comfort (OLS Regression)

The personalisation module fits an **Ordinary Least Squares (OLS)**
regression mapping model-predicted PMV to occupant TSV:

> $$\\text{TSV} = \\alpha \\times \\text{PMV} + \\beta$$
>
> where $\\alpha$ captures the occupant's sensitivity to PMV changes and
> $\\beta$ is a systematic offset (e.g. preference for cooler/warmer
> conditions than the model predicts). Seasonal personalisation fits separate
> $(\\alpha, \\beta)$ per season.
>
> **Reference:** Schweiker, M. et al. (2020), "Review of multi-domain
> approaches to indoor environmental perception and behaviour",
> *Building and Environment* 176.
""")

w(code_block(cell_map[32]))
w("**Output:**")
w(format_output(cell_map[32]))
w("")

w(code_block(cell_map[33]))
w("> *Note: Some PMV values fall outside the [-2, +2] applicability range of the pythermalcomfort library and are set to NaN. This is expected for extreme winter conditions with high clothing insulation.*")
w(format_output(cell_map[33]))
w("")

w(code_block(cell_map[34]))
w("")

# --- Section 5: Pollutant IAQ ---
w("""---

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
""")

w(code_block(cell_map[36]))
w("**Output:**")
w(format_output(cell_map[36]))
w("")

w(code_block(cell_map[37]))
w("")

# --- Section 6: Advanced Domains ---
w("""---

## 6. Advanced Domains (optional extras)

These domains require additional dependencies installed via extras:
`[daylighting]`, `[color]`, `[acoustics]`, `[psychrometrics]`.

### 6a. Daylighting — Radiance (`[daylighting]`)

Daylighting metrics (Daylight Autonomy, sDA, ASE) require a Radiance scene
(`.oct` file). On Windows, Radiance needs WSL; on Linux/macOS it installs
via `pip install comfio[daylighting]`.
""")

w(code_block(cell_map[39]))
w("**Output:**")
w(format_output(cell_map[39]))
w("")

w("""### 6b. Colour Quality (CRI / CCT) — `pip install comfio[color]`

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
""")

w(code_block(cell_map[41]))
w("**Output:**")
w(format_output(cell_map[41]))
w("")

w("""### 6c. Reverberation Time (RT60) — `pip install comfio[acoustics]`

RT60 is the time required for the sound pressure level to decrease by 60 dB
after the sound source stops. `comfio` uses the Sabine or Eyring formula
via the `python-acoustics` library.

> **Theory — Sabine formula.**
>
> $$T_{60} = 0.161 \\, \\frac{V}{\\sum_i S_i \\, \\alpha_i}$$
>
> where $V$ is room volume (m³), $S_i$ is surface area (m²), and $\\alpha_i$
> is the absorption coefficient of surface $i$.
>
> **Reference:** Sabine, W.C. (1922), *Collected Papers on Acoustics*,
> Harvard University Press.
""")

w(code_block(cell_map[43]))
w("**Output:**")
w(format_output(cell_map[43]))
w("")

w("""### 6d. Speech Intelligibility (STI) — `pip install comfio[acoustics]`

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
""")

w(code_block(cell_map[45]))
w("**Output:**")
w(format_output(cell_map[45]))
w("")

w("""### 6e. Ventilation Rate (CO₂ decay) — `pip install comfio[psychrometrics]`

The ventilation rate (air changes per hour, ACH) can be estimated from the
CO₂ decay curve when a space transitions from occupied to unoccupied. The
exponential decay constant maps directly to ACH.

> **Theory — CO₂ decay method.** When occupancy stops, CO₂ decays as:
>
> $$C(t) = C_{\\infty} + (C_0 - C_{\\infty})\\,e^{-Nt}$$
>
> where $N$ is the ACH (1/h), $C_0$ is the initial concentration, and
> $C_{\\infty}$ is the outdoor (steady-state) concentration. Fitting the
> decay curve yields $N$.
>
> **Reference:** ASTM D7297-14, *Standard Practice for Determining
> Ventilation Effectiveness of Residential and Commercial Buildings*.
""")

w(code_block(cell_map[47]))
w("**Output:**")
w(format_output(cell_map[47]))
w("")

w("""### 6f. Psychrometrics — `pip install comfio[psychrometrics]`

Full psychrometric properties of moist air (wet bulb, dew point, enthalpy,
humidity ratio, vapor pressure, specific volume, degree of saturation) are
computed via PsychroLib.

> **Reference:** ASHRAE Handbook — Fundamentals (2017), Chapter 1:
> *Psychrometrics*. PsychroLib: https://github.com/psychrometrics/psychrolib
""")

w(code_block(cell_map[49]))
w("**Output:**")
w(format_output(cell_map[49]))
w("")

w(code_block(cell_map[50]))
w("")

# --- Section 7: Global IEQ Index ---
w("""---

## 7. Global IEQ Index & Weight Presets

The **Global IEQ Index** aggregates per-domain scores (0–100) into a single
index using configurable weights. `comfio` provides:

- `default_weights()` — equal weights across available domains
- `preset_weights("thermal_first")` — 50% thermal, 17% each other domain
- `preset_weights("visual_first")` — emphasises lighting
- `custom_weights({"thermal": 0.5, "visual": 0.1, ...})` — user-defined

> **Theory — Weighted aggregation.**
>
> $$\\text{IEQ} = \\sum_{d \\in D} w_d \\, s_d \\quad / \\quad \\sum_{d \\in D} w_d$$
>
> where $s_d$ is the domain score and $w_d$ is the weight. Weights are
> normalised so they sum to 1. Missing domains are excluded from both
> numerator and denominator.
>
> **Reference:** CEN/TC 156 WG, *Indoor Environmental Quality (IEQ)
> assessment methods*, prEN 16798-1:2019.
""")

w(code_block(cell_map[52]))
w("**Output:**")
w(format_output(cell_map[52]))
w("")

w(code_block(cell_map[53]))
w("**Output:**")
w(format_output(cell_map[53]))
w("")

w(code_block(cell_map[54]))
w("**Output:**")
w(format_output(cell_map[54]))
w("")

w(code_block(cell_map[55]))
w("")

w(code_block(cell_map[56]))
w("")

# --- Section 8: Compliance & Performance Contracts ---
w("""---

## 8. Compliance & Performance Contracts

`calculate_compliance()` evaluates the IEQ Index against a threshold
(default 80/100) and produces a `ComplianceReport` with compliant hours,
total hours, and a compliance rate. This is the basis for **performance
contract verification** in smart-building applications.

The `comfio.contracts` module generates **Solidity-ready ABI fragments** and
JSON payloads for on-chain compliance attestation via oracle contracts.
""")

w(code_block(cell_map[58]))
w("**Output:**")
w(format_output(cell_map[58]))
w("")

w(code_block(cell_map[59]))
w("**Output:**")
w(format_output(cell_map[59]))
w("")

w(code_block(cell_map[60]))
w("**Output:**")
w(format_output(cell_map[60]))
w("")

w(code_block(cell_map[61]))
w("**Output:**")
w(format_output(cell_map[61]))
w("")

# --- Section 9: ML Integration ---
w("""---

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
""")

w(code_block(cell_map[63]))
w("**Output:**")
w(format_output(cell_map[63]))
w("")

w("""### 9a. scikit-learn — `IEQFeatureExtractor` + RandomForest

`IEQFeatureExtractor` is an sklearn-compatible transformer that wraps comfio
domain evaluations. Given a DataFrame with sensor columns, it computes the
Global IEQ Index and per-domain scores per row.
""")

w(code_block(cell_map[65]))
w("**Output:**")
w(format_output(cell_map[65]))
w("")

w(code_block(cell_map[66]))
w("")

w("""### 9b. PyTorch — `IEQTimeSeriesDataset` + LSTM

`IEQTimeSeriesDataset` wraps the 10-min sensor DataFrame into windowed samples
with computed IEQ scores. We use 1-day windows (144 timesteps at 10-min
sampling) and predict the next day's mean IEQ Index.
""")

w(code_block(cell_map[68]))
w("**Output:**")
w(format_output(cell_map[68]))
w("")

w(code_block(cell_map[69]))
w("**Output:**")
w(format_output(cell_map[69]))
w("")

w(code_block(cell_map[70]))
w("")

w("""### 9c. Keras / TensorFlow — `IEQPreprocessingLayer`

`IEQPreprocessingLayer` wraps comfio domain evaluations as a callable Keras
preprocessing layer. It accepts a pandas DataFrame and returns IEQ features
as a TensorFlow tensor.
""")

w(code_block(cell_map[72]))
w("**Output:**")
w(format_output(cell_map[72]))
w("")

w(code_block(cell_map[73]))
w("**Output:**")
w(format_output(cell_map[73]))
w("")

w(code_block(cell_map[74]))
w("")

w("**Model comparison:**")
w(format_output(cell_map[75]))
w("")

# --- Section 10: LLM Integration ---
w("""---

## 10. LLM Integration

`comfio.llm` provides three layers for integrating IEQ data with Large
Language Models:

1. **Interpreters** — `ieq_to_markdown()` and `ieq_to_summary_dict()`
   serialise IEQ results into token-efficient structured text for LLM context.
2. **Prompts** — Guarded system prompts and diagnostic templates for
   building-comfort LLM agents.
3. **Tools** — `to_openai_tools()` and `to_langchain_tools()` generate
   function-calling schemas that let LLMs invoke comfio evaluations.
""")

w(code_block(cell_map[77]))
w("**Output:**")
w(format_output(cell_map[77]))
w("")

w(code_block(cell_map[78]))
w("**Output:**")
w(format_output(cell_map[78]))
w("")

w(code_block(cell_map[79]))
w("**Output:**")
w(format_output(cell_map[79]))
w("")

w(code_block(cell_map[80]))
w("**Output:**")
w(format_output(cell_map[80]))
w("")

w(code_block(cell_map[81]))
w("**Output:**")
w(format_output(cell_map[81]))
w("")

w(code_block(cell_map[82]))
w("**Output:**")
w(format_output(cell_map[82]))
w("")

# --- Section 11: Smart-Contract Export ---
w("""---

## 11. Smart-Contract Export (web3.py)

`comfio.contracts` generates Solidity ABI fragments and JSON payloads for
on-chain compliance attestation. The `web3.py` integration (optional)
demonstrates how to encode the ABI call for a blockchain transaction.
""")

w(code_block(cell_map[84]))
w("**Output:**")
w(format_output(cell_map[84]))
w("")

w(code_block(cell_map[85]))
w("**Output:**")
w(format_output(cell_map[85]))
w("")

# --- Section 12: Reports ---
w("""---

## 12. Reports — CSV / PDF / DOCX / Intelligent Pipeline

`comfio.reports` provides:

- `ieq_to_csv()` — per-timestamp IEQ scores and compliance flags as CSV
- `ieq_to_pdf()` — formatted PDF report (requires `reportlab`)
- `ieq_to_docx()` — Word document report (requires `python-docx`)
- `run_pipeline()` — intelligent pipeline that auto-detects capabilities
  and runs all possible evaluations
- `generate_pipeline_script()` — exports a reproducible Python script
""")

w(code_block(cell_map[87]))
w("**Output:**")
w(format_output(cell_map[87]))
w("")

w(code_block(cell_map[88]))
w("**Output:**")
w(format_output(cell_map[88]))
w("")

w(code_block(cell_map[89]))
w("**Output:**")
w(format_output(cell_map[89]))
w("")

w(code_block(cell_map[90]))
w("**Output:**")
w(format_output(cell_map[90]))
w("")

w(code_block(cell_map[91]))
w("**Output:**")
w(format_output(cell_map[91]))
w("")

# --- References ---
w("""---

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
python -m nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.timeout=600 \
    examples/walkthrough_executed.ipynb
```

*To regenerate this markdown:*

```bash
python extract_outputs.py
python build_walkthrough_md.py
```
""")

# --- Write the file ---
output = "\n".join(md_parts)
out_path = Path("docs/tutorials/06_walkthrough.md")
out_path.parent.mkdir(parents=True, exist_ok=True)
out_path.write_text(output, encoding="utf-8")
print(f"Wrote {len(output):,} chars to {out_path}")
