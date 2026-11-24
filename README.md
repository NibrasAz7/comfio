<!-- markdownlint-disable MD033 MD041 -->
<p align="center">
  <img src="assets/icon.svg" width="96" alt="comfio logo" />
</p>

<h1 align="center">comfio</h1>

<p align="center">
  <b>Multi-domain Indoor Environmental Quality (IEQ) and performance-contract framework for smart buildings.</b>
</p>

<p align="center">
  <a href="https://pypi.org/project/comfio/"><img src="https://img.shields.io/pypi/v/comfio?color=3776AB&logo=pypi&logoColor=white" alt="PyPI version" /></a>
  <a href="https://pypi.org/project/comfio/"><img src="https://img.shields.io/pypi/pyversions/comfio?color=3776AB" alt="Python versions" /></a>
  <a href="https://github.com/NibrasAz7/comfio/actions/workflows/test.yml"><img src="https://github.com/NibrasAz7/comfio/actions/workflows/test.yml/badge.svg?branch=main" alt="Tests" /></a>
  <a href="https://comfio.readthedocs.io/"><img src="https://readthedocs.org/projects/comfio/badge/?version=stable" alt="Documentation" /></a>
  <a href="https://github.com/NibrasAz7/comfio/blob/main/LICENSE"><img src="https://img.shields.io/pypi/l/comfio?color=blue" alt="License: MIT" /></a>
  <a href="https://pepy.tech/project/comfio"><img src="https://static.pepy.tech/badge/comfio/months/py3?color=blue" alt="Downloads" /></a>
  <img src="https://img.shields.io/badge/status-beta-orange" alt="Status: beta" />
</p>

<p align="center">
  <a href="https://comfio.readthedocs.io/">Documentation</a> ·
  <a href="https://github.com/NibrasAz7/comfio/tree/main/examples">Examples</a> ·
  <a href="#quick-start">Quick start</a>
</p>

comfio bridges the gap between raw building sensor data and actionable smart-building management. It unifies the siloed disciplines of building physics — **thermal**, **visual**, **acoustic**, and **indoor air quality** — under a single Python API, and compresses them into a weighted **Global IEQ Index** (0–100) with compliance tracking and contract-ready outputs.

Built for time-series data (IoT sensors, edge computing) and comfort-based performance contracts. NumPy-native, optional pandas/numpy, ML framework adapters, and a Streamlit analyst workbench.

## Contents

- [Why comfio](#why-comfio)
- [Installation](#installation)
- [Quick start](#quick-start)
- [Long-term IEQ — time in categories](#long-term-ieq--time-in-categories-en-16798-1)
- [GUI analyst workbench](#gui--analyst-workbench)
- [Domain modules](#domain-modules)
- [Try it in Colab](#try-it-in-colab)
- [Architecture](#architecture)
- [Citation](#citation)
- [Development](#development)

## Why comfio

- **Silo-breaking** — thermal, visual, acoustic, and IAQ metrics unified under one API and one index
- **Data-native** — built for massive time-series arrays (Pandas/NumPy), not single-point calculations
- **Standards-grounded** — ISO 7730, ASHRAE 55/62.1, EN 12464-1, EN 16798-1, WHO/EPA/WELL pollutant thresholds
- **Long-term compliance** — EN 16798-1 Annex B time-in-category (% occupied time per I–IV) for every domain
- **Smart-contract ready** — structured JSON outputs with formal ABI schemas for blockchain oracle integration
- **ML/DL compatible** — optional adapters for scikit-learn, PyTorch, and TensorFlow/Keras
- **Fast & light** — core depends only on `numpy`, `pandas`, and `pythermalcomfort`

## Installation

```bash
pip install comfio
```

Optional extras:

```bash
pip install comfio[ml]             # scikit-learn
pip install comfio[torch]          # PyTorch
pip install comfio[keras]          # TensorFlow/Keras
pip install comfio[gui]            # Streamlit workbench + notebook widget
pip install comfio[gui-reports]    # + PDF/DOCX exports
pip install comfio[daylighting]    # pyradiance ray-tracing
pip install comfio[color]          # CRI / CCT (colour-science)
pip install comfio[acoustics]      # RT60, STI (python-acoustics, pyroomacoustics)
pip install comfio[psychrometrics] # PsychroLib (psychrometrics, CO₂ decay ACH)
pip install comfio[all]            # everything
```

Requires Python ≥ 3.11.

## Quick start

### 1. Evaluate domains

```python
import numpy as np
from comfio import evaluate_thermal, evaluate_visual, evaluate_acoustic, evaluate_iaq

thermal = evaluate_thermal(
    tdb=np.array([24.0, 25.0, 26.0]),   # air temp °C
    tr=np.array([24.0, 25.0, 26.0]),    # radiant temp °C
    vr=np.array([0.1, 0.1, 0.1]),        # air velocity m/s
    rh=np.array([50.0, 50.0, 50.0]),     # relative humidity %
    met=1.2, clo=0.5, category="B",      # ISO 7730 category
)
print(f"PMV: {thermal.pmv}, PPD: {thermal.ppd}")

visual = evaluate_visual(illuminance=np.array([450.0, 500.0, 600.0]), task_type="office_writing")
acoustic = evaluate_acoustic(laeq=np.array([35.0, 40.0, 45.0]), nc_level="NC-35")
iaq = evaluate_iaq(co2=np.array([700.0, 900.0, 1100.0]), threshold_level="good")
```

### 2. Global IEQ index

```python
from comfio import calculate_global_ieq, default_weights

ieq = calculate_global_ieq(
    thermal=thermal, visual=visual, acoustic=acoustic, iaq=iaq,
    weights=default_weights(),  # thermal 40% · iaq 25% · visual 20% · acoustic 15%
)
print(f"Global IEQ Index: {ieq.index}")
```

### 3. Compliance & contract outputs

```python
from comfio import calculate_compliance

report = calculate_compliance(ieq, threshold=80.0)
print(f"Compliance rate: {report.compliance_rate_pct:.1f}%")
contract_json = report.to_contract_json()   # blockchain-oracle-ready JSON
```

### 4. Time-series data

```python
import pandas as pd
from comfio import SensorData

df = pd.read_csv("sensor_data.csv")
sensor = SensorData(df=df)                  # auto-detects column names
print(sensor.available_domains())           # ['thermal', 'visual', 'acoustic', 'iaq']
sensor.validate()                           # NaN handling, physical bounds
```

Every `Result` dataclass supports `.to_dict()`, `.to_json()`, and `.to_dataframe()`.

## Long-term IEQ — time in categories (EN 16798-1)

Percentage of occupied time per category I–IV, cumulative I+II / I+II+III shares, and occupied hours — over a whole series or per day:

```python
from comfio import evaluate_temporal_thermal, evaluate_temporal_iaq

summary = evaluate_temporal_thermal(thermal_result, timestamps=df["timestamp"])
summary.percentages   # {'I': 20.0, 'II': 68.0, 'III': 7.0, 'IV': 5.0}
summary.cumulative    # {'I': 20.0, 'I+II': 88.0, 'I+II+III': 95.0, ...}
summary.hours         # occupied hours per category (time-weighted)

daily = evaluate_temporal_thermal(thermal_result, timestamps=df["timestamp"], by="D")
```

One wrapper per domain — all return `TimeInCategoryResult`:

| Function | Input |
| --- | --- |
| `evaluate_temporal_thermal` | PMV / sPMV / sPMVopt |
| `evaluate_temporal_tsv` | TSV occupant votes |
| `evaluate_temporal_adaptive` | adaptive Δt |
| `evaluate_temporal_iaq` | CO₂ (categories on ppm *above outdoor*) |
| `evaluate_temporal_pollutant` | PM2.5, PM10, TVOC, HCHO, CO |
| `evaluate_temporal_visual` | illuminance |
| `evaluate_temporal_acoustic` | LAeq |

`run_pipeline()` computes all of these automatically (`result.temporal_results`; opt out with `config={"temporal": False}`), and `temporal_to_markdown()` renders them for LLM/report pipelines.

> [!TIP]
> Full worked example over a synthetic year of hourly data — all 10 evaluations — in [`examples/temporal_ieq_example.py`](examples/temporal_ieq_example.py) (also exercised by CI). See the [how-to guide](https://comfio.readthedocs.io/en/stable/how_to/long_term_ieq/) for occupancy masks and NaN handling.

## GUI — analyst workbench

```python
import comfio
comfio.gui()              # browser app on localhost:8501
comfio.gui(df)            # preloaded DataFrame
comfio.gui(desktop=True)  # native pywebview window
comfio.gui(tools=["workbench", "calculator"])  # enable only some tools
```

A multipage set of independent tools (Streamlit ≥ 1.51):

| Tool | id | What it does |
| --- | --- | --- |
| Analyse my data | `workbench` | upload → mapping → Global IEQ verdict → evidence/exports (default page) |
| Occupant survey | `survey` | satisfaction-vote import, sensor-vs-occupant check |
| Comfort calculator | `calculator` | PMV/PPD, SET, adaptive models, compare & ranges, psychrometric chart |
| MRT room designer | `mrt` | plan-view MRT/PMV map of a rectangular room |
| Setpoint explorer | `setpoint` | comfort-band edges and deadband widening |
| Global database | `benchmark` | ASHRAE DB II adaptive scatter, sensation bins, PMV bias |
| Learn | `learn` | guided lessons that hand inputs to the calculator |
| About & sources | `about` | enabled tools, provenance, licences, citations (always on) |

The workbench three-act flow: **upload** CSV/Excel → **inspect** auto-detected column mapping → **verdict** (Global IEQ gauge, per-domain traffic lights, comfort scatter, exceedance heatmap, occupant-vote calibration) → **evidence** (compliance rates, WELL v2 T06 / RESET Air presets, CSV/DOCX/PDF/JSON/Markdown exports). Assumptions such as `MRT = air temp` are always shown, never applied silently.

In Jupyter: `comfio.gui_notebook(df)` returns an ipywidgets mini-workbench.

## Domain modules

### Thermal comfort

```python
from comfio import evaluate_thermal, evaluate_spmv, evaluate_adaptive_ashrae, evaluate_adaptive_en

# Full PMV/PPD (ISO 7730 / ASHRAE 55) — see Quick start above
# Simplified PMV (Buratti et al. 2009) — temperature + humidity only
spmv = evaluate_spmv(indoor_temp=tdb, indoor_rh=rh, season="mid")

# Adaptive comfort for naturally ventilated buildings
ashrae = evaluate_adaptive_ashrae(tdb=tdb, tr=tr, t_prevail=20.0, acceptability=80)
en = evaluate_adaptive_en(tdb=tdb, tr=tr, t_running_mean=20.0, category="ii")
```

Plus local thermal discomfort: `evaluate_ankle_draft` (ASHRAE 55 §5.3.3) and `evaluate_vertical_gradient` (ISO 7730 §6.1), combined via `local_discomfort_score()`.

### IAQ — CO₂ & pollutants

```python
from comfio import evaluate_iaq_pollutants

pollutant = evaluate_iaq_pollutants(
    pm25=np.array([8.0, 12.0, 35.0]),
    tvoc=np.array([150.0, 300.0, 500.0]),
    formaldehyde=np.array([20.0, 27.0, 50.0]),
    co=np.array([1.5, 5.0, 10.0]),
    threshold_level="good",   # WHO / EPA NAAQS / WELL v2 tiers
)
print(f"Pollutant IAQ score: {pollutant.score}")
```

Covers PM2.5, PM10, TVOC, formaldehyde, and CO against WHO (2021), EPA NAAQS, and WELL Building Standard v2 thresholds.

### Occupant feedback (TSV)

```python
from comfio import augment_tsv_cdf, evaluate_tsv

# Augment sparse votes to dense sensor timestamps (CDF remapping)
augmented = augment_tsv_cdf(
    sparse_votes=np.array([-2, -1, 0, 0, 1, 1, 2, -1, 0, 1]),
    vote_timestamps=np.arange(10),
    target_timestamps=np.arange(100),
)
tsv_result = evaluate_tsv(augmented)   # ASHRAE 55-2023 Appendix L
```

`train_personalisation` / `evaluate_personalised_pmv` fit an OLS personalisation index `TSV = α·PMV + β` to align model predictions with real occupant feedback.

### Advanced physics (extras)

```python
from comfio import (
    evaluate_reverberation, evaluate_speech_intelligibility,
    evaluate_ventilation, get_psychrometrics, calculate_global_ieq,
)

reverb = evaluate_reverberation(surfaces, absorption, volume, room_type="office")
sti = evaluate_speech_intelligibility(ir_signal, sample_rate=16000)   # pyroomacoustics
vent = evaluate_ventilation(co2_array, timestamps, occupancy_type="office")  # CO₂ decay
psych = get_psychrometrics(tdb=25.0, rh=0.50)                        # psychrolib

ieq = calculate_global_ieq(
    thermal=thermal, visual=visual, acoustic=acoustic, iaq=iaq,
    reverberation=reverb, speech_intelligibility=sti, ventilation=vent,
)
```

Also `evaluate_color_quality` (CRI / CCT / D_uv, colour-science) and `evaluate_daylighting` (pyradiance).

### Weather integration

`fetch_prevailing_temp()` (ASHRAE 55, 7-day mean) and `fetch_running_mean()` (EN 16798-1, exponentially weighted) pull outdoor temperature from meteostat — no manual `t_prevail` calculation. Cached under `~/.cache/comfio/weather/`.

## Try it in Colab

Run the tutorials instantly in your browser — no local install required.

| Notebook | Description | Open in Colab |
| --- | --- | --- |
| `walkthrough_executed.ipynb` | End-to-end: data gen, all domains, Global IEQ, compliance, ML, LLM, smart contracts, reports | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NibrasAz7/comfio/blob/main/examples/walkthrough_executed.ipynb) |
| `simple_tutorial.ipynb` | Quick start: core domains, Global IEQ Index, compliance | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NibrasAz7/comfio/blob/main/examples/simple_tutorial.ipynb) |
| `advanced_domains_tutorial.ipynb` | Daylighting, CRI/CCT, RT60, STI, ventilation, psychrometrics | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NibrasAz7/comfio/blob/main/examples/advanced_domains_tutorial.ipynb) |
| `llm_tutorial.ipynb` | LLM integration: interpreters, prompts, OpenAI/LangChain tool schemas | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NibrasAz7/comfio/blob/main/examples/llm_tutorial.ipynb) |
| `contracts_ml_tutorial.ipynb` | Smart contracts & ML: compliance ABI, web3.py, sklearn/PyTorch/Keras | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NibrasAz7/comfio/blob/main/examples/contracts_ml_tutorial.ipynb) |

Executed versions with pre-run outputs are alongside each notebook in [`examples/`](https://github.com/NibrasAz7/comfio/tree/main/examples), and five more docs tutorials live in [`docs/tutorials/`](https://github.com/NibrasAz7/comfio/tree/main/docs/tutorials). In Colab, run `!pip install comfio[all]` in the first cell.

## Architecture

comfio operates on a **4-layer data flow**:

```text
Layer 1: Data Ingestion (SensorData)
    ↓ Pandas/NumPy time-series arrays
Layer 2: Single-Domain Modules (domains/)
    ↓ Thermal (pythermalcomfort) | Visual (EN 12464-1) | Acoustic (NC) | IAQ | Pollutants | Adaptive | Temporal
Layer 3: Multi-Domain Integration (integration/)
    ↓ Global IEQ Index (0-100) with configurable weighting
Layer 4: Application & Contracts (performance/, reports/, gui/)
    → Compliance rates, exports, smart contract ABI schemas
```

**Key design principle — decoupling**: `integration/` only talks to `domains/`, never to `pythermalcomfort` directly. If upstream releases a breaking change, only `domains/thermal.py` needs updating.

### Weighting presets

| Preset | Thermal | IAQ | Visual | Acoustic | Use case |
| --- | --- | --- | --- | --- | --- |
| `default` | 40% | 25% | 20% | 15% | General (Pierson et al. 2019) |
| `equal` | 25% | 25% | 25% | 25% | Equal weighting |
| `school` | 27% | 26% | 24% | 23% | School children (Yang et al. 2020) |
| `office` | 45% | 30% | 15% | 10% | Office workers |
| `healthcare` | 25% | 40% | 15% | 20% | Healthcare facilities |

```python
from comfio.integration.weights import preset_weights, custom_weights

weights = preset_weights("office")
weights = custom_weights(thermal=0.5, visual=0.2, acoustic=0.1, iaq=0.2)
```

## Citation

If you use comfio in your research, please cite it:

```bibtex
@software{comfio,
  author       = {ABO ALZAHAB, Nibras},
  title        = {comfio: A Multi-Domain IEQ \& Performance Contract Framework for Smart Buildings},
  year         = {2026},
  url          = {https://github.com/NibrasAz7/comfio},
  version      = {0.1.9},
}
```

comfio wraps peer-reviewed methods — when you use a specific function, also cite the underlying work (pythermalcomfort — Tartarini & Schiavon 2020, Fanger 1970, Buratti et al. 2009, de Dear & Brager 1998, Pierson et al. 2019, WHO 2021, and more). The full per-function citation table is in the [docs](https://comfio.readthedocs.io/en/stable/).

## Development

```bash
pip install -e ".[dev]"
pytest                # tests (3.11–3.13 in CI)
ruff check src/ tests/
mypy src/comfio/
```

## License

MIT — see [LICENSE](LICENSE).
