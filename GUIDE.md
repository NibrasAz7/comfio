# comfio — Comprehensive User Guide

> **A multi-domain IEQ & performance contract framework for smart buildings.**

[![PyPI version](https://img.shields.io/pypi/v/comfio.svg)](https://pypi.org/project/comfio/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Installation](#2-installation)
3. [Quick Start](#3-quick-start)
4. [Architecture Overview](#4-architecture-overview)
5. [Domain Evaluation](#5-domain-evaluation)
   - 5.1 [Thermal Comfort (PMV/PPD)](#51-thermal-comfort-pmvppd)
   - 5.2 [Simplified PMV (sPMV)](#52-simplified-pmv-spmv)
   - 5.3 [Adaptive Comfort](#53-adaptive-comfort)
   - 5.4 [TSV Augmentation](#54-tsv-augmentation)
   - 5.5 [Personalised Comfort](#55-personalised-comfort)
   - 5.6 [Visual Comfort](#56-visual-comfort)
   - 5.7 [Acoustic Comfort](#57-acoustic-comfort)
   - 5.8 [IAQ — CO₂](#58-iaq--co₂)
   - 5.9 [Pollutant IAQ](#59-pollutant-iaq)
6. [Advanced Domain Modules](#6-advanced-domain-modules)
   - 6.1 [Daylighting](#61-daylighting)
   - 6.2 [Color Quality](#62-color-quality)
   - 6.3 [Reverberation (RT60)](#63-reverberation-rt60)
   - 6.4 [Speech Intelligibility (STI)](#64-speech-intelligibility-sti)
   - 6.5 [Ventilation from CO₂ Decay](#65-ventilation-from-co₂-decay)
   - 6.6 [Psychrometrics](#66-psychrometrics)
7. [Global IEQ Index](#7-global-ieq-index)
8. [Performance Contracts & Compliance](#8-performance-contracts--compliance)
9. [ML/DL Integration](#9-mldl-integration)
   - 9.1 [scikit-learn Transformers](#91-scikit-learn-transformers)
   - 9.2 [PyTorch Dataset](#92-pytorch-dataset)
   - 9.3 [TensorFlow/Keras Adapter](#93-tensorflowkeras-adapter)
10. [LLM Integration](#10-llm-integration)
11. [Weighting Schemas](#11-weighting-schemas)
12. [Standards Referenced](#12-standards-referenced)
13. [Limitations & Caveats](#13-limitations--caveats)
14. [Citation](#14-citation)

---

## 1. Introduction

comfio bridges the gap between raw building sensor data and actionable smart building management. It unifies **Thermal**, **Visual**, **Acoustic**, and **Indoor Air Quality (IAQ)** metrics into a single **Global IEQ Index** — designed for time-series IoT data and comfort-based performance contracts.

### Key Features

- **Multi-Domain IEQ**: Thermal, Visual, Acoustic, IAQ unified into a single 0–100 index
- **Time-Series Native**: Built for Pandas/NumPy arrays, not single-point calculations
- **Performance Contracts**: Compliance rates + structured JSON for blockchain Oracle integration
- **ML/DL Compatible**: Optional adapters for scikit-learn, PyTorch, and TensorFlow/Keras
- **Advanced Physics**: Radiance daylighting, CRI/CCT, RT60 reverberation, STI, CO₂ decay, psychrometrics
- **Pollutant IAQ**: PM2.5, PM10, TVOC, formaldehyde, CO against WHO/EPA/WELL thresholds
- **Adaptive Comfort**: ASHRAE 55-2023 and EN 16798-1:2019 adaptive models
- **sPMV**: Buratti et al. (2009) seasonal simplified PMV
- **TSV Augmentation**: CDF-based quantile mapping for sparse occupant vote expansion
- **Personalised Comfort**: OLS regression personalisation with per-season support
- **LLM-Native**: Diagnostic prompts, OpenAI/LangChain tool schemas, markdown summaries

---

## 2. Installation

### Core installation

```bash
pip install comfio
```

### Optional extras

```bash
pip install comfio[ml]              # scikit-learn
pip install comfio[torch]           # PyTorch
pip install comfio[keras]           # TensorFlow/Keras
pip install comfio[all]             # All ML frameworks + advanced domains
pip install comfio[daylighting]     # pyradiance (Radiance ray-tracing)
pip install comfio[color]           # colour-science (CRI, CCT)
pip install comfio[acoustics]       # python-acoustics + pyroomacoustics
pip install comfio[psychrometrics]  # PsychroLib
```

### Requirements

- Python ≥ 3.10
- numpy, pandas, pythermalcomfort (core)
- scikit-learn (optional, `[ml]`)
- torch (optional, `[torch]`)
- tensorflow (optional, `[keras]`)

---

## 3. Quick Start

```python
import numpy as np
from comfio import (
    evaluate_thermal, evaluate_visual,
    evaluate_acoustic, evaluate_iaq,
    calculate_global_ieq, calculate_compliance,
)

# 1. Evaluate each domain
thermal = evaluate_thermal(
    tdb=np.array([24.0, 25.0, 26.0]),
    tr=np.array([24.0, 25.0, 26.0]),
    vr=np.array([0.1, 0.1, 0.1]),
    rh=np.array([50.0, 50.0, 50.0]),
    met=1.2, clo=0.5, category="B",
)
visual = evaluate_visual(illuminance=np.array([450.0, 500.0, 600.0]))
acoustic = evaluate_acoustic(laeq=np.array([35.0, 40.0, 45.0]))
iaq = evaluate_iaq(co2=np.array([700.0, 900.0, 1100.0]))

# 2. Merge into a single 0–100 score
ieq = calculate_global_ieq(
    thermal=thermal, visual=visual,
    acoustic=acoustic, iaq=iaq,
)
print(f"Global IEQ Index: {ieq.index}")

# 3. Compliance tracking & contract-ready JSON
report = calculate_compliance(ieq, threshold=80.0)
print(f"Compliance rate: {report.compliance_rate_pct:.1f}%")
print(report.to_contract_json())
```

---

## 4. Architecture Overview

```
comfio/
├── core/
│   ├── data_handler.py      # SensorData: ingestion, column detection, validation
│   └── results.py           # Shared dataclasses (DomainResult, etc.)
├── domains/
│   ├── thermal.py           # PMV/PPD (ISO 7730 / ASHRAE 55)
│   ├── thermal_spmv.py      # Simplified PMV (Buratti et al. 2009)
│   ├── thermal_adaptive.py  # ASHRAE 55-2023 & EN 16798-1:2019
│   ├── thermal_tsv.py       # TSV augmentation (CDF) & compliance
│   ├── thermal_personal.py  # OLS regression personalisation
│   ├── visual.py            # Illuminance (EN 12464-1)
│   ├── visual_daylight.py   # Radiance daylighting (optional)
│   ├── visual_color.py      # CRI/CCT color quality (optional)
│   ├── acoustic.py          # Noise level (NC curves)
│   ├── acoustic_reverb.py   # RT60 reverberation (optional)
│   ├── acoustic_sti.py      # Speech Intelligibility (optional)
│   ├── iaq.py               # CO₂-based IAQ (ASHRAE 62.1)
│   ├── iaq_pollutants.py    # PM2.5, PM10, TVOC, HCHO, CO
│   └── iaq_ventilation.py   # CO₂ decay ventilation (optional)
├── integration/
│   ├── global_ieq.py        # GlobalIEQResult, calculate_global_ieq()
│   └── weights.py           # WeightSchema, presets
├── performance/
│   └── contracts.py         # ComplianceReport, calculate_compliance()
├── ml/
│   ├── sklearn_transformers.py  # IEQFeatureExtractor
│   ├── torch_dataset.py         # IEQTimeSeriesDataset
│   └── keras_adapter.py         # KerasIEQAdapter
├── llm/
│   ├── tools.py             # to_openai_tools, to_langchain_tools
│   ├── interpreters.py      # Function call interpreters
│   └── prompts.py           # Diagnostic prompt templates
└── utils/
    ├── validation.py        # Input array validation
    └── psychrometrics.py    # PsychroLib wrapper (optional)
```

### Data Flow

```
Sensor Data → SensorData → Domain Evaluations → Global IEQ → Compliance Report → Contract JSON
```

Each domain evaluation produces a result dataclass with:
- `score`: 0–100 numeric score
- `compliant`: boolean array per timestamp
- Domain-specific fields (e.g., `pmv`, `ppd` for thermal)

---

## 5. Domain Evaluation

### 5.1 Thermal Comfort (PMV/PPD)

The PMV model predicts the mean thermal sensation vote on a 7-point scale (−3 to +3) based on six parameters: air temperature, radiant temperature, air velocity, relative humidity, metabolic rate, and clothing insulation.

```python
from comfio import evaluate_thermal

thermal = evaluate_thermal(
    tdb=np.array([24.0, 25.0, 26.0]),    # air temperature (°C)
    tr=np.array([24.0, 25.0, 26.0]),     # mean radiant temperature (°C)
    vr=np.array([0.1, 0.1, 0.1]),        # air velocity (m/s)
    rh=np.array([50.0, 50.0, 50.0]),     # relative humidity (%)
    met=1.2,                              # metabolic rate (met)
    clo=0.5,                              # clothing insulation (clo)
    category="B",                         # ISO 7730 category (A/B/C)
)

print(thermal.pmv)        # PMV values per timestamp
print(thermal.ppd)        # PPD values per timestamp
print(thermal.score)      # 0–100 score per timestamp
print(thermal.compliant)  # boolean array
```

**ISO 7730 Categories:**

| Category | PMV range | PPD limit |
|---|---|---|
| A | −0.2 to +0.2 | ≤ 6% |
| B | −0.5 to +0.5 | ≤ 10% |
| C | −0.7 to +0.7 | ≤ 15% |

See [Theory — PMV/PPD](docs/theory/thermal_pmv_ppd.md) for the mathematical background.

### 5.2 Simplified PMV (sPMV)

The sPMV model by Buratti et al. (2009) requires only indoor temperature and relative humidity — ideal for BMS with limited sensors.

```python
from comfio import evaluate_spmv

spmv = evaluate_spmv(
    indoor_temp=np.array([23.0, 24.0, 25.0]),
    indoor_rh=np.array([50.0, 50.0, 50.0]),
    season="mid",  # "winter", "mid", "summer"
)
print(spmv.spmv)  # simplified PMV values
print(spmv.score) # 0–100 score
```

### 5.3 Adaptive Comfort

Adaptive models are for naturally ventilated buildings where occupants adapt to outdoor conditions.

```python
from comfio import evaluate_adaptive_ashrae, evaluate_adaptive_en

# ASHRAE 55-2023
ashrae = evaluate_adaptive_ashrae(
    tdb=np.array([24.0, 25.0, 26.0]),
    tr=np.array([24.0, 25.0, 26.0]),
    t_prevail=20.0,       # prevailing mean outdoor temp (°C)
    acceptability=80,     # 80% or 90%
)

# EN 16798-1:2019
en = evaluate_adaptive_en(
    tdb=np.array([24.0, 25.0, 26.0]),
    tr=np.array([24.0, 25.0, 26.0]),
    t_running_mean=20.0,  # running mean outdoor temp (°C)
    category="ii",        # I, II, III
)
```

### 5.4 TSV Augmentation

Expand sparse occupant votes to dense sensor timestamps using CDF-based quantile mapping.

```python
from comfio import augment_tsv_cdf, evaluate_tsv

# Sparse votes from surveys
sparse_votes = np.array([-2, -1, 0, 0, 1, 1, 2, -1, 0, 1])
vote_timestamps = np.arange(10)
target_timestamps = np.arange(100)

# Augment to dense timestamps
augmented = augment_tsv_cdf(
    sparse_votes=sparse_votes,
    vote_timestamps=vote_timestamps,
    target_timestamps=target_timestamps,
)

# Evaluate TSV compliance (ASHRAE 55-2023 Appendix L)
tsv_result = evaluate_tsv(augmented)
print(f"Compliance rate: {tsv_result.compliance_rate:.1%}")
```

### 5.5 Personalised Comfort

Personalise PMV predictions to match individual occupant preferences using OLS regression.

```python
from comfio import train_personalisation, evaluate_personalised_pmv

# Train on historical data
index = train_personalisation(
    pmv=historical_pmv,
    tsv=historical_tsv,
)

# Apply to new data
result = evaluate_personalised_pmv(
    tdb=tdb, tr=tr, vr=vr, rh=rh, met=1.2, clo=0.5,
    personalisation_index=index,
)
```

### 5.6 Visual Comfort

Evaluates illuminance against EN 12464-1 task-based thresholds.

```python
from comfio import evaluate_visual

visual = evaluate_visual(
    illuminance=np.array([450.0, 500.0, 600.0]),
    task_type="office",  # "office", "reading", "drafting", etc.
)
print(visual.score)  # 0–100
```

### 5.7 Acoustic Comfort

Evaluates A-weighted equivalent noise levels against NC (Noise Criterion) curves.

```python
from comfio import evaluate_acoustic

acoustic = evaluate_acoustic(
    laeq=np.array([35.0, 40.0, 45.0]),
    space_type="office",  # "office", "classroom", "hospital", etc.
)
print(acoustic.score)  # 0–100
```

### 5.8 IAQ — CO₂

Evaluates CO₂ concentration against ASHRAE 62.1 ventilation thresholds.

```python
from comfio import evaluate_iaq

iaq = evaluate_iaq(
    co2=np.array([700.0, 900.0, 1100.0]),
    occupancy_type="office",  # "office", "classroom", "residential"
)
print(iaq.score)  # 0–100
```

### 5.9 Pollutant IAQ

Evaluates indoor pollutants against WHO/EPA/WELL thresholds.

```python
from comfio import evaluate_iaq_pollutants

pollutant = evaluate_iaq_pollutants(
    pm25=np.array([8.0, 12.0, 35.0]),         # μg/m³
    pm10=np.array([20.0, 35.0, 80.0]),        # μg/m³
    tvoc=np.array([150.0, 300.0, 500.0]),     # μg/m³
    formaldehyde=np.array([20.0, 27.0, 50.0]),# μg/m³
    co=np.array([1.5, 5.0, 10.0]),            # mg/m³
    threshold_level="good",  # "good", "moderate", "permissive"
)
print(pollutant.score)  # 0–100 (weakest-link across pollutants)
```

---

## 6. Advanced Domain Modules

### 6.1 Daylighting

Radiance-based daylighting evaluation (requires `pip install comfio[daylighting]`).

```python
from comfio import evaluate_daylighting

daylight = evaluate_daylighting(
    scene_file="room.rad",
    sky_type="clear",
    point_coords=[(2.0, 3.0, 0.8)],
)
print(daylight.illuminance)  # lux at each point
print(daylight.sDA)          # spatial daylight autonomy
```

### 6.2 Color Quality

CRI (Color Rendering Index) and CCT (Correlated Color Temperature) evaluation (requires `pip install comfio[color]`).

```python
from comfio import evaluate_color_quality

color = evaluate_color_quality(
    spectral_data=spectral_array,  # nm vs intensity
    reference_cct=4000,             # K
)
print(color.cri)  # Ra value (0–100)
print(color.cct)  # K
```

### 6.3 Reverberation (RT60)

Sabine/Eyring reverberation time calculation (requires `pip install comfio[acoustics]`).

```python
from comfio import evaluate_reverberation

reverb = evaluate_reverberation(
    surfaces={"floor": 50.0, "ceiling": 50.0, "walls": 120.0},
    absorption={"floor": 0.05, "ceiling": 0.80, "walls": 0.10},
    volume=300.0,       # m³
    room_type="office",
)
print(reverb.rt60)  # seconds
```

### 6.4 Speech Intelligibility (STI)

Speech Transmission Index from impulse response (requires `pip install comfio[acoustics]`).

```python
from comfio import evaluate_speech_intelligibility

sti = evaluate_speech_intelligibility(
    ir_signal=impulse_response,  # numpy array
    sample_rate=16000,
)
print(sti.sti)   # 0.0–1.0
print(sti.score) # 0–100
```

### 6.5 Ventilation from CO₂ Decay

Estimate ventilation rate from CO₂ decay curve (requires `pip install comfio[psychrometrics]`).

```python
from comfio import evaluate_ventilation

vent = evaluate_ventilation(
    co2=co2_array,           # ppm time-series
    timestamps=ts_array,     # hours
    occupancy_type="office",
)
print(vent.ach)   # air changes per hour
print(vent.score) # 0–100
```

### 6.6 Psychrometrics

Psychrometric properties via PsychroLib (requires `pip install comfio[psychrometrics]`).

```python
from comfio import get_psychrometrics

psych = get_psychrometrics(tdb=25.0, rh=0.50)
print(f"Dew point: {psych.dew_point:.1f}°C")
print(f"Enthalpy: {psych.enthalpy:.1f} kJ/kg")
print(f"Humidity ratio: {psych.humidity_ratio:.4f} kg/kg")
```

---

## 7. Global IEQ Index

The Global IEQ Index merges all domain scores into a unified 0–100 metric using configurable weights.

```python
from comfio import calculate_global_ieq

# Basic: 4 domains
ieq = calculate_global_ieq(
    thermal=thermal,
    visual=visual,
    acoustic=acoustic,
    iaq=iaq,
)

# With advanced domains
ieq = calculate_global_ieq(
    thermal=thermal,
    visual=visual,
    acoustic=acoustic,
    iaq=iaq,
    pollutant_iaq=pollutant,           # blends 50/50 with CO₂ IAQ
    tsv=tsv_result,                    # overrides PMV thermal score
    reverberation=reverb,              # augments acoustic
    speech_intelligibility=sti,        # augments acoustic
    ventilation=vent,                  # augments IAQ
)

print(ieq.index)       # 0–100 per timestamp
print(ieq.weights)     # weights used
print(ieq.domain_scores)  # individual domain scores
```

### Default Weights

| Domain | Weight |
|---|---|
| Thermal | 40% |
| IAQ | 25% |
| Visual | 20% |
| Acoustic | 15% |

See [Weighting Schemas](#11-weighting-schemas) for customization.

---

## 8. Performance Contracts & Compliance

```python
from comfio import calculate_compliance

report = calculate_compliance(
    ieq_result=ieq,
    threshold=80.0,               # minimum IEQ index for compliance
    period_start=1717200000.0,    # Unix timestamp
    period_end=1717286400.0,
)

print(f"IEQ Index (avg): {report.ieq_index_avg:.1f}")
print(f"Compliance rate: {report.compliance_rate_pct:.1f}%")
print(f"Is compliant: {report.is_compliant}")

# Export for blockchain Oracle
contract_json = report.to_contract_json()
```

### JSON Output

```json
{
  "period_start": 1717200000.0,
  "period_end": 1717286400.0,
  "ieq_index_avg": 82.5,
  "ieq_index_min": 65.0,
  "ieq_index_max": 95.0,
  "compliance_rate_pct": 87.5,
  "is_compliant": true,
  "threshold": 80.0,
  "schema_version": "0.1.0"
}
```

---

## 9. ML/DL Integration

### 9.1 scikit-learn Transformers

```python
from comfio.ml.sklearn_transformers import IEQFeatureExtractor

extractor = IEQFeatureExtractor()
features = extractor.fit_transform(ieq_results)
# Returns statistical features (mean, std, min, max, percentiles) per domain
```

### 9.2 PyTorch Dataset

```python
from comfio.ml.torch_dataset import IEQTimeSeriesDataset
from torch.utils.data import DataLoader

dataset = IEQTimeSeriesDataset(
    ieq_results=ieq_list,
    window_size=24,
    horizon=1,
)
loader = DataLoader(dataset, batch_size=32, shuffle=True)
```

### 9.3 TensorFlow/Keras Adapter

```python
from comfio.ml.keras_adapter import KerasIEQAdapter

adapter = KerasIEQAdapter(
    window_size=24,
    horizon=1,
)
model = adapter.build_model(input_dim=4)
model.fit(loader, epochs=50)
```

---

## 10. LLM Integration

### Tool Schemas

```python
from comfio.llm.tools import to_openai_tools, to_langchain_tools

# OpenAI function-calling format
openai_tools = to_openai_tools()

# LangChain Tool objects
langchain_tools = to_langchain_tools()
```

### Diagnostic Summaries

```python
from comfio import generate_markdown_summary, ieq_to_summary_dict

summary_dict = ieq_to_summary_dict(ieq)
markdown_report = generate_markdown_summary(ieq)
```

### Prompt Templates

```python
from comfio import DIAGNOSTIC_PROMPT_TEMPLATE, EDGE_SYSTEM_PROMPT, format_prompt

prompt = format_prompt(
    DIAGNOSTIC_PROMPT_TEMPLATE,
    ieq_summary=summary_dict,
    building_type="office",
)
```

---

## 11. Weighting Schemas

### Presets

| Preset | Thermal | IAQ | Visual | Acoustic | Use Case |
|---|---|---|---|---|---|
| default | 40% | 25% | 20% | 15% | General buildings |
| equal | 25% | 25% | 25% | 25% | Equal weighting |
| school | 27% | 26% | 24% | 23% | School children |
| office | 45% | 30% | 15% | 10% | Office workers |
| healthcare | 25% | 40% | 15% | 20% | Healthcare facilities |

### Custom Weights

```python
from comfio.integration.weights import custom_weights, WeightSchema

# Via helper (auto-normalizes)
weights = custom_weights(thermal=0.50, visual=0.15, acoustic=0.10, iaq=0.25)

# Via dataclass
weights = WeightSchema(thermal=0.40, visual=0.20, acoustic=0.15, iaq=0.25)

# Use in calculation
ieq = calculate_global_ieq(..., weights=weights)
```

---

## 12. Standards Referenced

| Standard | Domain |
|---|---|
| ISO 7730 | Thermal comfort — PMV/PPD |
| ASHRAE 55 | Thermal environmental conditions |
| ASHRAE 55-2023 Appendix L | TSV compliance (|TSV| ≤ 1.5) |
| EN 16798-1:2019 | Adaptive thermal comfort |
| EN 12464-1 | Lighting of work places |
| ASHRAE 62.1 | Ventilation for acceptable IAQ |
| WHO Air Quality Guidelines (2021) | PM2.5, PM10 thresholds |
| EPA NAAQS | Criteria pollutant thresholds |
| WELL Building Standard v2 | Feature A01 pollutant thresholds |

---

## 13. Limitations & Caveats

- **PMV**: Steady-state model, not suitable for transient conditions or outdoor spaces
- **Adaptive**: Only for naturally ventilated buildings with occupant window control
- **sPMV**: Reduced accuracy vs full PMV; assumes default met/clo/vr
- **TSV**: Distribution-preserving but synthetic; not a substitute for real surveys
- **Pollutant IAQ**: No synergistic effects; threshold-based scoring
- **Global IEQ**: Weighted sum assumes domain independence; no temporal dynamics
- **Advanced modules**: Require optional dependencies; see applicability notes

See [Limitations & Caveats](docs/theory/limitations.md) for full details.

---

## 14. Citation

```bibtex
@software{comfio,
  author       = {comfio Contributors},
  title        = {comfio: A Multi-Domain IEQ \& Performance Contract Framework for Smart Buildings},
  year         = {2025},
  url          = {https://github.com/NibrasAz7/comfio},
  version      = {0.1.0},
}
```

---

## License

MIT — see [LICENSE](https://github.com/NibrasAz7/comfio/blob/main/LICENSE).
