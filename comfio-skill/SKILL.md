---
name: comfio
description: >
  Use when working with the comfio Python package for Indoor Environmental Quality (IEQ)
  evaluation in smart buildings. Trigger when the user mentions IEQ, indoor environmental
  quality, thermal comfort, PMV, PPD, building sensors, smart buildings, performance
  contracts, comfio, SensorData, Global IEQ Index, or needs to evaluate building comfort
  from sensor data. Also trigger for ASHRAE 55, ISO 7730, EN 12464-1, EN 16798, adaptive
  comfort, TSV, pollutant IAQ, PM2.5, TVOC, blockchain building contracts, or any task
  involving multi-domain comfort evaluation from time-series sensor arrays. Use this skill
  whenever the user wants to install, use, debug, or integrate comfio into a project, even
  if they don't explicitly say "comfio" — any mention of evaluating building comfort from
  sensor data should trigger this skill.
license: MIT
metadata:
  author: https://github.com/NibrasAz7
  version: "0.1.2"
  domain: building-science
  triggers: IEQ, thermal comfort, PMV, PPD, building sensors, smart buildings, comfio,
    ASHRAE 55, ISO 7730, EN 12464, EN 16798, adaptive comfort, TSV, pollutant IAQ,
    Global IEQ Index, performance contracts, blockchain Oracle, building physics
  role: specialist
  scope: implementation
  output-format: code
risk: safe
---

# comfio Skill

A skill for AI agents working with the **comfio** Python package — a multi-domain
Indoor Environmental Quality (IEQ) and performance contract framework for smart buildings.

## What comfio Does

comfio unifies four IEQ domains — **Thermal**, **Visual**, **Acoustic**, and **Indoor Air
Quality (IAQ)** — into a single **Global IEQ Index** (0–100) from time-series sensor data.
It generates compliance reports with structured JSON payloads for blockchain Oracle
integration, provides LLM-native diagnostic adapters, and includes ML/DL pipeline adapters.

## Installation

```bash
pip install comfio                          # core (numpy, pandas, pythermalcomfort)
pip install comfio[ml]                      # + scikit-learn
pip install comfio[torch]                   # + PyTorch
pip install comfio[keras]                   # + TensorFlow/Keras
pip install comfio[agent]                   # + pydantic (LLM tool schemas)
pip install comfio[daylighting]             # + pyradiance
pip install comfio[color]                   # + colour-science
pip install comfio[acoustics]               # + python-acoustics, pyroomacoustics
pip install comfio[psychrometrics]          # + psychrolib
pip install comfio[all]                     # everything
```

## Architecture (4 layers)

```
Layer 1: Data Ingestion (core/)        → SensorData wraps Pandas DataFrame
Layer 2: Domain Modules (domains/)     → thermal, visual, acoustic, IAQ, + advanced
Layer 3: Multi-Domain Integration      → calculate_global_ieq() → 0-100 index
Layer 4: Application & Contracts       → compliance rates, JSON, Solidity ABI
```

**Key principle — Decoupling**: `integration/` only talks to `domain/` modules, never to
`pythermalcomfort` directly. Optional deps use lazy imports.

## Quick-Start Patterns

### Pattern 1: Evaluate all four domains → Global IEQ → Compliance

```python
import numpy as np
from comfio import (
    evaluate_thermal, evaluate_visual,
    evaluate_acoustic, evaluate_iaq,
    calculate_global_ieq, calculate_compliance,
)

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

ieq = calculate_global_ieq(
    thermal=thermal, visual=visual,
    acoustic=acoustic, iaq=iaq,
)
report = calculate_compliance(ieq, threshold=80.0)
print(f"IEQ: {ieq.index}, Compliance: {report.compliance_rate_pct:.1f}%")
```

### Pattern 2: SensorData from DataFrame (auto-detect columns)

```python
import pandas as pd
from comfio import SensorData, evaluate_thermal, calculate_global_ieq

df = pd.read_csv("sensor_data.csv")
sensor = SensorData(df=df)
sensor.validate()
print(sensor.available_domains())  # ['thermal', 'visual', 'acoustic', 'iaq']

thermal = evaluate_thermal(
    tdb=sensor.get_validated("air_temp_c"),
    tr=sensor.get_validated("radiant_temp_c"),
    vr=sensor.get_validated("air_velocity_ms"),
    rh=sensor.get_validated("relative_humidity_pct"),
    met=1.2, clo=0.5,
)
ieq = calculate_global_ieq(thermal=thermal)
```

### Pattern 3: LLM-native markdown diagnostic

```python
from comfio import generate_markdown_summary

md = generate_markdown_summary(df=sensor_df, threshold=80.0, zone_id="Zone-A")
print(md)  # ~200-token structured markdown for LLM context
```

### Pattern 4: Blockchain contract JSON

```python
from comfio import calculate_compliance

report = calculate_compliance(ieq, threshold=80.0)
contract_json = report.to_contract_json()
# Matches IEQComplianceOracle.submitCompliance ABI
```

### Pattern 5: Pollutant IAQ + advanced blending

```python
from comfio import evaluate_iaq_pollutants, calculate_global_ieq

pollutant = evaluate_iaq_pollutants(
    pm25=np.array([8.0, 12.0, 35.0]),
    tvoc=np.array([150.0, 300.0, 500.0]),
    formaldehyde=np.array([20.0, 27.0, 50.0]),
    co=np.array([1.5, 5.0, 10.0]),
)
ieq = calculate_global_ieq(
    thermal=thermal_res, iaq=iaq_res,
    pollutant_iaq=pollutant,  # 50/50 blend with IAQ score
)
```

## API Summary Table

| Function | Module | Key Params | Returns |
|---|---|---|---|
| `SensorData(df=)` | core | DataFrame, column_map (optional) | SensorData |
| `evaluate_thermal()` | domains | tdb, tr, vr, rh, met, clo, category | ThermalResult |
| `evaluate_visual()` | domains | illuminance, task_type | VisualResult |
| `evaluate_acoustic()` | domains | laeq, nc_level | AcousticResult |
| `evaluate_iaq()` | domains | co2, threshold_level | IAQResult |
| `evaluate_iaq_pollutants()` | domains | pm25, pm10, tvoc, formaldehyde, co | PollutantIAQResult |
| `evaluate_adaptive_ashrae()` | domains | tdb, tr, t_prevail, acceptability | AdaptiveThermalResult |
| `evaluate_adaptive_en()` | domains | tdb, tr, t_running_mean, category | AdaptiveThermalResult |
| `evaluate_spmv()` | domains | indoor_temp, indoor_rh, season | SPMVResult |
| `evaluate_tsv()` | domains | tsv_array | TSVResult |
| `augment_tsv_cdf()` | domains | sparse_votes, vote_timestamps, target_timestamps | np.ndarray |
| `train_personalisation()` | domains | pmv, tsv | PersonalisationIndex |
| `evaluate_personalised_pmv()` | domains | tdb, tr, vr, rh, met, clo, personalisation_index | PersonalisedPMVResult |
| `calculate_global_ieq()` | integration | thermal, visual, acoustic, iaq, weights, ... | GlobalIEQResult |
| `calculate_compliance()` | performance | ieq_result, threshold | ComplianceReport |
| `ieq_to_markdown()` | llm | ieq_result, compliance_report | str |
| `generate_markdown_summary()` | llm | df, threshold, zone_id | str |
| `evaluate_reverberation()` | domains | surfaces, absorption, volume | ReverberationResult |
| `evaluate_speech_intelligibility()` | domains | ir_signal, sample_rate | SpeechIntelligibilityResult |
| `evaluate_ventilation()` | domains | co2, timestamps, occupancy_type | VentilationResult |
| `get_psychrometrics()` | domains | tdb, rh | PsychrometricResult |
| `evaluate_daylighting()` | domains | (requires [daylighting] extra) | DaylightingResult |
| `evaluate_color_quality()` | domains | (requires [color] extra) | ColorQualityResult |

## Weighting Presets

| Preset | Thermal | IAQ | Visual | Acoustic | Source |
|---|---|---|---|---|---|
| `default` | 40% | 25% | 20% | 15% | Pierson et al. 2019 |
| `equal` | 25% | 25% | 25% | 25% | — |
| `school` | 27% | 26% | 24% | 23% | Yang et al. 2020 |
| `office` | 45% | 30% | 15% | 10% | — |
| `healthcare` | 25% | 40% | 15% | 20% | — |

```python
from comfio.integration.weights import preset_weights, custom_weights
weights = preset_weights("office")
weights = custom_weights(thermal=0.5, visual=0.2, acoustic=0.1, iaq=0.2)
```

## Standards Referenced

- ISO 7730 — Thermal comfort PMV/PPD
- ASHRAE 55-2023 — Thermal environmental conditions (+ Appendix L TSV)
- EN 12464-1 — Lighting of work places
- EN 16798-1:2019 — Adaptive thermal comfort
- ASHRAE 62.1 — Ventilation for acceptable IAQ
- WHO Air Quality Guidelines 2021 — PM2.5, PM10
- WELL Building Standard v2 — Pollutant thresholds

## Common Pitfalls

1. **Missing optional extras**: Advanced modules (daylighting, color, acoustics, psychrometrics)
   require `pip install comfio[extra]`. They import lazily and raise `ImportError` on use.
2. **Array shapes**: All domain functions expect 1-D numpy arrays of equal length (one per timestamp).
3. **Column auto-detection**: `SensorData` maps common aliases (tdb, ta, temperature → air_temp_c).
   For custom names, pass `column_map={"air_temp_c": "your_col_name"}`.
4. **NaN handling**: `sensor.validate()` handles NaNs via strategies: "interpolate" (default),
   "drop", "fill_zero", "raise".
5. **TSV override**: When both `thermal` and `tsv` are passed to `calculate_global_ieq`, TSV
   overrides the thermal score (occupant feedback = ground truth).
6. **Weight renormalization**: Missing domains are handled by renormalizing remaining weights.
   No need to adjust weights manually.

## Deep-Dive References

For detailed information, read the reference files in `referances/`:

- `referances/api_reference.md` — Complete public API with signatures and return types
- `referances/architecture.md` — 4-layer architecture, module map, decoupling principle
- `referances/domain_modules.md` — Each domain function explained in detail
- `referances/integration_contracts.md` — Global IEQ Index, weighting, compliance, Solidity ABI
- `referances/llm_adapters.md` — Markdown interpreters, prompts, OpenAI/LangChain tools
- `referances/ml_adapters.md` — scikit-learn, PyTorch, TensorFlow/Keras adapters

## Useful Scripts

- `scripts/verify_install.py` — Check comfio installation and available extras
- `scripts/generate_sensor_template.py` — Generate synthetic sensor CSV for testing
- `scripts/run_ieq_pipeline.py` — End-to-end CLI: CSV → domains → IEQ → compliance JSON

## Code Templates

- `templates/full_pipeline.py` — Complete evaluation pipeline
- `templates/llm_diagnostic.py` — LLM-native diagnostic workflow
- `templates/ml_sklearn_pipeline.py` — scikit-learn integration
- `templates/blockchain_export.py` — Blockchain contract JSON export

## Links

- GitHub: https://github.com/NibrasAz7/comfio
- PyPI: https://pypi.org/project/comfio/
- License: MIT
- Python: >= 3.10
