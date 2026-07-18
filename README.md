# comfio

**A multi-domain IEQ & performance contract framework for smart buildings.**

comfio bridges the gap between raw building sensor data and actionable smart building management. It breaks the silos between different building physics disciplines — bringing together **Thermal**, **Visual**, **Acoustic**, and **Indoor Air Quality (IAQ)** metrics into a unified **Global IEQ Index**.

Designed for time-series data (IoT sensors, edge computing) and comfort-based performance contracts, comfio enables researchers and building managers to automate compliance tracking and generate smart-contract-ready outputs.

## Why comfio?

- **Silo-Breaking**: Unifies separated domains (Thermal, Acoustic, Visual, IAQ) under a single Python API
- **Data-Native**: Built to ingest massive arrays of time-series data (Pandas/NumPy) rather than single-point calculations
- **Actionable Output**: Translates physical equations into compliance rates for building performance contracts
- **Smart Contract Ready**: Generates structured JSON outputs with formal ABI schemas for blockchain Oracle integration
- **ML/DL Compatible**: NumPy-native core with optional adapters for scikit-learn, PyTorch, and TensorFlow/Keras
- **Advanced Physics Modules**: Optional extras for Radiance daylighting, CRI/CCT color quality, RT60 reverberation, STI speech intelligibility, CO₂ decay ventilation, and full psychrometrics
- **Pollutant IAQ**: PM2.5, PM10, TVOC, formaldehyde, and CO evaluation against WHO, EPA NAAQS, and WELL Building Standard v2 thresholds
- **Adaptive Thermal Comfort**: ASHRAE 55-2023 and EN 16798-1:2019 adaptive models for naturally ventilated buildings
- **Simplified PMV (sPMV)**: Buratti et al. (2009) seasonal model requiring only temperature and humidity
- **TSV Augmentation**: CDF-based remapping (quantile mapping) to augment sparse occupant votes to dense sensor timestamps while preserving the empirical distribution
- **Personalised Comfort**: OLS regression-based personalisation of model predictions to match occupant feedback (TSV), with per-season support
- **Fast & Light**: Core depends only on numpy, pandas, and pythermalcomfort

## Installation

```bash
pip install comfio
```

With ML/DL framework support:

```bash
pip install comfio[ml]        # scikit-learn
pip install comfio[torch]     # PyTorch
pip install comfio[keras]     # TensorFlow/Keras
pip install comfio[all]       # All frameworks + advanced domains
```

Advanced physics-based domain evaluation (optional extras):

```bash
pip install comfio[daylighting]      # pyradiance (Radiance ray-tracing)
pip install comfio[color]            # colour-science (CRI, CCT)
pip install comfio[acoustics]        # python-acoustics + pyroomacoustics (RT60, STI)
pip install comfio[psychrometrics]   # PsychroLib (psychrometric properties, CO₂ decay ACH)
```

## Quick Start

### 1. Evaluate Individual Domains

```python
import numpy as np
from comfio import evaluate_thermal, evaluate_visual, evaluate_acoustic, evaluate_iaq

# Thermal comfort (ISO 7730 / ASHRAE 55)
thermal = evaluate_thermal(
    tdb=np.array([24.0, 25.0, 26.0]),  # air temp °C
    tr=np.array([24.0, 25.0, 26.0]),   # radiant temp °C
    vr=np.array([0.1, 0.1, 0.1]),       # air velocity m/s
    rh=np.array([50.0, 50.0, 50.0]),    # relative humidity %
    met=1.2,                            # metabolic rate
    clo=0.5,                            # clothing insulation
    category="B",                       # ISO 7730 category
)
print(f"PMV: {thermal.pmv}, PPD: {thermal.ppd}")

# Visual comfort (EN 12464-1)
visual = evaluate_visual(
    illuminance=np.array([450.0, 500.0, 600.0]),
    task_type="office_writing",
)

# Acoustic comfort (NC curves)
acoustic = evaluate_acoustic(
    laeq=np.array([35.0, 40.0, 45.0]),
    nc_level="NC-35",
)

# IAQ (ASHRAE 62.1 indicators)
iaq = evaluate_iaq(
    co2=np.array([700.0, 900.0, 1100.0]),
    threshold_level="good",
)
```

### 2. Calculate Global IEQ Index

```python
from comfio import calculate_global_ieq, default_weights

# Merge all domains into a single 0-100 score
ieq = calculate_global_ieq(
    thermal=thermal,
    visual=visual,
    acoustic=acoustic,
    iaq=iaq,
    weights=default_weights(),  # thermal=40%, iaq=25%, visual=20%, acoustic=15%
)
print(f"Global IEQ Index: {ieq.index}")
print(f"Domain scores: {ieq.domain_scores}")
```

### 3. Compliance Tracking & Contract Outputs

```python
from comfio import calculate_compliance

report = calculate_compliance(ieq, threshold=80.0)
print(f"Compliance rate: {report.compliance_rate_pct:.1f}%")
print(f"Average IEQ: {report.ieq_index_avg:.1f}")

# Generate JSON for blockchain Oracle
contract_json = report.to_contract_json()
print(contract_json)
```

### 4. Time-Series Data with SensorData

```python
import pandas as pd
from comfio import SensorData

# Load your sensor DataFrame
df = pd.read_csv("sensor_data.csv")
sensor = SensorData(df=df)

# Auto-detects column names (tdb, ta, temperature → air_temp_c, etc.)
print(sensor.available_domains())  # ['thermal', 'visual', 'acoustic', 'iaq']

# Validate data (NaN handling, physical bounds checking)
sensor.validate()
clean_temp = sensor.get_validated("air_temp_c")
```

## ML/DL Integration

### scikit-learn Pipeline

```python
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from comfio.ml.sklearn_transformers import IEQFeatureExtractor

pipe = Pipeline([
    ("ieq", IEQFeatureExtractor()),
    ("model", RandomForestRegressor()),
])
pipe.fit(train_df, train_labels)
predictions = pipe.predict(test_df)
```

### PyTorch DataLoader

```python
from torch.utils.data import DataLoader
from comfio.ml.torch_dataset import IEQTimeSeriesDataset

dataset = IEQTimeSeriesDataset(df, window_size=24, stride=1)
loader = DataLoader(dataset, batch_size=32, shuffle=True)
for batch in loader:
    raw = batch["raw"]           # (32, 24, n_sensors)
    ieq = batch["ieq_index"]     # (32, 24)
```

### TensorFlow/Keras

```python
from comfio.ml.keras_adapter import IEQPreprocessingLayer

layer = IEQPreprocessingLayer()
layer.adapt(train_df)
features = layer(train_df)  # tf.Tensor of IEQ features
```

## Advanced Domain Evaluation

comfio offers optional physics-based modules that go beyond simple threshold checks. These require separate extras but integrate seamlessly with the Global IEQ Index.

```python
from comfio import (
    evaluate_reverberation, evaluate_speech_intelligibility,
    evaluate_ventilation, get_psychrometrics,
    calculate_global_ieq,
)

# Reverberation time (python-acoustics)
reverb = evaluate_reverberation(surfaces, absorption, volume, room_type="office")

# Speech intelligibility from impulse response (pyroomacoustics)
sti = evaluate_speech_intelligibility(ir_signal, sample_rate=16000)

# Ventilation rate from CO₂ decay (psychrolib)
vent = evaluate_ventilation(co2_array, timestamps, occupancy_type="office")

# Psychrometric properties (psychrolib)
psych = get_psychrometrics(tdb=25.0, rh=0.50)

# Blend advanced results into Global IEQ Index
result = calculate_global_ieq(
    thermal=thermal, visual=visual, acoustic=acoustic, iaq=iaq,
    reverberation=reverb, speech_intelligibility=sti, ventilation=vent,
)
```

See the [User Guide](https://github.com/NibrasAz7/comfio#readme) for full documentation.

## New Domain Modules

### Pollutant IAQ

Evaluate PM2.5, PM10, TVOC, formaldehyde, and CO against health-based thresholds:

```python
from comfio import evaluate_iaq_pollutants

pollutant = evaluate_iaq_pollutants(
    pm25=np.array([8.0, 12.0, 35.0]),
    tvoc=np.array([150.0, 300.0, 500.0]),
    formaldehyde=np.array([20.0, 27.0, 50.0]),
    co=np.array([1.5, 5.0, 10.0]),
    threshold_level="good",
)
print(f"Pollutant IAQ score: {pollutant.score}")
```

### Adaptive Thermal Comfort

```python
from comfio import evaluate_adaptive_ashrae, evaluate_adaptive_en

# ASHRAE 55-2023 (naturally ventilated buildings)
ashrae = evaluate_adaptive_ashrae(
    tdb=np.array([24.0, 25.0, 26.0]),
    tr=np.array([24.0, 25.0, 26.0]),
    t_prevail=20.0,       # prevailing mean outdoor temp
    acceptability=80,
)

# EN 16798-1:2019
en = evaluate_adaptive_en(
    tdb=np.array([24.0, 25.0, 26.0]),
    tr=np.array([24.0, 25.0, 26.0]),
    t_running_mean=20.0,
    category="ii",
)
```

### Simplified PMV (sPMV)

```python
from comfio import evaluate_spmv

spmv = evaluate_spmv(
    indoor_temp=np.array([23.0, 24.0, 25.0]),
    indoor_rh=np.array([50.0, 50.0, 50.0]),
    season="mid",  # or "winter" / "summer"
)
print(f"sPMV: {spmv.spmv}, score: {spmv.score}")
```

### TSV Augmentation & Evaluation

```python
from comfio import augment_tsv_cdf, evaluate_tsv

# Augment sparse occupant votes to dense sensor timestamps
augmented = augment_tsv_cdf(
    sparse_votes=np.array([-2, -1, 0, 0, 1, 1, 2, -1, 0, 1]),
    vote_timestamps=np.arange(10),
    target_timestamps=np.arange(100),  # dense sensor timestamps
)

# Evaluate TSV for compliance (ASHRAE 55-2023 Appendix L)
tsv_result = evaluate_tsv(augmented)
print(f"Mean TSV: {tsv_result.mean_tsv}")
print(f"Compliance rate: {tsv_result.compliance_rate:.1%}")
```

### Personalised Thermal Comfort

```python
from comfio import train_personalisation, evaluate_personalised_pmv

# Train: fit OLS regression TSV = alpha * PMV + beta
index = train_personalisation(
    pmv=historical_pmv_array,
    tsv=historical_tsv_array,
)

# Apply: personalise future PMV predictions
result = evaluate_personalised_pmv(
    tdb=tdb, tr=tr, vr=vr, rh=rh, met=1.2, clo=0.5,
    personalisation_index=index,
)
print(f"Personalised PMV: {result.personalised_pmv}")
```

### Integration with Global IEQ Index

```python
from comfio import calculate_global_ieq

ieq = calculate_global_ieq(
    thermal=thermal_res,
    visual=visual_res,
    acoustic=acoustic_res,
    iaq=iaq_res,
    pollutant_iaq=pollutant_res,  # blends 50/50 with IAQ score
    tsv=tsv_res,                  # overrides thermal score (occupant feedback is ground truth)
)
```

## Architecture

comfio operates on a **4-layer data flow**:

```text
Layer 1: Data Ingestion (SensorData)
    ↓ Pandas/NumPy time-series arrays
Layer 2: Single-Domain Modules (domains/)
    ↓ Thermal (pythermalcomfort) | Visual (EN 12464-1) | Acoustic (NC) | IAQ (ASHRAE 62.1)
Layer 3: Multi-Domain Integration (integration/)
    ↓ Global IEQ Index (0-100) with configurable weighting
Layer 4: Application & Contracts (performance/)
    → Compliance rates, JSON reports, smart contract ABI schemas
```

**Key design principle — Decoupling**: `integration/` only talks to `domains/`, never to `pythermalcomfort` directly. If pythermalcomfort releases a breaking change, only `domains/thermal.py` needs updating.

## Weighting Presets

| Preset | Thermal | IAQ | Visual | Acoustic | Use Case |
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

## Standards Referenced

- **ISO 7730**: Thermal comfort — PMV/PPD calculation
- **ASHRAE 55**: Thermal environmental conditions for human occupancy
- **ASHRAE 55-2023 Appendix L**: TSV compliance threshold (|TSV| ≤ 1.5)
- **EN 16798-1:2019**: Adaptive thermal comfort for naturally ventilated buildings
- **EN 12464-1**: Light and lighting — lighting of work places
- **ASHRAE 62.1**: Ventilation for acceptable indoor air quality
- **WHO Air Quality Guidelines (2021)**: PM2.5, PM10 thresholds
- **EPA NAAQS**: Criteria pollutant thresholds
- **WELL Building Standard v2**: Feature A01 pollutant thresholds

## Academic Attribution

comfio utilizes the validated [pythermalcomfort](https://github.com/pythermalcomfort/pythermalcomfort) library as its core engine for thermal metrics, while focusing its novel architecture on multi-domain integration and temporal performance evaluation.

> Tartarini, F., Schiavon, S., 2020. pythermalcomfort: A Python package for thermal comfort research. SoftwareX 12, 100578. <https://doi.org/10.1016/j.softx.2020.100578>

## Development

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Lint
ruff check src/ tests/
ruff format src/ tests/

# Type check
mypy src/comfio/

# Build
python -m build
```

## Citation

A formal academic paper for comfio is in preparation. In the meantime, if you use comfio in your research, please cite it as:

```bibtex
@software{comfio,
  author       = {ABO ALZAHAB, Nibras},
  title        = {comfio: A Multi-Domain IEQ \& Performance Contract Framework for Smart Buildings},
  year         = {2026},
  url          = {https://github.com/NibrasAz7/comfio},
  version      = {0.1.5},
}
```

### Per-function citation

comfio wraps and implements methods from multiple peer-reviewed sources. When you use a specific function, **please also cite the underlying work**:

| comfio function | Underlying method | Reference |
| ----------------- | ------------------- | ----------- |
| `evaluate_thermal` | Fanger PMV/PPD | Fanger (1970); ISO 7730:2005; [pythermalcomfort](https://github.com/pythermalcomfort/pythermalcomfort) — Tartarini & Schiavon (2020) |
| `evaluate_spmv` | Simplified PMV | Buratti, Ricciardi & Vergoni (2009), *Building and Environment* 44(3), 441–449 |
| `evaluate_adaptive_ashrae` | ASHRAE 55 adaptive | de Dear & Brager (1998), ASHRAE RP-884; ASHRAE 55-2023 Appendix L |
| `evaluate_adaptive_en` | EN 16798-1 adaptive | Nicol & Humphreys (2010), *Energy and Buildings* 42(10), 1793–1801; EN 16798-1:2019 |
| `evaluate_visual` | Illuminance & UGR | EN 12464-1:2021; CIE 117-1995 (UGR) |
| `evaluate_acoustic` | Noise Criteria | Beranek (1957), *Noise Control* 3(1), 19–27; ASHRAE Handbook — HVAC Applications |
| `evaluate_iaq` | CO₂ ventilation indicator | ASHRAE 62.1-2022; Persily (2015), *Building and Environment* 91, 61–69 |
| `evaluate_pollutant_iaq` | PM2.5/TVOC/HCHO/CO | WHO (2021), *Global air quality guidelines*; WHO (2010), *IAQ selected pollutants* |
| `evaluate_color_quality` | CRI / CCT / D_uv | CIE 13.3-1995; CIE 015:2018 |
| `evaluate_reverberation` | RT60 (Sabine/Eyring) | Sabine (1922); Eyring (1930), *JASA* 1(2A), 217–241; ISO 3382-2:2008 |
| `evaluate_speech_intelligibility` | STI | Houtgast & Steeneken (1971), *Acustica* 25, 355–367; IEC 60268-16:2020 |
| `evaluate_ventilation` | CO₂ decay method | ASHRAE 62.1-2022; ASTM D7297-14 |
| `get_psychrometrics` | Psychrometric properties | Hyland & Wexler (1983), *ASHRAE Transactions* 89(2A); [PsychroLib](https://github.com/psychrometrics/psychrolib) |
| `calculate_global_ieq` | Weighted IEQ index | Pierson et al. (2019), *Building and Environment* 150, 230–239 |
| `train_personalisation` | OLS personalisation | Schweiker et al. (2020), *Building and Environment* 176, 106834 |
| `augment_tsv_cdf` | CDF remapping | ASHRAE 55-2023 Appendix L |

**BibTeX entries for the most commonly cited works:**

```bibtex
@book{fanger1970,
  author    = {Fanger, Povl Ole},
  title     = {Thermal Comfort},
  publisher = {Danish Technical Press},
  year      = {1970},
}

@article{tartarini2020,
  author  = {Tartarini, Federico and Schiavon, Stefano},
  title   = {pythermalcomfort: A Python package for thermal comfort research},
  journal = {SoftwareX},
  volume  = {12},
  pages   = {100578},
  year    = {2020},
  doi     = {10.1016/j.softx.2020.100578},
}

@article{buratti2009,
  author  = {Buratti, C. and Ricciardi, P. and Vergoni, M.},
  title   = {Simplified PMV model for HVAC systems control},
  journal = {Building and Environment},
  volume  = {44},
  number  = {3},
  pages   = {441--449},
  year    = {2009},
}

@article{dedear1998,
  author  = {de Dear, R. and Brager, G. S.},
  title   = {Developing an adaptive model of thermal comfort and preference},
  journal = {ASHRAE Transactions},
  volume  = {104},
  number  = {1},
  year    = {1998},
}

@article{pierson2019,
  author  = {Pierson, A. and others},
  title   = {Indoor environmental quality: Development of a weight-based scoring system},
  journal = {Building and Environment},
  volume  = {150},
  pages   = {230--239},
  year    = {2019},
}

@misc{who2021,
  author       = {{World Health Organization}},
  title        = {WHO global air quality guidelines},
  year         = {2021},
  url          = {https://www.who.int/publications/i/item/9789240034221},
}
```

## License

MIT — see [LICENSE](LICENSE).
