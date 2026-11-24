# API Reference

Complete public API for comfio v0.1.2. All functions are importable from the top-level
`comfio` package unless noted otherwise.

## Core

### SensorData

```python
from comfio import SensorData

SensorData(
    df: pd.DataFrame,
    column_map: dict[str, str] = {},    # optional manual column mapping
    timestamp_col: str | None = None,
)
```

**Methods:**
- `validate(nan_strategy="interpolate")` — Validate all columns against physical bounds
- `get_column(canonical_name) -> np.ndarray` — Raw array for a canonical variable
- `get_validated(canonical_name, nan_strategy="interpolate") -> np.ndarray` — Validated array
- `available_domains() -> list[str]` — Subset of `["thermal", "visual", "acoustic", "iaq"]`
- `available_advanced_domains() -> list[str]` — Advanced capability keys
- `get_timestamps() -> pd.Series | None`

**Canonical column names:** `air_temp_c`, `radiant_temp_c`, `relative_humidity_pct`,
`air_velocity_ms`, `illuminance_lux`, `co2_ppm`, `noise_laeq_db`, `metabolic_rate_met`,
`clothing_insulation_clo`, `pm25_ugm3`, `pm10_ugm3`, `tvoc_ugm3`, `formaldehyde_ppb`,
`co_ppm`, `outdoor_temp_c`, `prevailing_mean_outdoor_c`, `running_mean_outdoor_c`,
`spectral_power`, `impulse_response`, `room_volume_m3`, `n_occupants`,
`atmospheric_pressure_pa`

**Auto-detected aliases:** `tdb`, `ta`, `temperature` → `air_temp_c`; `tr` → `radiant_temp_c`;
`rh`, `humidity` → `relative_humidity_pct`; `vr`, `v`, `air_speed` → `air_velocity_ms`;
`lux`, `illuminance` → `illuminance_lux`; `co2` → `co2_ppm`; `laeq`, `noise`, `spl` →
`noise_laeq_db`; `met` → `metabolic_rate_met`; `clo` → `clothing_insulation_clo`

### Exceptions

```python
from comfio import MissingSensorDataError, OutOfRangeError, InvalidUnitError
```

### Validation

```python
from comfio import validate_input_array

validate_input_array(arr, name="array") -> np.ndarray
```

## Domain Modules

### Thermal (ISO 7730 / ASHRAE 55)

```python
from comfio import evaluate_thermal

evaluate_thermal(
    tdb: np.ndarray,        # air temperature °C
    tr: np.ndarray,         # radiant temperature °C
    vr: np.ndarray,         # air velocity m/s
    rh: np.ndarray,         # relative humidity %
    met: float = 1.0,       # metabolic rate
    clo: float = 0.5,       # clothing insulation
    category: str = "B",    # "A", "B", "C", or "none"
    standard: str = "7730-2005",
) -> ThermalResult
```

**ThermalResult:** `.pmv` (np.ndarray -3..+3), `.ppd` (np.ndarray 0-100),
`.compliant` (bool array), `.category` (str), `.score` (np.ndarray 0-100)

### Visual (EN 12464-1)

```python
from comfio import evaluate_visual

evaluate_visual(
    illuminance: np.ndarray,     # lux
    task_type: str = "general",  # "office_writing", "office_reading", "general", etc.
) -> VisualResult
```

**VisualResult:** `.illuminance`, `.score` (0-100), `.compliant` (bool array)

### Acoustic (NC curves)

```python
from comfio import evaluate_acoustic

evaluate_acoustic(
    laeq: np.ndarray,            # A-weighted equivalent sound level dB
    nc_level: str = "NC-35",     # "NC-25" through "NC-65"
) -> AcousticResult
```

**AcousticResult:** `.laeq`, `.score` (0-100), `.compliant` (bool array)

### IAQ (ASHRAE 62.1)

```python
from comfio import evaluate_iaq

evaluate_iaq(
    co2: np.ndarray,               # CO₂ concentration ppm
    threshold_level: str = "good", # "good" (≤1000), "moderate" (≤1400), "poor"
) -> IAQResult
```

**IAQResult:** `.co2`, `.score` (0-100), `.compliant` (bool array)

### Pollutant IAQ (WHO / EPA / WELL v2)

```python
from comfio import evaluate_iaq_pollutants

evaluate_iaq_pollutants(
    pm25: np.ndarray | None = None,        # µg/m³
    pm10: np.ndarray | None = None,        # µg/m³
    tvoc: np.ndarray | None = None,        # µg/m³
    formaldehyde: np.ndarray | None = None, # ppb
    co: np.ndarray | None = None,          # ppm
    threshold_level: str = "good",
) -> PollutantIAQResult
```

### Adaptive Thermal (ASHRAE 55-2023)

```python
from comfio import evaluate_adaptive_ashrae

evaluate_adaptive_ashrae(
    tdb: np.ndarray,
    tr: np.ndarray,
    t_prevail: float,         # prevailing mean outdoor temp °C
    acceptability: int = 80,  # 80 or 90
) -> AdaptiveThermalResult
```

### Adaptive Thermal (EN 16798-1:2019)

```python
from comfio import evaluate_adaptive_en

evaluate_adaptive_en(
    tdb: np.ndarray,
    tr: np.ndarray,
    t_running_mean: float,    # running mean outdoor temp °C
    category: str = "ii",     # "i", "ii", "iii", "iv"
) -> AdaptiveThermalResult
```

### Simplified PMV (Buratti et al. 2009)

```python
from comfio import evaluate_spmv

evaluate_spmv(
    indoor_temp: np.ndarray,    # °C
    indoor_rh: np.ndarray,      # %
    season: str = "mid",        # "winter", "mid", "summer"
) -> SPMVResult
```

### TSV Augmentation & Evaluation

```python
from comfio import augment_tsv_cdf, evaluate_tsv

augment_tsv_cdf(
    sparse_votes: np.ndarray,       # TSV values (-3..+3)
    vote_timestamps: np.ndarray,    # timestamps of votes
    target_timestamps: np.ndarray,  # dense sensor timestamps
) -> np.ndarray                     # augmented TSV at sensor resolution

evaluate_tsv(tsv_array: np.ndarray) -> TSVResult
```

**TSVResult:** `.tsv`, `.mean_tsv`, `.compliance_rate` (0-1), `.score` (0-100)

### Personalised Comfort

```python
from comfio import train_personalisation, evaluate_personalised_pmv

train_personalisation(
    pmv: np.ndarray,    # historical PMV predictions
    tsv: np.ndarray,    # historical occupant TSV votes
) -> PersonalisationIndex

evaluate_personalised_pmv(
    tdb, tr, vr, rh, met, clo,
    personalisation_index: PersonalisationIndex,
) -> PersonalisedPMVResult
```

Also: `train_seasonal_personalisation()`, `evaluate_personalised_spmv()`,
`evaluate_personalised_adaptive()`

### Advanced Modules (require optional extras)

```python
# [acoustics] extra
from comfio import evaluate_reverberation, evaluate_speech_intelligibility
evaluate_reverberation(surfaces, absorption, volume, room_type="office")
evaluate_speech_intelligibility(ir_signal, sample_rate=16000)

# [psychrometrics] extra
from comfio import evaluate_ventilation, get_psychrometrics
evaluate_ventilation(co2, timestamps, occupancy_type="office")
get_psychrometrics(tdb=25.0, rh=0.50)

# [daylighting] extra
from comfio import evaluate_daylighting

# [color] extra
from comfio import evaluate_color_quality
```

## Integration

### Global IEQ Index

```python
from comfio import calculate_global_ieq, default_weights, WeightSchema

calculate_global_ieq(
    thermal: ThermalResult | None = None,
    visual: VisualResult | None = None,
    acoustic: AcousticResult | None = None,
    iaq: IAQResult | None = None,
    weights: WeightSchema | None = None,
    daylighting: DaylightingResult | None = None,
    color_quality: ColorQualityResult | None = None,
    reverberation: ReverberationResult | None = None,
    speech_intelligibility: SpeechIntelligibilityResult | None = None,
    ventilation: VentilationResult | None = None,
    pollutant_iaq: PollutantIAQResult | None = None,
    tsv: TSVResult | None = None,
) -> GlobalIEQResult
```

**GlobalIEQResult:** `.index` (np.ndarray 0-100), `.domain_scores` (dict),
`.weights_used` (dict), `.domains` (list[str]), `.n_timestamps` (int)

### Weighting

```python
from comfio.integration.weights import preset_weights, custom_weights, default_weights

default_weights() -> WeightSchema
preset_weights("office") -> WeightSchema
custom_weights(thermal=0.5, visual=0.2, acoustic=0.1, iaq=0.2) -> WeightSchema
```

## Performance & Contracts

```python
from comfio import calculate_compliance, ComplianceReport

calculate_compliance(
    ieq_result: GlobalIEQResult,
    threshold: float = 80.0,
    period_start: float | None = None,    # Unix timestamp
    period_end: float | None = None,
    domain_compliant_arrays: dict | None = None,
) -> ComplianceReport
```

**ComplianceReport fields:** `period_start`, `period_end`, `ieq_index_avg`, `ieq_index_min`,
`ieq_index_max`, `ieq_index_std`, `compliance_rate_pct`, `threshold`, `total_hours`,
`compliant_hours`, `domain_compliance` (dict), `domain_scores_avg` (dict),
`weights_used` (dict), `domains` (list)

**Methods:**
- `.to_json(indent=2) -> str` — Full JSON serialization
- `.to_contract_payload(schema=None) -> dict` — Solidity ABI-compatible dict
- `.to_contract_json(schema=None) -> str` — JSON string for blockchain Oracle

## LLM Adapters

```python
from comfio import (
    ieq_to_markdown, ieq_to_summary_dict, generate_markdown_summary,
    EDGE_SYSTEM_PROMPT, DIAGNOSTIC_PROMPT_TEMPLATE, format_prompt,
)

ieq_to_markdown(ieq_result, compliance_report=None, zone_id=None) -> str
ieq_to_summary_dict(ieq_result, compliance_report=None) -> dict
generate_markdown_summary(df, window_hours=24, threshold=80.0, zone_id=None) -> str
format_prompt(template, **kwargs) -> str
```

**[agent] extra:**
```python
from comfio.llm.tools import to_openai_tools, to_langchain_tools
to_openai_tools() -> list[dict]
to_langchain_tools() -> list
```

## ML/DL Adapters

```python
# [ml] extra
from comfio.ml.sklearn_transformers import IEQFeatureExtractor

# [torch] extra
from comfio.ml.torch_dataset import IEQTimeSeriesDataset

# [keras] extra
from comfio.ml.keras_adapter import IEQPreprocessingLayer
```
