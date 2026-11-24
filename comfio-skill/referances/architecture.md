# Architecture

comfio operates on a **4-layer data flow** architecture designed for maintainability
through strict decoupling.

## Layer Overview

```
Layer 1: Data Ingestion (core/)
    ↓ Pandas/NumPy time-series arrays
Layer 2: Single-Domain Modules (domains/)
    ↓ Thermal | Visual | Acoustic | IAQ | Pollutant | Adaptive | sPMV | TSV
Layer 3: Multi-Domain Integration (integration/)
    ↓ Global IEQ Index (0-100) with configurable weighting
Layer 4: Application & Contracts (performance/)
    → Compliance rates, JSON reports, Solidity ABI schemas
```

## Module Map

| Directory | Purpose | Key Files |
|---|---|---|
| `src/comfio/core/` | Data ingestion & validation | `data_handler.py`, `exceptions.py` |
| `src/comfio/domains/` | Single-domain evaluation | `thermal.py`, `visual.py`, `acoustic.py`, `iaq.py`, `iaq_pollutants.py`, `thermal_adaptive.py`, `thermal_spmv.py`, `thermal_tsv.py`, `thermal_personal.py`, `acoustic_advanced.py`, `visual_advanced.py`, `iaq_advanced.py` |
| `src/comfio/integration/` | Multi-domain aggregation | `global_ieq.py`, `weights.py` |
| `src/comfio/performance/` | Compliance & contracts | `contracts.py`, `contract_schema.py` |
| `src/comfio/llm/` | LLM-native adapters | `interpreters.py`, `prompts.py`, `tools.py` |
| `src/comfio/ml/` | ML/DL adapters | `sklearn_transformers.py`, `torch_dataset.py`, `keras_adapter.py` |
| `src/comfio/utils/` | Validation utilities | `validation.py`, `units.py` |

## Decoupling Principle

The integration layer (`integration/global_ieq.py`) **only** talks to domain score
functions and result dataclasses. It never imports `pythermalcomfort` or other external
libraries directly.

**Why this matters:**
- If `pythermalcomfort` releases a breaking change → only `domains/thermal.py` needs updating
- If a new standard replaces EN 12464-1 → only `domains/visual.py` changes
- The Global IEQ Index calculation remains stable across dependency upgrades

## Optional Dependency Handling

Advanced modules use **lazy imports** — heavy dependencies are imported inside functions,
not at module level. This means:

- `import comfio` works with only numpy, pandas, pythermalcomfort installed
- Advanced functions raise `ImportError` with a helpful message when the extra is missing
- The result dataclasses (e.g., `ReverberationResult`) are always importable for type hints

**Extras and their dependencies:**

| Extra | Packages | Modules Unlocked |
|---|---|---|
| `[ml]` | scikit-learn | `ml.sklearn_transformers` |
| `[torch]` | torch | `ml.torch_dataset` |
| `[keras]` | tensorflow | `ml.keras_adapter` |
| `[agent]` | pydantic | `llm.tools` |
| `[daylighting]` | pyradiance | `domains.visual_advanced` (daylighting) |
| `[color]` | colour-science | `domains.visual_advanced` (color quality) |
| `[acoustics]` | python-acoustics, pyroomacoustics | `domains.acoustic_advanced` |
| `[psychrometrics]` | psychrolib | `domains.iaq_advanced` |

## Data Flow Example

```
User provides CSV with columns: tdb, rh, lux, co2, laeq
    ↓
SensorData(df) auto-detects → column_map
    ↓
sensor.validate() → physical bounds check, NaN handling
    ↓
sensor.available_domains() → ['thermal', 'visual', 'acoustic', 'iaq']
    ↓
evaluate_thermal(sensor.get_validated("air_temp_c"), ...) → ThermalResult
evaluate_visual(sensor.get_validated("illuminance_lux"), ...) → VisualResult
evaluate_acoustic(sensor.get_validated("noise_laeq_db"), ...) → AcousticResult
evaluate_iaq(sensor.get_validated("co2_ppm"), ...) → IAQResult
    ↓
calculate_global_ieq(thermal=..., visual=..., acoustic=..., iaq=...) → GlobalIEQResult
    ↓
calculate_compliance(ieq, threshold=80.0) → ComplianceReport
    ↓
report.to_contract_json() → Solidity ABI JSON string
```

## Key Design Decisions

1. **NumPy-native**: All domain functions accept and return `np.ndarray` for vectorized
   time-series processing. No Python loops over timestamps.

2. **Result dataclasses**: Each domain returns a typed `@dataclass` (e.g., `ThermalResult`)
   with `.score` (0-100) for integration. This is the contract between Layer 2 and Layer 3.

3. **Weight renormalization**: `WeightSchema.get_normalized()` redistributes weights when
   domains are missing. No need to manually adjust weights for partial sensor setups.

4. **TSV as ground truth**: When both `thermal` and `tsv` are provided to
   `calculate_global_ieq`, TSV overrides the thermal score. Occupant feedback always wins.

5. **50/50 pollutant blend**: When both CO₂-based IAQ and pollutant IAQ are provided,
   scores blend 50/50. If only pollutant IAQ is provided, it maps to the "iaq" domain.
