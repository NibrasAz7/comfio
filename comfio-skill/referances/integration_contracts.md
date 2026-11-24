# Integration & Contracts

How comfio merges domain scores into a Global IEQ Index and generates
blockchain-compatible compliance reports.

## Global IEQ Index

```python
from comfio import calculate_global_ieq, default_weights

ieq = calculate_global_ieq(
    thermal=thermal_result,       # ThermalResult | None
    visual=visual_result,         # VisualResult | None
    acoustic=acoustic_result,     # AcousticResult | None
    iaq=iaq_result,               # IAQResult | None
    weights=None,                 # WeightSchema (default if None)
    # Advanced overrides/blends:
    daylighting=None,             # DaylightingResult
    color_quality=None,           # ColorQualityResult
    reverberation=None,           # ReverberationResult
    speech_intelligibility=None,  # SpeechIntelligibilityResult
    ventilation=None,             # VentilationResult
    pollutant_iaq=None,           # PollutantIAQResult
    tsv=None,                     # TSVResult
) -> GlobalIEQResult
```

### Formula

```
IEQ = Σ(w'_d × s_d)   for d in available domains

where w'_d = w_d / Σ(w_d for available domains)
```

### Blending Rules

| Scenario | Rule |
|---|---|
| TSV + thermal both provided | TSV **overrides** thermal score (occupant = ground truth) |
| TSV only (no thermal) | TSV maps to "thermal" domain |
| Pollutant IAQ + CO₂ IAQ both | **50/50 blend** of scores |
| Pollutant IAQ only | Maps to "iaq" domain |
| Daylighting provided | **Overrides** simple visual score |
| Color quality provided | **70/30 blend**: 70% visual/daylighting + 30% color |
| Reverberation provided | **60/40 blend**: 60% noise + 40% reverberation |
| Speech intelligibility provided | **50/50 blend**: 50% current acoustic + 50% STI |
| Ventilation provided | **60/40 blend**: 60% CO₂ + 40% ventilation |

### Missing Domains

Weights are **renormalized** automatically. If acoustic sensor is missing, its 15% weight
is redistributed proportionally across thermal, visual, and IAQ. No manual adjustment needed.

---

## Weighting Schemas

### Presets

```python
from comfio.integration.weights import preset_weights, default_weights

weights = default_weights()          # thermal=40%, iaq=25%, visual=20%, acoustic=15%
weights = preset_weights("equal")    # 25% each
weights = preset_weights("school")   # thermal=27%, iaq=26%, visual=24%, acoustic=23%
weights = preset_weights("office")   # thermal=45%, iaq=30%, visual=15%, acoustic=10%
weights = preset_weights("healthcare") # thermal=25%, iaq=40%, visual=15%, acoustic=20%
```

### Custom Weights

```python
from comfio.integration.weights import custom_weights

weights = custom_weights(
    thermal=0.50,
    visual=0.20,
    acoustic=0.10,
    iaq=0.20,
)
# Must sum to 1.0 (±0.01 tolerance)
```

### WeightSchema Methods

- `.weights` — dict of domain → weight
- `.preset_name` — str ("default", "equal", "school", "office", "healthcare", "custom")
- `.get_normalized(available_domains) -> dict` — Renormalized for available domains
- `.get_array(available_domains) -> np.ndarray` — Weights as numpy array

---

## Compliance Reports

```python
from comfio import calculate_compliance

report = calculate_compliance(
    ieq_result=ieq,
    threshold=80.0,              # IEQ Index threshold for compliance
    period_start=None,           # Unix timestamp (default: now - duration)
    period_end=None,             # Unix timestamp (default: now)
    domain_compliant_arrays=None, # Optional per-domain boolean arrays
) -> ComplianceReport
```

### ComplianceReport Fields

| Field | Type | Description |
|---|---|---|
| `period_start` | float | Unix timestamp of period start |
| `period_end` | float | Unix timestamp of period end |
| `ieq_index_avg` | float | Average Global IEQ Index (0-100) |
| `ieq_index_min` | float | Minimum IEQ Index |
| `ieq_index_max` | float | Maximum IEQ Index |
| `ieq_index_std` | float | Standard deviation |
| `compliance_rate_pct` | float | % of timestamps ≥ threshold |
| `threshold` | float | IEQ threshold used |
| `total_hours` | float | Total hours in period |
| `compliant_hours` | float | Hours meeting threshold |
| `domain_compliance` | dict | Per-domain compliance rates (%) |
| `domain_scores_avg` | dict | Per-domain average scores (0-100) |
| `weights_used` | dict | Normalized weights applied |
| `domains` | list | Domains included |

### Compliance Rate Formula

```
compliance_rate = 100 × (1/n) × Σ 𝟙(IEQ_i ≥ threshold)
```

### Export Methods

```python
report.to_json(indent=2)                    # Full JSON string
report.to_contract_payload(schema=None)     # Solidity ABI dict
report.to_contract_json(schema=None)        # Solidity ABI JSON string
```

---

## Solidity ABI Schema

The default contract schema maps to `IEQComplianceOracle.submitCompliance`:

| Solidity Field | Type | Source |
|---|---|---|
| `periodStart` | uint256 | report.period_start |
| `periodEnd` | uint256 | report.period_end |
| `ieqIndexAvg` | uint8 | report.ieq_index_avg (rounded) |
| `complianceRatePct` | uint8 | report.compliance_rate_pct (rounded) |
| `thermalCompliant` | bool | domain_compliance.thermal ≥ 80 |
| `visualCompliant` | bool | domain_compliance.visual ≥ 80 |
| `acousticCompliant` | bool | domain_compliance.acoustic ≥ 80 |
| `iaqCompliant` | bool | domain_compliance.iaq ≥ 80 |
| `totalOccupiedHours` | uint32 | report.total_hours |
| `compliantHours` | uint32 | report.compliant_hours |

### Custom Schema

```python
from comfio.performance.contract_schema import ContractSchema, ContractField, default_compliance_schema

schema = default_compliance_schema()
print(schema.to_abi())   # ABI fragment dict
print(schema.to_dict())  # Human-readable schema

# Build custom schema
custom = ContractSchema(
    contract_name="MyOracle",
    function_name="submitReport",
    fields=[
        ContractField("ieqAvg", "uint8", "IEQ average", "report.ieq_index_avg"),
        ContractField("isCompliant", "bool", "Compliance flag", "report.compliance_rate_pct >= 80"),
    ],
)
report.to_contract_json(schema=custom)
```
