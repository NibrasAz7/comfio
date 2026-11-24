# Domain Modules

Detailed reference for each IEQ domain evaluation function in comfio.

## Table of Contents

1. [Thermal (PMV/PPD)](#thermal-pmvppd)
2. [Visual (Illuminance)](#visual-illuminance)
3. [Acoustic (NC Curves)](#acoustic-nc-curves)
4. [IAQ (CO₂)](#iaq-co₂)
5. [Pollutant IAQ](#pollutant-iaq)
6. [Adaptive Thermal — ASHRAE 55](#adaptive-thermal--ashrae-55)
7. [Adaptive Thermal — EN 16798](#adaptive-thermal--en-16798)
8. [Simplified PMV (sPMV)](#simplified-pmv-spmv)
9. [TSV Augmentation & Evaluation](#tsv-augmentation--evaluation)
10. [Personalised Comfort](#personalised-comfort)
11. [Advanced: Reverberation (RT60)](#advanced-reverberation-rt60)
12. [Advanced: Speech Intelligibility (STI)](#advanced-speech-intelligibility-sti)
13. [Advanced: Ventilation (CO₂ Decay)](#advanced-ventilation-co₂-decay)
14. [Advanced: Psychrometrics](#advanced-psychrometrics)
15. [Advanced: Daylighting](#advanced-daylighting)
16. [Advanced: Color Quality](#advanced-color-quality)

---

## Thermal (PMV/PPD)

**Standard:** ISO 7730 / ASHRAE 55
**Engine:** `pythermalcomfort` (lazy-imported)

```python
evaluate_thermal(
    tdb: np.ndarray,    # air temperature °C
    tr: np.ndarray,     # mean radiant temperature °C
    vr: np.ndarray,     # air velocity m/s
    rh: np.ndarray,     # relative humidity %
    met: float = 1.0,   # metabolic rate (1.0 = resting, 1.2 = office, 1.4 = standing)
    clo: float = 0.5,   # clothing insulation (0.5 = summer, 1.0 = winter)
    category: str = "B", # "A" (PPD≤6%), "B" (PPD≤10%), "C" (PPD≤15%), "none"
) -> ThermalResult
```

**Returns:** `ThermalResult` with `.pmv` (-3..+3), `.ppd` (0-100%), `.compliant` (bool),
`.score` (0-100, derived from PPD: `100 - ppd`)

**Category limits:**
| Category | PMV range | PPD limit |
|---|---|---|
| A | ±0.2 | ≤6% |
| B | ±0.5 | ≤10% |
| C | ±0.7 | ≤15% |

---

## Visual (Illuminance)

**Standard:** EN 12464-1

```python
evaluate_visual(
    illuminance: np.ndarray,      # lux
    task_type: str = "general",   # see task targets below
) -> VisualResult
```

**Task type targets (lux):**
| Task type | Target lux |
|---|---|
| `office_writing` | 500 |
| `office_reading` | 500 |
| `general` | 300 |
| `corridor` | 100 |
| `meeting` | 500 |

**Score logic:** Linear interpolation — 100 at target, 0 at 0 lux. Over-illumination
penalized above 2× target.

---

## Acoustic (NC Curves)

**Standard:** NC (Noise Criterion) curves

```python
evaluate_acoustic(
    laeq: np.ndarray,           # A-weighted equivalent sound level dB
    nc_level: str = "NC-35",    # "NC-25" through "NC-65"
) -> AcousticResult
```

**NC thresholds (dB):**
| Rating | Max dB |
|---|---|
| NC-25 | 25 |
| NC-30 | 30 |
| NC-35 | 35 |
| NC-40 | 40 |
| NC-45 | 45 |
| NC-50 | 50 |
| NC-55 | 55 |
| NC-60 | 60 |
| NC-65 | 65 |

**Score logic:** 100 at 0 dB above threshold, decreasing linearly. 0 score at +15 dB over.

---

## IAQ (CO₂)

**Standard:** ASHRAE 62.1

```python
evaluate_iaq(
    co2: np.ndarray,               # CO₂ concentration ppm
    threshold_level: str = "good", # "good", "moderate", "poor"
) -> IAQResult
```

**Threshold levels:**
| Level | CO₂ limit |
|---|---|
| good | ≤1000 ppm |
| moderate | ≤1400 ppm |
| poor | >1400 ppm |

**Score logic:** 100 at 400 ppm (ambient), linear decrease to 0 at 2000 ppm.

---

## Pollutant IAQ

**Standards:** WHO 2021, EPA NAAQS, WELL Building Standard v2

```python
evaluate_iaq_pollutants(
    pm25: np.ndarray | None = None,         # µg/m³
    pm10: np.ndarray | None = None,         # µg/m³
    tvoc: np.ndarray | None = None,         # µg/m³
    formaldehyde: np.ndarray | None = None,  # ppb
    co: np.ndarray | None = None,           # ppm
    threshold_level: str = "good",
) -> PollutantIAQResult
```

**WHO thresholds (good):** PM2.5 ≤15 µg/m³ (24h), PM10 ≤45 µg/m³ (24h),
TVOC ≤300 µg/m³, formaldehyde ≤27 ppb, CO ≤4 ppm (8h)

**Blending in Global IEQ:** When both `iaq` and `pollutant_iaq` are passed to
`calculate_global_ieq`, scores blend 50/50. If only `pollutant_iaq` is provided,
it maps to the "iaq" domain slot.

---

## Adaptive Thermal — ASHRAE 55

**Standard:** ASHRAE 55-2023 (naturally ventilated buildings)

```python
evaluate_adaptive_ashrae(
    tdb: np.ndarray,        # indoor air temp °C
    tr: np.ndarray,         # mean radiant temp °C
    t_prevail: float,       # prevailing mean outdoor temp °C
    acceptability: int = 80, # 80% or 90% acceptability
) -> AdaptiveThermalResult
```

**Comfort band:** Computed from prevailing mean outdoor temperature per ASHRAE 55-2023
Table 5.3.1.1. Operative temp = (tdb + tr) / 2.

---

## Adaptive Thermal — EN 16798

**Standard:** EN 16798-1:2019

```python
evaluate_adaptive_en(
    tdb: np.ndarray,
    tr: np.ndarray,
    t_running_mean: float,  # running mean outdoor temp °C
    category: str = "ii",   # "i", "ii", "iii", "iv"
) -> AdaptiveThermalResult
```

**Category bands:** Based on running mean outdoor temperature per EN 16798-1 Table B.1.

---

## Simplified PMV (sPMV)

**Source:** Buratti et al. (2009) seasonal model

```python
evaluate_spmv(
    indoor_temp: np.ndarray,   # °C
    indoor_rh: np.ndarray,     # %
    season: str = "mid",       # "winter", "mid", "summer"
) -> SPMVResult
```

**Advantage:** Requires only temperature and humidity (no met/clo/air velocity).
Uses season-specific coefficients to approximate PMV.

---

## TSV Augmentation & Evaluation

**TSV = Thermal Sensation Vote** (-3 cold .. +3 hot), occupant feedback.

### Augmentation (CDF-based quantile mapping)

```python
augment_tsv_cdf(
    sparse_votes: np.ndarray,       # e.g., [-2, -1, 0, 0, 1, 2]
    vote_timestamps: np.ndarray,    # when votes were collected
    target_timestamps: np.ndarray,  # dense sensor timestamps
) -> np.ndarray                     # TSV at sensor resolution
```

Preserves the empirical CDF of sparse votes while expanding to sensor density.

### Evaluation (ASHRAE 55-2023 Appendix L)

```python
evaluate_tsv(tsv_array: np.ndarray) -> TSVResult
```

**Compliance:** |mean TSV| ≤ 1.5 per ASHRAE 55-2023 Appendix L.

**In Global IEQ:** TSV overrides thermal score when both are provided
(occupant feedback = ground truth).

---

## Personalised Comfort

**Method:** OLS regression: TSV = α × PMV + β

```python
# Train
index = train_personalisation(pmv=historical_pmv, tsv=historical_tsv)

# Apply
result = evaluate_personalised_pmv(
    tdb=tdb, tr=tr, vr=vr, rh=rh, met=1.2, clo=0.5,
    personalisation_index=index,
)
```

**Seasonal variant:**
```python
seasonal_idx = train_seasonal_personalisation(pmv=..., tsv=..., seasons=...)
result = evaluate_personalised_spmv(
    indoor_temp=..., indoor_rh=..., season="summer",
    personalisation_index=seasonal_idx,
)
```

---

## Advanced: Reverberation (RT60)

**Extra:** `pip install comfio[acoustics]`

```python
evaluate_reverberation(
    surfaces: list,       # surface areas
    absorption: list,     # absorption coefficients
    volume: float,        # room volume m³
    room_type: str = "office",
) -> ReverberationResult
```

**Blending:** 60% noise score + 40% reverberation score in Global IEQ.

---

## Advanced: Speech Intelligibility (STI)

**Extra:** `pip install comfio[acoustics]`

```python
evaluate_speech_intelligibility(
    ir_signal: np.ndarray,    # impulse response
    sample_rate: int = 16000,
) -> SpeechIntelligibilityResult
```

**Blending:** 50% current acoustic + 50% STI score in Global IEQ.

---

## Advanced: Ventilation (CO₂ Decay)

**Extra:** `pip install comfio[psychrometrics]`

```python
evaluate_ventilation(
    co2: np.ndarray,
    timestamps: np.ndarray,
    occupancy_type: str = "office",
) -> VentilationResult
```

**Blending:** 60% CO₂ score + 40% ventilation score in Global IEQ.

---

## Advanced: Psychrometrics

**Extra:** `pip install comfio[psychrometrics]`

```python
get_psychrometrics(
    tdb: float = 25.0,    # dry-bulb temp °C
    rh: float = 0.50,     # relative humidity (0-1)
) -> PsychrometricResult
```

Returns dew point, wet bulb, enthalpy, humidity ratio, etc.

---

## Advanced: Daylighting

**Extra:** `pip install comfio[daylighting]`

```python
evaluate_daylighting(...) -> DaylightingResult
```

Uses Radiance ray-tracing. Overrides simple visual score in Global IEQ.

---

## Advanced: Color Quality

**Extra:** `pip install comfio[color]`

```python
evaluate_color_quality(...) -> ColorQualityResult
```

Computes CRI (Color Rendering Index) and CCT (Correlated Color Temperature).

**Blending:** 70% visual/daylighting + 30% color quality in Global IEQ.
