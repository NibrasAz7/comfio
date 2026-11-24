# Long-Term IEQ Evaluation (EN 16798-1, Annex B)

## Concept

Instantaneous comfort scores answer *"how is the room right now?"* —
long-term evaluation answers *"for what fraction of the occupied time
was the room acceptable?"*. EN 16798-1:2019 (which superseded
EN 15251:2007) defines, in Annex B, a representation based on the
**percentage of occupied time spent in each of the four categories**
for every IEQ domain:

| Domain | Cat. I | Cat. II | Cat. III | Cat. IV |
|---|---|---|---|---|
| Thermal Environment | 20 % | 68 % | 7 % | 5 % |
| Indoor Air Quality | 10 % | 76 % | 7 % | 7 % |

Results are reported per category **and cumulatively** — Cat. I,
Cat. I+II ("acceptable"), Cat. I+II+III, Cat. I+II+III+IV (= 100 %) —
over the whole monitored period or per day/week/month.

## Category boundaries used in comfio

### Thermal — PMV (ISO 7730 / EN 16798-1)

| Category | Criterion |
|---|---|
| I | \|PMV\| ≤ 0.2 |
| II | \|PMV\| ≤ 0.5 |
| III | \|PMV\| ≤ 0.7 |
| IV | outside Cat. III |

The same bands are applied to sPMV/sPMVopt arrays as an approximation
of the PMV-based classification (documented caveat, not a standard
provision). They are also applied to **TSV occupant votes** by
`evaluate_temporal_tsv()` — a reporting heuristic, since the bands are
defined on PMV, a model prediction, not on subjective votes.

### Thermal — adaptive model (EN 16798-1)

Classification on the absolute offset from the comfort temperature,
$|t_{op} - t_{comf}|$: Cat. I ≤ 2 K, Cat. II ≤ 3 K, Cat. III ≤ 4 K,
Cat. IV outside.

### Indoor Air Quality — CO₂ above outdoor

**Categories are defined on the excess over outdoor CO₂**, not on
absolute ppm:

| Category | CO₂ above outdoor |
|---|---|
| I | ≤ 400 ppm |
| II | ≤ 600 ppm |
| III | ≤ 1000 ppm |
| IV | > 1000 ppm |

This differs from `evaluate_iaq()`, whose thresholds are absolute-ppm
practice indicators (ASHRAE Position Document on Indoor CO₂), not EN
categories.

### Particulates & gases — WHO/EPA/WELL tiers

EN 16798-1 does **not** define categories for PM2.5, PM10, TVOC,
formaldehyde or CO. comfio maps the existing
`POLLUTANT_THRESHOLDS` tiers (WHO Air Quality Guidelines, EPA NAAQS,
WELL v2) onto the I–IV scale:

| Category | Criterion |
|---|---|
| I | concentration ≤ `excellent` threshold |
| II | ≤ `good` threshold |
| III | ≤ `moderate` threshold |
| IV | above `moderate` (the `poor` tier is a scoring anchor, not a bound) |

These are health-based bands, not a standard provision — documented
caveat. Custom bands can be passed via `thresholds=`.

### Visual and acoustic

Default category limits follow typical office requirements in the
EN 16798-1 annexes and are configurable per space/room type:

- Visual (maintained illuminance): Cat. I ≥ 500 lux, II ≥ 300 lux,
  III ≥ 200 lux, IV below.
- Acoustic (L_Aeq): Cat. I ≤ 35 dB, II ≤ 40 dB, III ≤ 45 dB, IV above.

## Aggregation rules in comfio

- **Occupancy**: an optional occupancy mask drops unoccupied samples
  from every denominator.
- **Invalid samples**: NaN labels (e.g. outside the PMV applicability
  domain) are excluded from the category percentages and reported
  separately as `excluded_pct`.
- **Time weighting**: with timestamps, each sample is weighted by its
  actual duration (interval to the next sample), so irregular sampling
  does not distort the percentages; occupied hours are reported per
  category. Without timestamps, samples have equal weight.
- **Grouping**: whole series by default, or any pandas offset alias
  (`by="D"`, `"W"`, `"MS"`, …) for per-period tables.

## References

- EN 16798-1:2019 — *Energy performance of buildings — Ventilation for
  buildings — Part 1: Indoor environmental input parameters for design
  and assessment of energy performance of buildings addressing indoor
  air quality, thermal environment, lighting and acoustics*, Annex B.
- ISO 7730:2005 — PMV/PPD category limits (Table 2).
