# Pollutant IAQ Thresholds

## Overview

comfio evaluates indoor air pollutants against health-based thresholds from three authoritative sources: WHO Air Quality Guidelines (2021), EPA NAAQS, and WELL Building Standard v2. Each pollutant is scored using a piecewise linear function that maps concentration to a 0–100 score.

---

## Pollutants Evaluated

| Pollutant | Unit | WHO (2021) | EPA NAAQS | WELL v2 |
|---|---|---|---|---|
| PM2.5 | μg/m³ | 15 (24h) | 35 (24h) | 15 (24h) |
| PM10 | μg/m³ | 45 (24h) | 150 (24h) | 50 (24h) |
| TVOC | μg/m³ | — | — | 500 (1h) |
| Formaldehyde | μg/m³ | 100 (30min) | — | 27 (1h) |
| CO | mg/m³ | 4 (24h) | 10 (8h) | — |

---

## Threshold Levels

comfio supports three threshold levels:

- **`good`** — strictest applicable standard (typically WHO or WELL)
- **`moderate`** — intermediate level
- **`permissive`** — most lenient applicable standard (typically EPA)

---

## Piecewise Scoring Function

For each pollutant with threshold $T$:

$$
\text{score}(c) = \begin{cases}
100 & \text{if } c \leq T \\
100 \cdot \left(1 - \frac{c - T}{T}\right) & \text{if } T < c \leq 2T \\
0 & \text{if } c > 2T
\end{cases}
$$

where $c$ is the measured concentration and $T$ is the threshold for the selected level.

### Overall Pollutant IAQ Score

The overall score is the minimum across all evaluated pollutants (weakest-link approach):

$$
\text{Pollutant IAQ Score} = \min_i(\text{score}_i)
$$

This ensures that a single pollutant exceeding its threshold can flag the air quality as poor, even if other pollutants are within limits.

---

## Integration with Global IEQ

When both basic IAQ (CO₂-based) and pollutant IAQ are provided, they are blended 50/50:

$$
\text{IAQ}_{\text{combined}} = 0.5 \cdot \text{IAQ}_{\text{CO2}} + 0.5 \cdot \text{IAQ}_{\text{pollutant}}
$$

See [Global IEQ Aggregation](weakest_link_aggregation.md) for details.

---

## References

- WHO Global Air Quality Guidelines (2021) — particulate matter, NO₂, SO₂, ozone, CO
- EPA NAAQS — Criteria Air Pollutants
- WELL Building Standard v2, Feature A01 — Fundamental Air Quality

See also: [API Reference — Pollutant IAQ](../reference/domains/iaq_pollutants.md)
