# Simplified PMV (Buratti)

## Overview

The Simplified Predicted Mean Vote (sPMV) model by Buratti et al. (2009) provides a reduced-form PMV calculation requiring only **indoor air temperature** and **relative humidity** — eliminating the need for metabolic rate, clothing insulation, air velocity, and radiant temperature measurements.

This makes it ideal for building management systems with limited sensor coverage.

---

## Seasonal Model

The sPMV model uses season-specific coefficients $(a, b, c)$:

$$
\text{sPMV} = a \cdot t_{in} + b \cdot RH + c
$$

where $t_{in}$ is indoor air temperature (°C) and $RH$ is relative humidity (%).

### Coefficients by season

| Season | $a$ | $b$ | $c$ |
|---|---|---|---|
| Winter | 0.212 | 0.0014 | −5.06 |
| Mid-season | 0.212 | 0.0014 | −5.06 |
| Summer | 0.212 | 0.0014 | −5.06 |

!!! note "Coefficient source"
    The coefficients are derived from regression analysis of full PMV calculations across typical indoor conditions. See Buratti et al. (2009) for the complete derivation and validation ranges.

---

## Vapor Pressure (Magnus Formula)

The model internally computes water vapor pressure using the Magnus formula:

$$
p_v = \frac{RH}{100} \times 6.112 \times \exp\left(\frac{17.62 \cdot t_{in}}{243.12 + t_{in}}\right)
$$

This is used to account for humidity effects on evaporative heat loss.

---

## Scoring

The sPMV score uses the same PMV-to-score mapping as the full PMV model:

- **Score = 100** when |sPMV| ≤ 0.5 (Category B equivalent)
- **Score decreases linearly** as |sPMV| increases
- **Score = 0** when |sPMV| ≥ 2.0

---

## Applicability

- Valid for typical office/residential indoor conditions
- Assumes default values: met = 1.2, clo = 1.0 (winter) / 0.5 (summer), v = 0.1 m/s
- **Less accurate** than full PMV when actual metabolic rate or clothing differ significantly from defaults
- Best suited for screening-level assessment or BMS integration with limited sensors

---

## References

- Buratti, C., Ricciardi, P., & Vergoni, M. (2009). Simplified PMV model for HVAC systems control. *Building and Environment*, 44(3), 441–449.

See also: [API Reference — sPMV](../reference/domains/thermal_spmv.md)
