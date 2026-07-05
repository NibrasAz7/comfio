# PMV/PPD Thermal Comfort

## Overview

The Predicted Mean Vote (PMV) model is the most widely used thermal comfort index, defined in ISO 7730 and ASHRAE 55. It predicts the mean thermal sensation vote of a large group of occupants on a 7-point scale (−3 cold to +3 hot) based on six environmental and personal parameters.

comfio wraps the validated [pythermalcomfort](https://github.com/pythermalcomfort/pythermalcomfort) library to compute PMV/PPD for vectorized time-series data.

---

## The PMV Heat Balance Equation

PMV is derived from the steady-state heat balance of the human body:

$$
0 = L = M - W - C - R - E_{sk} - C_{res} - E_{res}
$$

where:

- $M$ — metabolic rate (W/m²)
- $W$ — external work (W/m²)
- $C$ — convective heat loss from skin
- $R$ — radiative heat loss from skin
- $E_{sk}$ — evaporative heat loss from skin
- $C_{res}$ — convective heat loss from respiration
- $E_{res}$ — evaporative heat loss from respiration

The individual terms are:

$$
C = h_c \cdot (t_{cl} - t_a)
$$

$$
R = h_r \cdot (t_{cl} - \bar{t}_r)
$$

where $t_{cl}$ is clothing surface temperature, $t_a$ is air temperature, $\bar{t}_r$ is mean radiant temperature, and $h_c$, $h_r$ are convective and radiative heat transfer coefficients.

PMV is then:

$$
\text{PMV} = (0.303 \cdot e^{-0.036 \cdot M} + 0.028) \cdot L
$$

---

## PPD Relation

The Predicted Percentage of Dissatisfied (PPD) is a nonlinear function of PMV:

$$
\text{PPD} = 100 - 95 \cdot \exp\left(-0.03353 \cdot \text{PMV}^2 - 0.2179 \cdot \text{PMV}^4\right)
$$

or equivalently:

$$
\text{PPD} = 5 - 100 \cdot \exp\left(-0.03 - 3.14 \cdot \text{PMV}^2\right)
$$

PPD is symmetric around PMV = 0 and has a minimum of 5% (some people are always dissatisfied).

---

## ISO 7730 Categories

| Category | PMV range | PPD limit |
|---|---|---|
| A | −0.2 to +0.2 | ≤ 6% |
| B | −0.5 to +0.5 | ≤ 10% |
| C | −0.7 to +0.7 | ≤ 15% |

comfio uses these categories to determine compliance per timestamp.

---

## Scoring

The thermal score maps PMV to a 0–100 scale:

- **Score = 100** when |PMV| ≤ category limit (comfortable)
- **Score decreases linearly** as |PMV| exceeds the limit
- **Score = 0** when |PMV| ≥ 2.0 (extreme discomfort)

---

## Applicability

- Steady-state model — assumes constant conditions over time
- Valid for: 0 < met < 4, 0 < clo < 2, 10–40 °C air temp, 0–100% RH
- **Not suitable** for transient conditions, non-uniform environments, or outdoor spaces
- See [Limitations](limitations.md) for detailed caveats

---

## References

- ISO 7730:2005 — Ergonomics of the thermal environment — Analytical determination and interpretation of thermal comfort using calculation of the PMV and PPD indices
- ASHRAE 55-2017 — Thermal Environmental Conditions for Human Occupancy
- Fanger, P.O. (1970). *Thermal Comfort*. Danish Technical Press.

See also: [API Reference — Thermal](../reference/domains/thermal.md)
