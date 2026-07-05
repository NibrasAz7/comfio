# Adaptive Comfort Models

## Overview

Adaptive comfort models recognize that occupants in naturally ventilated buildings actively adapt to their environment (clothing, windows, posture) and that their comfort temperature tracks the prevailing outdoor temperature. Unlike the PMV model, adaptive models do not require metabolic rate or clothing insulation — only indoor and outdoor temperatures.

comfio implements both the ASHRAE 55-2023 and EN 16798-1:2019 adaptive models.

---

## ASHRAE 55-2023

The acceptable indoor operative temperature for naturally conditioned spaces:

$$
t_{comf} = 0.31 \cdot t_{prevail} + 17.8
$$

where $t_{comf}$ is the comfort operative temperature (°C) and $t_{prevail}$ is the prevailing mean outdoor air temperature (°C).

### Acceptability ranges

| Acceptability | Range |
|---|---|
| 80% | $t_{comf} \pm 3.5$ °C |
| 90% | $t_{comf} \pm 2.5$ °C |

### Applicability

- Naturally ventilated buildings only
- $10 \leq t_{prevail} \leq 33.5$ °C
- No mechanical cooling in operation
- Occupants have control over operable windows

---

## EN 16798-1:2019

The European adaptive model uses the exponentially weighted running mean of outdoor temperature:

$$
t_{comf} = 0.33 \cdot t_{rm} + 18.8
$$

where $t_{rm}$ is the running mean outdoor temperature (°C), computed as:

$$
t_{rm} = (1 - \alpha) \cdot \sum_{i=1}^{n} \alpha^{i-1} \cdot t_{out,i}
$$

with $\alpha = 0.8$ (recommended), $t_{out,i}$ is the daily mean outdoor temperature for day $i$.

### Category limits

| Category | Offset |
|---|---|
| I (High expectation) | $t_{comf} \pm 2$ °C |
| II (New buildings) | $t_{comf} \pm 3$ °C |
| III (Existing buildings) | $t_{comf} \pm 4$ °C |

### Applicability

- Offices and residential buildings without mechanical cooling
- $t_{rm}$ must be between 10–30 °C
- Window operation available to occupants

---

## Scoring

For both models, comfio maps the distance from the comfort temperature to a 0–100 score:

- **Score = 100** when indoor operative temperature is within the acceptability range
- **Score decreases linearly** outside the range
- **Score = 0** at 2× the range offset beyond the comfort temperature

---

## References

- ASHRAE 55-2023 — Thermal Environmental Conditions for Human Occupancy, Appendix G
- EN 16798-1:2019 — Energy performance of buildings — Ventilation for buildings
- de Dear, R., & Brager, G.S. (1998). Developing an adaptive model of thermal comfort and preference.

See also: [API Reference — Adaptive Thermal](../reference/domains/thermal_adaptive.md)
