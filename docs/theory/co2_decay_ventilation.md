# CO₂ Decay Ventilation Model

## Overview

The CO₂ decay method estimates outdoor air ventilation rates by monitoring
the exponential decay of CO₂ concentration after occupants leave a space.
This is an optional advanced module requiring the `psychrolib` extra.

---

## Exponential Decay Model

When a space is unoccupied, CO₂ concentration decays exponentially toward
the outdoor (background) level:

$$
C(t) = C_{\text{out}} + (C_0 - C_{\text{out}}) \, e^{-k \, t}
$$

where:

- $C(t)$ is the CO₂ concentration at time $t$ (ppm)
- $C_{\text{out}}$ is the outdoor CO₂ concentration (~420 ppm)
- $C_0$ is the initial CO₂ concentration at $t = 0$ (ppm)
- $k$ is the air exchange rate (1/h)
- $t$ is the elapsed time (h)

---

## Ventilation Rate Estimation

By fitting the exponential decay to measured CO₂ data, the air exchange rate
$k$ is extracted. The outdoor air ventilation rate per person is then:

$$
V_p = \frac{k \times V_{\text{room}}}{N}
$$

where $V_{\text{room}}$ is the room volume (m³) and $N$ is the design occupancy.

---

## Scoring

The ventilation score compares the estimated ventilation rate against
ASHRAE 62.1 minimum requirements:

| Occupancy Type | Min Ventilation (L/s·person) |
|----------------|-------------------------------|
| Office | 8.5 |
| Classroom | 6.7 |
| Meeting | 8.5 |

Score is 100 when ventilation meets or exceeds the requirement, decreasing
linearly to 0 at 50% of the requirement.

---

## Applicability

- Requires a period of unoccupancy (e.g., overnight) for decay observation
- Assumes well-mixed air (may not hold for large or partitioned spaces)
- Outdoor CO₂ concentration must be known or estimated
- Requires CO₂ sensor with sufficient accuracy (±50 ppm)

---

## References

- ASHRAE 62.1-2019 — Ventilation for Acceptable Indoor Air Quality
- ASTM D7297-14 — Standard Practice for Determining the Outdoor Air Ventilation Rate

See also: [API Reference — Advanced IAQ](../reference/domains/iaq.md)
