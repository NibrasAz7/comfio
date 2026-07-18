# Local Thermal Discomfort

## Overview

ISO 7730 and ASHRAE 55 distinguish between **whole-body** thermal comfort (PMV/PPD) and **local** (body-part-specific) discomfort. Even when PMV is near neutral, local effects can cause a significant percentage of occupants to be dissatisfied.

comfio v0.1.6 adds two local discomfort indices from pythermalcomfort:

1. **Ankle draft** — excessive air movement at 0.1 m above the floor
2. **Vertical temperature gradient** — head-to-feet temperature difference

## Ankle Draft (ASHRAE 55-2023 §5.3.3)

### Mechanism

Cold air pooling near the floor (from diffusers, infiltration, or stack effect) creates a local draft at ankle height (0.1 m). Sedentary occupants are particularly sensitive because their feet are stationary.

### Model

The PPD due to ankle draft is computed from the air speed at 0.1 m (`v_ankle`), along with the standard PMV inputs (tdb, tr, vr, rh, met, clo). ASHRAE 55-2023 specifies:

- **Acceptable**: PPD ≤ 20% for sedentary occupants
- Applicable range: 20 ≤ tdb ≤ 26 °C, 0 < vr < 0.2 m/s, 1.0 ≤ met ≤ 1.3, 0.5 ≤ clo ≤ 1.0

### Reference

ASHRAE Standard 55-2023, §5.3.3 "Local Thermal Discomfort."

## Vertical Air Temperature Gradient (ISO 7730 §6.1)

### Mechanism

Stratification causes the air temperature at head height (1.1 m seated / 1.7 m standing) to differ from that at ankle height (0.1 m). A gradient exceeding 3 °C/m causes discomfort.

### Model

The PPD due to vertical gradient is computed from the gradient (°C/m) and standard PMV inputs. The equation is only valid for `vr < 0.2 m/s`.

- **Acceptable**: PPD ≤ 5% (ISO 7730 Category A), gradient ≤ 3 °C/m
- Applicable range: 10 ≤ tdb ≤ 40 °C, 0 < vr < 0.2 m/s, 1.0 ≤ met ≤ 4.0, 0.0 ≤ clo ≤ 1.5

### Reference

ISO 7730:2005, §6.1 "Local thermal discomfort."

## Combined Compliance

Full ISO 7730 Category compliance requires **both**:

1. PMV/PPD within category limits (whole-body)
2. Local discomfort indices within acceptable limits

Use `local_discomfort_score()` to combine ankle draft and vertical gradient PPD into a single 0-100 score.
