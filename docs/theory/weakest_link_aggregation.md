# Global IEQ Aggregation

## Overview

The Global IEQ Index is the core innovation of comfio. It merges isolated domain scores (thermal, visual, acoustic, IAQ) into a unified 0–100 metric using configurable weighting schemas. This enables holistic building performance assessment rather than siloed domain evaluation.

---

## Weighted Sum

The base Global IEQ Index is a weighted sum of domain scores:

$$
\text{IEQ} = w_t \cdot S_t + w_v \cdot S_v + w_a \cdot S_a + w_i \cdot S_i
$$

where $S_t, S_v, S_a, S_i$ are thermal, visual, acoustic, and IAQ scores (0–100), and $w_t + w_v + w_a + w_i = 1$.

### Default Weights

| Domain | Weight | Rationale |
|---|---|---|
| Thermal | 40% | Primary comfort driver (Pierson et al. 2019) |
| IAQ | 25% | Health-critical, cumulative exposure |
| Visual | 20% | Significant but secondary |
| Acoustic | 15% | Context-dependent |

### Preset Weighting Schemas

| Pres | Thermal | IAQ | Visual | Acoustic | Use Case |
|---|---|---|---|---|---|
| default | 40% | 25% | 20% | 15% | General (Pierson et al. 2019) |
| equal | 25% | 25% | 25% | 25% | Equal weighting |
| school | 27% | 26% | 24% | 23% | School children (Yang et al. 2020) |
| office | 45% | 30% | 15% | 10% | Office workers |
| healthcare | 25% | 40% | 15% | 20% | Healthcare facilities |

---

## IAQ Blending

When both basic IAQ (CO₂) and pollutant IAQ are provided, they blend 50/50 before entering the weighted sum:

$$
S_i = 0.5 \cdot S_{\text{CO2}} + 0.5 \cdot S_{\text{pollutant}}
$$

This ensures that neither ventilation adequacy nor pollutant exposure dominates the IAQ assessment.

---

## TSV Thermal Override

When TSV (Thermal Sensation Vote) data is available, it **overrides** the PMV-based thermal score:

$$
S_t = S_{\text{TSV}} \quad \text{(when TSV is provided)}
$$

Rationale: occupant feedback is ground truth. If occupants report comfort (|TSV| ≤ 1.5), the space is comfortable regardless of what the PMV model predicts. This is consistent with ASHRAE 55-2023 Appendix L.

---

## Advanced Domain Integration

Advanced domain results (reverberation, speech intelligibility, ventilation, daylighting, color quality) can optionally augment the base domains:

- **Reverberation / STI** → augment acoustic score
- **Ventilation** → augment IAQ score
- **Daylighting / Color quality** → augment visual score

The augmentation uses a weighted blend between the basic and advanced scores.

---

## Weakest-Link Philosophy

While the Global IEQ Index uses a weighted sum (not a strict minimum), the pollutant IAQ sub-score uses a weakest-link approach (minimum across pollutants). This hybrid ensures:

1. **Holistic view** — no single domain dominates the overall score
2. **Health safety** — a single hazardous pollutant flags the IAQ component
3. **Flexibility** — users can customize weights for their building type

---

## References

- Pierson, A., et al. (2019). IEQ weighting schemes: A review. *Building and Environment*.
- Yang, W., et al. (2020). IEQ assessment in school buildings. *Indoor Air*.

See also: [API Reference — Integration](../reference/integration.md)
