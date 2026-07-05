# TSV CDF Remapping

## Overview

Thermal Sensation Vote (TSV) augmentation addresses a common problem in building comfort studies: occupant votes are sparse (collected via occasional surveys) while sensor data is dense (continuous time-series). The CDF (Cumulative Distribution Function) remapping method, also known as quantile mapping, expands sparse votes to dense sensor timestamps while preserving the empirical distribution of the original votes.

---

## CDF Matching Principle

The core idea is to match the cumulative distribution functions (CDFs) of the source and target datasets:

$$
F_{\text{target}}(y) = F_{\text{source}}(x)
$$

where:

- $F_{\text{source}}$ is the empirical CDF of the sparse votes
- $F_{\text{target}}$ is the CDF to be constructed on the dense timestamps
- $x$ is a sparse vote value
- $y$ is the corresponding dense (augmented) vote value

### Quantile Mapping

For each dense timestamp $t_i$, we:

1. Draw a random quantile $u_i \sim U(0, 1)$
2. Map through the source CDF inverse: $x_i = F_{\text{source}}^{-1}(u_i)$
3. Assign $x_i$ to timestamp $t_i$

This preserves the distribution of the original votes while populating every timestamp.

---

## Distribution Preservation

The key property of CDF remapping is that the augmented votes have the **same empirical distribution** as the source votes:

$$
\hat{F}_{\text{augmented}}(x) \approx F_{\text{source}}(x)
$$

This is critical for downstream compliance evaluation — the statistical properties (mean, variance, percentiles) of the original survey data are preserved.

---

## TSV Compliance (ASHRAE 55-2023 Appendix L)

ASHRAE 55-2023 Appendix L defines TSV-based compliance:

$$
\text{Compliance rate} = \frac{\#\{i : |\text{TSV}_i| \leq 1.5\}}{N} \times 100\%
$$

A space is compliant when the majority of votes fall within the ±1.5 range on the 7-point sensation scale.

---

## Group-By Support

comfio supports grouping by metadata (e.g., zone, season, occupant ID) to ensure distribution preservation within each group:

```python
augmented = augment_tsv_cdf(
    sparse_votes=votes,
    vote_timestamps=vote_times,
    target_timestamps=sensor_times,
    group_by=zone_labels,  # preserve distribution per zone
)
```

---

## References

- ASHRAE 55-2023, Appendix L — Evaluation of Occupant Thermal Comfort
- Cannon, A.J. (2015). Multivariate quantile mapping bias correction: an N-dimensional probability density function transform for climate model simulation of proxy variables.

See also: [API Reference — TSV](../reference/domains/thermal_tsv.md)
