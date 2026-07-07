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

---

## Time-Aware CDF Remapping

The default CDF remapping uses evenly spaced percentile ranks (`np.linspace(0, 1, m)`) based on position in the target array. This ignores the temporal structure of the votes — a target timestamp in February gets the same percentile rank whether the nearest real vote was in January or August.

### Limitation of the Default Method

Without time interpolation, all timestamps within the same period receive nearly identical values (since `linspace` position changes slowly). This results in poor temporal resolution: the augmented TSV is effectively constant within large time windows, regardless of when votes were actually cast.

### Time-Aware Algorithm

Time-aware mode adds a time-interpolation step before CDF remapping:

**Step 1 — Time interpolation:**

$$
c_j = \text{interp}(t_j, t_{\text{votes}}, v_{\text{votes}})
$$

Linearly interpolate sparse votes in time to obtain continuous values at every target timestamp.

**Step 2 — CDF intervals from original votes:**

$$
\text{PMF}(k) = \frac{n_k}{n}, \quad \text{CDF}(k) = \sum_{i \leq k} \text{PMF}(i)
$$

Partition $[0, 1]$ into intervals $(\text{low}_k, \text{high}_k)$ for each ordinal class $k$.

**Step 3 — Percentile rank mapping:**

$$
q_j = \text{rank}_{\text{pct}}(c_j), \quad \hat{v}_j = k \;\text{where}\; \text{low}_k \leq q_j < \text{high}_k
$$

The percentile rank of each continuous value is mapped through the CDF intervals to obtain the ordinal TSV class.

### Key Difference

The critical difference is Step 1. With time interpolation, the continuous values vary throughout the target timeline based on when nearby votes were cast. The CDF remapping then redistributes these values to match the original vote distribution while maintaining temporal structure.

This is especially important for datasets with large temporal gaps between votes (e.g., months with no surveys), where the default method assigns arbitrary values regardless of temporal proximity.

### Usage

```python
augmented = augment_tsv_cdf(
    sparse_votes=votes,
    vote_timestamps=vote_times,
    target_timestamps=sensor_times,
    time_aware=True,
)
```
