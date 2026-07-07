# TSV Augmentation

## Problem

You have sparse occupant thermal sensation votes (TSV) from occasional surveys but need dense time-series data aligned with sensor timestamps for compliance evaluation.

---

## Solution

```python
import numpy as np
from comfio import augment_tsv_cdf, evaluate_tsv

# Sparse votes (e.g., from 10 surveys)
sparse_votes = np.array([-2, -1, 0, 0, 1, 1, 2, -1, 0, 1])
vote_timestamps = np.arange(10)  # when votes were collected

# Dense sensor timestamps (100 readings)
target_timestamps = np.arange(100)

# Augment: expand sparse votes to dense timestamps
augmented = augment_tsv_cdf(
    sparse_votes=sparse_votes,
    vote_timestamps=vote_timestamps,
    target_timestamps=target_timestamps,
)
# Returns array of length 100 with preserved distribution
```

### Evaluate TSV compliance

```python
tsv_result = evaluate_tsv(augmented)
print(f"Mean TSV: {tsv_result.mean_tsv:.2f}")
print(f"Compliance rate: {tsv_result.compliance_rate:.1%}")
# ASHRAE 55-2023: compliant if |TSV| <= 1.5 for majority of timestamps
```

### Group by zone or season

```python
# Preserve distribution per group
zone_labels = np.array(["zone_a"] * 50 + ["zone_b"] * 50)

augmented = augment_tsv_cdf(
    sparse_votes=sparse_votes,
    vote_timestamps=vote_timestamps,
    target_timestamps=target_timestamps,
    group_by=zone_labels,
)
```

### Time-aware augmentation

When votes have large temporal gaps (e.g., months with no surveys), the default method assigns values based on position in the target array, ignoring when nearby votes were cast. Use `time_aware=True` to preserve temporal coherence:

```python
augmented = augment_tsv_cdf(
    sparse_votes=sparse_votes,
    vote_timestamps=vote_timestamps,
    target_timestamps=target_timestamps,
    time_aware=True,
)
```

This interpolates votes in time before CDF remapping, so nearby timestamps receive similar values based on when nearby votes were cast. The empirical distribution is still preserved. See [TSV CDF Remapping — Time-Aware Mode](../theory/tsv_cdf_remapping.md#time-aware-cdf-remapping) for details.

### Use augmented TSV in Global IEQ

```python
from comfio import calculate_global_ieq

ieq = calculate_global_ieq(
    thermal=thermal_result,
    visual=visual_result,
    acoustic=acoustic_result,
    iaq=iaq_result,
    tsv=tsv_result,  # overrides PMV-based thermal score
)
```

---

## How It Works

The CDF remapping preserves the empirical distribution of the original votes using quantile mapping. See [TSV CDF Remapping](../theory/tsv_cdf_remapping.md) for the mathematical details.

---

## See Also

- [Theory — TSV CDF Remapping](../theory/tsv_cdf_remapping.md)
- [API Reference — TSV](../reference/domains/thermal_tsv.md)
