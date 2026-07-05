# Train Personalisation

## Problem

The standard PMV model predicts the *average* occupant's comfort. You want to personalise predictions to match a specific occupant's reported thermal sensation votes (TSV).

---

## Solution

### 1. Train personalisation index

```python
import numpy as np
from comfio import train_personalisation

# Historical data: PMV predictions and occupant's actual TSV
historical_pmv = np.array([0.2, -0.3, 0.5, -0.1, 0.3, 0.0, -0.2, 0.4])
historical_tsv = np.array([0, -1, 1, 0, 0, -1, -1, 1])

# Fit OLS regression: TSV = alpha * PMV + beta
index = train_personalisation(
    pmv=historical_pmv,
    tsv=historical_tsv,
)
print(f"alpha={index.alpha:.3f}, beta={index.beta:.3f}")
```

### 2. Apply personalisation to new data

```python
from comfio import evaluate_personalised_pmv

result = evaluate_personalised_pmv(
    tdb=np.array([24.0, 25.0, 26.0]),
    tr=np.array([24.0, 25.0, 26.0]),
    vr=np.array([0.1, 0.1, 0.1]),
    rh=np.array([50.0, 50.0, 50.0]),
    met=1.2, clo=0.5,
    personalisation_index=index,
)
print(f"Personalised PMV: {result.personalised_pmv}")
```

### 3. Seasonal personalisation

```python
from comfio import train_seasonal_personalisation

# Train separate models per season
seasonal_index = train_seasonal_personalisation(
    pmv=historical_pmv,
    tsv=historical_tsv,
    seasons=np.array(["winter", "winter", "summer", "winter",
                       "summer", "summer", "winter", "summer"]),
)

# Apply with season selection
result = evaluate_personalised_pmv(
    tdb=tdb, tr=tr, vr=vr, rh=rh, met=1.2, clo=0.5,
    personalisation_index=seasonal_index,
    season="summer",
)
```

---

## How It Works

The personalisation fits an OLS regression:

$$
\text{TSV}_{\text{personalised}} = \alpha \cdot \text{PMV} + \beta
$$

where $\alpha$ and $\beta$ are learned from historical data. This shifts and scales the PMV prediction to match the occupant's individual comfort preferences.

---

## See Also

- [API Reference — Personalised Thermal](../reference/domains/thermal_personal.md)
- [Theory — PMV/PPD](../theory/thermal_pmv_ppd.md)
