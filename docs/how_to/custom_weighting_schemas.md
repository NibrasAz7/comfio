# Custom Weighting Schemas

## Problem

The default IEQ weights (thermal 40%, IAQ 25%, visual 20%, acoustic 15%) may not suit your building type. You need to customize the domain weights.

---

## Solution

### Use a preset

```python
from comfio.integration.weights import preset_weights

# Available presets: default, equal, school, office, healthcare
weights = preset_weights("office")
# thermal=45%, iaq=30%, visual=15%, acoustic=10%
```

### Define custom weights

```python
from comfio.integration.weights import custom_weights

weights = custom_weights(
    thermal=0.50,
    visual=0.15,
    acoustic=0.10,
    iaq=0.25,
)
# Weights are automatically normalized to sum to 1.0
```

### Use weights in Global IEQ

```python
from comfio import calculate_global_ieq

ieq = calculate_global_ieq(
    thermal=thermal_result,
    visual=visual_result,
    acoustic=acoustic_result,
    iaq=iaq_result,
    weights=weights,
)
```

### Use the WeightSchema dataclass directly

```python
from comfio.integration.weights import WeightSchema

weights = WeightSchema(
    thermal=0.40,
    visual=0.20,
    acoustic=0.15,
    iaq=0.25,
)
```

---

## Available Presets

| Preset | Thermal | IAQ | Visual | Acoustic | Use Case |
|---|---|---|---|---|---|
| default | 40% | 25% | 20% | 15% | General buildings |
| equal | 25% | 25% | 25% | 25% | Equal weighting |
| school | 27% | 26% | 24% | 23% | School children |
| office | 45% | 30% | 15% | 10% | Office workers |
| healthcare | 25% | 40% | 15% | 20% | Healthcare facilities |

---

## See Also

- [API Reference — Integration](../reference/integration.md)
- [Theory — Global IEQ Aggregation](../theory/weakest_link_aggregation.md)
