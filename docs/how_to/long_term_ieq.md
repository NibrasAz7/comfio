# Long-Term IEQ: Time-in-Category Summaries

## Problem

You need the EN 16798-1 (Annex B) long-term representation: the
percentage of occupied time a zone spends in comfort categories I–IV,
per domain, with cumulative I+II / I+II+III aggregates — for the whole
monitored period or per day.

## Solution

### 1. From an existing evaluation result

Every domain result can be summarized directly:

```python
import numpy as np
from comfio import evaluate_thermal, evaluate_temporal_thermal

thermal = evaluate_thermal(
    tdb=np.array([24.0, 26.0, 29.0]),
    tr=np.array([24.0, 26.0, 29.0]),
    vr=np.array([0.1, 0.1, 0.1]),
    rh=np.array([50.0, 50.0, 50.0]),
    met=1.2, clo=0.5,
)
summary = evaluate_temporal_thermal(thermal)
print(summary.percentages)  # {'I': ..., 'II': ..., 'III': ..., 'IV': ...}
print(summary.cumulative)   # {'I': ..., 'I+II': ..., 'I+II+III': ..., 'I+II+III+IV': 100.0}
```

The wrappers accept raw arrays too — `evaluate_temporal_thermal(pmv_array)`
works for PMV, sPMV and sPMVopt alike. Equivalent wrappers exist for
IAQ (`evaluate_temporal_iaq`), adaptive comfort
(`evaluate_temporal_adaptive`), visual (`evaluate_temporal_visual`)
and acoustic (`evaluate_temporal_acoustic`).

### 2. With timestamps — occupied hours and daily tables

```python
import pandas as pd

summary = evaluate_temporal_thermal(thermal, timestamps=df["timestamp"])
print(summary.total_hours, summary.hours)  # occupied hours per category

daily = evaluate_temporal_thermal(thermal, timestamps=df["timestamp"], by="D")
print(daily[["I", "II", "III", "IV", "I+II"]])  # one row per day
```

Timestamps enable **time weighting**: each sample counts for its actual
duration, so irregular sampling does not bias the percentages. `by`
accepts any pandas offset alias (`"D"`, `"W"`, `"MS"`, …).

### 3. Occupancy and invalid samples

```python
summary = evaluate_temporal_thermal(
    thermal,
    timestamps=df["timestamp"],
    occupied=df["occupied"].to_numpy(),  # unoccupied rows dropped
)
print(summary.excluded_pct)  # % occupied time with NaN labels (e.g. out-of-domain PMV)
```

### 4. Two-step use (custom classifiers)

```python
from comfio import thermal_category, iaq_category_en, summarize_time_in_categories

labels = thermal_category(pmv)                # 1-4, NaN for invalid
labels = iaq_category_en(co2, outdoor=420.0)  # NOTE: delta above outdoor, not absolute ppm
summary = summarize_time_in_categories(labels, timestamps=df["timestamp"], by="W")
```

### 5. In the pipeline and the GUI

`run_pipeline()` computes `result.temporal_results` automatically for
every available domain (disable with `config={"temporal": False}`), and
`temporal_to_markdown(result.temporal_results)` renders a compact
LLM-ready block. The GUI workbench shows the same data as stacked
category bars, a cumulative table, and a daily breakdown under
**Long-term IEQ** in the Verdict section.
