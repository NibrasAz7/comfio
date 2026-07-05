# Handling Missing Sensor Data

## Problem

IoT sensor data often contains gaps (NaN), out-of-range values, and inconsistent column names. comfio's `SensorData` class handles these issues automatically.

---

## Solution

### 1. Load data into SensorData

```python
import pandas as pd
from comfio import SensorData

df = pd.read_csv("sensor_data.csv")
sensor = SensorData(df=df)

# Auto-detects common column name variants
# e.g., "tdb", "ta", "temperature" → air_temp_c
print(sensor.available_domains())  # ['thermal', 'visual', 'acoustic', 'iaq']
```

### 2. Validate data

```python
# Checks NaN, physical bounds (e.g., temp must be -40 to 60 °C)
sensor.validate()

# Get validated column (NaNs replaced, out-of-range flagged)
clean_temp = sensor.get_validated("air_temp_c")
```

### 3. Handle NaNs explicitly

```python
import numpy as np

# Option A: Drop NaN timestamps
df_clean = df.dropna(subset=["air_temp_c", "relative_humidity_pct"])

# Option B: Forward-fill (for short gaps)
df_filled = df.ffill()

# Option C: Interpolate (for longer gaps)
df_interp = df.interpolate(method="time", limit=5)
```

### 4. Use validated arrays in evaluations

```python
from comfio import evaluate_thermal

thermal = evaluate_thermal(
    tdb=sensor.get_validated("air_temp_c"),
    tr=sensor.get_validated("radiant_temp_c"),
    vr=sensor.get_validated("air_velocity_ms"),
    rh=sensor.get_validated("relative_humidity_pct"),
    met=1.2, clo=0.5,
)
```

---

## Physical Bounds

SensorData enforces the following physical bounds:

| Parameter | Min | Max | Unit |
|---|---|---|---|
| air_temp_c | -40 | 60 | °C |
| radiant_temp_c | -40 | 100 | °C |
| air_velocity_ms | 0 | 10 | m/s |
| relative_humidity_pct | 0 | 100 | % |
| illuminance_lux | 0 | 100000 | lux |
| noise_laeq_db | 0 | 130 | dB |
| co2_ppm | 300 | 5000 | ppm |

Values outside these ranges are flagged as invalid during `validate()`.

---

## See Also

- [API Reference — SensorData](../reference/data_handler.md)
- [API Reference — Validation](../reference/validation.md)
