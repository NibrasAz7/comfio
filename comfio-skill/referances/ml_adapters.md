# ML/DL Adapters

comfio provides optional adapters for integrating IEQ evaluation into ML/DL pipelines.
Each adapter uses lazy imports so the core package works without ML frameworks installed.

## scikit-learn Pipeline

**Requires:** `pip install comfio[ml]`

### IEQFeatureExtractor

```python
from comfio.ml.sklearn_transformers import IEQFeatureExtractor

extractor = IEQFeatureExtractor()
```

A scikit-learn-compatible transformer that extracts IEQ features from a Pandas DataFrame.
Fits into standard `Pipeline`:

```python
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor

pipe = Pipeline([
    ("ieq", IEQFeatureExtractor()),
    ("model", RandomForestRegressor(n_estimators=100)),
])

pipe.fit(train_df, train_labels)
predictions = pipe.predict(test_df)
```

**Input:** Pandas DataFrame with sensor columns (auto-detected by SensorData).
**Output:** 2-D numpy array of IEQ features per timestamp (domain scores, IEQ index, etc.)

---

## PyTorch DataLoader

**Requires:** `pip install comfio[torch]`

### IEQTimeSeriesDataset

```python
from comfio.ml.torch_dataset import IEQTimeSeriesDataset
from torch.utils.data import DataLoader

dataset = IEQTimeSeriesDataset(
    df: pd.DataFrame,
    window_size: int = 24,    # hours per window
    stride: int = 1,          # step between windows
)

loader = DataLoader(dataset, batch_size=32, shuffle=True)
for batch in loader:
    raw = batch["raw"]          # (32, 24, n_sensors) — raw sensor values
    ieq_index = batch["ieq_index"]  # (32, 24) — IEQ index per timestamp
```

**Use case:** Time-series forecasting, anomaly detection, occupancy prediction from
IEQ-tagged sliding windows.

---

## TensorFlow/Keras

**Requires:** `pip install comfio[keras]`

### IEQPreprocessingLayer

```python
from comfio.ml.keras_adapter import IEQPreprocessingLayer

layer = IEQPreprocessingLayer()
layer.adapt(train_df)    # Learn normalization stats
features = layer(train_df)  # tf.Tensor of IEQ features
```

**Use case:** Keras model preprocessing layer that can be placed at the start of a
Sequential model or Functional API graph.

```python
import tensorflow as tf

model = tf.keras.Sequential([
    IEQPreprocessingLayer(),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(1, activation="linear"),
])
model.compile(optimizer="adam", loss="mse")
model.fit(train_df, train_labels, epochs=50)
```

---

## Common Patterns

### Feature extraction for predictive maintenance

```python
from comfio.ml.sklearn_transformers import IEQFeatureExtractor
from comfio import SensorData, evaluate_thermal, calculate_global_ieq

# Extract features from historical sensor data
sensor = SensorData(df=historical_df)
sensor.validate()

extractor = IEQFeatureExtractor()
features = extractor.fit_transform(historical_df)
# features shape: (n_timestamps, n_features)
# Features include: domain scores, IEQ index, compliance flags
```

### PyTorch time-series forecasting

```python
import torch
from torch import nn
from comfio.ml.torch_dataset import IEQTimeSeriesDataset

dataset = IEQTimeSeriesDataset(df, window_size=168, stride=24)  # 1-week windows
loader = DataLoader(dataset, batch_size=16, shuffle=True)

model = nn.LSTM(input_size=dataset.n_sensors, hidden_size=64, num_layers=2)
for batch in loader:
    out, _ = model(batch["raw"])  # (16, 168, 64)
```
