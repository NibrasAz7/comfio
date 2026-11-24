"""comfio scikit-learn pipeline template: IEQFeatureExtractor → ML model.

Requires: pip install comfio[ml]
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from comfio.ml.sklearn_transformers import IEQFeatureExtractor

# --- 1. Load sensor data with labels (e.g., occupant satisfaction scores) ---
df = pd.read_csv("sensor_data.csv")
labels = pd.read_csv("satisfaction_labels.csv")["satisfaction"].values

# --- 2. Build sklearn Pipeline with IEQ feature extraction ---
from sklearn.pipeline import Pipeline

pipe = Pipeline([
    ("ieq", IEQFeatureExtractor()),
    ("model", RandomForestRegressor(n_estimators=100, random_state=42)),
])

# --- 3. Train/test split ---
X_train, X_test, y_train, y_test = train_test_split(
    df, labels, test_size=0.2, random_state=42,
)

# --- 4. Fit and predict ---
pipe.fit(X_train, y_train)
predictions = pipe.predict(X_test)

mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print(f"MSE: {mse:.4f}")
print(f"R²: {r2:.4f}")

# --- 5. Feature importance (via the RandomForest) ---
rf = pipe.named_steps["model"]
importances = rf.feature_importances_
for i, imp in enumerate(sorted(importances, reverse=True)[:10]):
    print(f"  Feature {i}: importance={imp:.4f}")
