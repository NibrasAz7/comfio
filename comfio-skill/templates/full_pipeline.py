"""comfio full pipeline template: SensorData → domains → Global IEQ → compliance → contract JSON.

Copy this file and adapt the CSV path, threshold, and weighting preset.
"""

import json

import numpy as np
import pandas as pd

from comfio import (
    SensorData,
    calculate_compliance,
    calculate_global_ieq,
    evaluate_acoustic,
    evaluate_iaq,
    evaluate_iaq_pollutants,
    evaluate_thermal,
    evaluate_visual,
)
from comfio.integration.weights import preset_weights

# --- 1. Load sensor data ---
df = pd.read_csv("sensor_data.csv")
sensor = SensorData(df=df)
sensor.validate(nan_strategy="interpolate")

print(f"Available domains: {sensor.available_domains()}")

# --- 2. Evaluate each available domain ---
domain_kwargs = {}

if "thermal" in sensor.available_domains():
    domain_kwargs["thermal"] = evaluate_thermal(
        tdb=sensor.get_validated("air_temp_c"),
        tr=sensor.get_validated("radiant_temp_c"),
        vr=sensor.get_validated("air_velocity_ms"),
        rh=sensor.get_validated("relative_humidity_pct"),
        met=1.2,
        clo=0.5,
        category="B",
    )

if "visual" in sensor.available_domains():
    domain_kwargs["visual"] = evaluate_visual(
        illuminance=sensor.get_validated("illuminance_lux"),
        task_type="office_writing",
    )

if "acoustic" in sensor.available_domains():
    domain_kwargs["acoustic"] = evaluate_acoustic(
        laeq=sensor.get_validated("noise_laeq_db"),
        nc_level="NC-35",
    )

if "iaq" in sensor.available_domains():
    domain_kwargs["iaq"] = evaluate_iaq(
        co2=sensor.get_validated("co2_ppm"),
        threshold_level="good",
    )

# --- 3. Optional: pollutant IAQ (50/50 blend with CO₂ IAQ) ---
try:
    domain_kwargs["pollutant_iaq"] = evaluate_iaq_pollutants(
        pm25=sensor.get_validated("pm25_ugm3"),
        tvoc=sensor.get_validated("tvoc_ugm3"),
        formaldehyde=sensor.get_validated("formaldehyde_ppb"),
        co=sensor.get_validated("co_ppm"),
    )
except Exception:
    pass  # pollutant columns not available

# --- 4. Compute Global IEQ Index ---
weights = preset_weights("default")
ieq = calculate_global_ieq(weights=weights, **domain_kwargs)

print(f"Global IEQ Index (avg): {np.mean(ieq.index):.1f}")
print(f"Domains used: {ieq.domains}")

# --- 5. Compliance report ---
report = calculate_compliance(
    ieq_result=ieq,
    threshold=80.0,
)

print(f"Compliance rate: {report.compliance_rate_pct:.1f}%")
print(f"Compliant hours: {report.compliant_hours:.1f}/{report.total_hours:.1f}")

# --- 6. Export contract JSON for blockchain Oracle ---
contract_json = report.to_contract_json()
with open("compliance_report.json", "w") as f:
    f.write(contract_json)
print("Contract JSON saved to compliance_report.json")
