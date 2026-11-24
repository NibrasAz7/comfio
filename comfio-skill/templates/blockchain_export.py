"""comfio blockchain export template: compliance report → Solidity ABI JSON.

Generates a JSON payload matching the IEQComplianceOracle.submitCompliance ABI
for blockchain Oracle integration in performance-based smart contracts.
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
    evaluate_thermal,
    evaluate_visual,
)
from comfio.performance.contract_schema import default_compliance_schema

# --- 1. Run the IEQ pipeline ---
df = pd.read_csv("sensor_data.csv")
sensor = SensorData(df=df)
sensor.validate()

thermal = evaluate_thermal(
    tdb=sensor.get_validated("air_temp_c"),
    tr=sensor.get_validated("radiant_temp_c"),
    vr=sensor.get_validated("air_velocity_ms"),
    rh=sensor.get_validated("relative_humidity_pct"),
    met=1.2, clo=0.5, category="B",
)
visual = evaluate_visual(illuminance=sensor.get_validated("illuminance_lux"))
acoustic = evaluate_acoustic(laeq=sensor.get_validated("noise_laeq_db"))
iaq = evaluate_iaq(co2=sensor.get_validated("co2_ppm"))

ieq = calculate_global_ieq(
    thermal=thermal, visual=visual, acoustic=acoustic, iaq=iaq,
)

# --- 2. Generate compliance report ---
report = calculate_compliance(
    ieq_result=ieq,
    threshold=80.0,
    period_start=1717200000.0,  # Unix timestamp
    period_end=1717286400.0,
)

# --- 3. Export contract JSON using default schema ---
contract_payload = report.to_contract_payload()
contract_json = report.to_contract_json()

print("=== Contract Payload (Solidity ABI) ===")
print(json.dumps(contract_payload, indent=2))

# --- 4. Show the ABI schema ---
schema = default_compliance_schema()
print("\n=== ABI Schema ===")
print(json.dumps(schema.to_dict(), indent=2))

# --- 5. Save to file ---
with open("oracle_payload.json", "w") as f:
    f.write(contract_json)
print("\nSaved to oracle_payload.json")

# --- 6. Solidity struct for reference ---
print("""
struct ComplianceReport {
    uint256 periodStart;
    uint256 periodEnd;
    uint8   ieqIndexAvg;
    uint8   complianceRatePct;
    bool    thermalCompliant;
    bool    visualCompliant;
    bool    acousticCompliant;
    bool    iaqCompliant;
    uint32  totalOccupiedHours;
    uint32  compliantHours;
}
""")
