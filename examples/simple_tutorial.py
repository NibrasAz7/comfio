"""Simple tutorial: basic comfio workflow.

Demonstrates the core 4-step pipeline:
1. Create synthetic sensor data
2. Evaluate each domain
3. Calculate Global IEQ Index
4. Check compliance and export contract JSON
"""

import numpy as np
from comfio import (
    SensorData,
    evaluate_thermal,
    evaluate_visual,
    evaluate_acoustic,
    evaluate_iaq,
    calculate_global_ieq,
    calculate_compliance,
)


def main() -> None:
    # 1. Create synthetic sensor data
    rng = np.random.default_rng(42)
    n = 100
    tdb = rng.normal(24.0, 1.5, n)
    tr = rng.normal(24.0, 1.0, n)
    vr = rng.normal(0.1, 0.02, n)
    rh = rng.normal(50.0, 5.0, n)
    illuminance = rng.normal(500.0, 50.0, n)
    laeq = rng.normal(40.0, 5.0, n)
    co2 = rng.normal(800.0, 100.0, n)

    # 2. Evaluate each domain
    thermal = evaluate_thermal(
        tdb=tdb, tr=tr, vr=vr, rh=rh, met=1.2, clo=0.5, category="B",
    )
    visual = evaluate_visual(illuminance=illuminance)
    acoustic = evaluate_acoustic(laeq=laeq)
    iaq = evaluate_iaq(co2=co2)

    print(f"Thermal — Mean PMV: {np.mean(thermal.pmv):.2f}")
    print(f"Visual — Mean score: {np.mean(visual.score):.1f}/100")
    print(f"Acoustic — Mean score: {np.mean(acoustic.score):.1f}/100")
    print(f"IAQ — Mean score: {np.mean(iaq.score):.1f}/100")

    # 3. Calculate Global IEQ Index
    ieq = calculate_global_ieq(
        thermal=thermal, visual=visual, acoustic=acoustic, iaq=iaq,
    )
    print(f"\nGlobal IEQ Index — Mean: {np.mean(ieq.index):.1f}/100")

    # 4. Check compliance
    report = calculate_compliance(ieq, threshold=80.0)
    print(f"Compliance rate: {report.compliance_rate_pct:.1f}%")
    print(f"Is compliant: {report.is_compliant}")
    print(f"\nContract JSON:\n{report.to_contract_json()}")


if __name__ == "__main__":
    main()
