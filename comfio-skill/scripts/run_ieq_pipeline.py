#!/usr/bin/env python3
"""End-to-end CLI: CSV → domain evaluation → Global IEQ → compliance JSON.

Usage:
    python run_ieq_pipeline.py --input sensor.csv --threshold 80 --output report.json
"""

import argparse
import json
import sys

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


def run_pipeline(csv_path: str, threshold: float) -> dict:
    df = pd.read_csv(csv_path)
    sensor = SensorData(df=df)
    sensor.validate()

    domains = sensor.available_domains()
    print(f"Available domains: {domains}")

    kwargs = {}

    if "thermal" in domains:
        kwargs["thermal"] = evaluate_thermal(
            tdb=sensor.get_validated("air_temp_c"),
            tr=sensor.get_validated("radiant_temp_c"),
            vr=sensor.get_validated("air_velocity_ms"),
            rh=sensor.get_validated("relative_humidity_pct"),
            met=1.2,
            clo=0.5,
            category="B",
        )

    if "visual" in domains:
        kwargs["visual"] = evaluate_visual(
            illuminance=sensor.get_validated("illuminance_lux"),
        )

    if "acoustic" in domains:
        kwargs["acoustic"] = evaluate_acoustic(
            laeq=sensor.get_validated("noise_laeq_db"),
        )

    if "iaq" in domains:
        kwargs["iaq"] = evaluate_iaq(
            co2=sensor.get_validated("co2_ppm"),
        )

    advanced = sensor.available_advanced_domains()
    if "pollutant_iaq" in advanced:
        try:
            kwargs["pollutant_iaq"] = evaluate_iaq_pollutants(
                pm25=sensor.get_validated("pm25_ugm3"),
                tvoc=sensor.get_validated("tvoc_ugm3"),
                formaldehyde=sensor.get_validated("formaldehyde_ppb"),
                co=sensor.get_validated("co_ppm"),
            )
        except Exception:
            pass

    ieq = calculate_global_ieq(**kwargs)

    report = calculate_compliance(ieq, threshold=threshold)

    print(f"Global IEQ Index (avg): {report.ieq_index_avg:.1f}")
    print(f"Compliance rate: {report.compliance_rate_pct:.1f}%")
    print(f"Domains: {report.domains}")
    for domain, score in report.domain_scores_avg.items():
        print(f"  {domain}: {score:.1f}/100")

    return json.loads(report.to_json())


def main() -> int:
    parser = argparse.ArgumentParser(description="Run comfio IEQ pipeline on a CSV file")
    parser.add_argument("--input", "-i", required=True, help="Input sensor CSV path")
    parser.add_argument("--output", "-o", default="report.json", help="Output JSON report path")
    parser.add_argument("--threshold", "-t", type=float, default=80.0, help="IEQ compliance threshold")
    args = parser.parse_args()

    result = run_pipeline(args.input, args.threshold)

    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nReport saved to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
