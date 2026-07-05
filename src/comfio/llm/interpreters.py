"""Semantic interpreters — convert comfio results into token-efficient text.

These functions take existing ``GlobalIEQResult`` and ``ComplianceReport``
objects and produce structured markdown or summary dicts suitable for
injection into LLM context windows.

No external dependencies beyond numpy and pandas (already required by core).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from comfio.core.data_handler import SensorData
from comfio.domains.acoustic import evaluate_acoustic
from comfio.domains.iaq import evaluate_iaq
from comfio.domains.iaq_pollutants import evaluate_iaq_pollutants
from comfio.domains.thermal import evaluate_thermal
from comfio.domains.visual import evaluate_visual
from comfio.integration.global_ieq import GlobalIEQResult, calculate_global_ieq
from comfio.performance.contracts import ComplianceReport, calculate_compliance

_DOMAIN_DIAGNOSTICS: dict[str, str] = {
    "thermal": "PMV drifted outside the comfort band. Action: check cooling/heating setpoints or solar gains.",
    "visual": "Illuminance below target levels. Action: verify lighting fixtures or increase daylight access.",
    "acoustic": "Noise levels exceeded NC threshold. Action: inspect noise sources or add acoustic damping.",
    "iaq": "CO₂ thresholds breached. Action: trigger HVAC fresh air purge or increase ventilation rate.",
    "pollutant_iaq": "PM2.5/TVOC/formaldehyde/CO levels exceed WHO/WELL thresholds. Action: check filtration, source control, or increase ventilation.",
    "adaptive": "Operative temperature outside adaptive comfort band. Action: adjust natural ventilation or shading.",
    "spmv": "Simplified PMV outside comfort range. Action: verify seasonal coefficients and indoor temperature.",
    "tsv": "Occupant thermal sensation votes indicate discomfort. Action: review setpoints against occupant feedback.",
}


def ieq_to_markdown(
    ieq_result: GlobalIEQResult,
    compliance_report: ComplianceReport | None = None,
    zone_id: str | None = None,
) -> str:
    """Serialize a GlobalIEQResult into token-efficient structured markdown.

    Parameters
    ----------
    ieq_result : GlobalIEQResult
        The Global IEQ Index calculation result.
    compliance_report : ComplianceReport or None
        Optional compliance report for compliance context.
    zone_id : str or None
        Optional zone identifier for spatial context.

    Returns
    -------
    str
        Markdown string (~200 tokens) summarizing the IEQ state.
    """
    index_avg = float(np.mean(ieq_result.index))
    index_min = float(np.min(ieq_result.index))
    index_max = float(np.max(ieq_result.index))

    domain_avgs = {
        domain: float(np.mean(scores)) for domain, scores in ieq_result.domain_scores.items()
    }
    worst_domain = min(domain_avgs, key=domain_avgs.get)

    if compliance_report is not None:
        status = "COMPLIANT" if compliance_report.compliance_rate_pct > 85.0 else "NON-COMPLIANT"
        compliance_line = f"- **Contract Compliance Rate**: {compliance_report.compliance_rate_pct:.1f}%\n"
    else:
        status = "COMPLIANT" if index_avg >= 80.0 else "NON-COMPLIANT"
        compliance_line = ""

    zone_label = f"Zone {zone_id}" if zone_id else "Building"

    md = f"### Building System Report: {zone_label}\n"
    md += f"- **Current Operational State**: {status}\n"
    md += f"- **Global IEQ Index Score**: {index_avg:.1f}/100 "
    md += f"(min: {index_min:.1f}, max: {index_max:.1f})\n"
    md += compliance_line
    md += f"- **Timestamps Evaluated**: {ieq_result.n_timestamps}\n\n"
    md += "#### Domain Breakdown:\n"

    for domain, avg_score in domain_avgs.items():
        flag = "WARNING" if avg_score < 70 else "OK"
        md += f"  - [{flag}] {domain.upper()}: {avg_score:.1f}/100\n"

    md += f"\n#### Diagnostic Insight:\n"
    md += f"The primary limiting factor is the **{worst_domain.upper()}** domain "
    md += f"(score: {domain_avgs[worst_domain]:.1f}/100). "
    md += _DOMAIN_DIAGNOSTICS.get(worst_domain, "Investigate sensor readings for this domain.")

    return md


def ieq_to_summary_dict(
    ieq_result: GlobalIEQResult,
    compliance_report: ComplianceReport | None = None,
) -> dict[str, float | str | int | list[str]]:
    """Convert a GlobalIEQResult into a flat summary dict for structured LLM context.

    Parameters
    ----------
    ieq_result : GlobalIEQResult
        The Global IEQ Index calculation result.
    compliance_report : ComplianceReport or None
        Optional compliance report.

    Returns
    -------
    dict
        Flat dictionary with scalar aggregates.
    """
    domain_avgs = {
        domain: round(float(np.mean(scores)), 1) for domain, scores in ieq_result.domain_scores.items()
    }
    worst_domain = min(domain_avgs, key=domain_avgs.get)

    summary: dict[str, float | str | int | list[str]] = {
        "ieq_index_avg": round(float(np.mean(ieq_result.index)), 1),
        "ieq_index_min": round(float(np.min(ieq_result.index)), 1),
        "ieq_index_max": round(float(np.max(ieq_result.index)), 1),
        "ieq_index_std": round(float(np.std(ieq_result.index)), 1),
        "n_timestamps": ieq_result.n_timestamps,
        "domains": ieq_result.domains,
        "domain_scores_avg": domain_avgs,
        "worst_domain": worst_domain,
        "weights_used": ieq_result.weights_used,
    }

    if compliance_report is not None:
        summary["compliance_rate_pct"] = round(compliance_report.compliance_rate_pct, 1)
        summary["threshold"] = compliance_report.threshold
        summary["compliant_hours"] = round(compliance_report.compliant_hours, 0)
        summary["total_hours"] = round(compliance_report.total_hours, 0)

    return summary


def generate_markdown_summary(
    df: pd.DataFrame,
    window_hours: int = 24,
    threshold: float = 80.0,
    zone_id: str | None = None,
) -> str:
    """Run the full IEQ pipeline on a DataFrame and produce a dense markdown report.

    Evaluates all available domains from the sensor data, computes the Global
    IEQ Index, generates a compliance report, and returns a structured markdown
    string listing critical failures with timestamps.

    Parameters
    ----------
    df : pandas.DataFrame
        Sensor data with columns matching comfio canonical names or aliases.
    window_hours : int, default 24
        Expected number of hours (timestamps) per evaluation window.
    threshold : float, default 80.0
        IEQ Index threshold for compliance.
    zone_id : str or None
        Optional zone identifier.

    Returns
    -------
    str
        Markdown report string.
    """
    sensor = SensorData(df=df)
    sensor.validate()
    domains = sensor.available_domains()

    thermal_res = None
    visual_res = None
    acoustic_res = None
    iaq_res = None

    if "thermal" in domains:
        thermal_res = evaluate_thermal(
            tdb=sensor.get_validated("air_temp_c"),
            tr=sensor.get_validated("radiant_temp_c"),
            vr=sensor.get_validated("air_velocity_ms"),
            rh=sensor.get_validated("relative_humidity_pct"),
            met=sensor.get_validated("metabolic_rate_met")
            if "metabolic_rate_met" in sensor.column_map
            else 1.2,
            clo=sensor.get_validated("clothing_insulation_clo")
            if "clothing_insulation_clo" in sensor.column_map
            else 0.5,
        )

    if "visual" in domains:
        visual_res = evaluate_visual(
            illuminance=sensor.get_validated("illuminance_lux"),
            task_type="general",
        )

    if "acoustic" in domains:
        acoustic_res = evaluate_acoustic(
            laeq=sensor.get_validated("noise_laeq_db"),
        )

    if "iaq" in domains:
        iaq_res = evaluate_iaq(
            co2=sensor.get_validated("co2_ppm"),
        )

    # Pollutant IAQ detection
    pollutant_res = None
    advanced = sensor.available_advanced_domains()
    if "pollutant_iaq" in advanced:
        pollutant_kwargs = {}
        for col, key in [
            ("pm25_ugm3", "pm25"), ("pm10_ugm3", "pm10"),
            ("tvoc_ugm3", "tvoc"), ("formaldehyde_ppb", "formaldehyde"),
            ("co_ppm", "co"),
        ]:
            if col in sensor.column_map:
                pollutant_kwargs[key] = sensor.get_validated(col)
        if pollutant_kwargs:
            pollutant_res = evaluate_iaq_pollutants(**pollutant_kwargs)

    ieq_result = calculate_global_ieq(
        thermal=thermal_res,
        visual=visual_res,
        acoustic=acoustic_res,
        iaq=iaq_res,
        pollutant_iaq=pollutant_res,
    )

    report = calculate_compliance(ieq_result, threshold=threshold)

    zone_label = f"Zone {zone_id}" if zone_id else "Building"
    md = f"## IEQ Report: {zone_label}\n"
    md += f"* **Global IEQ Average:** {report.ieq_index_avg:.1f}/100\n"
    md += f"* **Compliance Rate:** {report.compliance_rate_pct:.1f}% (Threshold > {threshold:.0f})\n"
    md += f"* **Timestamps:** {ieq_result.n_timestamps}\n\n"

    md += "### Domain Summary\n"
    for domain, avg in report.domain_scores_avg.items():
        compliance = report.domain_compliance.get(domain, 0.0)
        md += f"* **{domain.upper()}:** avg={avg:.1f}/100, compliance={compliance:.1f}%\n"

    md += "\n### Critical Failures\n"
    non_compliant_mask = ieq_result.index < threshold
    failure_indices = np.where(non_compliant_mask)[0]

    if len(failure_indices) == 0:
        md += "* No critical failures detected.\n"
    else:
        worst_domain_scores = {}
        for domain, scores in ieq_result.domain_scores.items():
            worst_domain_scores[domain] = float(np.mean(scores[failure_indices]))

        worst_domain = min(worst_domain_scores, key=worst_domain_scores.get)
        md += f"* **{len(failure_indices)} timestamps** below threshold (IEQ < {threshold:.0f}).\n"
        md += f"* Primary cause: **{worst_domain.upper()}** domain "
        md += f"(avg score during failures: {worst_domain_scores[worst_domain]:.1f}/100).\n"
        md += f"* {_DOMAIN_DIAGNOSTICS.get(worst_domain, '')}\n"

    return md
