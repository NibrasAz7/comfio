"""Export buttons for the comfio Streamlit app (Act 3 — Evidence)."""

from __future__ import annotations

from typing import Any

import streamlit as st


def render_exports(
    ieq_result: Any,
    compliance_report: Any,
    zone_id: str,
    pipeline_config: dict[str, Any],
) -> None:
    """Render the six download buttons: CSV, DOCX, PDF, JSON, MD, script."""
    from comfio.llm.interpreters import ieq_to_markdown
    from comfio.reports import ieq_to_csv, ieq_to_docx, ieq_to_pdf
    from comfio.reports.script_export import generate_pipeline_script

    st.subheader("Evidence export")
    cols = st.columns(6)

    cols[0].download_button(
        "CSV",
        data=ieq_to_csv(ieq_result, compliance_report),
        file_name="ieq_report.csv",
        mime="text/csv",
    )

    try:
        cols[1].download_button(
            "DOCX",
            data=ieq_to_docx(ieq_result, compliance_report, zone_id=zone_id),
            file_name="ieq_report.docx",
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        )
    except ImportError:
        cols[1].caption("DOCX needs `pip install comfio[gui-reports]`")

    try:
        cols[2].download_button(
            "PDF",
            data=ieq_to_pdf(ieq_result, compliance_report, zone_id=zone_id),
            file_name="ieq_report.pdf",
            mime="application/pdf",
        )
    except ImportError:
        cols[2].caption("PDF needs `pip install comfio[gui-reports]`")

    cols[3].download_button(
        "Contract JSON",
        data=compliance_report.to_contract_json(),
        file_name="compliance_contract.json",
        mime="application/json",
    )

    cols[4].download_button(
        "Markdown",
        data=ieq_to_markdown(ieq_result, compliance_report, zone_id=zone_id),
        file_name="ieq_summary.md",
        mime="text/markdown",
    )

    cols[5].download_button(
        "Python script",
        data=generate_pipeline_script(pipeline_config),
        file_name="comfio_pipeline.py",
        mime="text/x-python",
        help="Standalone script reproducing this analysis — the GUI as API on-ramp.",
    )
