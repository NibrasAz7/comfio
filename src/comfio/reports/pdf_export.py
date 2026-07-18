"""PDF report export for IEQ results.

Provides :func:`ieq_to_pdf` to generate a formatted PDF report with
KPIs, domain breakdown, and diagnostic insights. Requires ``reportlab``.
"""

from __future__ import annotations

import io
from typing import Any

import numpy as np

from comfio.integration.global_ieq import GlobalIEQResult
from comfio.llm.interpreters import ieq_to_markdown
from comfio.performance.contracts import ComplianceReport

_IMPORT_ERROR_MSG = (
    "ieq_to_pdf requires reportlab. Install it with: pip install reportlab\n"
    "Or install comfio with reports support: pip install comfio[gui-reports]"
)


def ieq_to_pdf(
    ieq_result: GlobalIEQResult,
    compliance_report: ComplianceReport | None = None,
    zone_id: str | None = None,
) -> bytes:
    """Generate a formatted PDF report from IEQ results.

    Parameters
    ----------
    ieq_result : GlobalIEQResult
        The Global IEQ Index calculation result.
    compliance_report : ComplianceReport or None
        Optional compliance report for compliance context.
    zone_id : str or None
        Optional zone identifier for the report title.

    Returns
    -------
    bytes
        PDF file content.

    Raises
    ------
    ImportError
        If reportlab is not installed.

    Examples
    --------
    >>> import numpy as np
    >>> from comfio import evaluate_thermal, calculate_global_ieq
    >>> from comfio.reports import ieq_to_pdf
    >>> thermal = evaluate_thermal(
    ...     tdb=np.array([24.0, 25.0]),
    ...     tr=np.array([24.0, 25.0]),
    ...     vr=np.array([0.1, 0.1]),
    ...     rh=np.array([50.0, 50.0]),
    ...     met=1.2, clo=0.5,
    ... )
    >>> ieq = calculate_global_ieq(thermal=thermal)
    >>> pdf_bytes = ieq_to_pdf(ieq)  # doctest: +SKIP
    """
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.lib.units import mm
        from reportlab.platypus import (
            Paragraph,
            SimpleDocTemplate,
            Spacer,
            Table,
            TableStyle,
        )
    except ImportError as exc:
        raise ImportError(_IMPORT_ERROR_MSG) from exc

    # Generate markdown content for the body
    md = ieq_to_markdown(ieq_result, compliance_report=compliance_report, zone_id=zone_id)

    # Build PDF
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, margins=20 * mm)
    styles = getSampleStyleSheet()
    story: list[Any] = []

    # Title
    zone_label = f"Zone {zone_id}" if zone_id else "Building"
    story.append(Paragraph(f"IEQ Report: {zone_label}", styles["Title"]))
    story.append(Spacer(1, 12))

    # KPI summary table
    index_avg = float(np.mean(ieq_result.index))
    index_min = float(np.min(ieq_result.index))
    index_max = float(np.max(ieq_result.index))
    index_std = float(np.std(ieq_result.index))

    kpi_data = [
        ["Metric", "Value"],
        ["Global IEQ Index (avg)", f"{index_avg:.1f} / 100"],
        ["IEQ Index (min)", f"{index_min:.1f}"],
        ["IEQ Index (max)", f"{index_max:.1f}"],
        ["IEQ Index (std)", f"{index_std:.1f}"],
        ["Timestamps Evaluated", str(ieq_result.n_timestamps)],
        ["Domains Included", ", ".join(ieq_result.domains)],
    ]

    if compliance_report is not None:
        kpi_data.append(
            [
                "Compliance Rate",
                f"{compliance_report.compliance_rate_pct:.1f}% "
                f"(threshold: {compliance_report.threshold:.0f})",
            ]
        )

    kpi_table = Table(kpi_data, colWidths=[80 * mm, 90 * mm])
    kpi_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f77b4")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTSIZE", (0, 0), (-1, -1), 10),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f0f0f0")]),
            ]
        )
    )
    story.append(kpi_table)
    story.append(Spacer(1, 20))

    # Domain breakdown
    story.append(Paragraph("Domain Breakdown", styles["Heading2"]))
    story.append(Spacer(1, 6))

    domain_data = [["Domain", "Avg Score", "Weight", "Status"]]
    domain_avgs = {d: float(np.mean(s)) for d, s in ieq_result.domain_scores.items()}
    for domain, avg in domain_avgs.items():
        weight = ieq_result.weights_used.get(domain, 0.0)
        status = "OK" if avg >= 70 else "WARNING"
        domain_data.append(
            [
                domain.upper(),
                f"{avg:.1f}",
                f"{weight:.2f}",
                status,
            ]
        )

    domain_table = Table(domain_data, colWidths=[40 * mm, 35 * mm, 30 * mm, 35 * mm])
    domain_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2ca02c")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTSIZE", (0, 0), (-1, -1), 10),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f0f0f0")]),
            ]
        )
    )
    story.append(domain_table)
    story.append(Spacer(1, 20))

    # Diagnostic section
    story.append(Paragraph("Diagnostic Insight", styles["Heading2"]))
    story.append(Spacer(1, 6))

    worst_domain = min(domain_avgs, key=lambda k: domain_avgs[k]) if domain_avgs else "N/A"
    worst_score = domain_avgs.get(worst_domain, 0.0)
    diag_text = (
        f"The primary limiting factor is the <b>{worst_domain.upper()}</b> domain "
        f"(score: {worst_score:.1f}/100). "
    )
    story.append(Paragraph(diag_text, styles["Normal"]))
    story.append(Spacer(1, 12))

    # Full markdown as plain text (stripped of markdown syntax)
    story.append(Paragraph("Full Report", styles["Heading2"]))
    story.append(Spacer(1, 6))
    # Convert markdown to simple paragraphs
    for line in md.split("\n"):
        line = line.strip()
        if not line:
            continue
        # Strip markdown headers/bold for PDF
        clean = line.replace("###", "").replace("##", "").replace("#", "")
        clean = clean.replace("**", "<b>").replace("__", "<b>")
        # Close any opened bold tags (simple approach)
        if clean.count("<b>") % 2 == 1:
            clean += "</b>"
        try:
            story.append(Paragraph(clean, styles["Normal"]))
        except Exception:
            story.append(Paragraph(line, styles["Normal"]))

    doc.build(story)
    return buf.getvalue()
