"""Report generation utilities for comfio IEQ results.

Provides functions for exporting IEQ evaluations to various formats:
CSV, PDF, DOCX, and reproducible Python scripts.

Also provides the intelligent pipeline (:func:`detect_capabilities`,
:func:`run_pipeline`) for automatic domain detection and evaluation.
"""

from __future__ import annotations

from comfio.reports.csv_export import ieq_to_csv
from comfio.reports.pipeline import (
    PipelineResult,
    detect_capabilities,
    run_pipeline,
)
from comfio.reports.script_export import generate_pipeline_script

__all__ = [
    "PipelineResult",
    "detect_capabilities",
    "run_pipeline",
    "ieq_to_csv",
    "ieq_to_pdf",
    "ieq_to_docx",
    "generate_pipeline_script",
]


def ieq_to_pdf(*args, **kwargs):  # type: ignore[no-untyped-def]
    """Lazy-import wrapper for ieq_to_pdf (requires reportlab)."""
    from comfio.reports.pdf_export import ieq_to_pdf as _impl

    return _impl(*args, **kwargs)


def ieq_to_docx(*args, **kwargs):  # type: ignore[no-untyped-def]
    """Lazy-import wrapper for ieq_to_docx (requires python-docx)."""
    from comfio.reports.docx_export import ieq_to_docx as _impl

    return _impl(*args, **kwargs)
