"""CSV export for IEQ results.

Provides :func:`ieq_to_csv` to export per-timestamp domain scores,
IEQ index, and compliance flags as a CSV string.
"""

from __future__ import annotations

import io

import numpy as np
import pandas as pd

from comfio.integration.global_ieq import GlobalIEQResult
from comfio.performance.contracts import ComplianceReport


def ieq_to_csv(
    ieq_result: GlobalIEQResult,
    compliance_report: ComplianceReport | None = None,
) -> str:
    """Export per-timestamp IEQ scores and compliance flags to CSV.

    Parameters
    ----------
    ieq_result : GlobalIEQResult
        The Global IEQ Index calculation result.
    compliance_report : ComplianceReport or None
        Optional compliance report for threshold context.

    Returns
    -------
    str
        CSV string with one row per timestamp.

    Examples
    --------
    >>> import numpy as np
    >>> from comfio import evaluate_thermal, calculate_global_ieq, ieq_to_csv
    >>> thermal = evaluate_thermal(
    ...     tdb=np.array([24.0, 25.0]),
    ...     tr=np.array([24.0, 25.0]),
    ...     vr=np.array([0.1, 0.1]),
    ...     rh=np.array([50.0, 50.0]),
    ...     met=1.2, clo=0.5,
    ... )
    >>> ieq = calculate_global_ieq(thermal=thermal)
    >>> csv_str = ieq_to_csv(ieq)
    >>> "ieq_index" in csv_str
    True
    >>> csv_str.count("\\n")
    3
    """
    data: dict[str, np.ndarray] = {"ieq_index": ieq_result.index}

    for domain, scores in ieq_result.domain_scores.items():
        data[f"{domain}_score"] = scores

    if compliance_report is not None:
        threshold = compliance_report.threshold
        data["compliant"] = (ieq_result.index >= threshold).astype(int)
        data["threshold"] = np.full(ieq_result.n_timestamps, threshold)

    df = pd.DataFrame(data)
    buf = io.StringIO()
    df.to_csv(buf, index_label="timestamp_index", float_format="%.2f")
    return buf.getvalue()
