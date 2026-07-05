"""Compliance rate calculation and contract-ready JSON report generation.

Translates Global IEQ Index arrays into time-based compliance metrics
(e.g., "The space maintained >80% IEQ for 95% of occupied hours") and
generates structured JSON outputs ready to be sent to blockchain Oracles.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from comfio.integration.global_ieq import GlobalIEQResult
from comfio.performance.contract_schema import ContractSchema, default_compliance_schema


@dataclass
class ComplianceReport:
    """Structured compliance report for a reporting period.

    Attributes
    ----------
    period_start : float
        Unix timestamp of the period start.
    period_end : float
        Unix timestamp of the period end.
    ieq_index_avg : float
        Average Global IEQ Index over the period (0-100).
    ieq_index_min : float
        Minimum Global IEQ Index (0-100).
    ieq_index_max : float
        Maximum Global IEQ Index (0-100).
    ieq_index_std : float
        Standard deviation of the Global IEQ Index.
    compliance_rate_pct : float
        Percentage of timestamps that met the threshold (0-100).
    threshold : float
        IEQ Index threshold used for compliance (default 80).
    total_hours : float
        Total hours in the reporting period.
    compliant_hours : float
        Hours that met the threshold.
    domain_compliance : dict[str, float]
        Per-domain compliance rates (0-100%).
    domain_scores_avg : dict[str, float]
        Per-domain average scores (0-100).
    weights_used : dict[str, float]
        Weights applied in the IEQ calculation.
    domains : list[str]
        Domains included in the calculation.
    """

    period_start: float
    period_end: float
    ieq_index_avg: float
    ieq_index_min: float
    ieq_index_max: float
    ieq_index_std: float
    compliance_rate_pct: float
    threshold: float
    total_hours: float
    compliant_hours: float
    domain_compliance: dict[str, float] = field(default_factory=dict)
    domain_scores_avg: dict[str, float] = field(default_factory=dict)
    weights_used: dict[str, float] = field(default_factory=dict)
    domains: list[str] = field(default_factory=list)

    def to_json(self, indent: int = 2) -> str:
        """Serialize the report to a JSON string.

        Parameters
        ----------
        indent : int
            JSON indentation level.

        Returns
        -------
        str
            JSON string of the compliance report.

        Examples
        --------
        >>> report = ComplianceReport(
        ...     period_start=0.0, period_end=3600.0,
        ...     ieq_index_avg=85.0, ieq_index_min=70.0,
        ...     ieq_index_max=95.0, ieq_index_std=5.0,
        ...     compliance_rate_pct=90.0, threshold=80.0,
        ...     total_hours=1.0, compliant_hours=0.9,
        ... )
        >>> import json
        >>> data = json.loads(report.to_json())
        >>> data['ieq_index_avg']
        85.0
        """
        return json.dumps(asdict(self), indent=indent)

    def to_contract_payload(
        self,
        schema: ContractSchema | None = None,
    ) -> dict[str, Any]:
        """Generate a payload matching the smart contract schema.

        Parameters
        ----------
        schema : ContractSchema or None
            Contract schema to map against. Uses default if None.

        Returns
        -------
        dict
            Dictionary with field names matching the contract schema,
            with values converted to Solidity-compatible types.
        """
        if schema is None:
            schema = default_compliance_schema()

        # Map report fields to contract fields
        source_map: dict[str, Any] = {
            "report.period_start": int(self.period_start),
            "report.period_end": int(self.period_end),
            "report.ieq_index_avg": int(round(self.ieq_index_avg)),
            "report.compliance_rate_pct": int(round(self.compliance_rate_pct)),
            "report.domain_compliance.thermal": bool(
                self.domain_compliance.get("thermal", 0.0) >= 80.0
            ),
            "report.domain_compliance.visual": bool(
                self.domain_compliance.get("visual", 0.0) >= 80.0
            ),
            "report.domain_compliance.acoustic": bool(
                self.domain_compliance.get("acoustic", 0.0) >= 80.0
            ),
            "report.domain_compliance.iaq": bool(
                self.domain_compliance.get("iaq", 0.0) >= 80.0
            ),
            "report.total_occupied_hours": int(round(self.total_hours)),
            "report.compliant_hours": int(round(self.compliant_hours)),
        }

        payload: dict[str, Any] = {}
        for flapjack in schema.fields:
            payload[flapjack.name] = source_map.get(flapjack.source)
        return payload

    def to_contract_json(self, schema: ContractSchema | None = None) -> str:
        """Generate a JSON string matching the smart contract schema.

        Parameters
        ----------
        schema : ContractSchema or None
            Contract schema to map against.

        Returns
        -------
        str
            JSON string with contract-compatible field names and types.
        """
        return json.dumps(self.to_contract_payload(schema), indent=2)


def calculate_compliance(
    ieq_result: GlobalIEQResult,
    threshold: float = 80.0,
    period_start: float | None = None,
    period_end: float | None = None,
    domain_compliant_arrays: dict[str, np.ndarray] | None = None,
) -> ComplianceReport:
    """Calculate compliance rates from a Global IEQ result.

    Parameters
    ----------
    ieq_result : GlobalIEQResult
        The Global IEQ Index calculation result.
    threshold : float, default 80.0
        Minimum IEQ Index value to count as "compliant" (0-100).
    period_start : float or None
        Unix timestamp of period start. Defaults to current time minus period duration.
    period_end : float or None
        Unix timestamp of period end. Defaults to current time.
    domain_compliant_arrays : dict[str, np.ndarray] or None
        Per-domain boolean compliance arrays. If provided, domain-level
        compliance rates are calculated. If None, domain compliance is
        derived from domain scores >= 80.

    Returns
    -------
    ComplianceReport
        Structured compliance report with all metrics.

    Notes
    -----
    Compliance rate is the fraction of timestamps where the IEQ Index
    meets or exceeds the threshold:

    .. math::

        \text{compliance\_rate} = 100 \times \frac{1}{n} \sum_{i=1}^{n}
            \mathbb{1}(\text{IEQ}_i \geq \text{threshold})

    Domain compliance is derived from domain scores ≥ 80 unless
    explicit boolean compliance arrays are provided.
    """
    index = ieq_result.index
    n = ieq_result.n_timestamps

    # IEQ Index statistics
    ieq_avg = float(np.mean(index))
    ieq_min = float(np.min(index))
    ieq_max = float(np.max(index))
    ieq_std = float(np.std(index))

    # Compliance: percentage of timestamps where IEQ >= threshold
    compliant_mask = index >= threshold
    compliance_rate = float(np.mean(compliant_mask) * 100.0)

    # Time estimation: assume 1-hour intervals if no timestamp info
    # (This is a rough estimate; real implementation would use actual timestamps)
    total_hours = float(n)
    compliant_hours = float(np.sum(compliant_mask))

    # Period timestamps
    now = time.time()
    if period_end is None:
        period_end = now
    if period_start is None:
        period_start = period_end - total_hours * 3600.0

    # Domain compliance rates
    domain_compliance: dict[str, float] = {}
    domain_scores_avg: dict[str, float] = {}

    for domain_name, score_arr in ieq_result.domain_scores.items():
        domain_scores_avg[domain_name] = float(np.mean(score_arr))

        if domain_compliant_arrays and domain_name in domain_compliant_arrays:
            domain_compliance[domain_name] = float(
                np.mean(domain_compliant_arrays[domain_name]) * 100.0
            )
        else:
            # Derive from score: >= 80 is compliant
            domain_compliance[domain_name] = float(
                np.mean(score_arr >= 80.0) * 100.0
            )

    return ComplianceReport(
        period_start=period_start,
        period_end=period_end,
        ieq_index_avg=ieq_avg,
        ieq_index_min=ieq_min,
        ieq_index_max=ieq_max,
        ieq_index_std=ieq_std,
        compliance_rate_pct=compliance_rate,
        threshold=threshold,
        total_hours=total_hours,
        compliant_hours=compliant_hours,
        domain_compliance=domain_compliance,
        domain_scores_avg=domain_scores_avg,
        weights_used=ieq_result.weights_used,
        domains=ieq_result.domains,
    )
