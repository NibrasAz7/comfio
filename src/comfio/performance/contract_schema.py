"""Formal schema mapping IEQ compliance data to smart contract inputs.

Defines an ABI-like structure that describes how Global IEQ compliance
data maps to Solidity function parameters for blockchain Oracle integration.
This is a schema definition, not actual blockchain interaction code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# Solidity type mapping for contract inputs
SOLIDITY_TYPES = {
    "uint8": "uint8",
    "uint16": "uint16",
    "uint32": "uint32",
    "uint256": "uint256",
    "int8": "int8",
    "int256": "int256",
    "bool": "bool",
    "string": "string",
    "bytes32": "bytes32",
    "address": "address",
}


@dataclass
class ContractField:
    """A single field in the smart contract schema.

    Attributes
    ----------
    name : str
        Field name (matches Solidity parameter name).
    solidity_type : str
        Solidity type (e.g., "uint256", "bool").
    description : str
        Human-readable description.
    source : str
        Which comfio output this field maps to.
    """

    name: str
    solidity_type: str
    description: str
    source: str


@dataclass
class ContractSchema:
    """Schema defining how IEQ compliance data maps to a smart contract.

    Attributes
    ----------
    contract_name : str
        Name of the target smart contract.
    fields : list[ContractField]
        Ordered list of fields matching the contract function signature.
    function_name : str
        Name of the Solidity function that receives this data.
    """

    contract_name: str
    function_name: str
    fields: list[ContractField] = field(default_factory=list)

    def to_abi(self) -> dict[str, Any]:
        """Generate a JSON-serializable ABI fragment.

        Returns
        -------
        dict
            ABI-compatible dictionary describing the function inputs.
        """
        return {
            "name": self.function_name,
            "type": "function",
            "stateMutability": "nonpayable",
            "inputs": [
                {
                    "name": f.name,
                    "type": f.solidity_type,
                    "internalType": f.solidity_type,
                }
                for f in self.fields
            ],
            "outputs": [],
        }

    def to_dict(self) -> dict[str, Any]:
        """Return a human-readable schema description.

        Returns
        -------
        dict
            Dictionary with contract name, function, and field details.
        """
        return {
            "contract_name": self.contract_name,
            "function_name": self.function_name,
            "fields": [
                {
                    "name": f.name,
                    "type": f.solidity_type,
                    "description": f.description,
                    "source": f.source,
                }
                for f in self.fields
            ],
        }


def default_compliance_schema() -> ContractSchema:
    """Return the default IEQ compliance contract schema.

    This schema maps a compliance report to a Solidity function that
    a smart lease contract would call to trigger penalties/rewards.

    Returns
    -------
    ContractSchema
        Schema with standard fields for IEQ compliance reporting.
    """
    return ContractSchema(
        contract_name="IEQComplianceOracle",
        function_name="submitCompliance",
        fields=[
            ContractField(
                name="periodStart",
                solidity_type="uint256",
                description="Unix timestamp of the reporting period start.",
                source="report.period_start",
            ),
            ContractField(
                name="periodEnd",
                solidity_type="uint256",
                description="Unix timestamp of the reporting period end.",
                source="report.period_end",
            ),
            ContractField(
                name="ieqIndexAvg",
                solidity_type="uint8",
                description="Average Global IEQ Index (0-100) over the period.",
                source="report.ieq_index_avg",
            ),
            ContractField(
                name="complianceRatePct",
                solidity_type="uint8",
                description="Percentage of occupied hours that met the IEQ threshold (0-100).",
                source="report.compliance_rate_pct",
            ),
            ContractField(
                name="thermalCompliant",
                solidity_type="bool",
                description="Whether thermal comfort compliance was met.",
                source="report.domain_compliance.thermal",
            ),
            ContractField(
                name="visualCompliant",
                solidity_type="bool",
                description="Whether visual comfort compliance was met.",
                source="report.domain_compliance.visual",
            ),
            ContractField(
                name="acousticCompliant",
                solidity_type="bool",
                description="Whether acoustic comfort compliance was met.",
                source="report.domain_compliance.acoustic",
            ),
            ContractField(
                name="iaqCompliant",
                solidity_type="bool",
                description="Whether IAQ compliance was met.",
                source="report.domain_compliance.iaq",
            ),
            ContractField(
                name="totalOccupiedHours",
                solidity_type="uint32",
                description="Total number of occupied hours in the reporting period.",
                source="report.total_occupied_hours",
            ),
            ContractField(
                name="compliantHours",
                solidity_type="uint32",
                description="Number of occupied hours that met the IEQ threshold.",
                source="report.compliant_hours",
            ),
        ],
    )
