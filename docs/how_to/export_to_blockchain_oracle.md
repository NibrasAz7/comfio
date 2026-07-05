# Export to Blockchain Oracle

## Problem

You need to generate a structured JSON compliance report that can be consumed by a smart contract Oracle for building performance verification on-chain.

---

## Solution

```python
import numpy as np
from comfio import (
    evaluate_thermal, evaluate_visual,
    evaluate_acoustic, evaluate_iaq,
    calculate_global_ieq, calculate_compliance,
)

# 1. Evaluate all domains
thermal = evaluate_thermal(
    tdb=np.array([24.0, 25.0, 26.0]),
    tr=np.array([24.0, 25.0, 26.0]),
    vr=np.array([0.1, 0.1, 0.1]),
    rh=np.array([50.0, 50.0, 50.0]),
    met=1.2, clo=0.5, category="B",
)
visual = evaluate_visual(illuminance=np.array([450.0, 500.0, 600.0]))
acoustic = evaluate_acoustic(laeq=np.array([35.0, 40.0, 45.0]))
iaq = evaluate_iaq(co2=np.array([700.0, 900.0, 1100.0]))

# 2. Calculate Global IEQ
ieq = calculate_global_ieq(
    thermal=thermal, visual=visual,
    acoustic=acoustic, iaq=iaq,
)

# 3. Generate compliance report with contract JSON
report = calculate_compliance(
    ieq_result=ieq,
    threshold=80.0,
    period_start=1717200000.0,  # Unix timestamp
    period_end=1717286400.0,
)

# 4. Export JSON for Oracle
contract_json = report.to_contract_json()
print(contract_json)
```

### JSON Output Structure

```json
{
  "period_start": 1717200000.0,
  "period_end": 1717286400.0,
  "ieq_index_avg": 82.5,
  "ieq_index_min": 65.0,
  "ieq_index_max": 95.0,
  "compliance_rate_pct": 87.5,
  "is_compliant": true,
  "threshold": 80.0,
  "schema_version": "0.1.0"
}
```

### Solidity ABI Mapping

The JSON fields map directly to a Solidity struct:

```solidity
struct ComplianceReport {
    uint256 periodStart;
    uint256 periodEnd;
    uint256 ieqIndexAvg;    // scaled by 100 (8250 = 82.50)
    uint256 ieqIndexMin;
    uint256 ieqIndexMax;
    uint256 complianceRatePct;  // scaled by 100 (8750 = 87.50%)
    bool    isCompliant;
    uint256 threshold;
    string  schemaVersion;
}
```

---

## See Also

- [API Reference — Performance](../reference/performance.md)
- [Theory — Global IEQ Aggregation](../theory/weakest_link_aggregation.md)
