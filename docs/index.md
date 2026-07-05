# comfio

**A multi-domain IEQ & performance contract framework for smart buildings.**

[![PyPI version](https://img.shields.io/pypi/v/comfio.svg)](https://pypi.org/project/comfio/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

comfio bridges the gap between raw building sensor data and actionable smart building management. It unifies **Thermal**, **Visual**, **Acoustic**, and **Indoor Air Quality (IAQ)** metrics into a single **Global IEQ Index** — designed for time-series IoT data and comfort-based performance contracts.

---

## 30-Second Quickstart

```bash
pip install comfio
```

```python
import numpy as np
from comfio import (
    evaluate_thermal, evaluate_visual,
    evaluate_acoustic, evaluate_iaq,
    calculate_global_ieq, calculate_compliance,
)

# Evaluate each domain
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

# Merge into a single 0–100 score
ieq = calculate_global_ieq(
    thermal=thermal, visual=visual,
    acoustic=acoustic, iaq=iaq,
)
print(f"Global IEQ Index: {ieq.index}")

# Compliance tracking & contract-ready JSON
report = calculate_compliance(ieq, threshold=80.0)
print(f"Compliance rate: {report.compliance_rate_pct:.1f}%")
print(report.to_contract_json())
```

---

## Features

| Feature | Description |
|---|---|
| **Multi-Domain IEQ** | Thermal, Visual, Acoustic, IAQ unified into a single 0–100 index |
| **Time-Series Native** | Built for Pandas/NumPy arrays, not single-point calculations |
| **Performance Contracts** | Compliance rates + structured JSON for blockchain Oracle integration |
| **ML/DL Compatible** | Optional adapters for scikit-learn, PyTorch, and TensorFlow/Keras |
| **Advanced Physics** | Radiance daylighting, CRI/CCT, RT60 reverberation, STI, CO₂ decay, psychrometrics |
| **Pollutant IAQ** | PM2.5, PM10, TVOC, formaldehyde, CO against WHO/EPA/WELL thresholds |
| **Adaptive Comfort** | ASHRAE 55-2023 and EN 16798-1:2019 adaptive models |
| **sPMV** | Buratti et al. (2009) seasonal simplified PMV |
| **TSV Augmentation** | CDF-based quantile mapping for sparse occupant vote expansion |
| **Personalised Comfort** | OLS regression personalisation with per-season support |
| **LLM-Native** | Diagnostic prompts, OpenAI/LangChain tool schemas, markdown summaries |

---

## Documentation

Explore comfio through four lenses:

- **[Tutorials](tutorials/index.md)** — Step-by-step Jupyter notebooks covering the full workflow
- **[How-To Guides](how_to/index.md)** — Practical, copy-pasteable solutions to specific problems
- **[Theory](theory/index.md)** — Mathematical background, standards, and equations
- **[API Reference](reference/index.md)** — Auto-generated documentation for every public function

---

## Standards Referenced

| Standard | Domain |
|---|---|
| ISO 7730 | Thermal comfort — PMV/PPD |
| ASHRAE 55 | Thermal environmental conditions |
| ASHRAE 55-2023 Appendix L | TSV compliance (|TSV| ≤ 1.5) |
| EN 16798-1:2019 | Adaptive thermal comfort |
| EN 12464-1 | Lighting of work places |
| ASHRAE 62.1 | Ventilation for acceptable IAQ |
| WHO Air Quality Guidelines (2021) | PM2.5, PM10 thresholds |
| EPA NAAQS | Criteria pollutant thresholds |
| WELL Building Standard v2 | Feature A01 pollutant thresholds |

---

## Installation

```bash
pip install comfio               # core (numpy, pandas, pythermalcomfort)
pip install comfio[ml]           # + scikit-learn
pip install comfio[torch]        # + PyTorch
pip install comfio[keras]        # + TensorFlow/Keras
pip install comfio[all]          # All frameworks + advanced domains
```

Optional physics extras:

```bash
pip install comfio[daylighting]  # pyradiance (Radiance ray-tracing)
pip install comfio[color]        # colour-science (CRI, CCT)
pip install comfio[acoustics]    # python-acoustics + pyroomacoustics
pip install comfio[psychrometrics]  # PsychroLib
```

---

## Citation

```bibtex
@software{comfio,
  author       = {comfio Contributors},
  title        = {comfio: A Multi-Domain IEQ \& Performance Contract Framework for Smart Buildings},
  year         = {2025},
  url          = {https://github.com/NibrasAz7/comfio},
  version      = {0.1.0},
}
```

## License

MIT — see [LICENSE](https://github.com/NibrasAz7/comfio/blob/main/LICENSE).
