# comfio — Test Documentation

> **Test suite overview, coverage, and running instructions for the comfio package.**

---

## Table of Contents

1. [Running Tests](#1-running-tests)
2. [Test Structure](#2-test-structure)
3. [Test Categories](#3-test-categories)
4. [Coverage Report](#4-coverage-report)
5. [Doctests](#5-doctests)
6. [Continuous Integration](#6-continuous-integration)

---

## 1. Running Tests

### Install test dependencies

```bash
pip install comfio[all] pytest pytest-cov
```

### Run the full suite

```bash
pytest
```

### Run with coverage

```bash
pytest --cov=comfio --cov-report=term-missing
```

### Run a specific test file

```bash
pytest tests/test_thermal.py -v
```

### Run doctests

```bash
pytest --doctest-modules src/comfio/
```

---

## 2. Test Structure

```
tests/
├── conftest.py                 # Shared fixtures (synthetic data, result objects)
├── test_thermal.py             # PMV/PPD evaluation
├── test_thermal_spmv.py        # Simplified PMV
├── test_thermal_adaptive.py    # ASHRAE 55 & EN 16798 adaptive
├── test_thermal_tsv.py         # TSV augmentation & compliance
├── test_thermal_personal.py    # Personalised comfort
├── test_visual.py              # Visual comfort
├── test_acoustic.py            # Acoustic comfort
├── test_iaq.py                 # CO₂-based IAQ
├── test_iaq_pollutants.py      # Pollutant IAQ
├── test_global_ieq.py          # Global IEQ aggregation
├── test_weights.py             # Weighting schemas
├── test_compliance.py          # Compliance & contract JSON
├── test_data_handler.py        # SensorData
├── test_validation.py          # Input validation
├── test_ml_adapters.py         # ML/DL adapters (skipped if deps missing)
├── test_llm_tools.py           # LLM tool schemas & prompts
└── test_integration.py         # End-to-end integration tests
```

---

## 3. Test Categories

### Unit Tests

Each domain module has a dedicated test file that verifies:

- **Score range**: All scores are within 0–100
- **Compliance logic**: `compliant` array matches expected category limits
- **Edge cases**: Empty arrays, single-element arrays, extreme values
- **NaN handling**: NaN inputs produce NaN scores (not crashes)
- **Array shapes**: Output arrays match input array length

### Integration Tests

`test_integration.py` tests the full pipeline end-to-end:

```python
def test_full_pipeline():
    """Sensor data → domain evaluations → Global IEQ → compliance report."""
    thermal = evaluate_thermal(...)
    visual = evaluate_visual(...)
    acoustic = evaluate_acoustic(...)
    iaq = evaluate_iaq(...)
    ieq = calculate_global_ieq(thermal=thermal, visual=visual, ...)
    report = calculate_compliance(ieq, threshold=80.0)
    assert report.compliance_rate_pct >= 0.0
    assert report.compliance_rate_pct <= 100.0
    assert isinstance(report.to_contract_json(), str)
```

### Optional Dependency Tests

Tests for ML/DL and advanced physics modules are marked with `pytest.mark.skipif` to skip when optional dependencies are not installed:

```python
pytest.importorskip("torch")
pytest.importorskip("tensorflow")
pytest.importorskip("pyradiance")
pytest.importorskip("colour")
pytest.importorskip("pyroomacoustics")
```

---

## 4. Coverage Report

Run coverage with:

```bash
pytest --cov=comfio --cov-report=html
```

This generates `htmlcov/index.html` with line-by-line coverage.

### Target Coverage

| Module | Target | Notes |
|---|---|---|
| `domains/thermal.py` | ≥ 95% | Core PMV/PPD logic |
| `domains/iaq.py` | ≥ 95% | CO₂ scoring |
| `domains/iaq_pollutants.py` | ≥ 90% | Pollutant thresholds |
| `integration/global_ieq.py` | ≥ 95% | Aggregation logic |
| `performance/contracts.py` | ≥ 95% | Contract JSON |
| `core/data_handler.py` | ≥ 90% | Column detection |
| `ml/*` | ≥ 80% | Optional deps |
| `llm/*` | ≥ 85% | Tool schemas |

---

## 5. Doctests

comfio uses NumPy-style docstrings with doctests for key functions. Run them with:

```bash
pytest --doctest-modules src/comfio/
```

Doctests verify that the examples in docstrings are executable and produce correct output. They serve as both documentation and regression tests.

---

## 6. Continuous Integration

### GitHub Actions Workflow

```yaml
# .github/workflows/tests.yml
name: Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.10", "3.11", "3.12"]
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}
      - run: pip install comfio[all] pytest pytest-cov
      - run: pytest --cov=comfio --cov-report=xml
      - uses: codecov/codecov-action@v4
```

### MkDocs Strict Build

Documentation builds are verified with:

```bash
mkdocs build --strict
```

This catches broken links, malformed docstrings, and missing references.

---

## Test Fixtures

Shared fixtures in `conftest.py`:

- `synthetic_thermal_data()` — Random thermal sensor arrays
- `synthetic_visual_data()` — Random illuminance arrays
- `synthetic_acoustic_data()` — Random noise level arrays
- `synthetic_iaq_data()` — Random CO₂ arrays
- `sample_ieq_result()` — Pre-computed GlobalIEQResult for compliance tests
- `sample_compliance_report()` — Pre-computed ComplianceReport for JSON tests

---

## Reporting Issues

If you find a bug or have a test failure, please [open an issue](https://github.com/NibrasAz7/comfio/issues) with:

1. Python version
2. comfio version (`pip show comfio`)
3. Installed extras (`pip list | grep comfio`)
4. Full traceback
5. Minimal reproduction example
