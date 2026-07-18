# Changelog

All notable changes to comfio will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

Nothing yet.

## [0.1.6] - 2026-07-18

### Added

- **Local thermal discomfort**: `evaluate_ankle_draft()` and `evaluate_vertical_gradient()` wrapping pythermalcomfort's ASHRAE 55 / ISO 7730 local discomfort models, plus `local_discomfort_score()` for combining both into a 0-100 score
- **Weather integration**: `fetch_outdoor_temperature()`, `fetch_prevailing_temp()`, and `fetch_running_mean()` — retrieve historical outdoor temperature via `meteostat` for adaptive comfort models (ASHRAE 55 prevailing mean, EN 16798-1 running mean). Results cached under `~/.cache/comfio/weather/`
- **ResultBase mixin**: all 16+ Result dataclasses now provide `to_dict()`, `to_json()`, and `to_dataframe()` methods for easy serialization
- **Logging**: `setup_logging()` function and `logging.getLogger(__name__)` calls across `pipeline.py` — silent `except Exception` blocks now emit `logger.warning()` to stderr

### Changed

- **Python floor bumped from 3.10 to 3.11** (required by `meteostat`; Python 3.10 reaches EOL Oct 2026)
- `meteostat>=2.1` added to core dependencies
- CI matrix updated to test Python 3.11, 3.12, 3.13 (3.10 dropped)
- Ruff `target-version` and mypy `python_version` updated to 3.11
- Development Status classifier remains `4 - Beta`

### Documentation

- New theory article: [Local Discomfort](theory/local_discomfort.md)
- New API reference pages: thermal_local, weather, logging, result_base
- Limitations page updated with weather integration caveats
- mkdocs.yml navigation updated with all new pages

## [0.1.5] - 2026-07-07

### Added

- `time_aware` parameter to `augment_tsv_cdf` for time-interpolated CDF remapping that preserves temporal coherence
- `evaluate_seasonal_personalised_pmv` — auto-selects per-season personalisation index for Fanger PMV
- `evaluate_seasonal_personalised_adaptive` — auto-selects per-season personalisation index for adaptive comfort
- Docs: time-aware TSV augmentation how-to and theory sections

## [0.1.4] - 2026-07-07

### Added

- `evaluate_seasonal_personalised_spmv` — auto-selects per-season personalisation index for sPMV in a single call

### Changed

- Author metadata updated to "Nibras Abo Alzahab" across `pyproject.toml` and `CITATION.cff`
- GitHub Actions docs workflow: added `permissions: contents: write` for `gh-pages` deployment

### Fixed

- `mkdocs.yml`: removed unsupported mkdocstrings options (`show_docstring_notes`, `show_docstring_examples`)
- CI workflow paths corrected from `src/comfortpy/` to `src/comfio/` (mypy + pytest)
- PyPI publish workflow URL corrected from `comfortpy` to `comfio`

## [0.1.2] - 2026-07-06

### Fixed

- CI workflow paths corrected from `src/comfortpy/` to `src/comfio/` (mypy + pytest)
- PyPI publish workflow URL corrected from `comfortpy` to `comfio`
- 157 ruff lint errors fixed (unused imports, E712 comparisons, import sorting, line length)
- 33 files reformatted to satisfy `ruff format --check`
- Renamed `comfioError` to `ComfioError` (PEP 8 CapWords convention)
- Replaced `Union` with modern `X | Y` syntax in `units.py`
- Merged nested `if` in `data_handler.py` (SIM102)
- Broke long LaTeX docstring lines in `thermal_tsv.py`
- Added `__all__` to `domains/__init__.py` for re-export clarity
- Relaxed mypy `strict` mode to per-module overrides for intentional type patterns

## [0.1.1] - 2026-07-05

### Fixed

- Project URLs corrected from `NibrasAz7/Comfio` to `NibrasAz7/comfio` (all lowercase)
- Documentation regenerated after corruption: all docs/ pages restored
- GUIDE.md and TESTS.md restored with full content
- Example scripts restored in examples/
- mkdocs build --strict passes with 0 warnings

### Changed

- Version bumped to 0.1.1
- All internal references unified to lowercase `comfio`

## [0.1.0] - 2026-07-05

### Added

- Multi-domain IEQ framework: thermal, visual, acoustic, IAQ
- Thermal comfort scoring via PMV/PPD (ISO 7730, ASHRAE 55)
- Simplified PMV (sPMV) with seasonal Buratti model
- Adaptive thermal comfort (ASHRAE 55-2023, EN 16798-1:2019)
- Personalised thermal comfort via OLS regression
- TSV augmentation with CDF remapping
- IAQ pollutant scoring (PM2.5, PM10, TVOC, formaldehyde, CO)
- Advanced psychrometrics via psychrolib
- Ventilation rate estimation from CO₂ decay
- Global IEQ Index with configurable weighting schemas
- Performance contract compliance reporting with JSON export
- LLM-native tool schemas (OpenAI function calling, LangChain)
- scikit-learn transformer for IEQ feature extraction
- PyTorch dataset and Keras preprocessing layer for ML pipelines
- Sensor data handler with unit validation
- Comprehensive doctests across all domain modules

### Fixed

- Psychrolib 2.5.0 + numba 0.65 compatibility (pure Python fallback)
- Doctest expected values corrected for thermal, adaptive, sPMV, and IAQ scores
- ML adapter doctests marked `+SKIP` for optional dependencies
