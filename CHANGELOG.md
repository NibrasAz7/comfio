# Changelog

All notable changes to comfio will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.2] - 2025-07-06

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

## [0.1.1] - 2025-07-05

### Fixed

- Project URLs corrected from `NibrasAz7/Comfio` to `NibrasAz7/comfio` (all lowercase)
- Documentation regenerated after corruption: all docs/ pages restored
- GUIDE.md and TESTS.md restored with full content
- Example scripts restored in examples/
- mkdocs build --strict passes with 0 warnings

### Changed

- Version bumped to 0.1.1
- All internal references unified to lowercase `comfio`

## [0.1.0] - 2025-07-05

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
