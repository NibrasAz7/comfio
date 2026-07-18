# API Reference

Auto-generated documentation for every public function, class, and module in comfio.

---

## Core

- [SensorData](data_handler.md) — Data ingestion, column auto-detection, validation
- [Validation](validation.md) — Input array validation utilities
- [ResultBase](result_base.md) — to_dict / to_json / to_dataframe mixin (v0.1.6)

## Domains

- [Thermal](domains/thermal.md) — PMV/PPD evaluation (ISO 7730 / ASHRAE 55)
- [Local Discomfort](domains/thermal_local.md) — Ankle draft & vertical gradient (v0.1.6)
- [sPMV](domains/thermal_spmv.md) — Simplified PMV (Buratti et al. 2009)
- [Adaptive Thermal](domains/thermal_adaptive.md) — ASHRAE 55-2023 & EN 16798-1:2019
- [TSV](domains/thermal_tsv.md) — TSV augmentation and compliance
- [Personalised Thermal](domains/thermal_personal.md) — OLS regression personalisation
- [Visual](domains/visual.md) — Illuminance evaluation (EN 12464-1)
- [Acoustic](domains/acoustic.md) — Noise level evaluation (NC curves)
- [IAQ](domains/iaq.md) — CO₂-based IAQ evaluation (ASHRAE 62.1)
- [Pollutant IAQ](domains/iaq_pollutants.md) — PM2.5, PM10, TVOC, formaldehyde, CO

## Integration

- [Global IEQ](integration.md) — Multi-domain weighted aggregation
- [Weather](weather.md) — Outdoor temperature fetch via meteostat (v0.1.6)

## Performance & Contracts

- [Compliance](performance.md) — Compliance rates, contract JSON generation

## Utilities

- [Logging](logging.md) — comfio logger configuration (v0.1.6)

## ML/DL Adapters

- [ML Adapters](ml_adapters.md) — scikit-learn, PyTorch, TensorFlow/Keras

## LLM Tools

- [LLM Tools](llm_tools.md) — OpenAI/LangChain tool schemas, diagnostic prompts
