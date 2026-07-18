# Limitations & Caveats

## Overview

Scientific transparency is a core principle of comfio. This page documents the known limitations, applicability ranges, and caveats of each module to help users make informed decisions.

---

## Thermal Comfort (PMV)

- **Steady-state assumption**: PMV assumes thermal equilibrium. It does not capture transient effects (e.g., rapid temperature changes, metabolic transitions).
- **Uniform environment**: PMV assumes uniform air temperature, radiant temperature, and air velocity. In practice, stratification and local drafts cause deviations.
- **Applicability range**: Valid for 0 < met < 4, 0 < clo < 2, 10–40 °C, 0–100% RH. Outside these ranges, predictions become unreliable.
- **Not for outdoor spaces**: PMV is designed for indoor environments. Outdoor comfort requires UTCI or other models.
- **Individual prediction**: PMV predicts the *mean* vote of a large group. Individual comfort varies ±1 PMV unit.

---

## Adaptive Comfort

- **Naturally ventilated only**: The adaptive model is not applicable to mechanically cooled buildings.
- **Prevailing mean temperature**: Requires accurate outdoor temperature data over the preceding 7–30 days.
- **Occupant control**: Assumes occupants can open windows and adjust clothing. In buildings with locked windows, the model overestimates comfort.

---

## Simplified PMV (sPMV)

- **Default assumptions**: Uses fixed metabolic rate (1.2 met), clothing insulation (season-dependent), and air velocity (0.1 m/s). Deviations from these defaults are not captured.
- **Reduced accuracy**: Compared to full PMV, sPMV has higher error in non-typical conditions (high activity, unusual clothing, high air velocity).
- **No radiant temperature**: sPMV does not account for radiant asymmetry or hot/cold surfaces.

---

## TSV Augmentation

- **Distribution preservation only**: CDF remapping preserves the statistical distribution but does not capture temporal dynamics (e.g., thermal lag, adaptation effects).
- **Random sampling**: The augmentation uses random quantile draws. Results vary between runs unless a seed is set.
- **Not a substitute for surveys**: Augmented votes are synthetic. Real occupant feedback should be collected when possible.

---

## Pollutant IAQ

- **Threshold-based**: Scoring is based on fixed thresholds. Actual health effects depend on individual susceptibility, exposure duration, and pollutant mixtures.
- **No synergistic effects**: Pollutants are scored independently. Combined effects (e.g., PM2.5 + NO₂) are not modeled.
- **Missing pollutants**: Not all indoor pollutants are evaluated (e.g., radon, ozone, VOCs beyond TVOC).

---

## Global IEQ Index

- **Weight subjectivity**: Default weights are based on literature but may not be optimal for all building types, climates, or occupant populations.
- **Linear aggregation**: The weighted sum assumes domain independence. In reality, domains interact (e.g., high temperature increases perceived odor intensity).
- **No temporal dynamics**: The index is computed per timestamp. Cumulative exposure, circadian effects, and adaptation are not captured in the base calculation.

---

## Advanced Physics Modules

- **Radiance daylighting**: Requires Radiance installed on the system. Results depend on sky model and material reflectance assumptions.
- **Reverberation (RT60)**: Uses Sabine/Eyring formulas. Accurate for diffuse fields; less accurate for non-cubic or heavily partitioned spaces.
- **Speech Intelligibility (STI)**: Requires impulse response measurement. Results are sensitive to background noise assumptions.
- **CO₂ decay ventilation**: Assumes well-mixed air and known outdoor CO₂ concentration. In large or partitioned spaces, the well-mixed assumption may not hold.
- **Psychrometrics**: Wraps PsychroLib. Accuracy depends on pressure assumption (sea level default).

---

## ML/DL Integration

- **Feature engineering**: The IEQFeatureExtractor uses a fixed set of features. Custom features require subclassing.
- **PyTorch dataset**: The IEQTimeSeriesDataset assumes regularly sampled data. Irregular time-series require preprocessing.
- **Keras adapter**: TensorFlow is an optional dependency. Not tested on all TF versions.

---

## Weather Integration (v0.1.6)

- **Network dependency**: `fetch_outdoor_temperature()`, `fetch_prevailing_temp()`, and `fetch_running_mean()` make live network requests via the `meteostat` library. Importing `comfio` does **not** trigger any network activity — only explicit calls to these functions do.
- **Data availability**: Meteostat aggregates from national weather services (NOAA, DWD, etc.). Coverage varies by region; some locations may have no nearby stations.
- **Reproducibility**: Historical weather data can change (station corrections, provider updates). For reproducible research, cache the fetched data (automatic under `~/.cache/comfio/weather/`) and pin the cache file.
- **Python ≥3.11 required**: The `meteostat` package requires Python 3.11+. comfio v0.1.6 bumps its minimum Python version accordingly.

---

## Reporting

If you encounter results that seem incorrect or have questions about applicability, please [open an issue](https://github.com/NibrasAz7/comfio/issues).
