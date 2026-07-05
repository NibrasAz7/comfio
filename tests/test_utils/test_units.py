"""Tests for unit conversion utilities."""

from __future__ import annotations

import numpy as np
import pytest

from comfio.utils.units import (
    celsius_to_fahrenheit,
    clo_to_m2kw,
    db_to_pa,
    fahrenheit_to_celsius,
    met_to_wm2,
    mgm3_to_ppm,
    pa_to_db,
    ppm_to_mgm3,
)


class TestTemperatureConversions:
    def test_f_to_c_scalar(self) -> None:
        assert np.isclose(fahrenheit_to_celsius(32.0), 0.0)
        assert np.isclose(fahrenheit_to_celsius(212.0), 100.0)
        assert np.isclose(fahrenheit_to_celsius(98.6), 37.0, atol=0.01)

    def test_c_to_f_scalar(self) -> None:
        assert np.isclose(celsius_to_fahrenheit(0.0), 32.0)
        assert np.isclose(celsius_to_fahrenheit(100.0), 212.0)

    def test_roundtrip(self) -> None:
        for c in [-10.0, 0.0, 25.0, 50.0]:
            assert np.isclose(fahrenheit_to_celsius(celsius_to_fahrenheit(c)), c)

    def test_f_to_c_array(self) -> None:
        arr = np.array([32.0, 212.0, 50.0])
        result = fahrenheit_to_celsius(arr)
        expected = np.array([0.0, 100.0, 10.0])
        np.testing.assert_allclose(result, expected)


class TestGasConversions:
    def test_ppm_to_mgm3_co2(self) -> None:
        # CO2 at 25°C, 101.325 kPa
        result = ppm_to_mgm3(1000.0, molar_mass=44.01, temperature_c=25.0)
        # Expected: 1000 * 44.01 * 101.325 / (8.314 * 298.15) ≈ 1800 mg/m³
        assert 1700 < result < 1900

    def test_mgm3_to_ppm_roundtrip(self) -> None:
        ppm = 800.0
        mgm3 = ppm_to_mgm3(ppm, molar_mass=44.01, temperature_c=25.0)
        back = mgm3_to_ppm(mgm3, molar_mass=44.01, temperature_c=25.0)
        assert np.isclose(back, ppm, rtol=1e-6)


class TestAcousticConversions:
    def test_db_to_pa(self) -> None:
        # 0 dB SPL = reference pressure
        assert np.isclose(db_to_pa(0.0), 2e-5)

    def test_pa_to_db(self) -> None:
        assert np.isclose(pa_to_db(2e-5), 0.0, atol=0.01)

    def test_roundtrip(self) -> None:
        for db in [20.0, 40.0, 60.0, 80.0]:
            pa = db_to_pa(db)
            back = pa_to_db(pa)
            assert np.isclose(back, db, atol=0.01)


class TestClothingAndMetabolic:
    def test_clo_to_m2kw(self) -> None:
        assert np.isclose(clo_to_m2kw(1.0), 0.155)

    def test_met_to_wm2(self) -> None:
        assert np.isclose(met_to_wm2(1.0), 58.2)
