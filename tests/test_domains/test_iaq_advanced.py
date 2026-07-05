"""Tests for advanced IAQ module (ventilation + psychrometrics).

Tests are skipped if psychrolib is not installed.
"""

import numpy as np
import pytest

pytest.importorskip(
    "psychrolib", reason="psychrolib not installed — pip install comfio[psychrometrics]"
)


class TestEvaluateVentilation:
    """Tests for evaluate_ventilation (CO₂ decay + steady-state ACH)."""

    def test_co2_decay_method(self) -> None:
        """CO₂ decay method should detect ACH from exponential decay phase."""
        from comfio.domains.iaq_advanced import evaluate_ventilation

        # Simulate CO₂ decay: starts at 1200 ppm, decays to 420 ppm
        # with ACH = 2.0 (k = 2.0/3600 per second)
        timestamps = np.arange(0, 7200, 300, dtype=float)  # 0 to 2h, 5-min intervals
        ach_true = 2.0
        k = ach_true / 3600.0  # per second
        co2 = 420.0 + (1200.0 - 420.0) * np.exp(-k * timestamps)

        result = evaluate_ventilation(
            co2=co2,
            timestamps=timestamps,
            outdoor_co2=420.0,
            occupancy_type="office",
        )

        assert result.ach > 0, "ACH should be positive from decay method"
        assert result.ach_method == "co2_decay"
        # Should be within 20% of true ACH
        assert abs(result.ach - ach_true) < 0.5, f"ACH {result.ach} vs true {ach_true}"
        assert result.co2_peak == pytest.approx(1200.0, abs=10.0)

    def test_steady_state_method(self) -> None:
        """Steady-state method should estimate ACH from room parameters."""
        from comfio.domains.iaq_advanced import evaluate_ventilation

        # Simulate steady-state at 1000 ppm with known room params
        co2 = np.full(20, 1000.0)
        timestamps = np.arange(20, dtype=float) * 3600  # hourly

        result = evaluate_ventilation(
            co2=co2,
            timestamps=timestamps,
            outdoor_co2=420.0,
            occupancy_type="office",
            room_volume=50.0,
            n_occupants=5,
        )

        assert result.ach > 0, "ACH should be positive from steady-state"
        assert result.ach_method in ("steady_state", "co2_decay", "unknown")
        assert result.co2_steady_state == pytest.approx(1000.0, abs=50.0)

    def test_compliance_high_ach(self) -> None:
        """High ACH should be compliant for office."""
        from comfio.domains.iaq_advanced import evaluate_ventilation

        # Fast decay → high ACH
        timestamps = np.arange(0, 3600, 60, dtype=float)
        k = 5.0 / 3600.0  # ACH = 5
        co2 = 420.0 + (1500.0 - 420.0) * np.exp(-k * timestamps)

        result = evaluate_ventilation(
            co2=co2,
            timestamps=timestamps,
            outdoor_co2=420.0,
            occupancy_type="office",
        )

        assert bool(result.compliant) is True
        assert result.score > 50.0

    def test_score_range(self) -> None:
        """Score should always be in 0-100 range."""
        from comfio.domains.iaq_advanced import evaluate_ventilation

        # Flat CO₂ at outdoor level → no decay detectable
        co2 = np.full(10, 420.0)
        result = evaluate_ventilation(co2=co2, occupancy_type="general")

        assert 0 <= result.score <= 100

    def test_ventilation_efficiency(self) -> None:
        """Ventilation efficiency should be in 0-1 range."""
        from comfio.domains.iaq_advanced import evaluate_ventilation

        co2 = np.array([800.0, 900.0, 1000.0, 1100.0, 1000.0, 900.0, 800.0])
        result = evaluate_ventilation(co2=co2, outdoor_co2=420.0)

        assert 0 <= result.ventilation_efficiency <= 1.0


class TestGetPsychrometrics:
    """Tests for get_psychrometrics using psychrolib."""

    def test_standard_conditions(self) -> None:
        """Psychrometric properties at 25°C, 50% RH should be reasonable."""
        from comfio.domains.iaq_advanced import get_psychrometrics

        result = get_psychrometrics(tdb=25.0, rh=0.50, pressure=101325.0)

        assert result.tdb == 25.0
        assert result.rh == 0.50
        # Wet bulb should be between dew point and dry bulb
        assert result.tdew < result.twb < result.tdb
        # Dew point for 25°C, 50% RH is approximately 13.9°C
        assert 12.0 < result.tdew < 16.0, f"Dew point {result.tdew} unexpected"
        # Wet bulb should be around 18-20°C
        assert 16.0 < result.twb < 22.0, f"Wet bulb {result.twb} unexpected"
        # Enthalpy should be positive
        assert result.enthalpy > 0
        # Humidity ratio should be positive and small
        assert 0 < result.hum_ratio < 0.1
        # Vapor pressure should be positive
        assert result.vapor_pressure > 0
        # Moist air volume should be around 0.85-0.87 m³/kg
        assert 0.80 < result.moist_air_volume < 0.90
        # Degree of saturation should be 0-1
        assert 0 <= result.degree_of_saturation <= 1

    def test_dry_air(self) -> None:
        """At very low RH, dew point should be very low."""
        from comfio.domains.iaq_advanced import get_psychrometrics

        result = get_psychrometrics(tdb=20.0, rh=0.001, pressure=101325.0)

        # At very low RH, humidity ratio and vapor pressure should be very small
        assert result.hum_ratio == pytest.approx(0.0, abs=1e-3)
        assert result.vapor_pressure < 5.0

    def test_saturated_air(self) -> None:
        """At 100% RH, wet bulb = dry bulb = dew point."""
        from comfio.domains.iaq_advanced import get_psychrometrics

        result = get_psychrometrics(tdb=22.0, rh=1.0, pressure=101325.0)

        # At saturation, all three temperatures should be equal
        assert result.twb == pytest.approx(result.tdb, abs=0.5)
        assert result.tdew == pytest.approx(result.tdb, abs=0.5)

    def test_high_altitude(self) -> None:
        """At lower pressure (high altitude), properties should change."""
        from comfio.domains.iaq_advanced import get_psychrometrics

        # 1500m altitude ≈ 84500 Pa
        result_sea = get_psychrometrics(tdb=20.0, rh=0.50, pressure=101325.0)
        result_alt = get_psychrometrics(tdb=20.0, rh=0.50, pressure=84500.0)

        # At lower pressure, moist air volume should be larger
        assert result_alt.moist_air_volume > result_sea.moist_air_volume
