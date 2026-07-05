"""Tests for advanced visual comfort module (daylighting + color quality).

Tests are skipped if pyradiance or colour-science are not installed.
"""

import numpy as np
import pytest

pytest.importorskip("colour", reason="colour-science not installed — pip install comfio[color]")


class TestEvaluateColorQuality:
    """Tests for evaluate_color_quality using colour-science."""

    def test_cri_of_d65_illuminant(self) -> None:
        """D65 illuminant should have a very high CRI (near 100)."""
        from comfio.domains.visual_advanced import evaluate_color_quality

        # Use colour-science's built-in D65 illuminant
        import colour

        sd = colour.SDS_ILLUMINANTS["D65"]
        wavelengths = sd.wavelengths
        values = sd.values

        result = evaluate_color_quality(
            spectral_distribution=values,
            wavelengths=wavelengths,
            method="CIE 1995",
        )

        assert result.cri > 90.0, f"D65 CRI should be very high, got {result.cri}"
        assert 5000 < result.cct < 7000, f"D65 CCT should be ~6500K, got {result.cct}"
        assert result.compliant is True
        assert result.score > 80.0

    def test_cri_with_additional_data(self) -> None:
        """CRI with additional_data=True should return per-sample data."""
        from comfio.domains.visual_advanced import evaluate_color_quality

        import colour

        sd = colour.SDS_ILLUMINANTS["D65"]
        result = evaluate_color_quality(
            spectral_distribution=sd.values,
            wavelengths=sd.wavelengths,
            additional_data=True,
        )

        assert result.cri_data is not None
        assert "Q_as" in result.cri_data
        assert len(result.cri_data["Q_as"]) > 0

    def test_cct_range_reasonable(self) -> None:
        """CCT should be in a physically reasonable range."""
        from comfio.domains.visual_advanced import evaluate_color_quality

        import colour

        # Use FL2 fluorescent lamp (lower CRI, different CCT)
        sd = colour.SDS_ILLUMINANTS["FL2"]
        result = evaluate_color_quality(
            spectral_distribution=sd.values,
            wavelengths=sd.wavelengths,
        )

        assert 2000 < result.cct < 10000, f"CCT {result.cct} outside reasonable range"
        assert 0 <= result.cri <= 100
        assert 0 <= result.score <= 100

    def test_score_range(self) -> None:
        """Score should always be in 0-100 range."""
        from comfio.domains.visual_advanced import evaluate_color_quality

        import colour

        for lamp_name in ["A", "D65", "FL2"]:
            sd = colour.SDS_ILLUMINANTS[lamp_name]
            result = evaluate_color_quality(
                spectral_distribution=sd.values,
                wavelengths=sd.wavelengths,
            )
            assert 0 <= result.score <= 100, f"{lamp_name} score {result.score} out of range"

    def test_wavelength_mismatch_raises(self) -> None:
        """Should raise ValueError if SPD and wavelength lengths don't match."""
        from comfio.domains.visual_advanced import evaluate_color_quality

        with pytest.raises(ValueError, match="SPD length"):
            evaluate_color_quality(
                spectral_distribution=np.array([1.0, 2.0, 3.0]),
                wavelengths=np.array([380.0, 500.0, 600.0, 700.0]),
            )


class TestEvaluateDaylighting:
    """Tests for evaluate_daylighting using pyradiance.

    These are skipped if pyradiance is not installed, since it requires
    Radiance binaries (WSL recommended on Windows).
    """

    def test_import_error_without_pyradiance(self) -> None:
        """Should raise ImportError with helpful message if pyradiance missing."""
        try:
            import pyradiance  # noqa: F401
            pytest.skip("pyradiance is installed, skipping import error test")
        except ImportError:
            pass

        from comfio.domains.visual_advanced import evaluate_daylighting

        with pytest.raises(ImportError, match="pyradiance"):
            evaluate_daylighting(
                octree_file="nonexistent.oct",
                sensor_points=np.array([[0, 0, 0]]),
            )

    def test_invalid_sensor_points_shape(self) -> None:
        """Should raise ValueError for wrong sensor_points shape."""
        try:
            import pyradiance  # noqa: F401
        except ImportError:
            pytest.skip("pyradiance not installed")

        from comfio.domains.visual_advanced import evaluate_daylighting

        with pytest.raises(ValueError, match="sensor_points"):
            evaluate_daylighting(
                octree_file="test.oct",
                sensor_points=np.array([1, 2, 3]),  # 1D, not 2D
            )
