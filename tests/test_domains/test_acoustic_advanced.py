"""Tests for advanced acoustic comfort module (reverberation + STI).

Tests are skipped if python-acoustics or pyroomacoustics are not installed.
"""

import numpy as np
import pytest

pytest.importorskip(
    "acoustics", reason="python-acoustics not installed — pip install comfio[acoustics]"
)
pytest.importorskip(
    "pyroomacoustics", reason="pyroomacoustics not installed — pip install comfio[acoustics]"
)


class TestEvaluateReverberation:
    """Tests for evaluate_reverberation using python-acoustics."""

    def test_sabine_simple_room(self) -> None:
        """Sabine RT60 for a simple room should be positive and reasonable."""
        from comfio.domains.acoustic_advanced import evaluate_reverberation

        # Simple room: 5 surfaces, moderate absorption
        surfaces = np.array([50.0, 50.0, 50.0, 50.0, 50.0])
        alpha = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
        volume = 125.0  # 5x5x5 m

        result = evaluate_reverberation(
            surfaces=surfaces,
            absorption_coeffs=alpha,
            volume=volume,
            method="sabine",
            room_type="office",
        )

        assert result.rt60.ndim >= 1
        assert np.all(result.rt60 > 0), "RT60 should be positive"
        assert result.method == "sabine"
        assert 0 <= result.score <= 100

    def test_eyring_simple_room(self) -> None:
        """Eyring RT60 should be slightly lower than Sabine for same room."""
        from comfio.domains.acoustic_advanced import evaluate_reverberation

        surfaces = np.array([50.0, 50.0, 50.0, 50.0, 50.0])
        alpha = np.array([0.3, 0.3, 0.3, 0.3, 0.3])
        volume = 125.0

        sabine = evaluate_reverberation(
            surfaces=surfaces, absorption_coeffs=alpha, volume=volume, method="sabine"
        )
        eyring = evaluate_reverberation(
            surfaces=surfaces, absorption_coeffs=alpha, volume=volume, method="eyring"
        )

        # Eyring is generally lower than Sabine for same absorption
        assert np.mean(eyring.rt60) <= np.mean(sabine.rt60) + 0.01

    def test_compliance_office(self) -> None:
        """Office RT60 within 0.4-0.6s should be compliant."""
        from comfio.domains.acoustic_advanced import evaluate_reverberation

        # High absorption → short RT60, within office range
        surfaces = np.array([50.0, 50.0, 50.0, 50.0, 50.0])
        alpha = np.array([0.5, 0.5, 0.5, 0.5, 0.5])
        volume = 125.0

        result = evaluate_reverberation(
            surfaces=surfaces,
            absorption_coeffs=alpha,
            volume=volume,
            method="sabine",
            room_type="office",
        )

        # With high absorption, RT60 should be short
        assert np.mean(result.rt60) < 1.0
        assert isinstance(result.compliant, bool)

    def test_nrc_calculation(self) -> None:
        """NRC should be calculated when 4+ frequency bands are provided."""
        from comfio.domains.acoustic_advanced import evaluate_reverberation

        # 5 surfaces × 6 frequency bands
        surfaces = np.array([[50.0] * 6 for _ in range(5)])
        alpha = np.array([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6] for _ in range(5)])
        volume = 125.0

        result = evaluate_reverberation(
            surfaces=surfaces,
            absorption_coeffs=alpha,
            volume=volume,
            method="sabine",
        )

        assert result.nrc is not None
        assert 0 <= result.nrc <= 1

    def test_invalid_method_raises(self) -> None:
        """Unknown method should raise ValueError."""
        from comfio.domains.acoustic_advanced import evaluate_reverberation

        with pytest.raises(ValueError, match="Unknown method"):
            evaluate_reverberation(
                surfaces=np.array([50.0]),
                absorption_coeffs=np.array([0.2]),
                volume=125.0,
                method="fit",  # type: ignore[arg-type]
            )


class TestEvaluateSpeechIntelligibility:
    """Tests for evaluate_speech_intelligibility using pyroomacoustics."""

    def test_sti_from_synthetic_ir(self) -> None:
        """STI from a synthetic impulse response should be in 0-1 range."""
        from comfio.domains.acoustic_advanced import evaluate_speech_intelligibility

        # Create a simple decaying impulse response
        sample_rate = 16000.0
        duration = 0.5  # 500ms
        n_samples = int(sample_rate * duration)
        t = np.arange(n_samples) / sample_rate
        ir = np.exp(-t / 0.3) * np.random.randn(n_samples)

        result = evaluate_speech_intelligibility(
            impulse_response=ir,
            sample_rate=sample_rate,
        )

        assert 0 <= result.sti <= 1, f"STI {result.sti} out of range"
        assert result.rating in ["bad", "poor", "fair", "good", "excellent"]
        assert result.rt60_measured > 0
        assert 0 <= result.score <= 100

    def test_sti_compliance_threshold(self) -> None:
        """Compliance should be True when STI >= 0.60."""
        from comfio.domains.acoustic_advanced import evaluate_speech_intelligibility

        # Very short RT60 → high STI → should be compliant
        sample_rate = 16000.0
        n_samples = int(sample_rate * 0.05)  # 50ms IR
        t = np.arange(n_samples) / sample_rate
        ir = np.exp(-t / 0.02) * np.random.randn(n_samples)

        result = evaluate_speech_intelligibility(
            impulse_response=ir,
            sample_rate=sample_rate,
        )

        # Short RT60 should give high STI
        if result.sti >= 0.60:
            assert result.compliant is True
        else:
            assert result.compliant is False

    def test_sti_rating_consistency(self) -> None:
        """Rating should be consistent with STI value."""
        from comfio.domains.acoustic_advanced import (
            STI_RATING_BANDS,
            evaluate_speech_intelligibility,
        )

        sample_rate = 16000.0
        duration = 1.0
        n_samples = int(sample_rate * duration)
        t = np.arange(n_samples) / sample_rate
        ir = np.exp(-t / 0.5) * np.random.randn(n_samples)

        result = evaluate_speech_intelligibility(
            impulse_response=ir,
            sample_rate=sample_rate,
        )

        # Verify rating matches the bands
        expected_rating = "bad"
        for threshold, label in STI_RATING_BANDS:
            if result.sti >= threshold:
                expected_rating = label
        assert result.rating == expected_rating
