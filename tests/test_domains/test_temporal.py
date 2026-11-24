"""Tests for long-term (temporal) IEQ evaluation — EN 16798-1 Annex B."""

import numpy as np
import pandas as pd
import pytest

from comfio.domains.temporal import (
    acoustic_category,
    adaptive_category_en,
    evaluate_temporal_thermal,
    iaq_category_en,
    summarize_time_in_categories,
    thermal_category,
    visual_category,
)


class TestClassifiers:
    def test_thermal_bands(self) -> None:
        pmv = np.array([0.0, 0.2, -0.2, 0.21, 0.5, -0.5, 0.51, 0.7, 0.71, -1.5, np.nan])
        labels = thermal_category(pmv)
        expected = [1, 1, 1, 2, 2, 2, 3, 3, 4, 4]
        np.testing.assert_array_equal(labels[:-1], expected)
        assert np.isnan(labels[-1])

    def test_iaq_is_delta_above_outdoor(self) -> None:
        # 420 + 400 = 820 ppm is the Cat I/II boundary, not 1000 ppm absolute
        co2 = np.array([420.0, 820.0, 821.0, 1020.0, 1021.0, 1420.0, 1421.0])
        np.testing.assert_array_equal(iaq_category_en(co2), [1, 1, 2, 2, 3, 3, 4])

    def test_iaq_custom_outdoor(self) -> None:
        np.testing.assert_array_equal(iaq_category_en([900.0], outdoor=500.0), [1])

    def test_adaptive_bands(self) -> None:
        delta = np.array([0.0, 2.0, -2.0, 2.1, 3.0, 3.1, 4.0, 4.1])
        np.testing.assert_array_equal(adaptive_category_en(delta), [1, 1, 1, 2, 2, 3, 3, 4])

    def test_visual_bands(self) -> None:
        e = np.array([600.0, 500.0, 499.0, 300.0, 299.0, 200.0, 199.0])
        np.testing.assert_array_equal(visual_category(e), [1, 1, 2, 2, 3, 3, 4])

    def test_acoustic_bands(self) -> None:
        la = np.array([30.0, 35.0, 35.1, 40.0, 40.1, 45.0, 45.1])
        np.testing.assert_array_equal(acoustic_category(la), [1, 1, 2, 2, 3, 3, 4])

    def test_invalid_labels_rejected(self) -> None:
        with pytest.raises(ValueError, match="1-4"):
            summarize_time_in_categories(np.array([0.0, 5.0]))


class TestWholeSeriesSummary:
    def test_en_annex_example(self) -> None:
        """Reproduce the EN 16798-1 example: 5 / 7 / 68 / 20 %."""
        labels = np.concatenate(
            [
                np.full(5, 4),
                np.full(7, 3),
                np.full(68, 2),
                np.full(20, 1),
            ]
        ).astype(float)
        res = summarize_time_in_categories(labels)
        assert res.percentages == {"I": 20.0, "II": 68.0, "III": 7.0, "IV": 5.0}
        assert res.cumulative["I"] == 20.0
        assert res.cumulative["I+II"] == 88.0
        assert res.cumulative["I+II+III"] == 95.0
        assert res.cumulative["I+II+III+IV"] == 100.0
        assert res.n_samples == 100
        assert res.excluded_pct == 0.0
        assert res.hours is None

    def test_nan_excluded_from_denominator(self) -> None:
        labels = np.array([1.0, 1.0, 2.0, np.nan, np.nan])
        res = summarize_time_in_categories(labels)
        assert res.n_samples == 3
        assert res.percentages["I"] == pytest.approx(200.0 / 3)
        assert res.percentages["II"] == pytest.approx(100.0 / 3)
        assert res.excluded_pct == pytest.approx(40.0)

    def test_occupancy_mask_dropped_not_excluded(self) -> None:
        labels = np.array([1.0, 2.0, 3.0, 4.0])
        occupied = np.array([True, True, False, False])
        res = summarize_time_in_categories(labels, occupied=occupied)
        assert res.n_samples == 2
        assert res.percentages["I"] == 50.0
        assert res.percentages["II"] == 50.0
        assert res.excluded_pct == 0.0

    def test_empty_after_filter(self) -> None:
        res = summarize_time_in_categories(np.array([np.nan, np.nan]))
        assert res.n_samples == 0
        assert np.isnan(res.percentages["I"])

    def test_hours_from_timestamps(self) -> None:
        ts = pd.date_range("2024-01-01", periods=4, freq="h")
        labels = np.array([1.0, 2.0, 2.0, 4.0])
        res = summarize_time_in_categories(labels, timestamps=ts)
        assert res.total_hours == pytest.approx(4.0)
        assert res.hours == {"I": 1.0, "II": 2.0, "III": 0.0, "IV": 1.0}
        assert res.percentages["II"] == 50.0

    def test_irregular_sampling_time_weighted(self) -> None:
        # 3 samples: 1h Cat I, 3h Cat II, 3h (last, Cat IV)
        ts = pd.DatetimeIndex(["2024-01-01 00:00", "2024-01-01 01:00", "2024-01-01 04:00"])
        labels = np.array([1.0, 2.0, 4.0])
        res = summarize_time_in_categories(labels, timestamps=ts)
        assert res.total_hours == pytest.approx(7.0)
        assert res.percentages["I"] == pytest.approx(100.0 / 7)
        assert res.percentages["II"] == pytest.approx(300.0 / 7)
        assert res.percentages["IV"] == pytest.approx(300.0 / 7)


class TestGroupedSummary:
    def test_daily_grouping(self) -> None:
        ts = pd.date_range("2024-01-01", periods=48, freq="h")
        labels = np.concatenate([np.full(24, 1.0), np.full(24, 4.0)])
        df = summarize_time_in_categories(labels, timestamps=ts, by="D")
        assert len(df) == 2
        assert df.iloc[0]["I"] == 100.0
        assert df.iloc[1]["IV"] == 100.0
        assert df.iloc[0]["total_hours"] == pytest.approx(24.0)

    def test_by_requires_timestamps(self) -> None:
        with pytest.raises(ValueError, match="requires timestamps"):
            summarize_time_in_categories(np.array([1.0]), by="D")


class TestConvenienceWrappers:
    def test_from_thermal_result(self) -> None:
        from comfio import evaluate_thermal

        result = evaluate_thermal(
            tdb=np.array([24.0, 30.0]),
            tr=np.array([24.0, 30.0]),
            vr=np.array([0.1, 0.1]),
            rh=np.array([50.0, 50.0]),
            met=1.2,
            clo=0.5,
        )
        res = evaluate_temporal_thermal(result)
        assert res.n_samples == 2
        assert abs(sum(res.percentages.values()) - 100.0) < 1e-9

    def test_from_raw_pmv(self) -> None:
        res = evaluate_temporal_thermal(np.array([0.0, 1.5]))
        assert res.percentages["I"] == 50.0
        assert res.percentages["IV"] == 50.0

    def test_to_dict_and_json(self) -> None:
        res = evaluate_temporal_thermal(np.array([0.0, 1.5]))
        d = res.to_dict()
        assert d["cumulative"]["I+II"] == 50.0
        assert '"I+II+III+IV": 100.0' in res.to_json()


class TestPipelineIntegration:
    def test_pipeline_adds_temporal_results(self) -> None:
        import comfio

        df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=24, freq="h"),
                "air_temp_c": np.linspace(22, 26, 24),
                "radiant_temp_c": np.linspace(22, 26, 24),
                "air_velocity_ms": 0.1,
                "relative_humidity_pct": 50.0,
                "co2_ppm": np.linspace(600, 1400, 24),
            }
        )
        sensor = comfio.SensorData(df=df, timestamp_col="timestamp")
        result = comfio.run_pipeline(sensor)
        assert set(result.temporal_results) == {"thermal", "iaq"}
        assert result.temporal_results["iaq"].total_hours == pytest.approx(24.0)
        md = comfio.temporal_to_markdown(result.temporal_results)
        assert "EN 16798-1" in md and "THERMAL" in md

    def test_pipeline_temporal_opt_out(self) -> None:
        import comfio

        df = pd.DataFrame(
            {
                "air_temp_c": [24.0],
                "radiant_temp_c": [24.0],
                "air_velocity_ms": [0.1],
                "relative_humidity_pct": [50.0],
            }
        )
        sensor = comfio.SensorData(df=df)
        result = comfio.run_pipeline(sensor, config={"temporal": False})
        assert result.temporal_results == {}


class TestTemporalChart:
    def test_stacked_bars(self) -> None:
        pytest.importorskip("plotly")
        from comfio.gui._charts import temporal_bars

        res = evaluate_temporal_thermal(np.array([0.0, 1.5]))
        fig = temporal_bars({"thermal": res})
        assert len(fig.data) == 4
        assert fig.layout.barmode == "stack"


def test_evaluate_temporal_tsv() -> None:
    from comfio import evaluate_temporal_tsv, evaluate_tsv

    res = evaluate_tsv(np.array([-0.1, 0.6, 1.5]))
    summary = evaluate_temporal_tsv(res)
    assert summary.percentages["I"] == pytest.approx(100 / 3)
    assert summary.percentages["III"] == pytest.approx(100 / 3)
    assert summary.percentages["IV"] == pytest.approx(100 / 3)
    # raw array path works too
    assert evaluate_temporal_tsv(np.array([0.0])).percentages["I"] == 100.0


def test_evaluate_temporal_pollutant() -> None:
    from comfio import evaluate_iaq_pollutants, evaluate_temporal_pollutant

    # pm25 tiers: excellent 5, good 15, moderate 35 -> I/II/III, above -> IV
    pm25 = np.array([3.0, 10.0, 25.0, 60.0])
    summary = evaluate_temporal_pollutant(pm25, pollutant="pm25")
    assert summary.percentages["I"] == 25.0
    assert summary.percentages["II"] == 25.0
    assert summary.percentages["III"] == 25.0
    assert summary.percentages["IV"] == 25.0

    # works on a PollutantIAQResult too (reads the named attribute)
    res = evaluate_iaq_pollutants(pm25=pm25, pm10=np.array([20.0] * 4))
    s2 = evaluate_temporal_pollutant(res, pollutant="pm25")
    assert s2.percentages["IV"] == 25.0
    s3 = evaluate_temporal_pollutant(res, pollutant="pm10")
    assert s3.percentages["II"] == 100.0  # 20 > excellent(15), <= good(45)
