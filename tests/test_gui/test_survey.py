"""Tests for the occupant-survey tool (Task 5).

Pure-logic tests hit ``comfio.gui.tools.survey._logic``; the render test
uses ``AppTest.from_function`` with the bundled demo survey (no uploads
or workbench result needed).
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

st = pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

from comfio.gui.tools.survey import _logic  # noqa: E402


def test_summarize_known() -> None:
    df = pd.DataFrame(
        {
            "respondent_id": [f"R{i}" for i in range(4)],
            "sat_thermal": [-3, 0, 1, 2],
        }
    )
    out = _logic.summarize(df, {"sat_thermal": "sat_thermal"})
    assert len(out) == 1
    row = out.iloc[0]
    assert row["category"] == "sat_thermal"
    assert row["group"] == "all"
    assert row["n"] == 4
    assert row["pct_satisfied"] == pytest.approx(50.0)
    assert row["pct_neutral"] == pytest.approx(25.0)
    assert row["pct_dissatisfied"] == pytest.approx(25.0)
    assert row["mean"] == pytest.approx(0.0)


def test_seven_point_remap() -> None:
    df = pd.DataFrame({"sat_air": [7, 1, 4]})
    out = _logic.summarize(df, {"sat_air": "sat_air"}, scale="1..7")
    row = out.iloc[0]
    assert row["pct_satisfied"] == pytest.approx(100.0 / 3.0, abs=0.1)
    assert row["pct_dissatisfied"] == pytest.approx(100.0 / 3.0, abs=0.1)
    assert row["pct_neutral"] == pytest.approx(100.0 / 3.0, abs=0.1)
    # 7 -> +3, 1 -> -3, 4 -> 0
    assert row["mean"] == pytest.approx(0.0)
    # without the opt-in remap the same values are *not* re-interpreted
    out_raw = _logic.summarize(df, {"sat_air": "sat_air"}, scale="-3..3")
    assert out_raw.iloc[0]["pct_satisfied"] == pytest.approx(100.0)


def test_invalid_scale_reported() -> None:
    assert _logic.validate_scale(pd.Series(["very satisfied", "ok"])) == "invalid"
    assert _logic.validate_scale(pd.Series([0, 1, 9])) == "invalid"
    assert _logic.validate_scale(pd.Series([-3, 0, 3])) == "-3..3"
    assert _logic.validate_scale(pd.Series([1, 4, 7])) == "1..7"
    assert _logic.validate_scale(pd.Series([None, None])) == "invalid"


def test_detect_columns() -> None:
    df = pd.DataFrame(
        {
            "respondent": ["a"],
            "Thermal Satisfaction": [1],
            "sat_air": [0],
            "noise satisfaction": [-1],
            "overall": [2],
        }
    )
    found = _logic.detect_columns(df)
    assert found["sat_thermal"] == "Thermal Satisfaction"
    assert found["sat_air"] == "sat_air"
    assert found["sat_acoustic"] == "noise satisfaction"
    assert found["sat_overall"] == "overall"
    assert "sat_lighting" not in found


def test_sensor_vs_occupant_flags_disagreement() -> None:
    result = SimpleNamespace(
        ieq_result=SimpleNamespace(
            domain_scores={
                "thermal": np.array([90.0, 85.0, 88.0]),
                "acoustic": np.array([40.0, 45.0, 42.0]),
            }
        )
    )
    df = pd.DataFrame(
        {
            # occupants unhappy but sensors happy -> disagreement
            "sat_thermal": [-3, -3, -2, -3],
            # occupants happy and sensors unhappy -> disagreement
            "sat_acoustic": [3, 3, 2, 3],
        }
    )
    summary = _logic.summarize(
        df, {"sat_thermal": "sat_thermal", "sat_acoustic": "sat_acoustic"}
    )
    out = _logic.sensor_vs_occupant(summary, result)
    assert out is not None
    by_domain = out.set_index("domain")
    assert bool(by_domain.loc["thermal", "flag"])
    assert bool(by_domain.loc["acoustic", "flag"])
    assert by_domain.loc["thermal", "sensor_mean_score"] == pytest.approx(87.7, abs=0.1)

    # agreement: satisfied occupants + high sensor score -> no flag
    df_ok = pd.DataFrame({"sat_thermal": [3, 3, 2, 3]})
    out_ok = _logic.sensor_vs_occupant(
        _logic.summarize(df_ok, {"sat_thermal": "sat_thermal"}), result
    )
    assert out_ok is not None
    assert not bool(out_ok.iloc[0]["flag"])


def test_sensor_vs_occupant_none_without_result() -> None:
    df = pd.DataFrame({"sat_thermal": [1, 2]})
    summary = _logic.summarize(df, {"sat_thermal": "sat_thermal"})
    assert _logic.sensor_vs_occupant(summary, None) is None
    assert _logic.sensor_vs_occupant(
        summary, SimpleNamespace(ieq_result=None)
    ) is None


def test_demo_deterministic() -> None:
    a = _logic.demo_survey()
    b = _logic.demo_survey()
    pd.testing.assert_frame_equal(a, b)
    assert len(a) == 60
    assert a["zone"].nunique() == 2
    assert set(_logic.SAT_COLUMNS) <= set(a.columns)
    for col in _logic.SAT_COLUMNS:
        assert _logic.validate_scale(a[col]) == "-3..3"
    c = _logic.demo_survey(seed=99)
    assert not a.equals(c)


def test_template_csv_shape() -> None:
    text = _logic.template_csv()
    lines = [ln for ln in text.strip().splitlines() if ln]
    assert len(lines) == 4  # header + 3 example rows
    header = lines[0].split(",")
    assert header[:2] == ["respondent_id", "zone"]
    assert header[2:] == list(_logic.SAT_COLUMNS)


def _survey_app() -> None:
    from comfio.gui.tools.survey.page import render

    render()


@pytest.fixture()
def at() -> AppTest:
    st.cache_data.clear()
    return AppTest.from_function(_survey_app, default_timeout=60)


def test_page_demo_render(at: AppTest) -> None:
    at.run()
    assert not at.exception
    downloads = [d.label for d in at.get("download_button")]
    assert any("template" in d.lower() for d in downloads)
    demo = [b for b in at.button if "demo" in (b.label or "").lower()]
    assert demo
    demo[0].click().run()
    assert not at.exception
    # 8 category selectboxes + summary table + diverging chart
    assert len(at.selectbox) >= 8
    assert len(at.dataframe) >= 1
    assert len(at.get("plotly_chart")) >= 1
