"""Tests for the setpoint explorer tool (Task 6).

Model tests exercise ``comfort_band``/``deadband_table`` directly; the
page test uses ``AppTest.from_function``.
"""

from __future__ import annotations

import numpy as np
import pytest

st = pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

from comfio.gui.tools.setpoint._logic import (  # noqa: E402
    ENERGY_CALC_URL,
    comfort_band,
    deadband_table,
)


def _setpoint_app() -> None:
    from comfio.gui.tools.setpoint.page import render

    render()


@pytest.fixture()
def at() -> AppTest:
    return AppTest.from_function(_setpoint_app, default_timeout=60)


def test_band_ordering() -> None:
    band = comfort_band(1.2, 1.0, 0.5, 50.0)
    assert set(band) == {"heat_min_c", "cool_max_c", "cool_max_fan_c"}
    assert band["heat_min_c"] < band["cool_max_c"]
    # no fan requested -> the fan edge is NaN
    assert np.isnan(band["cool_max_fan_c"])


def test_fan_raises_cool_max() -> None:
    band = comfort_band(1.2, 1.0, 0.5, 50.0, vr_fan=0.8)
    assert band["cool_max_fan_c"] > band["cool_max_c"]


def test_more_clo_lowers_heat_min() -> None:
    light = comfort_band(1.2, 0.8, 0.5, 50.0)
    heavy = comfort_band(1.2, 1.2, 0.5, 50.0)
    assert heavy["heat_min_c"] < light["heat_min_c"]


def test_no_root_nan() -> None:
    # met 4.0 + heavy clothing + saturated air: PMV stays above +0.5 over
    # the whole 10–36 °C axis even at fan speed — no crossing exists
    band = comfort_band(4.0, 2.0, 2.0, 100.0, vr_fan=0.8)
    for key in ("heat_min_c", "cool_max_c", "cool_max_fan_c"):
        assert np.isnan(band[key]), key


def test_deadband_table_rows() -> None:
    band = comfort_band(1.2, 1.0, 0.5, 50.0, vr_fan=0.8)
    table = deadband_table(21.0, 25.0, band)
    assert list(table.columns) == [
        "scenario",
        "heating setpoint",
        "cooling setpoint",
        "deadband width",
        "note",
    ]
    assert list(table["scenario"]) == ["current", "proposed", "delta"]
    current, proposed, delta = table.iloc[0], table.iloc[1], table.iloc[2]
    assert current["deadband width"] == pytest.approx(25.0 - 21.0)
    assert proposed["cooling setpoint"] == pytest.approx(
        band["cool_max_fan_c"]
    )
    assert proposed["heating setpoint"] == pytest.approx(band["heat_min_c"])
    assert delta["deadband width"] == pytest.approx(
        proposed["deadband width"] - current["deadband width"]
    )
    # the fan edge is used for the proposed cooling setpoint when present
    assert proposed["cooling setpoint"] > band["cool_max_c"]


def test_page_shows_energy_link(at: AppTest) -> None:
    at.run()
    assert not at.exception
    urls = [b.url for b in at.get("link_button")]
    md = " ".join(str(e.value) for e in list(at.markdown) + list(at.caption))
    assert ENERGY_CALC_URL in urls or ENERGY_CALC_URL in md
