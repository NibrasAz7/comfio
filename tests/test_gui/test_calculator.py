"""Tests for the comfort calculator tool (Task 2).

Model tests run against pythermalcomfort/comfio directly; page tests use
``AppTest.from_function`` since the tool renders inside the multipage shell.
"""

from __future__ import annotations

import numpy as np
import pytest

st = pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

from comfio.domains.thermal import evaluate_thermal  # noqa: E402
from comfio.gui.tools.calculator._models import (  # noqa: E402
    METHODS,
    Inputs,
    en16798_category,
    evaluate_point,
    mrt_from_globe,
    rh_from,
    sweep,
)
from comfio.gui.tools.calculator._psychro import (  # noqa: E402
    humidity_ratio,
    pmv_zone_boundary,
)


def _calc_app() -> None:
    from comfio.gui.tools.calculator.page import render

    render()


@pytest.fixture()
def at() -> AppTest:
    return AppTest.from_function(_calc_app, default_timeout=60)


def test_ashrae_matches_pythermalcomfort() -> None:
    from pythermalcomfort.models import pmv_ppd_ashrae
    from pythermalcomfort.utilities import clo_dynamic_ashrae, v_relative

    inp = Inputs(tdb=27.0, tr=27.0, rh=50.0, vr=0.8, met=1.2, clo=0.5)
    res = evaluate_point(inp, METHODS[0])

    direct = pmv_ppd_ashrae(
        tdb=[27.0],
        tr=[27.0],
        vr=v_relative([0.8], [1.2]),
        rh=[50.0],
        met=[1.2],
        clo=clo_dynamic_ashrae([0.5], [1.2], model="55-2023"),
        model="55-2023",
        limit_inputs=True,
        round_output=False,
    )
    assert res.pmv == pytest.approx(float(direct.pmv[0]), abs=1e-6)
    assert res.ppd == pytest.approx(float(direct.ppd[0]), abs=1e-6)


def test_iso_matches_pipeline() -> None:
    inp = Inputs(tdb=24.0, tr=24.0, rh=50.0, vr=0.1, met=1.2, clo=0.5)
    res = evaluate_point(inp, METHODS[1])
    direct = evaluate_thermal(
        tdb=np.array([24.0]),
        tr=np.array([24.0]),
        vr=np.array([0.1]),
        rh=np.array([50.0]),
        met=1.2,
        clo=0.5,
    )
    assert res.pmv == pytest.approx(float(direct.pmv[0]), abs=1e-9)
    assert res.ppd == pytest.approx(float(direct.ppd[0]), abs=1e-9)


def test_en_category() -> None:
    assert en16798_category(0.3) == "II"
    assert en16798_category(0.6) == "III"
    assert en16798_category(0.8) == "beyond III"
    assert en16798_category(0.1) == "I"


def test_out_of_range_note() -> None:
    inp = Inputs(tdb=45.0, tr=45.0, rh=50.0, vr=0.1, met=1.2, clo=0.5)
    res = evaluate_point(inp, METHODS[1])
    assert np.isnan(res.pmv)
    assert res.compliant is None
    assert any("outside model range" in n for n in res.notes)


def test_adaptive_requires_running_mean() -> None:
    inp = Inputs(tdb=24.0, tr=24.0, rh=50.0, vr=0.1, met=1.2, clo=0.5)
    res = evaluate_point(inp, METHODS[3])
    assert np.isnan(res.pmv)
    assert any("needs" in n for n in res.notes)


def test_adaptive_compliant_point() -> None:
    # t_prevail 20 -> t_comf = 24.0; a 24/24 room passes the 80% band.
    inp = Inputs(tdb=24.0, tr=24.0, rh=50.0, vr=0.1, met=1.2, clo=0.5,
                 t_running_mean=20.0)
    res = evaluate_point(inp, METHODS[3])
    assert res.compliant is True
    assert res.category == "90 %"
    assert res.extra["t_comf"] == pytest.approx(24.0, abs=0.01)


def test_rh_from_dew_point_roundtrip() -> None:
    from pythermalcomfort.utilities import dew_point_tmp

    dew = float(dew_point_tmp(np.array([25.0]), np.array([50.0]))[0])
    assert rh_from("dew_point", dew, 25.0) == pytest.approx(50.0, abs=0.5)


def test_rh_from_dew_point_no_root() -> None:
    assert np.isnan(rh_from("dew_point", 40.0, 25.0))


def test_humidity_ratio_roundtrip() -> None:
    w = float(humidity_ratio(np.array([25.0]), np.array([50.0]))[0])
    assert rh_from("humidity_ratio", w, 25.0) == pytest.approx(50.0, abs=1e-6)


def test_mrt_from_globe_matches_pythermalcomfort() -> None:
    from pythermalcomfort.utilities import mean_radiant_tmp

    expected = float(mean_radiant_tmp(30.0, 25.0, 0.5, standard="ISO"))
    assert mrt_from_globe(30.0, 25.0, 0.5) == pytest.approx(expected)


def test_zone_band_brackets_neutral() -> None:
    zone = pmv_zone_boundary(1.2, 0.5, 0.1)
    row = zone.iloc[int(np.argmin(np.abs(zone["w"] - 0.008)))]
    assert row["t_cool"] < row["t_warm"]
    mid = 0.5 * (row["t_cool"] + row["t_warm"])
    # rh implied by w at the midpoint temperature
    from comfio.gui.tools.calculator._models import ashrae_pmv

    pv = row["w"] * 101325.0 / (0.621945 + row["w"])
    from pythermalcomfort.utilities import p_sat

    rh_mid = 100.0 * pv / float(p_sat(np.array([mid]))[0])
    pmv_mid = float(
        ashrae_pmv(
            np.array([mid]), np.array([mid]), np.array([0.1]),
            np.array([rh_mid]), np.array([1.2]), np.array([0.5]),
        )[0]
    )
    assert abs(pmv_mid) <= 0.5


def test_air_speed_widens_zone() -> None:
    z_slow = pmv_zone_boundary(1.2, 0.5, 0.1)
    z_fast = pmv_zone_boundary(1.2, 0.5, 0.8)
    i = int(np.argmin(np.abs(z_slow["w"] - 0.008)))
    assert z_fast.iloc[i]["t_warm"] > z_slow.iloc[i]["t_warm"]


def test_zone_no_root_nan() -> None:
    zone = pmv_zone_boundary(1.2, 0.5, 0.1, w_max=0.03)
    assert np.isnan(zone.iloc[-1]["t_cool"])


def test_sweep_columns() -> None:
    inp = Inputs(tdb=24.0, tr=24.0, rh=50.0, vr=0.1, met=1.2, clo=0.5)
    df = sweep(inp, METHODS[0], "tdb", np.linspace(20.0, 28.0, 5))
    assert list(df.columns) == ["tdb", "pmv", "ppd", "set_c"]
    assert len(df) == 5
    assert df["pmv"].is_monotonic_increasing


def test_default_render(at: AppTest) -> None:
    at.run()
    assert not at.exception
    labels = [m.label for m in at.metric]
    assert "PMV" in labels and "PPD" in labels


def test_compare_tab_renders_chart(at: AppTest) -> None:
    at.run()
    assert not at.exception
    # main + compare charts: result psychro chart + one compare chart
    assert len(at.get("arrow_vega_lite_chart")) >= 2 or len(
        at.get("plotly_chart")
    ) >= 2


def test_ranges_tab_renders_chart(at: AppTest) -> None:
    at.run()
    assert not at.exception
    n_charts = len(at.get("arrow_vega_lite_chart")) + len(
        at.get("plotly_chart")
    )
    assert n_charts >= 3  # result + compare + ranges


def test_prefill_from_calc_inputs(at: AppTest) -> None:
    at.session_state["calc_inputs"] = {
        "tdb": 30.0,
        "tr": 30.0,
        "rh": 50.0,
        "vr": 0.1,
        "met": 1.2,
        "clo": 0.5,
    }
    at.run()
    assert not at.exception
    tdb = [n for n in at.number_input if "Air temperature" in (n.label or "")]
    assert tdb and tdb[0].value == pytest.approx(30.0)
