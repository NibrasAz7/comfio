"""Tests for the comfort_tool GUI tool (classic interactive rebuild).

The physics lives in ``_assets/physics.js`` (a JS port of the
pythermalcomfort model family) and is verified through Node against the
installed Python package.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

st = pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

ASSETS = Path(__file__).parents[2] / "src" / "comfio" / "gui" / "tools" / "comfort_tool" / "_assets"
NODE_CHECK = Path(__file__).with_name("_node_check.js")

needs_node = pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")


def _node(fn: str, *args: float):
    out = subprocess.run(
        ["node", str(NODE_CHECK), fn, json.dumps(list(args))],
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(out.stdout)


def _assembled() -> str:
    from comfio.gui.tools.comfort_tool.page import _assemble_html

    html = _assemble_html()
    assert html is not None
    return html


def test_assets_exist() -> None:
    for name in ("page.html", "site.css", "physics.js", "app.js"):
        assert (ASSETS / name).is_file(), name
    html = _assembled()
    assert "Comfio" in html
    assert "CBE" not in html
    assert "cbe.berkeley.edu" not in html
    assert "Center for the Built Environment" not in html


@needs_node
def test_physics_pmv_matches_pythermalcomfort() -> None:
    from pythermalcomfort.models import pmv_ppd_ashrae
    from pythermalcomfort.utilities import clo_dynamic_ashrae, v_relative

    pts = [
        (25.0, 25.0, 0.1, 50.0, 1.2, 0.5),
        (20.0, 20.0, 0.1, 50.0, 1.0, 1.0),
        (28.0, 28.0, 0.3, 60.0, 1.4, 0.5),
        (23.0, 25.0, 0.15, 30.0, 1.6, 0.7),
        (30.0, 29.0, 0.6, 70.0, 1.1, 0.3),
    ]
    for ta, tr, v, rh, met, clo in pts:
        vr = float(v_relative(v, met))
        clo_d = float(clo_dynamic_ashrae(clo, met))
        ref = float(
            pmv_ppd_ashrae(
                tdb=ta,
                tr=tr,
                vr=vr,
                rh=rh,
                met=met,
                clo=clo_d,
                limit_inputs=False,
                round_output=False,
            ).pmv
        )
        got = _node("pmvAshrae", ta, tr, v, rh, met, clo)
        assert got["pmv"] == pytest.approx(ref, abs=0.02), (ta, tr, v, rh, met, clo)


@needs_node
def test_physics_set_matches_pythermalcomfort() -> None:
    from pythermalcomfort.models import set_tmp

    pts = [
        (25.0, 25.0, 0.1, 50.0, 1.2, 0.5),
        (20.0, 20.0, 0.1, 50.0, 1.0, 1.0),
        (30.0, 30.0, 0.3, 60.0, 1.2, 0.5),
        (23.0, 25.0, 0.1, 30.0, 1.6, 0.7),
        (18.0, 22.0, 0.05, 40.0, 1.4, 1.1),
    ]
    for ta, tr, v, rh, met, clo in pts:
        ref = float(
            set_tmp(
                tdb=ta, tr=tr, v=v, rh=rh, met=met, clo=clo, limit_inputs=False, round_output=False
            ).set
        )
        got = _node("setTmp", ta, tr, v, rh, met, clo)
        assert got == pytest.approx(ref, abs=0.3), (ta, tr, v, rh, met, clo)


@needs_node
def test_physics_adaptive_matches() -> None:
    from pythermalcomfort.models import adaptive_ashrae, adaptive_en

    for ta, tr, t_rm, v in [
        (25.0, 25.0, 20.0, 0.1),
        (28.0, 27.0, 24.0, 0.8),
        (20.0, 21.0, 12.0, 0.3),
    ]:
        ref = adaptive_ashrae(tdb=ta, tr=tr, t_running_mean=t_rm, v=v)
        got = _node("adaptiveAshrae", ta, tr, t_rm, v)
        assert got["tCmf"] == pytest.approx(float(ref.tmp_cmf), abs=0.05)
        assert got["tCmf80Low"] == pytest.approx(float(ref.tmp_cmf_80_low), abs=0.05)
        assert got["tCmf80Up"] == pytest.approx(float(ref.tmp_cmf_80_up), abs=0.05)
        assert got["tCmf90Low"] == pytest.approx(float(ref.tmp_cmf_90_low), abs=0.05)
        assert got["tCmf90Up"] == pytest.approx(float(ref.tmp_cmf_90_up), abs=0.05)
        assert bool(got["acceptability80"]) == bool(ref.acceptability_80)
        assert bool(got["acceptability90"]) == bool(ref.acceptability_90)

        ref_en = adaptive_en(tdb=ta, tr=tr, t_running_mean=t_rm, v=v)
        got_en = _node("adaptiveEn", ta, tr, t_rm, v)
        assert got_en["tCmf"] == pytest.approx(float(ref_en.tmp_cmf), abs=0.05)
        assert got_en["catI"]["low"] == pytest.approx(float(ref_en.tmp_cmf_cat_i_low), abs=0.05)
        assert got_en["catIII"]["up"] == pytest.approx(float(ref_en.tmp_cmf_cat_iii_up), abs=0.05)
        assert bool(got_en["acceptabilityIII"]) == bool(ref_en.acceptability_cat_iii)


@needs_node
def test_physics_corrections_and_ce() -> None:
    from pythermalcomfort.models import cooling_effect
    from pythermalcomfort.utilities import clo_dynamic_ashrae, v_relative

    assert _node("vRelative", 0.1, 1.0) == pytest.approx(0.1)
    assert _node("vRelative", 0.1, 1.4) == pytest.approx(0.22)
    assert _node("cloDynamicAshrae", 0.5, 1.0) == pytest.approx(0.5)
    assert _node("cloDynamicAshrae", 0.5, 1.4) == pytest.approx(0.443)

    assert float(v_relative(0.1, 1.4)) == pytest.approx(_node("vRelative", 0.1, 1.4))
    assert float(clo_dynamic_ashrae(0.5, 1.4)) == pytest.approx(_node("cloDynamicAshrae", 0.5, 1.4))

    ref_ce = float(cooling_effect(tdb=25, tr=25, vr=0.3, rh=50, met=1.2, clo=0.5).ce)
    got_ce = _node("coolingEffect", 25, 25, 0.3, 50, 1.2, 0.5)
    assert got_ce == pytest.approx(ref_ce, abs=0.1)


def _comfort_app() -> None:
    from comfio.gui.tools.comfort_tool.page import render

    render()


@pytest.fixture()
def at() -> AppTest:
    return AppTest.from_function(_comfort_app, default_timeout=60)


def test_render_runs(at: AppTest) -> None:
    at.run()
    assert not at.exception
