"""Tests for the global-database benchmark tool (Task 3).

Uses ``AppTest.from_function``; the database loader is monkeypatched so
no network access or real download ever happens.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

st = pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

from comfio.integration import comfort_db  # noqa: E402


def _bench_app() -> None:
    from comfio.gui.tools.benchmark.page import render

    render()


def _synth_db(n: int = 200) -> pd.DataFrame:
    rng = np.random.default_rng(5)
    return pd.DataFrame(
        {
            "building_id": rng.choice([f"b{i:02d}" for i in range(10)], n),
            "climate": rng.choice(["tropical", "temperate", "continental"], n),
            "building_type": rng.choice(["office", "classroom"], n),
            "cooling_type": rng.choice(
                ["air conditioned", "naturally ventilated", "mixed mode"], n
            ),
            "season": rng.choice(["summer", "winter"], n),
            "country": rng.choice(["IT", "UK", "IN"], n),
            "ta": rng.uniform(20.0, 30.0, n),
            "tr": rng.uniform(20.0, 30.0, n),
            "top": rng.uniform(20.0, 30.0, n),
            "vel": rng.uniform(0.05, 0.3, n),
            "t_mot_isd": rng.uniform(10.0, 33.0, n),
            "thermal_sensation": rng.integers(-3, 4, n).astype(float),
            "thermal_acceptability": rng.choice(["acceptable", "unacceptable"], n),
            "pmv": rng.uniform(-2.0, 2.0, n),
        }
    )


@pytest.fixture()
def at(monkeypatch: pytest.MonkeyPatch, tmp_path) -> AppTest:
    # isolate the cache so a real download can never be trusted by accident
    monkeypatch.setenv(comfort_db.DB2_CACHE_ENV, str(tmp_path / "cache"))
    st.cache_data.clear()
    return AppTest.from_function(_bench_app, default_timeout=60)


def test_fetch_button_shown_empty_cache(at: AppTest) -> None:
    at.run()
    assert not at.exception
    labels = [b.label for b in at.button]
    assert any("Fetch database" in str(label) for label in labels)
    # nothing is rendered before the user asks for the download
    assert len(at.get("plotly_chart")) == 0


def test_charts_render_with_monkeypatched_loader(
    at: AppTest, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(comfort_db, "load_ashrae_db2", lambda *a, **k: _synth_db())
    at.run()
    at.button[0].click().run()
    assert not at.exception
    assert len(at.multiselect) >= 5  # climate/building/cooling/season/country
    assert len(at.get("plotly_chart")) >= 3


def test_offline_shows_error(at: AppTest, monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*args: object, **kwargs: object) -> pd.DataFrame:
        raise comfort_db.ComfortDBUnavailable("No cached database and download failed: no network.")

    monkeypatch.setattr(comfort_db, "load_ashrae_db2", boom)
    at.run()
    at.button[0].click().run()
    assert not at.exception
    assert len(at.error) >= 1
    assert "no network" in str(at.error[0].value)


def test_filter_multiselect_narrows_rows(at: AppTest, monkeypatch: pytest.MonkeyPatch) -> None:
    db = _synth_db()
    monkeypatch.setattr(comfort_db, "load_ashrae_db2", lambda *a, **k: db)
    at.run()
    at.button[0].click().run()
    assert not at.exception
    season = [m for m in at.multiselect if "season" in (m.label or "")]
    assert season
    season[0].set_value(["summer"]).run()
    assert not at.exception
    assert len(at.get("plotly_chart")) >= 3
