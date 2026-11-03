"""Streamlit AppTest smoke tests for the comfio workbench.

Covers the three-act flow: ingest (demo data) → column mapping →
verdict + evidence. Requires ``comfio[gui]`` extras.
"""

from __future__ import annotations

import pytest

st = pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

APP = "src/comfio/gui/app.py"

_ST_MM = tuple(int(p) for p in st.__version__.split(".")[:2])
file_uploader_supported = pytest.mark.skipif(
    _ST_MM < (1, 56),
    reason="AppTest file_uploader needs streamlit>=1.56",
)


@pytest.fixture()
def at() -> AppTest:
    return AppTest.from_file(APP, default_timeout=60)


def _load_demo(at: AppTest) -> None:
    """Click 'Load demo data' and settle into the mapping screen."""
    at.run()
    for b in at.button:
        if "demo data" in (b.label or ""):
            b.click().run()
            break
    at.run()


@file_uploader_supported
def test_ingest_screen_shows_uploader(at: AppTest) -> None:
    at.run()
    assert not at.exception
    assert any("demo data" in (b.label or "") for b in at.button)
    assert len(at.file_uploader) == 1


def test_mapping_screen_after_demo(at: AppTest) -> None:
    _load_demo(at)
    assert not at.exception
    assert any("Column mapping" in s.value for s in at.subheader)
    # capability banner must list all four domains for demo data
    banner = " ".join(i.value for i in at.info)
    for domain in ("Thermal", "Visual", "Acoustic"):
        assert domain in banner
    # timestamp + canonical selectboxes are rendered
    assert len(at.selectbox) > 10


def test_full_flow_to_verdict(at: AppTest) -> None:
    _load_demo(at)
    for b in at.button:
        if "Run analysis" in (b.label or ''):
            b.click().run()
            break
    at.run()
    assert not at.exception
    subs = [s.value for s in at.subheader]
    assert "2 · Verdict" in subs
    assert "3 · Evidence" in subs
    labels = [m.label for m in at.metric]
    assert "Compliance rate" in labels
    assert "Mean PMV" in labels
    downloads = [d.label for d in at.get("download_button")]
    assert any("Per-row thermal" in d for d in downloads)


def test_tsv_calibration_view_in_demo(at: AppTest) -> None:
    _reach_verdict(at)
    labels = [e.label for e in at.expander]
    assert any("Occupant-vote calibration" in lb for lb in labels)
    metric_labels = [m.label for m in at.metric]
    assert "Slope (α)" in metric_labels and "R²" in metric_labels


def test_certification_preset_reset_air(at: AppTest) -> None:
    _reach_verdict(at)
    sels = [s for s in at.selectbox if "certification" in (s.label or "").lower()]
    assert sels
    sels[0].select("RESET Air (basic targets)").run()
    assert not at.exception
    captions = " ".join(c.value for c in at.caption)
    assert "TVOC" in captions  # demo data lacks TVOC -> reported missing


def test_certification_preset_well(at: AppTest) -> None:
    _reach_verdict(at)
    sels = [s for s in at.selectbox if "certification" in (s.label or "").lower()]
    sels[0].select("WELL v2 T06 (thermal monitoring)").run()
    assert not at.exception
    assert any("PMV" in str(c.value) for c in at.dataframe) or at.dataframe


def test_back_button_returns_to_mapping(at: AppTest) -> None:
    _load_demo(at)
    for b in at.button:
        if "Run analysis" in (b.label or ""):
            b.click().run()
            break
    at.run()
    for b in at.button:
        if "Back" in (b.label or ""):
            b.click().run()
            break
    at.run()
    assert not at.exception
    assert any("Column mapping" in s.value for s in at.subheader)


def _reach_verdict(at: AppTest) -> None:
    _load_demo(at)
    for b in at.button:
        if "Run analysis" in (b.label or ""):
            b.click().run()
            break
    at.run()


def _pin_scenario(at: AppTest) -> None:
    for b in at.button:
        if "Pin scenario" in (b.label or ""):
            b.click().run()
            return
    raise AssertionError("Pin scenario button not found")


def test_scenario_pin_and_compare(at: AppTest) -> None:
    _reach_verdict(at)
    _pin_scenario(at)
    at.run()
    # one pinned -> hint text, no compare table yet
    assert any("Scenario comparison" in s.value for s in at.subheader)
    _pin_scenario(at)
    at.run()
    # two pinned -> grouped compare chart + table rendered
    assert not at.exception
    subs = [s.value for s in at.subheader]
    assert "Scenario comparison" in subs
    assert len(at.dataframe) >= 1


def test_whatif_sweep(at: AppTest) -> None:
    _reach_verdict(at)
    for b in at.button:
        if "Run sweep" in (b.label or ""):
            b.click().run()
            break
    assert not at.exception


@file_uploader_supported
def test_scenario_save_restore_ui(at: AppTest) -> None:
    _reach_verdict(at)
    _pin_scenario(at)
    _pin_scenario(at)
    at.run()
    downloads = [d.label for d in at.get("download_button")]
    assert any("Save scenarios" in d for d in downloads)
    uploads = [u.label for u in at.file_uploader]
    assert any("Restore scenarios" in u for u in uploads)
