"""Tests for the ASHRAE DB II loader and benchmark logic — no network.

All tests either write synthetic cache files directly or monkeypatch
``urllib.request.urlopen``; the real database is never fetched.
"""

from __future__ import annotations

import hashlib
import io
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from comfio.integration import comfort_db

MEASUREMENTS_NAME = "db_measurements_v2.1.0.csv.gz"
METADATA_NAME = "db_metadata.csv"


def _measurements_df(n: int = 40) -> pd.DataFrame:
    rng = np.random.default_rng(3)
    return pd.DataFrame(
        {
            "building_id": rng.choice([f"b{i:02d}" for i in range(8)], n),
            "season": rng.choice(["summer", "winter"], n),
            "ta": rng.uniform(20.0, 30.0, n),
            "tr": rng.uniform(20.0, 30.0, n),
            "top": rng.uniform(20.0, 30.0, n),
            "vel": rng.uniform(0.05, 0.3, n),
            "t_mot_isd": rng.uniform(10.0, 33.0, n),
            "thermal_sensation": rng.integers(-3, 4, n).astype(float),
            "thermal_acceptability": rng.choice(
                ["acceptable", "unacceptable"], n
            ),
            "pmv": rng.uniform(-2.0, 2.0, n),
        }
    )


def _metadata_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "building_id": [f"b{i:02d}" for i in range(8)],
            "region": ["Europe"] * 8,
            "country": ["Italy"] * 4 + ["United Kingdom"] * 4,
            "climate": ["Mediterranean"] * 8,
            "building_type": ["office"] * 8,
            "cooling_type": ["naturally ventilated"] * 4 + ["air conditioned"] * 4,
        }
    )


def _measurements_bytes() -> bytes:
    buf = io.BytesIO()
    _measurements_df().to_csv(buf, index=False, compression="gzip")
    return buf.getvalue()


def _metadata_bytes() -> bytes:
    return _metadata_df().to_csv(index=False).encode("utf-8")


def _payloads() -> dict[str, bytes]:
    return {
        "measurements": _measurements_bytes(),
        "metadata": _metadata_bytes(),
    }


def _fake_db2_files(payloads: dict[str, bytes]) -> dict[str, tuple[str, str]]:
    return {
        key: (
            f"v2.1.0/{MEASUREMENTS_NAME if key == 'measurements' else METADATA_NAME}",
            hashlib.sha256(data).hexdigest(),
        )
        for key, data in payloads.items()
    }


def _write_cache(cache_dir: Path, payloads: dict[str, bytes]) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / MEASUREMENTS_NAME).write_bytes(payloads["measurements"])
    (cache_dir / METADATA_NAME).write_bytes(payloads["metadata"])


def _offline(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*args: object, **kwargs: object) -> object:
        raise urllib.error.URLError("no network")

    monkeypatch.setattr(urllib.request, "urlopen", boom)


@pytest.fixture()
def cached_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Path:
    """Cache dir pre-filled with synthetic DB2 files + patched checksums."""
    payloads = _payloads()
    cache = tmp_path / "db2"
    _write_cache(cache, payloads)
    monkeypatch.setattr(comfort_db, "DB2_FILES", _fake_db2_files(payloads))
    return cache


def test_load_from_cache_verifies_hash(
    cached_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _offline(monkeypatch)  # a valid cache must never touch the network
    df = comfort_db.load_ashrae_db2(cached_dir)
    assert len(df) == 40
    # merged metadata columns present alongside measurements
    for col in ("thermal_sensation", "t_mot_isd", "country", "climate", "cooling_type"):
        assert col in df.columns
    # second call: still a pure cache hit
    df2 = comfort_db.load_ashrae_db2(cached_dir)
    assert len(df2) == 40


def test_hash_mismatch_raises_and_cleans_tmp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payloads = _payloads()
    cache = tmp_path / "db2"
    _write_cache(cache, payloads)
    monkeypatch.setattr(comfort_db, "DB2_FILES", _fake_db2_files(payloads))
    bad = cache / MEASUREMENTS_NAME
    bad.write_bytes(b"corrupt-not-the-real-file")
    _offline(monkeypatch)
    with pytest.raises(comfort_db.ComfortDBUnavailable):
        comfort_db.load_ashrae_db2(cache)
    assert not list(cache.glob("*.tmp"))
    assert not bad.exists()  # corrupt cache entry was deleted


def test_download_hash_mismatch_discards_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payloads = _payloads()
    monkeypatch.setattr(comfort_db, "DB2_FILES", _fake_db2_files(payloads))
    # server returns garbage for every file
    monkeypatch.setattr(
        urllib.request, "urlopen", lambda *a, **k: io.BytesIO(b"junk")
    )
    cache = tmp_path / "db2"
    with pytest.raises(comfort_db.ComfortDBUnavailable):
        comfort_db.load_ashrae_db2(cache)
    assert not list(cache.glob("*.tmp"))
    assert not list(cache.glob("*.csv*"))


def test_download_then_cache_hit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payloads = _payloads()
    monkeypatch.setattr(comfort_db, "DB2_FILES", _fake_db2_files(payloads))

    def fake_urlopen(url: str, *args: object, **kwargs: object) -> io.BytesIO:
        if url.endswith(".csv.gz"):
            return io.BytesIO(payloads["measurements"])
        return io.BytesIO(payloads["metadata"])

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    cache = tmp_path / "db2"
    df = comfort_db.load_ashrae_db2(cache)
    assert len(df) == 40
    # second load is served from the verified cache
    _offline(monkeypatch)
    df2 = comfort_db.load_ashrae_db2(cache)
    assert len(df2) == 40


def test_offline_raises_unavailable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _offline(monkeypatch)
    with pytest.raises(comfort_db.ComfortDBUnavailable):
        comfort_db.load_ashrae_db2(tmp_path / "empty")
    with pytest.raises(comfort_db.ComfortDBUnavailable):
        comfort_db.load_ashrae_db2(tmp_path / "empty", download=False)


# --------------------------------------------------------------------------- #
# _logic helpers
# --------------------------------------------------------------------------- #

from comfio.gui.tools.benchmark import _logic  # noqa: E402


def test_filter_db() -> None:
    df = pd.DataFrame(
        {
            "climate": ["hot", "cold", "hot"],
            "building_type": ["office", "office", "classroom"],
            "cooling_type": ["nv", "ac", "nv"],
            "season": ["summer", "winter", "summer"],
            "country": ["IT", "UK", "IT"],
            "x": [1.0, 2.0, 3.0],
        }
    )
    out = _logic.filter_db(df, climate=["hot"], season=["summer"])
    assert len(out) == 2
    assert len(_logic.filter_db(df, country=["UK"])) == 1
    assert len(_logic.filter_db(df)) == 3  # no selections -> unfiltered
    assert len(_logic.filter_db(df, climate=["hot"], building_type=["classroom"])) == 1


def test_sensation_bins_known_values() -> None:
    df = pd.DataFrame(
        {
            "top": [24.2, 24.7, 26.1],
            "thermal_sensation": [0.0, 1.0, -1.0],
            "thermal_acceptability": [
                "acceptable",
                "unacceptable",
                "acceptable",
            ],
            "pmv": [0.1, 0.5, -0.8],
        }
    )
    bins = _logic.sensation_bins(df).set_index("t_bin")
    assert bins.loc[24.0, "n"] == 2
    assert bins.loc[24.0, "mean_tsv"] == pytest.approx(0.5)
    assert bins.loc[24.0, "pct_acceptable"] == pytest.approx(50.0)
    assert bins.loc[24.0, "mean_pmv"] == pytest.approx(0.3)
    assert bins.loc[26.0, "n"] == 1
    assert bins.loc[26.0, "mean_tsv"] == pytest.approx(-1.0)
    assert bins.loc[26.0, "pct_acceptable"] == pytest.approx(100.0)


def test_pmv_bias_sign() -> None:
    df = pd.DataFrame(
        {
            "cooling_type": ["air conditioned"] * 3 + ["naturally ventilated"] * 3,
            "pmv": [0.5, 0.6, 0.4] + [-0.5, -0.6, -0.4],
            "thermal_sensation": [0.0, 0.0, 0.0] + [0.2, 0.2, 0.2],
        }
    )
    bias = _logic.pmv_bias(df).set_index("cooling_type")
    assert bias.loc["air conditioned", "mean_bias"] > 0
    assert bias.loc["naturally ventilated", "mean_bias"] < 0
    assert bias.loc["air conditioned", "n"] == 3


def test_with_operative_computes_and_labels() -> None:
    df = pd.DataFrame(
        {
            "ta": [25.0, 26.0],
            "tr": [27.0, 26.0],
            "vel": [0.1, 0.1],
            "top": [np.nan, 26.5],
        }
    )
    out = _logic.with_operative(df)
    assert out["top"].iloc[0] == pytest.approx(26.0, abs=0.2)
    assert out["top_source"].iloc[0] == "operative_tmp(ta, tr, vel)"
    assert out["top"].iloc[1] == pytest.approx(26.5)
    assert out["top_source"].iloc[1] == "database"


def test_user_overlay_builds_columns() -> None:
    prepared = pd.DataFrame(
        {
            "ta": [24.0, 25.0],
            "tr": [25.0, 25.0],
            "tout": [15.0, 16.0],
            "vote": [0.0, 1.0],
        }
    )
    mapping = {
        "air_temp_c": "ta",
        "radiant_temp_c": "tr",
        "running_mean_outdoor_c": "tout",
        "tsv": "vote",
    }
    ov = _logic.user_overlay(prepared, mapping)
    assert ov is not None
    assert {"top", "t_mot", "thermal_sensation"} <= set(ov.columns)
    assert ov["t_mot"].tolist() == [15.0, 16.0]


def test_user_overlay_none_when_unusable() -> None:
    prepared = pd.DataFrame({"rh": [50.0, 55.0]})
    assert _logic.user_overlay(prepared, {"relative_humidity_pct": "rh"}) is None
