"""Unit tests for the GUI's pure column-mapping logic (no streamlit needed)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from comfio.gui._data import (
    attach_outdoor_temperature,
    convert_ip_to_si,
    demo_data,
    excel_sheet_names,
    prepare,
    read_uploaded,
    template_csv,
)
from comfio.gui._mapping import (
    auto_map_columns,
    domain_coverage,
    missing_required,
    normalize,
    unmapped_columns,
)


class TestNormalize:
    def test_strips_units_and_parens(self) -> None:
        assert normalize("Zone Air Temp (°C)") == "zone_air_temp"
        assert normalize("CO2 [ppm]") == "co2"
        assert normalize("RH (%)") == "rh"

    def test_lowercase_and_separators(self) -> None:
        assert normalize("Air-Temp C") == "air_temp_c"


class TestAutoMap:
    def test_canonical_names_map_directly(self) -> None:
        df = demo_data()
        m = auto_map_columns(df)
        assert m["air_temp_c"] == "air_temp_c"
        assert m["co2_ppm"] == "co2_ppm"
        assert m["outdoor_temp_c"] == "outdoor_temp_c"

    def test_alias_names_map(self) -> None:
        df = pd.DataFrame({"tdb": [20.0], "rh": [50.0], "co2": [700.0]})
        m = auto_map_columns(df)
        assert m["air_temp_c"] == "tdb"
        assert m["relative_humidity_pct"] == "rh"
        assert m["co2_ppm"] == "co2"

    def test_ambiguous_names_not_guessed(self) -> None:
        # 'Zone 3 Temp' / 'CO2 Level' must NOT silently map — user decides.
        df = pd.DataFrame({"Zone 3 Temp": [20.0], "CO2 Level": [700.0]})
        m = auto_map_columns(df)
        assert "air_temp_c" not in m
        assert "co2_ppm" not in m

    def test_units_in_header_stripped_for_aliases(self) -> None:
        df = pd.DataFrame({"rh (%)": [50.0], "co2 [ppm]": [800.0]})
        m = auto_map_columns(df)
        assert m["relative_humidity_pct"] == "rh (%)"
        assert m["co2_ppm"] == "co2 [ppm]"


class TestCoverage:
    def test_full_demo_covers_all_domains(self) -> None:
        m = auto_map_columns(demo_data())
        assert domain_coverage(m) == {
            "thermal": True,
            "visual": True,
            "acoustic": True,
            "iaq": True,
        }

    def test_partial_file(self) -> None:
        m = {"air_temp_c": "t", "relative_humidity_pct": "rh"}
        cov = domain_coverage(m)
        assert cov == {
            "thermal": True,
            "visual": False,
            "acoustic": False,
            "iaq": False,
        }

    def test_missing_required(self) -> None:
        assert missing_required({"air_temp_c": "t"}, {"air_temp_c", "co2_ppm"}) == [
            "co2_ppm"
        ]

    def test_unmapped_columns_excludes_timestamp_tsv(self) -> None:
        df = pd.DataFrame(
            {"timestamp": [1], "tsv": [0], "air_temp_c": [22.0], "junk": ["x"]}
        )
        m = {"air_temp_c": "air_temp_c"}
        assert unmapped_columns(df, m) == ["junk"]


class TestDataHelpers:
    def test_template_csv_headers(self) -> None:
        assert "air_temp_c" in template_csv("si")
        assert "Air temperature [F]" in template_csv("ip")

    def test_read_uploaded_csv(self, tmp_path) -> None:
        import io

        class FakeUpload(io.BytesIO):
            name = "data.csv"

        f = FakeUpload(b"a,b\n1,2\n")
        df = read_uploaded(f)
        assert list(df.columns) == ["a", "b"]
        assert len(df) == 1

    def test_read_uploaded_rejects_other_types(self) -> None:
        import io

        class FakeUpload(io.BytesIO):
            name = "data.txt"

        with pytest.raises(ValueError, match="Unsupported file type"):
            read_uploaded(FakeUpload(b"x"))

    def test_excel_sheet_picker(self) -> None:
        import io

        pytest.importorskip("openpyxl")

        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="openpyxl") as w:
            pd.DataFrame({"a": [1]}).to_excel(w, sheet_name="Zone1", index=False)
            pd.DataFrame({"b": [2]}).to_excel(w, sheet_name="Zone2", index=False)
        buf.name = "bms.xlsx"
        buf.seek(0)

        assert excel_sheet_names(buf) == ["Zone1", "Zone2"]
        buf.seek(0)
        df = read_uploaded(buf, sheet="Zone2")
        assert list(df.columns) == ["b"]

    def test_prepare_injects_assumptions(self) -> None:
        df = pd.DataFrame(
            {"t": [20.0, np.nan, 22.0], "rh": [50.0, 45.0, 55.0]}
        )
        m = {"air_temp_c": "t", "relative_humidity_pct": "rh"}
        cfg = {"vr": 0.15, "_mrt_equals_tdb": True}
        prep, m2, notes = prepare(df, m, cfg)
        assert m2["radiant_temp_c"] == "__comfio_radiant_temp_c"
        assert m2["air_velocity_ms"] == "__comfio_air_velocity_ms"
        assert prep["__comfio_radiant_temp_c"].iloc[0] == 20.0
        assert prep["__comfio_air_velocity_ms"].iloc[1] == 0.15
        assert not prep["t"].isna().any()  # interpolated
        assert any("Radiant temperature" in n for n in notes)
        assert any("constant 0.15 m/s" in n for n in notes)

    def test_prepare_drop_strategy(self) -> None:
        df = pd.DataFrame({"t": [20.0, np.nan, 22.0], "rh": [50.0, 45.0, 55.0]})
        m = {"air_temp_c": "t", "relative_humidity_pct": "rh"}
        prep, _, _ = prepare(df, m, {"_mrt_equals_tdb": True}, nan_strategy="drop")
        assert len(prep) == 2

    def test_demo_has_sparse_votes(self) -> None:
        df = demo_data()
        assert "tsv" in df.columns
        n_votes = int(df["tsv"].notna().sum())
        assert 0 < n_votes < len(df)  # sparse -> augmentation path

    def test_attach_outdoor_merges_nearest_hour(self, monkeypatch) -> None:
        import numpy as np

        df = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(
                    ["2026-03-01 09:10", "2026-03-01 10:40", "2026-03-01 23:30"]
                ),
                "air_temp_c": [22.0, 23.0, 21.0],
            }
        )
        wx = pd.DataFrame(
            {"temp": [5.0, 7.0]},
            index=pd.to_datetime(["2026-03-01 09:00", "2026-03-01 11:00"]),
        )
        import comfio.integration.weather as weather

        monkeypatch.setattr(
            weather, "fetch_outdoor_temperature", lambda *a, **k: wx
        )
        out = attach_outdoor_temperature(df, "timestamp", 40.0, -74.0)
        vals = out["outdoor_temp_c"].to_numpy()
        assert vals[0] == 5.0  # 09:10 -> nearest 09:00
        assert vals[1] == 7.0  # 10:40 -> nearest 11:00
        assert np.isnan(vals[2])  # 23:30 outside 1 h tolerance

    def test_convert_ip_to_si(self) -> None:
        df = pd.DataFrame({"t_f": [68.0], "v_fpm": [19.685], "rh": [50.0]})
        m = {"air_temp_c": "t_f", "air_velocity_ms": "v_fpm"}
        out = convert_ip_to_si(df, m)
        assert out["t_f"].iloc[0] == pytest.approx(20.0, abs=0.01)
        assert out["v_fpm"].iloc[0] == pytest.approx(0.1, abs=0.001)
        assert out["rh"].iloc[0] == 50.0  # untouched
        assert df["t_f"].iloc[0] == 68.0  # original not mutated
