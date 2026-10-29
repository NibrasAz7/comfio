"""Unit tests for the GUI's pure column-mapping logic (no streamlit needed)."""

from __future__ import annotations

import pandas as pd
import pytest

from comfio.gui._data import (
    convert_ip_to_si,
    demo_data,
    excel_sheet_names,
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

    def test_convert_ip_to_si(self) -> None:
        df = pd.DataFrame({"t_f": [68.0], "v_fpm": [19.685], "rh": [50.0]})
        m = {"air_temp_c": "t_f", "air_velocity_ms": "v_fpm"}
        out = convert_ip_to_si(df, m)
        assert out["t_f"].iloc[0] == pytest.approx(20.0, abs=0.01)
        assert out["v_fpm"].iloc[0] == pytest.approx(0.1, abs=0.001)
        assert out["rh"].iloc[0] == 50.0  # untouched
        assert df["t_f"].iloc[0] == 68.0  # original not mutated
