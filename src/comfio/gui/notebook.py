"""Jupyter notebook widget for comfio — a minimal in-notebook workbench.

``gui_notebook(df)`` returns an ``ipywidgets.VBox`` with a column mapper,
occupant parameters, and a Run button that executes ``run_pipeline`` and
renders the verdict inline. Pure ipywidgets — no streamlit dependency.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from comfio.core.data_handler import SensorData
from comfio.gui import _data, _mapping

_NOT_AVAILABLE = "— not available —"


def gui_notebook(df: pd.DataFrame) -> Any:
    """Create the comfio notebook widget for a DataFrame.

    Returns
    -------
    ipywidgets.VBox
        Display-ready widget. Requires ``ipywidgets`` (``comfio[gui]``).
    """
    try:
        import ipywidgets as widgets
        from IPython.display import display  # noqa: F401
    except ModuleNotFoundError as exc:
        raise ImportError(
            "gui_notebook requires ipywidgets: pip install comfio[gui]"
        ) from exc

    auto = _mapping.auto_map_columns(df)
    options = [_NOT_AVAILABLE] + list(df.columns)
    sensor_cols = [
        c for c in _mapping.CANONICAL_COLUMNS if c in _data.COLUMN_DESCRIPTIONS
    ]

    dropdowns: dict[str, Any] = {}
    rows = []
    for canonical in sensor_cols:
        dd = widgets.Dropdown(
            options=options,
            value=auto.get(canonical, _NOT_AVAILABLE),
            description=_data.COLUMN_DESCRIPTIONS[canonical],
            style={"description_width": "220px"},
            layout=widgets.Layout(width="480px"),
        )
        dropdowns[canonical] = dd
        rows.append(dd)

    ts_dd = widgets.Dropdown(
        options=options,
        value=next(
            (
                c
                for c in df.columns
                if _mapping.normalize(c)
                in {"timestamp", "time", "date", "datetime"}
            ),
            _NOT_AVAILABLE,
        ),
        description="Timestamp",
        style={"description_width": "220px"},
        layout=widgets.Layout(width="480px"),
    )

    met = widgets.FloatText(value=1.2, description="met")
    clo = widgets.FloatText(value=0.5, description="clo")
    vr = widgets.FloatText(value=0.1, description="air speed m/s")
    mrt = widgets.Checkbox(value=True, description="MRT = air temp")
    params = widgets.HBox([met, clo, vr, mrt])

    run_btn = widgets.Button(
        description="Run analysis", button_style="primary", icon="play"
    )
    out = widgets.Output()

    def _on_run(_: Any) -> None:
        with out:
            out.clear_output()
            mapping = {
                c: dd.value
                for c, dd in dropdowns.items()
                if dd.value != _NOT_AVAILABLE
            }
            coverage = _mapping.domain_coverage(mapping)
            if not any(coverage.values()):
                print("No evaluable domain — map more columns.")
                return
            config: dict[str, Any] = {
                "met": met.value,
                "clo": clo.value,
                "vr": vr.value,
                "category": "B",
                "task_type": "general",
                "nc_level": "NC-35",
                "co2_threshold": "good",
                "weight_preset": "default",
                "threshold": 80.0,
                "_mrt_equals_tdb": mrt.value,
            }
            prepared, mapping2, notes = _data.prepare(df, mapping, config)
            ts_col = None if ts_dd.value == _NOT_AVAILABLE else ts_dd.value
            sensor = SensorData(
                df=prepared, column_map=mapping2, timestamp_col=ts_col
            )
            from comfio.reports.pipeline import run_pipeline

            result = run_pipeline(
                sensor, {k: v for k, v in config.items() if not k.startswith("_")}
            )
            if result.ieq_result is None:
                print("No domain could be evaluated.")
                return
            ieq = result.ieq_result
            mean = float(np.nanmean(ieq.index))
            rows_html = "".join(
                f"<tr><td>{d}</td><td>{np.nanmean(s):.0f}</td></tr>"
                for d, s in ieq.domain_scores.items()
            )
            notes_html = "".join(f"<li>{n}</li>" for n in notes + result.warnings)
            widgets.HTML(  # render into the output area
                value=(
                    f"<h3>Global IEQ Index: {mean:.0f} / 100</h3>"
                    f"<table><tr><th>Domain</th><th>Score</th></tr>"
                    f"{rows_html}</table>"
                    f"<b>Assumptions & warnings</b><ul>{notes_html}</ul>"
                )
            )._ipython_display_()

    run_btn.on_click(_on_run)

    return widgets.VBox(
        [
            widgets.HTML("<h3>comfio — column mapping</h3>"),
            ts_dd,
            *rows,
            widgets.HTML("<b>Occupant parameters</b>"),
            params,
            run_btn,
            out,
        ]
    )
