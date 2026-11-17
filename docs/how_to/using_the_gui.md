# Using the comfio GUI

## Problem

You want to evaluate Indoor Environmental Quality from sensor data — or explore
comfort models — without writing Python code.

## Solution

comfio ships a Streamlit-based GUI: a multipage set of independent tools,
collectively called the **IEQ Analyst Workbench**.

### 1. Install the GUI extra

```bash
pip install comfio[gui]
```

This pulls in `streamlit>=1.51`, `plotly`, `openpyxl`, `ipywidgets`, and
`pywebview`. For PDF/DOCX report exports add the reports extra:

```bash
pip install comfio[gui-reports]
```

### 2. Launch the app

```python
import comfio

comfio.gui()               # browser app on http://localhost:8501
comfio.gui(df)             # preloaded pandas DataFrame
comfio.gui(desktop=True)   # native pywebview window instead of a browser
comfio.gui(tools=["workbench", "calculator"])  # enable only some tools
```

From the command line:

```bash
streamlit run streamlit_app.py
```

### 3. Pick a tool

The app opens on the **workbench** (the default page). The navigation groups
each tool under its own page:

| Tool | id | What it does |
| --- | --- | --- |
| Analyse my data | `workbench` | upload → mapping → Global IEQ verdict → evidence/exports |
| Occupant survey | `survey` | satisfaction-vote import, % satisfied, sensor-vs-occupant check |
| Comfort calculator | `calculator` | PMV/PPD, SET, adaptive models, compare & ranges, psychrometric chart |
| Comfort tool | `comfort_tool` | CBE-style embedded web app for quick comfort what-ifs |
| MRT room designer | `mrt` | plan-view MRT/PMV map of a rectangular room, solar ΔMRT |
| Setpoint explorer | `setpoint` | comfort-band edges and deadband widening |
| Global database | `benchmark` | ASHRAE DB II (CC0) adaptive scatter, sensation bins, PMV bias |
| Learn | `learn` | guided lessons that hand inputs to the calculator |
| About & sources | `about` | enabled tools, provenance, licences, citations (always on) |

The **About** page is always enabled and lists every other tool currently
turned on, with provenance and citations.

### 4. Workbench flow

The default page walks through four stages:

1. **Upload** — CSV or Excel (multi-sheet supported, SI or IP units) or a
   bundled demo dataset.
2. **Inspect** — review and correct the auto-detected column mapping; optionally
   fetch meteostat outdoor temperature to fill gaps.
3. **Verdict** — Global IEQ gauge (0–100), per-domain traffic lights, comfort
   scatter, hour-of-day exceedance heatmap, and occupant-vote calibration
   (PMV→TSV).
4. **Evidence** — compliance rates, WELL v2 T06 / RESET Air certification
   presets, and exports: CSV, DOCX, PDF, contract-JSON, Markdown, or a
   standalone Python script that reproduces the run.

Assumptions such as `MRT = air temperature` or constant air speed are shown
explicitly in the UI — they are never applied silently.

### 5. Use the GUI inside Jupyter

```python
w = comfio.gui_notebook(df)  # ipywidgets mini-workbench
w
```

### 6. Enable or disable tools

Three equivalent mechanisms, applied in order:

```python
# Per launch
comfio.gui(tools=["workbench", "calculator"])
```

```bash
# Environment variable (comma-separated tool ids)
COMFIO_GUI_TOOLS=calculator,about streamlit run streamlit_app.py
```

```python
# Permanently, in the tool registry
from comfio.gui.tools import BUILTIN_TOOLS
# edit comfio/gui/tools/__init__.py — one line per tool
```

### 7. Deploy the workbench

The repository ships `streamlit_app.py` and `requirements.txt`, so you can
point a [Streamlit Community Cloud](https://share.streamlit.io/) app directly at
`streamlit_app.py`. For self-hosting, run
`streamlit run streamlit_app.py` behind any reverse proxy.
