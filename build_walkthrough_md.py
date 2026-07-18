"""Build docs/tutorials/06_walkthrough.md from the executed notebook.

Pulls markdown cells, code cells, and outputs directly from the notebook
(single source of truth). The notebook's markdown cells contain all
narrative, theory boxes, and references.
"""
from __future__ import annotations

import json
from pathlib import Path

with open("notebook_outputs.json", encoding="utf-8") as f:
    cells = json.load(f)


def get_stream_outputs(cell: dict) -> str:
    """Concatenate all stream outputs from a code cell, filtering warnings."""
    parts = []
    for o in cell.get("outputs", []):
        if o["type"] == "stream":
            text = o["text"]
            lines = text.split("\n")
            filtered = [
                l for l in lines
                if "UserWarning" not in l
                and "pythermalcomfort" not in l
                and "valid_range" not in l
                and "pmv_valid" not in l
                and "outside the applicability" not in l
                and "it/s]" not in l
                and "s/it]" not in l
                and l.strip() != ""
            ]
            if filtered:
                parts.append("\n".join(filtered))
    return "\n".join(parts)


def get_result_output(cell: dict) -> str:
    for o in cell.get("outputs", []):
        if o["type"] == "result":
            return o["text"]
    return ""


def format_output(cell: dict) -> str:
    """Format all text outputs as a markdown code block."""
    stream = get_stream_outputs(cell)
    result = get_result_output(cell)
    parts = []
    if stream.strip():
        parts.append(stream.strip())
    if result.strip():
        parts.append(result.strip())
    if not parts:
        return ""
    combined = "\n".join(parts)
    if len(combined) > 3000:
        combined = combined[:3000] + "\n... (truncated)"
    return f"```\n{combined}\n```"


# ===========================================================================
# Build the markdown document by iterating through notebook cells
# ===========================================================================
md_parts: list[str] = []


def w(s: str) -> None:
    md_parts.append(s)


# Title and intro (not in the notebook — markdown-specific framing)
w("""# comfio — Complete Walkthrough

> **A living document**: this walkthrough grows with the package. Every public
> API is exercised here on a single synthetic dataset so the behaviour is
> end-to-end reproducible. The companion executed notebook is at
> [`examples/walkthrough_executed.ipynb`](../../examples/walkthrough_executed.ipynb).

> **Run this notebook yourself** with
> `pip install comfio[ml,torch,keras,agent,acoustics,color,psychrometrics]`
> plus `plotly tensorflow langchain web3 reportlab python-docx`.
> On Windows, `[daylighting]` (Radiance) needs WSL.

---

""")

# Iterate through all notebook cells in order.
# Cell 0 is the notebook title (skip — we have our own above).
# Markdown cells are emitted as-is.
# Code cells are emitted as ```python blocks followed by their output.
for cell in cells[1:]:
    if cell["type"] == "markdown":
        w(cell["source"])
        w("")
    elif cell["type"] == "code":
        # Skip empty code cells
        if not cell["source"].strip():
            continue
        w(f"```python\n{cell['source']}\n```")
        output = format_output(cell)
        if output:
            w("")
            w("**Output:**")
            w("")
            w(output)
        w("")
        w("---")
        w("")

# Footer
w("""
*This walkthrough is a living document. To regenerate the executed notebook:*

```bash
python build_walkthrough_nb.py
python -m nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.timeout=600 \
    examples/walkthrough_executed.ipynb
```

*To regenerate this markdown:*

```bash
python extract_outputs.py
python build_walkthrough_md.py
```
""")

# --- Write the file ---
output = "\n".join(md_parts)
out_path = Path("docs/tutorials/06_walkthrough.md")
out_path.parent.mkdir(parents=True, exist_ok=True)
out_path.write_text(output, encoding="utf-8")
print(f"Wrote {len(output):,} chars to {out_path}")
