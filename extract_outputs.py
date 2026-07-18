"""Extract cells from the executed notebook into notebook_outputs.json.

Produces a simplified JSON list of cells with type, source, and outputs
that build_walkthrough_md.py consumes.
"""
from __future__ import annotations

import json
from pathlib import Path

nb_path = Path("examples/walkthrough_executed.ipynb")
with open(nb_path, encoding="utf-8") as f:
    nb = json.load(f)

cells = []
for cell in nb["cells"]:
    entry: dict = {
        "type": cell["cell_type"],
        "source": "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"],
        "outputs": [],
    }
    if cell["cell_type"] == "code":
        for o in cell.get("outputs", []):
            if o["output_type"] == "stream":
                entry["outputs"].append({
                    "type": "stream",
                    "text": "".join(o["text"]) if isinstance(o["text"], list) else o["text"],
                })
            elif o["output_type"] == "execute_result":
                data = o.get("data", {})
                text = ""
                if "text/plain" in data:
                    t = data["text/plain"]
                    text = "".join(t) if isinstance(t, list) else t
                entry["outputs"].append({"type": "result", "text": text})
            elif o["output_type"] == "error":
                entry["outputs"].append({
                    "type": "stream",
                    "text": "\n".join(o.get("traceback", [])),
                })
    cells.append(entry)

out_path = Path("notebook_outputs.json")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(cells, f, ensure_ascii=False, indent=2)

print(f"Wrote {len(cells)} cells to {out_path}")
