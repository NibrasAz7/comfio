# LLM Adapters

comfio provides LLM-native adapters for integrating IEQ evaluation with large language
models for automated building diagnostics.

## Table of Contents

1. [Markdown Interpreters](#markdown-interpreters)
2. [Summary Dict](#summary-dict)
3. [Full Pipeline Summary](#full-pipeline-summary)
4. [Prompt Templates](#prompt-templates)
5. [OpenAI Tool Schemas](#openai-tool-schemas)
6. [LangChain Tools](#langchain-tools)

---

## Markdown Interpreters

### ieq_to_markdown

```python
from comfio import ieq_to_markdown

md = ieq_to_markdown(
    ieq_result: GlobalIEQResult,
    compliance_report: ComplianceReport | None = None,
    zone_id: str | None = None,
) -> str
```

Produces a ~200-token structured markdown report:

```markdown
### Building System Report: Zone-A
- **Current Operational State**: COMPLIANT
- **Global IEQ Index Score**: 85.3/100 (min: 70.1, max: 95.2)
- **Contract Compliance Rate**: 90.0%
- **Timestamps Evaluated**: 24

#### Domain Breakdown:
  - [OK] THERMAL: 88.2/100
  - [OK] VISUAL: 82.1/100
  - [WARNING] ACOUSTIC: 65.3/100
  - [OK] IAQ: 91.0/100

#### Diagnostic Insight:
The primary limiting factor is the ACOUSTIC domain (score: 65.3/100).
Noise levels exceeded NC threshold. Action: inspect noise sources or add acoustic damping.
```

**Status logic:** "COMPLIANT" if compliance_rate > 85% or ieq_avg ≥ 80, else "NON-COMPLIANT".
**WARNING flag:** Domain avg < 70.

---

## Summary Dict

### ieq_to_summary_dict

```python
from comfio import ieq_to_summary_dict

summary = ieq_to_summary_dict(
    ieq_result: GlobalIEQResult,
    compliance_report: ComplianceReport | None = None,
) -> dict
```

Returns a flat dict with scalar aggregates:

```python
{
    "ieq_index_avg": 85.3,
    "ieq_index_min": 70.1,
    "ieq_index_max": 95.2,
    "ieq_index_std": 5.2,
    "n_timestamps": 24,
    "domains": ["thermal", "visual", "acoustic", "iaq"],
    "domain_scores_avg": {"thermal": 88.2, "visual": 82.1, ...},
    "worst_domain": "acoustic",
    "weights_used": {"thermal": 0.4, "iaq": 0.25, ...},
    # If compliance_report provided:
    "compliance_rate_pct": 90.0,
    "threshold": 80.0,
    "compliant_hours": 22.0,
    "total_hours": 24.0,
}
```

---

## Full Pipeline Summary

### generate_markdown_summary

```python
from comfio import generate_markdown_summary

md = generate_markdown_summary(
    df: pd.DataFrame,        # Raw sensor data
    window_hours: int = 24,
    threshold: float = 80.0,
    zone_id: str | None = None,
) -> str
```

Runs the **entire pipeline** automatically:
1. Creates `SensorData(df)` and validates
2. Detects available domains
3. Evaluates each domain (thermal, visual, acoustic, IAQ, pollutant IAQ if available)
4. Computes Global IEQ Index
5. Generates compliance report
6. Returns structured markdown with critical failure analysis

Output includes:
- Global IEQ average and compliance rate
- Per-domain summary (avg score + compliance %)
- Critical failures section (timestamps below threshold, worst domain, diagnostic suggestion)

---

## Prompt Templates

### EDGE_SYSTEM_PROMPT

```python
from comfio import EDGE_SYSTEM_PROMPT
```

A guarded system prompt for edge building diagnostics with safety rules:
1. Never invent physical metrics — rely on comfio outputs
2. Don't recommend increasing outdoor air if ambient > 30°C and PMV is high
3. Output reasoning first, then BACnet/Modbus control command in JSON
4. Prioritize filtration over ventilation if outdoor air quality is poor
5. Trust TSV over PMV when conflicting
6. Check adaptive comfort band before recommending mechanical intervention

Contains `{context}` and `{query}` placeholders.

### DIAGNOSTIC_PROMPT_TEMPLATE

```python
from comfio import DIAGNOSTIC_PROMPT_TEMPLATE
```

Template for diagnostic prompts with placeholders:
`{ieq_report}`, `{complaint}`, `{pollutant_status}`, `{adaptive_status}`, `{tsv_status}`

Requests: root cause analysis, immediate remediation, long-term recommendations,
model-vs-occupant discrepancy notes.

### format_prompt

```python
from comfio import format_prompt

prompt = format_prompt(EDGE_SYSTEM_PROMPT, context=md_report, query="Why is Zone-A uncomfortable?")
```

---

## OpenAI Tool Schemas

**Requires:** `pip install comfio[agent]`

```python
from comfio.llm.tools import to_openai_tools

tools = to_openai_tools()
# Returns list of dicts:
# [
#   {"type": "function", "function": {"name": "evaluate_thermal", ...}},
#   {"type": "function", "function": {"name": "evaluate_visual", ...}},
#   {"type": "function", "function": {"name": "evaluate_acoustic", ...}},
#   {"type": "function", "function": {"name": "evaluate_iaq", ...}},
#   {"type": "function", "function": {"name": "evaluate_ieq", ...}},
#   {"type": "function", "function": {"name": "calculate_global_ieq", ...}},
# ]
```

Use with OpenAI function calling:

```python
from openai import OpenAI
client = OpenAI()

response = client.chat.completions.create(
    model="gpt-4",
    tools=to_openai_tools(),
    messages=[{"role": "user", "content": "Evaluate thermal comfort for 24°C, 50% RH"}],
)
```

---

## LangChain Tools

**Requires:** `pip install comfio[agent]`

```python
from comfio.llm.tools import to_langchain_tools

lc_tools = to_langchain_tools()
# Returns list of LangChain Tool objects with .name, .description, .args_schema
```

Use with LangChain agents:

```python
from langchain.agents import AgentExecutor, create_openai_tools_agent

agent = create_openai_tools_agent(llm, lc_tools, prompt)
executor = AgentExecutor(agent=agent, tools=lc_tools)
result = executor.invoke({"input": "Check IEQ compliance for Zone-A"})
```

---

## Available Tool Functions

| Tool | Description |
|---|---|
| `evaluate_thermal_tool` | Evaluate thermal comfort from sensor arrays |
| `evaluate_visual_tool` | Evaluate visual comfort from illuminance |
| `evaluate_acoustic_tool` | Evaluate acoustic comfort from noise levels |
| `evaluate_iaq_tool` | Evaluate IAQ from CO₂ |
| `evaluate_ieq_tool` | Full multi-domain IEQ evaluation |
| `calculate_global_ieq_tool` | Merge domain results into Global IEQ Index |
