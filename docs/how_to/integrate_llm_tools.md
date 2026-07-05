# Integrate LLM Tools

## Problem

You want to use comfio's IEQ evaluation functions as tools callable by LLM agents (OpenAI function calling, LangChain tools) for automated building diagnostics.

---

## Solution

### Generate OpenAI tool schemas

```python
from comfio.llm.tools import to_openai_tools

# Get OpenAI-compatible tool definitions
tools = to_openai_tools()
# Returns a list of dicts with "type": "function", "function": {...}
# covering evaluate_thermal, evaluate_iaq, calculate_global_ieq, etc.
```

### Generate LangChain tools

```python
from comfio.llm.tools import to_langchain_tools

# Get LangChain Tool objects
lc_tools = to_langchain_tools()
# Each tool wraps a comfio function with name, description, and args_schema
```

### Generate diagnostic summaries

```python
from comfio import (
    calculate_global_ieq,
    generate_markdown_summary,
    ieq_to_summary_dict,
)

# After calculating IEQ
ieq = calculate_global_ieq(
    thermal=thermal, visual=visual,
    acoustic=acoustic, iaq=iaq,
)

# Convert to summary dict for LLM consumption
summary = ieq_to_summary_dict(ieq)

# Generate markdown report
markdown = generate_markdown_summary(ieq)
print(markdown)
```

### Use diagnostic prompt templates

```python
from comfio import DIAGNOSTIC_PROMPT_TEMPLATE, EDGE_SYSTEM_PROMPT, format_prompt

# Format a diagnostic prompt with IEQ data
prompt = format_prompt(
    DIAGNOSTIC_PROMPT_TEMPLATE,
    ieq_summary=summary,
    building_type="office",
)
```

---

## Available LLM Functions

| Function | Description |
|---|---|
| `to_openai_tools()` | OpenAI function-calling tool schemas |
| `to_langchain_tools()` | LangChain Tool objects |
| `ieq_to_summary_dict(ieq)` | Convert IEQ result to dict |
| `ieq_to_markdown(ieq)` | Convert IEQ result to markdown table |
| `generate_markdown_summary(ieq)` | Full diagnostic markdown report |
| `EDGE_SYSTEM_PROMPT` | System prompt for edge diagnostic agent |
| `DIAGNOSTIC_PROMPT_TEMPLATE` | Template for diagnostic prompts |
| `format_prompt(template, **kwargs)` | Format a prompt template with values |

---

## See Also

- [API Reference — LLM Tools](../reference/llm_tools.md)
- [Tutorial 4: LLM Diagnostics](../tutorials/04_llm_diagnostics.ipynb)
