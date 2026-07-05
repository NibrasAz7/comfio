"""LLM-native adapters for comfio — semantic serialization, tool schemas, and prompt templates.

Public API:
    interpreters  — ieq_to_markdown, ieq_to_summary_dict, generate_markdown_summary
    prompts       — EDGE_SYSTEM_PROMPT, DIAGNOSTIC_PROMPT_TEMPLATE, format_prompt
    tools         — evaluate_thermal_tool, evaluate_ieq_tool, to_openai_tools, to_langchain_tools

The ``interpreters`` and ``prompts`` modules require no extra dependencies.
The ``tools`` module requires ``pip install comfio[agent]`` (adds pydantic).
"""

from comfio.llm.interpreters import (
    generate_markdown_summary,
    ieq_to_markdown,
    ieq_to_summary_dict,
)
from comfio.llm.prompts import (
    DIAGNOSTIC_PROMPT_TEMPLATE,
    EDGE_SYSTEM_PROMPT,
    format_prompt,
)

__all__ = [
    "ieq_to_markdown",
    "ieq_to_summary_dict",
    "generate_markdown_summary",
    "EDGE_SYSTEM_PROMPT",
    "DIAGNOSTIC_PROMPT_TEMPLATE",
    "format_prompt",
]
