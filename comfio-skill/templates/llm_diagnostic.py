"""comfio LLM diagnostic template: IEQ → markdown summary → prompt → diagnostic output.

Requires: pip install comfio (core only, no extras needed for interpreters/prompts)
"""

import pandas as pd

from comfio import (
    DIAGNOSTIC_PROMPT_TEMPLATE,
    EDGE_SYSTEM_PROMPT,
    format_prompt,
    generate_markdown_summary,
    ieq_to_markdown,
    ieq_to_summary_dict,
)

# --- 1. Generate full markdown diagnostic from raw sensor DataFrame ---
md_report = generate_markdown_summary(
    df=pd.read_csv("sensor_data.csv"),
    window_hours=24,
    threshold=80.0,
    zone_id="Zone-A",
)

print("=== Markdown Diagnostic Report ===")
print(md_report)

# --- 2. Format a diagnostic prompt for an LLM ---
prompt = format_prompt(
    DIAGNOSTIC_PROMPT_TEMPLATE,
    ieq_report=md_report,
    complaint="Occupants report feeling too warm in the afternoon",
    pollutant_status="PM2.5 within WHO limits",
    adaptive_status="Not applicable (mechanically ventilated)",
    tsv_status="No TSV data available",
)

print("\n=== Formatted Diagnostic Prompt ===")
print(prompt)

# --- 3. System prompt for an edge diagnostic agent ---
# Contains safety rules: never invent metrics, trust TSV over PMV, etc.
print("\n=== Edge System Prompt (first 500 chars) ===")
print(EDGE_SYSTEM_PROMPT[:500])

# --- 4. If you have an LLM API, send the prompt ---
# Example with OpenAI (requires pip install openai):
#
# from openai import OpenAI
# client = OpenAI()
# response = client.chat.completions.create(
#     model="gpt-4",
#     messages=[
#         {"role": "system", "content": EDGE_SYSTEM_PROMPT},
#         {"role": "user", "content": prompt},
#     ],
# )
# print(response.choices[0].message.content)
