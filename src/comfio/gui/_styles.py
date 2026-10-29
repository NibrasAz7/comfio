"""Light CSS polish for the comfio Streamlit app."""

from __future__ import annotations

import streamlit as st

_CSS = """
<style>
  /* tighter canvas */
  .block-container { padding-top: 2rem; padding-bottom: 3rem; }

  /* metric cards */
  [data-testid="stMetric"] {
      background: #f7f9fa;
      border: 1px solid #e2e8ec;
      border-radius: 8px;
      padding: 10px 14px;
  }

  /* section headers match the brand palette */
  h1, h2, h3 { color: #264653; }
</style>
"""


def inject() -> None:
    """Apply the app-wide stylesheet (call once after set_page_config)."""
    st.markdown(_CSS, unsafe_allow_html=True)
