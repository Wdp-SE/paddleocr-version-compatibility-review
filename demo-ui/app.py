"""Streamlit entry point for versioned PaddleOCR knowledge and application review."""

from __future__ import annotations

import streamlit as st

st.set_page_config(
    page_title="PaddleOCR 版本知识与应用兼容性审查",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

from public_workbench import render

render()
