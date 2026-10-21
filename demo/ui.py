"""Shared UI helpers: styling, cached loaders, and 'pending' notices."""
from __future__ import annotations

import streamlit as st

from src import results_io as rio
from src.config import ProjectPaths
from src.predict import Predictor

CSS = """
<style>
.block-container {padding-top: 2rem; max-width: 1150px;}
.ad-hero {padding: 1.6rem 1.8rem; border-radius: 14px; border: 1px solid rgba(128,128,128,.35);
          background: linear-gradient(135deg, rgba(99,102,241,.14), rgba(20,184,166,.12));}
.ad-hero h1 {margin: 0 0 .2rem 0; font-size: 2.3rem;}
.ad-hero p {margin: 0; opacity: .85; font-size: 1.05rem;}
.ad-card {padding: 1.1rem 1.3rem; border-radius: 12px; border: 1px solid rgba(128,128,128,.35); margin: .4rem 0;}
.ad-result {padding: 1.3rem 1.5rem; border-radius: 14px; border-width: 3px; border-style: solid; margin: .8rem 0;}
.ad-result .label {font-size: 1.8rem; font-weight: 700;}
.ad-sarcastic {border-color: #d97706; background: rgba(217,119,6,.12);}
.ad-notsarcastic {border-color: #0f9d8a; background: rgba(15,157,138,.12);}
.ad-pill {display:inline-block; padding: .1rem .6rem; border-radius: 999px; font-size: .8rem;
          border: 1px solid rgba(128,128,128,.5); margin-right: .3rem;}
</style>
"""


def get_paths() -> ProjectPaths:
    return ProjectPaths()


@st.cache_resource
def get_predictor() -> Predictor:
    """One Predictor per server process; saved pipelines are loaded lazily and cached (no retraining)."""
    return Predictor(ProjectPaths())


def inject_css():
    st.markdown(CSS, unsafe_allow_html=True)


def pending_notice():
    st.info("No trained models were found yet, so there is nothing to display. This page only shows "
            "**real saved results**. Train the models first (PowerShell, project folder, venv active):")
    st.code("python -m src.prepare_data\npython -m src.split\npython -m src.train", language="powershell")


def smoke_test_banner(paths: ProjectPaths):
    meta = rio.load_run_metadata(paths)
    if meta and meta.get("smoke_test"):
        st.error("These results come from a SMOKE TEST (reduced training data). They are not a valid experiment.")
    return meta
