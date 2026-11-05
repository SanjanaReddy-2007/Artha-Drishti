"""Extra look-and-feel for the whole app. Additive only: it layers on top of ui.inject_css() and uses
translucent colours, so it works in both light and dark mode."""
import streamlit as st

CSS = """
<style>
.block-container { padding-top: 2rem; max-width: 1200px; }
h1 { letter-spacing: -0.02em; }
div[data-testid="stMetric"] { border: 1px solid rgba(128,128,128,.28); border-radius: 12px;
    padding: .7rem .95rem; background: rgba(128,128,128,.07); }
div[data-testid="stMetricLabel"] p { font-size: .85rem; opacity: .85; }
.stButton > button, .stDownloadButton > button { border-radius: 10px; font-weight: 600; }
textarea { border-radius: 10px !important; }
div[data-testid="stExpander"] { border-radius: 12px; }
div[data-testid="stDataFrame"] { border-radius: 10px; overflow: hidden; }
section[data-testid="stSidebar"] .stRadio label { padding: .15rem 0; }
</style>
"""


def inject():
    st.markdown(CSS, unsafe_allow_html=True)
