"""Artha-Drishti website.  Run:  streamlit run demo\\app.py"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st  # noqa: E402

from demo import ui  # noqa: E402
from demo.views import about, batch, comparison, detector, home, performance  # noqa: E402
from src import results_io as rio  # noqa: E402

st.set_page_config(page_title="Artha-Drishti | Sarcasm Detection", page_icon="🔍", layout="wide")
ui.inject_css()

PAGES = {"Home": home, "Sarcasm Detector": detector, "Batch Detector": batch, "Model Comparison": comparison,
         "Model Performance": performance, "About the Project": about}

with st.sidebar:
    st.markdown("## 🔍 Artha-Drishti")
    st.caption("Machine Learning-Based Sarcasm Detection in English Tweets")
    page = st.radio("Navigate", list(PAGES), label_visibility="collapsed")
    st.divider()
    paths = ui.get_paths()
    done = rio.completed_models(paths)
    st.caption(f"Trained models available: **{len(done)}**")
    st.caption("Text-only, topic-independent baseline.")

PAGES[page].render()
