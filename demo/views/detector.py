import html

import streamlit as st

from demo import ui
from src.config import MAX_INPUT_CHARS
from src.predict import InputError

EXAMPLES = [
    "Oh great, another Monday meeting that could have been an email. Just what I needed.",
    "The match starts at 6 pm and the stadium opens at 4.",
    "I love how the wifi only dies when I have a deadline. Truly perfect timing.",
]


def _set(text):
    st.session_state["tweet_text"] = text


def render():
    paths = ui.get_paths()
    st.title("Sarcasm Detector")
    ui.smoke_test_banner(paths)
    predictor = ui.get_predictor()
    models = predictor.available_models()
    if not models:
        ui.pending_notice()
        return
    default = predictor.default_model_name()

    c1, c2 = st.columns(2)
    default_label = f"Default (selected on validation data: {default})" if default else "Default"
    choice = c1.selectbox("Model", [default_label] + models)
    thr_label = c2.radio("Decision threshold", ["Validation-selected", "Model default"], horizontal=True)
    model_name = None if choice == default_label else choice
    thr_type = "validation_selected" if thr_label == "Validation-selected" else "default"

    st.caption("Examples (for trying the interface only; they are not from the dataset or used in any metric):")
    cols = st.columns(len(EXAMPLES))
    for i, ex in enumerate(EXAMPLES):
        cols[i].button(f"Example {i + 1}", key=f"ex{i}", on_click=_set, args=(ex,))

    text = st.text_area("Enter an English tweet", key="tweet_text", height=140, max_chars=MAX_INPUT_CHARS,
                        placeholder="Type or paste a tweet...")
    st.caption(f"{len(text)} / {MAX_INPUT_CHARS} characters (tweets are at most 280)")

    if st.button("Analyze Tweet", type="primary"):
        try:
            res = predictor.predict(text, model_name, thr_type)
        except InputError as e:
            st.warning(str(e))
            return
        except FileNotFoundError as e:
            st.error(str(e))
            return
        css = "ad-sarcastic" if res["is_sarcastic"] else "ad-notsarcastic"
        icon = "🙄" if res["is_sarcastic"] else "✅"
        st.markdown(f'<div class="ad-result {css}"><div class="label">{icon} {html.escape(res["label"])}</div>'
                    f'<div>Model: <b>{html.escape(res["model_display"])}</b></div></div>', unsafe_allow_html=True)
        m1, m2, m3 = st.columns(3)
        m1.metric("Model score", f"{res['score']:.3f}")
        m2.metric("Threshold used", f"{res['threshold']:.3f}")
        m3.metric("Threshold type", res["threshold_type"].replace("_", " "))
        st.caption(f"**Score meaning:** {res['score_meaning']}. A tweet is labelled Sarcastic when its score "
                   "passes the threshold.")
        for w in res["warnings"]:
            st.warning(w)
        with st.expander("Text as seen by the model (after preprocessing)"):
            st.code(res["cleaned_text"])
    st.info("**Limitations:** this is a text-only statistical model trained on one tweet dataset. It cannot see "
            "context, tone, the author, or cultural references, and it will make mistakes. Do not use it for "
            "decisions about people.")
