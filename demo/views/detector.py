import html

import pandas as pd
import streamlit as st

from demo import ui
from src import batch as bt
from src import batch_multi as bm
from src.config import MAX_INPUT_CHARS, MULTI_TWEET_MAX, TWEET_CHARS
from src.predict import InputError

EXAMPLES = [
    "Oh great, another Monday meeting that could have been an email. Just what I needed.",
    "The match starts at 6 pm and the stadium opens at 4.",
    "I love how the wifi only dies when I have a deadline. Truly perfect timing.",
]
ICONS = {"Sarcastic": "🙄 Sarcastic", "Not Sarcastic": "✅ Not Sarcastic"}


def _set(text):
    st.session_state["tweet_text"] = text


def _show_single(res):
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


def _show_many(saved):
    rows, cfg = saved["rows"], saved["cfg"]
    s = bt.summarize(rows)
    st.subheader(f"Results for {s['total_rows']:,} tweets")
    st.caption(f"Model: **{rows.attrs['model_display']}** | threshold: "
               f"**{cfg['thr_type'].replace('_', ' ')}** ({rows['threshold'].dropna().iloc[0]:.3f})"
               if s["scored"] else f"Model: **{rows.attrs['model_display']}**")
    c = st.columns(4)
    c[0].metric("Tweets entered", f"{s['total_rows']:,}")
    c[1].metric("🙄 Sarcastic", f"{s['sarcastic']:,}", f"{s['sarcastic_pct']:.0f}% of scored", delta_color="off")
    c[2].metric("✅ Not Sarcastic", f"{s['not_sarcastic']:,}", f"{s['not_sarcastic_pct']:.0f}% of scored",
                delta_color="off")
    c[3].metric("Not scored", f"{s['skipped']:,}")
    t = bm.lines_table(rows)
    t["Predicted label"] = t["Predicted label"].map(ICONS).fillna("➖ Not scored")
    st.dataframe(t, hide_index=True, column_config={
        "#": st.column_config.NumberColumn(width="small"),
        "Tweet": st.column_config.TextColumn(width="large"),
        "Score": st.column_config.NumberColumn(format="%.3f"),
        "Threshold used": st.column_config.NumberColumn(format="%.3f")})
    st.caption(f"**Score meaning:** {rows.attrs['score_meaning']}. A tweet is labelled Sarcastic when its score "
               "passes the threshold. Tweets that could not be scored (no readable words, or over "
               f"{MAX_INPUT_CHARS} characters) are listed with the reason.")
    st.download_button("Download these results (CSV)", bt.to_csv_bytes(t), "sarcasm_results.csv", "text/csv")
    with st.expander("Text as seen by the model (after preprocessing) - first 5 scored tweets"):
        for tw in rows[rows.status == bt.STATUS_OK].tweet.head(5):
            st.code(bt.clean_text(tw))


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
    cols = st.columns(len(EXAMPLES) + 1)
    for i, ex in enumerate(EXAMPLES):
        cols[i].button(f"Example {i + 1}", key=f"ex{i}", on_click=_set, args=(ex,))
    cols[-1].button("All 3 examples", key="ex_all", on_click=_set, args=("\n".join(EXAMPLES),))

    text = st.text_area("Enter one or more English tweets (one tweet per line)", key="tweet_text", height=160,
                        max_chars=MULTI_TWEET_MAX * TWEET_CHARS,
                        placeholder="Type or paste a tweet...\nPress Enter to start the next tweet on a new line.")
    per_line = st.checkbox("Treat each line as a separate tweet", value=True,
                           help="Untick this if your single tweet itself contains line breaks.")
    lines, n_found = bm.split_tweets(text, MULTI_TWEET_MAX)
    many = per_line and n_found > 1
    if many:
        st.caption(f"{n_found} tweets detected (blank lines are ignored).")
    else:
        st.caption(f"{len(text)} / {MAX_INPUT_CHARS} characters (tweets are at most 280)")

    if st.button("Analyze Tweets" if many else "Analyze Tweet", type="primary"):
        st.session_state.pop("detector_many", None)
        if many:
            if n_found > MULTI_TWEET_MAX:
                st.warning(f"Only the first {MULTI_TWEET_MAX} of {n_found} tweets are analysed per click.")
            try:
                rows = bt.run_batch(predictor, pd.DataFrame({"tweet": lines}), "tweet", model_name, thr_type)
            except FileNotFoundError as e:
                st.error(str(e))
                return
            except Exception as e:  # keep the page usable; show a message, not a stack trace
                st.error(f"Could not analyse these tweets ({type(e).__name__}). Please try again.")
                return
            st.session_state["detector_many"] = {
                "rows": rows, "cfg": {"text": text, "model_name": model_name, "thr_type": thr_type}}
        else:
            try:
                res = predictor.predict(lines[0] if (per_line and lines) else text, model_name, thr_type)
            except InputError as e:
                st.warning(str(e))
                return
            except FileNotFoundError as e:
                st.error(str(e))
                return
            _show_single(res)

    saved = st.session_state.get("detector_many")
    if saved and many:
        if saved["cfg"] != {"text": text, "model_name": model_name, "thr_type": thr_type}:
            st.warning("The text or settings changed since the last analysis. Click **Analyze Tweets** to update "
                       "the results below.")
        _show_many(saved)
    st.info("**Limitations:** this is a text-only statistical model trained on one tweet dataset. It cannot see "
            "context, tone, the author, or cultural references, and it will make mistakes. Do not use it for "
            "decisions about people.")
