import hashlib

import altair as alt
import pandas as pd
import streamlit as st

from demo import ui
from src import batch as bt
from src.config import BATCH_MAX_MB, BATCH_MAX_ROWS

COLORS = {"Sarcastic": "#d97706", "Not Sarcastic": "#0f9d8a"}


def _donut(s):
    d = pd.DataFrame({"Label": ["Sarcastic", "Not Sarcastic"], "Tweets": [s["sarcastic"], s["not_sarcastic"]]})
    return (alt.Chart(d).mark_arc(innerRadius=60)
            .encode(theta="Tweets:Q",
                    color=alt.Color("Label:N", scale=alt.Scale(domain=list(COLORS), range=list(COLORS.values())),
                                    title=None),
                    tooltip=["Label", "Tweets"])
            .properties(width=320, height=240))


def _show_results(res, parsed_df, cfg, predictor, paths):
    rows, s = res["rows"], res["summary"]
    st.divider()
    st.subheader("Results")
    st.caption(f"Model: **{cfg['model_display']}** | threshold: **{cfg['threshold_mode'].replace('_', ' ')}** "
               f"| tweet column: **{cfg['tweet_col']}** | {s['total_rows']:,} rows uploaded")
    c = st.columns(5)
    c[0].metric("Total tweets", f"{s['total_rows']:,}")
    c[1].metric("Sarcastic", f"{s['sarcastic']:,}", f"{s['sarcastic_pct']:.1f}% of scored", delta_color="off")
    c[2].metric("Not Sarcastic", f"{s['not_sarcastic']:,}", f"{s['not_sarcastic_pct']:.1f}% of scored", delta_color="off")
    c[3].metric("Scored", f"{s['scored']:,}")
    c[4].metric("Skipped", f"{s['skipped']:,}")
    st.caption("Percentages are shares of the tweets that were scored (skipped rows excluded). "
               f"Sarcastic + Not Sarcastic + skipped = {s['total_rows']:,} total rows.")
    if s["scored"]:
        st.altair_chart(_donut(s))
    if s["duplicate_rows"]:
        st.caption(f"{s['duplicate_rows']:,} duplicate tweet(s) kept in the output; each unique text was scored once.")
    if s["skipped"]:
        with st.expander(f"Skipped rows ({s['skipped']:,})"):
            sk = rows[rows.status != bt.STATUS_OK][["row_number", "tweet", "status"]]
            st.dataframe(sk.rename(columns={"row_number": "Row", "tweet": "Tweet", "status": "Reason"}),
                         hide_index=True)
    st.info("**Caution:** these are model predictions, and individual predictions can be wrong. On the held-out "
            "test set the default model reaches only about 0.37 sarcastic-class F1 (wide confidence intervals; see "
            "Model Performance). Scores are **uncalibrated**: "
            f"{cfg['score_meaning'][0].lower() + cfg['score_meaning'][1:]}.")

    # ---- table with filter / search / sort
    st.subheader("Classification of every tweet")
    f1, f2, f3 = st.columns([1.2, 1.6, 1.2])
    flt = f1.radio("Show", ["All", "Sarcastic", "Not Sarcastic"], horizontal=True)
    q = f2.text_input("Search tweet text", placeholder="type part of a tweet...")
    srt = f3.selectbox("Sort by", ["Original order", "Score: high to low", "Score: low to high"])
    t = rows[rows.status == bt.STATUS_OK]
    if flt != "All":
        t = t[t.predicted_label == flt]
    if q.strip():
        t = t[t.tweet.str.contains(q.strip(), case=False, regex=False)]
    if srt != "Original order":
        t = t.sort_values("score", ascending=srt.endswith("low to high"), kind="stable")
    show = t[["row_number", "tweet", "predicted_label", "score", "threshold"]].rename(columns={
        "row_number": "Row", "tweet": "Tweet", "predicted_label": "Predicted label", "score": "Score",
        "threshold": "Threshold used"})
    st.caption(f"Showing {len(show):,} of {s['scored']:,} scored tweets.")
    st.dataframe(show, hide_index=True, column_config={
        "Score": st.column_config.NumberColumn(format="%.3f"),
        "Threshold used": st.column_config.NumberColumn(format="%.3f")})
    with st.expander("Text as seen by the model (after preprocessing) - first 5 scored tweets"):
        for tw in rows[rows.status == bt.STATUS_OK].tweet.head(5):
            st.code(bt.clean_text(tw))

    # ---- downloads
    out = bt.build_output(parsed_df, rows)
    d1, d2 = st.columns(2)
    d1.download_button("Download full results (CSV)", bt.to_csv_bytes(out), "sarcasm_batch_results.csv",
                       "text/csv", type="primary")
    d2.download_button("Download summary (CSV)",
                       bt.to_csv_bytes(bt.summary_frame(s, cfg["model_name"], cfg["threshold_mode"])),
                       "sarcasm_batch_summary.csv", "text/csv")

    # ---- optional: labelled check
    if cfg.get("label_col"):
        try:
            lc = bt.labelled_check(parsed_df, rows, cfg["label_col"])
        except bt.BatchError as e:
            st.warning(str(e))
        else:
            st.subheader("Ad-hoc check against your own labels")
            st.warning("This is an ad-hoc check on your own file. It is NOT the official test-set result in the "
                       "Performance page. If your file contains tweets the models were trained on (for example "
                       "the iSarcasmEval training file), the numbers will look better than they should.")
            m = st.columns(5)
            m[0].metric("Sarcastic precision", f"{lc['precision']:.3f}")
            m[1].metric("Sarcastic recall", f"{lc['recall']:.3f}")
            m[2].metric("Sarcastic F1", f"{lc['f1']:.3f}")
            m[3].metric("Accuracy", f"{lc['accuracy']:.3f}")
            m[4].metric("Rows used", f"{lc['n_used']:,}")
            st.caption(f"{lc['n_sarcastic_true']:,} of {lc['n_used']:,} labelled rows are Sarcastic. "
                       "Accuracy is misleading when classes are imbalanced; look at the sarcastic-class metrics."
                       + (f" {lc['n_invalid_labels']:,} scored row(s) had no valid 0/1 label and were left out."
                          if lc["n_invalid_labels"] else ""))
            st.dataframe(lc["confusion"])

    # ---- optional: compare models
    with st.expander("Compare models (runs all trained models on this file)", expanded=False):
        models = predictor.available_models()
        st.caption(f"Scores every unique valid tweet with each of the {len(models)} trained models at the selected "
                   "threshold mode. This can take a while for large files, so it only runs when you click.")
        if len(rows[rows.status == bt.STATUS_OK]) > 2000:
            st.warning("Large file: this may take a minute or more.")
        if st.button("Run all models"):
            bar = st.progress(0.0, text="Running models...")
            cmp_df = bt.compare_models(predictor, rows, models, cfg["threshold_mode"],
                                       progress=lambda f: bar.progress(f, text="Running models..."))
            bar.empty()
            st.session_state["batch_compare"] = (cfg, cmp_df)
        saved = st.session_state.get("batch_compare")
        if saved and saved[0] == cfg:
            st.dataframe(saved[1], hide_index=True)
            st.caption("Counts of tweets each model labels Sarcastic. Models disagree a lot because every one of "
                       "them is weak on this task; a higher count does not mean a better model.")


def render():
    paths = ui.get_paths()
    st.title("Batch Detector")
    ui.smoke_test_banner(paths)
    predictor = ui.get_predictor()
    models = predictor.available_models()
    if not models:
        ui.pending_notice()
        return
    default = predictor.default_model_name()

    st.write("Upload a CSV (or a .txt with one tweet per line) and classify every tweet with a saved model. "
             "Nothing is retrained and nothing is sent over the network.")
    st.caption(f"Limits: at most {BATCH_MAX_ROWS:,} rows and {BATCH_MAX_MB:g} MB. Only the tweet column you choose "
               "is used as model input.")

    c1, c2 = st.columns(2)
    default_label = f"Default (selected on validation data: {default})" if default else "Default"
    choice = c1.selectbox("Model", [default_label] + models)
    thr_label = c2.radio("Decision threshold", ["Validation-selected", "Model default"], horizontal=True)
    model_name = None if choice == default_label else choice
    thr_type = "validation_selected" if thr_label == "Validation-selected" else "default"

    up = st.file_uploader("Upload tweets", type=["csv", "txt"])
    has_header = st.checkbox("The first row is a header (uncheck if your CSV has no header row)", value=True)
    if up is None:
        st.info("No file yet. You can try `demo/sample_tweets.csv` (made-up demo text, not dataset data).")
        return

    raw = up.getvalue()
    try:
        parsed = bt.read_upload(raw, up.name, has_header=has_header)
    except bt.BatchError as e:
        st.error(str(e))
        return
    for n in parsed.notes:
        st.info(n)
    df = parsed.df
    st.caption(f"Read {len(df):,} rows and {df.shape[1]} column(s) (encoding: {parsed.encoding}).")
    st.write("Preview (first 5 rows):")
    st.dataframe(df.head(5), hide_index=True)

    cols = list(df.columns)
    tweet_col = st.selectbox("Column that contains the tweets", cols, index=cols.index(bt.detect_text_column(df)))

    label_col = None
    other = [c for c in cols if c != tweet_col]
    if other:
        guess = bt.detect_label_column(df, tweet_col)
        if st.checkbox("My file also has a label column (0 = Not Sarcastic, 1 = Sarcastic) - show an ad-hoc check",
                       value=False, key=f"use_labels_{guess}"):
            label_col = st.selectbox("Label column", other, index=other.index(guess) if guess in other else 0)
            st.caption("The label column is NOT used as model input; it is only compared with the predictions "
                       "afterwards.")

    cfg = {"file": hashlib.md5(raw).hexdigest(), "header": has_header, "tweet_col": tweet_col,
           "model_name": model_name or default, "threshold_mode": thr_type, "label_col": label_col}

    if st.button("Classify all tweets", type="primary"):
        bar = st.progress(0.0, text="Classifying tweets...")
        try:
            rows = bt.run_batch(predictor, df, tweet_col, model_name, thr_type,
                                progress=lambda f: bar.progress(f, text=f"Classifying tweets... {f:.0%}"))
        except bt.BatchError as e:
            bar.empty()
            st.error(str(e))
            return
        except FileNotFoundError as e:
            bar.empty()
            st.error(str(e))
            return
        except Exception as e:  # keep the page usable; show a message, not a stack trace
            bar.empty()
            st.error(f"Could not classify this file ({type(e).__name__}). Check the model files and the "
                     "selected column, then try again.")
            return
        bar.empty()
        if not (rows.status == bt.STATUS_OK).any():
            st.warning("No tweet in this column could be scored (all rows were blank or had no readable words). "
                       "Please check that you selected the right column.")
            st.session_state.pop("batch_result", None)
            return
        cfg["model_display"] = rows.attrs["model_display"]
        cfg["score_meaning"] = rows.attrs["score_meaning"]
        st.session_state["batch_result"] = {"cfg": cfg, "df": df,
                                            "res": {"rows": rows, "summary": bt.summarize(rows)}}

    saved = st.session_state.get("batch_result")
    if saved:
        scfg = saved["cfg"]
        stale = {k: scfg[k] for k in ("file", "header", "tweet_col", "model_name", "threshold_mode", "label_col")} != \
                {k: cfg[k] for k in ("file", "header", "tweet_col", "model_name", "threshold_mode", "label_col")}
        if stale:
            st.warning("The file or settings changed since the last run. The results below are from the previous "
                       "run; click **Classify all tweets** to update them.")
        _show_results(saved["res"], saved["df"], scfg, predictor, paths)
