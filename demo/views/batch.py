import hashlib
from dataclasses import replace

import altair as alt
import pandas as pd
import streamlit as st

from demo import ui
from src import batch as bt
from src import batch_multi as bm
from src.config import BATCH_MAX_FILES, BATCH_MAX_MB, BATCH_MAX_ROWS, BATCH_MAX_TOTAL_ROWS

COLORS = {"Sarcastic": "#d97706", "Not Sarcastic": "#0f9d8a"}
_SCALE = alt.Scale(domain=list(COLORS), range=list(COLORS.values()))
STALE_KEYS = ("files", "header", "tweet_cols", "label_cols", "model_name", "threshold_mode")


@st.cache_data(show_spinner=False, max_entries=64)
def _load(raw: bytes, name: str, has_header: bool):
    """Parse one uploaded file (cached, so changing a setting does not re-read every file)."""
    return bm.load_file(raw, name, has_header=has_header)


def _donut(s):
    d = pd.DataFrame({"Label": ["Sarcastic", "Not Sarcastic"], "Tweets": [s["sarcastic"], s["not_sarcastic"]]})
    return (alt.Chart(d).mark_arc(innerRadius=60)
            .encode(theta="Tweets:Q", color=alt.Color("Label:N", scale=_SCALE, title=None),
                    tooltip=["Label", "Tweets"])
            .properties(width=320, height=240))


def _files_chart(runs):
    d = pd.DataFrame([{"File": r.name, "Label": lab, "Tweets": r.summary[key]}
                      for r in runs if r.summary is not None
                      for lab, key in (("Sarcastic", "sarcastic"), ("Not Sarcastic", "not_sarcastic"))])
    return (alt.Chart(d).mark_bar()
            .encode(y=alt.Y("File:N", title=None, sort=None), x=alt.X("Tweets:Q", title="Scored tweets"),
                    color=alt.Color("Label:N", scale=_SCALE, title=None), tooltip=["File", "Label", "Tweets"])
            .properties(width=420, height=max(80, 32 * d.File.nunique())))


def _show_results(saved, predictor):
    cfg, runs, rows, s, out = saved["cfg"], saved["runs"], saved["combined"], saved["summary"], saved["outputs"]
    done = [r for r in runs if r.rows is not None]
    failed = [r for r in runs if r.rows is None]
    multi = len(done) > 1

    st.divider()
    st.subheader("Results")
    st.caption(f"Model: **{cfg['model_display']}** | threshold: **{cfg['threshold_mode'].replace('_', ' ')}** "
               f"| {len(done)} file(s) classified | {s['total_rows']:,} rows in total")
    for r in failed:
        st.error(f"**{r.name}** was not classified. {r.error}")

    c = st.columns(5)
    c[0].metric("Total tweets", f"{s['total_rows']:,}")
    c[1].metric("Sarcastic", f"{s['sarcastic']:,}", f"{s['sarcastic_pct']:.1f}% of scored", delta_color="off")
    c[2].metric("Not Sarcastic", f"{s['not_sarcastic']:,}", f"{s['not_sarcastic_pct']:.1f}% of scored",
                delta_color="off")
    c[3].metric("Scored", f"{s['scored']:,}")
    c[4].metric("Skipped", f"{s['skipped']:,}")
    st.caption("Percentages are shares of the tweets that were scored (skipped rows excluded). "
               f"Sarcastic + Not Sarcastic + skipped = {s['total_rows']:,} total rows"
               + (" (this holds for every file too)." if multi else "."))
    if s["scored"]:
        if multi:
            g1, g2 = st.columns(2)
            g1.altair_chart(_donut(s))
            g2.altair_chart(_files_chart(done))
        else:
            st.altair_chart(_donut(s))
    if multi:
        st.subheader("Summary per file")
        st.dataframe(out["summary_df"].drop(columns=["model", "threshold_mode"]), hide_index=True)
    if s["duplicate_rows"]:
        st.caption(f"{s['duplicate_rows']:,} duplicate tweet(s) kept in the output; each unique text was scored "
                   "once" + (f" (across all files; {bm.cross_file_duplicates(runs):,} distinct tweet(s) appear in "
                             "more than one file)." if multi else "."))
    if s["skipped"]:
        with st.expander(f"Skipped rows ({s['skipped']:,})"):
            sk = rows[rows.status != bt.STATUS_OK][["source_file", "row_number", "tweet", "status"]]
            sk = sk.rename(columns={"source_file": "File", "row_number": "Row", "tweet": "Tweet",
                                    "status": "Reason"})
            st.dataframe(sk if multi else sk.drop(columns=["File"]), hide_index=True)
    st.info("**Caution:** these are model predictions, and individual predictions can be wrong. On the held-out "
            "test set the default model reaches only about 0.37 sarcastic-class F1 (wide confidence intervals; see "
            "Model Performance). Scores are **uncalibrated**: "
            f"{cfg['score_meaning'][0].lower() + cfg['score_meaning'][1:]}.")

    # ---- table with filter / search / sort (all files together)
    st.subheader("Classification of every tweet")
    f1, f2, f3 = st.columns([1.2, 1.6, 1.2])
    flt = f1.radio("Show", ["All", "Sarcastic", "Not Sarcastic"], horizontal=True)
    q = f2.text_input("Search tweet text", placeholder="type part of a tweet...")
    srt = f3.selectbox("Sort by", ["Original order", "Score: high to low", "Score: low to high"])
    t = rows[rows.status == bt.STATUS_OK]
    if multi:
        pick = st.multiselect("Files", [r.name for r in done], default=[r.name for r in done])
        t = t[t.source_file.isin(pick)]
    if flt != "All":
        t = t[t.predicted_label == flt]
    if q.strip():
        t = t[t.tweet.str.contains(q.strip(), case=False, regex=False)]
    if srt != "Original order":
        t = t.sort_values("score", ascending=srt.endswith("low to high"), kind="stable")
    cols = ["source_file", "row_number", "tweet", "predicted_label", "score", "threshold"]
    show = t[cols].rename(columns={"source_file": "File", "row_number": "Row", "tweet": "Tweet",
                                   "predicted_label": "Predicted label", "score": "Score",
                                   "threshold": "Threshold used"})
    if not multi:
        show = show.drop(columns=["File"])
    st.caption(f"Showing {len(show):,} of {s['scored']:,} scored tweets.")
    st.dataframe(show, hide_index=True, column_config={
        "Score": st.column_config.NumberColumn(format="%.3f"),
        "Threshold used": st.column_config.NumberColumn(format="%.3f")})
    with st.expander("Text as seen by the model (after preprocessing) - first 5 scored tweets"):
        for tw in rows[rows.status == bt.STATUS_OK].tweet.head(5):
            st.code(bt.clean_text(tw))

    # ---- downloads
    st.subheader("Downloads")
    d = st.columns(3 if multi else 2)
    if multi:
        d[0].download_button("Combined results, all files (CSV)", out["combined"], "sarcasm_batch_combined.csv",
                             "text/csv", type="primary")
        d[1].download_button("Summary per file (CSV)", out["summary"], "sarcasm_batch_summary.csv", "text/csv")
        d[2].download_button("Everything (ZIP, one results file per file)", out["zip"],
                             "sarcasm_batch_results.zip", "application/zip")
        with st.expander("Download the results of one file (your original columns + predictions)"):
            for i, r in enumerate(done):
                st.download_button(f"{r.name} (CSV)", out["files"][r.name], f"{r.name.rsplit('.', 1)[0]}_results.csv",
                                   "text/csv", key=f"dl_file_{i}")
    else:
        r = done[0]
        d[0].download_button("Download full results (CSV)", out["files"][r.name], "sarcasm_batch_results.csv",
                             "text/csv", type="primary")
        d[1].download_button("Download summary (CSV)", out["summary"], "sarcasm_batch_summary.csv", "text/csv")
    st.caption("CSV files are UTF-8 with a byte-order mark so Excel shows special characters correctly.")

    # ---- optional: labelled check, per file that has a label column
    for r in done:
        if not r.label_col:
            continue
        try:
            lc = bt.labelled_check(r.df, r.rows, r.label_col)
        except bt.BatchError as e:
            st.warning(f"{r.name}: {e}")
            continue
        st.subheader("Ad-hoc check against your own labels" + (f" - {r.name}" if multi else ""))
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

    # ---- optional: compare models (all files together)
    with st.expander("Compare models (runs all trained models on these files)", expanded=False):
        models = predictor.available_models()
        st.caption(f"Scores every unique valid tweet with each of the {len(models)} trained models at the selected "
                   "threshold mode. This can take a while for large uploads, so it only runs when you click.")
        if len(rows[rows.status == bt.STATUS_OK].tweet.unique()) > 2000:
            st.warning("Large upload: this may take a minute or more.")
        if st.button("Run all models"):
            bar = st.progress(0.0, text="Running models...")
            cmp_df = bt.compare_models(predictor, rows, models, cfg["threshold_mode"],
                                       progress=lambda f: bar.progress(f, text="Running models..."))
            bar.empty()
            st.session_state["batch_compare"] = (cfg, cmp_df)
        saved_cmp = st.session_state.get("batch_compare")
        if saved_cmp and saved_cmp[0] == cfg:
            st.dataframe(saved_cmp[1], hide_index=True)
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

    st.write("Upload one or more CSV files (or .txt files with one tweet per line) and classify every tweet with "
             "a saved model. Nothing is retrained and nothing is sent over the network.")
    st.caption(f"Limits: each file at most {BATCH_MAX_ROWS:,} rows and {BATCH_MAX_MB:g} MB; at most "
               f"{BATCH_MAX_FILES} files and {BATCH_MAX_TOTAL_ROWS:,} rows in total per run. Only the tweet column "
               "you choose in each file is used as model input.")

    c1, c2 = st.columns(2)
    default_label = f"Default (selected on validation data: {default})" if default else "Default"
    choice = c1.selectbox("Model", [default_label] + models)
    thr_label = c2.radio("Decision threshold", ["Validation-selected", "Model default"], horizontal=True)
    model_name = None if choice == default_label else choice
    thr_type = "validation_selected" if thr_label == "Validation-selected" else "default"

    ups = st.file_uploader("Upload tweet files (you can select several at once)", type=["csv", "txt"],
                           accept_multiple_files=True)
    has_header = st.checkbox("The first row is a header (uncheck if your CSV files have no header row)", value=True)
    if not ups:
        st.info("No file yet. You can try `demo/sample_tweets.csv` (made-up demo text, not dataset data).")
        return

    raws = [u.getvalue() for u in ups]
    names = bm.unique_names([u.name for u in ups])
    loaded = [replace(_load(raw, u.name, has_header), name=n) for raw, u, n in zip(raws, ups, names)]
    loaded = bm.apply_limits(loaded)
    digests = [hashlib.md5(raw).hexdigest() for raw in raws]

    # ---- overview of every file
    ov = [{"File": l.name,
           "Rows": len(l.parsed.df) if l.ok else None,
           "Columns": l.parsed.df.shape[1] if l.ok else None,
           "Encoding": l.parsed.encoding if l.ok else "",
           "Status": ("ready" + (f" ({len(l.parsed.notes)} note(s), see below)" if l.parsed.notes else ""))
           if l.ok else l.error} for l in loaded]
    st.subheader(f"Files ({len(loaded)})")
    st.dataframe(pd.DataFrame(ov), hide_index=True)
    ready = [(i, l) for i, l in enumerate(loaded) if l.ok]
    if not ready:
        st.error("None of the uploaded files could be read. See the Status column above for the reason.")
        return
    total_rows = sum(len(l.parsed.df) for _, l in ready)
    st.caption(f"{len(ready)} file(s) ready, {total_rows:,} rows in total.")

    # ---- per-file settings: tweet column (auto-detected) and optional label column
    st.subheader("Choose the tweet column for each file")
    tweet_cols, label_cols = [], []
    for i, l in ready:
        df = l.parsed.df
        cols = list(df.columns)
        auto = bt.detect_text_column(df)
        key = f"{i}_{digests[i][:8]}_{int(has_header)}"
        with st.expander(f"{l.name} - {len(df):,} rows, tweet column: {auto}", expanded=len(ready) == 1):
            for n in l.parsed.notes:
                st.info(n)
            st.caption(f"Read {len(df):,} rows and {df.shape[1]} column(s) (encoding: {l.parsed.encoding}).")
            st.write("Preview (first 5 rows):")
            st.dataframe(df.head(5), hide_index=True)
            tc = st.selectbox("Column that contains the tweets", cols, index=cols.index(auto), key=f"tc_{key}")
            lc = None
            other = [c for c in cols if c != tc]
            if other:
                guess = bt.detect_label_column(df, tc)
                if st.checkbox("This file also has a label column (0 = Not Sarcastic, 1 = Sarcastic) - show an "
                               "ad-hoc check", value=False, key=f"use_labels_{key}_{guess}"):
                    lc = st.selectbox("Label column", other, index=other.index(guess) if guess in other else 0,
                                      key=f"lc_{key}")
                    st.caption("The label column is NOT used as model input; it is only compared with the "
                               "predictions afterwards.")
        tweet_cols.append(tc)
        label_cols.append(lc)

    cfg = {"files": [(l.name, digests[i]) for i, l in ready], "header": has_header,
           "tweet_cols": tweet_cols, "label_cols": label_cols,
           "model_name": model_name or default, "threshold_mode": thr_type}

    label = "Classify all tweets" if len(ready) == 1 else f"Classify all tweets in {len(ready)} files"
    if st.button(label, type="primary"):
        bar = st.progress(0.0, text="Classifying tweets...")
        try:
            rows_list = bm.run_batch_multi(predictor, [(l.parsed.df, tc) for (_, l), tc in zip(ready, tweet_cols)],
                                           model_name, thr_type,
                                           progress=lambda f: bar.progress(f, text=f"Classifying tweets... {f:.0%}"))
        except (bt.BatchError, FileNotFoundError) as e:
            bar.empty()
            st.error(str(e))
            return
        except Exception as e:  # keep the page usable; show a message, not a stack trace
            bar.empty()
            st.error(f"Could not classify these files ({type(e).__name__}). Check the model files and the "
                     "selected columns, then try again.")
            return
        bar.empty()
        runs, k = [], 0
        for l in loaded:
            if l.ok:
                rows = rows_list[k]
                runs.append(bm.FileRun(l.name, l.parsed.df, tweet_cols[k], rows, bt.summarize(rows),
                                       label_col=label_cols[k]))
                k += 1
            else:
                runs.append(bm.FileRun(l.name, error=l.error))
        combined = bm.combine_runs(runs)
        if not (combined.status == bt.STATUS_OK).any():
            st.warning("No tweet in the selected columns could be scored (all rows were blank or had no readable "
                       "words). Please check that you selected the right column for each file.")
            st.session_state.pop("batch_multi_result", None)
            return
        cfg["model_display"] = rows_list[0].attrs["model_display"]
        cfg["score_meaning"] = rows_list[0].attrs["score_meaning"]
        st.session_state["batch_multi_result"] = {
            "cfg": cfg, "runs": runs, "combined": combined, "summary": bt.summarize(combined),
            "outputs": bm.build_outputs(runs, cfg["model_name"], thr_type)}

    saved = st.session_state.get("batch_multi_result")
    if saved:
        if {k: saved["cfg"][k] for k in STALE_KEYS} != {k: cfg[k] for k in STALE_KEYS}:
            st.warning("The files or settings changed since the last run. The results below are from the previous "
                       "run; click **" + label + "** to update them.")
        _show_results(saved, predictor)
