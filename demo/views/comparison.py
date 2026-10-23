import altair as alt
import pandas as pd
import streamlit as st

from demo import ui
from src import results_io as rio

COLS = {"model": "Model", "f1": "Sarcastic F1", "precision": "Precision", "recall": "Recall", "pr_auc": "PR-AUC",
        "accuracy": "Accuracy", "macro_f1": "Macro F1", "roc_auc": "ROC-AUC", "final_fit_time_s": "Fit time (s)",
        "tuning_time_s": "Tuning time (s)", "inference_ms_per_tweet": "Inference (ms/tweet)"}


def _bar(df, value, title, order, rule=None, err=None):
    base = alt.Chart(df).encode(x=alt.X("model:N", sort=order, title=None, axis=alt.Axis(labelAngle=-40, labelOverlap=False, labelLimit=200)))
    layers = [base.mark_bar().encode(y=alt.Y(f"{value}:Q", title=title), color=alt.Color("experiment:N", title="Group"),
                                      tooltip=["model", alt.Tooltip(f"{value}:Q", format=".3f"), "experiment"])]
    if err:
        layers.append(base.mark_rule(color="black").encode(y=alt.Y(f"{err[0]}:Q", title=title), y2=f"{err[1]}:Q"))
    if rule is not None:
        layers.append(alt.Chart(pd.DataFrame({"y": [rule[0]]})).mark_rule(strokeDash=[5, 4], color="crimson")
                      .encode(y=alt.Y("y:Q", title=title)))
    return alt.layer(*layers).properties(height=300, title=title)


def render():
    paths = ui.get_paths()
    st.title("Model Comparison")
    ui.smoke_test_banner(paths)
    metrics = rio.load_metrics(paths)
    if metrics is None:
        ui.pending_notice()
        return
    base = rio.load_baselines(paths)
    status = rio.load_status(paths)
    default = rio.load_default_model(paths)

    c1, c2 = st.columns(2)
    stage = c1.radio("Evaluation stage", ["validation", "test"], horizontal=True,
                     format_func=lambda s: "Validation (used for tuning/selection)" if s == "validation" else "Final test (reported once)")
    thr = c2.radio("Decision threshold", ["default", "validation_selected"], horizontal=True,
                   format_func=lambda s: "Default (0.5 / 0)" if s == "default" else "Validation-selected")
    st.caption(f"Showing **{stage}** results at the **{thr.replace('_', ' ')}** threshold. "
               "Validation-selected thresholds are chosen on validation data only, never on test labels.")
    df = metrics[(metrics.stage == stage) & (metrics.threshold_type == thr)].copy()
    b = base[base.stage == stage].set_index("baseline") if base is not None else None
    f1_all = float(b.loc["all_sarcastic", "f1"]) if b is not None else None
    if default:
        st.info(f"Default model (chosen on validation PR-AUC; models within {default['tie_tolerance']} count as "
                f"tied and the simplest wins): **{default['name']}**")

    st.subheader("Results table (click a header to sort)")
    show = df[list(COLS)].rename(columns=COLS)
    st.dataframe(show.round(3), hide_index=True)
    if b is not None:
        st.caption("Baselines on this stage: " + "; ".join(
            f"{k.replace('_', ' ')} F1 = {b.loc[k, 'f1']:.3f}" for k in b.index) +
            f". Sarcastic share = {b.loc['all_sarcastic', 'prevalence_sarcastic']:.3f} (PR-AUC of a random scorer).")
    if stage == "test":
        st.caption("Bootstrap 95% CIs for test F1 are drawn as black lines. Overlapping intervals mean the models "
                   "cannot be separated on this test set; do not read small differences as superiority.")

    st.subheader("Model status")
    st.dataframe(pd.DataFrame([{"Model": k, "Status": v["status"], "Note": v.get("reason", "")}
                               for k, v in status.items()]), hide_index=True)

    order = df.sort_values("f1", ascending=False).model.tolist()
    err = ("f1_ci_low", "f1_ci_high") if stage == "test" and df.f1_ci_low.notna().all() else None
    st.subheader("Sarcastic-class F1")
    st.altair_chart(_bar(df, "f1", "Sarcastic-class F1", order, (f1_all,) if f1_all is not None else None, err))
    if f1_all is not None:
        st.caption("Dashed red line = F1 of the always-'Sarcastic' baseline. A model below it is not useful at this threshold.")

    st.subheader("Precision and recall")
    long = df.melt(id_vars=["model"], value_vars=["precision", "recall"], var_name="metric", value_name="value")
    st.altair_chart(alt.Chart(long).mark_bar().encode(
        x=alt.X("model:N", sort=order, title=None, axis=alt.Axis(labelAngle=-40, labelOverlap=False, labelLimit=200)), y=alt.Y("value:Q", title="Score"),
        color="metric:N", xOffset="metric:N", tooltip=["model", "metric", alt.Tooltip("value:Q", format=".3f")]).properties(height=300))

    st.subheader("PR-AUC")
    st.altair_chart(_bar(df, "pr_auc", "PR-AUC (average precision)", df.sort_values("pr_auc", ascending=False).model.tolist(),
                         (float(b.loc["all_sarcastic", "prevalence_sarcastic"]),) if b is not None else None))
    st.caption("Dashed line = sarcastic share, the PR-AUC of random scores.")

    st.subheader("Training and inference time")
    tdf = df.melt(id_vars=["model"], value_vars=["tuning_time_s", "final_fit_time_s"], var_name="kind", value_name="seconds")
    tdf["kind"] = tdf.kind.map({"tuning_time_s": "Hyper-parameter tuning (5-fold CV grid)", "final_fit_time_s": "Final fit"})
    st.altair_chart(alt.Chart(tdf).mark_bar().encode(
        x=alt.X("model:N", title=None, axis=alt.Axis(labelAngle=-40, labelOverlap=False, labelLimit=200)), y=alt.Y("seconds:Q", title="Seconds"),
        color="kind:N", xOffset="kind:N", tooltip=["model", "kind", alt.Tooltip("seconds:Q", format=".2f")]).properties(height=300))
    st.caption("Inference time per tweet (ms, including TF-IDF) is in the table above.")

    st.subheader("Confusion matrix")
    sel = st.selectbox("Model", df.model.tolist(), key="cm_model")
    detail = rio.load_model_detail(paths, sel)
    if detail:
        cm = rio.confusion_df(detail, stage, thr)
        mid = cm.Count.max() / 2
        h = alt.Chart(cm).encode(x=alt.X("Predicted:N", sort=["Not Sarcastic", "Sarcastic"]),
                                 y=alt.Y("Actual:N", sort=["Not Sarcastic", "Sarcastic"]))
        rect = h.mark_rect().encode(color=alt.Color("Count:Q", scale=alt.Scale(scheme="blues"), legend=None))
        txt = h.mark_text(fontSize=22).encode(text="Count:Q", color=alt.condition(alt.datum.Count > mid, alt.value("white"), alt.value("black")))
        st.altair_chart((rect + txt).properties(width=320, height=260))

    st.subheader("Precision-recall curves")
    curves = [c for c in (rio.pr_curve(paths, m, stage) for m in df.model) if c is not None]
    if curves:
        cdf = pd.concat(curves)
        lines = alt.Chart(cdf).mark_line().encode(x=alt.X("recall:Q", scale=alt.Scale(domain=[0, 1])),
                                                  y=alt.Y("precision:Q", scale=alt.Scale(domain=[0, 1])),
                                                  color="model:N", tooltip=["model"]).properties(height=380)
        st.altair_chart(lines)
        st.caption("Curves are computed from the saved scores and true labels of each model.")
    else:
        st.info("Prediction files for the curves are not available.")
