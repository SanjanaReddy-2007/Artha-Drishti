import altair as alt
import pandas as pd
import streamlit as st

from demo import ui
from src import results_io as rio
from src.config import CLASS_NAMES

GENERAL = {  # general, well-known properties of each algorithm (not results)
    "SVM": "Strong on sparse high-dimensional text. Produces a raw margin, not a probability. Sensitive to class imbalance.",
    "SVM_balanced": "Class-weighted variant: usually trades precision for recall on the minority (sarcastic) class.",
    "LogReg": "Simple, fast, interpretable coefficients, gives probabilities. Linear decision boundary only.",
    "LogReg_balanced": "Class-weighted variant: higher recall for the minority class, typically lower precision.",
    "NaiveBayes": "Very fast and robust on small data. Assumes feature independence; probabilities are often over-confident.",
    "DecisionTree": "Interpretable rules, but unstable and prone to over-fitting on sparse high-dimensional TF-IDF.",
    "RandomForest": "Reduces tree variance; slower and larger; often weak on sparse text unless features are informative.",
    "KNN": "No training step, but needs all training data at prediction time; cosine distance on sparse text is noisy.",
    "GradientBoosting": "Flexible and often strong on tabular data; slower to train and can over-fit small text datasets.",
    "XGBoost": "Fast, regularised boosting; needs tuning and can over-fit small text datasets.",
    "MLP": "Can learn non-linear feature interactions; needs more data and care to avoid over-fitting TF-IDF inputs.",
}


def _interpret(d, base, stage="test", key="validation_selected"):
    if base is None or stage not in d:
        return None
    b = base[(base.stage == stage) & (base.baseline == "all_sarcastic")]
    if b.empty:
        return None
    f1, f1b = d[stage][key]["f1"], float(b.f1.iloc[0])
    rel = "above" if f1 > f1b else "below"
    return (f"On the {stage} set at the {key.replace('_', ' ')} threshold, sarcastic-class F1 is **{f1:.3f}**, "
            f"{rel} the always-sarcastic baseline (**{f1b:.3f}**). PR-AUC is **{d[stage][key]['pr_auc']:.3f}** "
            f"against a random-scorer level of {float(b.prevalence_sarcastic.iloc[0]):.3f}.")


def render():
    paths = ui.get_paths()
    st.title("Model Performance")
    ui.smoke_test_banner(paths)
    done = rio.completed_models(paths)
    if not done:
        ui.pending_notice()
        return
    name = st.selectbox("Model", done)
    d = rio.load_model_detail(paths, name)
    if d is None:
        st.error("Result file for this model is missing.")
        return
    base = rio.load_baselines(paths)
    c1, c2 = st.columns(2)
    stage = c1.radio("Stage", ["validation", "test"], horizontal=True)
    key = c2.radio("Threshold", ["default", "validation_selected"], horizontal=True,
                   format_func=lambda s: s.replace("_", " "))

    t1, t2, t3, t4, t5 = st.tabs(["Configuration", "Metrics", "Curves", "Threshold analysis", "Interpretation"])
    with t1:
        st.markdown(f"**{d['display']}**  \nImplementation: `{d['implementation']}`  \nFeatures: {d['feature_description']}")
        st.write(f"Score type: `{d['score_type']}`")
        st.write("Hyper-parameters selected by 5-fold CV on the training set (scoring: sarcastic F1):")
        st.json(d["best_params"])
        st.write("Search grid:"); st.json(d["grid"])
        st.write(f"Best CV F1: **{d['cv_best_f1']:.3f}** | Thresholds: "
                 f"default = {d['thresholds']['default']['value']:.3f}, validation-selected = "
                 f"{d['thresholds']['validation_selected']['value']:.3f}")
        tm = d["timing"]
        st.write(f"Tuning time {tm['tuning_time_s']:.1f}s | final fit {tm['final_fit_time_s']:.2f}s | "
                 f"inference {tm.get('inference_ms_per_tweet_test', float('nan')):.2f} ms/tweet (test)")
        if d.get("notes"):
            st.caption(d["notes"])
    with t2:
        m = d[stage][key]
        cols = st.columns(5)
        for c, k, lbl in zip(cols, ("f1", "precision", "recall", "pr_auc", "accuracy"),
                             ("Sarcastic F1", "Precision", "Recall", "PR-AUC", "Accuracy")):
            c.metric(lbl, f"{m[k]:.3f}")
        st.write(f"Macro F1 {m['macro_f1']:.3f} | Weighted F1 {m['weighted_f1']:.3f} | ROC-AUC {m['roc_auc']:.3f}")
        cm = rio.confusion_df(d, stage, key)
        mid = cm.Count.max() / 2
        h = alt.Chart(cm).encode(x=alt.X("Predicted:N", sort=CLASS_NAMES), y=alt.Y("Actual:N", sort=CLASS_NAMES))
        rect = h.mark_rect().encode(color=alt.Color("Count:Q", scale=alt.Scale(scheme="blues"), legend=None))
        txt = h.mark_text(fontSize=22).encode(text="Count:Q", color=alt.condition(alt.datum.Count > mid, alt.value("white"), alt.value("black")))
        st.altair_chart((rect + txt).properties(width=320, height=260))
        rep = d[stage][f"{key}_classification_report"]
        st.write("Classification report")
        st.dataframe(pd.DataFrame(rep).T.round(3))
        ci = d["test"].get(f"{key}_bootstrap_ci") if "test" in d else None
        if stage == "test" and ci:
            st.write(f"Bootstrap 95% confidence intervals (test set, {ci['n_boot']} resamples):")
            st.dataframe(pd.DataFrame({k: v for k, v in ci.items() if isinstance(v, dict)}).T.round(3))
            pv = {k: v for k, v in d["test"].items() if k.startswith(f"{key}_paired_vs_")}
            for k, v in pv.items():
                ref = k.split("_paired_vs_")[1]
                st.caption(f"Paired difference vs **{ref}** (same test examples): F1 diff {v['f1_diff']['mean']:+.3f} "
                           f"[{v['f1_diff']['low']:+.3f}, {v['f1_diff']['high']:+.3f}] - "
                           + ("interval excludes 0." if v['f1_diff']['excludes_zero'] else "interval includes 0, so not conclusive."))
        elif stage == "validation":
            st.caption("Bootstrap intervals are computed for the final test stage only.")
    with t3:
        pr = rio.pr_curve(paths, name, stage)
        if pr is not None:
            st.altair_chart(alt.Chart(pr).mark_line().encode(x=alt.X("recall:Q", scale=alt.Scale(domain=[0, 1])),
                                                              y=alt.Y("precision:Q", scale=alt.Scale(domain=[0, 1]))).properties(height=340))
        if d.get("loss_curve"):
            lc = pd.DataFrame({"epoch": range(1, len(d["loss_curve"]) + 1), "training loss": d["loss_curve"]})
            st.write("Training history (MLP training loss per epoch)")
            st.altair_chart(alt.Chart(lc).mark_line(point=True).encode(x="epoch:Q", y="training loss:Q").properties(height=240))
        else:
            st.caption("This model type does not produce a training history, so no learning curve is shown.")
    with t4:
        tt = pd.DataFrame(d["validation"]["threshold_table"])
        long = tt.melt(id_vars="threshold", var_name="metric", value_name="value")
        chart = alt.Chart(long).mark_line().encode(x="threshold:Q", y=alt.Y("value:Q", scale=alt.Scale(domain=[0, 1])), color="metric:N")
        sel = pd.DataFrame({"x": [d["thresholds"]["validation_selected"]["value"]]})
        st.altair_chart((chart + alt.Chart(sel).mark_rule(strokeDash=[5, 4], color="crimson").encode(x="x:Q")).properties(height=320))
        st.caption("Computed on VALIDATION data only. The dashed line is the threshold that maximises sarcastic-class F1 on "
                   "validation. With about 350 validation tweets this choice is noisy.")
    with t5:
        st.write(GENERAL.get(name, ""))
        txt = _interpret(d, base)
        if txt:
            st.markdown("**From the saved results:** " + txt)
        st.caption("Interpretation of general strengths/weaknesses is background knowledge about the algorithm, not a "
                   "finding of this experiment. Limitations: single dataset, small test set, text-only features.")
