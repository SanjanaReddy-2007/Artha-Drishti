import streamlit as st

from demo import ui
from src import results_io as rio


def render():
    paths = ui.get_paths()
    st.markdown('<div class="ad-hero"><h1>Artha-Drishti</h1>'
                '<p>Machine Learning-Based Sarcasm Detection in English Tweets</p></div>', unsafe_allow_html=True)
    ui.smoke_test_banner(paths)
    st.write("")
    st.subheader("Why sarcasm matters")
    st.write("Sarcasm says one thing and means another, so a model that only reads the literal words can get the "
             "sentiment of a tweet exactly backwards. Detecting it is a hard, open problem in natural language "
             "processing, and it matters for opinion mining, moderation and social-media analysis.")
    st.success("**Try it:** open **Sarcasm Detector** in the sidebar and analyse a tweet.")

    st.subheader("Dataset and experiment")
    rep = rio.load_data_report(paths)
    c1, c2, c3 = st.columns(3)
    if rep:
        c1.metric("Tweets used", f"{rep['n_rows_final']:,}")
        c2.metric("Sarcastic", f"{rep['n_sarcastic']:,}")
        c3.metric("Not sarcastic", f"{rep['n_not_sarcastic']:,}")
        st.caption("Counts come from the saved data report after cleaning and de-duplication.")
    else:
        st.info("Dataset statistics will appear here after `python -m src.prepare_data` has been run.")
    done = rio.completed_models(paths)
    st.write(f"**Models trained and evaluated:** {len(done)}" + (f" ({', '.join(done)})" if done else ""))

    st.subheader("Methodology in brief")
    st.markdown(
        "1. **Data:** iSarcasmEval English training tweets, binary label (sarcastic / not sarcastic).\n"
        "2. **Preprocessing:** URLs, mentions and hashtags normalised; emojis converted to words; lowercase.\n"
        "3. **Features:** TF-IDF word n-grams, learned from training data only.\n"
        "4. **Models:** SVM, Logistic Regression, Naive Bayes, Decision Tree, Random Forest, KNN, "
        "Gradient Boosting / XGBoost and an MLP, under identical data conditions.\n"
        "5. **Evaluation:** sarcastic-class F1, precision, recall, PR-AUC; validation-selected thresholds; "
        "bootstrap confidence intervals; comparison with trivial baselines.")
    st.warning("**Disclaimer:** predictions are experimental and may be wrong. Sarcasm depends on context, tone "
               "and shared knowledge that a text-only model cannot see.")
