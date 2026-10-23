import streamlit as st

from demo import ui
from src import results_io as rio


def render():
    paths = ui.get_paths()
    st.title("About the Project")
    st.info("This site presents a **text-only, topic-independent baseline**. Topic-aware modelling is a planned later extension.")
    st.markdown("""
### Pages of this site
Home; Sarcasm Detector (one tweet); Batch Detector (upload a CSV or .txt of tweets, classify all of them, download the
results); Model Comparison; Model Performance; About the Project. Both detector pages use the same saved models,
preprocessing and thresholds.

### Problem statement
Given an English tweet, decide whether it is **sarcastic** or **not sarcastic** (binary classification; sarcastic = positive class).

### Objectives
* Build a fair, reproducible comparison of classical machine-learning classifiers on the same data and split.
* Evaluate with metrics that suit an imbalanced problem, and compare against trivial baselines.
* Provide a working web application backed by the saved, evaluated models.

### Dataset and its limitations
iSarcasmEval (SemEval-2022 Task 6) English training tweets. Labels were provided by the tweet authors, which is
valuable but means a small, single-source, English-only dataset with a minority of sarcastic examples. Results may not transfer to other platforms,
time periods or languages. Only the tweet text and label are used; other columns that exist only for sarcastic tweets
(such as the author's rephrasing) are excluded to prevent target leakage.

### Preprocessing
HTML entities decoded; URLs -> `url`; mentions -> `@user`; hashtags keep their word; emojis converted to words;
lowercased; punctuation and repeated letters kept (possible sarcasm cues). Empty and duplicate texts removed
(duplicates with conflicting labels dropped).

### Feature extraction
TF-IDF on word 1-2-grams (an optional word + character n-gram configuration is available). The vocabulary is learned
from training data only.

### Algorithms compared
Linear SVM, Logistic Regression, Multinomial Naive Bayes, Decision Tree, Random Forest, K-Nearest Neighbours,
Gradient Boosting (and XGBoost when installed) and a Multilayer Perceptron. Class-weighted SVM and Logistic Regression
are reported as a separate class-imbalance experiment. *Maximum Entropy* is mathematically logistic regression, so it is not
counted as a separate algorithm. A GloVe + BiGRU model is a pending future phase.

### Experimental methodology
Stratified 80/10/10 split (fixed seed, indices saved). Hyper-parameters tuned by 5-fold cross-validation on the training
set. Validation data selects the decision threshold and the default model. The test set is used once for final reporting.

### Evaluation metrics
Sarcastic-class precision, recall and F1; accuracy; macro/weighted F1; PR-AUC; ROC-AUC; confusion matrices; training and
inference time; bootstrap 95% confidence intervals and paired bootstrap comparisons. Baselines: always-sarcastic and majority-class.

### Limitations and future work
Small dataset and small test set (wide confidence intervals); text-only; no calibration analysis yet; no context or
conversation features; the literature review that would support any novelty claim is still to be completed; planned: GloVe + BiGRU,
probability calibration, robustness tests and, later, a topic-aware extension.
""")
    meta = rio.load_run_metadata(paths)
    if meta:
        st.subheader("Reproducibility record of the saved run")
        st.json({k: meta[k] for k in ("timestamp_utc", "python", "scikit_learn", "seed", "features", "cv_folds",
                                       "n_bootstrap", "git_commit", "default_model", "smoke_test") if k in meta})
    st.caption("Project owner: Are Sanjana Reddy (24BDS0187).")
