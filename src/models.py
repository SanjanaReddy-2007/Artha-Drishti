"""Model registry: one place that defines every candidate, its hyper-parameter grid and score type.

Maximum Entropy: a (multinomial) logistic-regression classifier IS a maximum-entropy classifier.
It is therefore NOT implemented a second time; 'LogReg' covers it. Listing it separately would
count one algorithm twice.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

from src.config import SEED


@dataclass
class ModelSpec:
    name: str
    display: str
    family: str
    implementation: str
    builder: Callable[[int], object]          # n_jobs -> estimator
    grid: dict
    experiment: str = "main"                  # 'main' or 'class_weight'
    notes: str = ""
    available: bool = True
    unavailable_reason: str = ""


def _specs() -> list[ModelSpec]:
    s = [
        ModelSpec("SVM", "Support Vector Machine (linear)", "linear", "sklearn.svm.LinearSVC",
                  lambda n: LinearSVC(random_state=SEED, max_iter=5000), {"clf__C": [0.1, 0.3, 1, 3, 10]},
                  notes="Linear kernel; raw decision score, not a probability."),
        ModelSpec("SVM_balanced", "SVM (class_weight=balanced)", "linear", "sklearn.svm.LinearSVC",
                  lambda n: LinearSVC(random_state=SEED, max_iter=5000, class_weight="balanced"),
                  {"clf__C": [0.1, 0.3, 1, 3, 10]}, experiment="class_weight",
                  notes="Class-imbalance experiment, reported separately from the unweighted SVM."),
        ModelSpec("LogReg", "Logistic Regression (= Maximum Entropy)", "linear", "sklearn.linear_model.LogisticRegression",
                  lambda n: LogisticRegression(max_iter=2000, random_state=SEED), {"clf__C": [0.1, 1, 3, 10, 30]},
                  notes="Also the maximum-entropy classifier; not implemented twice."),
        ModelSpec("LogReg_balanced", "Logistic Regression (class_weight=balanced)", "linear",
                  "sklearn.linear_model.LogisticRegression",
                  lambda n: LogisticRegression(max_iter=2000, random_state=SEED, class_weight="balanced"),
                  {"clf__C": [0.1, 1, 3, 10, 30]}, experiment="class_weight",
                  notes="Class-imbalance experiment, reported separately."),
        ModelSpec("NaiveBayes", "Multinomial Naive Bayes", "probabilistic", "sklearn.naive_bayes.MultinomialNB",
                  lambda n: MultinomialNB(), {"clf__alpha": [0.03, 0.1, 0.3, 1.0]}),
        ModelSpec("DecisionTree", "Decision Tree", "tree", "sklearn.tree.DecisionTreeClassifier",
                  lambda n: DecisionTreeClassifier(random_state=SEED),
                  {"clf__max_depth": [5, 10, 20, None], "clf__min_samples_leaf": [1, 3, 5]}),
        ModelSpec("RandomForest", "Random Forest", "tree ensemble", "sklearn.ensemble.RandomForestClassifier",
                  lambda n: RandomForestClassifier(random_state=SEED, n_jobs=n),
                  {"clf__n_estimators": [200], "clf__max_depth": [None, 40]}),
        ModelSpec("KNN", "K-Nearest Neighbours (cosine)", "instance based", "sklearn.neighbors.KNeighborsClassifier",
                  lambda n: KNeighborsClassifier(metric="cosine", algorithm="brute"),
                  {"clf__n_neighbors": [3, 5, 9, 15]}),
        ModelSpec("GradientBoosting", "Gradient Boosting (scikit-learn)", "tree ensemble",
                  "sklearn.ensemble.GradientBoostingClassifier",
                  lambda n: GradientBoostingClassifier(random_state=SEED),
                  {"clf__n_estimators": [100, 200], "clf__max_depth": [2, 3]}),
        ModelSpec("MLP", "Multilayer Perceptron", "neural", "sklearn.neural_network.MLPClassifier",
                  lambda n: MLPClassifier(early_stopping=True, max_iter=200, random_state=SEED),
                  {"clf__hidden_layer_sizes": [(64,), (128,)], "clf__alpha": [1e-4, 1e-2]},
                  notes="Single hidden layer on TF-IDF; early stopping uses a split of the training data only."),
    ]
    try:
        from xgboost import XGBClassifier
        s.append(ModelSpec("XGBoost", "XGBoost", "tree ensemble", "xgboost.XGBClassifier",
                           lambda n: XGBClassifier(eval_metric="logloss", tree_method="hist",
                                                   random_state=SEED, n_jobs=n),
                           {"clf__n_estimators": [200, 400], "clf__max_depth": [3, 6], "clf__learning_rate": [0.05, 0.1]}))
    except Exception as e:  # not installed -> recorded as skipped, NOT silently replaced
        s.append(ModelSpec("XGBoost", "XGBoost", "tree ensemble", "xgboost.XGBClassifier", lambda n: None, {},
                           available=False, unavailable_reason=f"xgboost not importable ({type(e).__name__}); "
                           "pip install xgboost to enable. scikit-learn GradientBoosting is reported separately."))
    return s


def get_registry() -> dict[str, ModelSpec]:
    return {m.name: m for m in _specs()}


def score_info(estimator) -> tuple[str, float]:
    """(score_type, default_threshold) for a fitted/unfitted estimator."""
    if hasattr(estimator, "predict_proba"):
        return "probability", 0.5
    return "decision_function", 0.0


def get_scores(pipeline, texts):
    """Continuous sarcasm score for each text. Probability if supported, else decision score."""
    clf = pipeline.named_steps["clf"]
    if hasattr(clf, "predict_proba"):
        return pipeline.predict_proba(texts)[:, 1]
    return pipeline.decision_function(texts)


SCORE_LABELS = {
    "probability": "Model probability of 'Sarcastic' (uncalibrated)",
    "decision_function": "Decision score (distance from the SVM boundary; not a probability)",
}
