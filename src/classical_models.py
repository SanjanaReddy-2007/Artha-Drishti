"""Step 4 (M9-M11) - TF-IDF + five classical classifiers.

SVM, Logistic Regression, Naive Bayes, Random Forest, KNN.
Tuning: 5-fold stratified CV on the TRAIN split, scoring = F1 of sarcastic class.
Final model refit on train, predictions saved for val and test.
Classes are NOT balanced by default (use --balanced to switch class_weight on).
"""
import argparse
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score
from config import SPLIT_CSV, RESULTS, MODELS, SEED


def build(balanced):
    cw = "balanced" if balanced else None

    def tfidf():
        return TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True,
                               token_pattern=r"(?u)\b\w+\b|[!?]+")
    return {
        "NaiveBayes": (Pipeline([("tfidf", tfidf()), ("clf", MultinomialNB())]),
                       {"clf__alpha": [0.1, 0.3, 1.0]}),
        "LogReg": (Pipeline([("tfidf", tfidf()),
                             ("clf", LogisticRegression(max_iter=2000, class_weight=cw, random_state=SEED))]),
                   {"clf__C": [0.3, 1, 3, 10]}),
        "SVM": (Pipeline([("tfidf", tfidf()),
                          ("clf", LinearSVC(class_weight=cw, random_state=SEED))]),
                {"clf__C": [0.1, 0.3, 1, 3]}),
        "KNN": (Pipeline([("tfidf", tfidf()),
                          ("clf", KNeighborsClassifier(metric="cosine", algorithm="brute"))]),
                {"clf__n_neighbors": [5, 11, 21], "clf__weights": ["uniform", "distance"]}),
        "RandomForest": (Pipeline([("tfidf", tfidf()),
                                   ("clf", RandomForestClassifier(n_estimators=300, class_weight=cw,
                                                                  n_jobs=-1, random_state=SEED))]),
                         {"clf__max_depth": [None, 30], "clf__min_samples_leaf": [1, 2]}),
    }


def main(balanced, text_col):
    df = pd.read_csv(SPLIT_CSV)
    df[text_col] = df[text_col].fillna("")
    tr = df[df["split"] == "train"]
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    summary = []
    for name, (pipe, grid) in build(balanced).items():
        gs = GridSearchCV(pipe, grid, scoring="f1", cv=cv, n_jobs=-1)
        gs.fit(tr[text_col], tr["label"])
        best = gs.best_estimator_
        for split in ("val", "test"):
            part = df[df["split"] == split]
            pred = best.predict(part[text_col])
            out = part[["id", "label", "topic"]].assign(pred=pred)
            out.to_csv(RESULTS / f"preds_{name}_{split}.csv", index=False)
        vf1 = f1_score(df[df.split == "val"]["label"], best.predict(df[df.split == "val"][text_col]))
        joblib.dump(best, MODELS / f"{name}.joblib")
        pd.Series(gs.best_params_).to_json(RESULTS / f"best_params_{name}.json")
        print(f"{name:13s} cv_f1={gs.best_score_:.3f}  val_f1={vf1:.3f}  params={gs.best_params_}")
        summary.append({"model": name, "cv_f1": gs.best_score_, "val_f1": vf1,
                        "params": str(gs.best_params_)})
    pd.DataFrame(summary).to_csv(RESULTS / "classical_tuning_summary.csv", index=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--balanced", action="store_true", help="use class_weight='balanced'")
    ap.add_argument("--text_col", default="text_clean")
    a = ap.parse_args()
    main(a.balanced, a.text_col)
