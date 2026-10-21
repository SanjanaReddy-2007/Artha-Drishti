"""Read saved results for the dashboard. No Streamlit imports here, so it is unit-testable.
Every function returns None / empty when the file does not exist (the UI then shows 'pending')."""
from __future__ import annotations

import json

import pandas as pd
from sklearn.metrics import precision_recall_curve

from src.config import ProjectPaths


def _json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
    except json.JSONDecodeError:
        return None


def load_status(p: ProjectPaths): return _json(p.status_json) or {}
def load_run_metadata(p: ProjectPaths): return _json(p.run_metadata)
def load_data_report(p: ProjectPaths): return _json(p.data_report)
def load_default_model(p: ProjectPaths): return _json(p.default_model_json)
def load_model_detail(p: ProjectPaths, name: str): return _json(p.model_results / f"{name}.json")


def load_metrics(p: ProjectPaths):
    return pd.read_csv(p.metrics_csv) if p.metrics_csv.exists() else None


def load_baselines(p: ProjectPaths):
    return pd.read_csv(p.baselines_csv) if p.baselines_csv.exists() else None


def completed_models(p: ProjectPaths) -> list[str]:
    st = load_status(p)
    return [n for n, v in st.items() if v.get("status") == "completed" and (p.models / f"{n}.joblib").exists()]


def load_predictions(p: ProjectPaths, name: str, stage: str):
    f = p.predictions / f"{name}_{stage}.csv"
    return pd.read_csv(f) if f.exists() else None


def pr_curve(p: ProjectPaths, name: str, stage: str):
    """Precision-recall points computed from the SAVED scores and true labels."""
    d = load_predictions(p, name, stage)
    if d is None:
        return None
    prec, rec, _ = precision_recall_curve(d.y_true, d.score)
    return pd.DataFrame({"recall": rec, "precision": prec, "model": name, "stage": stage})


def confusion_df(detail: dict, stage: str, key: str) -> pd.DataFrame:
    m = detail[stage][key]
    return pd.DataFrame([
        {"Actual": "Not Sarcastic", "Predicted": "Not Sarcastic", "Count": m["tn"]},
        {"Actual": "Not Sarcastic", "Predicted": "Sarcastic", "Count": m["fp"]},
        {"Actual": "Sarcastic", "Predicted": "Not Sarcastic", "Count": m["fn"]},
        {"Actual": "Sarcastic", "Predicted": "Sarcastic", "Count": m["tp"]},
    ])
