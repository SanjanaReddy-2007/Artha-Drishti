"""Evaluation utilities. Positive class = 1 = 'Sarcastic'.

Two decision rules are reported separately:
  * default threshold  : 0.5 for probabilities, 0 for SVM decision scores (strict '>', as predict())
  * validation-selected: threshold that maximises sarcastic-class F1 on VALIDATION data ('>=')
Thresholds are never chosen on test labels.
"""
from __future__ import annotations

import time

import numpy as np
from sklearn.metrics import (accuracy_score, average_precision_score, classification_report, confusion_matrix,
                             f1_score, precision_recall_curve, precision_score, recall_score, roc_auc_score)

from src.config import CLASS_NAMES


def decide(scores, thr: float, strict: bool) -> np.ndarray:
    scores = np.asarray(scores)
    return (scores > thr if strict else scores >= thr).astype(int)


def compute_metrics(y, pred, scores) -> dict:
    y, pred = np.asarray(y), np.asarray(pred)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    both = len(np.unique(y)) == 2
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, pos_label=1, zero_division=0)),
        "recall": float(recall_score(y, pred, pos_label=1, zero_division=0)),
        "f1": float(f1_score(y, pred, pos_label=1, zero_division=0)),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y, pred, average="weighted", zero_division=0)),
        "pr_auc": float(average_precision_score(y, scores)),
        "roc_auc": float(roc_auc_score(y, scores)) if both else float("nan"),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def report_dict(y, pred) -> dict:
    return classification_report(y, pred, labels=[0, 1], target_names=CLASS_NAMES, output_dict=True, zero_division=0)


def select_threshold(y, scores) -> tuple[float, float]:
    """Validation-only: threshold maximising sarcastic-class F1 (rule: score >= threshold)."""
    p, r, t = precision_recall_curve(y, scores)
    if len(t) == 0:
        return float(np.min(scores)), float("nan")
    f1 = 2 * p[:-1] * r[:-1] / np.clip(p[:-1] + r[:-1], 1e-12, None)
    i = int(np.argmax(f1))
    return float(t[i]), float(f1[i])


def threshold_table(y, scores, max_points: int = 120) -> list[dict]:
    p, r, t = precision_recall_curve(y, scores)
    f1 = 2 * p[:-1] * r[:-1] / np.clip(p[:-1] + r[:-1], 1e-12, None)
    idx = np.arange(len(t))
    if len(idx) > max_points:
        idx = np.unique(np.linspace(0, len(t) - 1, max_points).astype(int))
    return [{"threshold": float(t[i]), "precision": float(p[i]), "recall": float(r[i]), "f1": float(f1[i])} for i in idx]


def baselines(y) -> dict:
    """Always-'Sarcastic' and majority-class ('Not Sarcastic') baselines on the given labels."""
    y = np.asarray(y)
    const = np.zeros(len(y))
    return {"all_sarcastic": compute_metrics(y, np.ones(len(y), dtype=int), const),
            "majority_class": compute_metrics(y, np.zeros(len(y), dtype=int), const),
            "prevalence_sarcastic": float(y.mean())}


# ---- bootstrap -------------------------------------------------------------------------------
def _batch_prf(y, pred, idx):
    yy, pp = y[idx], pred[idx]
    tp = (yy * pp).sum(1).astype(float); fp = ((1 - yy) * pp).sum(1).astype(float)
    fn = (yy * (1 - pp)).sum(1).astype(float); tn = ((1 - yy) * (1 - pp)).sum(1).astype(float)
    div = lambda a, b: np.divide(a, b, out=np.zeros_like(a), where=b > 0)
    prec, rec = div(tp, tp + fp), div(tp, tp + fn)
    f1, f1_neg = div(2 * tp, 2 * tp + fp + fn), div(2 * tn, 2 * tn + fn + fp)
    return {"precision": prec, "recall": rec, "f1": f1, "macro_f1": (f1 + f1_neg) / 2}


def _boot_idx(n, n_boot, seed):
    return np.random.default_rng(seed).integers(0, n, size=(n_boot, n))


def _prauc_batch(y, scores, idx):
    out = np.empty(len(idx))
    for i, ix in enumerate(idx):
        out[i] = average_precision_score(y[ix], scores[ix]) if y[ix].any() else np.nan
    return out


def _ci(a):
    a = a[~np.isnan(a)]
    return {"mean": float(a.mean()), "low": float(np.percentile(a, 2.5)), "high": float(np.percentile(a, 97.5))}


def bootstrap_ci(y, pred, scores, n_boot=1000, seed=42) -> dict:
    """Percentile 95% CIs (resampling test examples with replacement)."""
    y, pred, scores = np.asarray(y), np.asarray(pred), np.asarray(scores)
    idx = _boot_idx(len(y), n_boot, seed)
    b = _batch_prf(y, pred, idx)
    out = {k: _ci(v) for k, v in b.items()}
    out["pr_auc"] = _ci(_prauc_batch(y, scores, idx))
    out["n_boot"] = n_boot
    return out


def paired_bootstrap(y, pred_a, scores_a, pred_b, scores_b, n_boot=1000, seed=42) -> dict:
    """A minus B on the SAME resampled test examples. If the CI contains 0 the difference is not conclusive."""
    y = np.asarray(y)
    idx = _boot_idx(len(y), n_boot, seed)
    fa, fb = _batch_prf(y, np.asarray(pred_a), idx)["f1"], _batch_prf(y, np.asarray(pred_b), idx)["f1"]
    pa, pb = _prauc_batch(y, np.asarray(scores_a), idx), _prauc_batch(y, np.asarray(scores_b), idx)
    d_f1, d_pr = _ci(fa - fb), _ci(pa - pb)
    d_f1["excludes_zero"] = bool(d_f1["low"] > 0 or d_f1["high"] < 0)
    d_pr["excludes_zero"] = bool(d_pr["low"] > 0 or d_pr["high"] < 0)
    return {"f1_diff": d_f1, "pr_auc_diff": d_pr, "n_boot": n_boot}


def time_per_tweet_ms(fn, texts, repeats: int = 3) -> float:
    """Best-of-N wall time of fn(texts) in milliseconds per tweet (includes vectorisation)."""
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter(); fn(texts); best = min(best, time.perf_counter() - t0)
    return 1000.0 * best / max(len(texts), 1)
