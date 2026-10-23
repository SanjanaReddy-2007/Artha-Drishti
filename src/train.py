"""Train, tune, evaluate and save every candidate model (topic-independent).

Protocol
  1. Fit/tune on TRAIN only (5-fold stratified CV, scoring = F1 of the sarcastic class).
  2. VALIDATION: score, choose the sarcastic-F1 threshold, and select the default model
     (highest validation PR-AUC; models within 0.01 are tied and the simplest wins).
  3. TEST: evaluated once per model with thresholds already fixed on validation. Test labels are
     never used for tuning, thresholding or model selection.
Failures are recorded in results/status.json instead of being dropped.

Run:  python -m src.train [--models SVM,LogReg] [--features word|word_char] [--n-jobs 1]
"""
from __future__ import annotations

import argparse
import json
import logging
import platform
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline

from src import evaluate as ev
from src.config import SEED, SELECTION_METRIC, SIMPLICITY_ORDER, TIE_TOLERANCE, ProjectPaths
from src.features import FEATURE_CONFIGS, build_vectorizer
from src.models import ModelSpec, get_registry, get_scores, score_info
from src.preprocess import EMOJI_AVAILABLE
from src.split import load_split, verify_split

log = logging.getLogger("artha.train")


def _py(o):
    """Make numpy/pandas values JSON-serialisable."""
    if isinstance(o, dict):
        return {str(k): _py(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_py(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        return None if np.isnan(o) else float(o)
    if isinstance(o, np.ndarray):
        return _py(o.tolist())
    return o


def _git_commit(root):
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=root, capture_output=True,
                              text=True, timeout=5).stdout.strip() or None
    except Exception:
        return None


def _write_json(path, obj):
    path.write_text(json.dumps(_py(obj), indent=1), encoding="utf-8")


def select_default(val_results: dict) -> dict:
    """Pick the default model from VALIDATION results only."""
    best = max(v[SELECTION_METRIC] for v in val_results.values())
    tied = [n for n, v in val_results.items() if v[SELECTION_METRIC] >= best - TIE_TOLERANCE]
    rank = lambda n: SIMPLICITY_ORDER.index(n) if n in SIMPLICITY_ORDER else len(SIMPLICITY_ORDER)
    choice = sorted(tied, key=rank)[0]
    return {"name": choice, "selection_metric": f"validation {SELECTION_METRIC}", "tie_tolerance": TIE_TOLERANCE,
            "best_value": best, "chosen_value": val_results[choice][SELECTION_METRIC],
            "tied_candidates": tied, "rule": "highest validation PR-AUC; models within tolerance are tied; simplest wins"}


def _fit_one(spec: ModelSpec, feats: str, cv_folds: int, n_jobs: int, tr, va, paths: ProjectPaths):
    pipe = Pipeline([("features", build_vectorizer(feats)), ("clf", spec.builder(n_jobs))])
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=SEED)
    gs = GridSearchCV(pipe, spec.grid, scoring="f1", cv=cv, n_jobs=n_jobs, refit=True)
    t0 = time.perf_counter()
    gs.fit(tr.text_clean, tr.label.values)
    tune_time = time.perf_counter() - t0
    best = gs.best_estimator_
    score_type, thr_default = score_info(best.named_steps["clf"])

    s_val = get_scores(best, va.text_clean)
    thr_val, f1_at_thr = ev.select_threshold(va.label.values, s_val)
    thresholds = {"default": {"value": thr_default, "strict": True},
                  "validation_selected": {"value": thr_val, "strict": False}}
    val = {}
    for key, th in thresholds.items():
        pred = ev.decide(s_val, th["value"], th["strict"])
        val[key] = ev.compute_metrics(va.label.values, pred, s_val)
        val[f"{key}_classification_report"] = ev.report_dict(va.label.values, pred)
    val["threshold_table"] = ev.threshold_table(va.label.values, s_val)
    infer_ms = ev.time_per_tweet_ms(lambda x: get_scores(best, x), list(va.text_clean))

    joblib.dump(best, paths.models / f"{spec.name}.joblib", compress=3)
    pd.DataFrame({"row_id": va.row_id, "y_true": va.label, "score": s_val}).to_csv(
        paths.predictions / f"{spec.name}_validation.csv", index=False)

    clf = best.named_steps["clf"]
    detail = {
        "name": spec.name, "display": spec.display, "family": spec.family, "experiment": spec.experiment,
        "implementation": spec.implementation, "notes": spec.notes, "status": "completed",
        "score_type": score_type, "features": feats, "feature_description": FEATURE_CONFIGS[feats],
        "grid": {k: [str(x) for x in v] for k, v in spec.grid.items()},
        "best_params": {k: str(v) for k, v in gs.best_params_.items()},
        "cv_folds": cv_folds, "cv_best_f1": float(gs.best_score_), "thresholds": thresholds,
        "timing": {"tuning_time_s": tune_time, "final_fit_time_s": float(getattr(gs, "refit_time_", float("nan"))),
                   "inference_ms_per_tweet_validation": infer_ms},
        "validation": val, "artifact": f"models/{spec.name}.joblib",
        "loss_curve": list(getattr(clf, "loss_curve_", [])) or None,
        "n_features": int(best.named_steps["features"].transform(["probe"]).shape[1]),
    }
    return best, detail, s_val


def run(paths: ProjectPaths, model_names=None, features="word", cv_folds=5, n_jobs=1, n_boot=1000,
        limit=None) -> dict:
    paths.ensure()
    df = pd.read_csv(paths.processed_file)
    tr, va, te = load_split(paths, df)
    split_summary = verify_split(df, json.loads(paths.split_file.read_text(encoding="utf-8")))
    if limit:
        tr = tr.sample(min(limit, len(tr)), random_state=SEED).reset_index(drop=True)
        log.warning("SMOKE TEST: using only %d training rows. Results are NOT a valid experiment.", len(tr))
    log.info("train %d | val %d | test %d | sarcastic share %.3f/%.3f/%.3f", len(tr), len(va), len(te),
             tr.label.mean(), va.label.mean(), te.label.mean())

    registry = get_registry()
    names = model_names or list(registry)
    unknown = [n for n in names if n not in registry]
    if unknown:
        raise SystemExit(f"Unknown models {unknown}. Available: {list(registry)}")

    status, fitted, details, val_scores = {}, {}, {}, {}
    for name in names:
        spec = registry[name]
        if not spec.available:
            status[name] = {"status": "skipped", "reason": spec.unavailable_reason}
            log.warning("%s skipped: %s", name, spec.unavailable_reason)
            continue
        log.info("=== %s ===", name)
        try:
            best, detail, s_val = _fit_one(spec, features, cv_folds, n_jobs, tr, va, paths)
            fitted[name], details[name], val_scores[name] = best, detail, s_val
            status[name] = {"status": "completed", "reason": ""}
            log.info("%s: CV F1 %.3f | val F1 (default) %.3f | val PR-AUC %.3f | tune %.1fs", name,
                     detail["cv_best_f1"], detail["validation"]["default"]["f1"],
                     detail["validation"]["default"]["pr_auc"], detail["timing"]["tuning_time_s"])
        except Exception as exc:  # recorded, not hidden
            status[name] = {"status": "failed", "reason": f"{type(exc).__name__}: {exc}",
                            "traceback": traceback.format_exc()}
            log.error("%s FAILED: %s", name, exc)

    status["GloVe_BiGRU"] = {"status": "pending", "reason": "Future phase: not implemented in this baseline."}
    _write_json(paths.status_json, status)
    if not fitted:
        raise SystemExit("No model completed. See results/status.json and the log.")

    # ---- default model: VALIDATION data only -------------------------------------------------
    default = select_default({n: d["validation"]["default"] for n, d in details.items()})
    default["threshold_default"] = details[default["name"]]["thresholds"]["default"]
    default["threshold_validation_selected"] = details[default["name"]]["thresholds"]["validation_selected"]
    _write_json(paths.default_model_json, default)
    log.info("Default model (validation-selected): %s", default["name"])

    # ---- TEST: final evaluation, thresholds already fixed ------------------------------------
    s_test, pred_test = {}, {}
    for name, pipe in fitted.items():
        d = details[name]
        s = get_scores(pipe, te.text_clean)
        s_test[name] = s
        pd.DataFrame({"row_id": te.row_id, "y_true": te.label, "score": s}).to_csv(
            paths.predictions / f"{name}_test.csv", index=False)
        d["timing"]["inference_ms_per_tweet_test"] = ev.time_per_tweet_ms(lambda x: get_scores(pipe, x), list(te.text_clean))
        d["test"] = {}
        pred_test[name] = {}
        for key, th in d["thresholds"].items():
            p = ev.decide(s, th["value"], th["strict"])
            pred_test[name][key] = p
            d["test"][key] = ev.compute_metrics(te.label.values, p, s)
            d["test"][f"{key}_classification_report"] = ev.report_dict(te.label.values, p)
            d["test"][f"{key}_bootstrap_ci"] = ev.bootstrap_ci(te.label.values, p, s, n_boot, SEED)
    ref = default["name"]
    for name in fitted:
        if name == ref:
            continue
        for key in ("default", "validation_selected"):
            details[name]["test"][f"{key}_paired_vs_{ref}"] = ev.paired_bootstrap(
                te.label.values, pred_test[name][key], s_test[name], pred_test[ref][key], s_test[ref], n_boot, SEED)

    # ---- flat tables ---------------------------------------------------------------------------
    rows = []
    for name, d in details.items():
        for stage in ("validation", "test"):
            for key in ("default", "validation_selected"):
                m = d[stage][key]
                ci = d[stage].get(f"{key}_bootstrap_ci", {})
                rows.append({"model": name, "display": d["display"], "experiment": d["experiment"],
                             "stage": stage, "threshold_type": key, "threshold": d["thresholds"][key]["value"],
                             **{k: m[k] for k in ("accuracy", "precision", "recall", "f1", "macro_f1", "weighted_f1",
                                                  "pr_auc", "roc_auc", "tn", "fp", "fn", "tp")},
                             "f1_ci_low": ci.get("f1", {}).get("low"), "f1_ci_high": ci.get("f1", {}).get("high"),
                             "pr_auc_ci_low": ci.get("pr_auc", {}).get("low"), "pr_auc_ci_high": ci.get("pr_auc", {}).get("high"),
                             "cv_best_f1": d["cv_best_f1"], "tuning_time_s": d["timing"]["tuning_time_s"],
                             "final_fit_time_s": d["timing"]["final_fit_time_s"],
                             "inference_ms_per_tweet": d["timing"][f"inference_ms_per_tweet_{'validation' if stage == 'validation' else 'test'}"],
                             "status": "completed"})
        _write_json(paths.model_results / f"{name}.json", d)
    pd.DataFrame(rows).to_csv(paths.metrics_csv, index=False)

    brow = []
    for stage, y in (("validation", va.label.values), ("test", te.label.values)):
        b = ev.baselines(y)
        for k in ("all_sarcastic", "majority_class"):
            brow.append({"baseline": k, "stage": stage, **{m: b[k][m] for m in ("accuracy", "precision", "recall", "f1", "macro_f1", "pr_auc")},
                         "prevalence_sarcastic": b["prevalence_sarcastic"]})
    pd.DataFrame(brow).to_csv(paths.baselines_csv, index=False)

    meta = {"timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "python": sys.version.split()[0],
            "platform": platform.platform(), "scikit_learn": sklearn.__version__, "seed": SEED,
            "features": features, "feature_description": FEATURE_CONFIGS[features], "cv_folds": cv_folds,
            "n_bootstrap": n_boot, "emoji_conversion_available": EMOJI_AVAILABLE, "git_commit": _git_commit(paths.root),
            "smoke_test": bool(limit), "train_rows_used": int(len(tr)), "split_summary": split_summary,
            "models_requested": names, "default_model": default["name"],
            "selection_rule": default["rule"]}
    _write_json(paths.run_metadata, meta)
    log.info("Done. Results in %s", paths.results)
    return {"status": status, "default": default, "meta": meta}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", default="", help="comma-separated names (default: all)")
    ap.add_argument("--features", default="word", choices=list(FEATURE_CONFIGS))
    ap.add_argument("--cv-folds", type=int, default=5)
    ap.add_argument("--n-jobs", type=int, default=1)
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--limit", type=int, default=0, help="SMOKE TEST ONLY: use this many training rows")
    a = ap.parse_args(argv)
    paths = ProjectPaths().ensure()
    logfile = paths.logs / f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.StreamHandler(), logging.FileHandler(logfile, encoding="utf-8")])
    names = [m.strip() for m in a.models.split(",") if m.strip()] or None
    run(paths, names, a.features, a.cv_folds, a.n_jobs, a.bootstrap, a.limit or None)


if __name__ == "__main__":
    main()
