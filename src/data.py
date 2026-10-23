"""Load, validate, clean and de-duplicate the iSarcasmEval English training file."""
from __future__ import annotations

import json
import logging

import pandas as pd

from src.config import LABEL_COLUMNS, TEXT_COLUMNS, ProjectPaths
from src.preprocess import EMOJI_AVAILABLE, clean_text

log = logging.getLogger("artha.data")


def _find(df: pd.DataFrame, candidates, kind: str) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(f"No {kind} column found. Tried {list(candidates)}; file has {list(df.columns)}")


def load_raw(path) -> tuple[pd.DataFrame, dict]:
    """Return (frame with orig_row/text_raw/label, report). Only text + label are kept:
    other columns (e.g. 'rephrase', sarcasm-type flags) exist only for sarcastic tweets or are
    derived from the label, so using them would be target leakage."""
    df = pd.read_csv(path, encoding="utf-8-sig")
    text_col, label_col = _find(df, TEXT_COLUMNS, "text"), _find(df, LABEL_COLUMNS, "label")
    report = {"raw_file": str(path), "n_rows_raw": int(len(df)), "columns_in_file": list(df.columns),
              "text_column": text_col, "label_column": label_col,
              "columns_ignored_to_prevent_leakage": [c for c in df.columns if c not in (text_col, label_col)]}
    out = pd.DataFrame({"orig_row": range(len(df)), "text_raw": df[text_col], "label": df[label_col]})
    blank = out.text_raw.isna() | (out.text_raw.astype(str).str.strip() == "")
    report["n_missing_or_blank_text"] = int(blank.sum())
    out = out[~blank].copy()
    out["label"] = pd.to_numeric(out.label, errors="coerce")
    bad = ~out.label.isin([0, 1])
    report["n_invalid_label"] = int(bad.sum())
    out = out[~bad].copy()
    out["label"] = out.label.astype(int)
    out["text_raw"] = out.text_raw.astype(str)
    return out.reset_index(drop=True), report


def build_processed(paths: ProjectPaths, convert_emoji: bool = True) -> dict:
    """Clean text, drop empty/duplicate texts, write data/processed.csv and results/data_report.json."""
    paths.ensure()
    df, report = load_raw(paths.raw_file)
    report["raw_file"] = paths.raw_file.relative_to(paths.root).as_posix()   # relative, not machine-specific
    report["n_rows_after_validation"] = int(len(df))
    df["text_clean"] = [clean_text(t, convert_emoji) for t in df.text_raw]
    empty = df.text_clean == ""
    report["n_empty_after_cleaning"] = int(empty.sum())
    df = df[~empty]

    # Duplicate cleaned texts: drop groups with conflicting labels entirely, keep first otherwise.
    g = df.groupby("text_clean").label.nunique()
    conflicting = set(g[g > 1].index)
    report["n_duplicate_groups_conflicting_labels_dropped"] = len(conflicting)
    df = df[~df.text_clean.isin(conflicting)]
    n_before = len(df)
    df = df.drop_duplicates("text_clean", keep="first")
    report["n_duplicate_texts_dropped"] = int(n_before - len(df))

    df = df.reset_index(drop=True)
    df.insert(0, "row_id", range(len(df)))
    df.to_csv(paths.processed_file, index=False, encoding="utf-8")
    report.update({
        "n_rows_final": int(len(df)),
        "n_sarcastic": int((df.label == 1).sum()),
        "n_not_sarcastic": int((df.label == 0).sum()),
        "sarcastic_share": round(float((df.label == 1).mean()), 4),
        "emoji_conversion": bool(convert_emoji and EMOJI_AVAILABLE),
        "processed_file": paths.processed_file.relative_to(paths.root).as_posix(),
    })
    paths.data_report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    log.info("Processed data: %s", {k: report[k] for k in ("n_rows_raw", "n_rows_final", "n_sarcastic", "n_not_sarcastic")})
    return report
