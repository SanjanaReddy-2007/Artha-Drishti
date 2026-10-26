"""Multi-file Batch Detector and multi-tweet text box logic. No Streamlit imports, so it is unit-testable.

This module sits ON TOP of src/batch.py and does not change it: every file still goes through
read_upload(), prepare_rows() and the same scoring path as before, so a single file behaves exactly as
it did. What is new:

* several files in one run, with per-file and overall summaries (count identity holds for both);
* every unique text is scored ONCE across all files, then mapped back to every row;
* a failed file is reported and skipped; the other files still run;
* output: one combined CSV, one summary CSV, and a ZIP with one results CSV per file;
* split_tweets(): one tweet per line for the Sarcasm Detector text box.
"""
from __future__ import annotations

import io
import re
import zipfile
from collections import Counter
from dataclasses import dataclass
from pathlib import PurePath

import pandas as pd

from src.batch import (PRED_COLS, SKIP_REASONS, STATUS_OK, BatchError, ParsedFile, build_output,
                       prepare_rows, read_upload, summarize, to_csv_bytes)
from src.config import (BATCH_MAX_FILES, BATCH_MAX_MB, BATCH_MAX_ROWS, BATCH_MAX_TOTAL_ROWS,
                        MULTI_TWEET_MAX)

COMBINED_COLS = ["source_file", "row_number", "tweet"] + PRED_COLS
SUMMARY_COLS = ["file", "status", "total_rows", "rows_scored", "sarcastic", "sarcastic_percent_of_scored",
                "not_sarcastic", "not_sarcastic_percent_of_scored", "skipped_rows", "model",
                "threshold_mode"]
ALL_FILES = "ALL FILES"


# ---- loading several files ---------------------------------------------------------------------
@dataclass
class LoadedFile:
    """One uploaded file after parsing: either `parsed` is set, or `error` explains why not."""
    name: str
    parsed: ParsedFile | None = None
    error: str = ""

    @property
    def ok(self) -> bool:
        return self.parsed is not None and not self.error


@dataclass
class FileRun:
    """One file after classification (or a file that could not be processed, with `error` set)."""
    name: str
    df: pd.DataFrame | None = None
    tweet_col: str = ""
    rows: pd.DataFrame | None = None
    summary: dict | None = None
    error: str = ""
    label_col: str | None = None


def unique_names(names) -> list[str]:
    """Make file names unique ('a.csv', 'a (2).csv', ...) so two uploads with the same name stay apart."""
    used, out = set(), []
    for n in names:
        p, cand, k = PurePath(str(n)), str(n), 1
        while cand in used:
            k += 1
            cand = f"{p.stem} ({k}){p.suffix}"
        used.add(cand)
        out.append(cand)
    return out


def load_file(raw: bytes, filename: str, has_header: bool = True, max_rows: int = BATCH_MAX_ROWS,
              max_mb: float = BATCH_MAX_MB) -> LoadedFile:
    """read_upload() that returns the problem instead of raising it, so one bad file cannot stop the rest."""
    try:
        return LoadedFile(filename, read_upload(raw, filename, has_header, max_rows, max_mb))
    except BatchError as e:
        return LoadedFile(filename, None, str(e))


def apply_limits(files: list[LoadedFile], max_files: int = BATCH_MAX_FILES,
                 max_total_rows: int = BATCH_MAX_TOTAL_ROWS) -> list[LoadedFile]:
    """Enforce the limits across all files, in upload order. A file that would break a limit is replaced
    by an error entry ('not processed') and never silently dropped."""
    out, total = [], 0
    for i, f in enumerate(files):
        if i >= max_files:
            out.append(LoadedFile(f.name, None, f"Not processed: at most {max_files} files can be classified "
                                                "in one run."))
        elif not f.ok:
            out.append(f)
        else:
            n = len(f.parsed.df)
            if total + n > max_total_rows:
                out.append(LoadedFile(f.name, None,
                                      f"Not processed: adding this file ({n:,} rows) would exceed the limit of "
                                      f"{max_total_rows:,} rows across all files (already {total:,}). "
                                      "Classify it in a separate run."))
            else:
                total += n
                out.append(f)
    return out


# ---- classifying several files at once ---------------------------------------------------------
def run_batch_multi(predictor, jobs, model_name: str | None, threshold_type: str, progress=None) -> list[pd.DataFrame]:
    """jobs = [(DataFrame, tweet_column), ...]. Returns one per-row table per job, in the same format as
    src.batch.run_batch(). All valid texts are de-duplicated ACROSS the files and scored in one call, so a
    tweet that appears in several files is scored once and gets the same result everywhere."""
    prepared = [prepare_rows(df, col) for df, col in jobs]
    uniq = list(dict.fromkeys(t for r in prepared for t in r.loc[r.status == STATUS_OK, "tweet"]))
    res = predictor.predict_batch(uniq, model_name, threshold_type, progress=progress)
    idx = {t: i for i, t in enumerate(uniq)}
    out = []
    for rows in prepared:
        ok = rows.status == STATUS_OK
        rows["predicted_label"] = pd.Series([None] * len(rows), dtype=object)
        rows["predicted_class"] = pd.Series([pd.NA] * len(rows), dtype="Int64")
        rows["score"] = float("nan")
        rows["threshold"] = float("nan")
        for k in rows.index[ok]:
            j = idx[rows.at[k, "tweet"]]
            rows.at[k, "predicted_label"] = res["labels"][j]
            rows.at[k, "predicted_class"] = int(res["is_sarcastic"][j])
            rows.at[k, "score"] = float(res["scores"][j])
            rows.at[k, "threshold"] = res["threshold"]
        rows["model"] = res["model_name"]
        rows["threshold_mode"] = res["threshold_type"]
        rows["status"] = rows["status"].map(lambda s: STATUS_OK if s == STATUS_OK else "skipped: " + SKIP_REASONS[s])
        rows.attrs.update({"n_unique_predicted": len(uniq), "model_display": res["model_display"],
                           "score_meaning": res["score_meaning"]})
        out.append(rows)
    return out


# ---- combining and summarising -----------------------------------------------------------------
def combine_runs(runs: list[FileRun]) -> pd.DataFrame:
    """One table for all processed files: source_file, row_number, tweet + the prediction columns."""
    frames = []
    for r in runs:
        if r.rows is None:
            continue
        f = r.rows.copy()
        f.insert(0, "source_file", r.name)
        frames.append(f[COMBINED_COLS])
    if not frames:
        return pd.DataFrame(columns=COMBINED_COLS)
    return pd.concat(frames, ignore_index=True)


def cross_file_duplicates(runs: list[FileRun]) -> int:
    """Number of distinct valid tweets that appear in two or more different files."""
    seen = Counter()
    for r in runs:
        if r.rows is not None:
            seen.update(set(r.rows.loc[r.rows.status == STATUS_OK, "tweet"]))
    return sum(1 for n in seen.values() if n > 1)


def counts_consistent(runs: list[FileRun]) -> bool:
    """Sarcastic + Not Sarcastic + skipped == total rows, for every file and for the grand total."""
    done = [r for r in runs if r.summary is not None]
    per_file = all(r.summary["sarcastic"] + r.summary["not_sarcastic"] + r.summary["skipped"]
                   == r.summary["total_rows"] == len(r.df) for r in done)
    s = summarize(combine_runs(runs))
    overall = s["sarcastic"] + s["not_sarcastic"] + s["skipped"] == s["total_rows"] == \
        sum(r.summary["total_rows"] for r in done)
    return per_file and overall


def _summary_row(name: str, status: str, s: dict | None, model: str, thr: str) -> dict:
    if s is None:
        return {"file": name, "status": status, "total_rows": None, "rows_scored": None, "sarcastic": None,
                "sarcastic_percent_of_scored": None, "not_sarcastic": None,
                "not_sarcastic_percent_of_scored": None, "skipped_rows": None, "model": model,
                "threshold_mode": thr}
    return {"file": name, "status": status, "total_rows": s["total_rows"], "rows_scored": s["scored"],
            "sarcastic": s["sarcastic"], "sarcastic_percent_of_scored": round(s["sarcastic_pct"], 2),
            "not_sarcastic": s["not_sarcastic"],
            "not_sarcastic_percent_of_scored": round(s["not_sarcastic_pct"], 2),
            "skipped_rows": s["skipped"], "model": model, "threshold_mode": thr}


def multi_summary_frame(runs: list[FileRun], model: str, threshold_mode: str) -> pd.DataFrame:
    """One row per file (failed files included, with the reason) plus an 'ALL FILES' row."""
    recs = []
    for r in runs:
        if r.summary is None:
            recs.append(_summary_row(r.name, r.error or "not processed", None, model, threshold_mode))
        else:
            recs.append(_summary_row(r.name, "ok" if r.summary["scored"] else "no tweet could be scored",
                                     r.summary, model, threshold_mode))
    n_done = sum(r.summary is not None for r in runs)
    recs.append(_summary_row(ALL_FILES, f"{n_done} of {len(runs)} file(s) processed",
                             summarize(combine_runs(runs)), model, threshold_mode))
    return pd.DataFrame(recs, columns=SUMMARY_COLS)


# ---- downloads ---------------------------------------------------------------------------------
def zip_member_names(names) -> list[str]:
    """Safe, unique '<name>_results.csv' file names for the ZIP."""
    used, out = set(), []
    for n in names:
        stem = re.sub(r"[^\w.-]+", "_", PurePath(str(n)).stem).strip("._") or "file"
        cand, k = f"{stem}_results.csv", 1
        while cand in used:
            k += 1
            cand = f"{stem}_{k}_results.csv"
        used.add(cand)
        out.append(cand)
    return out


def build_outputs(runs: list[FileRun], model: str, threshold_mode: str) -> dict:
    """All downloads, built once after a run:
    combined  - one tidy CSV for every processed file (source_file column);
    summary   - per-file + overall summary CSV;
    files     - {file name: CSV with the file's ORIGINAL columns + prediction columns};
    zip       - the combined CSV, the summary CSV and one results CSV per file, in one ZIP.
    CSV files are UTF-8 with BOM so Excel opens them correctly."""
    combined = combine_runs(runs)
    summary_df = multi_summary_frame(runs, model, threshold_mode)
    done = [r for r in runs if r.rows is not None]
    files = {r.name: to_csv_bytes(build_output(r.df, r.rows)) for r in done}
    combined_b, summary_b = to_csv_bytes(combined), to_csv_bytes(summary_df)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("all_files_combined_results.csv", combined_b)
        z.writestr("summary.csv", summary_b)
        for member, r in zip(zip_member_names([r.name for r in done]), done):
            z.writestr(f"per_file/{member}", files[r.name])
    return {"combined": combined_b, "summary": summary_b, "summary_df": summary_df, "files": files,
            "zip": buf.getvalue()}


# ---- Sarcasm Detector text box: one tweet per line ----------------------------------------------
def split_tweets(text: str, max_items: int = MULTI_TWEET_MAX) -> tuple[list[str], int]:
    """Split the text box into tweets, one per non-blank line (surrounding spaces removed).
    Returns (tweets kept, number of non-blank lines found); tweets beyond max_items are not kept, so the
    caller can say how many were left out."""
    lines = [ln.strip() for ln in (text or "").splitlines()]
    lines = [ln for ln in lines if ln]
    return lines[:max(0, int(max_items))], len(lines)


def lines_table(rows: pd.DataFrame) -> pd.DataFrame:
    """Display table for the text box results (from src.batch.run_batch on the lines): scored tweets show
    label, score and threshold; skipped ones show the reason instead."""
    t = rows.copy()
    reason = t.status.where(t.status != STATUS_OK, t.note)
    return pd.DataFrame({"#": t.row_number, "Tweet": t.tweet, "Predicted label": t.predicted_label.fillna("-"),
                         "Score": t.score, "Threshold used": t.threshold, "Note": reason})
