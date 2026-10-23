"""Batch Detector logic. No Streamlit imports, so it is unit-testable.

Only the selected tweet column is ever passed to the model (leakage guard). A label column, if the
user chooses one, is used only afterwards to compute an ad-hoc check on the user's own file.
"""
from __future__ import annotations

import csv
import io
import math
import re
from dataclasses import dataclass, field

import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

from src.config import (BATCH_ENCODINGS, BATCH_MAX_MB, BATCH_MAX_ROWS, CLASS_NAMES, LABEL_COLUMNS,
                        MAX_INPUT_CHARS, TEXT_COLUMNS, TWEET_CHARS)
from src.preprocess import clean_text

PRED_COLS = ["predicted_label", "predicted_class", "score", "threshold", "model", "threshold_mode",
             "status", "note"]
STATUS_OK = "predicted"
SKIP_REASONS = {
    "blank": "blank or missing tweet",
    "no_words": "no readable English words after cleaning",
    "too_long": f"longer than {MAX_INPUT_CHARS} characters (the single-tweet page rejects this too)",
}


class BatchError(ValueError):
    """A problem with the uploaded file, shown to the user as a message (never a stack trace)."""


@dataclass
class ParsedFile:
    df: pd.DataFrame
    encoding: str
    notes: list = field(default_factory=list)


# ---- reading the upload ------------------------------------------------------------------------
def decode_bytes(raw: bytes) -> tuple[str, str]:
    """Try utf-8, utf-8-sig, latin-1 (latin-1 never fails). A BOM is stripped. Returns (text, encoding)."""
    for enc in BATCH_ENCODINGS:
        try:
            text = raw.decode(enc)
        except UnicodeDecodeError:
            continue
        if enc == "utf-8" and text.startswith("\ufeff"):
            return text[1:], "utf-8-sig"
        return text, enc
    raise BatchError("The file could not be decoded as text.")  # pragma: no cover (latin-1 always works)


def _delimiter(first_line: str) -> str:
    if "," in first_line:
        return ","
    for d in ("\t", ";", "|"):
        if d in first_line:
            return d
    return ","


def _count_ragged(text: str, sep: str, ncols: int) -> int:
    try:
        return sum(1 for r in csv.reader(io.StringIO(text), delimiter=sep) if r and len(r) != ncols)
    except csv.Error:
        return 0


def read_upload(raw: bytes, filename: str = "upload.csv", has_header: bool = True,
                max_rows: int = BATCH_MAX_ROWS, max_mb: float = BATCH_MAX_MB) -> ParsedFile:
    """Parse .csv (or .txt with one tweet per line) into a DataFrame of strings.
    Raises BatchError with a friendly message for empty / oversized / unreadable files."""
    if raw is None or len(raw) == 0:
        raise BatchError("The file is empty.")
    if len(raw) > max_mb * 1024 * 1024:
        raise BatchError(f"The file is {len(raw) / 1024 / 1024:.1f} MB; the limit is {max_mb:g} MB. "
                         "Please upload a smaller file.")
    text, enc = decode_bytes(raw)
    if not text.strip():
        raise BatchError("The file is empty (it contains only whitespace).")
    notes = []
    if enc == "latin-1":
        notes.append("The file is not valid UTF-8, so it was read as **latin-1**. Some special characters "
                     "(for example emojis) may look wrong.")
    elif enc == "utf-8-sig":
        notes.append("The file starts with a UTF-8 byte-order mark (BOM); it was removed.")

    if filename.lower().endswith(".txt"):
        lines = text.splitlines()
        if len(lines) > max_rows:
            raise BatchError(f"The file has {len(lines):,} lines; the limit is {max_rows:,} rows.")
        df = pd.DataFrame({"tweet": lines})
        notes.append("Plain-text file: each line was treated as one tweet (column name: tweet).")
        return ParsedFile(df, enc, notes)

    sep = _delimiter(text.split("\n", 1)[0])
    try:
        # index_col=False: never let pandas silently promote a column to the index when rows are ragged.
        df = pd.read_csv(io.StringIO(text), sep=sep, dtype=str, keep_default_na=False,
                         header=0 if has_header else None, engine="python", skip_blank_lines=False,
                         index_col=False)
    except pd.errors.EmptyDataError:
        raise BatchError("The file has no readable data.") from None
    except (pd.errors.ParserError, csv.Error) as e:
        raise BatchError(f"The file could not be read as CSV ({str(e)[:120]}).") from None
    n_ragged = _count_ragged(text, sep, df.shape[1])
    if n_ragged:
        notes.append(f"{n_ragged} line(s) had a different number of fields than the header (often an unquoted "
                     "comma inside a tweet). Extra fields were dropped and missing ones left empty; wrap such "
                     "tweets in double quotes to keep them whole.")
    if df.shape[1] == 0 or len(df) == 0:
        raise BatchError("The file has a header but no data rows." if has_header else "The file has no data rows.")
    if not has_header:
        df.columns = [f"column {i + 1}" for i in range(df.shape[1])]
    else:
        df.columns = [str(c).strip() if str(c).strip() else f"column {i + 1}" for i, c in enumerate(df.columns)]
        if df.columns.duplicated().any():  # keep every column; make names unique
            seen, new = {}, []
            for c in df.columns:
                seen[c] = seen.get(c, 0) + 1
                new.append(c if seen[c] == 1 else f"{c}.{seen[c] - 1}")
            df.columns = new
    if len(df) > max_rows:
        raise BatchError(f"The file has {len(df):,} rows; the limit is {max_rows:,} rows. "
                         "Please split it into smaller files.")
    return ParsedFile(df, enc, notes)


# ---- choosing the tweet column -----------------------------------------------------------------
def _is_number(x) -> bool:
    try:
        float(str(x).replace(",", ""))
        return True
    except ValueError:
        return False


def detect_text_column(df: pd.DataFrame) -> str:
    """'tweet' (case-insensitive), else 'text', else the first column that is mostly non-numeric text."""
    lower = {str(c).strip().lower(): c for c in df.columns}
    for key in ("tweet", "text"):
        if key in lower:
            return lower[key]
    for c in df.columns:
        vals = [v for v in df[c].head(200) if str(v).strip() != ""]
        if vals and sum(_is_number(v) for v in vals) / len(vals) < 0.5:
            return c
    return df.columns[0]


def detect_label_column(df: pd.DataFrame, tweet_col: str) -> str | None:
    """A column named like a label (sarcastic / label) whose non-empty values are all 0/1, else None."""
    names = {str(n).lower() for n in LABEL_COLUMNS}
    for c in df.columns:
        if c != tweet_col and str(c).strip().lower() in names and _label_series(df[c]).notna().any():
            return c
    return None


# ---- preparing rows (same checks as the single-tweet page) -------------------------------------
def prepare_rows(df: pd.DataFrame, tweet_col: str) -> pd.DataFrame:
    """One row per input row with columns: row_number (1-based), tweet, status, note.
    status = 'predicted' (valid, to be scored) or a skip reason key ('blank', 'no_words', 'too_long')."""
    if tweet_col not in df.columns:
        raise BatchError(f"Column '{tweet_col}' was not found in the file.")
    out = []
    for i, v in enumerate(df[tweet_col].tolist(), start=1):
        if v is None or (isinstance(v, float) and math.isnan(v)):
            text, status, note = "", "blank", ""
        else:
            text = v if isinstance(v, str) else str(v)          # non-string cells (numbers, bools) -> text
            if not text.strip():
                status, note = "blank", ""
            elif len(text) > MAX_INPUT_CHARS:
                status, note = "too_long", ""
            elif not re.search(r"[a-z]", clean_text(text)):
                status, note = "no_words", ""
            else:
                status = STATUS_OK
                note = (f"Longer than a tweet ({TWEET_CHARS} characters); the models were trained on tweets."
                        if len(text) > TWEET_CHARS else "")
        out.append({"row_number": i, "tweet": text, "status": status, "note": note})
    return pd.DataFrame(out, columns=["row_number", "tweet", "status", "note"])


def run_batch(predictor, df: pd.DataFrame, tweet_col: str, model_name: str | None, threshold_type: str,
              progress=None) -> pd.DataFrame:
    """Prepare rows, predict once per UNIQUE valid text, and map the results back to every row.
    Returns the per-row table (row_number, tweet, status, note, predicted_label, predicted_class,
    score, threshold, model, threshold_mode)."""
    rows = prepare_rows(df, tweet_col)
    ok = rows.status == STATUS_OK
    uniq = list(dict.fromkeys(rows.loc[ok, "tweet"]))             # order-preserving unique raw texts
    res = predictor.predict_batch(uniq, model_name, threshold_type, progress=progress)
    idx = {t: i for i, t in enumerate(uniq)}
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
    return rows


# ---- summary and output ------------------------------------------------------------------------
def summarize(rows: pd.DataFrame) -> dict:
    total = len(rows)
    n_s = int((rows.predicted_label == "Sarcastic").sum())
    n_n = int((rows.predicted_label == "Not Sarcastic").sum())
    skipped = total - n_s - n_n
    scored = n_s + n_n
    pct = lambda n, d: (100.0 * n / d) if d else 0.0
    return {"total_rows": total, "scored": scored, "sarcastic": n_s, "not_sarcastic": n_n, "skipped": skipped,
            "sarcastic_pct": pct(n_s, scored), "not_sarcastic_pct": pct(n_n, scored),
            "sarcastic_pct_of_total": pct(n_s, total), "not_sarcastic_pct_of_total": pct(n_n, total),
            "duplicate_rows": int(rows.loc[rows.status == STATUS_OK, "tweet"].duplicated().sum())}


def summary_frame(s: dict, model: str, threshold_mode: str) -> pd.DataFrame:
    items = [("model", model), ("threshold_mode", threshold_mode), ("total_rows", s["total_rows"]),
             ("rows_scored", s["scored"]), ("sarcastic", s["sarcastic"]),
             ("sarcastic_percent_of_scored", round(s["sarcastic_pct"], 2)),
             ("not_sarcastic", s["not_sarcastic"]),
             ("not_sarcastic_percent_of_scored", round(s["not_sarcastic_pct"], 2)),
             ("skipped_rows", s["skipped"])]
    return pd.DataFrame(items, columns=["metric", "value"])


def build_output(original: pd.DataFrame, rows: pd.DataFrame) -> pd.DataFrame:
    """Original columns (unchanged) + prediction columns. Every input row is kept.
    If an original column already uses a prediction column name, the original is renamed '<name>_original'."""
    orig = original.copy().reset_index(drop=True)
    orig.columns = [f"{c}_original" if c in PRED_COLS else c for c in orig.columns]
    pred = rows[PRED_COLS].reset_index(drop=True)
    return pd.concat([orig, pred], axis=1)


def to_csv_bytes(df: pd.DataFrame) -> bytes:
    """UTF-8 with BOM so Excel opens non-ASCII text correctly."""
    return df.to_csv(index=False).encode("utf-8-sig")


# ---- optional labelled-file check (ad-hoc, NOT the official test result) -----------------------
_TRUE = {"1", "1.0", "true", "yes", "sarcastic"}
_FALSE = {"0", "0.0", "false", "no", "not sarcastic", "not_sarcastic"}


def _label_series(s: pd.Series) -> pd.Series:
    def conv(v):
        k = str(v).strip().lower()
        return 1 if k in _TRUE else 0 if k in _FALSE else pd.NA
    return s.map(conv).astype("Int64")


def labelled_check(df: pd.DataFrame, rows: pd.DataFrame, label_col: str) -> dict:
    """Sarcastic-class precision/recall/F1/accuracy + confusion matrix on the user's own labelled file.
    Uses only rows that were scored and have a valid 0/1 label."""
    y = _label_series(df[label_col]).reset_index(drop=True)
    pred = rows.predicted_class.reset_index(drop=True)
    keep = y.notna() & pred.notna()
    n_bad = int((rows.predicted_class.notna().reset_index(drop=True) & y.isna()).sum())
    if keep.sum() == 0:
        raise BatchError(f"Column '{label_col}' has no usable 0/1 labels on the scored rows.")
    yt, yp = y[keep].astype(int).to_numpy(), pred[keep].astype(int).to_numpy()
    tn, fp, fn, tp = confusion_matrix(yt, yp, labels=[0, 1]).ravel()
    return {"n_used": int(keep.sum()), "n_invalid_labels": n_bad, "n_sarcastic_true": int(yt.sum()),
            "precision": float(precision_score(yt, yp, pos_label=1, zero_division=0)),
            "recall": float(recall_score(yt, yp, pos_label=1, zero_division=0)),
            "f1": float(f1_score(yt, yp, pos_label=1, zero_division=0)),
            "accuracy": float(accuracy_score(yt, yp)),
            "confusion": pd.DataFrame([[tn, fp], [fn, tp]], index=[f"Actual: {c}" for c in CLASS_NAMES],
                                      columns=[f"Predicted: {c}" for c in CLASS_NAMES])}


def compare_models(predictor, rows: pd.DataFrame, models, threshold_type: str, progress=None) -> pd.DataFrame:
    """Run every given model on the valid rows of a run_batch() table; sarcastic counts per model
    (counted per input row, duplicates included; each unique text is scored once)."""
    ok = rows[rows.status == STATUS_OK]
    counts = ok.tweet.value_counts()
    uniq = list(counts.index)
    weights = counts.to_numpy()
    total = int(weights.sum())
    out = []
    for i, m in enumerate(models):
        try:
            r = predictor.predict_batch(uniq, m, threshold_type)
            n = int(weights[r["is_sarcastic"]].sum())
            out.append({"Model": m, "Sarcastic": n, "Not Sarcastic": total - n,
                        "Sarcastic %": round(100.0 * n / max(1, total), 1), "Note": ""})
        except Exception as e:  # one broken model must not hide the others
            out.append({"Model": m, "Sarcastic": None, "Not Sarcastic": None, "Sarcastic %": None,
                        "Note": f"could not run ({type(e).__name__})"})
        if progress:
            progress((i + 1) / len(models))
    return pd.DataFrame(out)
