"""
LOAD DATASET (replaces Tweepy extraction; the "Loading dataset" fallback)

What it does:
  1. Reads one or MORE raw dataset files (csv / tsv / txt / json / jsonl) and merges them.
  2. Renames your columns to a standard format: tweet_id, text, label.
  3. Converts the label to 1 = sarcastic, 0 = not sarcastic.
  4. Removes empty and exact-duplicate tweets, prints a quality report.
  5. Saves data/processed/tweets_raw.csv for the next steps.

Run examples (from the project root folder):
  # file with a header row
  python src/load_dataset.py --input data/raw/my_data.csv --text-col tweet --label-col class --sarcastic-value sarcasm
  # tab-separated .txt files with NO header row, columns are label then tweet
  python src/load_dataset.py --input data/raw/train.txt data/raw/test.txt --no-header --columns label,text --text-col text --label-col label --sarcastic-value 1
"""
import argparse
import csv
import os
import sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent   # project root folder


def read_any(path, encoding, no_header, names):
    ext = os.path.splitext(path)[1].lower()
    kw = dict(encoding=encoding, on_bad_lines="skip")
    if no_header:
        kw.update(header=None, names=names)
    if ext in (".tsv", ".txt"):      # tab-separated; QUOTE_NONE so quotes inside tweets don't break parsing
        return pd.read_csv(path, sep="\t", quoting=csv.QUOTE_NONE, **kw)
    if ext == ".json":
        return pd.read_json(path)
    if ext == ".jsonl":
        return pd.read_json(path, lines=True)
    return pd.read_csv(path, **kw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, nargs="+",
                    help="one or more dataset files (they are merged into one pool)")
    ap.add_argument("--no-header", action="store_true",
                    help="the file has no header row; then you must give --columns")
    ap.add_argument("--columns", default="",
                    help="comma-separated column names for a headerless file, e.g. label,text or id,label,text")
    ap.add_argument("--text-col", required=True, help="column holding the tweet text")
    ap.add_argument("--label-col", required=True, help="column holding sarcasm label")
    ap.add_argument("--id-col", default=None, help="tweet id column (optional)")
    ap.add_argument("--sarcastic-value", default="1",
                    help="value in label column that means SARCASTIC (e.g. 1, sarcasm, True)")
    ap.add_argument("--encoding", default="utf-8", help="try latin-1 if you get decode errors")
    ap.add_argument("--output", default=str(ROOT / "data/processed/tweets_raw.csv"))
    a = ap.parse_args()

    names = [c.strip() for c in a.columns.split(",")] if a.columns else None
    if a.no_header and not names:
        sys.exit("ERROR: --no-header needs --columns, e.g. --columns label,text")
    frames = []
    for path in a.input:
        part = read_any(path, a.encoding, a.no_header, names)
        print(f"  {path}: {len(part):,} rows")
        frames.append(part)
    df = pd.concat(frames, ignore_index=True)
    print(f"Loaded {len(df):,} rows in total. Columns found: {list(df.columns)}")
    for c in [a.text_col, a.label_col] + ([a.id_col] if a.id_col else []):
        if c not in df.columns:
            sys.exit(f"ERROR: column '{c}' not found. Check spelling against the list above.")

    # --- show raw label values so YOU can confirm the mapping is right ---
    raw_vals = df[a.label_col].astype(str).str.strip().str.lower()
    print("\nRaw label values and counts (confirm which one means sarcastic):")
    print(raw_vals.value_counts().head(10))
    if raw_vals.nunique() != 2:
        sys.exit("ERROR: label column must have exactly 2 distinct values for binary sarcasm "
                 "detection. Filter or fix the dataset first.")
    if a.sarcastic_value.strip().lower() not in set(raw_vals):
        sys.exit(f"ERROR: --sarcastic-value '{a.sarcastic_value}' is not one of the values above.")

    out = pd.DataFrame({
        "tweet_id": df[a.id_col] if a.id_col else range(len(df)),
        "text": df[a.text_col].astype(str),
        "label": (raw_vals == a.sarcastic_value.strip().lower()).astype(int),
    })

    # --- basic cleaning of rows (not text) ---
    n0 = len(out)
    out["text"] = out["text"].str.strip()
    out = out[(out["text"] != "") & (out["text"].str.lower() != "nan")]
    n_empty = n0 - len(out)
    n1 = len(out)
    out = out.drop_duplicates(subset="text")
    n_dup = n1 - len(out)

    # --- quality report ---
    print(f"\nRemoved {n_empty:,} empty rows and {n_dup:,} exact-duplicate tweets.")
    print(f"Final rows: {len(out):,}")
    print("\nClass balance (1 = sarcastic):")
    print(out["label"].value_counts().rename({1: "sarcastic", 0: "not sarcastic"}))
    print(f"\nTweet length (characters): mean {out['text'].str.len().mean():.0f}, "
          f"median {out['text'].str.len().median():.0f}")

    leak = out["text"].str.contains(r"#(?:sarcasm|sarcastic|not)\b", case=False, regex=True)
    print(f"\nTweets still containing #sarcasm/#sarcastic/#not: {leak.sum():,} "
          f"({leak.mean():.1%}). They are removed in preprocessing so the model can't cheat.")
    if len(out) < 15000:
        print("WARNING: under ~15,000 tweets. Per-topic results may be noisy; use 4 topics.")

    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(a.output, index=False)
    print(f"\nSaved -> {a.output}")
    print("Sample rows:")
    print(out.sample(min(5, len(out)), random_state=42).to_string(index=False))


if __name__ == "__main__":
    main()
