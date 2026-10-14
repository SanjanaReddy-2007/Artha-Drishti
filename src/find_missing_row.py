"""Explains why 3,456 rows became 3,455 after tagging.

Compares tweets_raw.csv with tweets_tagged.csv and prints rows that are
missing, duplicated, or have empty text. Works on id if present, else on text.
"""
import pandas as pd
from config import RAW_CSV, TAGGED_CSV

raw = pd.read_csv(RAW_CSV)
tag = pd.read_csv(TAGGED_CSV)
print(f"raw={len(raw)}  tagged={len(tag)}")

key = "id" if "id" in raw.columns and "id" in tag.columns else "text"
missing = raw[~raw[key].isin(tag[key])]
extra = tag[~tag[key].isin(raw[key])]
print(f"\nIn raw but not in tagged ({len(missing)}):")
print(missing.to_string(max_colwidth=120) if len(missing) else "  none")
print(f"\nIn tagged but not in raw ({len(extra)}):")
print(extra.head().to_string(max_colwidth=120) if len(extra) else "  none")

blank = raw[raw["text"].isna() | (raw["text"].astype(str).str.strip() == "")]
print(f"\nBlank/NaN text rows in raw: {len(blank)}")
print(f"Duplicate text rows in raw: {raw.duplicated('text').sum()}")
print("\nTypical causes: (1) the tagger skipped an empty/very short text, "
      "(2) read_csv dropped a row with a malformed quote, (3) dedupe on text.")
