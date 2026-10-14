
"""Step 1a - load iSarcasmEval train.En.csv -> data/tweets_raw.csv.

Every dropped row is logged so the row count can be explained in the report.

Usage:  python src/load_dataset.py --input path/to/train.En.csv
"""
import argparse
import pandas as pd
from config import RAW_CSV


def main(path):
    df = pd.read_csv(path)
    print(f"Read {len(df)} rows; columns = {list(df.columns)}")

    text_col = next((c for c in ("tweet", "text") if c in df.columns), None)
    label_col = next((c for c in ("sarcastic", "label") if c in df.columns), None)
    if text_col is None or label_col is None:
        raise SystemExit("Could not find text/label columns - edit load_dataset.py")

    out = df[[text_col, label_col]].rename(columns={text_col: "text", label_col: "label"})
    out["orig_row"] = df.index  # keeps a pointer back to the original file

    n0 = len(out)
    empty = out["text"].isna() | (out["text"].astype(str).str.strip() == "")
    print(f"Dropping {empty.sum()} empty/NaN text rows")
    out = out[~empty]

    dup = out.duplicated(subset="text", keep="first")
    print(f"Dropping {dup.sum()} duplicate-text rows")
    print("  (labels of dropped duplicates:", out[dup]["label"].value_counts().to_dict(), ")")
    out = out[~dup]

    out["text"] = out["text"].astype(str).str.strip()
    out["label"] = out["label"].astype(int)
    out = out.reset_index(drop=True)
    out.insert(0, "id", range(len(out)))

    print(f"Kept {len(out)} of {n0} rows")
    print(out["label"].value_counts().rename({0: "not sarcastic", 1: "sarcastic"}))
    out.to_csv(RAW_CSV, index=False)
    print(f"Saved -> {RAW_CSV}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    main(ap.parse_args().input)
