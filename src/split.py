"""Reproducible stratified 80/10/10 split on the label only (topic-independent).

The split is saved as row_id lists in results/split_indices.json so every model, and the
dashboard, uses exactly the same examples. To reuse the split from the earlier project
(which was stratified on label x topic), use --import-existing; that keeps the same
train/val/test membership instead of making a new split.
"""
from __future__ import annotations

import argparse
import json
import logging

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import SEED, ProjectPaths

log = logging.getLogger("artha.split")


def make_split(df: pd.DataFrame, seed: int = SEED, val_frac: float = 0.1, test_frac: float = 0.1) -> dict:
    ids, y = df.row_id.values, df.label.values
    trainval_ids, test_ids, y_tv, _ = train_test_split(ids, y, test_size=test_frac, stratify=y, random_state=seed)
    train_ids, val_ids = train_test_split(trainval_ids, test_size=val_frac / (1 - test_frac),
                                          stratify=y_tv, random_state=seed)
    return {"strategy": f"stratified on label, seed={seed}", "seed": seed,
            "train_ids": sorted(map(int, train_ids)), "val_ids": sorted(map(int, val_ids)),
            "test_ids": sorted(map(int, test_ids))}


def import_existing(df: pd.DataFrame, train_csv, val_csv, test_csv, text_col: str, allow_unmatched=False) -> dict:
    """Map an earlier split (three CSVs with a text column) onto processed row_ids by exact text."""
    from src.preprocess import clean_text
    by_raw = dict(zip(df.text_raw.str.strip(), df.row_id))
    by_clean = dict(zip(df.text_clean, df.row_id))
    out, unmatched = {}, {}
    for name, path in (("train", train_csv), ("val", val_csv), ("test", test_csv)):
        texts = pd.read_csv(path)[text_col].astype(str)
        ids, miss = [], 0
        for t in texts:
            rid = by_raw.get(t.strip(), by_clean.get(clean_text(t)))
            if rid is None:
                miss += 1
            else:
                ids.append(int(rid))
        out[f"{name}_ids"], unmatched[name] = sorted(set(ids)), miss
    if any(unmatched.values()) and not allow_unmatched:
        raise SystemExit(f"Could not match some rows of the old split to processed data: {unmatched}. "
                         "Check --text-col, or pass --allow-unmatched to drop them (and document it).")
    return {"strategy": "imported from earlier split files", "seed": None, "unmatched_rows": unmatched, **out}


def verify_split(df: pd.DataFrame, split: dict) -> dict:
    """Raise if splits overlap or share duplicate text; return a summary."""
    sets = {k: set(split[f"{k}_ids"]) for k in ("train", "val", "test")}
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        if sets[a] & sets[b]:
            raise ValueError(f"Split leakage: {a} and {b} share {len(sets[a] & sets[b])} rows")
    summary = {}
    texts = {}
    for k, ids in sets.items():
        sub = df[df.row_id.isin(ids)]
        texts[k] = set(sub.text_clean)
        summary[k] = {"n": int(len(sub)), "n_sarcastic": int(sub.label.sum()),
                      "sarcastic_share": round(float(sub.label.mean()), 4)}
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        if texts[a] & texts[b]:
            raise ValueError(f"Duplicate-text leakage between {a} and {b}")
    return summary


def load_split(paths: ProjectPaths, df: pd.DataFrame):
    split = json.loads(paths.split_file.read_text(encoding="utf-8"))
    verify_split(df, split)
    return tuple(df[df.row_id.isin(set(split[f"{k}_ids"]))].reset_index(drop=True) for k in ("train", "val", "test"))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--import-existing", action="store_true")
    ap.add_argument("--train-csv"), ap.add_argument("--val-csv"), ap.add_argument("--test-csv")
    ap.add_argument("--text-col", default="text")
    ap.add_argument("--allow-unmatched", action="store_true")
    ap.add_argument("--force", action="store_true", help="overwrite an existing split file")
    a = ap.parse_args(argv)
    paths = ProjectPaths().ensure()
    if paths.split_file.exists() and not a.force:
        raise SystemExit(f"{paths.split_file} exists. Splits must not change silently; use --force to overwrite.")
    df = pd.read_csv(paths.processed_file)
    split = (import_existing(df, a.train_csv, a.val_csv, a.test_csv, a.text_col, a.allow_unmatched)
             if a.import_existing else make_split(df, a.seed))
    split["summary"] = verify_split(df, split)
    paths.split_file.write_text(json.dumps(split, indent=1), encoding="utf-8")
    print(json.dumps(split["summary"], indent=2))
    print(f"Saved {paths.split_file}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
