"""Step 3 (M5) - stratified 80/10/10 split.

Stratified on (label x topic) so every topic keeps sarcastic examples in val
and test (needed for per-topic evaluation). Cross-validation is done INSIDE the
training portion for hyper-parameter tuning (see classical_models.py).
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from config import CLEAN_CSV, SPLIT_CSV, SEED


def main():
    df = pd.read_csv(CLEAN_CSV)
    strat = df["label"].astype(str) + "_" + df["topic"].astype(str)
    tr, rest = train_test_split(df.index, test_size=0.20, stratify=strat, random_state=SEED)
    va, te = train_test_split(rest, test_size=0.50, stratify=strat[rest], random_state=SEED)
    df["split"] = "train"
    df.loc[va, "split"] = "val"
    df.loc[te, "split"] = "test"
    df.to_csv(SPLIT_CSV, index=False)
    print(df.groupby("split")["label"].agg(n="size", sarcastic="sum",
          share=lambda s: round(s.mean(), 3)))
    print(pd.crosstab([df["split"]], df["topic"]))
    print(f"Saved -> {SPLIT_CSV}")


if __name__ == "__main__":
    main()
