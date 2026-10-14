"""Step 7 (M13) - topic x classifier recommender.

Rule (state this explicitly in the report):
  1. For each topic, pick the model with the highest sarcastic-class F1 on the
     VALIDATION split (test is kept clean for reporting).
  2. Models within `--tie_eps` (default 0.01) of the best are treated as tied;
     the tie is broken by simplicity (MODEL_ORDER: NB < LogReg < SVM < KNN < RF < BiGRU).
Writes results/recommender.json and results/recommender_matrix.csv.
Also reports the TEST F1 of the recommended model vs. the best single global model.
"""
import argparse
import glob
import json
import re
import pandas as pd
from sklearn.metrics import f1_score
from config import RESULTS, MODEL_ORDER


def load(split):
    rows = []
    for f in glob.glob(str(RESULTS / f"preds_*_{split}.csv")):
        name = re.search(rf"preds_(.+)_{split}", f).group(1)
        d = pd.read_csv(f)
        for t, g in d.groupby("topic"):
            rows.append({"model": name, "topic": t,
                         "f1": f1_score(g["label"], g["pred"], zero_division=0)})
    return pd.DataFrame(rows)


def rank(m):
    return MODEL_ORDER.index(m) if m in MODEL_ORDER else len(MODEL_ORDER)


def main(eps):
    val = load("val")
    mat = val.pivot(index="topic", columns="model", values="f1").round(3)
    mat.to_csv(RESULTS / "recommender_matrix.csv")
    print("Validation F1 matrix (topic x model)\n", mat.to_string())
    rec = {}
    for topic, row in mat.iterrows():
        best = row.max()
        tied = [m for m in row.index if best - row[m] <= eps]
        rec[topic] = min(tied, key=rank)
        print(f"{topic}: best={row.idxmax()} ({best:.3f}); tied within {eps}: {tied} -> {rec[topic]}")
    json.dump(rec, open(RESULTS / "recommender.json", "w"), indent=2)

    # Compare on the test set
    test_files = {re.search(r"preds_(.+)_test", f).group(1): pd.read_csv(f)
                  for f in glob.glob(str(RESULTS / "preds_*_test.csv"))}
    glob_best = max(test_files, key=lambda m: f1_score(test_files[m]["label"], test_files[m]["pred"]))
    parts = []
    for topic, m in rec.items():
        parts.append(test_files[m][test_files[m]["topic"] == topic])
    routed = pd.concat(parts)
    print(f"\nTEST sarcastic F1 - recommender routing: {f1_score(routed['label'], routed['pred']):.3f}  |  "
          f"best single model ({glob_best}): {f1_score(test_files[glob_best]['label'], test_files[glob_best]['pred']):.3f}")
    print("If routing is not clearly better, report that honestly - it is still a valid finding.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tie_eps", type=float, default=0.01)
    main(ap.parse_args().tie_eps)
