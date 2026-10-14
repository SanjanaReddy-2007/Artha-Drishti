"""Step 6 (M12) - overall + per-topic evaluation of all models on the TEST split.

Outputs: results/overall.csv, results/per_topic.csv, results/per_topic_f1.png
Bootstrap 95% CIs (1000 resamples) are given for sarcastic-class F1 because the
per-topic test sets are small.
"""
import glob
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from config import RESULTS, SEED, MODEL_ORDER


def boot_ci(y, p, n=1000):
    rng = np.random.default_rng(SEED)
    vals = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        vals.append(f1_score(y[i], p[i], zero_division=0))
    return np.percentile(vals, [2.5, 97.5])


def main():
    files = sorted(glob.glob(str(RESULTS / "preds_*_test.csv")))
    if not files:
        raise SystemExit("No prediction files - run classical_models.py / neural_model.py first")
    overall, per_topic = [], []
    for f in files:
        name = re.search(r"preds_(.+)_test", f).group(1)
        d = pd.read_csv(f)
        y, p = d["label"].values, d["pred"].values
        lo, hi = boot_ci(y, p)
        overall.append({"model": name, "n": len(d),
                        "precision": precision_score(y, p, zero_division=0),
                        "recall": recall_score(y, p, zero_division=0),
                        "f1_sarcastic": f1_score(y, p, zero_division=0),
                        "f1_ci_low": lo, "f1_ci_high": hi,
                        "macro_f1": f1_score(y, p, average="macro", zero_division=0),
                        "accuracy": accuracy_score(y, p)})
        for t, g in d.groupby("topic"):
            yt, pt = g["label"].values, g["pred"].values
            per_topic.append({"model": name, "topic": t, "n": len(g), "n_sarcastic": int(yt.sum()),
                              "f1_sarcastic": f1_score(yt, pt, zero_division=0),
                              "precision": precision_score(yt, pt, zero_division=0),
                              "recall": recall_score(yt, pt, zero_division=0)})
    o = pd.DataFrame(overall).round(3)
    pt = pd.DataFrame(per_topic).round(3)
    o.to_csv(RESULTS / "overall.csv", index=False)
    pt.to_csv(RESULTS / "per_topic.csv", index=False)
    print("OVERALL (test)\n", o.sort_values("f1_sarcastic", ascending=False).to_string(index=False))
    piv = pt.pivot(index="topic", columns="model", values="f1_sarcastic")
    print("\nPER-TOPIC sarcastic F1 (test)\n", piv.to_string())
    print("\nNote: per-topic n_sarcastic is small; treat differences < ~0.05 as noise.")
    cols = [m for m in MODEL_ORDER if m in piv.columns] + [m for m in piv.columns if m not in MODEL_ORDER]
    piv[cols].plot(kind="bar", figsize=(9, 4.5))
    plt.ylabel("F1 (sarcastic class)"); plt.title("Per-topic F1 on test set")
    plt.xticks(rotation=15); plt.tight_layout()
    plt.savefig(RESULTS / "per_topic_f1.png", dpi=150)


if __name__ == "__main__":
    main()
