"""Learning curves: val F1 (sarcastic) vs. fraction of training data.

Classical models use their tuned params from classical_models.py.
Add --neural to include BiGRU (slower; needs torch + optional --glove).
"""
import argparse
import json
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.metrics import f1_score
import joblib
from config import SPLIT_CSV, RESULTS, MODELS, SEED, MODEL_ORDER

FRACS = [0.1, 0.2, 0.4, 0.6, 0.8, 1.0]


def main(neural, glove, text_col="text_clean"):
    df = pd.read_csv(SPLIT_CSV)
    df[text_col] = df[text_col].fillna("")
    tr, va = df[df.split == "train"], df[df.split == "val"]
    rows = []
    for name in MODEL_ORDER:
        path = MODELS / f"{name}.joblib"
        if not path.exists():
            continue
        base = clone(joblib.load(path))
        for fr in FRACS:
            # stratified subsample via groupby keeps the class ratio
            sub = pd.concat([tr[tr["label"] == c].sample(frac=fr, random_state=SEED)
                             for c in sorted(tr["label"].unique())])
            m = clone(base).fit(sub[text_col], sub["label"])
            rows.append({"model": name, "frac": fr, "n_train": len(sub),
                         "train_f1": f1_score(sub["label"], m.predict(sub[text_col])),
                         "val_f1": f1_score(va["label"], m.predict(va[text_col]))})
        print("done", name)
    if neural:
        from neural_model import train
        for fr in FRACS:
            model, vocab, best = train(df, glove, frac=fr, save=False, verbose=False)
            rows.append({"model": "BiGRU", "frac": fr, "n_train": int(len(tr) * fr),
                         "train_f1": float("nan"), "val_f1": best})
    r = pd.DataFrame(rows).round(3)
    r.to_csv(RESULTS / "learning_curves.csv", index=False)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for name, g in r.groupby("model"):
        ax.plot(g["n_train"], g["val_f1"], marker="o", label=name)
    ax.set_xlabel("training examples"); ax.set_ylabel("val F1 (sarcastic)")
    ax.set_title("Learning curves"); ax.legend(); fig.tight_layout()
    fig.savefig(RESULTS / "learning_curves.png", dpi=150)
    print(r.pivot(index="frac", columns="model", values="val_f1").to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--neural", action="store_true")
    ap.add_argument("--glove", default=None)
    a = ap.parse_args()
    main(a.neural, a.glove)
