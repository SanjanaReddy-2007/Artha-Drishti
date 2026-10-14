"""Step 1d - topic-tagging quality report.

Part A: confidence of the zero-shot assignments (no labels needed).
Part B: accuracy vs. your hand labels in data/judging_file.csv (if filled).
Paste the whole output into the chat / report.
"""
import math
import pandas as pd
from config import TAGGED_CSV, DATA, TOPIC_NAMES

CODE = {1: TOPIC_NAMES[0], 2: TOPIC_NAMES[1], 3: TOPIC_NAMES[2], 0: "None"}


def wilson(k, n, z=1.96):
    if n == 0:
        return (0, 0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def part_a(df):
    print("=" * 60, "\nPART A - assignment confidence\n" + "=" * 60)
    df = df.assign(n_words=df["text"].astype(str).str.split().str.len())
    g = df.groupby("topic").agg(n=("id", "size"),
                                mean_score=("topic_score", "mean"),
                                median_margin=("topic_margin", "median"),
                                pct_low_conf=("low_conf", "mean"),
                                mean_words=("n_words", "mean")).round(3)
    print(g.to_string())
    short = df["n_words"] <= 6
    print(f"\nShort tweets (<=6 words): {short.mean():.1%} of data; "
          f"mean topic_score {df[short]['topic_score'].mean():.3f} "
          f"vs {df[~short]['topic_score'].mean():.3f} for longer tweets")
    print("Interpretation: low score/margin on short tweets means those tags are close to arbitrary.")


def part_b(df):
    f = DATA / "judging_file.csv"
    k = DATA / "judging_key.csv"
    if not f.exists() or not k.exists():
        print("\n(Part B skipped - run make_judging_file.py first)")
        return
    j = pd.read_csv(f).merge(pd.read_csv(k), on="id")
    j = j[j["my_topic"].notna()]
    if len(j) == 0:
        print("\n(Part B skipped - fill the my_topic column first)")
        return
    j["human"] = j["my_topic"].astype(int).map(CODE)
    print("\n" + "=" * 60 + f"\nPART B - hand-check ({len(j)} tweets)\n" + "=" * 60)
    clear = j[j["human"] != "None"]
    acc_clear = (clear["human"] == clear["topic"]).mean() if len(clear) else float("nan")
    acc_all = (j["human"] == j["topic"]).mean()
    lo, hi = wilson(int((clear["human"] == clear["topic"]).sum()), len(clear))
    print(f"Tweets with no clear topic (human): {(j['human'] == 'None').sum()} ({(j['human']=='None').mean():.0%})")
    print(f"Accuracy on tweets that HAVE a topic: {acc_clear:.1%}  (95% CI {lo:.0%}-{hi:.0%}, n={len(clear)})")
    print(f"Accuracy counting 'none' as wrong:    {acc_all:.1%}")
    print("\nPer-topic accuracy (clear tweets):")
    for t, g in clear.groupby("topic"):
        print(f"  {t}: {(g['human'] == g['topic']).mean():.3f} (n={len(g)})")
    print("\nConfusion (rows = model, cols = human):")
    print(pd.crosstab(j["topic"], j["human"]).to_string())
    print("\nSuggested decision: >=75% accuracy -> keep TOPICS; 60-75% -> revise label wording "
          "or add a confidence threshold; <60% -> reconsider topics.")


if __name__ == "__main__":
    d = pd.read_csv(TAGGED_CSV)
    part_a(d)
    part_b(d)
