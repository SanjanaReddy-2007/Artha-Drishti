"""
src/check_topic_quality.py
Checks on the zero-shot topic tags. Uses NO sarcasm labels and NO classifiers.
  Part A: how confident the assignments are (topic_margin).
  Part B: your hand-check accuracy, from results/topic_check_sample_judged.csv.
Run from the project root:  python src/check_topic_quality.py
"""
import pandas as pd

# ---------- Part A: assignment confidence ----------
df = pd.read_csv("data/processed/tweets_with_topics.csv")
print("topic_margin = best topic score minus 2nd-best (small = nearly a coin flip)")
print(df["topic_margin"].describe().round(3))
print("\nMedian margin per topic:")
print(df.groupby("topic")["topic_margin"].median().round(3))

cut = df["topic_margin"].quantile(0.25)          # weakest 25% of margins overall
df["low_margin"] = df["topic_margin"] <= cut
print(f"\n% of each topic's tweets with margin <= {cut:.3f} (overall lowest 25%):")
print((df.groupby("topic")["low_margin"].mean() * 100).round(1))

# ---------- Part B: hand-check accuracy ----------
JUDGED = "results/topic_check_sample_judged.csv"
try:
    s = pd.read_csv(JUDGED)
except FileNotFoundError:
    print(f"\nPart B skipped: {JUDGED} not found yet.")
    raise SystemExit

if "topic" not in s.columns or "my_topic" not in s.columns:
    print("\nPart B needs columns 'topic' and 'my_topic'. Found:", list(s.columns))
    raise SystemExit

names = {1: "Politics & Society", 2: "Entertainment & Sports", 3: "Everyday Life"}
s = s.dropna(subset=["my_topic"]).copy()
s["mine"] = s["my_topic"].astype(int).map(names)   # 0 -> NaN = "no clear topic"
s["has_topic"] = s["mine"].notna()
s["match"] = s["mine"] == s["topic"]

rows = []
for t, g in s.groupby("topic"):
    d = g[g["has_topic"]]
    rows.append({
        "topic": t,
        "judged": len(g),
        "no_clear_topic": int((~g["has_topic"]).sum()),
        "correct": int(d["match"].sum()),
        "of_decidable": len(d),
        "accuracy_%": round(100 * d["match"].mean(), 1) if len(d) else None,
    })
print("\nHand-check result:")
print(pd.DataFrame(rows).to_string(index=False))

d = s[s["has_topic"]]
print(f"\nOverall: {int(d['match'].sum())}/{len(d)} correct "
      f"({100 * d['match'].mean():.1f} %), plus {int((~s['has_topic']).sum())} "
      f"tweets with no clear topic (report these separately).")