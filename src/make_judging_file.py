"""Step 1c - build a BLIND hand-check file of 60 tweets (20 per topic).

You see only the tweet text (not the model's topic) and fill `my_topic` with
one of: 1 = Everyday Life, 2 = Entertainment & Sports, 3 = Politics & Society,
0 = no clear topic.   Key goes to judging_key.csv.
"""
import argparse
import pandas as pd
from config import TAGGED_CSV, DATA, TOPIC_NAMES, SEED


def main(per_topic):
    df = pd.read_csv(TAGGED_CSV)
    parts = []
    for t in TOPIC_NAMES:
        sub = df[df["topic"] == t]
        parts.append(sub.sample(min(per_topic, len(sub)), random_state=SEED))
    s = pd.concat(parts).sample(frac=1, random_state=SEED).reset_index(drop=True)
    s[["id", "text"]].assign(my_topic="").to_csv(DATA / "judging_file.csv", index=False)
    s[["id", "topic", "topic_score", "label"]].to_csv(DATA / "judging_key.csv", index=False)
    print(f"Wrote {len(s)} tweets -> data/judging_file.csv")
    print("Codes: 1=Everyday Life  2=Entertainment & Sports  3=Politics & Society  0=no clear topic")
    print("Then run: python src/check_topic_quality.py")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--per_topic", type=int, default=20)
    main(ap.parse_args().per_topic)
