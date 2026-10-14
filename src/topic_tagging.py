"""
TOPIC TAGGING (zero-shot, sentence embeddings)

Idea: you describe each topic in plain English (see TOPICS below). A small pretrained
language model turns every tweet and every topic description into a vector, and each
tweet gets the topic whose description it is closest to (cosine similarity).
No training data for topics is needed, and the topic names are exactly the ones YOU chose.

Run from the project root:
  python src/topic_tagging.py

Input : data/processed/tweets_raw.csv           (from load_dataset.py)
Output: data/processed/tweets_with_topics.csv   (tweet + topic)
        models/topic_definitions.json           (used later by the demo to tag new tweets)
        results/topic_check_sample.csv          (tweets for YOU to hand-check)

First run downloads the model (~90 MB) from the internet, so you need to be online once.
"""
import argparse
import json
import re
import sys
from pathlib import Path
import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):                       # avoid emoji print errors on Windows
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

SEED = 42
ROOT = Path(__file__).resolve().parent.parent                # project root folder
MODEL_NAME = "all-MiniLM-L6-v2"

# ----------------------------------------------------------------------------------
# EDIT THIS to change topics. Name = the tag used in results. Descriptions = what the
# topic is about; more varied descriptions work better than one long sentence.
# ----------------------------------------------------------------------------------
TOPICS = {
    "Politics & Society": [
        "politics, government, elections and politicians",
        "news, protests, racism and social issues",
        "the coronavirus pandemic, lockdown and public health policy",
        "taxes, the economy, benefits and the law",
    ],
    "Entertainment & Sports": [
        "movies, TV shows and streaming",
        "music, celebrities and social media trends",
        "video games and gaming",
        "sports, football, baseball and athletes",
    ],
    "Everyday Life": [
        "school, college, exams and homework",
        "work, jobs, bosses and coworkers",
        "family, friends, relationships and dating",
        "daily routines, food, sleep, weather, travel and customer service",
    ],
}


def embed_clean(t):
    """Mild cleaning for embeddings: drop links, @mentions and label hashtags, keep the words."""
    t = re.sub(r"http\S+|www\.\S+", " ", t)
    t = re.sub(r"@\w+", " ", t)
    t = re.sub(r"#(sarcasm|sarcastic|not|irony|ironic)\b", " ", t, flags=re.I)
    t = t.replace("#", " ")
    return re.sub(r"\s+", " ", t).strip()


def assign_topics(texts, model, topics=TOPICS):
    """Return (topic names array, best score, margin between best and 2nd best topic)."""
    names = list(topics)
    topic_vecs = []
    for n in names:
        v = model.encode(topics[n], normalize_embeddings=True)
        m = v.mean(axis=0)
        topic_vecs.append(m / np.linalg.norm(m))
    T = np.vstack(topic_vecs)                                        # (n_topics, dim)
    X = model.encode(list(texts), normalize_embeddings=True, batch_size=64, show_progress_bar=True)
    S = X @ T.T                                                      # cosine similarities
    order = np.sort(S, axis=1)
    best = S.argmax(axis=1)
    return np.array(names)[best], S.max(axis=1), order[:, -1] - order[:, -2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=str(ROOT / "data/processed/tweets_raw.csv"))
    ap.add_argument("--balance", action="store_true",
                    help="downsample to equal sarcastic/non-sarcastic inside each topic "
                         "(NOT recommended for small datasets: it throws away data)")
    ap.add_argument("--min-margin", type=float, default=0.0,
                    help="drop tweets whose best topic beats the 2nd best by less than this (e.g. 0.03)")
    a = ap.parse_args()

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        sys.exit("Install it first:  pip install sentence-transformers")

    df = pd.read_csv(a.input)
    df["embed_text"] = df["text"].astype(str).apply(embed_clean)
    df = df[df["embed_text"].str.len() > 0].reset_index(drop=True)
    print(f"Tagging {len(df):,} tweets with {len(TOPICS)} topics using '{MODEL_NAME}' ...")

    model = SentenceTransformer(MODEL_NAME)
    df["topic"], df["topic_score"], df["topic_margin"] = assign_topics(df["embed_text"], model)

    n0 = len(df)
    df = df[df["topic_margin"] >= a.min_margin]
    if a.min_margin > 0:
        print(f"Dropped {n0 - len(df):,} ambiguous tweets (margin < {a.min_margin}).")

    print("\nRows per topic x label (0 = not sarcastic, 1 = sarcastic):")
    print(pd.crosstab(df["topic"], df["label"], margins=True))
    print("\nSarcastic share per topic:")
    print((df.groupby("topic")["label"].mean() * 100).round(1).astype(str) + " %")

    if a.balance:
        parts = []
        for _, g in df.groupby("topic"):
            n = g["label"].value_counts().min()
            for lab in (0, 1):
                parts.append(g[g["label"] == lab].sample(n, random_state=SEED))
        df = pd.concat(parts)
        print("\nAFTER balancing:")
        print(pd.crosstab(df["topic"], df["label"], margins=True))

    for t, g in df.groupby("topic"):
        if g["label"].value_counts().min() < 150:
            print(f"WARNING: topic '{t}' has fewer than 150 tweets in one class. "
                  f"Per-topic results will be very noisy; consider merging topics.")

    out_dir = ROOT / "data/processed"; out_dir.mkdir(parents=True, exist_ok=True)
    (ROOT / "models").mkdir(exist_ok=True); (ROOT / "results").mkdir(exist_ok=True)
    df[["tweet_id", "text", "label", "topic", "topic_score", "topic_margin"]].to_csv(
        out_dir / "tweets_with_topics.csv", index=False)
    with open(ROOT / "models/topic_definitions.json", "w", encoding="utf-8") as f:
        json.dump({"model": MODEL_NAME, "topics": TOPICS}, f, indent=2, ensure_ascii=False)

    # tweets for a manual accuracy check of the topic labels (fill 'your_topic', compare in your report)
    chk = pd.concat([g.sample(min(20, len(g)), random_state=SEED) for _, g in df.groupby("topic")])
    chk = chk[["tweet_id", "text", "topic"]].rename(columns={"topic": "model_topic"})
    chk["your_topic"] = ""
    chk.to_csv(ROOT / "results/topic_check_sample.csv", index=False)

    print("\n=== EYEBALL CHECK: 5 random tweets per topic ===")
    for t, g in df.groupby("topic"):
        print(f"\n--- {t} ---")
        for s in g.sample(min(5, len(g)), random_state=SEED)["text"]:
            print("   *", s[:120].replace("\n", " "))
    print("\nSaved -> data/processed/tweets_with_topics.csv, models/topic_definitions.json,")
    print("         results/topic_check_sample.csv (hand-check this: it gives you a topic-labelling accuracy)")


if __name__ == "__main__":
    main()