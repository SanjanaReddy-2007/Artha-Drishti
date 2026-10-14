"""Step 1b - zero-shot topic tagging (3 topics).

Guarantees ONE output row per input row (this fixes the 3,456 -> 3,455 loss:
empty / failed texts get topic "Unclear" instead of being dropped).

Usage:
  python src/topic_tagging.py                    # BART-MNLI zero-shot (needs transformers)
  python src/topic_tagging.py --backend keyword  # offline keyword fallback (testing only)
  python src/topic_tagging.py --low_conf 0.45    # flag (not drop) low-confidence tags
"""
import argparse
import re
import numpy as np
import pandas as pd
from config import RAW_CSV, TAGGED_CSV, TOPICS, TOPIC_NAMES, ZERO_SHOT_MODEL, HYPOTHESIS_TEMPLATE

KEYWORDS = {  # tiny lexicon, ONLY for offline smoke tests
    "Entertainment & Sports": r"\b(movie|film|song|music|album|game|team|win|match|season|concert|netflix|show|actor|football|cricket|nba)\b",
    "Politics & Society": r"\b(trump|biden|vote|election|government|president|police|law|tax|war|news|protest|congress|brexit)\b",
}


def keyword_scores(texts):
    rows = []
    for t in texts:
        s = [0.34, 0.33, 0.33]
        for i, name in enumerate(TOPIC_NAMES):
            if name in KEYWORDS and re.search(KEYWORDS[name], t.lower()):
                s[i] += 1.0
        s = np.array(s) / sum(s)
        rows.append(s)
    return np.array(rows)


def zero_shot_scores(texts, batch_size=16, model=ZERO_SHOT_MODEL):
    from transformers import pipeline  # imported lazily
    import torch
    device = 0 if torch.cuda.is_available() else -1
    clf = pipeline("zero-shot-classification", model=model, device=device)
    labels = list(TOPICS.values())
    out = np.zeros((len(texts), len(labels)))
    for start in range(0, len(texts), batch_size):
        chunk = texts[start:start + batch_size]
        res = clf(chunk, candidate_labels=labels, hypothesis_template=HYPOTHESIS_TEMPLATE,
                  multi_label=False)
        if isinstance(res, dict):
            res = [res]
        for j, r in enumerate(res):
            for lab, sc in zip(r["labels"], r["scores"]):
                out[start + j, labels.index(lab)] = sc
        if (start // batch_size) % 20 == 0:
            print(f"  tagged {min(start + batch_size, len(texts))}/{len(texts)}")
    return out


def main(backend, low_conf):
    df = pd.read_csv(RAW_CSV)
    n = len(df)
    texts = df["text"].fillna("").astype(str).tolist()
    valid = np.array([len(t.strip()) > 0 for t in texts])

    scores = np.full((n, len(TOPIC_NAMES)), np.nan)
    idx = np.where(valid)[0]
    vt = [texts[i] for i in idx]
    scores[idx] = keyword_scores(vt) if backend == "keyword" else zero_shot_scores(vt)

    for k, name in enumerate(TOPIC_NAMES):
        df[f"score_{k}"] = scores[:, k]
    sorted_s = np.sort(np.nan_to_num(scores, nan=0.0), axis=1)
    df["topic_score"] = sorted_s[:, -1]
    df["topic_margin"] = sorted_s[:, -1] - sorted_s[:, -2]
    top = np.nanargmax(np.nan_to_num(scores, nan=-1), axis=1)
    df["topic"] = [TOPIC_NAMES[t] for t in top]
    df.loc[~valid, "topic"] = "Unclear"
    df["low_conf"] = (df["topic_score"] < low_conf) | (~valid)

    assert len(df) == n, "Row count changed!"
    df.to_csv(TAGGED_CSV, index=False)
    print(f"Saved {len(df)} rows -> {TAGGED_CSV}")
    print(df.groupby("topic")["label"].agg(n="size", sarcastic="sum",
          share=lambda s: round(s.mean(), 3)))
    print(f"Low-confidence rows (<{low_conf}): {int(df['low_conf'].sum())}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["zeroshot", "keyword"], default="zeroshot")
    ap.add_argument("--low_conf", type=float, default=0.45)
    a = ap.parse_args()
    main(a.backend, a.low_conf)
