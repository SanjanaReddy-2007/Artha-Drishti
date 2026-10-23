"""SYNTHETIC data used ONLY by unit tests to check that the code runs. Never used for results."""
import random

import pandas as pd

SARC = ["yeah", "right", "sure", "totally", "great", "wow", "love", "just", "perfect", "thanks"]
NEUT = ["today", "match", "movie", "vote", "city", "coffee", "music", "weather", "team", "news"]


def make_raw_csv(path, n=600, seed=0):
    rnd = random.Random(seed)
    rows = []
    for i in range(n):
        lab = int(rnd.random() < 0.28)
        pool = (SARC * 2 + NEUT) if lab else (NEUT * 2 + SARC)
        text = " ".join(rnd.choice(pool) for _ in range(8)) + f" uniq{i}"
        rows.append({"tweet": text, "sarcastic": lab, "rephrase": "LEAKY " + text if lab else None})
    rows[3]["tweet"] = "  "                                   # blank text
    rows[5]["tweet"] = rows[4]["tweet"]; rows[5]["sarcastic"] = rows[4]["sarcastic"]      # exact duplicate
    rows[7]["tweet"] = rows[6]["tweet"]; rows[7]["sarcastic"] = 1 - rows[6]["sarcastic"]  # conflicting duplicate
    rows[8]["tweet"] = "@someone check https://t.co/abc #Wow so great"
    pd.DataFrame(rows).to_csv(path, index=False)
    return n
