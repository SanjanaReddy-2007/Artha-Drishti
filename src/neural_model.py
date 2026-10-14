"""Step 5 (M6-M8) - GloVe + BiGRU sarcasm classifier (PyTorch).

Embeddings: FROZEN by default (small dataset); --trainable fine-tunes them (ablation).
Get GloVe: https://nlp.stanford.edu/projects/glove/  (glove.twitter.27B.100d.txt recommended
for tweets, or glove.6B.100d.txt). Pass its path with --glove. Without it, embeddings are
random (a warning is printed) - results will be worse.

Model selection: best epoch by val F1 (sarcastic class), early stopping.

Usage:
  python src/neural_model.py --glove data/glove.twitter.27B.100d.txt
  python src/neural_model.py --glove ... --trainable --tag BiGRU_trainable
"""
import argparse
import json
import re
from collections import Counter
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from config import SPLIT_CSV, RESULTS, MODELS, SEED

PAD, UNK = 0, 1
MAXLEN = 60


def toks(s):
    return str(s).split()


def build_vocab(texts, min_freq=2):
    c = Counter(t for s in texts for t in toks(s))
    itos = ["<pad>", "<unk>"] + [w for w, f in c.items() if f >= min_freq]
    return {w: i for i, w in enumerate(itos)}


def encode(texts, vocab):
    X = np.zeros((len(texts), MAXLEN), dtype=np.int64)
    for i, s in enumerate(texts):
        ids = [vocab.get(t, UNK) for t in toks(s)][:MAXLEN]
        X[i, :len(ids)] = ids
    return torch.from_numpy(X)


def load_glove(path, vocab, dim):
    emb = np.random.normal(0, 0.1, (len(vocab), dim)).astype("float32")
    emb[PAD] = 0
    if path is None:
        print("WARNING: no --glove given, using random embeddings")
        return emb
    hit = 0
    with open(path, encoding="utf8") as f:
        for line in f:
            w, *vals = line.rstrip().split(" ")
            if w in vocab and len(vals) == dim:
                emb[vocab[w]] = np.asarray(vals, dtype="float32")
                hit += 1
    print(f"GloVe coverage: {hit}/{len(vocab)} vocab words ({hit/len(vocab):.0%})")
    return emb


class BiGRU(nn.Module):
    def __init__(self, emb, trainable, hidden=64, dropout=0.4):
        super().__init__()
        self.emb = nn.Embedding.from_pretrained(torch.tensor(emb), freeze=not trainable, padding_idx=PAD)
        self.gru = nn.GRU(emb.shape[1], hidden, batch_first=True, bidirectional=True)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden * 4, 1)

    def forward(self, x):
        mask = (x != PAD).unsqueeze(-1)
        h, _ = self.gru(self.drop(self.emb(x)))
        mx = h.masked_fill(~mask, -1e9).max(1).values          # max-pool
        mean = (h * mask).sum(1) / mask.sum(1).clamp(min=1)    # mean-pool
        return self.fc(self.drop(torch.cat([mx, mean], 1))).squeeze(-1)


def predict_proba(model, X, bs=256):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X), bs):
            out.append(torch.sigmoid(model(X[i:i + bs])))
    return torch.cat(out).numpy()


def train(df, glove=None, dim=100, trainable=False, epochs=30, patience=5, frac=1.0,
          balanced=False, tag="BiGRU", save=True, verbose=True):
    torch.manual_seed(SEED); np.random.seed(SEED)
    tr, va = df[df.split == "train"], df[df.split == "val"]
    if frac < 1.0:
        tr = tr.sample(frac=frac, random_state=SEED)
    vocab = build_vocab(tr["text_clean"].fillna(""))
    emb = load_glove(glove, vocab, dim)
    model = BiGRU(emb, trainable)
    Xtr, ytr = encode(tr["text_clean"].fillna(""), vocab), torch.tensor(tr["label"].values, dtype=torch.float32)
    Xva, yva = encode(va["text_clean"].fillna(""), vocab), va["label"].values
    pw = torch.tensor([(ytr == 0).sum() / max((ytr == 1).sum(), 1)]) if balanced else None
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pw)
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=2e-3)
    best, best_state, bad = -1, None, 0
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(Xtr))
        for i in range(0, len(perm), 64):
            idx = perm[i:i + 64]
            opt.zero_grad()
            loss_fn(model(Xtr[idx]), ytr[idx]).backward()
            opt.step()
        f1 = f1_score(yva, predict_proba(model, Xva) >= 0.5, zero_division=0)
        if verbose:
            print(f"epoch {ep+1:02d} val_f1={f1:.3f}")
        if f1 > best:
            best, bad = f1, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    if save:
        torch.save({"state": best_state, "vocab": vocab, "emb_shape": emb.shape,
                    "trainable": trainable}, MODELS / f"{tag}.pt")
    return model, vocab, best


def main(a):
    df = pd.read_csv(SPLIT_CSV)
    model, vocab, best = train(df, a.glove, a.dim, a.trainable, a.epochs, a.patience,
                               balanced=a.balanced, tag=a.tag)
    print(f"best val F1 = {best:.3f}")
    for split in ("val", "test"):
        part = df[df.split == split]
        p = predict_proba(model, encode(part["text_clean"].fillna(""), vocab))
        part[["id", "label", "topic"]].assign(pred=(p >= 0.5).astype(int), proba=p) \
            .to_csv(RESULTS / f"preds_{a.tag}_{split}.csv", index=False)
    json.dump(vars(a), open(RESULTS / f"{a.tag}_args.json", "w"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--glove", default=None)
    ap.add_argument("--dim", type=int, default=100)
    ap.add_argument("--trainable", action="store_true")
    ap.add_argument("--balanced", action="store_true")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--tag", default="BiGRU")
    main(ap.parse_args())
