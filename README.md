# Artha-Drishti - topic-aware sarcasm detection

Dataset: iSarcasmEval English training file (`train.En.csv`).
Pipeline: zero-shot topic tag (3 topics) -> preprocessing -> stratified 80/10/10 split ->
5 classical models (TF-IDF) + 1 BiGRU (GloVe) -> overall & per-topic evaluation ->
topic x classifier recommender -> Streamlit demo.

Decisions: classes NOT balanced by default (`--balanced` flag available) - metric is F1 of the
sarcastic class; emojis converted to words; stratified 80/10/10 with 5-fold CV for tuning;
GloVe frozen by default (`--trainable` ablation); recommender tie-break = simplest model within 0.01 F1.

## Setup
```bash
pip install -r requirements.txt
```
Put `train.En.csv` anywhere; for the neural model download GloVe (e.g. `glove.twitter.27B.100d.txt`) into `data/`.

## Run (from repo root)
```bash
bash run_all.sh path/to/train.En.csv data/glove.twitter.27B.100d.txt
```
or stage by stage from `src/`:

| Stage | Command | Output |
|---|---|---|
| 1a load | `python load_dataset.py --input train.En.csv` | data/tweets_raw.csv |
| 1b tag | `python topic_tagging.py` | data/tweets_tagged.csv (never drops rows) |
| 1c missing row | `python find_missing_row.py` | prints rows lost / duplicated |
| 1d hand-check | `python make_judging_file.py` -> fill `my_topic` in data/judging_file.csv -> `python check_topic_quality.py` | Part A + Part B |
| 2-3 preprocess | `python preprocess.py --emoji convert` | data/tweets_clean.csv |
| 5 split | `python split.py` | data/tweets_split.csv |
| 9-11 classical | `python classical_models.py` | results/preds_*, models/*.joblib |
| 6-8 neural | `python neural_model.py --glove ...` (`--trainable --tag BiGRU_trainable` for ablation) | results/preds_BiGRU_* |
| 12 evaluate | `python evaluate.py`, `python learning_curves.py [--neural]` | results/*.csv, *.png |
| 13 recommender | `python recommender.py` | results/recommender.json |
| 14 demo | `streamlit run demo/app.py` | |

## Notes
* `topic_tagging.py --backend keyword` is an offline smoke-test tagger only - never report its numbers.
* All code was smoke-tested on synthetic data; the neural model, zero-shot tagger and Streamlit demo
  need `torch`, `transformers`, `streamlit` and were NOT executed in the build environment.
* Model selection uses the validation split; the test split is touched only by `evaluate.py`.
