# Artha-Drishti
Topic-Aware Classifier Recommendation for Sarcasm Detection on Twitter Data (Team 24, G1 slot)

## Folder structure
```
artha_drishti_project/
├── README.md
├── requirements.txt
├── src/
│   ├── load_dataset.py      # loads + standardises the raw dataset
│   └── topic_tagging.py     # zero-shot topic tagging with sentence embeddings
├── data/
│   ├── raw/                 # PUT YOUR ORIGINAL DATASET FILE HERE (never edit it)
│   └── processed/           # files created by the scripts
├── models/                  # saved topic definitions (and later, trained models)
└── results/                 # tables and charts (used in later steps)
```

## Setup
```
pip install -r requirements.txt
```

## Run (always from the project root folder)
1. Put your dataset in `data/raw/` (we use the iSarcasmEval English train file, `train.En.csv`).
2. Load it:
```
python src/load_dataset.py --input data/raw/train.En.csv --text-col tweet --label-col sarcastic --sarcastic-value 1
```
   Output: `data/processed/tweets_raw.csv`
3. Tag topics (zero-shot, needs internet on first run to download the ~90 MB model):
```
python src/topic_tagging.py
```
   Topics and their descriptions are defined at the top of `src/topic_tagging.py` (edit `TOPICS` to change them).
   Output: `data/processed/tweets_with_topics.csv`, `models/topic_definitions.json`, `results/topic_check_sample.csv`
4. Open `results/topic_check_sample.csv`, fill the `your_topic` column by hand, and count how often it matches `model_topic`.
   That agreement rate is your topic-labelling accuracy for the report.

## Using the Pislar GitHub repository (Ghosh & Veale data)
```
git clone https://github.com/MirunaPislar/Sarcasm-Detection.git
```
1. Open `Sarcasm-Detection/res/datasets/ghosh/` and copy the ORIGINAL train/test files (not the token/POS versions) into `data/raw/`.
2. Look at the format first:  `head -n 5 data/raw/train.txt`   and   `wc -l data/raw/*.txt`
   (Windows PowerShell: `Get-Content data/raw/train.txt -TotalCount 5`)
3. Run `src/load_dataset.py` with the matching options (see the examples at the top of that file).