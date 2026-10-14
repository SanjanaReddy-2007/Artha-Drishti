"""Shared paths, constants and topic definitions for Artha-Drishti."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"
MODELS = ROOT / "models"
for _p in (DATA, RESULTS, MODELS):
    _p.mkdir(exist_ok=True)

SEED = 42

# Files produced by each stage
RAW_CSV = DATA / "tweets_raw.csv"          # id, text, label
TAGGED_CSV = DATA / "tweets_tagged.csv"    # + topic, topic_score, topic_margin
CLEAN_CSV = DATA / "tweets_clean.csv"      # + text_clean
SPLIT_CSV = DATA / "tweets_split.csv"      # + split (train/val/test)

# topic short name -> descriptive label shown to the zero-shot model.
# Descriptive labels usually work better than 1-word labels with NLI models.
TOPICS = {
    "Everyday Life": "everyday life, family, work, school, food, health or personal feelings",
    "Entertainment & Sports": "entertainment, movies, music, celebrities, games or sports",
    "Politics & Society": "politics, government, news, social issues or public events",
}
TOPIC_NAMES = list(TOPICS.keys())

ZERO_SHOT_MODEL = "facebook/bart-large-mnli"
HYPOTHESIS_TEMPLATE = "This tweet is about {}."

# Model names, ordered simple -> complex (used by the recommender tie-break)
MODEL_ORDER = ["NaiveBayes", "LogReg", "SVM", "KNN", "RandomForest", "BiGRU"]
