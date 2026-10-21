"""Feature extraction. The vectoriser lives inside the sklearn Pipeline, so during
cross-validation and final fitting its vocabulary/IDF are learned from training data only."""
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion

FEATURE_CONFIGS = {
    "word": "TF-IDF, word 1-2 grams, min_df=2, sublinear tf, max 20,000 features",
    "word_char": "TF-IDF word 1-2 grams (max 20,000) + char_wb 2-5 grams (max 30,000), min_df=2, sublinear tf",
}


def build_vectorizer(kind: str = "word"):
    word = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, max_features=20000)
    if kind == "word":
        return word
    if kind == "word_char":
        char = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True, max_features=30000)
        return FeatureUnion([("word", word), ("char", char)])
    raise ValueError(f"Unknown feature config {kind!r}; choose from {list(FEATURE_CONFIGS)}")
