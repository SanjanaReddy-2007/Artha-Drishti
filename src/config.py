"""Central configuration. All paths are derived from the project root (no absolute paths)."""
from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SEED = 42
RAW_FILENAME = "train.En.csv"
TEXT_COLUMNS = ("tweet", "text", "Tweet", "Text")
LABEL_COLUMNS = ("sarcastic", "label", "Sarcastic")

# Input limits for the website (a tweet is at most 280 characters).
TWEET_CHARS = 280
MAX_INPUT_CHARS = 1000

# Default-model selection rule (validation data only, never the test set).
SELECTION_METRIC = "pr_auc"          # threshold-independent
TIE_TOLERANCE = 0.01                 # models within 0.01 of the best count as tied
SIMPLICITY_ORDER = [                 # simplest first; used to break ties
    "NaiveBayes", "LogReg", "LogReg_balanced", "SVM", "SVM_balanced",
    "DecisionTree", "KNN", "RandomForest", "GradientBoosting", "XGBoost", "MLP",
]

CLASS_NAMES = ["Not Sarcastic", "Sarcastic"]   # label 0 / label 1; positive class = 1


class ProjectPaths:
    """All project directories/files, relative to one root (overridable for tests)."""

    def __init__(self, root: Path | str = PROJECT_ROOT):
        self.root = Path(root)
        self.data = self.root / "data"
        self.models = self.root / "models"
        self.results = self.root / "results"
        self.logs = self.root / "logs"
        self.raw_file = self.data / "raw" / RAW_FILENAME
        self.processed_file = self.data / "processed.csv"
        self.data_report = self.results / "data_report.json"
        self.split_file = self.results / "split_indices.json"
        self.model_results = self.results / "models"
        self.predictions = self.results / "predictions"
        self.metrics_csv = self.results / "metrics.csv"
        self.baselines_csv = self.results / "baselines.csv"
        self.status_json = self.results / "status.json"
        self.default_model_json = self.results / "default_model.json"
        self.run_metadata = self.results / "run_metadata.json"

    def ensure(self) -> "ProjectPaths":
        for p in (self.data, self.models, self.results, self.logs,
                  self.model_results, self.predictions):
            p.mkdir(parents=True, exist_ok=True)
        self.raw_file.parent.mkdir(parents=True, exist_ok=True)
        return self
