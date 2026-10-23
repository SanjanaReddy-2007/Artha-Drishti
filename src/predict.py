"""Prediction service used by the website. Loads SAVED pipelines; never trains."""
from __future__ import annotations

import re
from typing import Callable, Sequence
from dataclasses import asdict, dataclass

import joblib
import numpy as np

from src import results_io as rio
from src.config import BATCH_CHUNK_SIZE, MAX_INPUT_CHARS, TWEET_CHARS, ProjectPaths
from src.models import SCORE_LABELS, get_scores
from src.preprocess import EMOJI_AVAILABLE, clean_text
from src.evaluate import decide


class InputError(ValueError):
    """Invalid user input (shown to the user as a message, not a crash)."""


@dataclass
class Prediction:
    label: str
    is_sarcastic: bool
    score: float
    score_meaning: str
    threshold: float
    threshold_type: str
    model_name: str
    model_display: str
    cleaned_text: str
    warnings: list


class Predictor:
    def __init__(self, paths: ProjectPaths | None = None):
        self.paths = paths or ProjectPaths()
        self._pipes: dict = {}

    def available_models(self) -> list[str]:
        return rio.completed_models(self.paths)

    def default_model_name(self) -> str | None:
        d = rio.load_default_model(self.paths)
        return d["name"] if d and d["name"] in self.available_models() else None

    def _pipeline(self, name: str):
        if name not in self._pipes:
            self._pipes[name] = joblib.load(self.paths.models / f"{name}.joblib")
        return self._pipes[name]

    @staticmethod
    def validate(text: str) -> str:
        if text is None or not text.strip():
            raise InputError("Please enter a tweet to analyse.")
        if len(text) > MAX_INPUT_CHARS:
            raise InputError(f"Input is too long ({len(text)} characters; limit {MAX_INPUT_CHARS}).")
        return text

    def predict(self, text: str, model_name: str | None = None, threshold_type: str = "validation_selected") -> dict:
        text = self.validate(text)
        name = model_name or self.default_model_name()
        if name is None or name not in self.available_models():
            raise FileNotFoundError("No trained model found. Run: python -m src.train")
        if threshold_type not in ("default", "validation_selected"):
            raise ValueError("threshold_type must be 'default' or 'validation_selected'")
        cleaned = clean_text(text)
        if not re.search(r"[a-z]", cleaned):
            raise InputError("The text contains no readable English words after cleaning.")
        detail = rio.load_model_detail(self.paths, name)
        th = detail["thresholds"][threshold_type]
        score = float(get_scores(self._pipeline(name), [cleaned])[0])
        is_s = bool(decide([score], th["value"], th["strict"])[0])
        warns = []
        if len(text) > TWEET_CHARS:
            warns.append(f"Longer than a tweet ({TWEET_CHARS} characters); the models were trained on tweets.")
        meta = rio.load_run_metadata(self.paths) or {}
        if meta.get("smoke_test"):
            warns.append("These models come from a SMOKE TEST run (reduced data). Do not trust the output.")
        if meta.get("emoji_conversion_available") and not EMOJI_AVAILABLE:
            warns.append("Emoji conversion was used in training but the 'emoji' package is missing here.")
        return asdict(Prediction("Sarcastic" if is_s else "Not Sarcastic", is_s, score,
                                 SCORE_LABELS[detail["score_type"]], th["value"], threshold_type, name,
                                 detail["display"], cleaned, warns))

    def predict_batch(self, texts: Sequence[str], model_name: str | None = None,
                      threshold_type: str = "validation_selected",
                      progress: Callable[[float], None] | None = None,
                      chunk_size: int = BATCH_CHUNK_SIZE) -> dict:
        """Score many RAW tweets in one call, using exactly the same steps as predict():
        clean_text -> saved pipeline -> get_scores -> decide(saved threshold).
        Returns arrays aligned with `texts`: labels, is_sarcastic, scores, plus the threshold info.
        Callers must pass only valid texts (see src.batch.prepare_rows); invalid ones raise InputError
        here, exactly as predict() would. `progress(fraction)` is called after each chunk."""
        name = model_name or self.default_model_name()
        if name is None or name not in self.available_models():
            raise FileNotFoundError("No trained model found. Run: python -m src.train")
        if threshold_type not in ("default", "validation_selected"):
            raise ValueError("threshold_type must be 'default' or 'validation_selected'")
        cleaned = []
        for i, t in enumerate(texts):
            try:
                t = self.validate(t)
                c = clean_text(t)
                if not re.search(r"[a-z]", c):
                    raise InputError("The text contains no readable English words after cleaning.")
            except InputError as e:
                raise InputError(f"Item {i}: {e}") from None
            cleaned.append(c)
        detail = rio.load_model_detail(self.paths, name)
        th = detail["thresholds"][threshold_type]
        pipe = self._pipeline(name)
        scores = np.empty(len(cleaned), dtype=float)
        step = max(1, int(chunk_size))
        for start in range(0, len(cleaned), step):
            part = cleaned[start:start + step]
            scores[start:start + len(part)] = get_scores(pipe, part)
            if progress:
                progress(min(1.0, (start + len(part)) / max(1, len(cleaned))))
        is_s = decide(scores, th["value"], th["strict"]).astype(bool) if len(scores) else np.zeros(0, dtype=bool)
        return {"labels": ["Sarcastic" if x else "Not Sarcastic" for x in is_s],
                "is_sarcastic": is_s, "scores": scores, "cleaned": cleaned,
                "threshold": float(th["value"]), "threshold_type": threshold_type,
                "model_name": name, "model_display": detail["display"],
                "score_meaning": SCORE_LABELS[detail["score_type"]]}
