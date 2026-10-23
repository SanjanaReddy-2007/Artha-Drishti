# Batch Detector

Page: `demo/views/batch.py`. Logic (no Streamlit): `src/batch.py`. Vectorised prediction: `Predictor.predict_batch` in `src/predict.py`.

## What it does
Classifies every tweet of an uploaded CSV/TXT with a saved model and shows counts, percentages, a per-tweet table and downloads.
It is a convenience interface to the existing models; it does **not** change the split, the saved models, any metric in `results\`,
or `docs/PROTOCOL.md`, and it is not an evaluation.

## Same path as the single-tweet page
`predict_batch` repeats the steps of `Predictor.predict`: `validate` (empty / over 1,000 characters) -> `clean_text`
(same preprocessing, emoji conversion included) -> "contains a letter" check -> saved pipeline from `models\<name>.joblib` ->
`get_scores` -> `decide(score, threshold, strict)` using the threshold stored in `results\models\<name>.json`. Labels are
"Sarcastic" / "Not Sarcastic". `Predictor.predict` itself is unchanged. The default model comes from
`results\default_model.json`, not from code. Models are loaded once per process (`st.cache_resource` via `ui.get_predictor`).
Scores are scored in chunks of `BATCH_CHUNK_SIZE` rows (progress bar); the results are the same as one single call.

## Row handling
| Case | Behaviour |
|---|---|
| Blank / NaN / whitespace tweet | skipped, counted, listed |
| No readable English words after cleaning (e.g. only digits or symbols) | skipped (the single-tweet page raises the same error) |
| More than 1,000 characters | skipped (single-tweet page rejects it); 281-1,000 characters are scored with a note |
| Non-string cell (number, bool) | converted to text, then checked like any other |
| Duplicate tweets | all rows kept in the output; each unique text is scored once |
| Over `BATCH_MAX_ROWS` rows or `BATCH_MAX_MB` MB | whole file rejected with a message |
| Empty file, header only, wrong column, unreadable encoding | friendly message, no stack trace |
| Ragged lines (e.g. unquoted comma in a tweet) | reported; extra fields dropped, missing ones empty |

The count identity `Sarcastic + Not Sarcastic + skipped = total rows` always holds.

## Leakage guard
Only the selected tweet column is passed to `predict_batch`. The optional label column is used only after prediction, for the
ad-hoc precision / recall / F1 / confusion matrix, which carries the warning that it is not the official test result (and
will be optimistic if the file contains tweets the models were trained on).

## Tests
`tests/test_batch.py` (synthetic data): encodings, delimiters, no-header, limits, column detection, blank/duplicate/non-string
handling, batch-vs-single consistency (same label and score), count identity, output columns, BOM, labelled check, model comparison.
