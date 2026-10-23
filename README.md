# Artha-Drishti - Machine Learning-Based Sarcasm Detection in English Tweets

Owner: Are Sanjana Reddy (24BDS0187) | Task: binary sarcasm detection (Sarcastic / Not Sarcastic) | Interface: Streamlit

This build is a **text-only, topic-independent baseline**: ten scikit-learn candidates (plus XGBoost if installed) are trained
on the same data, split and protocol, saved to disk, and shown in a website that reads only **real saved results**.
Nothing in the repository contains pre-filled metrics; until you train, the dashboard says "pending".

## Status of this build (read this first)

| Item | Status |
|---|---|
| Code: data validation, cleaning, leakage-safe split, 10 models, evaluation, bootstrap CIs, predictor, 5-page website | **Written** |
| 22 unit/integration tests (1 skipped: needs the `emoji` package) | **Passed in my sandbox on SYNTHETIC data only** |
| Website page code executed with stubbed Streamlit/Altair (catches my own bugs, not Streamlit-API misuse) | **Passed in my sandbox** |
| Run on the **real** iSarcasmEval file | **NOT DONE** - the dataset was not available to me. Run the commands below. |
| Website launched in a real browser | **NOT DONE** - Streamlit was not installable in my sandbox. Please verify with `streamlit run demo\app.py`. |
| XGBoost | **Untested** here (not installed); recorded as "skipped" if it is missing |
| GloVe + BiGRU | **Pending** (future phase; shown as pending on the dashboard) |
| Literature review / novelty check | **NOT DONE** - see `docs/LITERATURE_REVIEW_STATUS.md`. Do not claim novelty. |
| Git branch / push | **NOT DONE** - I had no access to your repository. See `MERGE_GUIDE.md`. |
| Calibration of probabilities | **Not done**; scores are labelled "uncalibrated" |

Previously recorded numbers (3,456 tweets; SVM validation F1 0.360; ...) are historical. This code recomputes everything;
`python -m src.prepare_data` prints the real raw row count (this also settles whether the file has 3,456 or 3,468 rows).

## Setup (Windows PowerShell, Python 3.12)

```powershell
cd C:\Users\User\Downloads\Artha-Drishti
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install --default-timeout=120 --retries 10 -r requirements.txt
pip install --default-timeout=120 --retries 10 -r requirements-optional.txt   # optional: xgboost
```

## Train (one stage at a time; check each output before the next)

```powershell
# 0. put the iSarcasmEval file at data\train.En.csv
mkdir logs -ErrorAction SilentlyContinue
python -m src.prepare_data 2>&1 | Tee-Object logs\prepare_data.txt     # validates + cleans; read results\data_report.json
python -m src.split 2>&1 | Tee-Object logs\split.txt                   # saves results\split_indices.json (stratified on label)
python -m src.train --models SVM,LogReg --limit 300 --cv-folds 3       # SMOKE TEST ONLY (flagged in results, shown as a red banner)
python -m src.train 2>&1 | Tee-Object logs\train.txt                   # the real run (all models)
python -m unittest discover -s tests -t .                              # tests
```
Re-using the split from the earlier project (same train/val/test membership):
`python -m src.split --import-existing --train-csv <train.csv> --val-csv <val.csv> --test-csv <test.csv> --text-col <col> --force`.
A smoke test overwrites `results\` and `models\`; delete them (or re-run the full training) before reporting anything.
Options: `--features word_char` (adds character n-grams), `--n-jobs 4`, `--bootstrap 1000`, `--models A,B`.
Model names: SVM, SVM_balanced, LogReg, LogReg_balanced, NaiveBayes, DecisionTree, RandomForest, KNN, GradientBoosting, MLP, XGBoost.

## Launch the website

```powershell
streamlit run demo\app.py
```
Pages: Home, Sarcasm Detector, Model Comparison, Model Performance, About the Project.
If no model has been trained, pages show instructions instead of numbers. Models are loaded once from `models\*.joblib`; nothing is retrained on a request.

## Layout

```
src/config.py        paths (relative to project root), constants, default-model rule
src/preprocess.py    stateless text cleaning (+ emoji -> words)
src/data.py          load/validate/clean/de-duplicate -> data/processed.csv, results/data_report.json
src/split.py         stratified split saved as row_id lists; leakage verification; import of an older split
src/features.py      TF-IDF configs (vocabulary learned on training data only, inside the Pipeline)
src/models.py        model registry + grids (Maximum Entropy == LogReg, not duplicated)
src/evaluate.py      metrics, thresholds, baselines, bootstrap CIs, paired bootstrap, timing
src/train.py         tune on train -> validation (threshold + default model) -> test; saves everything
src/results_io.py    reads saved results (no Streamlit) ; src/predict.py  prediction service
demo/                Streamlit app (app.py, ui.py, views/)
tests/               unittest suite (synthetic data, tmp dirs)
docs/PROTOCOL.md     exact evaluation protocol ; docs/LITERATURE_REVIEW_*  (pending)
legacy_topic_aware/  place for your old topic-aware code (see its README)
results/ models/ logs/   generated artefacts (models/*.joblib are git-ignored)
```

## Saved artefacts (all generated by `src.train`)

`results/metrics.csv` (model x stage x threshold), `results/baselines.csv`, `results/status.json` (completed / failed / skipped / pending),
`results/default_model.json`, `results/run_metadata.json` (versions, seed, git commit, smoke-test flag), `results/models/<name>.json`
(config, params, thresholds, reports, bootstrap CIs, threshold table), `results/predictions/<name>_{validation,test}.csv` (scores used for PR curves),
`models/<name>.joblib`, `logs/train_<time>.log`. Pickled scikit-learn models require the same scikit-learn version to load.

## Known limitations

* Small dataset; the ~350-tweet validation and test sets give wide confidence intervals - treat F1 gaps below ~0.05 as noise.
* Thresholds tuned on a small validation set are noisy. Test results at the validation-selected threshold are honest but can still be unlucky.
* Scores are uncalibrated. Text-only; no conversational context. Single English dataset.
* Default-model selection uses validation PR-AUC (threshold-independent) with a tie rule; other rules are possible and would be a documented change.
* The class-weighted variants can be chosen as default if they win on validation; they are reported as a separate experiment group in the dashboard.

## Suggested next experiment (to confirm with your real results)

Once real results exist: (1) check whether any model beats the always-sarcastic F1 baseline and the PR-AUC random level with non-overlapping
bootstrap CIs; (2) compare `word` vs `word_char` features with the paired bootstrap; (3) add calibration; (4) then add GloVe + BiGRU
under the same protocol; (5) only then re-attach topic tags for the topic-aware extension.
