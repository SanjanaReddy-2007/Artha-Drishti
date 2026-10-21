# Evaluation protocol (topic-independent baseline)

* **Task:** binary classification of English tweets. Positive class = 1 = *Sarcastic*.
* **Data:** iSarcasmEval English training file. Only `tweet` text and `sarcastic` label are used.
  Other columns (e.g. `rephrase`, sarcasm-type flags) are excluded: they exist only for sarcastic tweets or encode the label.
* **Cleaning:** blank texts removed; text normalised (see About page); duplicate cleaned texts removed, and duplicate
  groups with conflicting labels dropped. Counts are recorded in `results/data_report.json`.
* **Split:** stratified on label, 80/10/10, seed 42, saved as `row_id` lists (`results/split_indices.json`).
  `verify_split` fails on any row overlap or duplicate-text overlap. The file is never overwritten without `--force`.
  `python -m src.split --import-existing ...` can reuse the membership of an earlier split.
* **Fitting:** vectoriser and classifier are one scikit-learn Pipeline fitted on the training set only
  (so vocabulary and IDF never see validation/test text, also inside cross-validation).
* **Tuning:** 5-fold stratified CV on the training set, scoring = sarcastic-class F1; small grids.
* **Thresholds:** (a) default: 0.5 for probabilities, 0 for SVM decision scores; (b) validation-selected: the score cut-off
  that maximises sarcastic F1 on the validation set. Never chosen on test labels.
* **Default model:** highest *validation* PR-AUC; models within 0.01 are tied and the simplest wins (`SIMPLICITY_ORDER`).
* **Test set:** scored once per model, after thresholds and the default model are fixed.
* **Metrics:** accuracy; sarcastic precision/recall/F1; macro and weighted F1; PR-AUC (average precision); ROC-AUC; confusion
  matrix; tuning time, final fit time, inference time per tweet (including TF-IDF).
* **Uncertainty:** percentile bootstrap 95% CIs on test (1000 resamples of test tweets); paired bootstrap differences against the
  default model on the *same* resamples. If a CI includes 0 the difference is not conclusive.
* **Baselines:** always-"Sarcastic" and majority-class ("Not Sarcastic"), plus the sarcastic share (PR-AUC of random scores).
* **Class imbalance:** `class_weight="balanced"` for SVM and Logistic Regression is a separate, labelled experiment.
* **Maximum Entropy** = logistic regression; implemented once.
* **Scores are not calibrated probabilities.** Logistic Regression/NB/RF/etc. "probabilities" are raw model outputs; the SVM gives a margin.
  Calibration is future work.
