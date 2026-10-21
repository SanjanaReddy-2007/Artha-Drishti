import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

from src import evaluate as ev
from src import results_io as rio
from src import train as trainmod
from src.config import ProjectPaths
from src.data import build_processed
from src.predict import InputError, Predictor
from src.preprocess import EMOJI_AVAILABLE, clean_text
from src.split import import_existing, make_split, verify_split
from tests.synthetic import make_raw_csv


class TestPreprocess(unittest.TestCase):
    def test_tokens(self):
        self.assertEqual(clean_text("@bob look https://x.co/a #Great &amp; fun"), "@user look url great & fun")

    def test_empty_and_nan(self):
        self.assertEqual(clean_text(None), ""); self.assertEqual(clean_text(float("nan")), "")

    @unittest.skipUnless(EMOJI_AVAILABLE, "emoji package not installed")
    def test_emoji_to_words(self):
        out = clean_text("so funny \U0001F602")
        self.assertNotIn("\U0001F602", out); self.assertIn("tears", out)


class TestEvaluate(unittest.TestCase):
    y = np.array([1, 1, 0, 0, 1, 0]); pred = np.array([1, 0, 0, 1, 1, 0]); s = np.array([.9, .4, .2, .6, .8, .1])

    def test_hand_computed_metrics(self):
        m = ev.compute_metrics(self.y, self.pred, self.s)
        self.assertEqual((m["tp"], m["fp"], m["fn"], m["tn"]), (2, 1, 1, 2))
        for k in ("precision", "recall", "f1"):
            self.assertAlmostEqual(m[k], 2 / 3)
        self.assertAlmostEqual(m["accuracy"], 4 / 6)

    def test_all_sarcastic_baseline(self):
        b = ev.baselines(self.y)["all_sarcastic"]
        self.assertAlmostEqual(b["recall"], 1.0); self.assertAlmostEqual(b["precision"], 0.5)
        self.assertAlmostEqual(b["f1"], 2 / 3)
        self.assertEqual(ev.baselines(self.y)["majority_class"]["f1"], 0.0)

    def test_decide_strictness(self):
        self.assertEqual(ev.decide([0.5], 0.5, True)[0], 0); self.assertEqual(ev.decide([0.5], 0.5, False)[0], 1)

    def test_bootstrap_and_paired(self):
        ci = ev.bootstrap_ci(self.y, self.pred, self.s, n_boot=200)
        self.assertLessEqual(ci["f1"]["low"], ci["f1"]["high"])
        same = ev.paired_bootstrap(self.y, self.pred, self.s, self.pred, self.s, n_boot=100)
        self.assertEqual(same["f1_diff"]["low"], 0.0); self.assertEqual(same["f1_diff"]["high"], 0.0)

    def test_threshold_uses_given_labels_only(self):
        thr, f1 = ev.select_threshold(self.y, self.s)
        self.assertGreaterEqual(f1, 0.8)


class TestSelection(unittest.TestCase):
    def test_simplest_within_tolerance_wins(self):
        r = {"SVM": {"pr_auc": .50}, "NaiveBayes": {"pr_auc": .495}, "RandomForest": {"pr_auc": .505}}
        self.assertEqual(trainmod.select_default(r)["name"], "NaiveBayes")

    def test_clear_winner(self):
        r = {"SVM": {"pr_auc": .60}, "NaiveBayes": {"pr_auc": .45}}
        self.assertEqual(trainmod.select_default(r)["name"], "SVM")


class TestDataAndSplit(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.paths = ProjectPaths(self.tmp.name).ensure()
        make_raw_csv(self.paths.raw_file, 600)
        self.report = build_processed(self.paths, convert_emoji=EMOJI_AVAILABLE)
        self.df = pd.read_csv(self.paths.processed_file)

    def tearDown(self): self.tmp.cleanup()

    def test_cleaning_report(self):
        r = self.report
        self.assertEqual(r["n_rows_raw"], 600); self.assertEqual(r["n_missing_or_blank_text"], 1)
        self.assertEqual(r["n_duplicate_texts_dropped"], 1); self.assertEqual(r["n_duplicate_groups_conflicting_labels_dropped"], 1)
        self.assertIn("rephrase", r["columns_ignored_to_prevent_leakage"])
        self.assertNotIn("rephrase", self.df.columns)
        self.assertEqual(r["n_rows_final"], len(self.df)); self.assertTrue(self.df.text_clean.is_unique)

    def test_split_stratified_disjoint_reproducible(self):
        s1, s2 = make_split(self.df, 42), make_split(self.df, 42)
        self.assertEqual(s1["test_ids"], s2["test_ids"])
        summ = verify_split(self.df, s1)
        self.assertEqual(sum(v["n"] for v in summ.values()), len(self.df))
        shares = [v["sarcastic_share"] for v in summ.values()]
        self.assertLess(max(shares) - min(shares), 0.03)
        self.assertAlmostEqual(summ["test"]["n"] / len(self.df), 0.1, delta=0.01)

    def test_overlap_detected(self):
        s = make_split(self.df, 42); s["val_ids"] = s["val_ids"] + s["test_ids"][:2]
        with self.assertRaises(ValueError): verify_split(self.df, s)

    def test_import_existing_split(self):
        s = make_split(self.df, 1)
        for k in ("train", "val", "test"):
            self.df[self.df.row_id.isin(s[f"{k}_ids"])][["text_raw"]].rename(columns={"text_raw": "t"}).to_csv(
                Path(self.tmp.name) / f"{k}.csv", index=False)
        imp = import_existing(self.df, *(Path(self.tmp.name) / f"{k}.csv" for k in ("train", "val", "test")), "t")
        self.assertEqual(imp["test_ids"], s["test_ids"]); verify_split(self.df, imp)


class TestEndToEnd(unittest.TestCase):
    """Runs the real pipeline on SYNTHETIC data. Checks the plumbing, not scientific results."""
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(); cls.paths = ProjectPaths(cls.tmp.name).ensure()
        make_raw_csv(cls.paths.raw_file, 700)
        build_processed(cls.paths, convert_emoji=EMOJI_AVAILABLE)
        df = pd.read_csv(cls.paths.processed_file)
        cls.paths.split_file.write_text(json.dumps(make_split(df, 42)))
        cls.out = trainmod.run(cls.paths, ["NaiveBayes", "LogReg", "SVM", "DecisionTree", "XGBoost"],
                               cv_folds=3, n_boot=60)

    @classmethod
    def tearDownClass(cls): cls.tmp.cleanup()

    def test_artifacts_and_status(self):
        st = rio.load_status(self.paths)
        for n in ("NaiveBayes", "LogReg", "SVM", "DecisionTree"):
            self.assertEqual(st[n]["status"], "completed")
            self.assertTrue((self.paths.models / f"{n}.joblib").exists())
            self.assertTrue((self.paths.model_results / f"{n}.json").exists())
        self.assertIn(st["XGBoost"]["status"], ("completed", "skipped"))
        self.assertEqual(st["GloVe_BiGRU"]["status"], "pending")

    def test_metrics_table_and_default_model(self):
        m = rio.load_metrics(self.paths)
        self.assertEqual(set(m.stage), {"validation", "test"}); self.assertEqual(set(m.threshold_type), {"default", "validation_selected"})
        v = m[(m.stage == "validation") & (m.threshold_type == "default")].set_index("model").pr_auc.to_dict()
        self.assertEqual(rio.load_default_model(self.paths)["name"], trainmod.select_default({k: {"pr_auc": x} for k, x in v.items()})["name"])
        self.assertTrue(m[m.stage == "test"].f1_ci_low.notna().all())

    def test_threshold_reproducible_from_validation_scores_only(self):
        d = rio.load_model_detail(self.paths, "LogReg"); p = rio.load_predictions(self.paths, "LogReg", "validation")
        thr, _ = ev.select_threshold(p.y_true.values, p.score.values)
        self.assertAlmostEqual(thr, d["thresholds"]["validation_selected"]["value"])

    def test_saved_metrics_match_recomputation(self):
        d = rio.load_model_detail(self.paths, "SVM"); p = rio.load_predictions(self.paths, "SVM", "test")
        th = d["thresholds"]["default"]; m = ev.compute_metrics(p.y_true, ev.decide(p.score, th["value"], th["strict"]), p.score)
        self.assertAlmostEqual(m["f1"], d["test"]["default"]["f1"])

    def test_confusion_and_pr_curve(self):
        d = rio.load_model_detail(self.paths, "NaiveBayes")
        self.assertEqual(rio.confusion_df(d, "validation", "default").Count.sum(), len(rio.load_predictions(self.paths, "NaiveBayes", "validation")))
        self.assertGreater(len(rio.pr_curve(self.paths, "NaiveBayes", "test")), 2)

    def test_predictor(self):
        pr = Predictor(self.paths)
        self.assertEqual(pr.default_model_name(), self.out["default"]["name"])
        r = pr.predict("yeah right great wow just perfect thanks")
        self.assertIn(r["label"], ("Sarcastic", "Not Sarcastic")); self.assertEqual(r["model_name"], pr.default_model_name())
        r2 = pr.predict("today match movie", model_name="SVM", threshold_type="default")
        self.assertEqual(r2["model_name"], "SVM"); self.assertIn("not a probability", r2["score_meaning"])
        self.assertEqual(set(pr._pipes), {pr.default_model_name(), "SVM"})   # loaded once each, never retrained
        pr.predict("yeah right"); self.assertEqual(len(pr._pipes), 2)
        for bad in ("", "   ", "x" * 1001):
            with self.assertRaises(InputError): pr.predict(bad)
        with self.assertRaises(InputError): pr.predict("12345 !!! ???")
        self.assertTrue(pr.predict("a " * 200 + "yeah")["warnings"])          # longer than a tweet

    def test_missing_models_handled(self):
        with tempfile.TemporaryDirectory() as t:
            with self.assertRaises(FileNotFoundError): Predictor(ProjectPaths(t).ensure()).predict("hello there")
            self.assertEqual(rio.completed_models(ProjectPaths(t)), []); self.assertIsNone(rio.load_metrics(ProjectPaths(t)))

    def test_failed_model_is_recorded_not_hidden(self):
        reg = trainmod.get_registry()
        reg["NaiveBayes"].builder = lambda n: (_ for _ in ()).throw(RuntimeError("boom"))
        with mock.patch.object(trainmod, "get_registry", return_value=reg):
            with tempfile.TemporaryDirectory() as t:
                p = ProjectPaths(t).ensure(); make_raw_csv(p.raw_file, 300); build_processed(p, convert_emoji=False)
                df = pd.read_csv(p.processed_file); p.split_file.write_text(json.dumps(make_split(df, 1)))
                trainmod.run(p, ["NaiveBayes", "LogReg"], cv_folds=3, n_boot=20)
                st = rio.load_status(p)
                self.assertEqual(st["NaiveBayes"]["status"], "failed"); self.assertIn("boom", st["NaiveBayes"]["reason"])
                self.assertEqual(st["LogReg"]["status"], "completed")


if __name__ == "__main__":
    unittest.main()
