"""Tests for the Batch Detector logic (src/batch.py, Predictor.predict_batch). SYNTHETIC data only."""
import io
import json
import tempfile
import unittest

import numpy as np
import pandas as pd

from src import batch as bt
from src import train as trainmod
from src.config import BATCH_MAX_ROWS, ProjectPaths
from src.data import build_processed
from src.predict import InputError, Predictor
from src.preprocess import EMOJI_AVAILABLE
from src.split import make_split
from tests.synthetic import make_raw_csv


def csv_bytes(text, enc="utf-8"):
    return text.encode(enc)


class TestParsing(unittest.TestCase):
    def test_utf8_plain(self):
        p = bt.read_upload(csv_bytes("tweet,id\nhello world,1\n"), "a.csv")
        self.assertEqual(p.encoding, "utf-8"); self.assertEqual(list(p.df.columns), ["tweet", "id"])
        self.assertEqual(p.notes, [])

    def test_utf8_bom_is_stripped(self):
        p = bt.read_upload(csv_bytes("tweet\ncaf\u00e9 time\n", "utf-8-sig"), "a.csv")
        self.assertEqual(p.encoding, "utf-8-sig"); self.assertEqual(list(p.df.columns), ["tweet"])
        self.assertTrue(p.notes)

    def test_latin1_fallback_is_reported(self):
        p = bt.read_upload(csv_bytes("tweet\ncaf\u00e9 time\n", "latin-1"), "a.csv")
        self.assertEqual(p.encoding, "latin-1"); self.assertEqual(p.df.tweet[0], "caf\u00e9 time")
        self.assertTrue(any("latin-1" in n for n in p.notes))

    def test_txt_one_tweet_per_line(self):
        p = bt.read_upload(csv_bytes("first tweet\nsecond tweet\n\nfourth"), "t.txt")
        self.assertEqual(list(p.df.columns), ["tweet"]); self.assertEqual(len(p.df), 4)

    def test_no_header_option(self):
        p = bt.read_upload(csv_bytes("hello there\nanother one\n"), "a.csv", has_header=False)
        self.assertEqual(len(p.df), 2); self.assertEqual(list(p.df.columns), ["column 1"])

    def test_semicolon_and_tab_delimiters(self):
        self.assertEqual(list(bt.read_upload(csv_bytes("tweet;id\nhi there;1\n"), "a.csv").df.columns), ["tweet", "id"])
        self.assertEqual(list(bt.read_upload(csv_bytes("tweet\tid\nhi there\t1\n"), "a.csv").df.columns), ["tweet", "id"])

    def test_quoted_commas_and_newlines(self):
        p = bt.read_upload(csv_bytes('tweet,id\n"well, this, works",1\n"two\nlines",2\n'), "a.csv")
        self.assertEqual(len(p.df), 2); self.assertEqual(p.df.tweet[0], "well, this, works")

    def test_empty_and_whitespace_files(self):
        for raw in (b"", b"   \n \n"):
            with self.assertRaises(bt.BatchError): bt.read_upload(raw, "a.csv")

    def test_header_only(self):
        with self.assertRaises(bt.BatchError) as cm: bt.read_upload(csv_bytes("tweet\n"), "a.csv")
        self.assertIn("no data rows", str(cm.exception))

    def test_row_limit_enforced(self):
        raw = csv_bytes("tweet\n" + "\n".join(f"tweet number {i}" for i in range(11)))
        with self.assertRaises(bt.BatchError) as cm: bt.read_upload(raw, "a.csv", max_rows=10)
        self.assertIn("limit", str(cm.exception))
        self.assertEqual(len(bt.read_upload(raw, "a.csv", max_rows=11).df), 11)
        with self.assertRaises(bt.BatchError): bt.read_upload(raw.replace(b"tweet\n", b"", 1), "a.txt", max_rows=10)
        self.assertGreaterEqual(BATCH_MAX_ROWS, 5000)

    def test_size_limit_enforced(self):
        with self.assertRaises(bt.BatchError) as cm: bt.read_upload(b"x" * 2048, "a.csv", max_mb=0.001)
        self.assertIn("MB", str(cm.exception))

    def test_ragged_lines_reported_and_columns_not_shifted(self):
        p = bt.read_upload(csv_bytes("tweet,id\nok one,1\nbad,1,2,3\nok two,2\n"), "a.csv")
        self.assertEqual(len(p.df), 3); self.assertEqual(list(p.df.columns), ["tweet", "id"])
        self.assertEqual(p.df.tweet.tolist(), ["ok one", "bad", "ok two"])      # no column promoted to index
        self.assertTrue(any("different number of fields" in n for n in p.notes))

    def test_duplicate_header_names_made_unique(self):
        p = bt.read_upload(csv_bytes("tweet,tweet\na b,c d\n"), "a.csv")
        self.assertEqual(len(set(p.df.columns)), 2)


class TestColumnDetection(unittest.TestCase):
    def test_tweet_case_insensitive_first(self):
        self.assertEqual(bt.detect_text_column(pd.DataFrame({"id": ["1"], "TWEET": ["hi"], "text": ["x"]})), "TWEET")

    def test_text_second(self):
        self.assertEqual(bt.detect_text_column(pd.DataFrame({"id": ["1"], "Text": ["hi"]})), "Text")

    def test_first_text_like_column(self):
        df = pd.DataFrame({"id": ["1", "2"], "score": ["0.5", "3"], "content": ["hello there", "good day"]})
        self.assertEqual(bt.detect_text_column(df), "content")

    def test_fallback_first_column(self):
        self.assertEqual(bt.detect_text_column(pd.DataFrame({"a": ["1"], "b": ["2"]})), "a")

    def test_label_column_detection(self):
        df = pd.DataFrame({"tweet": ["a b"], "sarcastic": ["1"], "other": ["x"]})
        self.assertEqual(bt.detect_label_column(df, "tweet"), "sarcastic")
        self.assertIsNone(bt.detect_label_column(pd.DataFrame({"tweet": ["a b"], "id": ["5"]}), "tweet"))


class TestPrepareRows(unittest.TestCase):
    def test_blank_nan_nonstring_long_and_wordless(self):
        df = pd.DataFrame({"t": ["good day", "", "   ", None, float("nan"), 12345, "!!! ???", "x" * 1001,
                                 "a " * 200 + "yeah"]})
        r = bt.prepare_rows(df, "t")
        self.assertEqual(r.status.tolist(), ["predicted", "blank", "blank", "blank", "blank", "no_words",
                                              "no_words", "too_long", "predicted"])
        self.assertEqual(r.row_number.tolist(), list(range(1, 10)))
        self.assertTrue(r.note[8])                       # longer than a tweet -> warning note
        self.assertEqual(r.tweet[5], "12345")            # non-string cell became text

    def test_missing_column(self):
        with self.assertRaises(bt.BatchError): bt.prepare_rows(pd.DataFrame({"a": ["x"]}), "tweet")


def _fitted_paths():
    tmp = tempfile.TemporaryDirectory(); paths = ProjectPaths(tmp.name).ensure()
    make_raw_csv(paths.raw_file, 700)
    build_processed(paths, convert_emoji=EMOJI_AVAILABLE)
    df = pd.read_csv(paths.processed_file)
    paths.split_file.write_text(json.dumps(make_split(df, 42)))
    trainmod.run(paths, ["NaiveBayes", "LogReg", "SVM", "DecisionTree"], cv_folds=3, n_boot=20)
    return tmp, paths


class TestPredictBatch(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp, cls.paths = _fitted_paths(); cls.pr = Predictor(cls.paths)

    @classmethod
    def tearDownClass(cls): cls.tmp.cleanup()

    texts = ["yeah right great wow just perfect thanks", "today match movie", "@bob look https://x.co #Wow so great",
             "coffee music weather team news", "totally love perfect just wow"]

    def test_shape_and_consistency_with_single_tweet(self):
        for model in (None, "SVM", "NaiveBayes"):
            for thr in ("default", "validation_selected"):
                b = self.pr.predict_batch(self.texts, model, thr)
                self.assertEqual(len(b["labels"]), len(self.texts)); self.assertEqual(b["scores"].shape, (len(self.texts),))
                for i, t in enumerate(self.texts):
                    s = self.pr.predict(t, model, thr)
                    self.assertEqual(b["labels"][i], s["label"]); self.assertEqual(bool(b["is_sarcastic"][i]), s["is_sarcastic"])
                    self.assertAlmostEqual(b["scores"][i], s["score"], places=9)
                    self.assertEqual(b["threshold"], s["threshold"]); self.assertEqual(b["model_name"], s["model_name"])
                    self.assertEqual(b["cleaned"][i], s["cleaned_text"])

    def test_default_model_read_from_metadata_and_chunking_and_progress(self):
        calls = []
        b = self.pr.predict_batch(self.texts, None, "default", progress=calls.append, chunk_size=2)
        self.assertEqual(b["model_name"], self.pr.default_model_name())
        self.assertEqual(len(calls), 3); self.assertAlmostEqual(calls[-1], 1.0)
        one = self.pr.predict_batch(self.texts, None, "default", chunk_size=1000)
        np.testing.assert_allclose(b["scores"], one["scores"])

    def test_invalid_input_errors_like_single(self):
        for bad in ("", "   ", "x" * 1001, "12345 !!!"):
            with self.assertRaises(InputError): self.pr.predict_batch(["fine text yeah", bad])
        with self.assertRaises(ValueError): self.pr.predict_batch(self.texts, None, "nope")
        self.assertEqual(len(self.pr.predict_batch([])["labels"]), 0)

    def test_no_models(self):
        with tempfile.TemporaryDirectory() as t:
            with self.assertRaises(FileNotFoundError): Predictor(ProjectPaths(t).ensure()).predict_batch(["hello there"])


class TestRunBatchAndOutputs(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp, cls.paths = _fitted_paths(); cls.pr = Predictor(cls.paths)
        cls.df = pd.DataFrame({
            "id": range(1, 10),
            "tweet": ["yeah right great wow", "today match movie", "", "yeah right great wow",   # duplicate of row 1
                      None, "12345", "coffee music weather", "x" * 1001, "totally love perfect just wow"],
            "sarcastic": ["1", "0", "0", "1", "0", "0", "0", "1", "1"],
            "predicted_label": ["keep me"] * 9})                                              # name collision

    @classmethod
    def tearDownClass(cls): cls.tmp.cleanup()

    def run_it(self, **kw):
        return bt.run_batch(self.pr, self.df, "tweet", kw.get("model"), kw.get("thr", "validation_selected"))

    def test_counts_add_up(self):
        rows = self.run_it(); s = bt.summarize(rows)
        self.assertEqual(s["total_rows"], 9); self.assertEqual(s["skipped"], 4)       # '', None, '12345', too long
        self.assertEqual(s["sarcastic"] + s["not_sarcastic"] + s["skipped"], s["total_rows"])
        self.assertEqual(s["scored"], 5); self.assertEqual(s["duplicate_rows"], 1)
        self.assertAlmostEqual(s["sarcastic_pct"] + s["not_sarcastic_pct"], 100.0)
        self.assertEqual(rows.attrs["n_unique_predicted"], 4)                          # duplicate scored once

    def test_duplicates_get_identical_predictions_and_match_single_page(self):
        rows = self.run_it()
        a, b = rows.iloc[0], rows.iloc[3]
        self.assertEqual((a.predicted_label, a.score), (b.predicted_label, b.score))
        single = self.pr.predict("yeah right great wow", None, "validation_selected")
        self.assertEqual(a.predicted_label, single["label"]); self.assertAlmostEqual(a.score, single["score"], places=9)

    def test_skipped_rows_have_no_prediction(self):
        rows = self.run_it(); sk = rows[rows.status != "predicted"]
        self.assertTrue(sk.predicted_label.isna().all()); self.assertTrue(sk.score.isna().all())
        self.assertTrue(all(s.startswith("skipped: ") for s in sk.status))

    def test_output_columns_and_order_and_bom(self):
        rows = self.run_it(thr="default", model="SVM"); out = bt.build_output(self.df, rows)
        self.assertEqual(len(out), len(self.df))                                      # every row kept
        self.assertEqual(list(out.columns)[:4], ["id", "tweet", "sarcastic", "predicted_label_original"])
        for c in ("predicted_label", "predicted_class", "score", "threshold", "model", "threshold_mode"):
            self.assertIn(c, out.columns)
        self.assertTrue((out.model == "SVM").all()); self.assertTrue((out.threshold_mode == "default").all())
        self.assertTrue(set(out.predicted_class.dropna().astype(int)) <= {0, 1})
        self.assertTrue((out.predicted_label_original == "keep me").all())
        raw = bt.to_csv_bytes(out); self.assertTrue(raw.startswith(b"\xef\xbb\xbf"))
        back = pd.read_csv(io.BytesIO(raw), encoding="utf-8-sig"); self.assertEqual(len(back), 9)

    def test_summary_csv(self):
        rows = self.run_it(); s = bt.summarize(rows)
        f = bt.summary_frame(s, "LogReg", "default").set_index("metric").value
        self.assertEqual(int(f["total_rows"]), 9); self.assertEqual(int(f["skipped_rows"]), 4)

    def test_label_column_never_reaches_model(self):
        seen = []
        orig = self.pr.predict_batch
        self.pr.predict_batch = lambda texts, *a, **k: (seen.extend(texts), orig(texts, *a, **k))[1]
        try: self.run_it()
        finally: self.pr.predict_batch = orig
        self.assertTrue(seen); self.assertTrue(set(seen) <= set(self.df.tweet.dropna()))
        self.assertNotIn("1", seen)

    def test_labelled_check(self):
        rows = self.run_it(); lc = bt.labelled_check(self.df, rows, "sarcastic")
        self.assertEqual(lc["n_used"], 5); self.assertEqual(int(lc["confusion"].to_numpy().sum()), 5)
        for k in ("precision", "recall", "f1", "accuracy"): self.assertTrue(0.0 <= lc[k] <= 1.0)
        with self.assertRaises(bt.BatchError): bt.labelled_check(self.df.assign(bad="zzz"), rows, "bad")

    def test_compare_models(self):
        rows = self.run_it(); cmp = bt.compare_models(self.pr, rows, self.pr.available_models(), "default")
        self.assertEqual(set(cmp.Model), set(self.pr.available_models()))
        self.assertTrue(((cmp.Sarcastic + cmp["Not Sarcastic"]) == 5).all())          # counted per scored row

    def test_all_rows_unscorable(self):
        rows = bt.prepare_rows(pd.DataFrame({"t": ["", "123"]}), "t")
        self.assertFalse((rows.status == "predicted").any())


if __name__ == "__main__":
    unittest.main()
