"""Tests for src/batch_multi.py (several files at once, one tweet per line).
Synthetic data and a fake predictor only: these check the plumbing, not scientific results."""
import io
import unittest
import zipfile

import numpy as np
import pandas as pd

from src import batch as bt
from src import batch_multi as bm


class FakePredictor:
    """Same return format as Predictor.predict_batch. A tweet is 'Sarcastic' if it contains 'sarcas'."""

    def __init__(self):
        self.calls = []

    def predict_batch(self, texts, model_name=None, threshold_type="validation_selected", progress=None,
                      chunk_size=500):
        texts = list(texts)
        self.calls.append(texts)
        is_s = np.array(["sarcas" in t.lower() for t in texts], dtype=bool)
        scores = np.where(is_s, 0.9, 0.1).astype(float)
        if progress:
            progress(1.0)
        return {"labels": ["Sarcastic" if x else "Not Sarcastic" for x in is_s], "is_sarcastic": is_s,
                "scores": scores, "cleaned": [t.lower() for t in texts], "threshold": 0.5,
                "threshold_type": threshold_type, "model_name": "FakeModel", "model_display": "Fake model",
                "score_meaning": "A fake score."}


def csv_bytes(*tweets, col="tweet", extra=None):
    df = pd.DataFrame({col: list(tweets)})
    if extra is not None:
        df["sarcastic"] = extra
    return df.to_csv(index=False).encode("utf-8")


def make_runs(pred, files, model=None, thr="validation_selected"):
    """files: list of (name, raw bytes). Returns FileRun list like the web page builds it."""
    loaded = bm.apply_limits([bm.load_file(raw, name) for name, raw in files])
    ok = [f for f in loaded if f.ok]
    cols = [bt.detect_text_column(f.parsed.df) for f in ok]
    rows_list = bm.run_batch_multi(pred, [(f.parsed.df, c) for f, c in zip(ok, cols)], model, thr)
    runs, k = [], 0
    for f in loaded:
        if f.ok:
            runs.append(bm.FileRun(f.name, f.parsed.df, cols[k], rows_list[k], bt.summarize(rows_list[k])))
            k += 1
        else:
            runs.append(bm.FileRun(f.name, error=f.error))
    return runs


A = csv_bytes("What a sarcastic day", "plain tweet", "", "shared tweet", "   ")
B = csv_bytes("shared tweet", "another sarcastic one", "12345 !!!")


class LoadingTests(unittest.TestCase):
    def test_unique_names(self):
        self.assertEqual(bm.unique_names(["a.csv", "a.csv", "b.csv", "a.csv"]),
                         ["a.csv", "a (2).csv", "b.csv", "a (3).csv"])

    def test_load_file_ok_and_error(self):
        self.assertTrue(bm.load_file(A, "a.csv").ok)
        bad = bm.load_file(b"", "empty.csv")
        self.assertFalse(bad.ok)
        self.assertIn("empty", bad.error.lower())

    def test_txt_file(self):
        f = bm.load_file(b"first sarcastic\nsecond\n", "t.txt")
        self.assertTrue(f.ok)
        self.assertEqual(list(f.parsed.df.tweet), ["first sarcastic", "second"])

    def test_total_row_limit_rejects_later_file_only(self):
        files = [bm.load_file(A, "a.csv"), bm.load_file(B, "b.csv"), bm.load_file(A, "c.csv")]
        out = bm.apply_limits(files, max_files=10, max_total_rows=8)   # a=5 rows, b=3 -> 8; c would be 13
        self.assertEqual([f.ok for f in out], [True, True, False])
        self.assertIn("exceed", out[2].error)

    def test_max_files_limit(self):
        files = [bm.load_file(A, f"{i}.csv") for i in range(4)]
        out = bm.apply_limits(files, max_files=2, max_total_rows=1000)
        self.assertEqual([f.ok for f in out], [True, True, False, False])
        self.assertEqual(len(out), 4)          # nothing silently dropped

    def test_one_bad_file_does_not_stop_the_others(self):
        runs = make_runs(FakePredictor(), [("a.csv", A), ("bad.csv", b""), ("b.csv", B)])
        self.assertEqual([r.rows is not None for r in runs], [True, False, True])
        self.assertTrue(runs[1].error)


class ClassificationTests(unittest.TestCase):
    def test_counts_identity_per_file_and_overall(self):
        runs = make_runs(FakePredictor(), [("a.csv", A), ("b.csv", B)])
        for r in runs:
            s = r.summary
            self.assertEqual(s["sarcastic"] + s["not_sarcastic"] + s["skipped"], s["total_rows"])
        self.assertEqual(runs[0].summary["total_rows"], 5)
        self.assertEqual(runs[0].summary["skipped"], 2)        # blank + whitespace
        self.assertEqual(runs[1].summary["skipped"], 1)        # '12345 !!!' has no words
        self.assertTrue(bm.counts_consistent(runs))

    def test_each_unique_text_scored_once_across_files(self):
        pred = FakePredictor()
        make_runs(pred, [("a.csv", A), ("b.csv", B)])
        self.assertEqual(len(pred.calls), 1)                    # one predict_batch call for everything
        texts = pred.calls[0]
        self.assertEqual(len(texts), len(set(texts)))
        self.assertEqual(texts.count("shared tweet"), 1)        # in both files, scored once

    def test_same_result_as_single_file_run_batch(self):
        pred = FakePredictor()
        runs = make_runs(pred, [("a.csv", A), ("b.csv", B)])
        for (name, raw), r in zip([("a.csv", A), ("b.csv", B)], runs):
            df = bt.read_upload(raw, name).df
            single = bt.run_batch(FakePredictor(), df, "tweet", None, "validation_selected")
            pd.testing.assert_frame_equal(r.rows.reset_index(drop=True), single.reset_index(drop=True),
                                          check_dtype=True)

    def test_cross_file_duplicates(self):
        runs = make_runs(FakePredictor(), [("a.csv", A), ("b.csv", B)])
        self.assertEqual(bm.cross_file_duplicates(runs), 1)

    def test_files_with_different_column_names(self):
        raw = csv_bytes("sarcastic remark", col="text")
        runs = make_runs(FakePredictor(), [("a.csv", A), ("t.csv", raw)])
        self.assertEqual(runs[1].tweet_col, "text")
        self.assertEqual(runs[1].summary["sarcastic"], 1)


class OutputTests(unittest.TestCase):
    def setUp(self):
        self.runs = make_runs(FakePredictor(), [("a.csv", A), ("bad.csv", b""), ("b.csv", B)])
        self.out = bm.build_outputs(self.runs, "FakeModel", "validation_selected")

    def test_combined_table(self):
        c = bm.combine_runs(self.runs)
        self.assertEqual(list(c.columns), bm.COMBINED_COLS)
        self.assertEqual(len(c), 5 + 3)
        self.assertEqual(set(c.source_file), {"a.csv", "b.csv"})

    def test_summary_has_failed_file_and_total_row(self):
        s = self.out["summary_df"]
        self.assertEqual(list(s.file), ["a.csv", "bad.csv", "b.csv", bm.ALL_FILES])
        self.assertTrue(pd.isna(s.loc[1, "total_rows"]))
        self.assertIn("empty", s.loc[1, "status"].lower())
        total = s[s.file == bm.ALL_FILES].iloc[0]
        self.assertEqual(total.total_rows, 8)
        self.assertEqual(total.sarcastic + total.not_sarcastic + total.skipped_rows, total.total_rows)

    def test_csv_has_bom(self):
        self.assertTrue(self.out["combined"].startswith(b"\xef\xbb\xbf"))
        self.assertTrue(self.out["summary"].startswith(b"\xef\xbb\xbf"))

    def test_zip_contents(self):
        z = zipfile.ZipFile(io.BytesIO(self.out["zip"]))
        self.assertEqual(sorted(z.namelist()), ["all_files_combined_results.csv", "per_file/a_results.csv",
                                                "per_file/b_results.csv", "summary.csv"])
        per_file = pd.read_csv(io.BytesIO(z.read("per_file/a_results.csv")), dtype=str, keep_default_na=False)
        self.assertEqual(len(per_file), 5)                       # every input row kept
        self.assertIn("tweet", per_file.columns)                 # original columns present
        self.assertIn("predicted_label", per_file.columns)

    def test_zip_member_names_unique_and_safe(self):
        names = bm.zip_member_names(["a.csv", "a.txt", "../we ird/na*me.csv", ""])
        self.assertEqual(len(set(names)), 4)
        self.assertTrue(all("/" not in n and "*" not in n and " " not in n for n in names))

    def test_no_processed_files_still_builds(self):
        runs = make_runs(FakePredictor(), [("bad.csv", b"")])
        out = bm.build_outputs(runs, "FakeModel", "default")
        self.assertEqual(zipfile.ZipFile(io.BytesIO(out["zip"])).namelist().count("summary.csv"), 1)
        self.assertEqual(out["files"], {})


class TextBoxTests(unittest.TestCase):
    def test_split_one_per_line_ignores_blanks(self):
        tweets, n = bm.split_tweets("  first one \r\n\r\n\nsecond one\n   \nthird")
        self.assertEqual(tweets, ["first one", "second one", "third"])
        self.assertEqual(n, 3)

    def test_split_single_line_and_empty(self):
        self.assertEqual(bm.split_tweets("just one"), (["just one"], 1))
        self.assertEqual(bm.split_tweets("  \n \n"), ([], 0))
        self.assertEqual(bm.split_tweets(None), ([], 0))

    def test_split_cap_reports_total(self):
        tweets, n = bm.split_tweets("\n".join(f"tweet {i}" for i in range(10)), max_items=4)
        self.assertEqual((len(tweets), n), (4, 10))

    def test_lines_table(self):
        lines = ["so sarcastic today", "meh", "12345"]
        rows = bt.run_batch(FakePredictor(), pd.DataFrame({"tweet": lines}), "tweet", None, "default")
        t = bm.lines_table(rows)
        self.assertEqual(list(t["Predicted label"]), ["Sarcastic", "Not Sarcastic", "-"])
        self.assertIn("skipped", t.loc[2, "Note"])
        self.assertEqual(len(t), 3)


if __name__ == "__main__":
    unittest.main()
