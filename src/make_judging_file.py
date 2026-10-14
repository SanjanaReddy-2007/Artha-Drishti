"""
src/make_judging_file.py
Creates results/topic_check_judge_me.csv: the 60 sample tweets with a blank
'my_topic' column and NO model topic shown, so your judgement is unbiased.
Also keeps the model's answers in a separate file for the checker to use.
Run from the project root:  python src/make_judging_file.py
"""
import pandas as pd

s = pd.read_csv("results/topic_check_sample.csv")
print("Columns found:", list(s.columns))

# Keep the model's answers separately (needed later to compute accuracy)
s.to_csv("results/topic_check_answers_DO_NOT_OPEN.csv", index=False, encoding="utf-8-sig")

# Judging file: drop anything that reveals the model's topic or score
hide = [c for c in s.columns if c.lower() in ("topic", "topic_score", "topic_margin")]
j = s.drop(columns=hide)
j["my_topic"] = ""   # fill: 1=Politics & Society, 2=Entertainment & Sports, 3=Everyday Life, 0=no clear topic
j.to_csv("results/topic_check_judge_me.csv", index=False, encoding="utf-8-sig")
print("Created results/topic_check_judge_me.csv  ->  open it and fill the my_topic column.")