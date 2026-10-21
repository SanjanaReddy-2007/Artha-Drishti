# How to merge this into your existing repository (safely)

I could not see your GitHub repository, so nothing here has been compared with your current files.
Several file names in this zip (`src/preprocess.py`, `src/split.py`, `demo/app.py`) **collide with files you already have**.
Do not unzip over your project. Use a feature branch and review every difference.

```powershell
cd C:\Users\User\Downloads\Artha-Drishti
git status                                   # note any uncommitted work first
git fetch origin
git checkout -b feature/topic-independent-baseline     # never work on main

# Unzip this project somewhere else first, e.g. C:\Users\User\Downloads\Artha-Drishti-new
# 1) Compare each colliding file before replacing it
git diff --no-index src\preprocess.py  ..\Artha-Drishti-new\src\preprocess.py
git diff --no-index src\split.py       ..\Artha-Drishti-new\src\split.py
git diff --no-index demo\app.py        ..\Artha-Drishti-new\demo\app.py
# also compare src\load_dataset.py and src\topic_tagging.py with the repo versions (your handoff note)

# 2) Preserve the topic-aware work, do not delete it
mkdir legacy_topic_aware -ErrorAction SilentlyContinue
git mv src\topic_tagging.py legacy_topic_aware\topic_tagging.py
git mv src\recommender.py   legacy_topic_aware\recommender.py
# (old preprocess/split/classical_models: keep a copy in legacy_topic_aware\ if you want them)

# 3) Copy the new files in (src\, demo\, tests\, docs\, requirements*.txt, README.md, .gitignore, run_pipeline.ps1)
# 4) Review, then commit WITHOUT .venv, GloVe files, zips or models\*.joblib
git status
git add src demo tests docs requirements.txt requirements-optional.txt README.md MERGE_GUIDE.md .gitignore run_pipeline.ps1 legacy_topic_aware
git commit -m "Topic-independent baseline: 10-model comparison, evaluation framework, Streamlit dashboard"
git push -u origin feature/topic-independent-baseline      # do NOT use --force
```
Check `git ls-files | findstr /i "venv glove .zip joblib"` returns nothing before pushing.
