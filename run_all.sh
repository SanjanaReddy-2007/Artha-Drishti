#!/usr/bin/env bash
# Full pipeline. Usage: bash run_all.sh path/to/train.En.csv [path/to/glove.txt]
set -e
cd "$(dirname "$0")/src"
python load_dataset.py --input "$1"
python topic_tagging.py                 # add --backend keyword for an offline smoke test
python find_missing_row.py
python make_judging_file.py             # <- now fill data/judging_file.csv, then:
python check_topic_quality.py
python preprocess.py --emoji convert
python split.py
python classical_models.py
if [ -n "$2" ]; then python neural_model.py --glove "$2"; else python neural_model.py; fi
python evaluate.py
python recommender.py
python learning_curves.py
echo "Demo: streamlit run ../demo/app.py"
