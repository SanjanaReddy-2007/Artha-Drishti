# Optional helper for Windows PowerShell. Run from the project folder.
#   .\run_pipeline.ps1            -> setup + prepare + split + train (full) + tests
#   .\run_pipeline.ps1 -Smoke     -> same, but training uses only 300 rows (NOT a valid experiment)
param([switch]$Smoke)
$ErrorActionPreference = "Stop"
if (-not (Test-Path .venv)) { python -m venv .venv }
. .venv\Scripts\Activate.ps1
pip install --default-timeout=120 --retries 10 -r requirements.txt
try { pip install --default-timeout=120 --retries 10 -r requirements-optional.txt } catch { Write-Warning "xgboost not installed; it will be recorded as skipped." }
if (-not (Test-Path data\train.En.csv)) { throw "Put the iSarcasmEval English training file at data\train.En.csv first." }
New-Item -ItemType Directory -Force logs, results, models | Out-Null
python -m src.prepare_data 2>&1 | Tee-Object logs\prepare_data.txt
if (-not (Test-Path results\split_indices.json)) { python -m src.split 2>&1 | Tee-Object logs\split.txt }
if ($Smoke) { python -m src.train --limit 300 --cv-folds 3 --bootstrap 100 } else { python -m src.train }
python -m unittest discover -s tests -t . 2>&1 | Tee-Object logs\tests.txt
Write-Host "Launch the website with:  streamlit run demo\app.py"
