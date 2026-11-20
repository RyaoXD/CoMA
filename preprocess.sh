#!/usr/bin/env bash
# Build processed CSV / pickle splits for each public dataset.
# Set COMA_DATA_DIR to the root that contains the raw dataset folders first.
set -euo pipefail

python ./coma/preprocess/mimic_cxr.py
python ./coma/preprocess/chexpert.py
python ./coma/preprocess/rsna.py
python ./coma/preprocess/covidx.py
python ./coma/preprocess/siim.py
python ./coma/preprocess/object_cxr.py
