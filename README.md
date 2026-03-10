# meta-cognition
Meta cognition in alarm flood management.

## Baseline pipeline (paper algorithm)
1. Put alarm event CSVs in `data/raw/`.
2. Adjust `configs/baseline.json` if your column names differ.
3. Run:
```bash
PYTHONPATH=src python scripts/run_baseline.py
```

Outputs go to `data/processed/`.
If `test_glob` is set, matches are computed from test → train sequences.

## If using TEP Dataverse .RData
Convert continuous signals into alarm events first (train/test):
```bash
python scripts/tep_to_alarm_events.py \
  --faultfree data/dataverse_files/TEP_FaultFree_Training.RData \
  --faulty data/dataverse_files/TEP_Faulty_Training.RData \
  --thresholds data/raw/te_alarm_thresholds.csv \
  --max-runs-per-fault 5 \
  --out data/raw/tep_alarm_events_train.csv

python scripts/tep_to_alarm_events.py \
  --faultfree data/dataverse_files/TEP_FaultFree_Training.RData \
  --faulty data/dataverse_files/TEP_Faulty_Testing.RData \
  --thresholds data/raw/te_alarm_thresholds.csv \
  --max-runs-per-fault 5 \
  --out data/raw/tep_alarm_events_test.csv
```
Then run the baseline pipeline with `train_glob` and `test_glob` set in `configs/baseline.json`.
