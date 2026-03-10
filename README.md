# meta-cognition
Meta cognition in alarm flood management.

## Paper-style evaluation (100 sequences, 3 faults)
To mimic the paper’s setup (100 sequences, 3 root causes), run a leave-one-out evaluation:
```bash
PYTHONPATH=src python scripts/eval_paper_protocol.py --faults 1,2,3 --total 100
```
Add `--min-duration` if you want to enforce a minimum sequence length.

## Paper-style digital twin (knowledge base + Jaccard filter)
This mirrors the paper’s assistance system (knowledge base + TF-IDF + Jaccard). You can control the initial knowledge size:
```bash
PYTHONPATH=src python scripts/run_paper_digital_twin.py --faults 1,2,3 --total 100 --knowledge-per-fault 5
```
To simulate dynamic knowledge updates:
```bash
PYTHONPATH=src python scripts/run_paper_digital_twin.py --faults 1,2,3 --total 100 --knowledge-per-fault 5 --dynamic-update
```

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
