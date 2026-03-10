#!/usr/bin/env python
import argparse
import csv
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import pyreadr


def load_rdata(path: Path) -> pd.DataFrame:
    res = pyreadr.read_r(str(path))
    # Assume single dataframe in file
    df = next(iter(res.values()))
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected DataFrame in {path}")
    return df


def infer_variable_columns(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if c.startswith("xmeas_") or c.startswith("xmv_")]


def load_thresholds(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"variable", "hi_alarm", "lo_alarm"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Thresholds file missing columns: {sorted(missing)}")
    return df


def build_alarm_events(
    df_faulty: pd.DataFrame,
    mean: pd.Series,
    std: pd.Series,
    k: float,
    out_csv: Path,
    thresholds: pd.DataFrame | None = None,
    max_runs_per_fault: int | None = None,
) -> None:
    var_cols = mean.index.tolist()
    thr_map: Dict[str, Tuple[float, float]] = {}
    if thresholds is not None:
        for _, row in thresholds.iterrows():
            thr_map[str(row["variable"])] = (float(row["lo_alarm"]), float(row["hi_alarm"]))

    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "timestamp",
                "alarm_id",
                "state",
                "root_cause",
                "faultNumber",
                "simulationRun",
                "run_id",
            ],
        )
        writer.writeheader()

        run_count_by_fault: Dict[int, int] = {}
        for (fault, run_id), run_df in df_faulty.groupby(["faultNumber", "simulationRun"]):
            fault = int(fault)
            if max_runs_per_fault is not None:
                run_count_by_fault.setdefault(fault, 0)
                if run_count_by_fault[fault] >= max_runs_per_fault:
                    continue
                run_count_by_fault[fault] += 1
            run_df = run_df.sort_values("sample")
            time = run_df["sample"].astype(int)
            root = fault
            run_uid = f"{int(fault)}_{int(run_id)}"

            for col in var_cols:
                if col in thr_map:
                    lo, hi = thr_map[col]
                else:
                    hi = mean[col] + k * std[col]
                    lo = mean[col] - k * std[col]

                series = run_df[col]
                high_state = (series > hi).astype(int)
                low_state = (series < lo).astype(int)

                for state_series, suffix in ((high_state, "high"), (low_state, "low")):
                    # Emit events only on state change
                    changes = state_series.diff().fillna(state_series).ne(0)
                    idxs = state_series.index[changes]
                    for idx in idxs:
                        writer.writerow(
                            {
                                "timestamp": int(time.loc[idx]),
                                "alarm_id": f"{col}_{suffix}",
                                "state": int(state_series.loc[idx]),
                                "root_cause": root,
                                "faultNumber": int(fault),
                                "simulationRun": int(run_id),
                                "run_id": run_uid,
                            }
                        )


def main() -> int:
    ap = argparse.ArgumentParser(description="Convert TEP RData to alarm event CSV")
    ap.add_argument("--faultfree", required=True, help="Path to TEP_FaultFree_Training.RData")
    ap.add_argument("--faulty", required=True, help="Path to TEP_Faulty_Training.RData")
    ap.add_argument("--k", type=float, default=3.0, help="Stddev multiplier for alarm thresholds")
    ap.add_argument("--thresholds", default=None, help="CSV with columns: variable, lo_alarm, hi_alarm")
    ap.add_argument(
        "--max-runs-per-fault",
        type=int,
        default=None,
        help="Limit runs per fault for faster debugging",
    )
    ap.add_argument("--out", default="data/raw/tep_alarm_events.csv", help="Output CSV path")
    args = ap.parse_args()

    df_ff = load_rdata(Path(args.faultfree))
    df_faulty = load_rdata(Path(args.faulty))

    var_cols = infer_variable_columns(df_ff)
    mean = df_ff[var_cols].mean()
    std = df_ff[var_cols].std(ddof=0)

    thresholds = None
    if args.thresholds:
        thresholds = load_thresholds(Path(args.thresholds))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    build_alarm_events(
        df_faulty,
        mean,
        std,
        args.k,
        out_path,
        thresholds=thresholds,
        max_runs_per_fault=args.max_runs_per_fault,
    )

    print(f"Wrote alarm events to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
