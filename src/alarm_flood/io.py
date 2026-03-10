import csv
import glob
import json
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Iterable, List, Optional, Tuple


@dataclass
class Event:
    ts: float
    alarm_id: str
    state: int
    root_cause: Optional[str] = None
    raw_ts: Optional[str] = None
    run_id: Optional[str] = None


def _parse_timestamp(value: str, mode: str) -> float:
    if mode == "auto":
        # Try ISO-8601 first, fallback to float seconds
        try:
            dt = datetime.fromisoformat(value)
            return dt.timestamp()
        except Exception:
            try:
                return float(value)
            except Exception as exc:
                raise ValueError(f"Unparseable timestamp: {value}") from exc
    if mode == "iso":
        return datetime.fromisoformat(value).timestamp()
    if mode == "unix":
        return float(value)
    raise ValueError(f"Unsupported timestamp_format: {mode}")


def load_events(config: Dict, override_glob: str | None = None) -> List[Event]:
    glob_pattern = override_glob or config["input_glob"]
    files = sorted(glob.glob(glob_pattern))
    if not files:
        raise FileNotFoundError(f"No files matched {glob_pattern}")

    ts_col = config["timestamp_col"]
    alarm_col = config["alarm_id_col"]
    state_col = config["state_col"]
    root_col = config.get("root_cause_col")
    run_col = config.get("run_id_col")
    ts_mode = config.get("timestamp_format", "auto")

    events: List[Event] = []
    for path in files:
        with open(path, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if ts_col not in row or alarm_col not in row or state_col not in row:
                    raise KeyError(
                        f"Missing required columns in {path}. "
                        f"Need: {ts_col}, {alarm_col}, {state_col}"
                    )
                ts_raw = row[ts_col]
                ts = _parse_timestamp(ts_raw, ts_mode)
                alarm_id = str(row[alarm_col]).strip()
                state = int(float(row[state_col]))
                root = None
                if root_col and root_col in row and row[root_col] != "":
                    root = str(row[root_col]).strip()
                run_id = None
                if run_col and run_col in row and row[run_col] != "":
                    run_id = str(row[run_col]).strip()
                events.append(
                    Event(
                        ts=ts,
                        alarm_id=alarm_id,
                        state=state,
                        root_cause=root,
                        raw_ts=ts_raw,
                        run_id=run_id,
                    )
                )

    events.sort(key=lambda e: e.ts)
    return events


def write_jsonl(path: str, rows: Iterable[Dict]) -> None:
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
