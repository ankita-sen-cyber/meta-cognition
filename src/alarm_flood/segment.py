from collections import Counter
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

from .io import Event


@dataclass
class Sequence:
    seq_id: int
    start_ts: float
    end_ts: float
    events: List[Event]
    alarm_counts: Counter
    root_cause: Optional[str]


def _most_common_root(events: List[Event]) -> Optional[str]:
    labels = [e.root_cause for e in events if e.root_cause]
    if not labels:
        return None
    return Counter(labels).most_common(1)[0][0]


def segment_sequences(events: List[Event], config: Dict) -> List[Sequence]:
    min_len = int(config.get("min_sequence_length", 2))
    merge_gap = float(config.get("merge_gap_seconds", 30))

    sequences: List[Sequence] = []
    current_events: List[Event] = []
    active = set()

    current_run_id = None
    for e in events:
        # hard boundary on run_id changes (TEP-style runs)
        if current_events and e.run_id is not None and e.run_id != current_run_id:
            counts = Counter(ev.alarm_id for ev in current_events)
            seq = Sequence(
                seq_id=len(sequences),
                start_ts=current_events[0].ts,
                end_ts=current_events[-1].ts,
                events=current_events,
                alarm_counts=counts,
                root_cause=_most_common_root(current_events),
            )
            sequences.append(seq)
            current_events = []
            active = set()
            current_run_id = None

        if not current_events:
            # Only start on activation to avoid stray deactivation events
            if e.state != 1:
                continue
            current_events.append(e)
            active.add(e.alarm_id)
            current_run_id = e.run_id
            continue

        current_events.append(e)
        if e.state == 1:
            active.add(e.alarm_id)
        else:
            if e.alarm_id in active:
                active.remove(e.alarm_id)

        if not active:
            # Close sequence at this event
            counts = Counter(ev.alarm_id for ev in current_events)
            seq = Sequence(
                seq_id=len(sequences),
                start_ts=current_events[0].ts,
                end_ts=current_events[-1].ts,
                events=current_events,
                alarm_counts=counts,
                root_cause=_most_common_root(current_events),
            )
            sequences.append(seq)
            current_events = []
            current_run_id = None

    # Merge adjacent sequences using coactivation constraint (shared alarms + short gap)
    if not sequences:
        return []

    merged: List[Sequence] = []
    i = 0
    while i < len(sequences):
        cur = sequences[i]
        j = i + 1
        while j < len(sequences):
            nxt = sequences[j]
            gap = nxt.start_ts - cur.end_ts
            if gap > merge_gap:
                break
            if set(cur.alarm_counts.keys()) & set(nxt.alarm_counts.keys()):
                # merge
                cur_events = cur.events + nxt.events
                cur = Sequence(
                    seq_id=cur.seq_id,
                    start_ts=cur.start_ts,
                    end_ts=nxt.end_ts,
                    events=cur_events,
                    alarm_counts=Counter(ev.alarm_id for ev in cur_events),
                    root_cause=_most_common_root(cur_events),
                )
                j += 1
                continue
            break
        if len(cur.events) >= min_len:
            merged.append(cur)
        i = j

    # Re-assign seq_id sequentially
    for idx, seq in enumerate(merged):
        seq.seq_id = idx

    return merged
