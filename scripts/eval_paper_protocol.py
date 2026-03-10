#!/usr/bin/env python
import argparse
import json
import random
from typing import Dict, List, Optional

from alarm_flood.io import load_events
from alarm_flood.match import best_match
from alarm_flood.segment import Sequence, segment_sequences
from alarm_flood.tfidf import build_tfidf


def load_config(path: str) -> Dict:
    with open(path, "r") as f:
        return json.load(f)


def filter_sequences(
    sequences: List[Sequence],
    faults: List[int],
    min_duration: Optional[float] = None,
) -> List[Sequence]:
    allowed = set(str(f) for f in faults)
    out = []
    for seq in sequences:
        if seq.root_cause is None:
            continue
        if str(seq.root_cause) not in allowed:
            continue
        if min_duration is not None:
            if (seq.end_ts - seq.start_ts) < min_duration:
                continue
        out.append(seq)
    return out


def sample_sequences(
    sequences: List[Sequence], faults: List[int], total: int, seed: int
) -> List[Sequence]:
    rng = random.Random(seed)
    by_fault: Dict[str, List[Sequence]] = {str(f): [] for f in faults}
    for seq in sequences:
        by_fault[str(seq.root_cause)].append(seq)

    # round-robin sample to balance across faults
    picked: List[Sequence] = []
    per_fault = max(1, total // len(faults))
    for f in faults:
        rng.shuffle(by_fault[str(f)])
        picked.extend(by_fault[str(f)][:per_fault])

    # top up if needed
    if len(picked) < total:
        remaining = []
        for f in faults:
            remaining.extend(by_fault[str(f)][per_fault:])
        rng.shuffle(remaining)
        picked.extend(remaining[: (total - len(picked))])

    return picked[:total]


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Paper-style evaluation: sample 100 sequences across 3 faults and run leave-one-out."
    )
    ap.add_argument("--config", default="configs/baseline.json")
    ap.add_argument(
        "--faults",
        default="1,2,3",
        help="Comma-separated fault numbers (e.g., 1,2,3)",
    )
    ap.add_argument("--total", type=int, default=100, help="Total sequences to sample")
    ap.add_argument(
        "--min-duration",
        type=float,
        default=None,
        help="Minimum duration filter in timestamp units",
    )
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    config = load_config(args.config)
    events = load_events(config, override_glob=config.get("train_glob"))
    sequences = segment_sequences(events, config)

    faults = [int(x.strip()) for x in args.faults.split(",") if x.strip()]
    filtered = filter_sequences(sequences, faults, min_duration=args.min_duration)
    if not filtered:
        print("No sequences after filtering. Check faults/min-duration.")
        return 1

    sampled = sample_sequences(filtered, faults, args.total, args.seed)
    idf, vectors = build_tfidf(sampled)

    correct = 0
    total = 0
    for i, seq in enumerate(sampled):
        candidate_indices = [j for j in range(len(sampled)) if j != i]
        candidates = [sampled[j] for j in candidate_indices]
        candidate_vecs = [vectors[j] for j in candidate_indices]
        match_id, _, _ = best_match(
            seq,
            vectors[i],
            candidates,
            candidate_vecs,
            float(config.get("jaccard_threshold", 0.6)),
        )
        if match_id is None:
            continue
        mapped = candidate_indices[match_id]
        if str(seq.root_cause) == str(sampled[mapped].root_cause):
            correct += 1
        total += 1

    acc = (correct / total) if total else 0.0
    print(f"Sampled sequences: {len(sampled)}")
    print(f"Evaluated sequences: {total}")
    print(f"Accuracy: {acc:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
