#!/usr/bin/env python
import argparse
import csv
import json
import os
import random
from typing import Dict, List, Optional

from alarm_flood.io import load_events
from alarm_flood.match import best_match
from alarm_flood.segment import Sequence, segment_sequences
from alarm_flood.tfidf import build_tfidf, vectorize_sequence


def load_config(path: str) -> Dict:
    with open(path, "r") as f:
        return json.load(f)


def filter_sequences(
    sequences: List[Sequence],
    faults: List[int],
    min_duration: Optional[float],
) -> List[Sequence]:
    allowed = set(str(f) for f in faults)
    out = []
    for seq in sequences:
        if seq.root_cause is None:
            continue
        if str(seq.root_cause) not in allowed:
            continue
        if min_duration is not None and (seq.end_ts - seq.start_ts) < min_duration:
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

    picked: List[Sequence] = []
    per_fault = max(1, total // len(faults))
    for f in faults:
        rng.shuffle(by_fault[str(f)])
        picked.extend(by_fault[str(f)][:per_fault])

    if len(picked) < total:
        remaining = []
        for f in faults:
            remaining.extend(by_fault[str(f)][per_fault:])
        rng.shuffle(remaining)
        picked.extend(remaining[: (total - len(picked))])

    return picked[:total]


def split_knowledge(
    sampled: List[Sequence], faults: List[int], per_fault: int, seed: int
) -> (List[Sequence], List[Sequence]):
    rng = random.Random(seed)
    by_fault: Dict[str, List[Sequence]] = {str(f): [] for f in faults}
    for seq in sampled:
        by_fault[str(seq.root_cause)].append(seq)

    knowledge: List[Sequence] = []
    remaining: List[Sequence] = []
    for f in faults:
        seqs = by_fault[str(f)]
        rng.shuffle(seqs)
        knowledge.extend(seqs[:per_fault])
        remaining.extend(seqs[per_fault:])

    rng.shuffle(remaining)
    return knowledge, remaining


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Paper-style digital twin evaluation with knowledge base and Jaccard filtering."
    )
    ap.add_argument("--config", default="configs/baseline.json")
    ap.add_argument("--faults", default="1,2,3")
    ap.add_argument("--total", type=int, default=100)
    ap.add_argument("--min-duration", type=float, default=None)
    ap.add_argument(
        "--knowledge-per-fault",
        type=int,
        default=5,
        help="Initial knowledge sequences per fault",
    )
    ap.add_argument(
        "--dynamic-update",
        action="store_true",
        help="Add each evaluated sequence to knowledge base (recompute TF-IDF)",
    )
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--jaccard", type=float, default=None)
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
    knowledge, eval_seqs = split_knowledge(
        sampled, faults, args.knowledge_per_fault, args.seed
    )
    if not knowledge:
        print("Knowledge base is empty. Increase --knowledge-per-fault.")
        return 1

    jaccard_threshold = (
        args.jaccard
        if args.jaccard is not None
        else float(config.get("jaccard_threshold", 0.6))
    )

    correct = 0
    total = 0
    idf, knowledge_vecs = build_tfidf(knowledge)

    matches_path = "data/processed/matches_paper_digital_twin.csv"
    os.makedirs("data/processed", exist_ok=True)
    with open(matches_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "seq_id",
                "root_cause",
                "match_root_cause",
                "distance",
                "jaccard",
            ],
        )
        writer.writeheader()

        seq_id = 0
        for seq in eval_seqs:
            q_vec = vectorize_sequence(seq, idf)
            match_id, dist, jac = best_match(
                seq, q_vec, knowledge, knowledge_vecs, jaccard_threshold
            )
            match_root = (
                knowledge[match_id].root_cause if match_id is not None else None
            )
            writer.writerow(
                {
                    "seq_id": seq_id,
                    "root_cause": seq.root_cause,
                    "match_root_cause": match_root,
                    "distance": dist,
                    "jaccard": jac,
                }
            )
            seq_id += 1

            if match_id is not None:
                total += 1
                if str(seq.root_cause) == str(match_root):
                    correct += 1

            if args.dynamic_update:
                knowledge.append(seq)
                idf, knowledge_vecs = build_tfidf(knowledge)

    acc = (correct / total) if total else 0.0
    print(f"Sampled sequences: {len(sampled)}")
    print(f"Knowledge base size: {len(knowledge)}")
    print(f"Evaluated sequences: {len(eval_seqs)}")
    print(f"Wrote: {matches_path}")
    print(f"Accuracy: {acc:.4f} ({correct}/{total})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
