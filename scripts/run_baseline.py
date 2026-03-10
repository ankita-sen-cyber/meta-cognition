#!/usr/bin/env python
import csv
import json
import os
import sys
from typing import Dict

from alarm_flood.io import load_events, write_jsonl
from alarm_flood.match import best_match
from alarm_flood.segment import segment_sequences
from alarm_flood.tfidf import build_tfidf, vectorize_sequence


def load_config(path: str) -> Dict:
    with open(path, "r") as f:
        return json.load(f)


def main() -> int:
    config_path = sys.argv[1] if len(sys.argv) > 1 else "configs/baseline.json"
    config = load_config(config_path)

    train_glob = config.get("train_glob")
    test_glob = config.get("test_glob")

    train_events = load_events(config, override_glob=train_glob)
    train_sequences = segment_sequences(train_events, config)

    os.makedirs("data/processed", exist_ok=True)

    seq_rows = []
    for seq in train_sequences:
        seq_rows.append({
            "seq_id": seq.seq_id,
            "start_ts": seq.start_ts,
            "end_ts": seq.end_ts,
            "alarm_counts": dict(seq.alarm_counts),
            "root_cause": seq.root_cause,
            "event_count": len(seq.events),
        })
    write_jsonl("data/processed/sequences_train.jsonl", seq_rows)

    idf, train_vectors = build_tfidf(train_sequences)
    tfidf_rows = []
    for seq, vec in zip(train_sequences, train_vectors):
        tfidf_rows.append({"seq_id": seq.seq_id, "tfidf": vec})
    write_jsonl("data/processed/tfidf_train.jsonl", tfidf_rows)

    test_sequences = None
    test_vectors = None
    if test_glob:
        test_events = load_events(config, override_glob=test_glob)
        test_sequences = segment_sequences(test_events, config)
        test_vectors = [vectorize_sequence(seq, idf) for seq in test_sequences]

        test_rows = []
        for seq in test_sequences:
            test_rows.append({
                "seq_id": seq.seq_id,
                "start_ts": seq.start_ts,
                "end_ts": seq.end_ts,
                "alarm_counts": dict(seq.alarm_counts),
                "root_cause": seq.root_cause,
                "event_count": len(seq.events),
            })
        write_jsonl("data/processed/sequences_test.jsonl", test_rows)

        tfidf_test_rows = []
        for seq, vec in zip(test_sequences, test_vectors):
            tfidf_test_rows.append({"seq_id": seq.seq_id, "tfidf": vec})
        write_jsonl("data/processed/tfidf_test.jsonl", tfidf_test_rows)

    matches_path = "data/processed/matches.csv"
    with open(matches_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "seq_id",
                "best_match_id",
                "distance",
                "jaccard",
                "root_cause",
                "match_root_cause",
            ],
        )
        writer.writeheader()
        if test_sequences is None:
            for i, seq in enumerate(train_sequences):
                candidate_indices = [j for j in range(len(train_sequences)) if j != i]
                candidate_seqs = [train_sequences[j] for j in candidate_indices]
                candidate_vecs = [train_vectors[j] for j in candidate_indices]
                match_id, dist, jac = best_match(
                    seq,
                    train_vectors[i],
                    candidate_seqs,
                    candidate_vecs,
                    float(config.get("jaccard_threshold", 0.6)),
                )
                mapped_match_id = (
                    candidate_indices[match_id] if match_id is not None else None
                )
                # map match_id to original index if needed
                writer.writerow({
                    "seq_id": seq.seq_id,
                    "best_match_id": mapped_match_id,
                    "distance": dist,
                    "jaccard": jac,
                    "root_cause": seq.root_cause,
                    "match_root_cause": train_sequences[mapped_match_id].root_cause
                    if mapped_match_id is not None
                    else None,
                })
        else:
            for i, seq in enumerate(test_sequences):
                match_id, dist, jac = best_match(
                    seq,
                    test_vectors[i],
                    train_sequences,
                    train_vectors,
                    float(config.get("jaccard_threshold", 0.6)),
                )
                writer.writerow({
                    "seq_id": seq.seq_id,
                    "best_match_id": match_id,
                    "distance": dist,
                    "jaccard": jac,
                    "root_cause": seq.root_cause,
                    "match_root_cause": train_sequences[match_id].root_cause if match_id is not None else None,
                })

    print(f"Train sequences: {len(train_sequences)}")
    print(f"Wrote: data/processed/sequences_train.jsonl")
    print(f"Wrote: data/processed/tfidf_train.jsonl")
    if test_sequences is not None:
        print(f"Test sequences: {len(test_sequences)}")
        print(f"Wrote: data/processed/sequences_test.jsonl")
        print(f"Wrote: data/processed/tfidf_test.jsonl")
    print(f"Wrote: {matches_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
