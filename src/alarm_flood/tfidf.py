import math
from typing import Dict, List, Tuple

from .segment import Sequence


def build_tfidf(sequences: List[Sequence]) -> Tuple[Dict[str, float], List[Dict[str, float]]]:
    # document frequency per alarm
    df: Dict[str, int] = {}
    for seq in sequences:
        for alarm_id in seq.alarm_counts.keys():
            df[alarm_id] = df.get(alarm_id, 0) + 1

    n_docs = len(sequences)
    idf: Dict[str, float] = {}
    for alarm_id, count in df.items():
        idf[alarm_id] = math.log((n_docs + 1) / (count + 1)) + 1.0

    vectors: List[Dict[str, float]] = []
    for seq in sequences:
        vectors.append(vectorize_sequence(seq, idf))

    return idf, vectors


def vectorize_sequence(seq: Sequence, idf: Dict[str, float]) -> Dict[str, float]:
    vec: Dict[str, float] = {}
    for alarm_id, tf in seq.alarm_counts.items():
        vec[alarm_id] = tf * idf.get(alarm_id, 0.0)
    return vec


def euclidean_distance(v1: Dict[str, float], v2: Dict[str, float]) -> float:
    keys = set(v1.keys()) | set(v2.keys())
    s = 0.0
    for k in keys:
        d = v1.get(k, 0.0) - v2.get(k, 0.0)
        s += d * d
    return math.sqrt(s)
