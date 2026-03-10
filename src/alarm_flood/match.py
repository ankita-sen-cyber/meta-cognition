from typing import Dict, List, Optional, Tuple

from .segment import Sequence
from .tfidf import euclidean_distance


def jaccard_distance(a: Dict[str, int], b: Dict[str, int]) -> float:
    set_a = set(a.keys())
    set_b = set(b.keys())
    if not set_a and not set_b:
        return 0.0
    inter = len(set_a & set_b)
    union = len(set_a | set_b)
    return 1.0 - (inter / union)


def best_match(
    query_seq: Sequence,
    query_vec: Dict[str, float],
    candidates: List[Sequence],
    candidate_vecs: List[Dict[str, float]],
    jaccard_threshold: float,
) -> Tuple[Optional[int], float, float]:
    best_idx = None
    best_dist = float("inf")
    best_jacc = 1.0

    for i, seq in enumerate(candidates):
        d = euclidean_distance(query_vec, candidate_vecs[i])
        j = jaccard_distance(query_seq.alarm_counts, seq.alarm_counts)
        if j >= jaccard_threshold:
            continue
        if d < best_dist:
            best_dist = d
            best_idx = i
            best_jacc = j

    return best_idx, best_dist, best_jacc
