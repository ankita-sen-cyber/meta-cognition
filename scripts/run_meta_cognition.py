#!/usr/bin/env python
import argparse
import csv
import json
import os
import sys
import urllib.request
import urllib.error
from typing import Dict, List, Optional, Tuple


def load_jsonl(path: str) -> List[Dict]:
    rows = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def load_kb(path: Optional[str]) -> Dict[str, Dict]:
    if not path:
        return {}
    if not os.path.exists(path):
        return {}
    kb = {}
    for row in load_jsonl(path):
        key = str(row.get("root_cause"))
        kb[key] = row
    return kb


def _alarm_id_to_variable(alarm_id: str) -> str:
    if alarm_id.endswith("_high") or alarm_id.endswith("_low"):
        return alarm_id.rsplit("_", 1)[0]
    return alarm_id


def load_mapping(path: Optional[str]) -> Dict[str, Dict]:
    if not path or not os.path.exists(path):
        return {}
    mapping = {}
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            var = str(row.get("variable"))
            mapping[var] = {
                "variable": var,
                "group": row.get("group"),
                "arr17_tag": row.get("arr17_tag"),
                "description": row.get("description"),
                "unit": row.get("unit"),
            }
    return mapping


def load_thresholds(path: Optional[str]) -> Dict[str, Dict]:
    if not path or not os.path.exists(path):
        return {}
    thresholds = {}
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            var = str(row.get("variable"))
            thresholds[var] = {
                "lo_alarm": row.get("lo_alarm"),
                "hi_alarm": row.get("hi_alarm"),
                "normal": row.get("normal"),
            }
    return thresholds


def load_sequences(path: Optional[str]) -> Dict[str, Dict]:
    if not path or not os.path.exists(path):
        return {}
    seqs = {}
    for row in load_jsonl(path):
        seq_id = str(row.get("seq_id"))
        seqs[seq_id] = row
    return seqs


def build_sequence_summary(
    seq: Optional[Dict],
    mapping: Dict[str, Dict],
    thresholds: Dict[str, Dict],
    top_n: int,
) -> Dict:
    if not seq:
        return {}
    counts = seq.get("alarm_counts", {})
    if not isinstance(counts, dict):
        return {}

    items = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:top_n]
    alarms = []
    for alarm_id, count in items:
        var = _alarm_id_to_variable(str(alarm_id))
        meta = mapping.get(var, {})
        thr = thresholds.get(var, {})
        alarms.append(
            {
                "alarm_id": alarm_id,
                "count": count,
                "variable": var,
                "description": meta.get("description"),
                "group": meta.get("group"),
                "unit": meta.get("unit"),
                "normal": thr.get("normal"),
                "lo_alarm": thr.get("lo_alarm"),
                "hi_alarm": thr.get("hi_alarm"),
            }
        )

    summary = {
        "unique_alarms": len(counts),
        "event_count": seq.get("event_count"),
        "start_ts": seq.get("start_ts"),
        "end_ts": seq.get("end_ts"),
        "top_alarms": alarms,
    }
    return summary


def heuristic_meta(
    row: Dict,
    kb: Dict[str, Dict],
    seq_summary: Dict,
) -> Dict:
    # Simple fallback if no LLM is configured.
    dist = row.get("distance")
    jac = row.get("jaccard")
    confidence = None
    if dist is not None and jac is not None:
        try:
            # Lower is better for both; map to a rough 0-1 confidence.
            confidence = max(0.0, 1.0 - min(1.0, float(jac))) * 0.7
            confidence += max(0.0, 1.0 / (1.0 + float(dist))) * 0.3
            confidence = round(confidence, 3)
        except Exception:
            confidence = None

    root = str(row.get("match_root_cause"))
    kb_entry = kb.get(root, {})
    return {
        "seq_id": row.get("seq_id"),
        "predicted_root_cause": row.get("match_root_cause"),
        "true_root_cause": row.get("root_cause"),
        "confidence": confidence,
        "rationale": "Heuristic confidence based on distance and Jaccard.",
        "sequence_summary": seq_summary,
        "recommended_actions": kb_entry.get("actions"),
        "notes": kb_entry.get("notes"),
    }


def build_prompt(
    row: Dict,
    kb: Dict[str, Dict],
    seq_summary: Dict,
) -> List[Dict]:
    root = str(row.get("match_root_cause"))
    kb_entry = kb.get(root, {})
    return [
        {
            "role": "system",
            "content": (
                "You are a safety-critical alarm assistance meta-cognition module. "
                "Given a predicted root cause and similarity metrics, produce a JSON object "
                "with fields: predicted_root_cause, confidence (0-1), rationale, "
                "alternative_hypotheses (list), recommended_actions (list), and missing_info."
            ),
        },
        {
            "role": "user",
            "content": json.dumps(
                {
                    "seq_id": row.get("seq_id"),
                    "predicted_root_cause": row.get("match_root_cause"),
                    "true_root_cause": row.get("root_cause"),
                    "distance": row.get("distance"),
                    "jaccard": row.get("jaccard"),
                    "sequence_summary": seq_summary,
                    "knowledge": kb_entry,
                }
            ),
        },
    ]


def _call_openai(messages: List[Dict]) -> Optional[Dict]:
    api_key = os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1")
    model = os.getenv("LLM_MODEL", "gpt-4o-mini")
    if not api_key:
        return None

    payload = {"model": model, "messages": messages, "temperature": 0.2}
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{base_url}/chat/completions",
        data=data,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            resp_data = json.loads(resp.read().decode("utf-8"))
        content = resp_data["choices"][0]["message"]["content"]
        return json.loads(content)
    except Exception:
        return None


def _call_ollama(messages: List[Dict]) -> Optional[Dict]:
    base_url = os.getenv("OLLAMA_URL", "http://localhost:11434")
    model = os.getenv("OLLAMA_MODEL", "gemma3:4b")
    payload = {"model": model, "messages": messages, "stream": False}
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{base_url}/api/chat",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            resp_data = json.loads(resp.read().decode("utf-8"))
        content = resp_data.get("message", {}).get("content", "")
        return json.loads(content)
    except Exception:
        return None


def call_llm(messages: List[Dict]) -> Optional[Dict]:
    provider = os.getenv("LLM_PROVIDER")
    if provider == "ollama":
        return _call_ollama(messages)
    if provider == "openai":
        return _call_openai(messages)

    # Auto-detect: prefer Ollama if local vars are set
    if os.getenv("OLLAMA_URL") or os.getenv("OLLAMA_MODEL"):
        return _call_ollama(messages)
    if os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY"):
        return _call_openai(messages)
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description="Meta-cognition layer on top of matches.")
    ap.add_argument(
        "--matches",
        default="data/processed/matches_paper_digital_twin.csv",
        help="Matches CSV from the paper-style digital twin run.",
    )
    ap.add_argument(
        "--kb",
        default=None,
        help="Optional knowledge base JSONL keyed by root_cause.",
    )
    ap.add_argument(
        "--out",
        default="data/processed/meta_cognition.jsonl",
        help="Output JSONL",
    )
    ap.add_argument(
        "--sequences",
        default=None,
        help="Optional sequences JSONL to enrich prompts (e.g., sequences_test.jsonl).",
    )
    ap.add_argument(
        "--mapping",
        default="data/raw/te_variable_mapping.csv",
        help="Variable mapping CSV for alarm descriptions.",
    )
    ap.add_argument(
        "--thresholds",
        default="data/raw/te_alarm_thresholds.csv",
        help="Thresholds CSV for alarm limits.",
    )
    ap.add_argument(
        "--top-alarms",
        type=int,
        default=10,
        help="Number of top alarms to include in the sequence summary.",
    )
    ap.add_argument(
        "--use-llm",
        action="store_true",
        help="Use LLM_API_KEY/LLM_BASE_URL/LLM_MODEL to generate meta-cognition",
    )
    args = ap.parse_args()

    if not os.path.exists(args.matches):
        print(f"Missing matches file: {args.matches}", file=sys.stderr)
        return 1

    kb = load_kb(args.kb)
    mapping = load_mapping(args.mapping)
    thresholds = load_thresholds(args.thresholds)
    sequences = load_sequences(args.sequences)

    out_rows: List[Dict] = []
    with open(args.matches, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            seq = sequences.get(str(row.get("seq_id"))) if sequences else None
            seq_summary = build_sequence_summary(
                seq, mapping, thresholds, args.top_alarms
            )
            if args.use_llm:
                prompt = build_prompt(row, kb, seq_summary)
                resp = call_llm(prompt)
                if resp is None:
                    out_rows.append(heuristic_meta(row, kb, seq_summary))
                else:
                    resp["seq_id"] = row.get("seq_id")
                    resp["true_root_cause"] = row.get("root_cause")
                    out_rows.append(resp)
            else:
                out_rows.append(heuristic_meta(row, kb, seq_summary))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        for row in out_rows:
            f.write(json.dumps(row) + "\n")

    print(f"Wrote: {args.out}")
    if args.use_llm:
        print("LLM mode enabled. If output is missing, check API key and model.")
    else:
        print("Heuristic meta-cognition used. Add --use-llm to call a model.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
