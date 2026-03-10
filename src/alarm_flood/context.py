from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Set

import pandas as pd


@dataclass
class AlarmMeta:
    variable: str
    group: Optional[str]
    arr17_tag: Optional[str]
    description: Optional[str]


def _alarm_id_to_variable(alarm_id: str) -> str:
    # Expected alarm_id like "xmeas_1_high" or "xmv_2_low"
    if alarm_id.endswith("_high") or alarm_id.endswith("_low"):
        return alarm_id.rsplit("_", 1)[0]
    return alarm_id


def load_mapping(path: str) -> Dict[str, AlarmMeta]:
    df = pd.read_csv(path)
    meta: Dict[str, AlarmMeta] = {}
    for _, row in df.iterrows():
        var = str(row.get("variable"))
        meta[var] = AlarmMeta(
            variable=var,
            group=str(row.get("group")) if "group" in row else None,
            arr17_tag=str(row.get("arr17_tag")) if "arr17_tag" in row else None,
            description=str(row.get("description")) if "description" in row else None,
        )
    return meta


def sequence_components(alarm_ids: Set[str], mapping: Dict[str, AlarmMeta]) -> Set[str]:
    comps: Set[str] = set()
    for alarm_id in alarm_ids:
        var = _alarm_id_to_variable(alarm_id)
        meta = mapping.get(var)
        if meta and meta.arr17_tag:
            comps.add(meta.arr17_tag)
    return comps


def sequence_groups(alarm_ids: Set[str], mapping: Dict[str, AlarmMeta]) -> Set[str]:
    groups: Set[str] = set()
    for alarm_id in alarm_ids:
        var = _alarm_id_to_variable(alarm_id)
        meta = mapping.get(var)
        if meta and meta.group:
            groups.add(meta.group)
    return groups
