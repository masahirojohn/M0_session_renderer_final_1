from __future__ import annotations
import json
from dataclasses import dataclass
from typing import Dict, List, Any

@dataclass
class Event:
    t_ms: int
    payload: Dict[str, Any]

class Timeline:
    def __init__(self, events: List[Event]):
        self.events = sorted(events, key=lambda e: e.t_ms)

    @staticmethod
    def load_json(path: str, key_map: Dict[str, str] | None = None) -> "Timeline":
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        events = []
        for item in raw:
            t = item.get("t_ms")
            if t is None:
                continue
            payload = {k: item[v] if key_map and v in item else item.get(k) for k, v in (key_map or {}).items()}
            # Fallback: if no key_map, take the whole item minus t_ms
            if not key_map:
                payload = {k: v for k, v in item.items() if k != "t_ms"}
            events.append(Event(int(t), payload))
        return Timeline(events)

    def value_at(self, t_ms: int) -> Dict[str, Any]:
        # Step function: last known payload at or before t_ms
        last = {}
        for e in self.events:
            if e.t_ms <= t_ms:
                last = e.payload
            else:
                break
        return last

    @staticmethod
    def merge_on_time(t_ms: int, *timelines: "Timeline") -> Dict[str, Any]:
        merged: Dict[str, Any] = {}
        for tl in timelines:
            if tl:
                merged.update(tl.value_at(t_ms))
        return merged
