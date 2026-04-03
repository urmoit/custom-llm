from __future__ import annotations

import hashlib
import json
import threading
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from .config import MEMORY_FILE, MEMORY_DIR

_MEMORY_LOCK = threading.Lock()


def _normalize_text(text: str) -> str:
    return " ".join(text.strip().lower().split())


def _truncate(text: str, limit: int = 1800) -> str:
    cleaned = " ".join(text.strip().split())
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: limit - 3].rstrip() + "..."


def _make_memory_key(question: str, answer: str, source: str) -> str:
    payload = "\n".join(
        [
            _normalize_text(question),
            _normalize_text(answer),
            _normalize_text(source),
        ]
    )
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def _memory_file_for(timestamp: datetime, memory_key: str) -> Path:
    date_folder = MEMORY_DIR / timestamp.strftime("%Y-%m-%d")
    time_part = timestamp.strftime("%H%M%S_%f")
    return date_folder / f"{time_part}_{memory_key[:12]}.json"


def append_chat_memory(
    *,
    question: str,
    answer: str,
    source: str,
    used_web: bool,
    thinking: str,
    confidence: float | None,
) -> None:
    question = _truncate(question, limit=500)
    answer = _truncate(answer, limit=1800)
    thinking = _truncate(thinking, limit=500)
    source = _truncate(source, limit=120)

    if not question or not answer:
        return

    MEMORY_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().astimezone()
    memory_key = _make_memory_key(question, answer, source)

    record: Dict[str, object] = {
        "timestamp": timestamp.isoformat(),
        "date": timestamp.strftime("%Y-%m-%d"),
        "topic": "Chat Memory",
        "section": "Interaction",
        "question": question,
        "answer": answer,
        "source": source,
        "used_web": used_web,
        "thinking": thinking,
        "confidence": confidence,
        "memory_key": memory_key,
    }

    with _MEMORY_LOCK:
        memory_path = _memory_file_for(timestamp, memory_key)
        memory_path.parent.mkdir(parents=True, exist_ok=True)
        memory_path.write_text(json.dumps(record, ensure_ascii=True, indent=2), encoding="utf-8")

        legacy_line = json.dumps(record, ensure_ascii=True)
        with MEMORY_FILE.open("a", encoding="utf-8") as f:
            f.write(legacy_line + "\n")


def load_chat_memory_records() -> List[Dict[str, object]]:
    records: List[Dict[str, object]] = []

    if MEMORY_FILE.exists():
        with MEMORY_FILE.open("r", encoding="utf-8") as f:
            for raw in f:
                line = raw.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except Exception:
                    continue
                if isinstance(item, dict):
                    records.append(item)

    if MEMORY_DIR.exists():
        for path in sorted(MEMORY_DIR.rglob("*.json")):
            if path == MEMORY_FILE:
                continue
            try:
                item = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if isinstance(item, dict):
                item.setdefault("source", str(path.relative_to(MEMORY_DIR)).replace("\\", "/"))
                item.setdefault("memory_file", str(path.relative_to(MEMORY_DIR)).replace("\\", "/"))
                records.append(item)
    return records