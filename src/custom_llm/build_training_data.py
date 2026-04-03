from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

from .config import KNOWLEDGE_DIR, TRAIN_DATA_FILE
from .memory_store import load_chat_memory_records


def _normalize_text(lines: List[str]) -> str:
    text = " ".join(line.strip() for line in lines if line.strip())
    return " ".join(text.split())


def _extract_sections(path: Path) -> Tuple[str, List[Tuple[str, str]]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    topic = path.stem.replace("_", " ").title()
    sections: List[Tuple[str, str]] = []

    current_section = "Overview"
    buf: List[str] = []

    for raw in lines:
        line = raw.strip()
        if not line:
            if buf:
                text = _normalize_text(buf)
                if text:
                    sections.append((current_section, text))
                buf = []
            continue

        if line.startswith("# "):
            topic = line[2:].strip()
            continue

        if line.startswith("## "):
            if buf:
                text = _normalize_text(buf)
                if text:
                    sections.append((current_section, text))
                buf = []
            current_section = line[3:].strip()
            continue

        buf.append(line)

    if buf:
        text = _normalize_text(buf)
        if text:
            sections.append((current_section, text))

    return topic, sections


def _make_chunks(topic: str, sections: List[Tuple[str, str]]) -> List[Dict[str, str]]:
    chunks: List[Dict[str, str]] = []

    full_summary = "\n\n".join(f"{name}: {text}" for name, text in sections)
    chunks.append(
        {
            "topic": topic,
            "section": "Full Summary",
            "text": f"Topic: {topic}\nSection: Full Summary\n{full_summary}",
        }
    )

    for section, text in sections:
        chunks.append(
            {
                "topic": topic,
                "section": section,
                "text": f"Topic: {topic}\nSection: {section}\n{text}",
            }
        )

    return chunks


def _make_synthetic_examples() -> List[Dict[str, str]]:
    topic_specs = [
        (
            "Conversation Basics",
            [
                "the user says hello",
                "the user says thanks",
                "the user says goodbye",
                "the message is vague",
                "the user asks how you are",
                "the user asks what you can do",
                "the user wants grammar cleanup",
                "the user writes with slang like u and thx",
                "the user has a short typo",
                "the user wants one short follow-up",
            ],
            [
                ("Brief greeting", "reply with a short friendly greeting."),
                ("Clarify", "ask one short clarifying follow-up."),
                ("Small talk", "answer naturally and stay helpful."),
                ("Grammar cleanup", "clean up the wording lightly without changing meaning."),
            ],
        ),
        (
            "General Knowledge Basics",
            [
                "weather versus climate",
                "percent and average",
                "the internet versus the web",
                "CPU, RAM, and storage",
                "variables and functions",
                "loops and conditionals",
                "Python script entry points",
                "debugging a failure",
                "JSON, CSV, and YAML",
                "basic security hygiene",
            ],
            [
                ("Simple explain", "explain the idea in plain language."),
                ("Concrete example", "give one concrete example."),
                ("Compare", "compare it with a related idea."),
                ("Best practice", "state one practical best practice."),
            ],
        ),
        (
            "Python",
            [
                "indentation and blocks",
                "if __name__ == '__main__'",
                "lists and dicts",
                "functions",
                "loops",
                "exceptions",
                "modules and packages",
                "virtual environments",
                "type hints",
                "script layout",
            ],
            [
                ("Simple explain", "explain the concept simply."),
                ("Tiny example", "show a tiny usage example."),
                ("Common pitfall", "mention one common pitfall."),
                ("Best practice", "give one coding best practice."),
            ],
        ),
        (
            "United States of America (USA)",
            [
                "the capital and currency",
                "states, districts, and territories",
                "geography",
                "government branches",
                "federalism and state powers",
                "the history timeline",
                "the economy",
                "technology and innovation",
                "the space program",
                "culture and society",
            ],
            [
                ("Brief fact", "state the fact briefly."),
                ("Broader context", "add one broader context point."),
                ("Regional variation", "mention one regional variation."),
                ("Practical implication", "mention one practical implication."),
            ],
        ),
        (
            "Tiny 1M-Style Model Capabilities",
            [
                "text classification",
                "intent detection",
                "fixed output formats",
                "simple extraction",
                "safety refusal",
                "short responses",
                "rule mapping",
                "ambiguity handling",
                "keyword matching",
                "broad topic repetition",
            ],
            [
                ("Classify", "classify the input."),
                ("Short reply", "keep the response short."),
                ("Fixed format", "use a fixed format."),
                ("Narrow answer", "prefer a narrow reliable answer."),
            ],
        ),
    ]

    examples: List[Dict[str, str]] = []
    counter = 1

    for topic, subjects, behaviors in topic_specs:
        for subject in subjects:
            for label, instruction in behaviors:
                section = f"Example {counter:03d} - {label}"
                text = (
                    f"Topic: {topic}\n"
                    f"Section: {section}\n"
                    f"When the user asks about {subject}, {instruction}"
                )
                examples.append(
                    {
                        "topic": topic,
                        "section": section,
                        "text": text,
                        "source": "synthetic/llm_200_examples",
                    }
                )
                counter += 1

    return examples


def _make_memory_examples() -> List[Dict[str, str]]:
    examples: List[Dict[str, str]] = []

    for index, item in enumerate(load_chat_memory_records(), start=1):
        question = str(item.get("question", "")).strip()
        answer = str(item.get("answer", "")).strip()
        if not question or not answer:
            continue

        topic = str(item.get("topic", "Chat Memory") or "Chat Memory").strip() or "Chat Memory"
        source = str(item.get("source", "memory/chat_memory.jsonl")).strip()
        section = f"Interaction {index:04d}"
        text = (
            f"Topic: {topic}\n"
            f"Section: {section}\n"
            f"User: {question}\n"
            f"Assistant: {answer}"
        )
        memory_key = str(item.get("memory_key", "")).strip()
        examples.append(
            {
                "topic": topic,
                "section": section,
                "text": text,
                "source": source,
                "memory_key": memory_key,
            }
        )

    return examples


def build_training_data() -> str:
    KNOWLEDGE_DIR.mkdir(parents=True, exist_ok=True)
    TRAIN_DATA_FILE.parent.mkdir(parents=True, exist_ok=True)

    all_chunks: List[Dict[str, str]] = []

    for path in sorted(KNOWLEDGE_DIR.rglob("*.md")):
        topic, sections = _extract_sections(path)
        if not sections:
            continue
        chunks = _make_chunks(topic, sections)
        rel_source = str(path.relative_to(KNOWLEDGE_DIR)).replace("\\", "/")
        for item in chunks:
            text = item.get("text", "")
            item["source"] = rel_source
            item["text"] = f"{text}\nSource: {rel_source}"
        all_chunks.extend(chunks)

    for item in _make_synthetic_examples():
        text = item.get("text", "").strip()
        if not text:
            continue
        item["text"] = f"{text}\nSource: {item.get('source', '')}"
        all_chunks.append(item)

    for item in _make_memory_examples():
        text = item.get("text", "").strip()
        if not text:
            continue
        item["text"] = f"{text}\nSource: {item.get('source', '')}"
        all_chunks.append(item)

    deduped: Dict[str, Dict[str, str]] = {}
    for item in all_chunks:
        topic = item.get("topic", "").strip()
        section = item.get("section", "").strip()
        text = item.get("text", "").strip()
        if not topic or not section or not text:
            continue
        source = item.get("source", "").strip().lower()
        memory_key = item.get("memory_key", "").strip().lower()
        if memory_key:
            key = f"memory::{memory_key}"
        else:
            key = f"{source}::{topic.lower()}::{section.lower()}"
        deduped[key] = {
            "topic": topic,
            "section": section,
            "text": text,
            "source": item.get("source", ""),
        }

    rows = list(deduped.values())
    with TRAIN_DATA_FILE.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")

    return (
        f"Built {len(rows)} trainable examples from knowledge files in '{KNOWLEDGE_DIR}' "
        f"and wrote '{TRAIN_DATA_FILE}'."
    )


if __name__ == "__main__":
    print(build_training_data())
