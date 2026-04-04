from __future__ import annotations

from dataclasses import dataclass
import json
import re
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from .config import (
    CUSTOM_LLM_FILE,
    DOCUMENTS_FILE,
    LOCAL_SIMILARITY_THRESHOLD,
    MATRIX_FILE,
    MODEL_META_FILE,
    TOKENIZER_FILE,
    TOP_K_LOCAL,
    VECTORIZER_FILE,
)
from .memory_store import append_chat_memory
from .length_knowledge import LengthContext, try_answer_length_question
from .search import format_web_results, web_search
from .tiny_tasks import try_handle_tiny_task


@dataclass
class AnswerResult:
    answer: str
    used_web: bool
    source: str
    thinking: str
    confidence: Optional[float] = None


# ---------------------------------------------------------------------------
# Conversation history for multi-turn context tracking
# ---------------------------------------------------------------------------
class ConversationHistory:
    def __init__(self, max_turns: int = 12):
        self.turns: List[Dict[str, str]] = []
        self.max_turns = max_turns

    def add(self, role: str, content: str) -> None:
        self.turns.append({"role": role, "content": content})
        if len(self.turns) > self.max_turns * 2:
            self.turns = self.turns[-self.max_turns * 2:]

    def last_user(self) -> str:
        for t in reversed(self.turns):
            if t["role"] == "user":
                return t["content"]
        return ""

    def last_bot(self) -> str:
        for t in reversed(self.turns):
            if t["role"] == "assistant":
                return t["content"]
        return ""

    def context_summary(self) -> str:
        recent = self.turns[-6:]
        return "\n".join(f"{t['role'].capitalize()}: {t['content'][:120]}" for t in recent)

    def has_context(self) -> bool:
        return len(self.turns) > 0


class SmartAssistant:
    def __init__(self) -> None:
        self.vectorizer = None
        self.doc_matrix = None
        self.documents: List[Dict[str, str]] = []
        self.model_meta: Dict[str, str] = {"backend": "tfidf"}
        self.embedder = None
        self.custom_llm = None
        self.custom_tokenizer = None
        self.length_context: Optional[LengthContext] = None
        self.conversation = ConversationHistory()
        self._load_artifacts()

    # ------------------------------------------------------------------
    # Text normalisation helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _normalize_question(text: str) -> str:
        text = text.lower().strip()
        text = text.replace("cityes", "cities")
        text = text.replace("bigest", "biggest")
        text = text.replace("amerca", "america")
        text = text.replace("uninted", "united")
        text = text.replace("everythiing", "everything")
        text = text.replace("kowladge", "knowledge")
        text = text.replace("knoe", "know")
        text = text.replace("ot ", "to ")
        text = text.replace(" u ", " you ")
        text = text.replace("wat ", "what ")
        text = text.replace("wats ", "whats ")
        text = text.replace("ai ", "ai ")
        text = text.replace("hw ", "how ")
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"[^a-z0-9 ?]", "", text)
        return text.strip()

    @staticmethod
    def _basic_rewrite(text: str) -> str:
        t = " ".join(text.strip().split())
        if not t:
            return ""
        replacements = {
            " u ": " you ",
            " ur ": " your ",
            " dont ": " don't ",
            " cant ": " can't ",
            " i ": " I ",
            " im ": " I'm ",
            " ive ": " I've ",
            " i ll ": " I'll ",
            " thx ": " thanks ",
        }
        padded = f" {t.lower()} "
        for src, dst in replacements.items():
            padded = padded.replace(src, dst)
        result = padded.strip()
        if result:
            result = result[0].upper() + result[1:]
        if result and result[-1] not in ".!?":
            result += "."
        return result

    # ------------------------------------------------------------------
    # Conversational response layer — improved multi-topic
    # ------------------------------------------------------------------
    def _conversation_answer(self, normalized_q: str, raw_question: str) -> Optional[AnswerResult]:
        q = normalized_q

        # Identity
        if any(p in q for p in ["what llm are you", "what model are you", "what is your name",
                                  "who are you", "llm name", "your name", "what are you"]):
            return AnswerResult(
                answer=(
                    "I am Custom LLM — a local retrieval-based assistant trained on a broad knowledge base "
                    "covering science, history, technology, geography, health, philosophy, economics, and more. "
                    "I can answer questions, explain concepts, write Python scripts, search the web when needed, "
                    "and hold a natural conversation. What would you like to explore?"
                ),
                used_web=False, source="conversation",
                thinking="Answered identity question with expanded description.", confidence=1.0,
            )

        # Greetings — varied responses
        if q in {"hi", "hello", "hey", "yo", "sup", "hola", "hi there", "hey there"}:
            import random
            greetings = [
                "Hello! What's on your mind?",
                "Hey there! How can I help you today?",
                "Hi! I'm ready to help — ask me anything.",
                "Hello! Feel free to ask about science, history, technology, or anything else.",
            ]
            return AnswerResult(
                answer=random.choice(greetings),
                used_web=False, source="conversation",
                thinking="Varied greeting response.", confidence=1.0,
            )

        # How are you
        if "how are you" in q:
            return AnswerResult(
                answer="I'm functioning well and ready to help! What would you like to know or discuss?",
                used_web=False, source="conversation",
                thinking="Small talk response.", confidence=1.0,
            )

        # What can you do — expanded
        if "what can you do" in q or ("help" in q and "me" in q and len(q) < 20):
            return AnswerResult(
                answer=(
                    "Here's what I can help with:\n\n"
                    "• Answer questions on science, history, technology, geography, health, "
                    "philosophy, economics, culture, and more\n"
                    "• Explain concepts at any depth — from simple overviews to technical detail\n"
                    "• Write and debug Python scripts (file organizers, API clients, log analyzers, etc.)\n"
                    "• Convert length and measurement units\n"
                    "• Classify text, detect intent, and handle NLP tasks\n"
                    "• Search the web for current information when my local knowledge isn't enough\n"
                    "• Hold a natural multi-turn conversation with context awareness\n\n"
                    "Just ask naturally — I'll do my best to help."
                ),
                used_web=False, source="conversation",
                thinking="Expanded capabilities description.", confidence=1.0,
            )

        # Capabilities follow-up
        if any(p in q for p in ["what do you know", "what topics", "what subjects", "knowledge base"]):
            return AnswerResult(
                answer=(
                    "My knowledge base covers:\n\n"
                    "Science: Physics, chemistry, biology, mathematics\n"
                    "Technology: AI/ML, software engineering, databases, networking, cloud\n"
                    "World knowledge: History, geography, countries (US, Estonia, UK, China, and more)\n"
                    "Health: Medicine, nutrition, fitness, mental health\n"
                    "Business: Economics, finance, entrepreneurship\n"
                    "Culture: Arts, music, film, literature, religion, media\n"
                    "Language: Linguistics, writing systems, world languages\n"
                    "Philosophy: Ethics, logic, epistemology, key thinkers\n"
                    "Programming: Python scripting, patterns, best practices\n\n"
                    "I also search the web when you need current or specific information."
                ),
                used_web=False, source="conversation",
                thinking="Detailed knowledge domain listing.", confidence=1.0,
            )

        # Thanks
        if q in {"thanks", "thank you", "thx", "ty", "thanks a lot", "thank you very much"}:
            return AnswerResult(
                answer="You're welcome! Let me know if there's anything else I can help with.",
                used_web=False, source="conversation",
                thinking="Polite acknowledgment.", confidence=1.0,
            )

        # Goodbye
        if q in {"bye", "goodbye", "see you", "see ya", "cya", "later"}:
            return AnswerResult(
                answer="Goodbye! Feel free to come back anytime you have questions.",
                used_web=False, source="conversation",
                thinking="Farewell response.", confidence=1.0,
            )

        # Tell me more / continue
        if any(p in q for p in ["tell me more", "go on", "continue", "elaborate", "expand on that",
                                  "more detail", "explain more"]):
            last_bot = self.conversation.last_bot()
            if last_bot:
                return AnswerResult(
                    answer=(
                        "To expand on what I said:\n\n"
                        + last_bot[:300]
                        + "\n\n...\n\nCould you tell me which specific part you'd like me to go deeper on? "
                        "That will help me give you the most useful detail."
                    ),
                    used_web=False, source="conversation",
                    thinking="Context-aware expansion request handled.", confidence=0.8,
                )

        # Grammar fix
        lower_raw = raw_question.lower().strip()
        if lower_raw.startswith("fix grammar:") or lower_raw.startswith("correct grammar:"):
            parts = raw_question.split(":", 1)
            source = parts[1].strip() if len(parts) > 1 else ""
            rewritten = self._basic_rewrite(source)
            if rewritten:
                return AnswerResult(
                    answer=f"Corrected:\n{rewritten}",
                    used_web=False, source="conversation",
                    thinking="Basic grammar cleanup applied.", confidence=1.0,
                )

        # Opinion/recommendation requests
        if any(p in q for p in ["what do you think", "your opinion", "recommend", "suggest",
                                  "which is better", "what should i"]):
            # Let this fall through to knowledge search — the knowledge base may have relevant content
            return None

        # Agreement/understanding checks
        if q in {"ok", "okay", "got it", "i see", "makes sense", "understood", "cool", "nice"}:
            import random
            acknowledgments = [
                "Great! Let me know if you have any other questions.",
                "Glad that makes sense! What else would you like to explore?",
                "Perfect! Feel free to ask about anything else.",
                "Awesome! I'm here if you need more help.",
            ]
            return AnswerResult(
                answer=random.choice(acknowledgments),
                used_web=False, source="conversation",
                thinking="User acknowledgment response.", confidence=1.0,
            )

        # Confusion/clarification requests
        if any(p in q for p in ["i don't understand", "confusing", "don't get it", "what do you mean",
                                  "can you explain", "help me understand"]):
            last_bot = self.conversation.last_bot()
            if last_bot:
                return AnswerResult(
                    answer=(
                        "Let me try to explain that differently:\n\n"
                        + last_bot[:250]
                        + "\n\nDoes that help clarify things? I can break it down further if needed."
                    ),
                    used_web=False, source="conversation",
                    thinking="Context-aware clarification request.", confidence=0.8,
                )

        # Casual chat patterns
        if any(p in q for p in ["you know", "i mean", "like", "um", "uh", "well", "so"]):
            # Check if it's a vague conversational prompt
            if len(q.split()) < 6 and not any(w in q for w in ["what", "how", "why", "when", "where", "who"]):
                return AnswerResult(
                    answer=(
                        "I'm here to help! Could you rephrase your question or tell me more about "
                        "what you'd like to know? I can discuss almost any topic."
                    ),
                    used_web=False, source="conversation",
                    thinking="Vague conversational prompt — asking for clarification.", confidence=0.7,
                )

        # Yes/no follow-ups
        if q in {"yes", "yeah", "yep", "sure", "absolutely", "definitely"}:
            last_bot = self.conversation.last_bot()
            if last_bot and "?" in last_bot:
                return AnswerResult(
                    answer=(
                        "Great! Based on what we discussed, is there anything specific you'd like "
                        "to explore further? I can go into more detail on any aspect."
                    ),
                    used_web=False, source="conversation",
                    thinking="Affirmative response to previous question.", confidence=0.8,
                )
            return AnswerResult(
                answer="Got it! What would you like to talk about next?",
                used_web=False, source="conversation",
                thinking="Simple affirmative response.", confidence=0.9,
            )

        if q in {"no", "nope", "nah", "not really"}:
            return AnswerResult(
                answer="No problem! What else can I help you with? I'm here for any questions or topics you'd like to explore.",
                used_web=False, source="conversation",
                thinking="Negative response — offering new direction.", confidence=0.9,
            )

        # Topic interest checks
        if any(p in q for p in ["interesting", "fascinating", "cool", "awesome", "amazing"]):
            return AnswerResult(
                answer=(
                    "Right? It's one of those topics that keeps getting more interesting the more you learn. "
                    "Would you like to dive deeper into any specific aspect, or explore something related?"
                ),
                used_web=False, source="conversation",
                thinking="User expressed interest — encouraging deeper exploration.", confidence=0.9,
            )

        return None

    # ------------------------------------------------------------------
    # Web search trigger logic — smart detection
    # ------------------------------------------------------------------
    def _should_use_web(self, normalized_q: str) -> bool:
        """Detect when web search would help regardless of local confidence."""
        web_triggers = [
            "latest", "recent", "news", "today", "current", "now", "2024", "2025", "2026",
            "stock price", "weather", "who won", "score", "live", "breaking",
            "new release", "just announced", "trending", "this week", "this month",
            "what happened", "update on", "status of",
        ]
        return any(t in normalized_q for t in web_triggers)

    # ------------------------------------------------------------------
    # Explain/definition handler
    # ------------------------------------------------------------------
    def _explanation_answer(self, normalized_q: str, raw_question: str) -> Optional[AnswerResult]:
        """Handle 'what is X', 'explain X', 'define X' type questions using local knowledge."""
        q = normalized_q
        explain_patterns = [
            r"what is (?:a |an |the )?(.+)",
            r"what are (.+)",
            r"explain (.+)",
            r"define (.+)",
            r"tell me about (.+)",
            r"describe (.+)",
            r"how does (.+) work",
        ]
        topic = None
        for pattern in explain_patterns:
            m = re.search(pattern, q)
            if m:
                topic = m.group(1).strip().rstrip("?").strip()
                break

        if not topic or len(topic) < 2:
            return None

        # Try to find in local knowledge base
        local = self._local_answer(raw_question)
        if local:
            text, score = local
            if score > 0.25:
                return None  # Let normal flow handle it with confidence score attached
        return None

    # ------------------------------------------------------------------
    # Python script generator — unchanged from original
    # ------------------------------------------------------------------
    def _python_script_answer(self, normalized_q: str, raw_question: str) -> Optional[AnswerResult]:
        wants_script = (
            "python script" in normalized_q
            or "write python" in normalized_q
            or "generate python" in normalized_q
            or "create python" in normalized_q
            or "script in python" in normalized_q
        )
        if not wants_script:
            return None

        if any(k in normalized_q for k in ["organize files", "file organizer", "sort files", "folder organizer"]):
            code = (
                "import argparse\n"
                "import shutil\n"
                "from pathlib import Path\n"
                "\n"
                "CATEGORY_MAP = {\n"
                "    'images': {'.png', '.jpg', '.jpeg', '.gif', '.webp'},\n"
                "    'docs': {'.pdf', '.docx', '.txt', '.md'},\n"
                "    'archives': {'.zip', '.rar', '.7z', '.tar', '.gz'},\n"
                "    'audio': {'.mp3', '.wav', '.flac', '.aac', '.ogg'},\n"
                "    'video': {'.mp4', '.avi', '.mkv', '.mov', '.wmv'},\n"
                "    'code': {'.py', '.js', '.ts', '.html', '.css', '.java', '.cpp'},\n"
                "}\n"
                "\n"
                "def pick_category(ext: str) -> str:\n"
                "    for name, exts in CATEGORY_MAP.items():\n"
                "        if ext in exts:\n"
                "            return name\n"
                "    return 'other'\n"
                "\n"
                "def organize(root: Path, dry_run: bool = False) -> dict[str, int]:\n"
                "    counts: dict[str, int] = {k: 0 for k in list(CATEGORY_MAP) + ['other']}\n"
                "    for item in root.iterdir():\n"
                "        if item.is_dir():\n"
                "            continue\n"
                "        category = pick_category(item.suffix.lower())\n"
                "        target_dir = root / category\n"
                "        target_dir.mkdir(exist_ok=True)\n"
                "        target = target_dir / item.name\n"
                "        index = 1\n"
                "        while target.exists():\n"
                "            target = target_dir / f'{item.stem}_{index}{item.suffix}'\n"
                "            index += 1\n"
                "        if dry_run:\n"
                "            print(f'[dry-run] Would move: {item.name} -> {category}/')\n"
                "        else:\n"
                "            shutil.move(str(item), str(target))\n"
                "        counts[category] += 1\n"
                "    return counts\n"
                "\n"
                "def main() -> None:\n"
                "    parser = argparse.ArgumentParser(description='Organize files by extension')\n"
                "    parser.add_argument('path', help='Target folder path')\n"
                "    parser.add_argument('--dry-run', action='store_true', help='Preview without moving')\n"
                "    args = parser.parse_args()\n"
                "\n"
                "    root = Path(args.path).expanduser().resolve()\n"
                "    if not root.exists() or not root.is_dir():\n"
                "        raise SystemExit(f'Invalid folder: {root}')\n"
                "\n"
                "    counts = organize(root, dry_run=args.dry_run)\n"
                "    label = '[dry-run] ' if args.dry_run else ''\n"
                "    print(f'{label}Organizing complete:')\n"
                "    for key, value in counts.items():\n"
                "        if value:\n"
                "            print(f'  {key}: {value} file(s)')\n"
                "\n"
                "if __name__ == '__main__':\n"
                "    main()\n"
            )
            return AnswerResult(
                answer=f"Here is a Python file organizer script (supports images, docs, audio, video, code, archives):\n\n{code}\nUsage: python organizer.py ./target_folder\nUsage: python organizer.py ./target_folder --dry-run",
                used_web=False, source="python-script-generator",
                thinking="Generated enhanced file organizer.", confidence=1.0,
            )

        if any(k in normalized_q for k in ["api", "rest", "http client", "fetch data", "request"]):
            code = (
                "import argparse\n"
                "import json\n"
                "import time\n"
                "from urllib.error import HTTPError, URLError\n"
                "from urllib.request import Request, urlopen\n"
                "\n"
                "def fetch_json(url: str, retries: int = 3, timeout: int = 10) -> dict:\n"
                "    last_error: Exception | None = None\n"
                "    for attempt in range(1, retries + 1):\n"
                "        try:\n"
                "            req = Request(url, headers={'User-Agent': 'custom-llm-client/1.0'})\n"
                "            with urlopen(req, timeout=timeout) as resp:\n"
                "                if resp.status != 200:\n"
                "                    raise RuntimeError(f'HTTP {resp.status}')\n"
                "                return json.loads(resp.read().decode('utf-8'))\n"
                "        except (HTTPError, URLError, TimeoutError, RuntimeError, json.JSONDecodeError) as exc:\n"
                "            last_error = exc\n"
                "            if attempt < retries:\n"
                "                wait = attempt * 2\n"
                "                print(f'Attempt {attempt} failed ({exc}). Retrying in {wait}s...')\n"
                "                time.sleep(wait)\n"
                "    raise RuntimeError(f'All {retries} attempts failed: {last_error}')\n"
                "\n"
                "def main() -> None:\n"
                "    parser = argparse.ArgumentParser(description='Fetch API JSON')\n"
                "    parser.add_argument('url', help='API endpoint URL')\n"
                "    parser.add_argument('-o', '--output', default='api_output.json')\n"
                "    parser.add_argument('--retries', type=int, default=3)\n"
                "    parser.add_argument('--timeout', type=int, default=10)\n"
                "    args = parser.parse_args()\n"
                "\n"
                "    data = fetch_json(args.url, retries=args.retries, timeout=args.timeout)\n"
                "    with open(args.output, 'w', encoding='utf-8') as f:\n"
                "        json.dump(data, f, indent=2, ensure_ascii=False)\n"
                "    print(f'Saved {len(str(data))} chars to {args.output}')\n"
                "\n"
                "if __name__ == '__main__':\n"
                "    main()\n"
            )
            return AnswerResult(
                answer=f"Here is a Python REST API client with retry logic:\n\n{code}\nUsage: python api_client.py https://api.example.com/data -o output.json",
                used_web=False, source="python-script-generator",
                thinking="Generated API client.", confidence=1.0,
            )

        if any(k in normalized_q for k in ["log analyzer", "analyze logs", "parse logs", "log parser"]):
            code = (
                "import argparse\n"
                "import re\n"
                "from collections import Counter\n"
                "from pathlib import Path\n"
                "\n"
                "LEVEL_RE = re.compile(r'\\b(DEBUG|INFO|WARNING|ERROR|CRITICAL)\\b')\n"
                "\n"
                "def analyze_log(path: str) -> tuple[Counter, Counter]:\n"
                "    levels: Counter = Counter()\n"
                "    errors: Counter = Counter()\n"
                "    with open(path, 'r', encoding='utf-8', errors='replace') as f:\n"
                "        for line in f:\n"
                "            m = LEVEL_RE.search(line)\n"
                "            if not m:\n"
                "                continue\n"
                "            level = m.group(1)\n"
                "            levels[level] += 1\n"
                "            if level in {'ERROR', 'CRITICAL'}:\n"
                "                errors[line.strip()] += 1\n"
                "    return levels, errors\n"
                "\n"
                "def main() -> None:\n"
                "    parser = argparse.ArgumentParser(description='Analyze log files')\n"
                "    parser.add_argument('file', help='Path to log file')\n"
                "    parser.add_argument('--top', type=int, default=5, help='Top N errors to show')\n"
                "    args = parser.parse_args()\n"
                "    if not Path(args.file).exists():\n"
                "        raise SystemExit(f'File not found: {args.file}')\n"
                "    levels, errors = analyze_log(args.file)\n"
                "    total = sum(levels.values())\n"
                "    print(f'Total log entries: {total}')\n"
                "    print('Level breakdown:')\n"
                "    for key in ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']:\n"
                "        count = levels.get(key, 0)\n"
                "        pct = 100 * count / total if total else 0\n"
                "        print(f'  {key:<10} {count:>6}  ({pct:.1f}%)')\n"
                "    print(f'Top {args.top} error lines:')\n"
                "    for msg, count in errors.most_common(args.top):\n"
                "        print(f'  [{count}x] {msg[:120]}')\n"
                "\n"
                "if __name__ == '__main__':\n"
                "    main()\n"
            )
            return AnswerResult(
                answer=f"Here is a Python log analyzer script:\n\n{code}\nUsage: python log_analyzer.py app.log --top 10",
                used_web=False, source="python-script-generator",
                thinking="Generated log analyzer.", confidence=1.0,
            )

        if any(k in normalized_q for k in ["csv", "spreadsheet", "csv report", "csv reader"]):
            code = (
                "import argparse\n"
                "import csv\n"
                "import json\n"
                "from collections import defaultdict\n"
                "\n"
                "def analyze_csv(path: str, group_col: str, value_col: str) -> dict:\n"
                "    groups: dict = defaultdict(list)\n"
                "    with open(path, newline='', encoding='utf-8') as f:\n"
                "        reader = csv.DictReader(f)\n"
                "        for row in reader:\n"
                "            key = row.get(group_col, 'UNKNOWN')\n"
                "            try:\n"
                "                groups[key].append(float(row[value_col]))\n"
                "            except (ValueError, KeyError):\n"
                "                pass\n"
                "    return {\n"
                "        k: {'count': len(v), 'total': sum(v), 'avg': sum(v)/len(v), 'min': min(v), 'max': max(v)}\n"
                "        for k, v in groups.items() if v\n"
                "    }\n"
                "\n"
                "def main() -> None:\n"
                "    parser = argparse.ArgumentParser(description='CSV report generator')\n"
                "    parser.add_argument('file', help='CSV file path')\n"
                "    parser.add_argument('--group', required=True, help='Column to group by')\n"
                "    parser.add_argument('--value', required=True, help='Numeric column to summarize')\n"
                "    parser.add_argument('--json', dest='as_json', action='store_true')\n"
                "    args = parser.parse_args()\n"
                "    results = analyze_csv(args.file, args.group, args.value)\n"
                "    if args.as_json:\n"
                "        print(json.dumps(results, indent=2))\n"
                "    else:\n"
                "        print(f'{'Group':<20} {'Count':>6} {'Total':>12} {'Avg':>10} {'Min':>10} {'Max':>10}')\n"
                "        print('-' * 72)\n"
                "        for k in sorted(results, key=lambda x: results[x]['total'], reverse=True):\n"
                "            r = results[k]\n"
                "            print(f'{k:<20} {r[\"count\"]:>6} {r[\"total\"]:>12.2f} {r[\"avg\"]:>10.2f} {r[\"min\"]:>10.2f} {r[\"max\"]:>10.2f}')\n"
                "\n"
                "if __name__ == '__main__':\n"
                "    main()\n"
            )
            return AnswerResult(
                answer=f"Here is a Python CSV report script:\n\n{code}\nUsage: python csv_report.py data.csv --group category --value amount",
                used_web=False, source="python-script-generator",
                thinking="Generated CSV report script.", confidence=1.0,
            )

        if any(k in normalized_q for k in ["sqlite", "task tracker", "todo", "database task"]):
            code = (
                "import argparse\n"
                "import sqlite3\n"
                "from pathlib import Path\n"
                "\n"
                "DB_FILE = 'tasks.db'\n"
                "\n"
                "def get_conn() -> sqlite3.Connection:\n"
                "    conn = sqlite3.connect(DB_FILE)\n"
                "    conn.execute('''CREATE TABLE IF NOT EXISTS tasks (\n"
                "        id INTEGER PRIMARY KEY AUTOINCREMENT,\n"
                "        title TEXT NOT NULL,\n"
                "        done INTEGER NOT NULL DEFAULT 0,\n"
                "        created_at TEXT DEFAULT (datetime('now'))\n"
                "    )''')\n"
                "    conn.commit()\n"
                "    return conn\n"
                "\n"
                "def list_tasks(conn: sqlite3.Connection) -> None:\n"
                "    rows = conn.execute('SELECT id, done, title FROM tasks ORDER BY id').fetchall()\n"
                "    if not rows:\n"
                "        print('No tasks.')\n"
                "        return\n"
                "    for row in rows:\n"
                "        status = '[x]' if row[1] else '[ ]'\n"
                "        print(f'  {row[0]:>3}. {status} {row[2]}')\n"
                "\n"
                "def main() -> None:\n"
                "    parser = argparse.ArgumentParser(description='SQLite task tracker')\n"
                "    sub = parser.add_subparsers(dest='cmd')\n"
                "    add_p = sub.add_parser('add', help='Add a task')\n"
                "    add_p.add_argument('title', nargs='+', help='Task title')\n"
                "    done_p = sub.add_parser('done', help='Mark task done')\n"
                "    done_p.add_argument('id', type=int)\n"
                "    del_p = sub.add_parser('delete', help='Delete a task')\n"
                "    del_p.add_argument('id', type=int)\n"
                "    sub.add_parser('list', help='List all tasks')\n"
                "    args = parser.parse_args()\n"
                "    conn = get_conn()\n"
                "    if args.cmd == 'add':\n"
                "        title = ' '.join(args.title)\n"
                "        conn.execute('INSERT INTO tasks (title) VALUES (?)', (title,))\n"
                "        conn.commit()\n"
                "        print(f'Added: {title}')\n"
                "    elif args.cmd == 'done':\n"
                "        conn.execute('UPDATE tasks SET done=1 WHERE id=?', (args.id,))\n"
                "        conn.commit()\n"
                "        print(f'Marked done: #{args.id}')\n"
                "    elif args.cmd == 'delete':\n"
                "        conn.execute('DELETE FROM tasks WHERE id=?', (args.id,))\n"
                "        conn.commit()\n"
                "        print(f'Deleted: #{args.id}')\n"
                "    else:\n"
                "        list_tasks(conn)\n"
                "    conn.close()\n"
                "\n"
                "if __name__ == '__main__':\n"
                "    main()\n"
            )
            return AnswerResult(
                answer=f"Here is a Python SQLite task tracker:\n\n{code}\nUsage: python tasks.py add Buy groceries\nUsage: python tasks.py list\nUsage: python tasks.py done 1",
                used_web=False, source="python-script-generator",
                thinking="Generated SQLite task tracker.", confidence=1.0,
            )

        # Generic starter
        code = (
            "import argparse\n"
            "import json\n"
            "from datetime import datetime\n"
            "from pathlib import Path\n"
            "\n"
            "def run(task: str, output: str) -> None:\n"
            "    payload = {\n"
            "        'task': task,\n"
            "        'status': 'ok',\n"
            "        'generated_at': datetime.utcnow().isoformat() + 'Z'\n"
            "    }\n"
            "    Path(output).parent.mkdir(parents=True, exist_ok=True)\n"
            "    with open(output, 'w', encoding='utf-8') as f:\n"
            "        json.dump(payload, f, indent=2)\n"
            "    print(f'Saved result to {output}')\n"
            "\n"
            "def main() -> None:\n"
            "    parser = argparse.ArgumentParser(description='Python script starter')\n"
            "    parser.add_argument('--task', default='demo_task', help='Task name')\n"
            "    parser.add_argument('--output', default='result.json', help='Output JSON file')\n"
            "    args = parser.parse_args()\n"
            "    run(args.task, args.output)\n"
            "\n"
            "if __name__ == '__main__':\n"
            "    main()\n"
        )
        return AnswerResult(
            answer=(
                "Here is a Python starter script. Tell me your specific task and I'll generate "
                "a specialized version (API client, file organizer, log analyzer, CSV report, "
                "SQLite tracker, and more).\n\n"
                f"{code}"
            ),
            used_web=False, source="python-script-generator",
            thinking="Generated starter script.", confidence=1.0,
        )

    # ------------------------------------------------------------------
    # Artifact loading
    # ------------------------------------------------------------------
    def _load_artifacts(self) -> None:
        if not (VECTORIZER_FILE.exists() and MATRIX_FILE.exists() and DOCUMENTS_FILE.exists()):
            self.vectorizer = None
            self.doc_matrix = None
            self.documents = []
            return

        self.vectorizer = joblib.load(VECTORIZER_FILE)
        self.doc_matrix = joblib.load(MATRIX_FILE)
        self.documents = joblib.load(DOCUMENTS_FILE)
        if MODEL_META_FILE.exists():
            try:
                self.model_meta = json.loads(MODEL_META_FILE.read_text(encoding="utf-8"))
            except Exception:
                self.model_meta = {"backend": "tfidf"}
        else:
            self.model_meta = {"backend": "tfidf"}

        if self.model_meta.get("backend") == "transformer":
            try:
                from sentence_transformers import SentenceTransformer  # type: ignore
                model_name = self.model_meta.get("model_name", "sentence-transformers/all-MiniLM-L6-v2")
                self.embedder = SentenceTransformer(model_name)
            except Exception:
                self.embedder = None
                self.model_meta = {"backend": "tfidf", "note": "Transformer runtime unavailable"}

        # Load fully custom LLM if available
        if self.model_meta.get("backend") == "custom":
            self._load_custom_llm()

    def _load_custom_llm(self) -> None:
        """Load the custom GPT model and tokenizer trained from scratch."""
        if not (CUSTOM_LLM_FILE.exists() and TOKENIZER_FILE.exists()):
            return
        try:
            import torch  # type: ignore
            from .model import CustomLanguageModel
            from .tokenizer import Tokenizer

            # weights_only=False is intentional: the checkpoint is produced locally
            # by train_custom_llm() and never downloaded from an external source,
            # so it is trusted.  Switch to weights_only=True if the checkpoint format
            # is ever changed to store only tensor state dicts.
            checkpoint = torch.load(CUSTOM_LLM_FILE, map_location="cpu", weights_only=False)
            cfg = checkpoint["config"]
            model = CustomLanguageModel(
                vocab_size=cfg["vocab_size"],
                d_model=cfg["d_model"],
                n_heads=cfg["n_heads"],
                n_layers=cfg["n_layers"],
                context_length=cfg["context_length"],
                d_ff=cfg["d_ff"],
            )
            model.load_state_dict(checkpoint["model_state"])
            model.eval()
            self.custom_llm = model
            self.custom_tokenizer = Tokenizer.load(TOKENIZER_FILE)
        except Exception:
            self.custom_llm = None
            self.custom_tokenizer = None

    @staticmethod
    def _save_memory(question: str, result: AnswerResult) -> None:
        try:
            append_chat_memory(
                question=question, answer=result.answer, source=result.source,
                used_web=result.used_web, thinking=result.thinking, confidence=result.confidence,
            )
        except Exception:
            pass

    def retrain_and_reload(self) -> str:
        from .build_training_data import build_training_data
        from .trainer import train_model
        build_msg = build_training_data()
        msg = train_model(backend="auto")
        self._load_artifacts()
        return build_msg + "\n" + msg

    # ------------------------------------------------------------------
    # Custom LLM generation (retrieval-augmented)
    # ------------------------------------------------------------------
    def _custom_llm_answer(self, question: str, context_text: str) -> Optional[AnswerResult]:
        """Generate an answer using the custom LLM with retrieved context.

        The prompt is formatted as::

            Context: {retrieved_text}
            Question: {question}
            Answer:

        The model then generates tokens until EOS or max_new_tokens.
        Falls back gracefully if the custom LLM is unavailable.
        """
        if self.custom_llm is None or self.custom_tokenizer is None:
            return None

        try:
            import torch  # type: ignore

            # Build RAG prompt: context + question
            prompt = (
                f"context: {context_text[:400]}\n"
                f"question: {question}\n"
                f"answer:"
            )
            prompt_ids = self.custom_tokenizer.encode(
                prompt, add_bos=True, add_eos=False,
                max_length=self.custom_llm.context_length - 60,
            )
            device_str = next(self.custom_llm.parameters()).device.type
            output_ids = self.custom_llm.generate(
                prompt_ids,
                max_new_tokens=80,
                temperature=0.7,
                top_k=40,
                device=device_str,
            )
            # Decode only the newly generated tokens (after the prompt)
            new_ids = output_ids[len(prompt_ids):]
            answer_text = self.custom_tokenizer.decode(new_ids, skip_special=True).strip()

            if not answer_text:
                return None

            return AnswerResult(
                answer=answer_text,
                used_web=False,
                source="custom-llm",
                thinking="Generated by fully custom transformer LLM (trained from scratch).",
                confidence=0.75,
            )
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Local knowledge retrieval
    # ------------------------------------------------------------------
    def _local_answer(self, question: str) -> Optional[Tuple[str, float]]:
        if self.doc_matrix is None or not self.documents:
            return None

        backend = self.model_meta.get("backend", "tfidf")
        if backend == "transformer" and self.embedder is not None:
            q_vec = self.embedder.encode(
                [question], normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False
            )[0]
            sims = np.dot(self.doc_matrix, q_vec).flatten()
        else:
            if self.vectorizer is None:
                return None
            q_vec = self.vectorizer.transform([question])
            sims = cosine_similarity(q_vec, self.doc_matrix).flatten()

        best_idx = int(sims.argmax())
        best_score = float(sims[best_idx])

        if best_score < LOCAL_SIMILARITY_THRESHOLD:
            return None

        ranked = sims.argsort()[::-1][:TOP_K_LOCAL]
        best_doc = self.documents[int(best_idx)]

        answer = [best_doc.get("text", "")]
        related = []
        for idx in ranked:
            if idx == best_idx:
                continue
            if float(sims[idx]) >= max(0.12, LOCAL_SIMILARITY_THRESHOLD * 0.6):
                doc = self.documents[int(idx)]
                topic = doc.get("topic", "")
                section = doc.get("section", "")
                label = ": ".join(part for part in [topic, section] if part).strip(": ")
                if label:
                    related.append(label)

        if related:
            answer.append("\nRelated topics I can explain:")
            answer.extend([f"- {topic}" for topic in related[:3]])
        return "\n".join(answer), best_score

    def _find_section(self, topic_contains: str, section_name: str) -> Optional[str]:
        t = topic_contains.lower().strip()
        s = section_name.lower().strip()
        for doc in self.documents:
            topic = str(doc.get("topic", "")).lower()
            section = str(doc.get("section", "")).lower()
            if t in topic and section == s:
                return str(doc.get("text", "")).strip()
        return None

    def _find_best_section_match(self, topic_contains: str, question: str) -> Optional[str]:
        t = topic_contains.lower().strip()
        q = self._normalize_question(question)

        alias_map = {
            "ongoing policy challenges": ["ongoing policy challenges", "policy challenges", "challenges"],
            "government and constitutional order": ["government", "federal government", "constitutional", "branches"],
            "economy": ["economy", "gdp", "economic"],
            "historical arc (high-level)": ["history", "timeline", "historical"],
            "population and demographics": ["population", "demographics"],
            "culture and society": ["culture", "society"],
            "sports": ["sports", "nfl", "nba", "mlb", "nhl", "mls"],
            "climate and biodiversity": ["climate", "biodiversity", "environment"],
            "cities and metropolitan areas": ["cities", "metro", "metropolitan"],
            "education and research institutions": ["education", "universities", "research"],
            "science, technology, and innovation": ["technology", "innovation", "science", "ai"],
            "military and foreign relations": ["military", "foreign relations", "nato", "defense"],
            "space program": ["space", "nasa"],
        }

        for doc in self.documents:
            topic = str(doc.get("topic", "")).lower()
            if t not in topic:
                continue
            section = str(doc.get("section", "")).lower()
            if section == "full summary":
                continue
            triggers = alias_map.get(section, [section])
            for trigger in triggers:
                if trigger in q:
                    return str(doc.get("text", "")).strip()
        return None

    # ------------------------------------------------------------------
    # Main ask method
    # ------------------------------------------------------------------
    def ask(self, question: str, force_web: bool = False) -> AnswerResult:
        normalized_q = self._normalize_question(question)

        # Record user turn
        self.conversation.add("user", question)

        # Step 1: Tiny tasks
        tiny_result = try_handle_tiny_task(question, normalized_q)
        if tiny_result is not None:
            result = AnswerResult(
                answer=tiny_result, used_web=False, source="tiny-model-tasks",
                thinking="Handled by lightweight tiny-task logic.", confidence=1.0,
            )
            self._save_memory(question, result)
            self.conversation.add("assistant", result.answer)
            return result

        # Step 2: Python scripts
        script_answer = self._python_script_answer(normalized_q, question)
        if script_answer is not None:
            self._save_memory(question, script_answer)
            self.conversation.add("assistant", script_answer.answer)
            return script_answer

        # Step 3: Conversational
        convo = self._conversation_answer(normalized_q, question)
        if convo is not None:
            self._save_memory(question, convo)
            self.conversation.add("assistant", convo.answer)
            return convo

        # Step 4: Force web
        if force_web or self._should_use_web(normalized_q):
            results = web_search(question)
            answer = format_web_results(results, question)
            if not force_web:
                answer = "Here's current information from the web:\n\n" + answer
            result = AnswerResult(
                answer=answer, used_web=True, source="web-search",
                thinking="Query required current/web information.",
                confidence=None,
            )
            self._save_memory(question, result)
            self.conversation.add("assistant", result.answer)
            return result

        # Step 5: USA section-specific matching
        usa_section_match = self._find_best_section_match("united states", question)
        if usa_section_match is not None:
            result = AnswerResult(
                answer=usa_section_match, used_web=False, source="usa-knowledge",
                thinking="Matched to specific USA section.", confidence=1.0,
            )
            self._save_memory(question, result)
            self.conversation.add("assistant", result.answer)
            return result

        # USA full summary
        if (
            ("usa" in normalized_q or "united states" in normalized_q or "america" in normalized_q)
            and ("everything" in normalized_q or "all" in normalized_q)
            and ("know" in normalized_q or "about" in normalized_q)
        ):
            full = self._find_section("united states", "full summary")
            if full:
                result = AnswerResult(
                    answer=full, used_web=False, source="usa-knowledge",
                    thinking="Used full USA summary.", confidence=1.0,
                )
                self._save_memory(question, result)
                self.conversation.add("assistant", result.answer)
                return result

        # USA biggest cities
        if (
            ("usa" in normalized_q or "united states" in normalized_q or "america" in normalized_q)
            and "cities" in normalized_q
            and ("biggest" in normalized_q or "largest" in normalized_q)
        ):
            result = AnswerResult(
                answer=(
                    "The largest U.S. cities by population:\n"
                    "1. New York City (~8.3M city, ~20M metro)\n"
                    "2. Los Angeles (~3.9M city)\n"
                    "3. Chicago (~2.7M city)\n"
                    "4. Houston (~2.3M city)\n"
                    "5. Phoenix (~1.6M city)\n"
                    "6. Philadelphia, San Antonio, San Diego, Dallas, San Jose follow.\n"
                    "Metro areas: New York, LA, Chicago, Dallas-Fort Worth, and Houston are the top 5."
                ),
                used_web=False, source="usa-knowledge",
                thinking="Local USA cities knowledge.", confidence=1.0,
            )
            self._save_memory(question, result)
            self.conversation.add("assistant", result.answer)
            return result

        # Step 6: Length conversions
        length_answer, updated_context = try_answer_length_question(question, context=self.length_context)
        self.length_context = updated_context
        if length_answer is not None:
            result = AnswerResult(
                answer=length_answer, used_web=False, source="length-engine",
                thinking="Answered using built-in length conversion.", confidence=1.0,
            )
            self._save_memory(question, result)
            self.conversation.add("assistant", result.answer)
            return result

        # Step 7: Local knowledge base + optional custom LLM generation
        local = self._local_answer(question)
        if local is not None:
            text, score = local

            # If the custom LLM is available, use retrieved context to generate a richer answer
            if self.custom_llm is not None:
                llm_result = self._custom_llm_answer(question, text)
                if llm_result is not None:
                    llm_result.confidence = score
                    self._save_memory(question, llm_result)
                    self.conversation.add("assistant", llm_result.answer)
                    return llm_result

            result = AnswerResult(
                answer=text, used_web=False, source="local-knowledge",
                thinking=f"Matched local knowledge (confidence: {score:.2f}).",
                confidence=score,
            )
            self._save_memory(question, result)
            self.conversation.add("assistant", result.answer)
            return result

        # Step 8: Web fallback
        results = web_search(question)
        answer = (
            "I don't have strong local knowledge on that, so I searched the web:\n\n"
            + format_web_results(results, question)
        )
        result = AnswerResult(
            answer=answer, used_web=True, source="web-search",
            thinking="Local knowledge insufficient; fell back to web search.",
            confidence=None,
        )
        self._save_memory(question, result)
        self.conversation.add("assistant", result.answer)
        return result
