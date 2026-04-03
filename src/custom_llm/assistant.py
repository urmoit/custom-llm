from __future__ import annotations

from dataclasses import dataclass
import json
import re
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from .config import (
    DOCUMENTS_FILE,
    LOCAL_SIMILARITY_THRESHOLD,
    MATRIX_FILE,
    MODEL_META_FILE,
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


class SmartAssistant:
    def __init__(self) -> None:
        self.vectorizer = None
        self.doc_matrix = None
        self.documents: List[Dict[str, str]] = []
        self.model_meta: Dict[str, str] = {"backend": "tfidf"}
        self.embedder = None
        self.length_context: Optional[LengthContext] = None
        self._load_artifacts()

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

    def _conversation_answer(self, normalized_q: str, raw_question: str) -> Optional[AnswerResult]:
        if any(
            phrase in normalized_q
            for phrase in [
                "what llm are you",
                "what model are you",
                "what is your name",
                "who are you",
                "llm name",
            ]
        ):
            return AnswerResult(
                answer="I am Custom LLM, a local retrieval-based assistant built for this workspace.",
                used_web=False,
                source="conversation",
                thinking="Answered the assistant identity question locally.",
                confidence=1.0,
            )

        if normalized_q in {"hi", "hello", "hey", "yo", "sup", "hola"}:
            return AnswerResult(
                answer="Hi. I am ready to chat. Ask me anything.",
                used_web=False,
                source="conversation",
                thinking="Handled a conversational greeting locally.",
                confidence=1.0,
            )

        if "how are you" in normalized_q:
            return AnswerResult(
                answer="I am doing well and ready to help. What do you want to talk about?",
                used_web=False,
                source="conversation",
                thinking="Handled small talk locally.",
                confidence=1.0,
            )

        if "what can you do" in normalized_q or "help me" in normalized_q:
            return AnswerResult(
                answer=(
                    "I can chat, explain topics from local knowledge, convert length units, "
                    "and do web fallback when needed."
                ),
                used_web=False,
                source="conversation",
                thinking="Returned core assistant capabilities.",
                confidence=1.0,
            )

        if normalized_q in {"thanks", "thank you", "thx"}:
            return AnswerResult(
                answer="You are welcome.",
                used_web=False,
                source="conversation",
                thinking="Handled courtesy response locally.",
                confidence=1.0,
            )

        if normalized_q in {"bye", "goodbye", "see you"}:
            return AnswerResult(
                answer="Bye. Talk to you soon.",
                used_web=False,
                source="conversation",
                thinking="Handled farewell locally.",
                confidence=1.0,
            )

        lower_raw = raw_question.lower().strip()
        if lower_raw.startswith("fix grammar:") or lower_raw.startswith("correct grammar:"):
            parts = raw_question.split(":", 1)
            source = parts[1].strip() if len(parts) > 1 else ""
            rewritten = self._basic_rewrite(source)
            if rewritten:
                return AnswerResult(
                    answer=f"Basic grammar rewrite:\n{rewritten}",
                    used_web=False,
                    source="conversation",
                    thinking="Applied basic grammar cleanup rules.",
                    confidence=1.0,
                )

        return None

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
                "}\n"
                "\n"
                "def pick_category(ext: str) -> str:\n"
                "    for name, exts in CATEGORY_MAP.items():\n"
                "        if ext in exts:\n"
                "            return name\n"
                "    return 'other'\n"
                "\n"
                "def organize(root: Path) -> dict[str, int]:\n"
                "    counts: dict[str, int] = {'images': 0, 'docs': 0, 'archives': 0, 'other': 0}\n"
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
                "        shutil.move(str(item), str(target))\n"
                "        counts[category] += 1\n"
                "    return counts\n"
                "\n"
                "def main() -> None:\n"
                "    parser = argparse.ArgumentParser(description='Organize files by extension')\n"
                "    parser.add_argument('path', help='Target folder path')\n"
                "    args = parser.parse_args()\n"
                "\n"
                "    root = Path(args.path).expanduser().resolve()\n"
                "    if not root.exists() or not root.is_dir():\n"
                "        raise SystemExit(f'Invalid folder: {root}')\n"
                "\n"
                "    counts = organize(root)\n"
                "    print('Organizing complete:')\n"
                "    for key, value in counts.items():\n"
                "        print(f'- {key}: {value}')\n"
                "\n"
                "if __name__ == '__main__':\n"
                "    main()\n"
            )
            return AnswerResult(
                answer=(
                    "Here is an intermediate Python file organizer script:\n\n"
                    f"{code}\n"
                    "Usage: python organizer.py ./target_folder"
                ),
                used_web=False,
                source="python-script-generator",
                thinking="Generated a non-basic local Python script from your request.",
                confidence=1.0,
            )

        if any(k in normalized_q for k in ["api", "rest", "http client", "fetch data"]):
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
                "                time.sleep(attempt)\n"
                "    raise RuntimeError(f'Failed after retries: {last_error}')\n"
                "\n"
                "def main() -> None:\n"
                "    parser = argparse.ArgumentParser(description='Fetch API JSON and save it')\n"
                "    parser.add_argument('url', help='API endpoint URL')\n"
                "    parser.add_argument('-o', '--output', default='api_output.json', help='Output JSON file')\n"
                "    args = parser.parse_args()\n"
                "\n"
                "    data = fetch_json(args.url)\n"
                "    with open(args.output, 'w', encoding='utf-8') as f:\n"
                "        json.dump(data, f, indent=2, ensure_ascii=False)\n"
                "    print(f'Saved response to {args.output}')\n"
                "\n"
                "if __name__ == '__main__':\n"
                "    main()\n"
            )
            return AnswerResult(
                answer=(
                    "Here is an intermediate Python REST API client script:\n\n"
                    f"{code}\n"
                    "Usage: python api_client.py https://api.example.com/data -o output.json"
                ),
                used_web=False,
                source="python-script-generator",
                thinking="Generated a non-basic local Python script from your request.",
                confidence=1.0,
            )

        if any(k in normalized_q for k in ["log analyzer", "analyze logs", "parse logs"]):
            code = (
                "import argparse\n"
                "import re\n"
                "from collections import Counter\n"
                "\n"
                "LEVEL_RE = re.compile(r'\\b(INFO|WARNING|ERROR|CRITICAL)\\b')\n"
                "\n"
                "def analyze_log(path: str) -> tuple[Counter, Counter]:\n"
                "    levels = Counter()\n"
                "    errors = Counter()\n"
                "\n"
                "    with open(path, 'r', encoding='utf-8', errors='replace') as f:\n"
                "        for line in f:\n"
                "            m = LEVEL_RE.search(line)\n"
                "            if not m:\n"
                "                continue\n"
                "            level = m.group(1)\n"
                "            levels[level] += 1\n"
                "            if level in {'ERROR', 'CRITICAL'}:\n"
                "                errors[line.strip()] += 1\n"
                "\n"
                "    return levels, errors\n"
                "\n"
                "def main() -> None:\n"
                "    parser = argparse.ArgumentParser(description='Analyze log levels and top errors')\n"
                "    parser.add_argument('file', help='Path to log file')\n"
                "    args = parser.parse_args()\n"
                "\n"
                "    levels, errors = analyze_log(args.file)\n"
                "    print('Level counts:')\n"
                "    for key in ['INFO', 'WARNING', 'ERROR', 'CRITICAL']:\n"
                "        print(f'- {key}: {levels.get(key, 0)}')\n"
                "\n"
                "    print('Top error lines:')\n"
                "    for msg, count in errors.most_common(5):\n"
                "        print(f'- {count}x {msg}')\n"
                "\n"
                "if __name__ == '__main__':\n"
                "    main()\n"
            )
            return AnswerResult(
                answer=(
                    "Here is an intermediate Python log analyzer script:\n\n"
                    f"{code}\n"
                    "Usage: python log_analyzer.py app.log"
                ),
                used_web=False,
                source="python-script-generator",
                thinking="Generated a non-basic local Python script from your request.",
                confidence=1.0,
            )

        if "calculator" in normalized_q:
            code = (
                "# Basic command-line calculator\n"
                "def calculate(a: float, op: str, b: float) -> float:\n"
                "    if op == '+':\n"
                "        return a + b\n"
                "    if op == '-':\n"
                "        return a - b\n"
                "    if op == '*':\n"
                "        return a * b\n"
                "    if op == '/':\n"
                "        if b == 0:\n"
                "            raise ValueError('Division by zero is not allowed.')\n"
                "        return a / b\n"
                "    raise ValueError('Unsupported operator. Use +, -, *, or /.')\n"
                "\n"
                "if __name__ == '__main__':\n"
                "    left = float(input('First number: '))\n"
                "    operator = input('Operator (+ - * /): ').strip()\n"
                "    right = float(input('Second number: '))\n"
                "    print('Result:', calculate(left, operator, right))\n"
            )
            return AnswerResult(
                answer=f"Here is a basic Python calculator script:\n\n{code}",
                used_web=False,
                source="python-script-generator",
                thinking="Generated a local starter Python script from your request.",
                confidence=1.0,
            )

        if "todo" in normalized_q or "to do" in normalized_q:
            code = (
                "# Basic in-memory to-do list\n"
                "tasks = []\n"
                "\n"
                "def show_tasks() -> None:\n"
                "    if not tasks:\n"
                "        print('No tasks yet.')\n"
                "        return\n"
                "    for i, task in enumerate(tasks, start=1):\n"
                "        print(f'{i}. {task}')\n"
                "\n"
                "while True:\n"
                "    cmd = input('Command (add/list/done/quit): ').strip().lower()\n"
                "    if cmd == 'add':\n"
                "        tasks.append(input('Task: ').strip())\n"
                "    elif cmd == 'list':\n"
                "        show_tasks()\n"
                "    elif cmd == 'done':\n"
                "        show_tasks()\n"
                "        idx = int(input('Task number done: ')) - 1\n"
                "        if 0 <= idx < len(tasks):\n"
                "            print('Done:', tasks.pop(idx))\n"
                "        else:\n"
                "            print('Invalid number')\n"
                "    elif cmd == 'quit':\n"
                "        break\n"
                "    else:\n"
                "        print('Unknown command')\n"
            )
            return AnswerResult(
                answer=f"Here is a basic Python to-do script:\n\n{code}",
                used_web=False,
                source="python-script-generator",
                thinking="Generated a local starter Python script from your request.",
                confidence=1.0,
            )

        code = (
            "import argparse\n"
            "import json\n"
            "from datetime import datetime\n"
            "\n"
            "def run(task: str, output: str) -> None:\n"
            "    payload = {\n"
            "        'task': task,\n"
            "        'status': 'ok',\n"
            "        'generated_at': datetime.utcnow().isoformat() + 'Z'\n"
            "    }\n"
            "    with open(output, 'w', encoding='utf-8') as f:\n"
            "        json.dump(payload, f, indent=2)\n"
            "    print(f'Saved result to {output}')\n"
            "\n"
            "def main() -> None:\n"
            "    parser = argparse.ArgumentParser(description='Intermediate Python script starter')\n"
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
                "Here is an intermediate Python starter script. If you tell me the exact task, "
                "I can generate a specialized script (API, logs, file organizer, and more).\n\n"
                f"{code}"
            ),
            used_web=False,
            source="python-script-generator",
            thinking="Generated a non-basic local Python script from your request.",
            confidence=1.0,
        )

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

    @staticmethod
    def _save_memory(question: str, result: AnswerResult) -> None:
        try:
            append_chat_memory(
                question=question,
                answer=result.answer,
                source=result.source,
                used_web=result.used_web,
                thinking=result.thinking,
                confidence=result.confidence,
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
            answer.append("\nRelated sections I can explain:")
            answer.extend([f"- {topic}" for topic in related[:2]])
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

    def ask(self, question: str, force_web: bool = False) -> AnswerResult:
        normalized = question.strip().lower()
        normalized_q = self._normalize_question(question)

        tiny_result = try_handle_tiny_task(question, normalized_q)
        if tiny_result is not None:
            result = AnswerResult(
                answer=tiny_result,
                used_web=False,
                source="tiny-model-tasks",
                thinking="Handled request with lightweight local tiny-task logic.",
                confidence=1.0,
            )
            self._save_memory(question, result)
            return result

        script_answer = self._python_script_answer(normalized_q, question)
        if script_answer is not None:
            self._save_memory(question, script_answer)
            return script_answer

        convo = self._conversation_answer(normalized_q, question)
        if convo is not None:
            self._save_memory(question, convo)
            return convo

        usa_section_match = self._find_best_section_match("united states", question)
        if usa_section_match is not None:
            result = AnswerResult(
                answer=usa_section_match,
                used_web=False,
                source="usa-knowledge",
                thinking="Matched your question to a specific USA section in local knowledge.",
                confidence=1.0,
            )
            self._save_memory(question, result)
            return result

        if (
            ("usa" in normalized_q or "united states" in normalized_q or "america" in normalized_q)
            and ("everything" in normalized_q or "all" in normalized_q)
            and ("know" in normalized_q or "about" in normalized_q)
        ):
            full = self._find_section("united states", "full summary")
            if full:
                result = AnswerResult(
                    answer=full,
                    used_web=False,
                    source="usa-knowledge",
                    thinking="Used deep local USA summary knowledge.",
                    confidence=1.0,
                )
                self._save_memory(question, result)
                return result

        if (
            ("usa" in normalized_q or "united states" in normalized_q or "america" in normalized_q)
            and ("cities" in normalized_q)
            and ("biggest" in normalized_q or "largest" in normalized_q)
        ):
            result = AnswerResult(
                answer=(
                    "The biggest U.S. cities are New York City, Los Angeles, Chicago, "
                    "Houston, and Phoenix."
                ),
                used_web=False,
                source="usa-knowledge",
                thinking="Handled a common USA city question with local knowledge.",
                confidence=1.0,
            )
            self._save_memory(question, result)
            return result

        if ("usa" in normalized_q or "united states" in normalized_q or "america" in normalized_q) and (
            "history" in normalized_q or "timeline" in normalized_q
        ):
            history = self._find_section("united states", "short history timeline")
            if history:
                result = AnswerResult(
                    answer=history,
                    used_web=False,
                    source="usa-knowledge",
                    thinking="Matched your question to local USA history knowledge.",
                    confidence=1.0,
                )
                self._save_memory(question, result)
                return result

        length_answer, updated_context = try_answer_length_question(
            question, context=self.length_context
        )
        self.length_context = updated_context
        if length_answer is not None:
            result = AnswerResult(
                answer=length_answer,
                used_web=False,
                source="length-engine",
                thinking="Used the built-in length knowledge and conversion rules.",
                confidence=1.0,
            )
            self._save_memory(question, result)
            return result

        if force_web:
            results = web_search(question)
            result = AnswerResult(
                answer=format_web_results(results, question),
                used_web=True,
                source="web-search",
                thinking="You forced web mode, so I looked up web sources directly.",
                confidence=None,
            )
            self._save_memory(question, result)
            return result

        local = self._local_answer(question)
        if local is not None:
            text, score = local
            result = AnswerResult(
                answer=text,
                used_web=False,
                source="local-knowledge",
                thinking="Matched your question against trained local knowledge.",
                confidence=score,
            )
            self._save_memory(question, result)
            return result

        results = web_search(question)
        answer = (
            "I do not have a strong local answer yet, so I searched the web.\n\n"
            + format_web_results(results, question)
        )
        result = AnswerResult(
            answer=answer,
            used_web=True,
            source="web-search",
            thinking="Local confidence was low, so I fell back to web search.",
            confidence=None,
        )
        self._save_memory(question, result)
        return result
