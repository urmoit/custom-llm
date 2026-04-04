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


def _make_chunks(topic: str, sections: List[Tuple[str, str]], source: str = "") -> List[Dict[str, str]]:
    chunks: List[Dict[str, str]] = []

    full_summary = "\n\n".join(f"{name}: {text}" for name, text in sections)
    chunks.append({
        "topic": topic,
        "section": "Full Summary",
        "text": f"Topic: {topic}\nSection: Full Summary\n{full_summary}",
        "source": source,
    })

    for section, text in sections:
        chunks.append({
            "topic": topic,
            "section": section,
            "text": f"Topic: {topic}\nSection: {section}\n{text}",
            "source": source,
        })

    return chunks


def _make_synthetic_examples() -> List[Dict[str, str]]:
    """Extended synthetic examples covering expanded knowledge domains."""
    topic_specs = [
        # === EXISTING TOPICS (keeping for continuity) ===
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
                "the user writes with slang",
                "the user has a short typo",
                "the user wants more detail",
                "the user asks about your capabilities",
                "the user wants to chat casually",
                "the user asks for an opinion",
                "the user needs advice",
                "the user is frustrated",
                "the user is confused",
                "the user makes a joke",
                "the user is testing you",
                "the user changes topic mid-conversation",
                "the user references earlier conversation",
            ],
            [
                ("Brief greeting", "reply with a short friendly greeting."),
                ("Clarify", "ask one short clarifying follow-up."),
                ("Small talk", "answer naturally and stay helpful."),
                ("Grammar cleanup", "clean up the wording lightly."),
                ("Empathize", "acknowledge their feeling and stay supportive."),
                ("Engage", "keep the conversation going naturally."),
            ],
        ),
        (
            "General Knowledge",
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
                "decorators",
                "asyncio and concurrency",
                "testing with pytest",
                "dataclasses",
                "context managers",
                "performance optimization",
            ],
            [
                ("Simple explain", "explain the concept simply."),
                ("Tiny example", "show a tiny usage example."),
                ("Common pitfall", "mention one common pitfall."),
                ("Best practice", "give one coding best practice."),
            ],
        ),
        (
            "Science",
            [
                "Newton's laws of motion",
                "thermodynamics and entropy",
                "quantum mechanics basics",
                "relativity and E=mc2",
                "the periodic table",
                "chemical bonding",
                "DNA and genetics",
                "evolution and natural selection",
                "the Big Bang",
                "black holes",
                "climate change and greenhouse gases",
                "renewable energy",
                "cell biology",
                "the immune system",
                "calculus basics",
                "probability and statistics",
                "linear algebra",
            ],
            [
                ("Simple explain", "explain the concept simply."),
                ("Key fact", "state one key fact."),
                ("Real-world application", "give a real-world application."),
                ("Common misconception", "correct one common misconception."),
            ],
        ),
        (
            "World History",
            [
                "ancient civilizations",
                "the Roman Empire",
                "the Industrial Revolution",
                "World War I causes",
                "World War II and the Holocaust",
                "the Cold War",
                "the French Revolution",
                "decolonization",
                "the moon landing",
                "the internet era",
            ],
            [
                ("Brief summary", "summarize briefly."),
                ("Key cause or effect", "state one key cause or effect."),
                ("Historical significance", "explain its significance."),
                ("Timeline fact", "give one specific date or timeline fact."),
            ],
        ),
        (
            "United States of America",
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
                "ongoing policy challenges",
                "military and foreign policy",
            ],
            [
                ("Brief fact", "state the fact briefly."),
                ("Broader context", "add one broader context point."),
                ("Regional variation", "mention one regional variation."),
                ("Practical implication", "mention one practical implication."),
            ],
        ),
        (
            "Countries of the World",
            [
                "China's government and economy",
                "India's democracy and culture",
                "UK's parliament and history",
                "Germany's economy and history",
                "Japan's culture and economy",
                "Estonia's digital government",
                "Brazil's geography and culture",
                "Russia's geography and politics",
                "France's culture and history",
                "Canada's government and geography",
            ],
            [
                ("Identity fact", "state a key identity fact."),
                ("Economic fact", "state a key economic fact."),
                ("Cultural fact", "state a key cultural fact."),
                ("Historical fact", "state a key historical fact."),
            ],
        ),
        (
            "Health and Medicine",
            [
                "nutrition and macronutrients",
                "exercise and fitness benefits",
                "heart disease risk factors",
                "diabetes prevention",
                "mental health and depression",
                "vaccines and immunity",
                "antibiotics and resistance",
                "cancer basics",
                "sleep and health",
                "public health and epidemiology",
            ],
            [
                ("Simple explain", "explain simply."),
                ("Key fact", "state one key fact."),
                ("Practical advice", "give one practical health tip."),
                ("Common misconception", "correct one common misconception."),
            ],
        ),
        (
            "Philosophy",
            [
                "consequentialism and utilitarianism",
                "deontology and Kant",
                "virtue ethics",
                "logical fallacies",
                "the mind-body problem",
                "free will and determinism",
                "epistemology and knowledge",
                "Socrates and the Socratic method",
                "existentialism",
                "scientific method and critical thinking",
            ],
            [
                ("Simple explain", "explain the concept simply."),
                ("Key thinker", "name one key thinker and their idea."),
                ("Real-world application", "give a real-world application."),
                ("Contrast", "contrast it with an opposing view."),
            ],
        ),
        (
            "Economics and Business",
            [
                "supply and demand",
                "GDP and macroeconomics",
                "inflation and interest rates",
                "stock markets and investing",
                "startup and entrepreneurship",
                "international trade",
                "business strategy and competitive advantage",
                "behavioral economics",
                "cryptocurrency basics",
                "the gig economy",
            ],
            [
                ("Simple explain", "explain simply."),
                ("Key concept", "define the key concept."),
                ("Real example", "give a real-world example."),
                ("Practical implication", "state one practical implication."),
            ],
        ),
        (
            "Technology and AI",
            [
                "machine learning basics",
                "neural networks and deep learning",
                "large language models",
                "computer vision",
                "cybersecurity threats",
                "cloud computing",
                "blockchain and cryptocurrency",
                "quantum computing",
                "the ethics of AI",
                "automation and the future of work",
            ],
            [
                ("Simple explain", "explain simply."),
                ("Key fact", "state one key fact."),
                ("Real-world use", "give a real-world use case."),
                ("Challenge or risk", "state one key challenge or risk."),
            ],
        ),
        (
            "Mathematics and Statistics",
            [
                "algebra and equations",
                "geometry and shapes",
                "trigonometry and angles",
                "probability theory",
                "statistical distributions",
                "hypothesis testing",
                "regression analysis",
                "Bayesian inference",
                "game theory",
                "graph theory",
                "number theory",
                "calculus and derivatives",
                "integration techniques",
                "differential equations",
                "matrix operations",
                "eigenvalues and eigenvectors",
            ],
            [
                ("Simple explain", "explain the concept simply."),
                ("Key formula", "state the key formula."),
                ("Real example", "give a worked example."),
                ("Common mistake", "correct one common mistake."),
            ],
        ),
        (
            "Literature and Writing",
            [
                "Shakespeare and his works",
                "classic American literature",
                "poetry and poetic devices",
                "the novel as a form",
                "narrative structure",
                "character development",
                "literary movements",
                "modernist literature",
                "postcolonial literature",
                "science fiction genre",
                "mystery and detective fiction",
                "creative writing techniques",
                "editing and revision",
                "storytelling across cultures",
            ],
            [
                ("Simple explain", "explain the concept simply."),
                ("Key work", "name one key work and its significance."),
                ("Literary device", "explain one literary device."),
                ("Cultural impact", "describe the cultural impact."),
            ],
        ),
        (
            "Music and Sound",
            [
                "music theory basics",
                "scales and chords",
                "rhythm and tempo",
                "classical music periods",
                "jazz history",
                "rock and roll evolution",
                "hip-hop culture",
                "electronic music",
                "world music traditions",
                "music production",
                "sound engineering",
                "the psychology of music",
            ],
            [
                ("Simple explain", "explain the concept simply."),
                ("Key artist", "name one key artist or composer."),
                ("Historical context", "give historical context."),
                ("Technical detail", "explain one technical aspect."),
            ],
        ),
        (
            "Art and Visual Culture",
            [
                "Renaissance art",
                "impressionism",
                "modern art movements",
                "photography history",
                "film theory",
                "animation techniques",
                "graphic design principles",
                "color theory",
                "perspective in drawing",
                "digital art tools",
                "street art and graffiti",
                "art criticism",
            ],
            [
                ("Simple explain", "explain the concept simply."),
                ("Key artist", "name one key artist."),
                ("Movement", "describe the art movement."),
                ("Technique", "explain one technique."),
            ],
        ),
        (
            "Psychology and Human Behavior",
            [
                "cognitive biases",
                "memory and learning",
                "motivation and goals",
                "personality types",
                "social psychology",
                "developmental psychology",
                "behavioral conditioning",
                "emotional intelligence",
                "decision-making",
                "habits and behavior change",
                "stress and coping",
                "positive psychology",
            ],
            [
                ("Simple explain", "explain the concept simply."),
                ("Key study", "describe one key study."),
                ("Practical tip", "give one practical tip."),
                ("Common bias", "explain one common bias."),
            ],
        ),
        (
            "Environment and Ecology",
            [
                "ecosystems and biodiversity",
                "food chains and webs",
                "climate systems",
                "ocean currents",
                "the water cycle",
                "carbon cycle",
                "deforestation",
                "endangered species",
                "conservation biology",
                "sustainable agriculture",
                "pollution and health",
                "environmental policy",
            ],
            [
                ("Simple explain", "explain the concept simply."),
                ("Key fact", "state one key environmental fact."),
                ("Action", "suggest one actionable step."),
                ("Impact", "describe the human impact."),
            ],
        ),
        (
            "Space and Astronomy",
            [
                "the solar system",
                "planets and moons",
                "stars and their life cycles",
                "galaxies and clusters",
                "the Milky Way",
                "exoplanets",
                "space exploration history",
                "the International Space Station",
                "Mars missions",
                "telescopes and observation",
                "cosmic phenomena",
                "astrophysics basics",
            ],
            [
                ("Simple explain", "explain the concept simply."),
                ("Key fact", "state one key astronomical fact."),
                ("Mission", "describe one space mission."),
                ("Discovery", "mention one key discovery."),
            ],
        ),
        (
            "Law and Justice",
            [
                "constitutional law",
                "criminal law basics",
                "civil law basics",
                "international law",
                "human rights",
                "contract law",
                "property law",
                "intellectual property",
                "court systems",
                "legal reasoning",
                "justice and ethics",
                "civil liberties",
            ],
            [
                ("Simple explain", "explain the concept simply."),
                ("Key principle", "state one key legal principle."),
                ("Example", "give one legal example."),
                ("Right", "describe one key right."),
            ],
        ),
        (
            "Education and Learning",
            [
                "learning styles",
                "effective study techniques",
                "critical thinking skills",
                "online education",
                "educational psychology",
                "assessment and testing",
                "curriculum design",
                "special education",
                "higher education trends",
                "lifelong learning",
                "skill development",
                "teaching methods",
            ],
            [
                ("Simple explain", "explain the concept simply."),
                ("Key strategy", "describe one learning strategy."),
                ("Research", "cite one key research finding."),
                ("Tip", "give one practical tip."),
            ],
        ),
        (
            "Sports and Athletics",
            [
                "soccer rules and history",
                "basketball fundamentals",
                "track and field events",
                "swimming techniques",
                "tennis basics",
                "Olympic Games history",
                "sports psychology",
                "training and conditioning",
                "nutrition for athletes",
                "sports injuries",
                "team dynamics",
                "sports ethics",
            ],
            [
                ("Simple explain", "explain the sport simply."),
                ("Key fact", "state one key sports fact."),
                ("Technique", "describe one technique."),
                ("Rule", "explain one key rule."),
            ],
        ),
        (
            "Food and Cooking",
            [
                "cooking methods",
                "baking basics",
                "knife skills",
                "flavor pairing",
                "world cuisines",
                "food safety",
                "meal planning",
                "vegetarian cooking",
                "fermentation",
                "kitchen equipment",
                "recipe development",
                "food preservation",
            ],
            [
                ("Simple explain", "explain the technique simply."),
                ("Key tip", "give one key cooking tip."),
                ("Recipe idea", "suggest one recipe approach."),
                ("Safety", "state one food safety rule."),
            ],
        ),
        (
            "Travel and Culture",
            [
                "travel planning",
                "cultural etiquette",
                "language barriers",
                "budget travel",
                "sustainable tourism",
                "world heritage sites",
                "local customs",
                "festivals worldwide",
                "adventure travel",
                "digital nomad lifestyle",
                "packing strategies",
                "navigation skills",
            ],
            [
                ("Simple explain", "explain the concept simply."),
                ("Destination", "describe one destination."),
                ("Tip", "give one travel tip."),
                ("Custom", "explain one cultural custom."),
            ],
        ),
        (
            "Relationships and Communication",
            [
                "active listening",
                "conflict resolution",
                "empathy skills",
                "boundaries in relationships",
                "online communication",
                "public speaking",
                "negotiation tactics",
                "building trust",
                "giving feedback",
                "nonverbal communication",
                "assertiveness",
                "relationship stages",
            ],
            [
                ("Simple explain", "explain the concept simply."),
                ("Key skill", "describe one key skill."),
                ("Technique", "explain one communication technique."),
                ("Example", "give one practical example."),
            ],
        ),
        (
            "Personal Finance",
            [
                "budgeting basics",
                "saving strategies",
                "debt management",
                "credit scores",
                "insurance basics",
                "retirement planning",
                "tax fundamentals",
                "investing for beginners",
                "real estate basics",
                "financial goals",
                "emergency funds",
                "compound interest",
            ],
            [
                ("Simple explain", "explain the concept simply."),
                ("Key rule", "state one financial rule."),
                ("Example", "give one financial example."),
                ("Tip", "give one money tip."),
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
                examples.append({
                    "topic": topic,
                    "section": section,
                    "text": text,
                    "source": "synthetic/llm_expanded_examples",
                })
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
        examples.append({
            "topic": topic,
            "section": section,
            "text": text,
            "source": source,
            "memory_key": memory_key,
        })
    return examples


def build_training_data() -> str:
    KNOWLEDGE_DIR.mkdir(parents=True, exist_ok=True)
    TRAIN_DATA_FILE.parent.mkdir(parents=True, exist_ok=True)

    all_chunks: List[Dict[str, str]] = []
    file_count = 0

    for path in sorted(KNOWLEDGE_DIR.rglob("*.md")):
        topic, sections = _extract_sections(path)
        if not sections:
            continue
        rel_source = str(path.relative_to(KNOWLEDGE_DIR)).replace("\\", "/")
        chunks = _make_chunks(topic, sections, source=rel_source)
        for item in chunks:
            text = item.get("text", "")
            item["text"] = f"{text}\nSource: {rel_source}"
        all_chunks.extend(chunks)
        file_count += 1

    print(f"[build] Processed {file_count} knowledge files.")

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
        f"Built {len(rows)} training examples from {file_count} knowledge files "
        f"in '{KNOWLEDGE_DIR}'. Wrote '{TRAIN_DATA_FILE}'."
    )


if __name__ == "__main__":
    print(build_training_data())
