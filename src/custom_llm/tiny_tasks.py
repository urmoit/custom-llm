from __future__ import annotations

import re
from typing import List, Optional


def _extract_payload(raw_question: str) -> str:
    if ":" in raw_question:
        return raw_question.split(":", 1)[1].strip()
    parts = raw_question.split(maxsplit=2)
    if len(parts) >= 3:
        return parts[2].strip()
    return ""


def _extract_last_quoted_text(raw_question: str) -> str:
    matches = re.findall(r'"([^"]+)"', raw_question)
    if matches:
        return matches[-1].strip()
    return ""


def _extract_after_keyword(raw_question: str, keyword: str) -> str:
    pattern = re.compile(rf"{re.escape(keyword)}\s*(.+)", re.IGNORECASE)
    m = pattern.search(raw_question)
    if not m:
        return ""
    value = m.group(1).strip()
    value = re.split(r"\b(output|answer|assistant|npc)\s*:\s*", value, flags=re.IGNORECASE)[0].strip()
    return value


def _classify_spam(text: str) -> str:
    t = text.lower()
    spam_terms = [
        "buy now",
        "free",
        "winner",
        "win big",
        "click here",
        "limited time",
        "urgent",
        "subscribe",
        "promo",
        "discount",
    ]
    return "spam" if any(term in t for term in spam_terms) else "not spam"


def _classify_toxic(text: str) -> str:
    t = text.lower()
    toxic_terms = [
        "idiot",
        "stupid",
        "hate you",
        "trash",
        "shut up",
        "moron",
        "dumb",
    ]
    return "toxic" if any(term in t for term in toxic_terms) else "safe"


def _classify_topic(text: str) -> str:
    t = text.lower()
    topic_keywords = {
        "sports": ["football", "basketball", "soccer", "nba", "nfl", "goal", "match"],
        "tech": ["python", "code", "ai", "software", "computer", "gpu", "app"],
        "music": ["song", "music", "playlist", "album", "artist"],
        "finance": ["stock", "market", "money", "bank", "investment", "budget"],
        "home": ["lights", "kitchen", "room", "home", "thermostat"],
    }
    for topic, keys in topic_keywords.items():
        if any(k in t for k in keys):
            return topic
    return "general"


def _detect_intent(text: str) -> str:
    t = text.lower().strip()
    if "turn on" in t and ("light" in t or "lights" in t):
        return "turn_on_lights"
    if "turn off" in t and ("light" in t or "lights" in t):
        return "turn_off_lights"
    if "play" in t and "music" in t:
        return "play_music"
    if "weather" in t:
        return "get_weather"
    if "set" in t and "alarm" in t:
        return "set_alarm"
    return "unknown"


def _detect_trade_intent(text: str) -> str:
    t = text.lower().strip()
    if any(k in t for k in ["hello", "hi", "hey", "good morning", "good evening"]):
        return "greet"
    if any(k in t for k in ["buy", "purchase", "get", "order"]):
        return "buy"
    if any(k in t for k in ["sell", "list for sale", "trade in"]):
        return "sell"
    if any(k in t for k in ["help", "support", "assist", "how do i"]):
        return "help"
    return "unknown"


def _simple_generation(prompt: str) -> str:
    p = prompt.lower().strip()
    if any(g in p for g in ["hi", "hello", "hey"]):
        return "Hello, how can I help?"
    if "thank" in p:
        return "You are welcome."
    if "bye" in p:
        return "Goodbye."
    return "Command not recognized."


def _short_assistant_reply(user_text: str) -> str:
    t = user_text.lower().strip()
    if any(g in t for g in ["hello", "hi", "hey"]):
        return "Hi, how can I help?"
    if "help" in t:
        return "I can help with simple tasks."
    if "bye" in t:
        return "Goodbye and have a great day."
    return "I can give short simple replies."


def _npc_shopkeeper_reply(player_text: str) -> str:
    t = player_text.lower().strip()
    if "worker" in t or "workers" in t:
        return "Hire workers from the shop desk."
    if any(k in t for k in ["buy", "purchase"]):
        return "You can buy items in my shop."
    if "sell" in t:
        return "Sell goods here for quick coins."
    return "Welcome. Buy or sell when ready."


def _yes_no_toxic(text: str) -> str:
    return "YES" if _classify_toxic(text) == "toxic" else "NO"


def _mapping_reply(text: str) -> Optional[str]:
    t = text.lower().strip()
    mapping = {
        "hello": "Hi there!",
        "bye": "Goodbye!",
        "help": "I can help you.",
        "thanks": "You're welcome!",
    }
    return mapping.get(t)


def _extract_mapping_input(raw_question: str) -> str:
    matches = re.findall(r"input\s*:\s*(.+?)(?:\s+output\s*:|$)", raw_question, flags=re.IGNORECASE)
    if not matches:
        return ""
    return matches[-1].strip().strip('"')


def _parse_numbers(text: str) -> List[float]:
    return [float(x) for x in re.findall(r"-?\d+(?:\.\d+)?", text)]


def _pattern_analysis(text: str) -> str:
    nums = _parse_numbers(text)
    if len(nums) < 3:
        return "Need at least 3 numbers to analyze a pattern."

    diffs = [nums[i + 1] - nums[i] for i in range(len(nums) - 1)]

    if all(abs(d - diffs[0]) < 1e-9 for d in diffs[1:]):
        nxt = nums[-1] + diffs[0]
        return f"Arithmetic pattern detected. Next value is likely {nxt:g}."

    if len(nums) >= 4:
        for p in range(1, min(4, len(nums) // 2 + 1)):
            unit = nums[:p]
            expected = [unit[i % p] for i in range(len(nums))]
            mismatches = [i for i, (a, b) in enumerate(zip(nums, expected)) if abs(a - b) > 1e-9]
            if not mismatches:
                return f"Repeating pattern detected with cycle length {p}."
            if len(mismatches) == 1:
                idx = mismatches[0]
                return (
                    "Mostly repeating pattern with one anomaly at "
                    f"position {idx + 1}: expected {expected[idx]:g}, got {nums[idx]:g}."
                )

    return "No strong simple pattern detected."


def try_handle_tiny_task(raw_question: str, normalized_q: str) -> Optional[str]:
    q = normalized_q

    if (
        "classify the user input" in q
        and "greet" in q
        and "buy" in q
        and "sell" in q
        and "help" in q
    ):
        payload = _extract_last_quoted_text(raw_question)
        if not payload:
            payload = _extract_payload(raw_question)
        if payload:
            return _detect_trade_intent(payload)
        return "unknown"

    if "classify intent" in q or "intent detection" in q:
        payload = _extract_payload(raw_question)
        if not payload:
            payload = _extract_last_quoted_text(raw_question)
        if payload:
            return _detect_trade_intent(payload)
        return "unknown"

    if "you are a simple assistant" in q and "respond in one short sentence" in q:
        user_text = ""
        if "user:" in raw_question.lower():
            user_text = raw_question.split(":", 1)[1].strip()
        if not user_text:
            user_text = _extract_last_quoted_text(raw_question) or _extract_payload(raw_question)
        return _short_assistant_reply(user_text)

    if "npc shopkeeper" in q or "you are an npc shopkeeper" in q:
        player_text = ""
        if "player:" in raw_question.lower():
            player_text = raw_question.lower().split("player:", 1)[1].strip()
        if not player_text:
            player_text = _extract_last_quoted_text(raw_question) or _extract_payload(raw_question)
        return _npc_shopkeeper_reply(player_text)

    if "answer only yes or no" in q or "is this message toxic" in q:
        payload = _extract_last_quoted_text(raw_question)
        if not payload:
            payload = _extract_payload(raw_question)
        if not payload:
            payload = _extract_after_keyword(raw_question, "toxic?")
        if not payload:
            return "NO"
        return _yes_no_toxic(payload)

    if "match the input to the correct response" in q:
        payload = _extract_mapping_input(raw_question)
        if not payload:
            payload = _extract_payload(raw_question)
        if not payload:
            payload = _extract_last_quoted_text(raw_question)
        reply = _mapping_reply(payload) if payload else None
        return reply if reply else "I do not have a mapped response for that input."

    if "classify text" in q or "text classification" in q:
        payload = _extract_payload(raw_question)
        if not payload:
            return "Please provide text after ':' for classification. Example: classify text: Win free prize now"
        spam = _classify_spam(payload)
        toxic = _classify_toxic(payload)
        topic = _classify_topic(payload)
        return (
            "Tiny-model classification result:\n"
            f"- Spam: {spam}\n"
            f"- Safety: {toxic}\n"
            f"- Topic: {topic}"
        )

    if "detect intent" in q or "intent" in q:
        payload = _extract_payload(raw_question)
        if not payload:
            payload = raw_question
        intent = _detect_intent(payload)
        return f"Detected intent: {intent}"

    if "generate" in q and ("simple" in q or "short" in q):
        payload = _extract_payload(raw_question)
        if not payload:
            payload = raw_question
        return _simple_generation(payload)

    if "pattern" in q or "anomaly" in q or "sequence" in q:
        payload = _extract_payload(raw_question)
        if not payload:
            payload = raw_question
        return _pattern_analysis(payload)

    return None
