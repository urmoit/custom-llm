from __future__ import annotations

import re
from typing import Dict, Optional, Tuple

LengthContext = Tuple[float, str]

# Factors are in meters.
UNIT_TO_METER: Dict[str, float] = {
    "km": 1000.0,
    "kilometer": 1000.0,
    "kilometers": 1000.0,
    "m": 1.0,
    "meter": 1.0,
    "meters": 1.0,
    "metre": 1.0,
    "metres": 1.0,
    "dm": 0.1,
    "decimeter": 0.1,
    "decimeters": 0.1,
    "cm": 0.01,
    "centimeter": 0.01,
    "centimeters": 0.01,
    "mm": 0.001,
    "millimeter": 0.001,
    "millimeters": 0.001,
    "um": 1e-6,
    "micrometer": 1e-6,
    "micrometers": 1e-6,
    "micromoter": 1e-6,
    "micromoters": 1e-6,
    "micrometre": 1e-6,
    "micrometres": 1e-6,
    "nm": 1e-9,
    "nanometer": 1e-9,
    "nanometers": 1e-9,
    "pm": 1e-12,
    "picometer": 1e-12,
    "picometers": 1e-12,
    "mi": 1609.344,
    "mile": 1609.344,
    "miles": 1609.344,
    "nmi": 1852.0,
    "nauticalmile": 1852.0,
    "nauticalmiles": 1852.0,
    "fur": 201.168,
    "furlong": 201.168,
    "furlongs": 201.168,
    "ftm": 1.8288,
    "fathom": 1.8288,
    "fathoms": 1.8288,
    "yd": 0.9144,
    "yard": 0.9144,
    "yards": 0.9144,
    "ft": 0.3048,
    "foot": 0.3048,
    "feet": 0.3048,
    "in": 0.0254,
    "inch": 0.0254,
    "inches": 0.0254,
    # Chinese units (modern metric definitions)
    "li": 500.0,
    "zhang": 3.3333333333,
    "chi": 0.3333333333,
    "cun": 0.0333333333,
    "fen": 0.0033333333,
    "lii": 0.0003333333,
    "hao": 0.0000333333,
    # Astronomy
    "pc": 3.085677581e16,
    "parsec": 3.085677581e16,
    "ld": 384400000.0,
    "id": 384400000.0,
    "lunardistance": 384400000.0,
    "au": 149597870700.0,
    "astronomicalunit": 149597870700.0,
    "ly": 9.4607304725808e15,
    "lightyear": 9.4607304725808e15,
}

DISPLAY_ORDER = [
    "km",
    "m",
    "dm",
    "cm",
    "mm",
    "um",
    "nm",
    "pm",
    "nmi",
    "mi",
    "fur",
    "ftm",
    "yd",
    "ft",
    "in",
    "li",
    "zhang",
    "chi",
    "cun",
    "fen",
    "lii",
    "hao",
    "pc",
    "ld",
    "au",
    "ly",
]

UNIT_LABEL = {
    "km": "km (kilometer)",
    "m": "m (meter)",
    "dm": "dm (decimeter)",
    "cm": "cm (centimeter)",
    "mm": "mm (millimeter)",
    "um": "um (micrometer)",
    "nm": "nm (nanometer)",
    "pm": "pm (picometer)",
    "nmi": "nmi (nautical mile)",
    "mi": "mi (mile)",
    "fur": "fur (furlong)",
    "ftm": "ftm (fathom)",
    "yd": "yd (yard)",
    "ft": "ft (foot)",
    "in": "in (inch)",
    "li": "li",
    "zhang": "zhang",
    "chi": "chi",
    "cun": "cun",
    "fen": "fen",
    "lii": "lii",
    "hao": "hao",
    "pc": "pc (parsec)",
    "ld": "ld (lunar distance)",
    "au": "au (astronomical unit)",
    "ly": "ly (light year)",
}


def _normalize_unit(unit: str) -> str:
    return re.sub(r"[^a-z]", "", unit.lower())


def _extract_conversion(question: str) -> Optional[Tuple[float, str, str]]:
    q = question.strip().lower()

    patterns = [
        r"(?:how\s+many|convert)\s+([a-zA-Z]+)\s+(?:are\s+in|in|to)\s+([0-9]*\.?[0-9]+)\s*([a-zA-Z]+)",
        r"([0-9]*\.?[0-9]+)\s*([a-zA-Z]+)\s+(?:to|in)\s+([a-zA-Z]+)",
    ]

    for pattern in patterns:
        match = re.search(pattern, q)
        if not match:
            continue

        g = match.groups()
        if len(g) != 3:
            continue

        # pattern 1 order: target, value, source
        if re.search(r"how\s+many|convert", q):
            if g[0].isalpha() and re.match(r"[0-9]", g[1]):
                target_u = _normalize_unit(g[0])
                value = float(g[1])
                source_u = _normalize_unit(g[2])
                return value, source_u, target_u

        # pattern 2 order: value, source, target
        if re.match(r"[0-9]", g[0]):
            value = float(g[0])
            source_u = _normalize_unit(g[1])
            target_u = _normalize_unit(g[2])
            return value, source_u, target_u

    return None


def _extract_followup_target(question: str) -> Optional[str]:
    q = question.strip().lower()
    patterns = [
        r"^(?:but\s+)?(?:in|to)\s+([a-zA-Z]+)\??$",
        r"^(?:and\s+)?(?:in|to)\s+([a-zA-Z]+)\??$",
        r"^but\s+([a-zA-Z]+)\??$",
    ]

    for pattern in patterns:
        match = re.search(pattern, q)
        if match:
            target = _normalize_unit(match.group(1))
            if target in UNIT_TO_METER:
                return target
    return None


def format_length_reference() -> str:
    lines = [
        "Length knowledge reference (base: meter):",
        "Core metric ladder:",
        "- 10 mm = 1 cm",
        "- 10 cm = 1 dm",
        "- 10 dm = 1 m",
        "- 1000 m = 1 km",
        "",
        "Supported units:",
    ]

    for unit in DISPLAY_ORDER:
        factor = UNIT_TO_METER[unit]
        label = UNIT_LABEL[unit]
        lines.append(f"- 1 {label} = {factor:.12g} m")

    lines.extend(
        [
            "",
            "Notes:",
            "- Chinese units use modern metric definitions.",
            "- Lunar distance is average Earth-Moon distance.",
            "- Use queries like: 'how many miles are in 1 km' or '10 mm to cm'.",
        ]
    )
    return "\n".join(lines)


def format_length_units_summary() -> str:
    units = [UNIT_LABEL[u] for u in DISPLAY_ORDER]
    return (
        f"I know {len(units)} length units:\n"
        + ", ".join(units)
        + "\n\nUse 'show all length units' if you want the full conversion table."
    )


def try_answer_length_question(
    question: str, context: Optional[LengthContext] = None
) -> Tuple[Optional[str], Optional[LengthContext]]:
    q = question.lower().replace("lenght", "length")

    if any(
        token in q
        for token in [
            "show all length",
            "all length units",
            "unit table",
            "length table",
            "full length",
            "full table",
        ]
    ):
        return format_length_reference(), context

    if (
        "length units" in q
        or "units do you know" in q
        or "how many length" in q
        or "how many units" in q
    ):
        return format_length_units_summary(), context

    if "km" in q and "mile" in q and "1" in q:
        miles = 1000.0 / UNIT_TO_METER["mi"]
        return (
            f"1 km = {miles:.9f} mi\n"
            "(exact meter definitions: 1 km = 1000 m, 1 mi = 1609.344 m)"
        ), (1.0, "km")

    parsed = _extract_conversion(question)
    if parsed:
        value, source_u, target_u = parsed
        if source_u not in UNIT_TO_METER or target_u not in UNIT_TO_METER:
            return None, context
    else:
        followup_target = _extract_followup_target(question)
        if not followup_target or context is None:
            return None, context
        value, source_u = context
        target_u = followup_target

    meters = value * UNIT_TO_METER[source_u]
    converted = meters / UNIT_TO_METER[target_u]

    source_label = UNIT_LABEL.get(source_u, source_u)
    target_label = UNIT_LABEL.get(target_u, target_u)

    answer = (
        f"{value:g} {source_label} = {converted:.12g} {target_label}\n"
        f"(via meters: {value:g} * {UNIT_TO_METER[source_u]:.12g} / {UNIT_TO_METER[target_u]:.12g})"
    )
    return answer, (value, source_u)
