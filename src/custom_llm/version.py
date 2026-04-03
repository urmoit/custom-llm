CLI_UI_VERSION = "0.1.0"

# Web UI version is tracked separately.
WEB_UI_VERSION = "0.1.0"

# Model version reflects the 10M-style upgrade.
LLM_MODEL_VERSION = "0.1.0"

# Backward-compatible alias.
VERSION = CLI_UI_VERSION

# Model profile label
MODEL_PROFILE = "10m-style"
MODEL_DESCRIPTION = (
    "Custom LLM v0.1.0 — 10M-style local knowledge model. "
    "Knowledge base covers science, history, technology, geography, health, "
    "philosophy, economics, culture, language, sports, and world countries. "
    "Upgraded TF-IDF: trigrams, 80k vocabulary, sublinear TF. "
    "Multi-turn conversation memory, smart web search triggers, "
    "and expanded Python code generation."
)
