CLI_UI_VERSION = "0.2.0"

# Web UI version is tracked separately.
WEB_UI_VERSION = "0.2.0"

# Model version reflects the fully custom LLM upgrade.
LLM_MODEL_VERSION = "0.2.0"

# Backward-compatible alias.
VERSION = CLI_UI_VERSION

# Model profile label
MODEL_PROFILE = "custom-transformer"
MODEL_DESCRIPTION = (
    "Custom LLM v0.2.0 — fully custom GPT-style transformer, trained from scratch. "
    "No pre-trained weights. Custom tokenizer + custom multi-head self-attention + "
    "custom feed-forward transformer blocks. "
    "Retrieval-augmented generation: TF-IDF context injection + custom LLM generation. "
    "Knowledge base covers science, history, technology, geography, health, "
    "philosophy, economics, culture, language, sports, and world countries."
)
