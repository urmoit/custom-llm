from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
KNOWLEDGE_DIR = ROOT / "knowledge"
MEMORY_DIR = DATA_DIR / "memory"
MEMORY_FILE = MEMORY_DIR / "chat_memory.jsonl"
ARTIFACTS_DIR = ROOT / "artifacts"
TRAIN_DATA_FILE = DATA_DIR / "training_data.jsonl"
VECTORIZER_FILE = ARTIFACTS_DIR / "vectorizer.joblib"
MATRIX_FILE = ARTIFACTS_DIR / "doc_matrix.joblib"
DOCUMENTS_FILE = ARTIFACTS_DIR / "documents.joblib"
MODEL_META_FILE = ARTIFACTS_DIR / "model_meta.json"

# Custom LLM artifacts (fully trained from scratch)
CUSTOM_LLM_FILE = ARTIFACTS_DIR / "custom_llm.pt"
TOKENIZER_FILE = ARTIFACTS_DIR / "tokenizer.json"

# Custom LLM hyperparameters — ~10M parameter model for richer knowledge & reasoning
CUSTOM_LLM_D_MODEL = 512
CUSTOM_LLM_N_HEADS = 8
CUSTOM_LLM_N_LAYERS = 6
CUSTOM_LLM_CONTEXT_LENGTH = 384
CUSTOM_LLM_D_FF = 1024
CUSTOM_LLM_DROPOUT = 0.1
CUSTOM_LLM_EPOCHS = 8
CUSTOM_LLM_BATCH_SIZE = 4
CUSTOM_LLM_LR = 2e-4

# 10M-style model settings — lower threshold for broader recall with richer vocab
TOP_K_LOCAL = 5                    # more related results shown
LOCAL_SIMILARITY_THRESHOLD = 0.18  # slightly lower to catch more with richer n-grams

MAX_WEB_RESULTS = 6
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)
