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

# 10M-style model settings — lower threshold for broader recall with richer vocab
TOP_K_LOCAL = 5                    # more related results shown
LOCAL_SIMILARITY_THRESHOLD = 0.18  # slightly lower to catch more with richer n-grams

MAX_WEB_RESULTS = 6
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)
