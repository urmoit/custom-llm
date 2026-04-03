import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from .config import (
    ARTIFACTS_DIR,
    DOCUMENTS_FILE,
    MATRIX_FILE,
    MODEL_META_FILE,
    TRAIN_DATA_FILE,
    VECTORIZER_FILE,
)


def _load_training_data(path: Path) -> List[Dict[str, str]]:
    docs: List[Dict[str, str]] = []
    if not path.exists():
        return docs

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            text = str(item.get("text", "")).strip()
            topic = str(item.get("topic", "")).strip()
            section = str(item.get("section", "")).strip()

            if text:
                docs.append({"topic": topic, "section": section, "text": text})
                continue

            # Backward compatibility for old q/a rows.
            q = str(item.get("question", "")).strip()
            a = str(item.get("answer", "")).strip()
            if q and a:
                docs.append({"topic": "legacy", "section": q, "text": f"Q: {q}\nA: {a}"})
    return docs


def _save_meta(meta: Dict[str, str]) -> None:
    MODEL_META_FILE.write_text(json.dumps(meta, ensure_ascii=True, indent=2), encoding="utf-8")


def _choose_backend(requested: str) -> Tuple[str, str]:
    if requested == "auto":
        return "tfidf", "Auto mode defaults to lightweight tiny-model profile (TF-IDF)"

    if requested == "tfidf":
        return "tfidf", "Explicitly requested TF-IDF backend"

    try:
        import torch  # type: ignore
        from sentence_transformers import SentenceTransformer  # type: ignore  # noqa: F401

        if torch.cuda.is_available():
            return "transformer", "Using GPU transformer embeddings"
    except Exception:
        pass

    return "tfidf", "GPU transformer backend unavailable; using TF-IDF"


def train_model(backend: str = "auto") -> str:
    docs = _load_training_data(TRAIN_DATA_FILE)
    if not docs:
        raise RuntimeError(
            f"No training examples found in {TRAIN_DATA_FILE}. Add JSONL data first."
        )

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    selected_backend, backend_note = _choose_backend(backend)
    corpus = [d["text"] for d in docs]

    if selected_backend == "transformer":
        import torch  # type: ignore
        from sentence_transformers import SentenceTransformer  # type: ignore

        model_name = "sentence-transformers/all-MiniLM-L6-v2"
        model = SentenceTransformer(model_name, device="cuda")
        embeddings = model.encode(
            corpus,
            batch_size=32,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        matrix = np.asarray(embeddings, dtype=np.float32)
        joblib.dump(None, VECTORIZER_FILE)
        joblib.dump(matrix, MATRIX_FILE)
        _save_meta(
            {
                "backend": "transformer",
                "model_name": model_name,
                "device": "cuda" if torch.cuda.is_available() else "cpu",
                "note": backend_note,
                "model_profile": "large-embedding",
            }
        )
    else:
        vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=1)
        matrix = vectorizer.fit_transform(corpus)
        joblib.dump(vectorizer, VECTORIZER_FILE)
        joblib.dump(matrix, MATRIX_FILE)
        _save_meta(
            {
                "backend": "tfidf",
                "model_name": "tfidf",
                "device": "cpu",
                "note": backend_note,
                "model_profile": "tiny-1m-style",
                "approx_parameters": 1000000,
            }
        )

    joblib.dump(docs, DOCUMENTS_FILE)

    return (
        f"Training complete. Backend={selected_backend}. Saved {len(docs)} examples "
        f"to artifacts in '{ARTIFACTS_DIR}'."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train local knowledge model")
    parser.add_argument(
        "--backend",
        choices=["auto", "tfidf", "transformer"],
        default="auto",
        help="Training backend selection",
    )
    args = parser.parse_args()
    print(train_model(backend=args.backend))
