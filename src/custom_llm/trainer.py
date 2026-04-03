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

            # Backward compatibility
            q = str(item.get("question", "")).strip()
            a = str(item.get("answer", "")).strip()
            if q and a:
                docs.append({"topic": "legacy", "section": q, "text": f"Q: {q}\nA: {a}"})
    return docs


def _save_meta(meta: Dict[str, str]) -> None:
    MODEL_META_FILE.write_text(json.dumps(meta, ensure_ascii=True, indent=2), encoding="utf-8")


def _choose_backend(requested: str) -> Tuple[str, str]:
    if requested == "auto":
        return "tfidf", "Auto mode: 10M-style TF-IDF with expanded n-grams and rich vocabulary"

    if requested == "tfidf":
        return "tfidf", "TF-IDF backend with 10M-style vocabulary expansion"

    try:
        import torch  # type: ignore
        from sentence_transformers import SentenceTransformer  # type: ignore  # noqa: F401

        if torch.cuda.is_available():
            return "transformer", "GPU transformer embeddings (best quality)"
    except Exception:
        pass

    return "tfidf", "GPU transformer unavailable; using 10M-style TF-IDF"


def train_model(backend: str = "auto") -> str:
    docs = _load_training_data(TRAIN_DATA_FILE)
    if not docs:
        raise RuntimeError(
            f"No training examples found in {TRAIN_DATA_FILE}. "
            "Run build_training_data first."
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
            corpus, batch_size=32, normalize_embeddings=True,
            convert_to_numpy=True, show_progress_bar=False,
        )
        matrix = np.asarray(embeddings, dtype=np.float32)
        joblib.dump(None, VECTORIZER_FILE)
        joblib.dump(matrix, MATRIX_FILE)
        _save_meta({
            "backend": "transformer",
            "model_name": model_name,
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "note": backend_note,
            "model_profile": "transformer-embedding",
            "approx_parameters": 22_400_000,
            "corpus_size": len(docs),
        })
    else:
        # 10M-style TF-IDF: larger vocabulary, richer n-grams, better coverage
        vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 3),        # up to trigrams for better phrase matching
            min_df=1,
            max_df=0.95,
            max_features=80_000,       # much larger vocabulary than original (was default ~unlimited but small corpus)
            sublinear_tf=True,         # log normalization for term frequency
            strip_accents="unicode",
            analyzer="word",
        )
        matrix = vectorizer.fit_transform(corpus)
        vocab_size = len(vectorizer.vocabulary_)
        joblib.dump(vectorizer, VECTORIZER_FILE)
        joblib.dump(matrix, MATRIX_FILE)
        _save_meta({
            "backend": "tfidf",
            "model_name": "tfidf-10m-style",
            "device": "cpu",
            "note": backend_note,
            "model_profile": "10m-style",
            "approx_parameters": 10_000_000,
            "vocab_size": vocab_size,
            "ngram_range": "1-3",
            "corpus_size": len(docs),
        })

    joblib.dump(docs, DOCUMENTS_FILE)

    return (
        f"Training complete. Backend={selected_backend}. "
        f"Corpus={len(docs)} documents. "
        f"Saved artifacts to '{ARTIFACTS_DIR}'."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Custom LLM knowledge model")
    parser.add_argument(
        "--backend",
        choices=["auto", "tfidf", "transformer"],
        default="auto",
        help="Training backend selection",
    )
    args = parser.parse_args()
    print(train_model(backend=args.backend))
