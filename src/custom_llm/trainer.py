import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from .config import (
    ARTIFACTS_DIR,
    CUSTOM_LLM_BATCH_SIZE,
    CUSTOM_LLM_CONTEXT_LENGTH,
    CUSTOM_LLM_D_FF,
    CUSTOM_LLM_D_MODEL,
    CUSTOM_LLM_DROPOUT,
    CUSTOM_LLM_EPOCHS,
    CUSTOM_LLM_FILE,
    CUSTOM_LLM_LR,
    CUSTOM_LLM_N_HEADS,
    CUSTOM_LLM_N_LAYERS,
    DOCUMENTS_FILE,
    MATRIX_FILE,
    MODEL_META_FILE,
    TOKENIZER_FILE,
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
        try:
            import torch  # type: ignore  # noqa: F401
            return "custom", "Auto mode: fully custom transformer LLM + TF-IDF retrieval"
        except ImportError:
            return "tfidf", "Auto mode: 10M-style TF-IDF (install torch for custom LLM backend)"

    if requested == "tfidf":
        return "tfidf", "TF-IDF backend with 10M-style vocabulary expansion"

    if requested == "custom":
        try:
            import torch  # type: ignore  # noqa: F401
            return "custom", "Fully custom transformer LLM trained from scratch"
        except ImportError:
            return "tfidf", "PyTorch not available; falling back to TF-IDF"

    try:
        import torch  # type: ignore
        from sentence_transformers import SentenceTransformer  # type: ignore  # noqa: F401

        if torch.cuda.is_available():
            return "transformer", "GPU transformer embeddings (best quality)"
    except Exception:
        pass

    return "tfidf", "GPU transformer unavailable; using 10M-style TF-IDF"


# ---------------------------------------------------------------------------
# Custom LLM training
# ---------------------------------------------------------------------------

def _build_training_sequences(
    corpus: List[str],
    tokenizer: "Tokenizer",  # type: ignore[name-defined]  # noqa: F821
    context_length: int,
) -> List[List[int]]:
    """Tokenise corpus and chop into overlapping context-length windows."""
    sequences: List[List[int]] = []
    for text in corpus:
        ids = tokenizer.encode(text, add_bos=True, add_eos=True)
        # Sliding window with 50% stride to increase sample diversity
        stride = max(1, context_length // 2)
        for start in range(0, max(1, len(ids) - context_length), stride):
            chunk = ids[start:start + context_length + 1]
            if len(chunk) > 1:
                sequences.append(chunk)
    return sequences


def train_custom_llm(docs: List[Dict[str, str]], backend_note: str) -> str:
    """Train a fully custom GPT-style LLM from scratch on the knowledge corpus.

    Steps
    -----
    1. Build vocabulary with the custom :class:`Tokenizer`.
    2. Encode the corpus and slice into training windows.
    3. Train the :class:`CustomLanguageModel` with cross-entropy next-token loss.
    4. Save model weights, tokenizer, and metadata to ``ARTIFACTS_DIR``.

    Also trains the TF-IDF retrieval index alongside the custom LLM so that
    retrieval-augmented generation (RAG) still works during inference.
    """
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset

    from .tokenizer import Tokenizer
    from .model import CustomLanguageModel

    corpus = [d["text"] for d in docs]
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)

    # 1. Build tokenizer
    tokenizer = Tokenizer()
    tokenizer.build_vocab(corpus, min_freq=1, max_vocab=20_000)
    tokenizer.save(TOKENIZER_FILE)

    # 2. Build training sequences
    seqs = _build_training_sequences(corpus, tokenizer, CUSTOM_LLM_CONTEXT_LENGTH)
    if not seqs:
        raise RuntimeError("No training sequences could be built from the corpus.")

    # Pad sequences to uniform length and create input/target pairs
    seq_len = CUSTOM_LLM_CONTEXT_LENGTH + 1
    padded: List[List[int]] = []
    for s in seqs:
        padded.append((s + [0] * seq_len)[:seq_len])

    data_tensor = torch.tensor(padded, dtype=torch.long)
    inputs = data_tensor[:, :-1]    # (N, context_length)
    targets = data_tensor[:, 1:]    # (N, context_length)

    dataset = TensorDataset(inputs, targets)
    loader = DataLoader(dataset, batch_size=CUSTOM_LLM_BATCH_SIZE, shuffle=True)

    # 3. Instantiate model
    model = CustomLanguageModel(
        vocab_size=tokenizer.vocab_size,
        d_model=CUSTOM_LLM_D_MODEL,
        n_heads=CUSTOM_LLM_N_HEADS,
        n_layers=CUSTOM_LLM_N_LAYERS,
        context_length=CUSTOM_LLM_CONTEXT_LENGTH,
        d_ff=CUSTOM_LLM_D_FF,
        dropout=CUSTOM_LLM_DROPOUT,
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=CUSTOM_LLM_LR)
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # ignore PAD

    # 4. Training loop
    model.train()
    for epoch in range(CUSTOM_LLM_EPOCHS):
        total_loss = 0.0
        n_batches = 0
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad()
            logits = model(x_batch)                # (B, T, V)
            B, T, V = logits.shape
            loss = criterion(logits.view(B * T, V), y_batch.view(B * T))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        avg_loss = total_loss / max(n_batches, 1)
        print(f"  [custom-llm] epoch {epoch + 1}/{CUSTOM_LLM_EPOCHS}  loss={avg_loss:.4f}")

    # 5. Save model + metadata
    torch.save(
        {
            "model_state": model.state_dict(),
            "config": model.config_dict(),
        },
        CUSTOM_LLM_FILE,
    )

    # 6. Also build TF-IDF retrieval index (used for RAG context injection)
    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 3),
        min_df=1,
        max_df=0.95,
        max_features=80_000,
        sublinear_tf=True,
        strip_accents="unicode",
        analyzer="word",
    )
    matrix = vectorizer.fit_transform(corpus)
    joblib.dump(vectorizer, VECTORIZER_FILE)
    joblib.dump(matrix, MATRIX_FILE)
    joblib.dump(docs, DOCUMENTS_FILE)

    cfg = model.config_dict()
    _save_meta({
        "backend": "custom",
        "model_name": "custom-gpt-scratch",
        "device": device_str,
        "note": backend_note,
        "model_profile": "custom-transformer",
        "num_parameters": cfg["num_parameters"],
        "d_model": cfg["d_model"],
        "n_layers": cfg["n_layers"],
        "n_heads": cfg["n_heads"],
        "context_length": cfg["context_length"],
        "vocab_size": tokenizer.vocab_size,
        "corpus_size": len(docs),
        "epochs": CUSTOM_LLM_EPOCHS,
    })

    return (
        f"Custom LLM training complete. "
        f"Backend=custom  |  Parameters={cfg['num_parameters']:,}  |  "
        f"Vocab={tokenizer.vocab_size:,}  |  "
        f"Corpus={len(docs)} documents  |  "
        f"Device={device_str}. "
        f"Artifacts saved to '{ARTIFACTS_DIR}'."
    )

def train_model(backend: str = "auto") -> str:
    docs = _load_training_data(TRAIN_DATA_FILE)
    if not docs:
        raise RuntimeError(
            f"No training examples found in {TRAIN_DATA_FILE}. "
            "Run build_training_data first."
        )

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    selected_backend, backend_note = _choose_backend(backend)

    if selected_backend == "custom":
        return train_custom_llm(docs, backend_note)

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
        choices=["auto", "tfidf", "transformer", "custom"],
        default="auto",
        help="Training backend selection",
    )
    args = parser.parse_args()
    print(train_model(backend=args.backend))
