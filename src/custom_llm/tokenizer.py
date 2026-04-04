"""Custom word-level tokenizer — built and trained from scratch on the local corpus."""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
BOS_TOKEN = "<BOS>"
EOS_TOKEN = "<EOS>"
SEP_TOKEN = "<SEP>"

SPECIAL_TOKENS = [PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN, SEP_TOKEN]

PAD_ID = 0
UNK_ID = 1
BOS_ID = 2
EOS_ID = 3
SEP_ID = 4


class Tokenizer:
    """Word-level tokenizer trained from scratch on the local knowledge corpus.

    Vocabulary is built by frequency from the training corpus.  Special tokens
    are always placed at fixed IDs 0–4 so that PAD/UNK/BOS/EOS/SEP remain
    stable across re-trains.
    """

    def __init__(self) -> None:
        self.word_to_id: Dict[str, int] = {}
        self.id_to_word: Dict[int, str] = {}
        self._init_specials()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _init_specials(self) -> None:
        for tok in SPECIAL_TOKENS:
            idx = len(self.word_to_id)
            self.word_to_id[tok] = idx
            self.id_to_word[idx] = tok

    @property
    def vocab_size(self) -> int:
        return len(self.word_to_id)

    @staticmethod
    def _tokenize_text(text: str) -> List[str]:
        r"""Split text into lowercase word/punctuation tokens.

        The pattern captures:
        - ``[a-z0-9]+(?:'[a-z]+)?`` — words (optionally with contractions like "don't")
        - ``[.,!?;:()\[\]{}\"/\\-]``  — common punctuation as individual tokens
        """
        text = text.lower()
        tokens = re.findall(r"[a-z0-9]+(?:'[a-z]+)?|[.,!?;:()\[\]{}\"/\\-]", text)
        return tokens

    # ------------------------------------------------------------------
    # Vocabulary construction
    # ------------------------------------------------------------------
    def build_vocab(self, corpus: List[str], min_freq: int = 1, max_vocab: int = 20_000) -> None:
        """Build vocabulary from a list of text strings.

        Tokens are ordered by descending frequency after the fixed special
        tokens so the most common words get the smallest IDs.
        """
        freq: Counter = Counter()
        for text in corpus:
            freq.update(self._tokenize_text(text))

        for word, count in freq.most_common(max_vocab):
            if count < min_freq:
                break
            if word not in self.word_to_id:
                idx = len(self.word_to_id)
                self.word_to_id[word] = idx
                self.id_to_word[idx] = word

    # ------------------------------------------------------------------
    # Encode / decode
    # ------------------------------------------------------------------
    def encode(
        self,
        text: str,
        add_bos: bool = True,
        add_eos: bool = True,
        max_length: int = 0,
    ) -> List[int]:
        """Encode *text* to a list of integer token IDs."""
        tokens = self._tokenize_text(text)
        ids = [self.word_to_id.get(t, UNK_ID) for t in tokens]
        if add_bos:
            ids = [BOS_ID] + ids
        if add_eos:
            ids = ids + [EOS_ID]
        if max_length > 0:
            ids = ids[:max_length]
        return ids

    def decode(self, ids: List[int], skip_special: bool = True) -> str:
        """Decode a list of token IDs back to a text string."""
        words: List[str] = []
        for i in ids:
            word = self.id_to_word.get(i, UNK_TOKEN)
            if skip_special and word in SPECIAL_TOKENS:
                continue
            words.append(word)
        return " ".join(words)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self, path: Path) -> None:
        data = {"word_to_id": self.word_to_id}
        path.write_text(json.dumps(data, ensure_ascii=True, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "Tokenizer":
        data = json.loads(path.read_text(encoding="utf-8"))
        obj: "Tokenizer" = cls.__new__(cls)
        obj.word_to_id = data["word_to_id"]
        obj.id_to_word = {int(v): k for k, v in data["word_to_id"].items()}
        return obj
