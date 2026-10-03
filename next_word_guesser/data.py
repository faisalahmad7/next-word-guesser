"""Dataset loading, tokenization and sequence preparation.

This module deliberately has no TensorFlow dependency so the preprocessing
logic can be unit-tested (and reused) on its own.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

DEFAULT_DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "quotes.json"

# Same punctuation set Keras' Tokenizer strips (apostrophes are kept, so
# "can't" stays one token).
_FILTERS = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'
_TRANSLATION = str.maketrans({ch: " " for ch in _FILTERS})


def load_quotes(path: str | Path = DEFAULT_DATA_PATH) -> list[dict[str, str]]:
    """Load the quote dataset: a list of ``{"quote": ..., "movie": ...}`` records."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def normalize(text: str) -> list[str]:
    """Lowercase, strip punctuation and split into word tokens."""
    return text.lower().translate(_TRANSLATION).split()


class WordTokenizer:
    """Minimal word-level tokenizer (index 0 is reserved for padding).

    Words are indexed by descending frequency, so the mapping is deterministic
    and can be saved to / restored from JSON for inference.
    """

    def __init__(self, word_index: dict[str, int] | None = None) -> None:
        self.word_index: dict[str, int] = dict(word_index or {})
        self.index_word: dict[int, str] = {i: w for w, i in self.word_index.items()}

    @property
    def vocab_size(self) -> int:
        """Number of classes the model must predict (vocabulary + padding id 0)."""
        return len(self.word_index) + 1

    def fit(self, texts: Iterable[str]) -> "WordTokenizer":
        counts = Counter(word for text in texts for word in normalize(text))
        self.word_index = {w: i for i, (w, _) in enumerate(counts.most_common(), start=1)}
        self.index_word = {i: w for w, i in self.word_index.items()}
        return self

    def encode(self, text: str) -> list[int]:
        """Convert text to token ids. Unknown words are dropped."""
        return [self.word_index[w] for w in normalize(text) if w in self.word_index]

    def save(self, path: str | Path) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.word_index, f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path: str | Path) -> "WordTokenizer":
        with open(path, encoding="utf-8") as f:
            return cls(json.load(f))


def build_ngram_sequences(quotes: Iterable[str], tokenizer: WordTokenizer) -> list[list[int]]:
    """Turn each quote into growing prefixes.

    ``"may the force"`` -> ``[may, the]``, ``[may, the, force]`` ...
    Each prefix is one training example: all-but-last word is the input, the
    last word is the label.
    """
    sequences: list[list[int]] = []
    for quote in quotes:
        ids = tokenizer.encode(quote)
        for i in range(1, len(ids)):
            sequences.append(ids[: i + 1])
    return sequences


def pad_pre(sequences: Sequence[Sequence[int]], maxlen: int) -> np.ndarray:
    """Left-pad with zeros (and left-truncate) to a fixed length."""
    out = np.zeros((len(sequences), maxlen), dtype=np.int32)
    for row, seq in enumerate(sequences):
        seq = list(seq)[-maxlen:]
        if seq:
            out[row, maxlen - len(seq):] = seq
    return out


def max_sequence_length(quotes: Iterable[str], tokenizer: WordTokenizer) -> int:
    """Length of the longest tokenized quote (model input is this minus one)."""
    return max(len(tokenizer.encode(q)) for q in quotes)


def prepare_xy(
    sequences: Sequence[Sequence[int]], max_seq_len: int
) -> tuple[np.ndarray, np.ndarray]:
    """Split padded n-gram sequences into inputs ``X`` and integer labels ``y``."""
    padded = pad_pre(sequences, max_seq_len)
    return padded[:, :-1], padded[:, -1]


def split_quotes(
    records: Sequence[dict[str, str]], val_fraction: float, seed: int = 42
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Hold out *whole quotes* for validation.

    Splitting at the quote level (not at the n-gram level) prevents prefixes of
    the same quote from leaking between train and validation sets.
    """
    if not 0.0 <= val_fraction < 1.0:
        raise ValueError("val_fraction must be in [0, 1)")
    indices = np.random.default_rng(seed).permutation(len(records))
    n_val = int(round(len(records) * val_fraction))
    val_idx = set(indices[:n_val].tolist())
    train = [r for i, r in enumerate(records) if i not in val_idx]
    val = [r for i, r in enumerate(records) if i in val_idx]
    return train, val


def find_quote(records: Sequence[dict[str, str]], text: str) -> dict[str, str] | None:
    """Return the first dataset quote that starts with the given words."""
    tokens = normalize(text)
    if not tokens:
        return None
    for record in records:
        if normalize(record["quote"])[: len(tokens)] == tokens:
            return record
    return None
