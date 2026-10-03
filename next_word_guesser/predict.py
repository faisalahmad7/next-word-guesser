"""Inference: load a trained model and predict / sample / complete text."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from next_word_guesser.data import (
    DEFAULT_DATA_PATH,
    WordTokenizer,
    find_quote,
    load_quotes,
    pad_pre,
)
from next_word_guesser.sampling import apply_temperature, sample_index, top_k_indices


class NextWordPredictor:
    """Wraps a trained Keras model together with its tokenizer."""

    def __init__(self, model, tokenizer: WordTokenizer, max_seq_len: int,
                 records: list[dict[str, str]] | None = None) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.records = records or []

    @classmethod
    def load(cls, model_dir: str | Path = "models",
             data_path: str | Path = DEFAULT_DATA_PATH) -> "NextWordPredictor":
        import os
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

        from next_word_guesser._quiet import suppress_stderr
        from next_word_guesser.train import CONFIG_FILE, MODEL_FILE, TOKENIZER_FILE

        model_dir = Path(model_dir)
        model_path = model_dir / MODEL_FILE
        if not model_path.exists():
            raise FileNotFoundError(
                f"No trained model at {model_path}. Run `python -m next_word_guesser.train` first."
            )
        config = json.loads((model_dir / CONFIG_FILE).read_text())

        with suppress_stderr():  # hides TensorFlow's startup warnings only
            import logging
            logging.getLogger("tensorflow").setLevel(logging.ERROR)
            from tensorflow import keras
            model = keras.models.load_model(model_path)

        return cls(
            model,
            WordTokenizer.load(model_dir / TOKENIZER_FILE),
            int(config["max_seq_len"]),
            load_quotes(data_path),
        )

    def _probabilities(self, text: str) -> np.ndarray:
        ids = self.tokenizer.encode(text)
        if not ids:
            raise ValueError("None of the input words are in the model's vocabulary.")
        x = pad_pre([ids], self.max_seq_len - 1)
        probs = self.model.predict(x, verbose=0)[0].astype(np.float64)
        probs[0] = 0.0  # id 0 is padding, never a valid prediction
        return probs / probs.sum()

    def _scaled_probabilities(self, text: str, temperature: float) -> np.ndarray:
        scaled = apply_temperature(self._probabilities(text), temperature)
        scaled[0] = 0.0  # keep the padding id out after rescaling
        return scaled / scaled.sum()

    def predict(self, text: str, top_k: int = 5, temperature: float = 1.0) -> list[tuple[str, float]]:
        """Top-k next-word candidates with (temperature-scaled) probabilities."""
        scaled = self._scaled_probabilities(text, temperature)
        return [(self.tokenizer.index_word[int(i)], float(scaled[i]))
                for i in top_k_indices(scaled, top_k)]

    def sample(self, text: str, temperature: float = 0.8, top_k: int = 5,
               rng: np.random.Generator | None = None) -> str:
        """Randomly sample the next word (higher temperature = more variety)."""
        probs = self._scaled_probabilities(text, temperature)
        return self.tokenizer.index_word[sample_index(probs, top_k=top_k, rng=rng)]

    def complete(self, text: str, n_words: int = 5, temperature: float = 0.5,
                 top_k: int = 5, rng: np.random.Generator | None = None) -> str:
        """Generate ``n_words`` more words, one at a time."""
        out = text.strip()
        for _ in range(n_words):
            out += " " + self.sample(out, temperature=temperature, top_k=top_k, rng=rng)
        return out

    def find_quote(self, text: str) -> dict[str, str] | None:
        return find_quote(self.records, text)
