"""Pure-NumPy helpers for turning model probabilities into predictions."""
from __future__ import annotations

import numpy as np


def apply_temperature(probs: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Rescale a probability distribution by a temperature.

    ``temperature < 1`` sharpens the distribution (more confident);
    ``temperature > 1`` flattens it (more random). Ranking is unchanged, so the
    effect shows up when *sampling* or when reading the probabilities.
    """
    if temperature <= 0:
        raise ValueError("temperature must be > 0")
    logits = np.log(np.asarray(probs, dtype=np.float64) + 1e-10) / temperature
    logits -= logits.max()  # numerical stability
    exp = np.exp(logits)
    return exp / exp.sum()


def top_k_indices(probs: np.ndarray, k: int) -> np.ndarray:
    """Indices of the ``k`` most probable entries, best first."""
    k = max(1, min(k, len(probs)))
    return np.argsort(probs)[::-1][:k]


def sample_index(
    probs: np.ndarray, top_k: int | None = None, rng: np.random.Generator | None = None
) -> int:
    """Sample one index from ``probs``, optionally restricted to the top-k."""
    rng = rng or np.random.default_rng()
    probs = np.asarray(probs, dtype=np.float64)
    if top_k is not None:
        keep = top_k_indices(probs, top_k)
        masked = np.zeros_like(probs)
        masked[keep] = probs[keep]
        probs = masked
    probs = probs / probs.sum()
    return int(rng.choice(len(probs), p=probs))
