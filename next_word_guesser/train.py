"""Train the next-word model.

Example:
    python -m next_word_guesser.train --epochs 250
    python -m next_word_guesser.train --epochs 100 --val-split 0.2   # held-out quotes
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

from next_word_guesser.data import (
    DEFAULT_DATA_PATH,
    WordTokenizer,
    build_ngram_sequences,
    load_quotes,
    max_sequence_length,
    prepare_xy,
    split_quotes,
)
from next_word_guesser.model import build_model

MODEL_FILE = "next_word_guesser.keras"
TOKENIZER_FILE = "tokenizer.json"
CONFIG_FILE = "config.json"
HISTORY_FILE = "history.json"
PLOT_FILE = "training_curves.png"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default=str(DEFAULT_DATA_PATH), help="path to quotes JSON")
    p.add_argument("--output-dir", default="models", help="where to save model artifacts")
    p.add_argument("--epochs", type=int, default=250)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--embedding-dim", type=int, default=128)
    p.add_argument("--lstm-units", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument(
        "--val-split", type=float, default=0.0,
        help="fraction of whole quotes held out for validation (0 = train on everything)",
    )
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def save_plot(history: dict[str, list[float]], path: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, metric, title in zip(axes, ("loss", "accuracy"), ("Loss", "Accuracy")):
        ax.plot(history[metric], label="train")
        if f"val_{metric}" in history:
            ax.plot(history[f"val_{metric}"], label="validation")
        ax.set_title(title)
        ax.set_xlabel("epoch")
        ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main() -> None:
    args = parse_args()

    import numpy as np
    import tensorflow as tf

    tf.keras.utils.set_random_seed(args.seed)

    records = load_quotes(args.data)
    all_quotes = [r["quote"] for r in records]

    tokenizer = WordTokenizer().fit(all_quotes)
    max_seq_len = max_sequence_length(all_quotes, tokenizer)

    train_records, val_records = split_quotes(records, args.val_split, args.seed)
    X_train, y_train = prepare_xy(
        build_ngram_sequences([r["quote"] for r in train_records], tokenizer), max_seq_len
    )
    validation_data = None
    if val_records:
        X_val, y_val = prepare_xy(
            build_ngram_sequences([r["quote"] for r in val_records], tokenizer), max_seq_len
        )
        validation_data = (X_val, y_val)

    print(f"Quotes: {len(records)} | vocab size: {tokenizer.vocab_size} | "
          f"context length: {max_seq_len - 1}")
    print(f"Training examples: {len(X_train)}"
          + (f" | validation examples: {len(X_val)}" if validation_data else ""))

    model = build_model(
        tokenizer.vocab_size, max_seq_len - 1,
        embedding_dim=args.embedding_dim, lstm_units=args.lstm_units, dropout=args.dropout,
    )
    model.summary()

    history = model.fit(
        X_train, y_train,
        epochs=args.epochs, batch_size=args.batch_size,
        validation_data=validation_data, verbose=2,
    ).history

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    model.save(out / MODEL_FILE)
    tokenizer.save(out / TOKENIZER_FILE)
    (out / CONFIG_FILE).write_text(json.dumps(
        {"max_seq_len": max_seq_len, "vocab_size": tokenizer.vocab_size, **vars(args)}, indent=2))
    (out / HISTORY_FILE).write_text(json.dumps({k: [float(v) for v in vs] for k, vs in history.items()}, indent=2))
    save_plot(history, out / PLOT_FILE)

    print(f"\nFinal train accuracy: {history['accuracy'][-1]:.3f}")
    if validation_data:
        print(f"Final validation accuracy: {history['val_accuracy'][-1]:.3f}")
    print(f"Artifacts saved to: {out.resolve()}")


if __name__ == "__main__":
    main()
