"""Model definition."""
from __future__ import annotations


def build_model(
    vocab_size: int,
    input_length: int,
    embedding_dim: int = 128,
    lstm_units: int = 256,
    dropout: float = 0.0,
):
    """Embedding -> LSTM -> softmax over the vocabulary.

    Args:
        vocab_size: number of output classes (vocabulary size + 1 for padding).
        input_length: number of context words fed to the model.
        embedding_dim: size of the learned word vectors.
        lstm_units: hidden size of the LSTM.
        dropout: optional dropout applied after the LSTM (0 disables it).
    """
    from tensorflow import keras
    from tensorflow.keras import layers

    stack = [
        layers.Input(shape=(input_length,)),
        layers.Embedding(vocab_size, embedding_dim),
        layers.LSTM(lstm_units),
    ]
    if dropout > 0:
        stack.append(layers.Dropout(dropout))
    stack.append(layers.Dense(vocab_size, activation="softmax"))

    model = keras.Sequential(stack, name="next_word_lstm")
    model.compile(
        loss="sparse_categorical_crossentropy",  # integer labels, no one-hot matrix needed
        optimizer="adam",
        metrics=["accuracy"],
    )
    return model
