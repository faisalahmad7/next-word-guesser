# 🎬 Next Word Guesser

An LSTM language model, built with TensorFlow/Keras, that predicts the next word of famous movie quotes. Type the start of a quote and the model returns its top candidate next words with probabilities, can generate a continuation, and looks up the full quote and movie from the dataset.

> **Note:** this is a learning project on a deliberately small dataset (37 quotes). It demonstrates the full NLP pipeline (tokenization, n-gram sequence generation, sequence modeling, temperature-controlled decoding), and with this little data the model largely memorizes the quotes. See [Limitations](#limitations).

## Features

- **Next-word prediction**: top-*k* candidates with probabilities
- **Temperature scaling and sampling**: control how confident or varied the output is
- **Text completion**: generate several words, one at a time
- **Quote lookup**: matches your input against the dataset and shows the movie
- **Reproducible training CLI**: hyperparameters via flags, seeded runs, saved training curves
- **Held-out evaluation**: optional validation split by *whole quote* to avoid prefix leakage
- **Tested preprocessing**: tokenizer, sequence building, splitting and sampling covered by unit tests

## How it works

```
"may the force"  ──►  tokenize  ──►  [14, 1, 29]  ──►  pad (left)  ──►  Embedding(128)
                                                                              │
              top-k words + probabilities  ◄──  Dense + softmax  ◄──  LSTM(256)
```

1. **Tokenize**: lowercase, strip punctuation, map each word to an integer id (frequency-ordered; id 0 is reserved for padding).
2. **Build n-gram sequences**: every quote becomes growing prefixes (`may the` → `may the force` → …). The last word of each prefix is the label, the rest is the input.
3. **Pad** inputs on the left to a fixed length.
4. **Model**: `Embedding(128) → LSTM(256) → Dense(softmax)` trained with sparse categorical cross-entropy and Adam.
5. **Decode**: apply temperature to the output distribution, then take the top-*k* words or sample from them.

Dataset stats: 37 quotes · 136 unique words · 166 training sequences · context window of 13 words.

## Project structure

```
next-word-guesser/
├── data/
│   └── quotes.json            # quote + movie records
├── next_word_guesser/
│   ├── data.py                # loading, tokenizer, n-gram sequences, splitting, quote lookup
│   ├── model.py               # Keras model definition
│   ├── train.py               # training CLI (saves model, tokenizer, config, curves)
│   ├── sampling.py            # temperature, top-k, sampling (NumPy only)
│   ├── predict.py             # NextWordPredictor: predict / sample / complete
│   └── cli.py                 # interactive demo
├── tests/                     # unit tests (no TensorFlow needed)
├── models/                    # trained artifacts are written here (git-ignored)
├── requirements.txt
└── README.md
```

## Getting started

```bash
git clone https://github.com/faisalahmad7/next-word-guesser.git
cd next-word-guesser

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 1. Train

```bash
python -m next_word_guesser.train --epochs 250
```

Artifacts are saved to `models/`: the model (`.keras`), `tokenizer.json`, `config.json`, `history.json` and `training_curves.png`.

Useful options:

| Flag | Default | Description |
|---|---|---|
| `--epochs` | 250 | training epochs |
| `--batch-size` | 32 | batch size |
| `--embedding-dim` | 128 | word-vector size |
| `--lstm-units` | 256 | LSTM hidden size |
| `--dropout` | 0.0 | dropout after the LSTM |
| `--val-split` | 0.0 | fraction of *quotes* held out for validation |
| `--seed` | 42 | random seed |

### 2. Try it

```bash
python -m next_word_guesser.cli
```

```
You: may the force
🤖 Predicted next words (top 5):
   <word>       → <probability>
   ...

🎬 Full quote: May the Force be with you
📽️  Movie: Star Wars
```

Type `complete <text>` to generate a continuation, or `exit` to quit. Use `--temperature` and `--top-k` to change decoding.

### Use it from Python

```python
from next_word_guesser.predict import NextWordPredictor

p = NextWordPredictor.load("models")
p.predict("you can't handle", top_k=3, temperature=0.8)   # [(word, prob), ...]
p.complete("i'll be", n_words=3)                          # generated continuation
```

### 3. Run the tests

```bash
python -m unittest discover -s tests -v
```

## Results

Fill these in after your own training run (`models/history.json` and `models/training_curves.png` have the numbers):

| Setting | Train accuracy | Validation accuracy |
|---|---|---|
| All quotes, 250 epochs (`--val-split 0.910`) | 0.910 | n/a |
| Held-out quotes (`--val-split 0.2`) | _ | _ |

## Limitations

- **Tiny dataset**: with 37 quotes the model memorizes rather than generalizes. High training accuracy here means "memorized the quotes", not "understands language". Validation on held-out quotes will be much lower, which is expected.
- **Closed vocabulary**: words not seen in training are ignored at inference.
- **Word-level model**: no sub-word handling, and no meaning beyond what's in the 37 quotes.

## Ideas for improvement

- Train on a larger corpus (e.g. a movie-dialogue dataset such as Cornell Movie-Dialogs) so the validation score means something
- Add an `<OOV>` token, dropout and early stopping
- Swap the LSTM for a Transformer, or fine-tune a pretrained model (GPT-2) and compare
- Wrap the predictor in a small Streamlit or Gradio web demo
- Report top-*k* accuracy and perplexity instead of top-1 accuracy alone

## License

MIT, see [LICENSE](LICENSE).
