import tempfile
import unittest
from pathlib import Path

from next_word_guesser.data import (
    WordTokenizer,
    build_ngram_sequences,
    find_quote,
    load_quotes,
    max_sequence_length,
    normalize,
    pad_pre,
    prepare_xy,
    split_quotes,
)


class NormalizeTests(unittest.TestCase):
    def test_lowercases_and_strips_punctuation(self):
        self.assertEqual(normalize("Here's looking at you, kid!"), ["here's", "looking", "at", "you", "kid"])

    def test_dotted_abbreviation_splits(self):
        self.assertEqual(normalize("E.T. phone home"), ["e", "t", "phone", "home"])

    def test_empty(self):
        self.assertEqual(normalize("  ...  "), [])


class TokenizerTests(unittest.TestCase):
    def setUp(self):
        self.tok = WordTokenizer().fit(["the cat sat", "the dog sat", "the end"])

    def test_frequency_ordering_and_padding_id(self):
        self.assertEqual(self.tok.word_index["the"], 1)
        self.assertNotIn(0, self.tok.index_word)
        self.assertEqual(self.tok.vocab_size, len(self.tok.word_index) + 1)

    def test_encode_drops_unknown_words(self):
        self.assertEqual(self.tok.encode("the zebra sat"), [self.tok.word_index["the"], self.tok.word_index["sat"]])

    def test_save_load_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "tok.json"
            self.tok.save(path)
            loaded = WordTokenizer.load(path)
        self.assertEqual(loaded.word_index, self.tok.word_index)
        self.assertEqual(loaded.index_word, self.tok.index_word)


class SequenceTests(unittest.TestCase):
    def setUp(self):
        self.quotes = ["a b c d", "a b"]
        self.tok = WordTokenizer().fit(self.quotes)

    def test_ngram_sequences(self):
        ids = self.tok.encode("a b c d")
        seqs = build_ngram_sequences(["a b c d"], self.tok)
        self.assertEqual(seqs, [ids[:2], ids[:3], ids[:4]])

    def test_pad_pre_left_pads_and_truncates_left(self):
        padded = pad_pre([[1, 2], [1, 2, 3, 4, 5]], 4)
        self.assertEqual(padded.tolist(), [[0, 0, 1, 2], [2, 3, 4, 5]])

    def test_prepare_xy_shapes(self):
        seqs = build_ngram_sequences(self.quotes, self.tok)
        max_len = max_sequence_length(self.quotes, self.tok)
        X, y = prepare_xy(seqs, max_len)
        self.assertEqual(X.shape, (len(seqs), max_len - 1))
        self.assertEqual(y.shape, (len(seqs),))
        self.assertTrue((y > 0).all())


class SplitAndLookupTests(unittest.TestCase):
    def setUp(self):
        self.records = [{"quote": f"quote number {i}", "movie": f"m{i}"} for i in range(10)]

    def test_split_is_disjoint_and_sized(self):
        train, val = split_quotes(self.records, 0.2, seed=1)
        self.assertEqual(len(val), 2)
        self.assertEqual(len(train) + len(val), 10)
        self.assertFalse({r["quote"] for r in train} & {r["quote"] for r in val})

    def test_split_zero_keeps_everything(self):
        train, val = split_quotes(self.records, 0.0)
        self.assertEqual((len(train), len(val)), (10, 0))

    def test_split_is_reproducible(self):
        self.assertEqual(split_quotes(self.records, 0.3, seed=7), split_quotes(self.records, 0.3, seed=7))

    def test_invalid_fraction(self):
        with self.assertRaises(ValueError):
            split_quotes(self.records, 1.0)

    def test_find_quote_prefix_is_word_based(self):
        recs = [{"quote": "May the Force be with you", "movie": "Star Wars"}]
        self.assertEqual(find_quote(recs, "may the force")["movie"], "Star Wars")
        self.assertIsNone(find_quote(recs, "may the for"))  # partial word is not a match
        self.assertIsNone(find_quote(recs, ""))


class DatasetFileTests(unittest.TestCase):
    def test_bundled_dataset(self):
        records = load_quotes()
        self.assertEqual(len(records), 37)
        self.assertTrue(all({"quote", "movie"} <= r.keys() for r in records))
        tok = WordTokenizer().fit(r["quote"] for r in records)
        seqs = build_ngram_sequences([r["quote"] for r in records], tok)
        self.assertGreater(len(seqs), 100)
        self.assertEqual(max(len(s) for s in seqs), max_sequence_length([r["quote"] for r in records], tok))


if __name__ == "__main__":
    unittest.main()
