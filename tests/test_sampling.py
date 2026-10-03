import unittest

import numpy as np

from next_word_guesser.sampling import apply_temperature, sample_index, top_k_indices


class TemperatureTests(unittest.TestCase):
    def setUp(self):
        self.probs = np.array([0.6, 0.3, 0.1])

    def test_sums_to_one(self):
        for t in (0.2, 1.0, 5.0):
            self.assertAlmostEqual(apply_temperature(self.probs, t).sum(), 1.0)

    def test_temperature_one_is_identity(self):
        np.testing.assert_allclose(apply_temperature(self.probs, 1.0), self.probs, atol=1e-6)

    def test_low_temperature_sharpens_high_flattens(self):
        self.assertGreater(apply_temperature(self.probs, 0.5)[0], 0.6)
        self.assertLess(apply_temperature(self.probs, 2.0)[0], 0.6)

    def test_ranking_unchanged(self):
        np.testing.assert_array_equal(np.argsort(apply_temperature(self.probs, 0.3)), np.argsort(self.probs))

    def test_invalid_temperature(self):
        with self.assertRaises(ValueError):
            apply_temperature(self.probs, 0)


class TopKAndSampleTests(unittest.TestCase):
    def test_top_k_order(self):
        self.assertEqual(top_k_indices(np.array([0.1, 0.5, 0.4]), 2).tolist(), [1, 2])

    def test_top_k_larger_than_vocab(self):
        self.assertEqual(len(top_k_indices(np.array([0.5, 0.5]), 10)), 2)

    def test_sample_respects_top_k(self):
        probs = np.array([0.05, 0.5, 0.4, 0.05])
        rng = np.random.default_rng(0)
        draws = {sample_index(probs, top_k=2, rng=rng) for _ in range(200)}
        self.assertEqual(draws, {1, 2})

    def test_sample_deterministic_with_seed(self):
        probs = np.array([0.25] * 4)
        a = [sample_index(probs, rng=np.random.default_rng(3)) for _ in range(5)]
        b = [sample_index(probs, rng=np.random.default_rng(3)) for _ in range(5)]
        self.assertEqual(a, b)


if __name__ == "__main__":
    unittest.main()
