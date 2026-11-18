from pathlib import Path
import tempfile
import unittest
import numpy as np
import pandas as pd
from validation import choose_folds


class ValidationTests(unittest.TestCase):
    def test_ten_folds_coverage_and_seed(self):
        data = range(23)
        splits = choose_folds(data)
        self.assertEqual(len(splits), 10)
        observed = []
        repeated = choose_folds(data)
        for (train, test), (train2, test2) in zip(splits, repeated):
            self.assertFalse(set(train) & set(test))
            self.assertEqual(set(train) | set(test), set(data))
            self.assertIn(len(test), (2, 3))
            np.testing.assert_array_equal(train, train2)
            np.testing.assert_array_equal(test, test2)
            observed.extend(test)
        self.assertEqual(sorted(observed), list(data))

    def test_saved_ten_folds_and_reject_old_five(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "folds.csv"
            ids = np.repeat(np.arange(10), 2)
            pd.DataFrame({"sample_index": range(20), "fold": ids}).to_csv(path, index=False)
            splits = choose_folds(range(20), path)
            self.assertEqual(len(splits), 10)
            np.testing.assert_array_equal(splits[9][1], [18, 19])
            pd.DataFrame({"sample_index": range(20), "fold": np.repeat(np.arange(5), 4)}).to_csv(path, index=False)
            with self.assertRaises(ValueError):
                choose_folds(range(20), path)

    def test_insufficient_samples(self):
        with self.assertRaises(ValueError):
            choose_folds(range(9))


if __name__ == "__main__":
    unittest.main()
