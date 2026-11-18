"""Casos sintéticos exclusivamente para verificação do código."""
from pathlib import Path
import tempfile
import unittest
import numpy as np
import pandas as pd
from PIL import Image
from metrics import regression_metrics
from mlp_features import mlp_feature_vector
from views import (plot_target, plot_comparison, plot_absolute_errors,
                   summarize_folds, compare_result_directories)


class ComparisonTests(unittest.TestCase):
    def test_paper_metrics(self):
        result = regression_metrics([1, 2, 3], [1, 2, 4])
        self.assertAlmostEqual(result["r2"], 0.5)
        self.assertAlmostEqual(result["wi"], 12/13)
        self.assertAlmostEqual(result["rmse"], np.sqrt(1/3))
        self.assertAlmostEqual(result["mae"], 1/3)
        self.assertAlmostEqual(result["mape_percent"], 100/9)

    def test_undefined_metrics(self):
        self.assertTrue(np.isnan(regression_metrics([0, 1], [0, 1])["mape_percent"]))
        self.assertTrue(np.isnan(regression_metrics([1, 1], [1, 1])["r2"]))
        self.assertTrue(np.isnan(regression_metrics([1, 1], [1, 1])["wi"]))
        with self.assertRaises(ValueError):
            regression_metrics([1, np.nan], [1, 2])

    def test_mlp_features_follow_java(self):
        image = Image.new("RGB", (8, 6), (60, 120, 30))
        np.testing.assert_allclose(mlp_feature_vector(image, 0), [0, 1])
        np.testing.assert_allclose(mlp_feature_vector(image, 1), [-79.4859, 150/330], rtol=1e-6)

    def test_fold_mean_preserves_undefined(self):
        rows = []
        for fold in range(5):
            result = regression_metrics([1, 2], [1, 2])
            if fold == 0:
                result["mape_percent"] = np.nan
            rows.append({"model": "mlp", "target": "altura", "fold": fold, **result})
        summary = summarize_folds(pd.DataFrame(rows))
        row = summary[summary.metric == "mape_percent"].iloc[0]
        self.assertTrue(np.isnan(row["mean"]))
        self.assertEqual(row.valid_folds, 4)

    def test_plot_generation_and_comparison_alignment(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            tables, predictions = [], []
            for model in ("mlp", "mobilenet"):
                rows, preds = [], []
                for target in ("altura", "biomassa"):
                    for fold in range(5):
                        y = np.array([1, 2]) + fold * 2
                        p = y + (0.2 if model == "mlp" else -0.3)
                        rows.append({"model": model, "target": target, "fold": fold, **regression_metrics(y, p)})
                        for offset in range(2):
                            preds.append({"model": model, "target": target, "fold": fold,
                                          "sample_index": 2*fold+offset, "observed": y[offset], "predicted": p[offset]})
                tables.append(pd.DataFrame(rows))
                predictions.append(pd.DataFrame(preds))
                path = directory / model
                path.mkdir()
                tables[-1].to_csv(path / "fold_metrics.csv", index=False)
                predictions[-1].to_csv(path / "oof_predictions.csv", index=False)
            plot_target([1, 2, 3], [1, 2.2, 2.9], "altura", directory, "Teste sintético")
            combined = pd.concat(tables)
            plot_comparison(combined, directory)
            plot_absolute_errors(pd.concat(predictions), directory)
            compare_result_directories([directory / "mlp", directory / "mobilenet"], directory / "comparison")
            for name in ("diagnostico_altura.png", "comparacao_altura.png", "comparacao_biomassa.png", "erros_absolutos_altura.png"):
                self.assertGreater((directory / name).stat().st_size, 1000)
            wrong = predictions[1].copy()
            wrong.loc[0, "fold"] = 4
            wrong.to_csv(directory / "mobilenet" / "oof_predictions.csv", index=False)
            with self.assertRaises(ValueError):
                compare_result_directories([directory / "mlp", directory / "mobilenet"], directory / "invalid")


if __name__ == "__main__":
    unittest.main()
