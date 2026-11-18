"""Smoke tests sintéticos; exigem o ambiente torch/torchvision do projeto."""
import importlib.util
from pathlib import Path
import tempfile
import unittest
import numpy as np
from PIL import Image

AVAILABLE = importlib.util.find_spec("torch") is not None and importlib.util.find_spec("torchvision") is not None


@unittest.skipUnless(AVAILABLE, "torch/torchvision não instalados neste ambiente")
class TorchIntegrationTests(unittest.TestCase):
    def test_shapes_and_same_augmented_pixels(self):
        import torch
        from main import ExperimentDataset
        from models.MLP import MLPSingleRegressor
        class Base:
            def __len__(self):
                return 5
            def __getitem__(self, index):
                image = Image.fromarray(np.arange(8*8*3, dtype=np.uint8).reshape(8, 8, 3))
                return image, torch.tensor([0.5, 0.5])
        for target in (0, 1):
            model = MLPSingleRegressor(target)
            self.assertEqual(model(torch.zeros(4, 2)).shape, (4,))
        a = ExperimentDataset(Base(), [0, 1], True, True, 0, 42)
        b = ExperimentDataset(Base(), [0, 1], True, True, 0, 42)
        a.epoch = b.epoch = 3
        torch.manual_seed(1)
        first = a[0][0]
        torch.manual_seed(999)
        second = b[0][0]
        torch.testing.assert_close(first, second)

    def test_mlp_training_smoke(self):
        import torch
        from main import run_experiment
        class SyntheticBase:
            max_altura = 40
            max_biomassa = 1000
            def __len__(self):
                return 10
            def __getitem__(self, index):
                rng = np.random.default_rng(index)
                image = Image.fromarray(rng.integers(20, 220, (16, 16, 3), dtype=np.uint8))
                return image, torch.tensor([0.2 + index/20, 0.1 + index/20])
        with tempfile.TemporaryDirectory() as directory:
            predictions, metrics = run_experiment(SyntheticBase(), ["mlp"], directory, epochs=1, device="cpu")
            self.assertEqual(len(predictions), 20)
            self.assertEqual(len(metrics), 20)
            self.assertTrue(np.isfinite(predictions.predicted).all())
            self.assertTrue((Path(directory) / "mlp/fold_0/altura/model.pt").exists())


if __name__ == "__main__":
    unittest.main()
