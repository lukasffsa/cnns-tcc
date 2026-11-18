"""Descritores usados pelas Activities Java, incluindo suas convenções."""
from dataclasses import dataclass, asdict
import math
import numpy as np
from PIL import Image


@dataclass(frozen=True)
class Features:
    entropy: float
    homogeneity: float
    angular_second_moment: float
    gray_mean: float
    gray_std: float
    dark_area_percent: float
    gli_mean: float
    valid_pixels: int
    width: int
    height: int

    def to_dict(self):
        return asdict(self)


def load_image(path, size=None):
    """Sem redimensionamento ou rotação EXIF implícitos; size=(largura, altura)."""
    with Image.open(path) as source:
        image = source.convert("RGBA")
    if size is not None:
        if len(size) != 2 or any(int(v) != v or v < 1 for v in size):
            raise ValueError("Dimensões devem ser inteiros positivos.")
        image = image.resize(tuple(map(int, size)), Image.Resampling.BILINEAR)
    return np.array(image, dtype=np.uint8)


def extract_features(image):
    """Aceita RGB/RGBA uint8 em ordem H×W×C; não aceita BGR do OpenCV."""
    pixels = np.asarray(image)
    if pixels.dtype != np.uint8 or pixels.ndim != 3 or pixels.shape[2] not in (3, 4):
        raise ValueError("Imagem deve ser um array uint8 RGB ou RGBA.")
    h, w, _ = pixels.shape
    if h < 1 or w < 2:
        raise ValueError("Imagem exige altura >= 1 e largura >= 2 para a GLCM.")
    valid = pixels[:, :, 3] != 0 if pixels.shape[2] == 4 else np.ones((h, w), bool)
    count = int(valid.sum())
    if count == 0:
        raise ValueError("Imagem não contém pixels com alfa diferente de zero.")
    rgb = pixels[:, :, :3].astype(np.float64)
    total = rgb[:, :, 0] + rgb[:, :, 1] + rgb[:, :, 2]
    gray = total / 3.0
    gray_values = gray[valid]
    mean = float(np.cumsum(gray_values)[-1]) / count
    denominator = 2 * rgb[:, :, 1] + rgb[:, :, 0] + rgb[:, :, 2]
    gli = np.zeros((h, w), dtype=np.float64)
    np.divide(2 * rgb[:, :, 1] - rgb[:, :, 0] - rgb[:, :, 2], denominator,
              out=gli, where=denominator != 0)
    gli_mean = float(np.cumsum(gli[valid])[-1]) / count
    # O Java converte o acumulador para float32 depois de cada pixel.
    variance_sum = 0.0
    for value in gray_values:
        variance_sum = float(np.float32(variance_sum + math.pow(float(value) - mean, 2)))
    std = float(np.float32(math.sqrt(variance_sum / count)))
    levels = (total / 96.0).astype(np.int64)
    pair_mask = valid[:, :-1]
    # Somente o alfa do pixel esquerdo é examinado no original.
    codes = (levels[:, :-1] * 8 + levels[:, 1:])[pair_mask]
    glcm = np.bincount(codes, minlength=64).reshape(8, 8).astype(np.float64)
    glcm /= (w - 1) * h
    entropy = homogeneity = angular = 0.0
    for i in range(8):
        for j in range(8):
            p = float(glcm[i, j])
            angular += math.pow(p, 2)
            if p != 0:
                entropy += p * (math.log(p) / math.log(2))
            homogeneity += p / (1 + math.pow(i - j, 2))
    dark = 100 * (int((gray_values <= 30).sum()) / count)
    return Features(entropy, homogeneity, angular, mean, std, dark, gli_mean, count, w, h)
