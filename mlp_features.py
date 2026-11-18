import numpy as np
from pastagem_ml.features import extract_features


FEATURE_NAMES = {
    0: ["entropia", "homogeneidade"],
    1: ["altura_linear_java", "gli_medio"],
}


def mlp_feature_vector(image, target_idx):
    """Descritores sobre RGB após augmentation, antes de Normalize(ImageNet)."""
    if target_idx not in FEATURE_NAMES:
        raise ValueError("Alvo inválido.")
    features = extract_features(np.asarray(image.convert("RGB"), dtype=np.uint8))
    if target_idx == 0:
        vector = [features.entropy, features.homogeneity]
    else:
        linear_height = (-437.2714 - 41.8438 * features.entropy
                         + 357.7855 * features.homogeneity - 0.24237 * features.gray_std
                         + 0.1394 * features.dark_area_percent)
        vector = [linear_height, features.gli_mean]
    return np.asarray(vector, dtype=np.float32)
