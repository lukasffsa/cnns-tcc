"""Métricas das equações 4.1–4.5 do PDF; indefinições são explícitas."""
import numpy as np


METRICS = ("r2", "wi", "rmse", "mae", "mape_percent")


def regression_metrics(observed, predicted):
    y = np.asarray(observed, dtype=np.float64)
    p = np.asarray(predicted, dtype=np.float64)
    if y.ndim != 1 or y.shape != p.shape or len(y) == 0:
        raise ValueError("Observado e predito devem ser vetores não vazios do mesmo tamanho.")
    if not np.isfinite(y).all() or not np.isfinite(p).all():
        raise ValueError("Valores não finitos na avaliação.")
    residual = p - y
    error_sum = np.sum(residual ** 2)
    center = y.mean()
    r2_denominator = np.sum((y - center) ** 2)
    wi_denominator = np.sum((np.abs(y - center) + np.abs(p - center)) ** 2)
    mape = np.mean(np.abs(residual / y)) if np.all(y != 0) else np.nan
    return {
        "r2": float(1 - error_sum / r2_denominator) if len(y) >= 2 and r2_denominator > 0 else np.nan,
        "wi": float(1 - error_sum / wi_denominator) if wi_denominator > 0 else np.nan,
        "rmse": float(np.sqrt(np.mean(residual ** 2))),
        "mae": float(np.mean(np.abs(residual))),
        "mape_fraction": float(mape),
        "mape_percent": float(100 * mape),
        "n_samples": int(len(y)),
        "zero_targets": int(np.sum(y == 0)),
    }
