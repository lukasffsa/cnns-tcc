"""Gráficos e tabelas em unidades originais; somente previsões OOF."""
from pathlib import Path
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from metrics import regression_metrics, METRICS


UNITS = {"altura": "cm", "biomassa": "kg/ha"}
LABELS = {"r2": "R²", "wi": "WI", "rmse": "RMSE", "mae": "MAE", "mape_percent": "MAPE (%)"}


def plot_target(y, p, target, directory, model_name="Modelo", folds=None):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    y, p = np.asarray(y), np.asarray(p)
    result = regression_metrics(y, p)
    unit = UNITS[target]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.7))
    color = "#166a72" if target == "altura" else "#477536"
    axes[0].scatter(y, p, s=32, alpha=0.75, c=color, edgecolors="white", linewidths=0.4)
    lo, hi = min(y.min(), p.min()), max(y.max(), p.max())
    if hi == lo:
        lo, hi = lo - 1, hi + 1
    padding = (hi - lo) * 0.04
    lo, hi = lo - padding, hi + padding
    axes[0].plot([lo, hi], [lo, hi], "--", color="#bc4340", label="Ideal: y = x")
    axes[0].set(xlabel=f"Observado ({unit})", ylabel=f"Predito ({unit})", xlim=(lo, hi), ylim=(lo, hi))
    axes[0].set_aspect("equal", adjustable="box")
    axes[0].legend(frameon=False)
    axes[0].set_title(f"R²={result['r2']:.3f}  WI={result['wi']:.3f}\nRMSE={result['rmse']:.2f}  MAE={result['mae']:.2f} {unit}")
    axes[1].scatter(y, p-y, s=32, alpha=0.75, c=color, edgecolors="white", linewidths=0.4)
    axes[1].axhline(0, linestyle="--", color="#bc4340")
    axes[1].set(xlabel=f"Observado ({unit})", ylabel=f"Resíduo: predito − observado ({unit})")
    axes[1].set_title(f"Viés médio={np.mean(p-y):.2f} {unit}\nMAPE={result['mape_percent']:.2f}%")
    for axis in axes:
        axis.grid(alpha=0.18)
        axis.spines[["top", "right"]].set_visible(False)
    fig.suptitle(f"{model_name} · {target.capitalize()} · previsões out-of-fold")
    fig.tight_layout()
    fig.savefig(directory / f"diagnostico_{target}.png", dpi=200)
    plt.close(fig)


def plot_separated_kfold_results(y_true_alt, y_pred_alt, y_true_bio, y_pred_bio,
                                 output_dir="resultados", model_name="Modelo"):
    """Compatível com a chamada original de quatro argumentos."""
    for target, y, p in (("altura", y_true_alt, y_pred_alt), ("biomassa", y_true_bio, y_pred_bio)):
        result = regression_metrics(y, p)
        print(f"{model_name} / {target}: " + " | ".join(f"{LABELS[key]}={result[key]:.4f}" for key in METRICS))
        if result["zero_targets"]:
            print("MAPE indefinido: existem alvos zero; nenhuma amostra foi removida.")
        plot_target(y, p, target, output_dir, model_name)


def summarize_folds(fold_metrics):
    rows = []
    for (model, target), group in fold_metrics.groupby(["model", "target"], sort=False):
        for metric in METRICS:
            values = group[metric].to_numpy(dtype=float)
            complete = np.isfinite(values).all()
            rows.append({"model": model, "target": target, "metric": metric,
                         "mean": float(values.mean()) if complete else np.nan,
                         "std_population": float(values.std(ddof=0)) if complete else np.nan,
                         "std_sample": float(values.std(ddof=1)) if complete and len(values) > 1 else np.nan,
                         "valid_folds": int(np.isfinite(values).sum()), "total_folds": len(values)})
    return pd.DataFrame(rows)


def plot_comparison(fold_metrics, output_dir):
    """Pontos de cada fold: não apresenta dispersão como intervalo de confiança."""
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    for target in ("altura", "biomassa"):
        data = fold_metrics[fold_metrics.target == target]
        if data.empty:
            continue
        models = list(data.model.unique())
        fig, axes = plt.subplots(2, 3, figsize=(13, 7))
        for axis, metric in zip(axes.flat, METRICS):
            for index, model in enumerate(models):
                rows = data[data.model == model].sort_values("fold")
                values = rows[metric].to_numpy(dtype=float)
                axis.scatter(np.full(len(values), index) + np.linspace(-0.12, 0.12, len(values)), values, s=28, alpha=0.8)
                if np.isfinite(values).all():
                    axis.plot([index - 0.22, index + 0.22], [values.mean()] * 2, color="black", linewidth=2)
            axis.set_xticks(range(len(models)), models, rotation=18, ha="right")
            suffix = f" ({UNITS[target]})" if metric in ("mae", "rmse") else ""
            axis.set_title(LABELS[metric] + suffix)
            axis.grid(axis="y", alpha=0.2)
            axis.spines[["top", "right"]].set_visible(False)
        axes.flat[-1].axis("off")
        axes.flat[-1].text(0, 0.85, "Cada ponto: um fold\nTraço preto: média dos folds\nR² e WI: maior é melhor\nRMSE, MAE e MAPE: menor é melhor\nDispersão entre folds não é IC.", va="top", fontsize=11)
        fig.suptitle(f"Comparação · {target} · mesmos folds")
        fig.tight_layout()
        fig.savefig(directory / f"comparacao_{target}.png", dpi=200)
        plt.close(fig)


def plot_absolute_errors(predictions, output_dir):
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    for target in ("altura", "biomassa"):
        table = predictions[predictions.target == target]
        models = list(table.model.unique())
        if not models:
            continue
        groups = [np.abs(table.loc[table.model == model, "predicted"].to_numpy() -
                         table.loc[table.model == model, "observed"].to_numpy()) for model in models]
        fig, axis = plt.subplots(figsize=(7.5, 4.7))
        axis.boxplot(groups, patch_artist=True, medianprops={"color": "black"},
                     boxprops={"facecolor": "#a7c9ce"})
        axis.set_xticks(np.arange(1, len(models) + 1), models)
        axis.set(ylabel=f"Erro absoluto ({UNITS[target]})", title=f"{target.capitalize()} · erros por amostra OOF")
        axis.grid(axis="y", alpha=0.2)
        axis.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        fig.savefig(directory / f"erros_absolutos_{target}.png", dpi=200)
        plt.close(fig)


def compare_result_directories(paths, output_dir):
    tables, prediction_tables, baseline, source_hash = [], [], None, None
    used_models = set()
    for path in paths:
        directory = Path(path)
        source = directory / "dataset_source.json"
        if source.exists():
            current_hash = json.loads(source.read_text())["csv_sha256"]
            if source_hash is not None and current_hash != source_hash:
                raise ValueError("Os resultados foram produzidos com arquivos CSV diferentes.")
            source_hash = current_hash
        predictions = pd.read_csv(directory / "oof_predictions.csv")
        prediction_tables.append(predictions)
        if predictions.duplicated(["sample_index", "target", "model"]).any():
            raise ValueError("Previsões OOF duplicadas.")
        for model, model_predictions in predictions.groupby("model"):
            if model in used_models:
                raise ValueError("Mesmo modelo fornecido em mais de uma pasta.")
            used_models.add(model)
            reference = model_predictions.sort_values(["sample_index", "target"])[["sample_index", "target", "fold", "observed"]].reset_index(drop=True)
            if baseline is None:
                baseline = reference
            else:
                if not reference[["sample_index", "target", "fold"]].equals(baseline[["sample_index", "target", "fold"]]):
                    raise ValueError("Modelos usam amostras ou folds diferentes.")
                np.testing.assert_allclose(reference.observed, baseline.observed, rtol=1e-6, atol=1e-6)
        tables.append(pd.read_csv(directory / "fold_metrics.csv"))
    combined = pd.concat(tables, ignore_index=True)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_dir / "fold_metrics.csv", index=False)
    summarize_folds(combined).to_csv(output_dir / "summary_folds.csv", index=False)
    plot_comparison(combined, output_dir)
    plot_absolute_errors(pd.concat(prediction_tables, ignore_index=True), output_dir)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, default=Path("comparacao"))
    args = parser.parse_args()
    compare_result_directories(args.paths, args.output)
