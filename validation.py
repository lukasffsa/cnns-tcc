"""Partição compartilhada pelos quatro modelos, sem dependência de PyTorch."""
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

N_FOLDS = 10
KFOLD_SEED = 42


def choose_folds(dataset, path=None):
    if len(dataset) < N_FOLDS:
        raise ValueError(f"{N_FOLDS} folds exigem pelo menos {N_FOLDS} amostras.")
    if path is None:
        return list(KFold(n_splits=N_FOLDS, shuffle=True, random_state=KFOLD_SEED)
                    .split(np.arange(len(dataset))))
    table = pd.read_csv(path)
    if set(table.columns) != {"sample_index", "fold"} or table.sample_index.duplicated().any():
        raise ValueError("Folds exigem sample_index e fold únicos por amostra.")
    if set(table.sample_index) != set(range(len(dataset))) or set(table.fold) != set(range(N_FOLDS)):
        raise ValueError(f"Os folds devem cobrir todas as amostras e usar índices 0 a {N_FOLDS - 1}; arquivos antigos de 5 folds não são aceitos.")
    folds = table.set_index("sample_index").loc[range(len(dataset)), "fold"].to_numpy()
    return [(np.flatnonzero(folds != i), np.flatnonzero(folds == i)) for i in range(N_FOLDS)]
