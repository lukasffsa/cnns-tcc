import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error

def plot_separated_kfold_results(y_true_alt, y_pred_alt, y_true_bio, y_pred_bio):
    print("\n" + "="*50)
    print("MÉTRICAS GLOBAIS - 5-FOLD CROSS-VALIDATION")
    print("="*50)

    # Métricas para Altura
    rmse_alt = np.sqrt(mean_squared_error(y_true_alt, y_pred_alt))
    r2_alt = r2_score(y_true_alt, y_pred_alt)

    print(f"[ MODELO DEDICADO: ALTURA ]")
    print(f"Coeficiente de Determinação (R²): {r2_alt:.4f}")
    print(f"Erro Quadrático Médio (RMSE): {rmse_alt:.4f} cm\n")

    # Métricas para Biomassa
    rmse_bio = np.sqrt(mean_squared_error(y_true_bio, y_pred_bio))
    r2_bio = r2_score(y_true_bio, y_pred_bio)

    print(f"[ MODELO DEDICADO: BIOMASSA ]")
    print(f"Coeficiente de Determinação (R²): {r2_bio:.4f}")
    print(f"Erro Quadrático Médio (RMSE): {rmse_bio:.4f} kg/ha")
    print("="*50 + "\n")

    # Gráfico de Dispersão: Altura
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true_alt, y_pred_alt, alpha=0.7, color='navy', edgecolor='k')
    min_val_a = min(min(y_true_alt), min(y_pred_alt))
    max_val_a = max(max(y_true_alt), max(y_pred_alt))
    plt.plot([min_val_a, max_val_a], [min_val_a, max_val_a], 'r--', label='Ideal (y = x)')
    plt.xlabel('Altura Real (cm)')
    plt.ylabel('Altura Predita (cm)')
    plt.title(f'Altura da Alfafa (5-Fold)\nR²: {r2_alt:.3f} | RMSE: {rmse_alt:.2f} cm')
    plt.legend(loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig('dispersao_altura_kfold.png')
    plt.close()

    # Gráfico de Dispersão: Biomassa
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true_bio, y_pred_bio, alpha=0.7, color='darkgreen', edgecolor='k')
    min_val_b = min(min(y_true_bio), min(y_pred_bio))
    max_val_b = max(max(y_true_bio), max(y_pred_bio))
    plt.plot([min_val_b, max_val_b], [min_val_b, max_val_b], 'r--', label='Ideal (y = x)')
    plt.xlabel('Biomassa Real (MS kg/ha)')
    plt.ylabel('Biomassa Predita (MS kg/ha)')
    plt.title(f'Biomassa da Alfafa (5-Fold)\nR²: {r2_bio:.3f} | RMSE: {rmse_bio:.2f} kg/ha')
    plt.legend(loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig('dispersao_biomassa_kfold.png')
    plt.close()