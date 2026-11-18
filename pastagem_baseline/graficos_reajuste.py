#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Boxplots do reajuste (le os CSV gravados por reajuste_baseline.py).

Uso:
    python graficos_reajuste.py resultados

Dependencias: numpy, matplotlib
"""
import csv
import os
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# cores e tinta
FUNDO, TINTA, TINTA2, MUDO, GRADE, EIXO = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
AZUL, LARANJA, AQUA, CINZA = "#2a78d6", "#eb6834", "#1baf7a", "#b5b3ab"

ALVOS = [("altura", "Altura", "cm"), ("biomassa", "Biomassa", "kg/ha")]
MODELOS = [("lasso", "LASSO"), ("svr", "SVR"), ("mlp", "MLP")]
REFERENCIAS = [("media_treino", "Média do\ntreino"), ("media_cultivar_momento", "Média por\ncultivar e\nmomento")]
CENARIOS = [("publicado_imagem", "Seleção nos folds de teste, folds por imagem", AZUL),
            ("publicado_parcela", "Seleção nos folds de teste, folds por parcela", LARANJA),
            ("aninhado_parcela", "Validação aninhada, folds por parcela", AQUA)]
# Numeros do rascunho do TCC do Lukas (ResNet50, 5 folds, divisao nao informada)
CNN_RASCUNHO = {"altura": 4.1135, "biomassa": 286.3794}


def ler(caminho):
    with open(caminho, encoding="utf-8") as f:
        return list(csv.DictReader(f, delimiter=";"))


def estilo(ax):
    ax.set_facecolor(FUNDO)
    for lado in ("top", "right", "left"):
        ax.spines[lado].set_visible(False)
    ax.spines["bottom"].set_color(EIXO)
    ax.grid(axis="y", color=GRADE, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUDO, length=0, labelsize=9)
    for r in ax.get_xticklabels():
        r.set_color(TINTA2)


def caixa(ax, valores, posicao, cor, largura=0.6):
    bp = ax.boxplot([valores], positions=[posicao], widths=largura, patch_artist=True,
                    flierprops=dict(marker="o", markersize=3, markerfacecolor=cor, markeredgecolor="none", alpha=0.7),
                    medianprops=dict(color=TINTA, linewidth=1.4),
                    whiskerprops=dict(color=TINTA2, linewidth=0.9),
                    capprops=dict(color=TINTA2, linewidth=0.9),
                    boxprops=dict(facecolor=cor, edgecolor="none", alpha=0.9))
    return bp


def figura_aninhado(por_fold, pasta):
    """RMSE por fold na validacao aninhada, com as regras de referencia."""
    reps = {int(l["rep"]) for l in por_fold if l["cenario"] == "aninhado_parcela"}
    dados = defaultdict(list)
    for l in por_fold:
        # as regras de referencia entram com as mesmas repeticoes (mesmos folds) do aninhado
        if int(l["rep"]) in reps:
            dados[(l["cenario"], l["alvo"], l["modelo"])].append(float(l["RMSE"]))
    fig, eixos = plt.subplots(1, 2, figsize=(11, 4.6), facecolor=FUNDO)
    for ax, (alvo, nome, unidade) in zip(eixos, ALVOS):
        estilo(ax)
        rotulos = []
        for i, (m, r) in enumerate(MODELOS):
            caixa(ax, dados[("aninhado_parcela", alvo, m)], i, AZUL)
            rotulos.append(r)
        for j, (m, r) in enumerate(REFERENCIAS):
            caixa(ax, dados[("publicado_parcela", alvo, m)], len(MODELOS) + j, CINZA)
            rotulos.append(r)
        ax.set_xticks(range(len(rotulos)))
        ax.set_xticklabels(rotulos)
        ax.set_title("%s: RMSE por fold (%s)" % (nome, unidade), loc="left", fontsize=11, color=TINTA)
        ax.set_ylim(bottom=0)
    fig.legend(handles=[Patch(color=AZUL, label="Atributos manuais + modelo reajustado"),
                        Patch(color=CINZA, label="Regra de referência, sem usar a imagem")],
               loc="lower center", ncol=2, frameon=False, fontsize=9, labelcolor=TINTA2)
    fig.suptitle("Baseline reajustado, validação aninhada com folds por parcela",
                 x=0.02, ha="left", fontsize=13, color=TINTA)
    fig.tight_layout(rect=(0, 0.07, 1, 0.94))
    fig.savefig(os.path.join(pasta, "reajuste_boxplot_rmse_por_fold.png"), dpi=160, facecolor=FUNDO)
    plt.close(fig)


def figura_cenarios(por_rep, pasta):
    """RMSE global por repeticao: efeito de como os folds e a selecao sao feitos."""
    dados = defaultdict(list)
    for l in por_rep:
        dados[(l["cenario"], l["alvo"], l["modelo"])].append(float(l["RMSE"]))
    fig, eixos = plt.subplots(1, 2, figsize=(11, 5.3), facecolor=FUNDO)
    for ax, (alvo, nome, unidade) in zip(eixos, ALVOS):
        estilo(ax)
        for i, (m, r) in enumerate(MODELOS):
            for j, (c, _, cor) in enumerate(CENARIOS):
                v = dados[(c, alvo, m)]
                if v:
                    caixa(ax, v, i + (j - 1) * 0.27, cor, largura=0.22)
        ax.set_xticks(range(len(MODELOS)))
        ax.set_xticklabels([r for _, r in MODELOS])
        ax.axhline(CNN_RASCUNHO[alvo], color=TINTA2, linewidth=1.0, linestyle=(0, (4, 3)), zorder=1)
        ax.set_title("%s: RMSE global por repetição (%s)" % (nome, unidade), loc="left", fontsize=11, color=TINTA)
        ax.set_ylim(bottom=0)
    fig.legend(handles=[Patch(color=cor, label=rot) for _, rot, cor in CENARIOS]
               + [Line2D([0], [0], color=TINTA2, linewidth=1.0, linestyle=(0, (4, 3)),
                         label="CNN no rascunho do Lukas (forma de dividir os folds não informada)")],
               loc="lower center", ncol=1, frameon=False, fontsize=9, labelcolor=TINTA2)
    fig.suptitle("O mesmo baseline sob três formas de validar",
                 x=0.02, ha="left", fontsize=13, color=TINTA)
    fig.tight_layout(rect=(0, 0.17, 1, 0.94))
    fig.savefig(os.path.join(pasta, "reajuste_boxplot_cenarios.png"), dpi=160, facecolor=FUNDO)
    plt.close(fig)


def main():
    pasta = sys.argv[1] if len(sys.argv) > 1 else "resultados"
    figura_aninhado(ler(os.path.join(pasta, "reajuste_metricas_por_fold.csv")), pasta)
    figura_cenarios(ler(os.path.join(pasta, "reajuste_metricas_por_repeticao.csv")), pasta)
    print("Figuras gravadas em", pasta)


if __name__ == "__main__":
    main()
