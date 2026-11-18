#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Junta todos os modelos (baseline, CNNs com fine-tuning, CNN congelada e
variantes com cultivar/momento) nos mesmos folds por parcela e gera a tabela
e os boxplots finais.

Uso:
    python consolidar_resultados.py dataset/metadata.csv resultados \
        --cnn ResNet-50=predicoes_resnet.csv --cnn MobileNetV3-L=predicoes_mobilenet.csv \
        --cnn EfficientNet-B0=predicoes_efficientnet.csv

Le de <resultados>/: folds_por_parcela.csv, reajuste_predicoes_aninhado.csv e
predicoes_embeddings.csv. A CR17 fica fora da biomassa (MS vazio no metadata).
Grava resultados_finais.csv, resultados_finais.txt e
resultados_finais_boxplot_<alvo>.png.
"""
import argparse
import csv
import math
import os
import re
from collections import defaultdict

import numpy as np

PADRAO = re.compile(r"^([A-Za-z]{2})(\d{2})[Rr0](\d)_M(\d)")
K = 5
REPS = range(10)
COL = {"altura": "Média", "biomassa": "MS (kg/ha)"}
UNIDADE = {"altura": "cm", "biomassa": "kg/ha"}

# grupo, rotulo curto, chave interna
MODELOS = [
    ("ref", "Média do\ntreino", "media_treino"),
    ("ref", "Média por\ncultivar e\nmomento", "media_cultivar_momento"),
    ("manual", "LASSO\nmanual", "lasso"),
    ("manual", "SVR\nmanual", "svr"),
    ("manual", "MLP\nmanual", "mlp"),
    ("ft", "ResNet-50\nfine-tuning", "cnn:ResNet-50"),
    ("ft", "MobileNetV3\nfine-tuning", "cnn:MobileNetV3-L"),
    ("ft", "EfficientNet\nfine-tuning", "cnn:EfficientNet-B0"),
    ("cong", "ResNet-50\ncongelada", "emb/lasso"),
    ("cong", "Congelada\n+ manual", "emb+manual/lasso"),
    ("ctx", "Manual\n+ contexto", "manual+ctx/lasso"),
    ("ctx", "Congelada\n+ manual\n+ contexto", "emb+manual+ctx/lasso"),
]
NOME_GRUPO = {"ref": "Regra de referência, sem olhar a imagem",
              "manual": "Atributos manuais (método do artigo reajustado)",
              "ft": "CNN com fine-tuning (Lukas)",
              "cong": "ResNet-50 congelada como extrator",
              "ctx": "Com cultivar e momento como entrada"}
COR = {"ref": "#b5b3ab", "manual": "#2a78d6", "ft": "#eb6834", "cong": "#1baf7a", "ctx": "#4a3aa7"}


def ler(c):
    with open(c, encoding="utf-8-sig") as f:
        return list(csv.DictReader(f, delimiter=";"))


def chave(arq):
    m = PADRAO.match(arq)
    cul, num, _, mom = m.groups()
    return cul.upper() + num, cul.upper(), "M" + mom


def metricas(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    sq = np.sum((y - p) ** 2)
    st = np.sum((y - y.mean()) ** 2)
    wi = np.sum((np.abs(p - y.mean()) + np.abs(y - y.mean())) ** 2)
    return {"R2": 1 - sq / st, "WI": 1 - sq / wi, "RMSE": math.sqrt(sq / len(y)),
            "MAE": float(np.mean(np.abs(y - p))), "MAPE": float(100 * np.mean(np.abs((p - y) / y)))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("metadata")
    ap.add_argument("resultados")
    ap.add_argument("--cnn", action="append", default=[], help="Nome=arquivo.csv")
    ap.add_argument("--embeddings", default=None, help="predicoes_embeddings.csv (padrao: em resultados)")
    a = ap.parse_args()

    meta = {l["ALTURA"].strip().upper(): l for l in ler(a.metadata)}
    folds = {l["arquivo"]: l for l in ler(os.path.join(a.resultados, "folds_por_parcela.csv"))}
    info = {}
    for arq in folds:
        parcela, cul, mom = chave(arq)
        m = meta[parcela + mom]
        info[arq] = {"parcela": parcela, "cultivar": cul, "momento": mom,
                     "altura": float(m["Média"]),
                     "biomassa": float(m["MS (kg/ha)"]) if m["MS (kg/ha)"].strip() else None}

    pred = defaultdict(dict)  # (alvo, modelo, rep) -> {arquivo: previsto}
    for l in ler(os.path.join(a.resultados, "reajuste_predicoes_aninhado.csv")):
        pred[(l["alvo"], l["modelo"], int(l["rep"]))][l["arquivo"]] = float(l["previsto"])
    for l in ler(a.embeddings or os.path.join(a.resultados, "predicoes_embeddings.csv")):
        pred[(l["alvo"], l["modelo"], int(l["rep"]))][l["arquivo"]] = float(l["previsto"])
    for item in a.cnn:
        nome, caminho = item.split("=", 1)
        for l in ler(caminho):
            pred[(l["alvo"], "cnn:" + nome, int(l["rep"]))][l["arquivo"]] = float(l["previsto"])

    linhas, texto, por_fold_todos = [], [], {}
    for alvo in ("altura", "biomassa"):
        arqs = sorted(x for x in info if info[x][alvo] is not None)
        y = np.array([info[x][alvo] for x in arqs])
        for rep in REPS:
            k = {x: int(folds[x]["rep%02d" % rep]) for x in arqs}
            for x in arqs:
                tr = [o for o in arqs if k[o] != k[x]]
                pred[(alvo, "media_treino", rep)][x] = float(np.mean([info[o][alvo] for o in tr]))
                ig = [o for o in tr if info[o]["cultivar"] == info[x]["cultivar"]
                      and (alvo == "biomassa" or info[o]["momento"] == info[x]["momento"])]
                pred[(alvo, "media_cultivar_momento", rep)][x] = float(np.mean([info[o][alvo] for o in ig]))

        texto.append("%s (%s), folds por parcela, rep00 a rep09%s" % (
            alvo.upper(), UNIDADE[alvo], ", CR17 fora" if alvo == "biomassa" else ""))
        texto.append("  %-34s %-15s %-13s %-15s %s" % ("modelo", "R2 por foto", "RMSE por foto",
                                                       "R2 por parcela", "RMSE por parcela"))
        por_fold = {}
        for grupo, rot, m in MODELOS:
            g_foto, g_parc, folds_rmse = [], [], []
            ok = True
            for rep in REPS:
                d = pred.get((alvo, m, rep))
                if not d or any(x not in d for x in arqs):
                    ok = False
                    break
                p = np.array([d[x] for x in arqs])
                g_foto.append(metricas(y, p))
                kk = np.array([int(folds[x]["rep%02d" % rep]) for x in arqs])
                for f in range(K):
                    folds_rmse.append(metricas(y[kk == f], p[kk == f])["RMSE"])
                # avaliacao por parcela e momento: media das fotos
                grp = defaultdict(list)
                for x, v in zip(arqs, p):
                    grp[(info[x]["parcela"], info[x]["momento"])].append((v, info[x][alvo]))
                yp = np.array([v[0][1] for v in grp.values()])
                pp = np.array([np.mean([t[0] for t in v]) for v in grp.values()])
                g_parc.append(metricas(yp, pp))
            if not ok:
                texto.append("  %-34s (sem predicoes completas)" % rot.replace("\n", " "))
                continue
            por_fold[m] = folds_rmse
            med = lambda L, c: (np.mean([g[c] for g in L]), np.std([g[c] for g in L]))
            r2f, rmf, r2p, rmp = med(g_foto, "R2"), med(g_foto, "RMSE"), med(g_parc, "R2"), med(g_parc, "RMSE")
            nome = rot.replace("\n", " ")
            texto.append("  %-34s %6.2f (%.2f)   %7.2f (%.2f)  %6.2f (%.2f)    %7.2f (%.2f)" % (
                nome, r2f[0], r2f[1], rmf[0], rmf[1], r2p[0], r2p[1], rmp[0], rmp[1]))
            row = {"alvo": alvo, "grupo": NOME_GRUPO[grupo], "modelo": nome}
            for c in ("R2", "WI", "RMSE", "MAE", "MAPE"):
                row["%s_foto" % c] = round(med(g_foto, c)[0], 4)
                row["%s_foto_dp" % c] = round(med(g_foto, c)[1], 4)
            for c in ("R2", "RMSE"):
                row["%s_parcela" % c] = round(med(g_parc, c)[0], 4)
                row["%s_parcela_dp" % c] = round(med(g_parc, c)[1], 4)
            linhas.append(row)
        texto.append("  media (desvio padrao) entre as 10 repeticoes; 'por parcela' = media das fotos de cada parcela e momento")
        texto.append("")
        por_fold_todos[alvo] = por_fold

    with open(os.path.join(a.resultados, "resultados_finais.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(linhas[0].keys()), delimiter=";")
        w.writeheader()
        w.writerows(linhas)
    with open(os.path.join(a.resultados, "resultados_finais.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(texto))
    print("\n".join(texto))
    figuras(por_fold_todos, a.resultados)


def figuras(por_fold_todos, pasta):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    FUNDO, TINTA, TINTA2, MUDO, GRADE, EIXO = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
    for alvo, por_fold in por_fold_todos.items():
        itens = [(g, r, m) for g, r, m in MODELOS if m in por_fold]
        fig, ax = plt.subplots(figsize=(13, 5.4), facecolor=FUNDO)
        ax.set_facecolor(FUNDO)
        for lado in ("top", "right", "left"):
            ax.spines[lado].set_visible(False)
        ax.spines["bottom"].set_color(EIXO)
        ax.grid(axis="y", color=GRADE, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.tick_params(colors=MUDO, length=0, labelsize=8.5)
        for i, (g, r, m) in enumerate(itens):
            ax.boxplot([por_fold[m]], positions=[i], widths=0.62, patch_artist=True,
                       flierprops=dict(marker="o", markersize=3, markerfacecolor=COR[g], markeredgecolor="none", alpha=0.7),
                       medianprops=dict(color=TINTA, linewidth=1.4),
                       whiskerprops=dict(color=TINTA2, linewidth=0.9), capprops=dict(color=TINTA2, linewidth=0.9),
                       boxprops=dict(facecolor=COR[g], edgecolor="none", alpha=0.9))
        # referencia: mediana da regra de cultivar e momento
        ref = np.median(por_fold["media_cultivar_momento"])
        ax.axhline(ref, color=TINTA2, linewidth=0.9, linestyle=(0, (4, 3)), zorder=1)
        ax.set_xticks(range(len(itens)))
        ax.set_xticklabels([r for _, r, _ in itens])
        for t in ax.get_xticklabels():
            t.set_color(TINTA2)
        ax.set_ylim(bottom=0)
        ax.set_title("%s: RMSE por fold (%s), 10 repetições x 5 folds por parcela" % (alvo.capitalize(), UNIDADE[alvo]),
                     loc="left", fontsize=12, color=TINTA)
        grupos = [g for g in ("ref", "manual", "ft", "cong", "ctx") if any(x[0] == g for x in itens)]
        fig.legend(handles=[Patch(color=COR[g], label=NOME_GRUPO[g]) for g in grupos],
                   loc="lower center", ncol=3, frameon=False, fontsize=9, labelcolor=TINTA2)
        fig.text(0.99, 0.135, "linha tracejada: mediana da regra por cultivar e momento", ha="right",
                 fontsize=8, color=MUDO)
        fig.tight_layout(rect=(0, 0.13, 1, 1))
        fig.savefig(os.path.join(pasta, "resultados_finais_boxplot_%s.png" % alvo), dpi=160, facecolor=FUNDO)
        plt.close(fig)


if __name__ == "__main__":
    main()
