#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compara as predicoes da CNN com o baseline reajustado, nos mesmos folds.

Uso:
    python comparar_com_cnn.py predicoes_cnn.csv dataset/metadata.csv -o resultados

O arquivo da CNN deve ser um CSV com ponto e virgula e as colunas
    arquivo;rep;fold;alvo;previsto
onde alvo e "altura" ou "biomassa", e rep/fold seguem o folds_por_parcela.csv
(rep 0 = coluna rep00). Cada imagem deve ser prevista pelo modelo treinado SEM
o fold dela.

Le de <saida>/: folds_por_parcela.csv, reajuste_predicoes_aninhado.csv e
baseline_python.csv. Grava comparacao_resumo.txt, comparacao_por_fold.csv e
comparacao_boxplot_altura.png e comparacao_boxplot_biomassa.png.

Dependencias: numpy, matplotlib (e as do reajuste_baseline.py, que e importado)
"""
import argparse
import csv
import os
import sys
from collections import defaultdict
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import reajuste_baseline as rb  # noqa: E402

CNN = "MobileNet V3-Large"


NOMES = {"cnn": CNN, "lasso": "LASSO", "svr": "SVR", "mlp": "MLP",
         "media_treino": "Média do treino", "media_cultivar_momento": "Média por cultivar e momento"}
ORDEM = ["cnn", "lasso", "svr", "mlp", "media_treino", "media_cultivar_momento"]
UNIDADE = {"altura": "cm", "biomassa": "kg/ha"}


def ler(caminho):
    with open(caminho, encoding="utf-8-sig") as f:
        return list(csv.DictReader(f, delimiter=";"))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("predicoes_cnn")
    ap.add_argument("metadata")
    ap.add_argument("-o", "--saida", default="resultados")
    args = ap.parse_args(argv)

    dados = rb.carregar(os.path.join(args.saida, "baseline_python.csv"), args.metadata)
    info = {d["arquivo"]: d for d in dados}
    folds = {l["arquivo"]: l for l in ler(os.path.join(args.saida, "folds_por_parcela.csv"))}

    # pred[(alvo, modelo, rep)][arquivo] = previsto
    pred = defaultdict(dict)
    avisos = []
    fold_errado = 0
    for l in ler(args.predicoes_cnn):
        arq, rep, alvo = l["arquivo"].strip(), int(l["rep"]), l["alvo"].strip().lower()
        if arq not in info:
            avisos.append("Imagem desconhecida no arquivo da CNN: %s" % arq)
            continue
        if folds[arq]["rep%02d" % rep] != str(int(l["fold"])):
            fold_errado += 1
        pred[(alvo, "cnn", rep)][arq] = float(l["previsto"].replace(",", "."))
    if fold_errado:
        avisos.append("ATENCAO: %d linhas da CNN com fold diferente do folds_por_parcela.csv. "
                      "Se os folds nao forem os mesmos, a comparacao nao vale." % fold_errado)
    for l in ler(os.path.join(args.saida, "reajuste_predicoes_aninhado.csv")):
        pred[(l["alvo"], l["modelo"], int(l["rep"]))][l["arquivo"]] = float(l["previsto"])

    linhas_fold, texto = [], []
    for alvo in ("altura", "biomassa"):
        validas = [d for d in dados if not np.isnan(d[alvo])]
        reps_cnn = sorted(r for (a, m, r) in pred if a == alvo and m == "cnn")
        reps_base = {r for (a, m, r) in pred if a == alvo and m == "lasso"}
        reps = [r for r in reps_cnn if r in reps_base]
        fora = [r for r in reps_cnn if r not in reps_base]
        if fora:
            avisos.append("%s: repeticoes %s da CNN nao tem baseline aninhado (o baseline cobre rep00 a rep%02d)."
                          % (alvo, fora, max(reps_base)))
        if not reps:
            texto.append("%s: nenhuma repeticao da CNN para comparar.\n" % alvo)
            continue

        globais = defaultdict(list)   # modelo -> metricas globais por rep
        por_fold = defaultdict(list)  # modelo -> RMSE por fold (na mesma ordem para todos)
        for rep in reps:
            k_de = {d["arquivo"]: int(folds[d["arquivo"]]["rep%02d" % rep]) for d in validas}
            faltam = [d["arquivo"] for d in validas if d["arquivo"] not in pred[(alvo, "cnn", rep)]]
            if faltam:
                avisos.append("%s rep%02d: a CNN nao previu %d imagens (ex.: %s). Repeticao ignorada."
                              % (alvo, rep, len(faltam), faltam[0]))
                continue
            # regras de referencia nos mesmos folds
            for d in validas:
                treino = [o for o in validas if k_de[o["arquivo"]] != k_de[d["arquivo"]]]
                pred[(alvo, "media_treino", rep)][d["arquivo"]] = float(np.mean([o[alvo] for o in treino]))
                iguais = [o for o in treino if o["cultivar"] == d["cultivar"]
                          and (alvo == "biomassa" or o["momento"] == d["momento"])]
                pred[(alvo, "media_cultivar_momento", rep)][d["arquivo"]] = float(np.mean([o[alvo] for o in iguais]))
            y = np.array([d[alvo] for d in validas])
            k = np.array([k_de[d["arquivo"]] for d in validas])
            for m in ORDEM:
                p = np.array([pred[(alvo, m, rep)][d["arquivo"]] for d in validas])
                globais[m].append(rb.metricas(y, p))
                for f in range(rb.K):
                    mf = rb.metricas(y[k == f], p[k == f])
                    por_fold[m].append(mf["RMSE"])
                    linhas_fold.append([alvo, m, rep, f] + [repr(mf[c]) for c in ("R2", "WI", "RMSE", "MAE", "MAPE")])

        n = len(globais["cnn"])
        if n == 0:
            texto.append("%s: nenhuma repeticao completa da CNN para comparar.\n" % alvo)
            continue
        texto.append("%s (%s), %d repeticao(oes), folds por parcela" % (alvo.upper(), UNIDADE[alvo], n))
        texto.append("  metricas sobre todas as imagens fora do fold: media (desvio padrao) entre repeticoes")
        for m in ORDEM:
            t = "  %-30s" % NOMES[m]
            for c in ("R2", "WI", "RMSE", "MAE", "MAPE"):
                v = np.array([g[c] for g in globais[m]])
                t += "  %s=%8.3f (%.3f)" % (c, v.mean(), v.std())
            texto.append(t)
        texto.append("  RMSE por fold da CNN menos o de cada alternativa (negativo = CNN melhor), %d folds" % len(por_fold["cnn"]))
        cnn = np.array(por_fold["cnn"])
        for m in ORDEM[1:]:
            dif = cnn - np.array(por_fold[m])
            texto.append("  %-30s  media=%8.3f  mediana=%8.3f  CNN melhor em %d de %d folds"
                         % ("vs " + NOMES[m], dif.mean(), np.median(dif), int((dif < 0).sum()), len(dif)))
        texto.append("")
        figura(alvo, por_fold, n, args.saida)

    if avisos:
        texto = ["AVISOS"] + ["  " + a for a in avisos] + [""] + texto
    texto.append("Obs.: os folds de repeticoes diferentes reaproveitam as mesmas parcelas, entao nao sao\n"
                 "amostras independentes. A contagem de folds serve como descricao, nao como teste estatistico.")
    saida_txt = "\n".join(texto)
    print(saida_txt)
    with open(os.path.join(args.saida, "comparacao_resumo.txt"), "w", encoding="utf-8") as f:
        f.write(saida_txt + "\n")
    with open(os.path.join(args.saida, "comparacao_por_fold.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow(["alvo", "modelo", "rep", "fold", "R2", "WI", "RMSE", "MAE", "MAPE"])
        w.writerows(linhas_fold)


def figuras(por_fold_todos, pasta):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    FUNDO, TINTA, TINTA2, MUDO, GRADE, EIXO = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
    
    for alvo, por_fold in por_fold_todos.items():
        itens = [(g, r, m) for g, r, m in MODELOS if m in por_fold]
        
        # Reduzimos a largura física, mas aumentamos a altura relativa
        fig, ax = plt.subplots(figsize=(10, 7), facecolor=FUNDO)
        ax.set_facecolor(FUNDO)
        
        for lado in ("top", "right", "left"):
            ax.spines[lado].set_visible(False)
        ax.spines["bottom"].set_color(EIXO)
        ax.grid(axis="y", color=GRADE, linewidth=0.8)
        ax.set_axisbelow(True)
        
        # AUMENTAMOS O TAMANHO DA FONTE DOS EIXOS AQUI (de 8.5 para 12)
        ax.tick_params(colors=MUDO, length=0, labelsize=12)
        
        for i, (g, r, m) in enumerate(itens):
            ax.boxplot([por_fold[m]], positions=[i], widths=0.62, patch_artist=True,
                       flierprops=dict(marker="o", markersize=4, markerfacecolor=COR[g], markeredgecolor="none", alpha=0.7),
                       medianprops=dict(color=TINTA, linewidth=1.4),
                       whiskerprops=dict(color=TINTA2, linewidth=0.9), capprops=dict(color=TINTA2, linewidth=0.9),
                       boxprops=dict(facecolor=COR[g], edgecolor="none", alpha=0.9))
        
        ref = np.median(por_fold["media_cultivar_momento"])
        ax.axhline(ref, color=TINTA2, linewidth=1.2, linestyle=(0, (4, 3)), zorder=1)
        ax.set_xticks(range(len(itens)))
        
        # Rótulos do eixo X (mantidos em 45 graus)
        ax.set_xticklabels([r for _, r, _ in itens], rotation=45, ha="right", rotation_mode="anchor")
        for t in ax.get_xticklabels():
            t.set_color(TINTA2)
            t.set_fontsize(11) # Fonte dos modelos bem visível
            
        ax.set_ylim(bottom=0)
        
        # Título bem maior
        ax.set_title("%s: RMSE por fold (%s)" % (alvo.capitalize(), UNIDADE[alvo]),
                     loc="left", fontsize=15, color=TINTA, pad=15)
        
        grupos = [g for g in ("ref", "manual", "ft", "cong", "ctx") if any(x[0] == g for x in itens)]
        
        # Legenda com fonte maior e movida um pouco mais para baixo
        fig.legend(handles=[Patch(color=COR[g], label=NOME_GRUPO[g]) for g in grupos],
                   loc="lower center", ncol=2, frameon=False, fontsize=11, labelcolor=TINTA2, 
                   bbox_to_anchor=(0.5, 0.01))
        
        # Ajuste de margens: deixa muito espaço embaixo (0.35) para os textos inclinados e legenda
        fig.tight_layout(rect=(0, 0.25, 1, 1))
        
        fig.savefig(os.path.join(pasta, "resultados_finais_boxplot_%s.pdf" % alvo), format="pdf", dpi=300, facecolor=FUNDO)
        plt.close(fig)


if __name__ == "__main__":
    main()
