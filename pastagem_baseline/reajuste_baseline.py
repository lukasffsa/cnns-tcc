#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Reajuste da metodologia de Franco et al. (2022) nos dados do Lukas.

O app Pastagem traz modelos treinados em outro conjunto de dados, que nao
funcionam aqui. Este script refaz o METODO do artigo com os dados novos:

  atributos manuais por imagem (os mesmos do app/artigo)
    -> LASSO, SVR e MLP
    -> hiperparametros e selecao de atributos por Evolucao Diferencial (DE)
    -> validacao cruzada com 5 subconjuntos, varias repeticoes

Tres cenarios de validacao:

  publicado_imagem   DE olha os mesmos 5 folds em que o resultado e reportado,
                     folds sorteados por imagem (como o artigo parece ter feito)
  publicado_parcela  idem, mas todas as fotos de uma parcela ficam no mesmo fold
  aninhado_parcela   folds por parcela e DE rodando so dentro do treino de cada
                     fold. E a estimativa honesta, a que vale para comparar

Uso:
    python reajuste_baseline.py resultados/baseline_python.csv dataset/metadata.csv

Dependencias: numpy, scipy, scikit-learn
"""
import argparse
import csv
import json
import os
import re
import sys
import time
import warnings

import numpy as np
from scipy.optimize import differential_evolution
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Atributos do artigo (Tabela 2). A altura estimada so entra na biomassa.
ATRIBUTOS = ["gli_medio", "mpri_medio", "vari_medio", "tgi_medio", "homogeneidade",
             "entropia", "segundo_momento_angular", "cobertura_verde_pct"]
# Altura estimada: regressao linear com as mesmas variaveis da formula do app
ATRIBUTOS_ALTURA_LINEAR = ["entropia", "homogeneidade", "desvio_padrao_cinza", "area_escura_pct"]

MODELOS = ("lasso", "svr", "mlp")
ALVOS = ("altura", "biomassa")
CENARIOS = ("publicado_imagem", "publicado_parcela", "aninhado_parcela")
REFERENCIAS = ("media_treino", "media_cultivar_momento")
K = 5
PADRAO_NOME = re.compile(r"^([A-Za-z]{2})(\d{2})[Rr0](\d)_M(\d)")


# --------------------------------------------------------------------------
# Dados
# --------------------------------------------------------------------------
def carregar(caminho_atributos, caminho_metadata):
    with open(caminho_metadata, encoding="utf-8-sig") as f:
        meta = {l["ALTURA"].strip().upper(): l for l in csv.DictReader(f, delimiter=";")}
    with open(caminho_atributos, encoding="utf-8") as f:
        linhas = list(csv.DictReader(f, delimiter=";"))
    dados = []
    for l in linhas:
        m = PADRAO_NOME.match(l["arquivo"])
        if not m:
            continue
        cultivar, numero, rep, momento = m.groups()
        parcela = cultivar.upper() + numero
        chave = "%sM%s" % (parcela, momento)
        if chave not in meta:
            continue
        ms = (meta[chave].get("MS (kg/ha)") or "").strip()
        d = {"arquivo": l["arquivo"], "parcela": parcela, "cultivar": cultivar.upper(),
             "momento": "M" + momento,
             "altura": float(meta[chave]["Média"]),
             "biomassa": float(ms) if ms else float("nan")}
        for c in set(ATRIBUTOS + ATRIBUTOS_ALTURA_LINEAR):
            d[c] = float(l[c])
        dados.append(d)
    return dados


def folds_por_parcela(dados, semente):
    """5 folds com todas as fotos da parcela juntas, equilibrando as cultivares."""
    rng = np.random.RandomState(semente)
    fold_da_parcela = {}
    for cultivar in sorted({d["cultivar"] for d in dados}):
        parcelas = sorted({d["parcela"] for d in dados if d["cultivar"] == cultivar})
        rng.shuffle(parcelas)
        for i, p in enumerate(parcelas):
            fold_da_parcela[p] = i % K
    return np.array([fold_da_parcela[d["parcela"]] for d in dados])


def folds_por_imagem(dados, semente):
    rng = np.random.RandomState(semente)
    ordem = rng.permutation(len(dados))
    folds = np.empty(len(dados), dtype=int)
    folds[ordem] = np.arange(len(dados)) % K
    return folds


# --------------------------------------------------------------------------
# Preparo de cada divisao treino/teste
# --------------------------------------------------------------------------
def preparar_divisoes(X, XL, y, y_altura, folds, com_altura):
    """Para cada fold: matriz padronizada pelo treino e alvo padronizado.

    Quando com_altura=True (biomassa), a altura estimada entra como nona
    coluna. Ela vem de uma regressao linear ajustada so com o treino.
    """
    divisoes = []
    for k in sorted(set(folds)):
        tr, te = np.where(folds != k)[0], np.where(folds == k)[0]
        Xtr, Xte = X[tr], X[te]
        if com_altura:
            rl = LinearRegression().fit(XL[tr], y_altura[tr])
            Xtr = np.column_stack([Xtr, rl.predict(XL[tr])])
            Xte = np.column_stack([Xte, rl.predict(XL[te])])
        mu, sd = Xtr.mean(0), Xtr.std(0)
        sd[sd == 0] = 1.0
        y_mu, y_sd = y[tr].mean(), y[tr].std()
        divisoes.append({"te": te, "Xtr": (Xtr - mu) / sd, "Xte": (Xte - mu) / sd,
                         "ytr": (y[tr] - y_mu) / y_sd, "y_mu": y_mu, "y_sd": y_sd})
    return divisoes


# --------------------------------------------------------------------------
# Modelos e codificacao do individuo da DE (faixas da Tabela 4 do artigo)
# --------------------------------------------------------------------------
N_GENES_MODELO = {"lasso": 1, "svr": 4, "mlp": 2}


def decodificar(modelo, vetor, n_atr):
    mascara = tuple(bool(v >= 0.5) for v in vetor[:n_atr])
    g = vetor[n_atr:]
    if modelo == "lasso":
        # mesma faixa do artigo, ate 20, mas percorrida em escala logaritmica:
        # so alphas pequenos geram modelo nao constante com o alvo padronizado
        cfg = {"alpha": float("%.3g" % (10 ** (-4 + g[0] * (np.log10(20.0) + 4))))}
    elif modelo == "svr":
        cfg = {"C": round(1.0 + 99.0 * g[0], 3),
               "kernel": "rbf" if g[1] < 0.5 else "linear",
               "gamma": round(0.001 + 0.999 * g[2], 4),
               "epsilon": round(float(g[3]), 4)}
    else:
        cfg = {"camadas": 1 if g[0] < 0.5 else 2,
               "neuronios": int(min(50, 1 + int(g[1] * 50)))}
    return mascara, cfg


def criar_modelo(modelo, cfg, semente):
    if modelo == "lasso":
        return Lasso(alpha=cfg["alpha"], max_iter=5000)
    if modelo == "svr":
        return SVR(C=cfg["C"], kernel=cfg["kernel"], gamma=cfg["gamma"], epsilon=cfg["epsilon"])
    return MLPRegressor(hidden_layer_sizes=(cfg["neuronios"],) * cfg["camadas"],
                        activation="relu", solver="lbfgs", max_iter=200,
                        random_state=semente)


def prever_fora_do_fold(modelo, cfg, mascara, divisoes, n, semente):
    cols = np.where(mascara)[0]
    pred = np.full(n, np.nan)
    for d in divisoes:
        m = criar_modelo(modelo, cfg, semente).fit(d["Xtr"][:, cols], d["ytr"])
        pred[d["te"]] = m.predict(d["Xte"][:, cols]) * d["y_sd"] + d["y_mu"]
    return pred


def buscar_de(modelo, divisoes, y, n_atr, semente, np_pop, geracoes):
    """Evolucao Diferencial classica (rand/1/bin) minimizando o RMSE da validacao."""
    cache = {}
    idx = np.concatenate([d["te"] for d in divisoes])

    def custo(vetor):
        mascara, cfg = decodificar(modelo, vetor, n_atr)
        if not any(mascara):
            return 1e9
        chave = (mascara, tuple(sorted(cfg.items())))
        if chave not in cache:
            pred = prever_fora_do_fold(modelo, cfg, mascara, divisoes, len(y), semente)
            cache[chave] = float(np.sqrt(np.mean((pred[idx] - y[idx]) ** 2)))
        return cache[chave]

    if modelo == "lasso":
        # o ajuste e barato: a busca do LASSO recebe um orcamento maior
        np_pop, geracoes = 2 * np_pop, 3 * geracoes
    dim = n_atr + N_GENES_MODELO[modelo]
    rng = np.random.RandomState(semente)
    res = differential_evolution(custo, [(0.0, 1.0)] * dim, strategy="rand1bin",
                                 init=rng.rand(np_pop, dim), maxiter=geracoes,
                                 mutation=0.8, recombination=0.9, tol=0.0,
                                 polish=False, seed=semente, updating="immediate")
    return decodificar(modelo, res.x, n_atr)


# --------------------------------------------------------------------------
# Metricas (as mesmas do artigo)
# --------------------------------------------------------------------------
def metricas(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    sq = np.sum((y - p) ** 2)
    st = np.sum((y - y.mean()) ** 2)
    wi = np.sum((np.abs(p - y.mean()) + np.abs(y - y.mean())) ** 2)
    return {"R2": float(1 - sq / st) if st > 0 else float("nan"),
            "WI": float(1 - sq / wi) if wi > 0 else float("nan"),
            "RMSE": float(np.sqrt(sq / len(y))),
            "MAE": float(np.mean(np.abs(y - p))),
            "MAPE": float(100 * np.mean(np.abs((p - y) / y)))}


# --------------------------------------------------------------------------
# Uma tarefa = cenario x alvo x modelo x repeticao
# --------------------------------------------------------------------------
def rodar_tarefa(dados, cenario, alvo, modelo, rep, np_pop, geracoes):
    t0 = time.time()
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    validos = np.array([not np.isnan(d[alvo]) for d in dados])
    sub = [d for d, v in zip(dados, validos) if v]
    folds_todos = (folds_por_imagem if cenario.endswith("imagem") else folds_por_parcela)(dados, rep)
    folds = folds_todos[validos]
    X = np.array([[d[c] for c in ATRIBUTOS] for d in sub])
    XL = np.array([[d[c] for c in ATRIBUTOS_ALTURA_LINEAR] for d in sub])
    y = np.array([d[alvo] for d in sub])
    y_altura = np.array([d["altura"] for d in sub])
    com_altura = alvo == "biomassa"
    n_atr = len(ATRIBUTOS) + (1 if com_altura else 0)
    nomes = ATRIBUTOS + (["altura_estimada"] if com_altura else [])
    escolhas = []

    if modelo in REFERENCIAS:
        pred = np.full(len(y), np.nan)
        for k in range(K):
            tr, te = np.where(folds != k)[0], np.where(folds == k)[0]
            for i in te:
                if modelo == "media_treino":
                    iguais = tr
                else:
                    iguais = [j for j in tr if sub[j]["cultivar"] == sub[i]["cultivar"]
                              and (com_altura or sub[j]["momento"] == sub[i]["momento"])]
                pred[i] = y[iguais].mean()
    elif cenario.startswith("publicado"):
        divisoes = preparar_divisoes(X, XL, y, y_altura, folds, com_altura)
        mascara, cfg = buscar_de(modelo, divisoes, y, n_atr, rep, np_pop, geracoes)
        pred = prever_fora_do_fold(modelo, cfg, mascara, divisoes, len(y), rep)
        escolhas.append({"fold": "todos", "atributos": [n for n, m in zip(nomes, mascara) if m], **cfg})
    else:  # aninhado: a DE so enxerga o treino de cada fold externo
        pred = np.full(len(y), np.nan)
        parcelas = np.array([d["parcela"] for d in sub])
        for k in range(K):
            tr, te = np.where(folds != k)[0], np.where(folds == k)[0]
            # folds internos por parcela, dentro do treino
            rng = np.random.RandomState(1000 * rep + k)
            unicas = np.array(sorted(set(parcelas[tr])))
            rng.shuffle(unicas)
            interno = {p: i % (K - 1) for i, p in enumerate(unicas)}
            folds_int = np.array([interno[p] for p in parcelas[tr]])
            div_int = preparar_divisoes(X[tr], XL[tr], y[tr], y_altura[tr], folds_int, com_altura)
            mascara, cfg = buscar_de(modelo, div_int, y[tr], n_atr, 1000 * rep + k, np_pop, geracoes)
            # ajusta no treino inteiro e preve o fold de teste
            ext = np.ones(len(y), dtype=int)
            ext[te] = 0  # 0 = teste
            div_ext = preparar_divisoes(X, XL, y, y_altura, ext, com_altura)[0]
            pred[te] = prever_fora_do_fold(modelo, cfg, mascara, [div_ext], len(y), rep)[te]
            escolhas.append({"fold": k, "atributos": [n for n, m in zip(nomes, mascara) if m], **cfg})

    por_fold = [dict(fold=int(k), **metricas(y[folds == k], pred[folds == k])) for k in range(K)]
    return {"cenario": cenario, "alvo": alvo, "modelo": modelo, "rep": rep,
            "global": metricas(y, pred), "por_fold": por_fold, "escolhas": escolhas,
            "arquivos": [d["arquivo"] for d in sub], "folds": folds.tolist(),
            "pred": [round(float(v), 4) for v in pred], "segundos": round(time.time() - t0, 1)}


def rodar_com_cache(args_tarefa, pasta_cache, dados, np_pop, geracoes):
    cenario, alvo, modelo, rep = args_tarefa
    arq = os.path.join(pasta_cache, "%s__%s__%s__%02d.json" % (cenario, alvo, modelo, rep))
    if os.path.exists(arq):
        with open(arq, encoding="utf-8") as f:
            return json.load(f)
    r = rodar_tarefa(dados, cenario, alvo, modelo, rep, np_pop, geracoes)
    with open(arq + ".tmp", "w", encoding="utf-8") as f:
        json.dump(r, f)
    os.replace(arq + ".tmp", arq)
    return r


# --------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("atributos", help="CSV gerado por pastagem_baseline.py")
    ap.add_argument("metadata", help="metadata.csv do dataset")
    ap.add_argument("-o", "--saida", default="resultados", help="pasta de saida")
    ap.add_argument("--reps", type=int, default=30, help="repeticoes do cenario publicado_parcela")
    ap.add_argument("--reps-imagem", type=int, default=10, help="repeticoes do cenario publicado_imagem")
    ap.add_argument("--reps-aninhado", type=int, default=10, help="repeticoes do cenario aninhado")
    ap.add_argument("--np", type=int, default=15, dest="np_pop", help="tamanho da populacao da DE")
    ap.add_argument("--geracoes", type=int, default=12, help="geracoes da DE")
    ap.add_argument("--modelos", default=",".join(MODELOS))
    ap.add_argument("--cenarios", default=",".join(CENARIOS))
    ap.add_argument("--jobs", type=int, default=2, help="processos em paralelo")
    ap.add_argument("--so-folds", action="store_true", help="so grava o arquivo de folds e sai")
    ap.add_argument("--so-relatorio", action="store_true", help="so monta as tabelas com o que ja foi calculado")
    args = ap.parse_args(argv)

    dados = carregar(args.atributos, args.metadata)
    os.makedirs(args.saida, exist_ok=True)

    # Arquivo de folds para a CNN usar exatamente as mesmas divisoes
    n_reps = max(args.reps, args.reps_aninhado, args.reps_imagem)
    with open(os.path.join(args.saida, "folds_por_parcela.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow(["arquivo", "parcela", "cultivar", "momento"] + ["rep%02d" % r for r in range(n_reps)])
        colunas = [folds_por_parcela(dados, r) for r in range(n_reps)]
        for i, d in enumerate(dados):
            w.writerow([d["arquivo"], d["parcela"], d["cultivar"], d["momento"]] + [int(c[i]) for c in colunas])
    if args.so_folds:
        return

    pasta_cache = os.path.join(args.saida, "_cache_reajuste")
    os.makedirs(pasta_cache, exist_ok=True)
    tarefas = []
    for cenario in args.cenarios.split(","):
        reps = {"aninhado_parcela": args.reps_aninhado, "publicado_imagem": args.reps_imagem}.get(cenario, args.reps)
        for alvo in ALVOS:
            modelos = list(args.modelos.split(","))
            if cenario.startswith("publicado"):
                modelos += list(REFERENCIAS)
            for modelo in modelos:
                for rep in range(reps):
                    tarefas.append((cenario, alvo, modelo, rep))
    # tarefas rapidas primeiro
    tarefas.sort(key=lambda t: (t[2] == "mlp", t[0].startswith("aninhado"), t[3]))

    if args.so_relatorio:
        resultados = []
        for t in tarefas:
            arq = os.path.join(pasta_cache, "%s__%s__%s__%02d.json" % t)
            if os.path.exists(arq):
                with open(arq, encoding="utf-8") as f:
                    resultados.append(json.load(f))
        gravar_saidas(resultados, args.saida)
        return

    from joblib import Parallel, delayed
    resultados = Parallel(n_jobs=args.jobs, verbose=5)(
        delayed(rodar_com_cache)(t, pasta_cache, dados, args.np_pop, args.geracoes) for t in tarefas)

    gravar_saidas(resultados, args.saida)


def gravar_saidas(resultados, pasta):
    nomes_m = ["R2", "WI", "RMSE", "MAE", "MAPE"]
    with open(os.path.join(pasta, "reajuste_metricas_por_repeticao.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow(["cenario", "alvo", "modelo", "rep"] + nomes_m)
        for r in resultados:
            w.writerow([r["cenario"], r["alvo"], r["modelo"], r["rep"]] + [repr(r["global"][m]) for m in nomes_m])
    with open(os.path.join(pasta, "reajuste_metricas_por_fold.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow(["cenario", "alvo", "modelo", "rep", "fold"] + nomes_m)
        for r in resultados:
            for pf in r["por_fold"]:
                w.writerow([r["cenario"], r["alvo"], r["modelo"], r["rep"], pf["fold"]] + [repr(pf[m]) for m in nomes_m])
    with open(os.path.join(pasta, "reajuste_configuracoes.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow(["cenario", "alvo", "modelo", "rep", "fold", "atributos", "hiperparametros"])
        for r in resultados:
            for e in r["escolhas"]:
                hp = {k: v for k, v in e.items() if k not in ("fold", "atributos")}
                w.writerow([r["cenario"], r["alvo"], r["modelo"], r["rep"], e["fold"],
                            ",".join(e["atributos"]), json.dumps(hp)])
    # predicoes fora do fold do cenario aninhado, no formato esperado da CNN
    with open(os.path.join(pasta, "reajuste_predicoes_aninhado.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow(["arquivo", "rep", "fold", "alvo", "modelo", "previsto"])
        for r in resultados:
            if r["cenario"] == "aninhado_parcela":
                for a, k, p in zip(r["arquivos"], r["folds"], r["pred"]):
                    w.writerow([a, r["rep"], k, r["alvo"], r["modelo"], p])

    linhas = []
    for cenario in CENARIOS:
        for alvo in ALVOS:
            linhas.append("%s | %s" % (cenario, alvo))
            for modelo in MODELOS + REFERENCIAS:
                rs = [r for r in resultados if (r["cenario"], r["alvo"], r["modelo"]) == (cenario, alvo, modelo)]
                if not rs:
                    continue
                txt = "  %-24s n_rep=%2d" % (modelo, len(rs))
                for m in nomes_m:
                    v = np.array([r["global"][m] for r in rs])
                    txt += "  %s=%9.3f (%.3f)" % (m, v.mean(), v.std())
                linhas.append(txt)
            linhas.append("")
    texto = "\n".join(linhas)
    with open(os.path.join(pasta, "reajuste_resumo.txt"), "w", encoding="utf-8") as f:
        f.write("Media (desvio padrao) entre repeticoes, metricas sobre todas as imagens fora do fold\n\n" + texto)
    print(texto)


if __name__ == "__main__":
    main()
