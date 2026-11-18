#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CNN pre-treinada como extrator congelado + modelos lineares, no protocolo do
baseline (folds por parcela, selecao so no treino, mesmas repeticoes).

Experimentos (cada um gera predicoes no formato de comparar_com_cnn.py):

  emb          LASSO / SVR sobre o embedding da CNN (PCA ajustado no treino)
  emb+manual   embedding + os 9 atributos manuais do baseline
  manual+ctx   atributos manuais + cultivar e momento (one-hot)
  emb+ctx      embedding + cultivar e momento

Passo 1 (uma vez, lento na CPU): extrair os embeddings
    python embeddings_cnn.py extrair dataset/images -o resultados/embeddings_resnet50.npz
Passo 2: rodar os experimentos
    python embeddings_cnn.py avaliar resultados/embeddings_resnet50.npz resultados/baseline_python.csv dataset/metadata.csv -o resultados

Dependencias: torch, torchvision (passo 1); numpy, scikit-learn (passo 2)
"""
import argparse
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import reajuste_baseline as rb  # noqa: E402


# --------------------------------------------------------------------------
def extrair(pasta, saida, pesos_h5=None, lado=448):
    """Embedding de 2048 dimensoes por imagem: imagem inteira (lado menor = `lado`),
    media global, e media com o flip horizontal para reduzir variancia."""
    import torch
    from PIL import Image
    torch.set_num_threads(max(1, os.cpu_count() or 1))
    if pesos_h5:
        import resnet50_keras as rk
        net = rk.carregar(pesos_h5)
        prep = rk.preparar
    else:
        import torchvision
        from torchvision import transforms
        net = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2)
        net.fc = torch.nn.Identity()
        net.eval()
        norm = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        prep = lambda a: norm(Image.fromarray(a)).unsqueeze(0)
    arquivos = sorted(a for a in os.listdir(pasta) if a.lower().endswith((".jpg", ".jpeg", ".png")))
    feats = []
    with torch.no_grad():
        for i, a in enumerate(arquivos, 1):
            im = Image.open(os.path.join(pasta, a)).convert("RGB")
            w, h = im.size
            s = lado / min(w, h)
            im = im.resize((int(round(w * s)), int(round(h * s))), Image.BILINEAR)
            x = prep(np.asarray(im))
            f = (net(x) + net(torch.flip(x, dims=[3]))) / 2
            feats.append(f[0].numpy())
            if i % 20 == 0 or i == len(arquivos):
                print("%d/%d" % (i, len(arquivos)), flush=True)
    np.savez_compressed(saida, arquivos=np.array(arquivos), X=np.vstack(feats).astype(np.float32))
    print("gravado", saida)


# --------------------------------------------------------------------------
def montar(dados, emb, usar_emb, usar_manual, usar_ctx, alvo):
    """Blocos de atributos por imagem. O PCA do embedding e ajustado por fold."""
    blocos = []
    if usar_emb:
        blocos.append(("emb", np.array([emb[d["arquivo"]] for d in dados])))
    if usar_manual:
        X = np.array([[d[c] for c in rb.ATRIBUTOS] for d in dados])
        blocos.append(("manual", X))
    if usar_ctx:
        X = np.array([[d["cultivar"] == "TR", d["momento"] == "M2"] for d in dados], dtype=float)
        blocos.append(("ctx", X))
    return blocos


def ajustar_prever(blocos, XL, y_altura, y, tr, te, modelo, alvo, n_pca, semente):
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LinearRegression

    partes_tr, partes_te = [], []
    for nome, X in blocos:
        Xtr, Xte = X[tr], X[te]
        if nome == "emb":
            pca = PCA(n_components=min(n_pca, len(tr) - 1), random_state=semente).fit(Xtr)
            Xtr, Xte = pca.transform(Xtr), pca.transform(Xte)
        if nome == "manual" and alvo == "biomassa":
            rl = LinearRegression().fit(XL[tr], y_altura[tr])
            Xtr = np.column_stack([Xtr, rl.predict(XL[tr])])
            Xte = np.column_stack([Xte, rl.predict(XL[te])])
        partes_tr.append(Xtr)
        partes_te.append(Xte)
    Xtr, Xte = np.hstack(partes_tr), np.hstack(partes_te)
    mu, sd = Xtr.mean(0), Xtr.std(0)
    sd[sd == 0] = 1
    Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
    ymu, ysd = y[tr].mean(), y[tr].std()
    ytr = (y[tr] - ymu) / ysd
    return Xtr, Xte, ytr, ymu, ysd


def avaliar(emb_npz, atributos, metadata, saida, reps, n_pca):
    from sklearn.linear_model import LassoCV
    from sklearn.model_selection import GroupKFold, GridSearchCV
    from sklearn.svm import SVR
    import warnings
    warnings.filterwarnings("ignore")

    z = np.load(emb_npz, allow_pickle=True)
    emb = dict(zip(z["arquivos"].tolist(), z["X"]))
    dados = rb.carregar(atributos, metadata)
    with open(os.path.join(saida, "folds_por_parcela.csv"), encoding="utf-8") as f:
        folds_arq = {l["arquivo"]: l for l in csv.DictReader(f, delimiter=";")}

    experimentos = {
        "emb": (True, False, False),
        "emb+manual": (True, True, False),
        "manual+ctx": (False, True, True),
        "emb+ctx": (True, False, True),
        "emb+manual+ctx": (True, True, True),
    }
    linhas = []
    resumo = []
    for alvo in ("altura", "biomassa"):
        sub = [d for d in dados if not np.isnan(d[alvo])]
        y = np.array([d[alvo] for d in sub])
        y_alt = np.array([d["altura"] for d in sub])
        XL = np.array([[d[c] for c in rb.ATRIBUTOS_ALTURA_LINEAR] for d in sub])
        grupos = np.array([d["parcela"] for d in sub])
        for nome, (ue, um, uc) in experimentos.items():
            blocos = montar(sub, emb, ue, um, uc, alvo)
            for regressor in ("lasso", "svr"):
                rmses = []
                for rep in range(reps):
                    k = np.array([int(folds_arq[d["arquivo"]]["rep%02d" % rep]) for d in sub])
                    pred = np.full(len(y), np.nan)
                    for f in range(rb.K):
                        tr, te = np.where(k != f)[0], np.where(k == f)[0]
                        Xtr, Xte, ytr, ymu, ysd = ajustar_prever(blocos, XL, y_alt, y, tr, te, regressor, alvo, n_pca, rep)
                        cv_int = list(GroupKFold(4).split(Xtr, ytr, grupos[tr]))
                        if regressor == "lasso":
                            m = LassoCV(cv=cv_int, alphas=30, max_iter=3000, n_jobs=1, random_state=rep).fit(Xtr, ytr)
                        else:
                            m = GridSearchCV(SVR(), {"C": [1, 10, 30], "gamma": ["scale", 0.01],
                                                     "epsilon": [0.1]}, n_jobs=1,
                                             cv=cv_int, scoring="neg_root_mean_squared_error").fit(Xtr, ytr)
                        pred[te] = m.predict(Xte) * ysd + ymu
                        for d, p, kk in zip([sub[i] for i in te], pred[te], k[te]):
                            linhas.append([d["arquivo"], rep, kk, alvo, "%s/%s" % (nome, regressor), round(float(p), 4)])
                    mt = rb.metricas(y, pred)
                    rmses.append((mt["R2"], mt["RMSE"]))
                r2 = np.array([a for a, _ in rmses]); rm = np.array([b for _, b in rmses])
                txt = "%-9s %-18s %-6s R2=%7.3f (%.3f)  RMSE=%8.3f (%.3f)" % (alvo, nome, regressor, r2.mean(), r2.std(), rm.mean(), rm.std())
                print(txt, flush=True)
                resumo.append(txt)
    with open(os.path.join(saida, "predicoes_embeddings_lukas.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow(["arquivo", "rep", "fold", "alvo", "modelo", "previsto"])
        w.writerows(linhas)
    with open(os.path.join(saida, "embeddings_resumo.txt"), "w", encoding="utf-8") as f:
        f.write("Media (desvio padrao) entre repeticoes, folds por parcela, selecao so no treino\n\n" + "\n".join(resumo) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    a = sp.add_parser("extrair"); a.add_argument("pasta"); a.add_argument("-o", "--saida", default="resultados/embeddings_resnet50.npz")
    a.add_argument("--pesos-h5", default=None, help="pesos Keras (resnet50_keras.py) em vez do torchvision")
    b = sp.add_parser("avaliar"); b.add_argument("embeddings"); b.add_argument("atributos"); b.add_argument("metadata")
    b.add_argument("-o", "--saida", default="resultados"); b.add_argument("--reps", type=int, default=10)
    b.add_argument("--pca", type=int, default=32)
    args = ap.parse_args()
    if args.cmd == "extrair":
        extrair(args.pasta, args.saida, args.pesos_h5)
    else:
        avaliar(args.embeddings, args.atributos, args.metadata, args.saida, args.reps, args.pca)


if __name__ == "__main__":
    main()
