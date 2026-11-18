#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Baseline do app Pastagem (Android/Java) portado para Python.

Reproduz a parte de CALCULO do app, sem tela, login, GPS ou Firebase:

  imagem RGB
    -> nivel de cinza (media de R, G, B) quantizado em 8 niveis
    -> matriz de co-ocorrencia 8x8 (vizinho da direita)
    -> entropia, homogeneidade, segundo momento angular
    -> desvio padrao do cinza, % de area escura, GLI medio
    -> ALTURA   = MLP(entropia, homogeneidade)                (AlturaActivity)
    -> BIOMASSA = MLP(altura_linear, GLI medio)               (BiomassaActivity)
       onde altura_linear e uma regressao linear dos atributos de textura

Os pesos das redes vem de pesos_app.json, extraidos automaticamente do codigo
Java. Nao ha treinamento aqui: o app so embarca modelos ja treinados.

Uso:
    python pastagem_baseline.py PASTA_DE_IMAGENS -o baseline.csv

Dependencias: numpy e Pillow  (pip install numpy pillow)
"""
import argparse
import csv
import json
import math
import os
import sys

import numpy as np
from PIL import Image

AQUI = os.path.dirname(os.path.abspath(__file__))
EXTENSOES = (".jpg", ".jpeg", ".png")


# --------------------------------------------------------------------------
# Modelos embarcados no app
# --------------------------------------------------------------------------
def mlp_predict(modelo, entradas):
    """Passagem direta da rede, igual ao metodo predict() das classes Java.

    Camadas ocultas com tanh, saida linear. As somas sao feitas na mesma
    ordem do Java (bias primeiro, depois cada entrada) para o resultado
    bater ate a ultima casa.
    """
    pesos, bias = modelo["weights"], modelo["bias"]
    ativ = list(entradas)
    n_camadas = len(pesos)
    for i in range(n_camadas):
        prox = []
        for j in range(len(bias[i])):
            soma = bias[i][j]
            for l in range(len(ativ)):
                soma += ativ[l] * pesos[i][l][j]
            prox.append(soma)
        if i < n_camadas - 1:
            prox = [math.tanh(v) for v in prox]
        ativ = prox
    return ativ[0]


# --------------------------------------------------------------------------
# Atributos da imagem
# --------------------------------------------------------------------------
def _soma_sequencial(v):
    """Soma na ordem dos pixels, como o laco do Java (np.sum usa outra ordem)."""
    return float(np.cumsum(v, dtype=np.float64)[-1]) if v.size else 0.0


def _soma_float32_do_app(x):
    """Reproduz `somaDP = (float)(somaDP + x)` do Java.

    No app o acumulador do desvio padrao e convertido para float (32 bits) a
    cada pixel, o que perde precisao quando a soma fica grande. Para comparar
    com o app sem mudar a metodologia, essa perda e reproduzida aqui.
    Dentro de uma mesma potencia de 2 o arredondamento de cada parcela nao
    depende do acumulado, entao da para vetorizar por faixas.
    """
    s = 0.0
    n = x.size
    idx = 0
    bloco = 1 << 16
    while idx < n:
        if s <= 0.0:
            s = float(np.float32(s + x[idx]))
            idx += 1
            continue
        expo = math.frexp(s)[1]            # s em [2^(expo-1), 2^expo)
        ulp = math.ldexp(1.0, expo - 24)   # passo do float32 nessa faixa
        topo = math.ldexp(1.0, expo)
        trecho = x[idx:idx + bloco]
        acumulado = s + np.cumsum(np.rint(trecho / ulp)) * ulp
        pos = int(np.searchsorted(acumulado, topo, side="left"))
        if pos > 0:
            s = float(acumulado[pos - 1])
            idx += pos
        if pos < trecho.size:              # este pixel muda de faixa: faz na mao
            s = float(np.float32(s + x[idx]))
            idx += 1
    return s


def atributos(rgb, dp_float32=True):
    """Calcula os atributos de uma imagem (array HxWx3, uint8).

    Segue AlturaActivity.textura() e BiomassaActivity.previsaoBiomassa().
    """
    alt, larg = rgb.shape[:2]
    r = rgb[..., 0].astype(np.int64)
    g = rgb[..., 1].astype(np.int64)
    b = rgb[..., 2].astype(np.int64)
    soma_rgb = r + g + b                    # 0..765
    n = alt * larg

    # nivel de cinza: avg = (r+g+b)/3
    avg = soma_rgb.astype(np.float64) / 3.0
    avg_medio = _soma_sequencial(avg.ravel()) / n

    # GLI = (2g - r - b)/(2g + r + b); pixels pretos (denominador 0) nao somam,
    # mas a media divide por TODOS os pixels (igual ao app)
    den = (2 * g + r + b).astype(np.float64)
    num = (2 * g - r - b).astype(np.float64)
    gli = np.divide(num, den, out=np.zeros_like(num), where=den != 0)
    gli_medio = _soma_sequencial(gli.ravel()) / n

    # matriz de co-ocorrencia 8x8: pixel (linha) x vizinho da direita (coluna)
    nivel = soma_rgb // 96                  # == (int)avg/32 e (int)((r+g+b)/96)
    pares = nivel[:, :-1] * 8 + nivel[:, 1:]
    cooc = np.bincount(pares.ravel(), minlength=64).astype(np.float64).reshape(8, 8)
    cooc = cooc / float((larg - 1) * alt)

    entropia = 0.0          # atencao: o app NAO troca o sinal (fica negativa)
    homogeneidade = 0.0
    segundo_momento = 0.0
    for i in range(8):
        for j in range(8):
            p = float(cooc[i, j])
            segundo_momento += p * p
            if p != 0:
                entropia += p * (math.log(p) / math.log(2))
            homogeneidade += p / (1 + (i - j) ** 2)

    # desvio padrao do cinza e % de pixels escuros (avg <= 30)
    quad = ((avg - avg_medio) ** 2).ravel()
    if dp_float32:
        var = float(np.float32(_soma_float32_do_app(quad))) / n
        desvio = float(np.float32(math.sqrt(var)))
    else:
        desvio = math.sqrt(_soma_sequencial(quad) / n)
    area_escura = 100.0 * (float(np.count_nonzero(soma_rgb <= 90)) / float(n))

    return {
        "entropia": entropia,
        "homogeneidade": homogeneidade,
        "segundo_momento_angular": segundo_momento,
        "desvio_padrao_cinza": desvio,
        "area_escura_pct": area_escura,
        "gli_medio": gli_medio,
    }


def atributos_extras(rgb):
    """Outros indices que o app mostra na tela mas que NAO entram nos modelos.

    Calculados em precisao dupla (o app usa float de 32 bits nessas telas,
    entao aqui os valores podem diferir do app nas ultimas casas).
    """
    r = rgb[..., 0].astype(np.float64)
    g = rgb[..., 1].astype(np.float64)
    b = rgb[..., 2].astype(np.float64)

    def media_valida(num, den):
        ok = den != 0
        return float(np.mean(num[ok] / den[ok])) if ok.any() else float("nan")

    return {
        "mpri_medio": media_valida(g - r, g + r),
        "vari_medio": media_valida(g - r, g + r - b),
        "tgi_medio": float(np.mean(g - 0.39 * r - 0.61 * b)),
        # Segmentacao, modo "green coverage": pixel verde se R < G e B < G
        "cobertura_verde_pct": 100.0 * float(np.mean((r < g) & (b < g))),
    }


def prever(attr, pesos):
    """Aplica os modelos embarcados no app aos atributos."""
    m_alt, m_bio, lin = pesos["altura"], pesos["biomassa"], pesos["altura_linear"]

    # AlturaActivity: MLP(entropia, homogeneidade), depois desfaz a padronizacao
    altura_mlp = mlp_predict(m_alt, [attr["entropia"], attr["homogeneidade"]])
    altura_mlp = altura_mlp * m_alt["saida_escala"] + m_alt["saida_media"]

    # BiomassaActivity: altura por regressao linear + GLI medio -> MLP
    altura_linear = (lin["intercepto"]
                     + attr["entropia"] * lin["entropia"]
                     + attr["homogeneidade"] * lin["homogeneidade"]
                     + attr["desvio_padrao_cinza"] * lin["desvio_padrao_cinza"]
                     + attr["area_escura_pct"] * lin["area_escura_pct"])
    biomassa = mlp_predict(m_bio, [altura_linear, attr["gli_medio"]])
    biomassa = biomassa * m_bio["saida_escala"] + m_bio["saida_media"]

    return {
        "altura_mlp_cm": altura_mlp,
        "altura_linear_cm": altura_linear,
        "biomassa_kg_ha": biomassa,
    }


# --------------------------------------------------------------------------
# Linha de comando
# --------------------------------------------------------------------------
def carregar_imagem(caminho, dividir=1.0):
    """Le a imagem como o app: sem girar pelo EXIF, em RGB.

    `dividir` reproduz a reducao que o app aplica em aparelhos lentos
    (largura e altura divididas por esse fator, interpolacao bilinear).
    """
    img = Image.open(caminho).convert("RGB")
    if dividir and dividir != 1.0:
        novo = (int(img.width / dividir), int(img.height / dividir))
        img = img.resize(novo, Image.BILINEAR)
    return np.asarray(img)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Baseline do app Pastagem em Python.")
    ap.add_argument("pasta", help="pasta com as imagens (.jpg, .jpeg, .png)")
    ap.add_argument("-o", "--saida", default="baseline.csv", help="CSV de saida")
    ap.add_argument("--pesos", default=os.path.join(AQUI, "pesos_app.json"))
    ap.add_argument("--dividir", type=float, default=1.0,
                    help="fator de reducao da imagem (padrao 1 = tamanho original)")
    ap.add_argument("--dp-float64", action="store_true",
                    help="desvio padrao em precisao dupla (nao imita o float do app)")
    args = ap.parse_args(argv)

    with open(args.pesos, encoding="utf-8") as f:
        pesos = json.load(f)

    arquivos = sorted(a for a in os.listdir(args.pasta) if a.lower().endswith(EXTENSOES))
    if not arquivos:
        sys.exit("Nenhuma imagem encontrada em " + args.pasta)

    linhas = []
    for k, nome in enumerate(arquivos, 1):
        rgb = carregar_imagem(os.path.join(args.pasta, nome), args.dividir)
        attr = atributos(rgb, dp_float32=not args.dp_float64)
        linha = {"arquivo": nome, "largura": rgb.shape[1], "altura_px": rgb.shape[0]}
        linha.update(attr)
        linha.update(prever(attr, pesos))
        linha.update(atributos_extras(rgb))
        linhas.append(linha)
        print("[%d/%d] %s  altura=%.1f cm  biomassa=%.1f kg/ha"
              % (k, len(arquivos), nome, linha["altura_mlp_cm"], linha["biomassa_kg_ha"]))

    with open(args.saida, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(linhas[0].keys()), delimiter=";")
        w.writeheader()
        for linha in linhas:
            w.writerow({c: (repr(v) if isinstance(v, float) else v) for c, v in linha.items()})
    print("Gravado:", args.saida)


if __name__ == "__main__":
    main()
