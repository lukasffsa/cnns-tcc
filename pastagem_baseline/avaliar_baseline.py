#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compara a saida de pastagem_baseline.py com as medidas de campo (metadata.csv).

Uso:
    python avaliar_baseline.py baseline.csv dataset/metadata.csv -o metricas.txt

Nome da imagem  ->  linha do metadata:
    CR01R2_M1.jpeg  ->  parcela CR01, repeticao 2, momento M1  ->  "CR01M1"
Referencia de altura  = coluna "Média" (cm)
Referencia de biomassa = coluna "MS (kg/ha)"

Metricas: R2, RMSE, MAE, MAPE, indice de Willmott (WI), correlacao de Pearson
e vies (media de previsto - medido). Calculadas por imagem e tambem pela media
das repeticoes de cada parcela/momento.
"""
import argparse
import csv
import math
import re
from collections import defaultdict

PADRAO = re.compile(r"^([A-Za-z]{2})(\d{2})[Rr0](\d)_M(\d)")


def chave_da_imagem(nome):
    m = PADRAO.match(nome)
    if not m:
        return None
    cultivar, parcela, _rep, momento = m.groups()
    return "%s%sM%s" % (cultivar.upper(), parcela, momento)


def num(txt):
    txt = (txt or "").strip().replace(",", ".")
    return float(txt) if txt else None


def metricas(medido, previsto):
    n = len(medido)
    if n < 2:
        return None
    mm = sum(medido) / n
    mp = sum(previsto) / n
    sq = sum((p - o) ** 2 for o, p in zip(medido, previsto))
    st = sum((o - mm) ** 2 for o in medido)
    sp = sum((p - mp) ** 2 for p in previsto)
    cov = sum((o - mm) * (p - mp) for o, p in zip(medido, previsto))
    wi_den = sum((abs(p - mm) + abs(o - mm)) ** 2 for o, p in zip(medido, previsto))
    return {
        "n": n,
        "R2": 1 - sq / st if st else float("nan"),
        "RMSE": math.sqrt(sq / n),
        "MAE": sum(abs(p - o) for o, p in zip(medido, previsto)) / n,
        "MAPE_%": 100 * sum(abs((p - o) / o) for o, p in zip(medido, previsto) if o) / n,
        "WI": 1 - sq / wi_den if wi_den else float("nan"),
        "r": cov / math.sqrt(st * sp) if st and sp else float("nan"),
        "vies": mp - mm,
        "dp_previsto": math.sqrt(sp / n),
        "dp_medido": math.sqrt(st / n),
    }


def formata(titulo, m):
    if m is None:
        return "%s: dados insuficientes" % titulo
    return ("%-44s n=%3d  R2=%9.3f  RMSE=%9.2f  MAE=%9.2f  MAPE=%7.1f%%  WI=%6.3f  r=%6.3f  "
            "vies=%9.2f  dp_prev=%8.2f  dp_med=%8.2f"
            % (titulo, m["n"], m["R2"], m["RMSE"], m["MAE"], m["MAPE_%"], m["WI"], m["r"],
               m["vies"], m["dp_previsto"], m["dp_medido"]))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("baseline", help="CSV gerado por pastagem_baseline.py")
    ap.add_argument("metadata", help="metadata.csv do dataset")
    ap.add_argument("-o", "--saida", default=None, help="arquivo texto para gravar o relatorio")
    args = ap.parse_args(argv)

    with open(args.metadata, encoding="utf-8-sig") as f:
        meta = {l["ALTURA"].strip().upper(): l for l in csv.DictReader(f, delimiter=";")}
    with open(args.baseline, encoding="utf-8") as f:
        base = list(csv.DictReader(f, delimiter=";"))

    alvos = [("altura_mlp_cm", "Média", "Altura - MLP do app (cm)"),
             ("altura_linear_cm", "Média", "Altura - formula linear do app (cm)"),
             ("biomassa_kg_ha", "MS (kg/ha)", "Biomassa - MLP do app (kg/ha)")]

    saida = []
    sem_par, sem_medida = [], set()
    for col_prev, col_med, titulo in alvos:
        por_imagem = defaultdict(lambda: ([], []))      # momento -> (medido, previsto)
        por_parcela = defaultdict(lambda: [None, []])   # chave -> [medido, [previstos]]
        for l in base:
            chave = chave_da_imagem(l["arquivo"])
            if chave is None or chave not in meta:
                if l["arquivo"] not in sem_par:
                    sem_par.append(l["arquivo"])
                continue
            medido = num(meta[chave].get(col_med))
            if medido is None:
                sem_medida.add("%s (%s)" % (chave, col_med))
                continue
            previsto = float(l[col_prev])
            for grupo in ("todos", chave[-2:]):
                por_imagem[grupo][0].append(medido)
                por_imagem[grupo][1].append(previsto)
            por_parcela[chave][0] = medido
            por_parcela[chave][1].append(previsto)

        saida.append(titulo)
        for grupo in sorted(por_imagem, key=lambda g: (g != "todos", g)):
            o, p = por_imagem[grupo]
            saida.append(formata("  por imagem, %s" % grupo, metricas(o, p)))
        o = [v[0] for v in por_parcela.values()]
        p = [sum(v[1]) / len(v[1]) for v in por_parcela.values()]
        saida.append(formata("  media das repeticoes por parcela/momento", metricas(o, p)))
        saida.append("")

    if sem_par:
        saida.append("Imagens sem linha correspondente no metadata: " + ", ".join(sem_par))
    if sem_medida:
        saida.append("Linhas do metadata sem a medida (ignoradas): " + ", ".join(sorted(sem_medida)))

    texto = "\n".join(saida)
    print(texto)
    if args.saida:
        with open(args.saida, "w", encoding="utf-8") as f:
            f.write(texto + "\n")


if __name__ == "__main__":
    main()
