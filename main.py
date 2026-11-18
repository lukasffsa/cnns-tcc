"""CNNs: cinco folds por parcela, nas divisões fornecidas pelo orientador."""
from datetime import datetime
from pathlib import Path
import json
import random

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from models.Mobilenet import MobileNetSingleRegressor
from models.Resnet import ResNet50SingleRegressor
from models.Efficentnet import EfficientNetSingleRegressor

# CONFIGURAÇÕES 
EPOCAS = 20
REPETICOES = 10                    # 5: rep00 a rep04; 10: rep00 a rep09.
MOSTRAR_LOSS_A_CADA = 5
BATCH_SIZE = 16
TAXA_APRENDIZADO = 1e-4
SEMENTE = 42
METADATA = Path("dataset/metadata.csv")
IMAGENS = Path("dataset/images")
FOLDS = Path("pastagem_baseline/resultados/folds_por_parcela.csv")
MODELOS = {
    "mobilenet": MobileNetSingleRegressor,
    "resnet": ResNet50SingleRegressor,
    "efficientnet": EfficientNetSingleRegressor,
}


def carregar_dados():
    meta = pd.read_csv(METADATA, sep=";", encoding="utf-8-sig")
    dados = pd.read_csv(FOLDS, sep=";", encoding="utf-8-sig")
    for coluna in ["ALTURA", "PARCELA", "Cultivar"]:
        meta[coluna] = meta[coluna].astype(str).str.strip().str.upper()
    for coluna in ["parcela", "cultivar", "momento"]:
        dados[coluna] = dados[coluna].astype(str).str.strip().str.upper()
    dados["arquivo"] = dados["arquivo"].str.strip()
    if meta.ALTURA.duplicated().any() or dados.arquivo.duplicated().any():
        raise ValueError("Há identificadores repetidos no metadata ou nos folds.")
    for coluna in ["Média", "MS (kg/ha)"]:
        texto = meta[coluna].map(
            lambda valor: str(valor).replace(",", ".") if pd.notna(valor) else np.nan)
        meta[coluna] = pd.to_numeric(texto, errors="raise")
    referencias = meta.set_index("ALTURA").loc[dados.parcela + dados.momento]
    if not np.array_equal(referencias.PARCELA, dados.parcela):
        raise ValueError("Parcelas do metadata e dos folds não correspondem.")
    if not np.array_equal(referencias.Cultivar, dados.cultivar):
        raise ValueError("Cultivares do metadata e dos folds não correspondem.")
    dados["altura"] = referencias["Média"].to_numpy()
    dados["biomassa"] = referencias["MS (kg/ha)"].to_numpy()
    if not np.isfinite(dados.altura).all() or np.isinf(dados.biomassa).any():
        raise ValueError("Confira alturas ausentes e valores não finitos no metadata.")
    if (dados[["altura", "biomassa"]] < 0).any().any():
        raise ValueError("Há referências negativas no metadata.")
    if dados.groupby("parcela").biomassa.nunique().gt(1).any():
        raise ValueError("A biomassa deve ser a mesma nos momentos de cada parcela.")
    for rep in range(REPETICOES):
        coluna = f"rep{rep:02d}"
        if set(dados[coluna]) != set(range(5)):
            raise ValueError(f"{coluna} deve conter os folds de 0 a 4.")
        if dados.groupby("parcela")[coluna].nunique().gt(1).any():
            raise ValueError(f"{coluna} separa imagens da mesma parcela.")
    for arquivo in dados.arquivo:
        if not (IMAGENS / arquivo).is_file():
            raise FileNotFoundError(f"Imagem não encontrada: {IMAGENS / arquivo}")
    return dados, meta


def dividir_fold(dados, alvo, rep, fold):
    amostras = dados.loc[dados[alvo].notna()]
    if alvo == "biomassa":
        # Preserva a exclusão de CR17 no baseline recebido, mesmo se for corrigida.
        amostras = amostras.loc[amostras.parcela != "CR17"]
    coluna = f"rep{rep:02d}"
    treino = amostras.loc[amostras[coluna] != fold].reset_index(drop=True)
    teste = amostras.loc[amostras[coluna] == fold].reset_index(drop=True)
    if treino.empty or teste.empty:
        raise ValueError(f"Treino ou teste vazio: {alvo}, repetição {rep}, fold {fold}.")
    escala = float(treino[alvo].max())
    if escala <= 0:
        raise ValueError("O máximo do alvo no treino deve ser positivo.")
    return treino, teste, escala


class Fotos(Dataset):
    def __init__(self, linhas, alvo, escala, treino):
        self.linhas, self.alvo, self.escala = linhas, alvo, escala
        etapas = [transforms.Resize((224, 224))]
        if treino:
            etapas += [transforms.RandomHorizontalFlip(0.5),
                       transforms.RandomVerticalFlip(0.5), transforms.RandomRotation(15),
                       transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1)]
        etapas += [transforms.ToTensor(),
                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
        self.transformar = transforms.Compose(etapas)

    def __len__(self):
        return len(self.linhas)

    def __getitem__(self, indice):
        linha = self.linhas.iloc[indice]
        with Image.open(IMAGENS / linha.arquivo) as imagem:
            entrada = self.transformar(imagem.convert("RGB"))
        alvo = torch.tensor(float(linha[self.alvo]) / self.escala, dtype=torch.float32)
        return entrada, alvo, linha.arquivo


def definir_semente(semente):
    random.seed(semente)
    np.random.seed(semente)
    torch.manual_seed(semente)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(semente)


def treinar(modelo, loader, dispositivo, contexto):
    otimizador = torch.optim.Adam(
        filter(lambda parametro: parametro.requires_grad, modelo.parameters()),
        lr=TAXA_APRENDIZADO, weight_decay=1e-4)
    criterio = nn.MSELoss()
    historico = []
    for epoca in range(1, EPOCAS + 1):
        modelo.train()
        soma = 0.0
        for imagens, alvos, _ in loader:
            imagens, alvos = imagens.to(dispositivo), alvos.to(dispositivo)
            otimizador.zero_grad()
            loss = criterio(modelo(imagens), alvos)
            if not torch.isfinite(loss):
                raise ValueError(f"Loss não finita em {contexto}.")
            loss.backward()
            otimizador.step()
            soma += loss.item() * len(alvos)
        media = soma / len(loader.dataset)
        historico.append({"epoca": epoca, "loss_treino_mse_normalizada": media})
        if epoca == 1 or epoca % MOSTRAR_LOSS_A_CADA == 0 or epoca == EPOCAS:
            print(f"[{contexto}] Época {epoca}/{EPOCAS} | Loss treino: {media:.8f}", flush=True)
    return historico


def prever(modelo, loader, dispositivo, escala, rep, fold, alvo):
    linhas = []
    modelo.eval()
    with torch.no_grad():
        for imagens, _, arquivos in loader:
            valores = modelo(imagens.to(dispositivo)).cpu().numpy() * escala
            if not np.isfinite(valores).all():
                raise ValueError("O modelo gerou previsões não finitas.")
            for arquivo, valor in zip(arquivos, valores):
                linhas.append({"arquivo": arquivo, "rep": rep, "fold": fold,
                               "alvo": alvo, "previsto": float(valor)})
    return linhas


def main():
    if min(EPOCAS, BATCH_SIZE, MOSTRAR_LOSS_A_CADA) < 1 or not 1 <= REPETICOES <= 10:
        raise ValueError("Use configurações positivas e de 1 a 10 repetições.")
    dados, meta = carregar_dados()
    dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    saida = Path("resultados_cnn") / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    saida.mkdir(parents=True)
    # Use esta cópia no comparador para excluir CR17 também na avaliação do baseline.
    meta.loc[meta.PARCELA == "CR17", "MS (kg/ha)"] = np.nan
    meta.to_csv(saida / "metadata_comparacao.csv", sep=";", index=False)
    configuracao = {"epocas": EPOCAS, "repeticoes": REPETICOES, "folds": 5,
                    "modelos": list(MODELOS), "semente": SEMENTE, "batch_size": BATCH_SIZE,
                    "lr": TAXA_APRENDIZADO, "optimizer": "Adam", "weight_decay": 1e-4,
                    "escala_alvo": "Máximo do treino de cada fold", "exclusao_biomassa": "CR17",
                    "parada_antecipada": False, "dispositivo": str(dispositivo)}
    (saida / "configuracao.json").write_text(
        json.dumps(configuracao, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Pasta de execução: {Path.cwd()}\nDispositivo: {dispositivo}", flush=True)
    if dispositivo.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(dispositivo)}", flush=True)
    print(f"Saída: {saida.resolve()} | {REPETICOES} repetições de 5 folds", flush=True)
    print("CR17 fica fora da biomassa, mantendo a comparação com o baseline recebido.", flush=True)
    for nome, criar_modelo in MODELOS.items():
        pasta = saida / nome
        pasta.mkdir()
        predicoes, escalas = [], []
        for rep in range(REPETICOES):
            for fold in range(5):
                for indice, alvo in enumerate(["altura", "biomassa"]):
                    definir_semente(SEMENTE + rep * 100 + fold * 2 + indice)
                    treino, teste, escala = dividir_fold(dados, alvo, rep, fold)
                    loader_treino = DataLoader(Fotos(treino, alvo, escala, True),
                                              batch_size=BATCH_SIZE, shuffle=True)
                    loader_teste = DataLoader(Fotos(teste, alvo, escala, False),
                                             batch_size=BATCH_SIZE, shuffle=False)
                    contexto = f"{nome} | rep{rep:02d} | fold {fold} | {alvo}"
                    print(f"\n[{contexto}] Treino: {len(treino)} fotos | Teste: {len(teste)} fotos", flush=True)
                    modelo = criar_modelo().to(dispositivo)
                    historico = treinar(modelo, loader_treino, dispositivo, contexto)
                    pd.DataFrame(historico).to_csv(
                        pasta / f"loss_rep{rep:02d}_fold{fold}_{alvo}.csv", sep=";", index=False)
                    predicoes += prever(modelo, loader_teste, dispositivo, escala, rep, fold, alvo)
                    escalas.append({"rep": rep, "fold": fold, "alvo": alvo, "maximo_treino": escala})
                    pd.DataFrame(predicoes).to_csv(pasta / "predicoes_parciais.csv", sep=";", index=False)
                    pd.DataFrame(escalas).to_csv(pasta / "escalas.csv", sep=";", index=False)
                    del modelo
                    if dispositivo.type == "cuda":
                        torch.cuda.empty_cache()
        (pasta / "predicoes_parciais.csv").rename(pasta / "predicoes_cnn.csv")
        print(f"\n{nome} concluída: {pasta / 'predicoes_cnn.csv'}", flush=True)
    print("\nTreinamentos concluídos. Use os CSVs finais e metadata_comparacao.csv no comparador.")


if __name__ == "__main__":
    main()
