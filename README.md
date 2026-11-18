# Predição de Altura e Biomassa de Alfafa a partir de Imagens RGB

Este projeto apresenta um pipeline experimental para estimar a altura e a biomassa da alfafa por meio de imagens RGB. A solução compara redes neurais convolucionais, um modelo MLP baseado em descritores de textura e o baseline móvel originalmente utilizado pelo aplicativo Pastagem.

O trabalho combina visão computacional, validação cruzada por parcela, normalização dos alvos, métricas de regressão e ferramentas de comparação entre resultados.

## Objetivos

- Estimar a altura das plantas em centímetros.
- Estimar a biomassa seca em kg/ha.
- Comparar arquiteturas CNN com modelos treinados a partir de descritores clássicos.
- Evitar vazamento entre parcelas durante a validação cruzada.
- Produzir métricas, previsões out-of-fold e gráficos diagnósticos reproduzíveis.
- Manter compatibilidade com os resultados e pesos do baseline desenvolvido em Java.

## Abordagens avaliadas

### Redes convolucionais

As redes CNN utilizam pesos pré-treinados no ImageNet e têm as últimas etapas de extração ajustadas ao domínio agrícola:

| Modelo | Backbone | Ajuste |
| --- | --- | --- |
| MLP | Descritores de textura | Treinado com vetores de duas dimensões |
| MobileNetV3-Large | MobileNetV3-Large | Últimos três blocos descongelados |
| ResNet50 | ResNet50 | `layer4` descongelada |
| EfficientNet-B0 | EfficientNet-B0 | Estágios finais 7 e 8 descongelados |

A saída final é uma regressão escalar. Altura e biomassa são treinadas por modelos separados.

### Baseline Pastagem

O diretório `pastagem_baseline/` contém uma implementação em Python do baseline móvel:

1. Conversão da imagem RGB para níveis de cinza.
2. Construção de uma matriz de coocorrência 8x8.
3. Cálculo de entropia, homogeneidade, desvio padrão do cinza e percentual de área escura.
4. Cálculo do índice GLI.
5. Aplicação das redes MLP e da regressão linear exportadas do aplicativo Java.

Os pesos originais são armazenados em `pastagem_baseline/pesos_app.json`.

## Conjunto de dados

Os dados utilizados ficam em `dataset/`:

- `dataset/images/`: imagens RGB das parcelas.
- `dataset/metadata.csv`: medidas de campo e informações das parcelas.
- Como as imagens são de domínio restrito da Embrapa, apenas o metadata.csv ficará disponível no repositório. Esse csv contem a ligação das imagens com seus valores de referência e é apenas o que o usuário precisa para ropdar o código.

Os nomes das imagens associam a observação à parcela e ao momento de coleta. O arquivo de metadados contém, entre outros campos:

- Parcela.
- Cultivar.
- Altura média em centímetros.
- Massa seca em kg/ha.
- Fertilização com K2O.

Para os experimentos CNN atuais, `pastagem_baseline/resultados/folds_por_parcela.csv` define a divisão por parcelas. O código impede:

- Identificadores repetidos.
- Separação de imagens da mesma parcela entre treino e teste.
- Referências negativas.
- Valores não finitos no conjunto de dados.

A imagem de biomassa da parcela `CR17` é excluída explicitamente do experimento de biomassa para preservar a comparação com o baseline recebido, mas caso o usuário queira rodar com essa parcela, os valores de referência já estão ajustados corretamente.

## Métricas

A avaliação considera:

- R².
- Índice de Willmott (WI).
- RMSE.
- MAE.
- MAPE.
- Bias médio.
- Previsões out-of-fold.

As métricas são implementadas em `metrics.py`. O MAPE fica indefinido quando o alvo observado é zero; nesse caso, os valores são registrados como ausentes sem remover amostras.

## Estrutura do repositório

```text
.
|-- main.py                     Pipeline principal das CNN
|-- Dataset.py                  Dataset legado de altura e biomassa
|-- models/
|   |-- MLP.py                  Regressor MLP com descritores
|   |-- Resnet.py               Adaptacao da ResNet50
|   |-- Mobilenet.py            Adaptacao da MobileNetV3-Large
|   `-- Efficentnet.py          Adaptacao da EfficientNet-B0
|-- pastagem_ml/
|   `-- features.py             Extracao dos descritores utilizados pelo MLP
|-- pastagem_baseline/
|   |-- pastagem_baseline.py    Implementacao Python do app Pastagem
|   |-- avaliar_baseline.py    Avaliacao contra o metadata
|   `-- pesos_app.json          Pesos exportados do aplicativo Java
|-- dataset/
|   |-- images/
|   `-- metadata.csv
|-- metrics.py                  Metricas de regressao
|-- mlp_features.py             Vetores usados pelo MLP
|-- validation.py               Validacao em 10 folds
|-- views.py                    Graficos e comparacoes
|-- tests/                      Testes unitarios e de integracao sintetica
|-- pyproject.toml              Dependencias e metadados do projeto
`-- uv.lock                     Lockfile do ambiente
```

## Requisitos

- Python 3.14 ou superior.
- PyTorch e TorchVision.
- NumPy.
- pandas.
- scikit-learn.
- Matplotlib.
- Pillow.

As dependências principais estão declaradas em `pyproject.toml`.

## Instalação

Com `uv`:

```bash
uv sync
```

Com um ambiente virtual padrão:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

No Windows PowerShell, a ativação do ambiente virtual é:

```powershell
.venv\Scripts\Activate.ps1
```

## Executando os experimentos

### Treinamento das CNN

Na raiz do projeto:

```bash
python main.py
```

O pipeline:

- Executa até 10 repetições de validação em cinco folds.
- Treina MobileNetV3-Large, ResNet50 e EfficientNet-B0.
- Avalia altura e biomassa separadamente.
- Detecta automaticamente CPU ou GPU CUDA.
- Grava as previsões, escalas, perdas e configurações em `resultados_cnn/`.

Os parâmetros principais estão no início de `main.py`:

```python
EPOCAS = 20
REPETICOES = 10
BATCH_SIZE = 16
TAXA_APRENDIZADO = 1e-4
```

### Avaliação do baseline móvel

Para executar o baseline sobre uma pasta de imagens:

```bash
python pastagem_baseline/pastagem_baseline.py dataset/images
```

O comando pode informar explicitamente o arquivo de saída:

```bash
python pastagem_baseline/pastagem_baseline.py dataset/images -o pastagem_baseline/resultados/baseline_python.csv
```

Depois, compare as previsões com o metadata:

```bash
python pastagem_baseline/avaliar_baseline.py \
  pastagem_baseline/resultados/baseline_python.csv \
  dataset/metadata.csv \
  -o pastagem_baseline/resultados/metricas_baseline.txt
```

### Comparação dos modelos

O módulo `views.py` combina diretórios de resultados que contenham `fold_metrics.csv` e `oof_predictions.csv`:

```bash
python views.py \
  resultados_todos_10folds_20260930_170636_592764/mlp \
  resultados_todos_10folds_20260930_170636_592764/mobilenet \
  resultados_todos_10folds_20260930_170636_592764/resnet \
  resultados_todos_10folds_20260930_170636_592764/efficientnet \
  --output comparacoes
```

Os diretórios comparados devem usar o mesmo conjunto de dados e os mesmos folds.

## Testes

Execute todos os testes disponíveis com:

```bash
python -m unittest discover -s tests -v
```

Os testes validam:

- Fórmulas de R², WI, RMSE, MAE e MAPE.
- Comportamento indefinido de métricas.
- Extratores de características equivalentes ao código Java.
- Cobertura e reprodução dos 10 folds.
- Rejeição de arquivos antigos com cinco folds.
- Geração dos gráficos e alinhamento das previsões.
- Integração sintética com PyTorch, quando disponível.

## Resultados disponíveis

O diretório `resultados_todos_10folds_20260930_170636_592764/` contém uma execução consolidada em dez folds. A tabela abaixo resume a média dos folds registrados nesse diretório.

| Modelo | Alvo | R² médio | RMSE médio |
| --- | --- | --- | --- |
| MLP | Altura | -4.877 | 10.673 cm |
| MLP | Biomassa | -0.087 | 476.317 kg/ha |
| MobileNetV3-Large | Altura | -0.947 | 7.194 cm |
| MobileNetV3-Large | Biomassa | 0.149 | 410.148 kg/ha |
| ResNet50 | Altura | -0.264 | 5.837 cm |
| ResNet50 | Biomassa | 0.397 | 351.668 kg/ha |
| EfficientNet-B0 | Altura | -1.373 | 7.980 cm |
| EfficientNet-B0 | Biomassa | 0.289 | 378.305 kg/ha |

Esses valores são resultados experimentais registrados no repositório e não substituem uma avaliação independente. Entre os modelos CNN consolidados, a ResNet50 apresentou o melhor RMSE médio nos dois alvos dessa execução.

## Organização dos resultados

Os resultados principais podem incluir:

- `oof_predictions.csv`: previsões out-of-fold por amostra.
- `fold_metrics.csv`: métricas por modelo, alvo e fold.
- `summary_folds.csv`: média e desvio das métricas entre os folds.
- `diagnostico_*.png`: gráficos observado versus predito e resíduos.
- `comparacao_*.png`: comparação dos modelos por fold.
- `erros_absolutos_*.png`: distribuição dos erros absolutos.
- `loss_*_*.csv`: histórico de perda durante o treinamento CNN.

## Limitações

- O conjunto de dados é limitado para redes profundas em larga escala.
- Algumas métricas por fold podem ficar indefinidas na presença de alvos iguais ou iguais a zero.
- Os resultados dependem da identidade visual e da diversidade das parcelas.
- O treinamento deve evitar critérios de divisão aleatórios que misturem a mesma parcela entre treino e teste.
- Pesos pré-treinados exigem acesso à internet na primeira execução, caso ainda não estejam armazenados no cache local.
- Os resultados devem ser interpretados junto com a estrutura amostral e a distribuição dos alvos.

## Extensões possíveis

- Ajuste sistemático de hiperparâmetros.
- Aumento e balanceamento do conjunto de dados.
- Validação externa por campanha, cultivar ou região.
- Modelo único com duas saídas.
- Segmentação da planta antes da extração de atributos.
- Comparação com modelos tabulares e arquiteturas recentes.
- Empacotamento da inferência em uma aplicação ou API.

## Licença

MIT License

Copyright (c) 2026 Lukas de Sá

