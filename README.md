# Predição de altura e biomassa de alfafa por imagens RGB

## Visão geral

Este projeto tem como objetivo estimar, a partir de imagens de plantas de alfafa, duas variáveis agronômicas importantes: altura e biomassa. A abordagem utiliza redes neurais convolucionais pré-treinadas em imagens naturais e adapta o modelo para regressão, buscando inferir valores contínuos relacionados ao crescimento da cultura.

O fluxo de trabalho combina:

- processamento e organização do conjunto de dados;
- alinhamento de imagens com metadados agronômicos;
- uso de transfer learning com arquiteturas populares de visão computacional;
- validação cruzada por folds para avaliação robusta;
- análise de métricas de regressão e visualização dos resultados.

## Motivação

A determinação da altura e da biomassa da alfafa é fundamental para monitoramento agronômico, avaliação de produtividade e tomada de decisão em sistemas de manejo da cultura. Métodos tradicionais frequentemente dependem de medições manuais e demoradas, enquanto a visão computacional oferece uma alternativa mais escalável e automatizada.

## Objetivos

- desenvolver um pipeline de regressão visual para estimar altura da alfafa;
- estimar a biomassa da cultura a partir das imagens;
- comparar diferentes arquiteturas de redes convolucionais;
- avaliar o desempenho com validação cruzada em múltiplas dobras;
- gerar métricas quantitativas e gráficos de dispersão dos resultados.

## Arquiteturas avaliadas

O projeto inclui experimentos com redes de transferência bem conhecidas:

- ResNet50;
- EfficientNet-B0;
- MobileNetV3 Large.

Em todas as abordagens, a etapa inicial do modelo é mantida com pesos pré-treinados do ImageNet e algumas camadas finais são descongeladas para adaptação ao domínio agrícola. O cabeçalho final é substituído por uma camada de regressão linear unidimensional, adequada para predição contínua.

## Base de dados

Os dados utilizados são compostos por:

- imagens armazenadas em dataset/images;
- metadados de campo em dataset/metadata.csv.

O arquivo CSV contém indicadores agronômicos como altura, biomassa, cultivar, parcela e fertilização com K2O. A associação entre imagem e amostra ocorre a partir de identificadores gerados pela combinação da parcela e do mês do registro.

A estrutura do conjunto é organizada para permitir que cada imagem seja associada ao valor real de altura e biomassa correspondente, seguindo um processo de normalização para estabilizar o treinamento.

## Pipeline do projeto

### 1. Carregamento e pré-processamento

A classe AlfalfaDataset realiza:

- leitura do CSV com separador semicolon;
- seleção das colunas relevantes;
- associação entre imagem e linha do metadado;
- carregamento da imagem em RGB;
- conversão dos alvos em valores normalizados;
- retorno do tensor da imagem e dos valores de altura e biomassa.

### 2. Transformações de dados

São aplicadas transformações de pré-processamento e aumento de dados, incluindo:

- resize para resolução 224x224;
- flip horizontal e vertical;
- rotação moderada;
- ajuste de brilho, contraste e saturação;
- normalização com estatísticas do ImageNet.

Essas transformações ajudam a aumentar a robustez do treinamento e a reduzir o risco de overfitting.

### 3. Regressão por redes convolucionais

A implementação usa regressão com perda MSE e otimizador Adam. O treinamento é realizado separadamente para cada alvo:

- modelo dedicado para altura;
- modelo dedicado para biomassa.

A ideia é especializar cada rede para uma variável específica, em vez de otimizar uma saída múltipla com menor controle sobre a tarefa.

### 4. Validação cruzada

A validação é feita via 5-fold cross-validation com embaralhamento, usando KFold do scikit-learn. Em cada dobra:

- uma parte dos dados é usada para treino;
- outra parte é usada para teste;
- os modelos são treinados e avaliados independently;
- os resultados são agregados para métricas globais.

### 5. Avaliação e visualização

O projeto calcula:

- RMSE;
- R²;
- gráficos de dispersão real versus predito.

Os gráficos são salvos localmente como imagens, permitindo uma análise visual direta do comportamento dos modelos.

## Estrutura do repositório

- main.py: pipeline principal de treinamento e validação em 5 folds;
- Dataset.py: definição do dataset e carregamento dos dados;
- views.py: funções para cálculo de métricas e geração dos gráficos;
- models/: implementações das arquiteturas ResNet, EfficientNet e MobileNet;
- dataset/: conjunto de metadados e imagens do experimento;
- resultados-*/: arquivos de saída e relatórios de execução;
- pyproject.toml: configuração do projeto e dependências.

## Dependências

O projeto utiliza as seguintes bibliotecas principais:

- Python;
- PyTorch;
- TorchVision;
- scikit-learn;
- pandas;
- matplotlib;
- PIL/Pillow.

A configuração do projeto também contempla dependências do ambiente via pyproject.toml.

## Como executar

### Instalação

Recomendado:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Ou, se o ambiente for gerenciado com uv:

```bash
uv sync
```

### Treinamento

```bash
python main.py
```

Esse comando inicia a validação cruzada e executa o treinamento dos modelos dedicados para altura e biomassa. Ao final, as métricas e os gráficos de dispersão são gerados e exibidos/armazenados no diretório do projeto.

## Resultados observados

Os resultados obtidos durante a execução do pipeline indicam desempenho variável entre as tarefas:

| Tarefa | R² | RMSE |
| --- | --- | --- |
| Altura | 0.2031 | 4.8654 cm |
| Biomassa | 0.5419 | 319.6064 kg/ha |

Esses valores demonstram que o modelo se mostrou mais estável para estimar biomassa do que altura, embora o desempenho geral ainda seja passível de melhoria por meio de refinamento do conjunto de dados, hiperparâmetros e estratégias de treinamento.

## Limitações e possibilidades de evolução

- o conjunto de dados é relativamente pequeno para o treinamento de redes profundas em larga escala;
- a associação entre imagem e metadado depende de convenções de nomenclatura;
- o projeto trabalha com modelos de regressão independentes para cada variável;
- há espaço para exploração de arquiteturas mais recentes, tuning de hiperparâmetros e técnicas de regularização.

Entre possíveis melhorias:

- uso de modelos multitarefa para prever altura e biomassa em uma única rede;
- aumento do conjunto de dados com mais imagens e diversidade agronômica;
- aplicação de técnicas de validação temporal ou por parcela;
- comparação com modelos baseados em segmentação ou extração de features clássicas.

## Conclusão

O projeto demonstra a viabilidade de utilizar aprendizado profundo em imagens RGB para estimar variáveis agronômicas de alfafa. Embora os resultados ainda tenham margem para melhoria, a abordagem mostra potencial para automatizar monitoramento de culturas e reduzir a dependência de medições manuais intensivas.

A estrutura modular do código permite expansão para novas arquiteturas, novos experimentos e melhor comparação de desempenho em cenários agrícolas reais.
