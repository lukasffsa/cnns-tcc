# Experimentos extras e consolidação final

Copie estes arquivos para dentro da pasta `pastagem_baseline` (os scripts na raiz, o conteúdo de `resultados/` em `resultados/`).

## O que foi feito

Todos os experimentos usam os mesmos folds por parcela (`folds_por_parcela.csv`, rep00 a rep09) e escolhem hiperparâmetros só com o treino de cada fold, por validação interna também por parcela. A CR17 fica fora da biomassa, como no baseline e nas suas CNNs.

- ResNet-50 congelada como extrator: imagem inteira (lado menor 448), média global, média com o flip horizontal. O vetor de 2048 dimensões passa por PCA com 32 componentes ajustado no treino e entra em LASSO ou SVR.
- Embedding + os atributos manuais do baseline.
- Atributos manuais + cultivar e momento (one-hot), com e sem o embedding.
- Avaliação por parcela: média das predições das fotos de cada parcela e momento, para todos os modelos.

Pesos da ResNet-50: como o servidor do torchvision não estava acessível no ambiente onde rodei, usei os pesos ImageNet do Keras Applications (GitHub), convertidos em `resnet50_keras.py`. Na sua máquina, o `embeddings_cnn.py extrair` sem `--pesos-h5` usa os pesos normais do torchvision. Os números devem mudar pouco.

## Como rodar

```
pip install torch torchvision numpy scikit-learn matplotlib h5py
python embeddings_cnn.py extrair dataset\images -o resultados\embeddings_resnet50.npz
python embeddings_cnn.py avaliar resultados\embeddings_resnet50.npz resultados\baseline_python.csv dataset\metadata.csv -o resultados
python consolidar_resultados.py dataset\metadata.csv resultados --cnn ResNet-50=predicoes_resnet.csv --cnn MobileNetV3-L=predicoes_mobilenet.csv --cnn EfficientNet-B0=predicoes_efficientnet.csv
```

O primeiro passo pode ser pulado: o `embeddings_resnet50.npz` já está em `resultados/`. Use o metadata original (com MS da CR17 vazio) para manter a CR17 fora da biomassa, ou o corrigido se decidir incluí-la, e nesse caso rode tudo de novo, baseline incluído.

## Saídas

- `resultados_finais.txt` e `resultados_finais.csv`: tabela com todos os modelos, métricas por foto e por parcela, média e desvio padrão nas 10 repetições.
- `resultados_finais_boxplot_altura.png` e `resultados_finais_boxplot_biomassa.png`: RMSE por fold (50 folds) de todos os modelos.
- `predicoes_embeddings.csv`: predições por imagem dos experimentos extras.
