# Baseline do app Pastagem, fora do Android

Este pacote isola a parte de cálculo do app Android "Pastagem" (o código Java que o Lukas recebeu) para rodar em qualquer computador, sem Android Studio, login, GPS ou Firebase.

## O que é o código Java

O zip é um projeto Android completo. São 25 arquivos `.java` e cerca de 5.900 linhas, mas quase tudo é tela, permissão de câmera, GPS, login e gravação no Firestore. Não existe código de treinamento nem dados de treino no projeto. O app só carrega modelos já treinados, com os pesos escritos direto no código.

O que interessa para a comparação está em quatro arquivos:

| Arquivo | O que faz |
|---|---|
| `AlturaActivity.java`, método `textura()` | Extrai os atributos de textura da foto e estima a altura |
| `BiomassaActivity.java`, método `previsaoBiomassa()` | Mesmos atributos, mais o GLI médio, e estima a biomassa |
| `MLPRegressionAltura.java` | Rede neural da altura, com pesos fixos |
| `MLPClassifier.java` | Rede neural da biomassa, com pesos fixos. Apesar do nome, é um regressor |

O restante (`Segmentacao`, `MPRIActivity`, login, adapters, banco local) não participa das estimativas de altura e biomassa.

## O que o app calcula

1. Converte cada pixel para cinza pela média simples de R, G e B e quantiza em 8 níveis.
2. Monta uma matriz de co-ocorrência 8x8 com o vizinho da direita e normaliza.
3. Tira dela entropia, homogeneidade e segundo momento angular. A entropia fica negativa, porque o app soma p·log2(p) sem trocar o sinal.
4. Calcula o desvio padrão do cinza, a porcentagem de pixels escuros (cinza até 30) e o GLI médio.
5. Altura: rede com entradas (entropia, homogeneidade), camadas ocultas de 3 e 50 neurônios com tanh. A saída é multiplicada por 8,6485 e somada a 21,125 para desfazer a padronização.
6. Biomassa: primeiro uma altura por regressão linear dos atributos de textura, depois uma rede com entradas (altura linear, GLI médio), camadas ocultas de 2 e 250 neurônios com tanh. A saída é multiplicada por 3916,23 e somada a 6428,87.

As constantes de padronização da saída indicam que os modelos foram treinados em dados com altura média de 21,1 cm e biomassa média de 6.429 kg/ha.

## O que tem neste pacote

- `pastagem_baseline.py`: o mesmo cálculo em Python. Só precisa de numpy e Pillow.
- `pesos_app.json`: pesos das duas redes e coeficientes da regressão, extraídos do Java por script. Nenhum número foi digitado à mão.
- `avaliar_baseline.py`: cruza a saída com o `metadata.csv` e calcula R², RMSE, MAE, MAPE, WI, correlação e viés.
- `referencia_java/`: os dois arquivos de rede originais do app e um programa de linha de comando (`PastagemRef.java`) com a lógica de `textura()` e `previsaoBiomassa()` copiada, trocando só a API de imagem do Android pela do Java padrão. Serve para auditar o port.
- `resultados/`: saída do Python e do Java nas 236 imagens do dataset do Lukas, e as métricas.
- `reajuste_baseline.py`, `graficos_reajuste.py` e `comparar_com_cnn.py`: reajuste da metodologia do artigo nos dados do Lukas, os boxplots e a comparação final com a CNN. Ver a seção "Reajuste da metodologia".

## Como rodar no Windows

No Prompt de Comando, dentro desta pasta:

```
pip install numpy pillow
python pastagem_baseline.py dataset\images -o resultados\baseline_python.csv
python avaliar_baseline.py resultados\baseline_python.csv dataset\metadata.csv -o resultados\metricas.txt
```

Leva em torno de um minuto para as 236 imagens. Para rodar a referência Java (opcional, precisa de JDK):

```
cd referencia_java
javac -encoding UTF-8 *.java
java PastagemRef ..\dataset\images ..\resultados\referencia_java.csv
```

O CSV de saída traz, por imagem, os seis atributos usados pelos modelos, as três estimativas (altura pela rede, altura linear, biomassa) e quatro índices extras que o app mostra em outras telas (MPRI, VARI, TGI e cobertura verde). Os extras não entram nos modelos do app e foram calculados em precisão dupla, então podem diferir do app nas últimas casas.

## Validação do port

O Python e a referência Java deram valores idênticos, bit a bit, nas 236 imagens e nas nove colunas calculadas pelos dois. Isso vale lendo os JPEG diretamente e também com as imagens convertidas para PNG. Para chegar nisso o Python reproduz dois detalhes do Java: a ordem das somas e o acumulador do desvio padrão em float de 32 bits.

O que não foi possível conferir é o app rodando num celular de verdade. A leitura do JPEG no Android pode diferir em um ou outro nível de cor.

## Resultado no dataset do Lukas, com os modelos do app como estão

| Estimativa | R² | RMSE | Viés | Desvio padrão do previsto | Desvio padrão do medido |
|---|---|---|---|---|---|
| Altura, rede do app | -3,03 | 10,9 cm | -9,5 cm | 0,15 cm | 5,45 cm |
| Altura, fórmula linear do app | -114,7 | 58,6 cm | -58,0 cm | 6,40 cm | 5,45 cm |
| Biomassa, rede do app | -29,6 | 2.500 kg/ha | +2.427 kg/ha | 544 kg/ha | 452 kg/ha |

Na prática as saídas são constantes. A rede da altura devolve entre 29,6 e 30,5 cm para todas as imagens. A fórmula linear dá altura negativa em 233 das 236. A rede da biomassa devolve 3.344,0 kg/ha em 215 das 236.

## Por que isso acontece

A rede da biomassa, como está no app, funciona como um degrau: devolve cerca de 3.344 kg/ha para qualquer altura linear abaixo de -3 e cerca de 8.135 kg/ha para qualquer altura acima de 5. Só varia de verdade com a entrada perto de zero, que é a faixa de uma variável padronizada (z-score). A rede da altura mostra o mesmo padrão: com os valores brutos de entropia e homogeneidade, duas das três unidades da primeira camada oculta ficam saturadas.

A leitura mais provável é que as redes foram treinadas com as entradas padronizadas (StandardScaler) e o app só desfaz a padronização da saída. O comentário no código menciona o StandardScaler apenas na saída, e a média e o desvio das entradas não estão em lugar nenhum do projeto. Isso é uma inferência a partir dos pesos, não algo que o código afirme.

O artigo (Franco et al., 2022, Evolutionary Intelligence) e a dissertação não mencionam padronização, mas as estatísticas que publicam sustentam essa leitura. Padronizando as entradas com as médias das Tabelas 5 e 6 do artigo e desvios aproximados pelos quartis, a rede da altura devolve 22,4 cm no ponto médio (a média de treino é 21,1) e cobre de 4 a 34 cm, e a da biomassa devolve 6.218 kg/ha no ponto médio (média de treino 6.429). Os desvios são aproximados, então isso serve como evidência e não como correção do app.

Mesmo com essa correção os modelos do app não servem para este dataset. As fotos do Lukas ficam a cerca de 4 desvios da média de treino em homogeneidade (0,57 a 0,75 contra 0,69 a 0,90). Com as entradas padronizadas a altura prevista cai para entre -3 e 3 cm e a biomassa fica com R² de -17,8.

Há um segundo problema, independente desse: os modelos foram ajustados para pastos com média de 6.429 kg/ha, e o dataset do Lukas vai de 239 a 2.141 kg/ha. Mesmo com as entradas corrigidas, seria uma extrapolação.

## Consequência para a comparação

Comparar a CNN com o app do jeito que ele está não é uma comparação de metodologias, porque o baseline devolve uma constante. Para comparar métodos, o caminho é usar os mesmos atributos manuais (as colunas do CSV) e reajustar os modelos da dissertação nos dados do Lukas, com os mesmos folds usados na CNN. Esse reajuste está feito e descrito na próxima seção.

## Reajuste da metodologia nos dados do Lukas

O `reajuste_baseline.py` refaz o método do artigo com os dados novos. Cada imagem é uma amostra, descrita pelos atributos do artigo: GLI, MPRI, VARI, TGI, homogeneidade, entropia, segundo momento angular e cobertura verde, mais a altura estimada no caso da biomassa. Os modelos são LASSO, SVR e MLP, com hiperparâmetros e seleção de atributos por Evolução Diferencial nas faixas da Tabela 4 do artigo, validação cruzada com 5 subconjuntos e as métricas R², WI, RMSE, MAE e MAPE.

O script roda três formas de validar, porque a forma muda muito o resultado:

- `publicado_imagem`: a busca escolhe o modelo olhando os mesmos folds em que o resultado é reportado, e os folds são sorteados por imagem. É o que o artigo parece ter feito.
- `publicado_parcela`: igual, mas todas as fotos de uma parcela ficam no mesmo fold.
- `aninhado_parcela`: folds por parcela, e a busca só enxerga o treino de cada fold. É a estimativa honesta e a que vale para comparar com a CNN.

Resultado na validação aninhada, com média e desvio padrão entre 10 repetições:

| Alvo | Modelo | R² | RMSE |
|---|---|---|---|
| Altura | LASSO | 0,41 (0,03) | 4,20 (0,09) cm |
| Altura | SVR | 0,37 (0,04) | 4,32 (0,14) cm |
| Altura | MLP | 0,38 (0,09) | 4,29 (0,29) cm |
| Altura | Média do treino | -0,02 (0,01) | 5,51 (0,03) cm |
| Altura | Média por cultivar e momento | 0,53 (0,03) | 3,74 (0,10) cm |
| Biomassa | LASSO | 0,07 (0,07) | 436 (16) kg/ha |
| Biomassa | SVR | 0,03 (0,10) | 445 (23) kg/ha |
| Biomassa | MLP | -0,12 (0,10) | 478 (20) kg/ha |
| Biomassa | Média do treino | -0,04 (0,02) | 461 (4) kg/ha |
| Biomassa | Média por cultivar | 0,04 (0,02) | 444 (4) kg/ha |

As duas últimas linhas de cada alvo são regras de referência que não olham a imagem: prever a média do treino, ou a média do treino para a mesma cultivar e o mesmo momento. As figuras `resultados/reajuste_boxplot_rmse_por_fold.png` e `resultados/reajuste_boxplot_cenarios.png` mostram a dispersão.

O que os números dizem:

- Na altura, os atributos manuais explicam cerca de 40% da variância, mas perdem para a regra que só conhece a cultivar e o momento. Boa parte da variação de altura neste dataset é a diferença entre M1 e M2.
- Na biomassa, os atributos manuais não preveem nada em parcelas que o modelo não viu. O erro é o mesmo de prever a média.
- Dividir os folds por imagem infla tudo. Com folds por imagem o SVR chega a R² de 0,64 e RMSE de 273 kg/ha na biomassa, e a R² de 0,74 e RMSE de 2,76 cm na altura. Cada parcela tem até 6 fotos com o mesmo valor de biomassa, então o modelo reconhece a parcela em vez de estimar a biomassa.

O rascunho do TCC do Lukas reporta, para a CNN, R² de 0,43 e RMSE de 4,11 cm na altura, e R² de 0,63 e RMSE de 286 kg/ha na biomassa. O número da biomassa coincide com o do baseline quando os folds são por imagem. Antes de qualquer conclusão é preciso saber como ele dividiu os folds e rodar a CNN no `resultados/folds_por_parcela.csv`.

Para a CNN mostrar ganho com folds por parcela, ela precisa ficar abaixo de 3,7 cm de RMSE na altura (a regra de cultivar e momento) e abaixo de uns 440 kg/ha na biomassa.

Escolhas minhas onde o artigo não diz nada:

- Entradas e alvo padronizados com a média e o desvio do treino de cada fold.
- MLP com ReLU, como no artigo, otimizador L-BFGS e 200 iterações.
- Evolução Diferencial rand/1/bin com F = 0,8 e CR = 0,9, população de 15 e 12 gerações (30 e 36 para o LASSO, que é barato). O critério da busca é o RMSE sobre todas as imagens fora do fold.
- No LASSO, a faixa de alpha do artigo (até 20) é percorrida em escala logarítmica. Em escala linear a busca ficava presa em modelos constantes em parte das repetições.
- Cobertura verde pela regra do código do app (R < G e B < G). O texto descreve HSV, superpixel e limiar no canal H, mas não dá os limiares.
- Altura estimada por regressão linear sobre entropia, homogeneidade, desvio padrão do cinza e área escura, ajustada só no treino. É a mesma forma da fórmula que está no app.
- Oito atributos candidatos para a altura, como no artigo. A dissertação usa quatro.

Para rodar (leva perto de duas horas com 2 núcleos, quase tudo na MLP):

```
pip install numpy scipy scikit-learn matplotlib
python reajuste_baseline.py resultados\baseline_python.csv dataset\metadata.csv -o resultados
python graficos_reajuste.py resultados
```

Arquivos gerados em `resultados/`: `folds_por_parcela.csv`, `reajuste_resumo.txt`, `reajuste_metricas_por_repeticao.csv`, `reajuste_metricas_por_fold.csv`, `reajuste_configuracoes.csv` (atributos e hiperparâmetros escolhidos) e `reajuste_predicoes_aninhado.csv`.

### Como a CNN entra na comparação

No `folds_por_parcela.csv`, cada coluna `rep00` a `rep29` é uma divisão em 5 folds. Para o fold k de uma repetição, treinar com as imagens em que o valor é diferente de k e prever as imagens em que é igual a k. Ajuste de hiperparâmetros e parada antecipada só com o treino. O baseline aninhado usa `rep00` a `rep09`.

As predições da CNN devem vir em CSV com ponto e vírgula e as colunas `arquivo;rep;fold;alvo;previsto`, com `alvo` igual a `altura` ou `biomassa`. É o mesmo formato do `reajuste_predicoes_aninhado.csv`, sem a coluna do modelo.

Com esse arquivo pronto, a comparação sai com um comando:

```
python comparar_com_cnn.py predicoes_cnn.csv dataset\metadata.csv -o resultados
```

Ele confere se os folds batem com o `folds_por_parcela.csv`, calcula as métricas da CNN, do baseline e das regras de referência nas mesmas repetições, mostra a diferença de RMSE fold a fold e grava `comparacao_resumo.txt`, `comparacao_por_fold.csv` e um boxplot por alvo. Não é preciso rodar o `reajuste_baseline.py` de novo, as predições do baseline já estão em `resultados/`.

## Detalhes do app que afetam a reprodução

- Redução de imagem. O app reduz a foto por um fator que depende do tempo que o celular levou para copiar a imagem na memória. Não é reproduzível. O padrão aqui é não reduzir, e `--dividir` aplica um fator fixo. Testei fatores de 1,5 a 4 e as saídas continuam praticamente constantes.
- Orientação. O app não gira a imagem pelo EXIF, e a co-ocorrência usa só o vizinho horizontal. O Python faz igual. As imagens do dataset não têm marca de orientação.
- GLI médio. Pixels totalmente pretos não entram na soma, mas a média divide por todos os pixels.

## Pontos do dataset para o Lukas conferir

- `metadata.csv`, parcela CR17: a coluna de K2O tem 560,3 e a de MS está vazia. As doses de K2O das outras parcelas são 0, 50, 100, 200 e 400, então parece que o valor de MS caiu na coluna errada. A avaliação ignora a CR17 na biomassa.
- A MS é a mesma nas linhas M1 e M2 de cada parcela. A biomassa foi medida uma vez só, então a mesma referência vale para as fotos dos dois momentos.
- Dois arquivos fogem do padrão de nome: `CR0103_M1.jpeg` (tratei como CR01, repetição 3) e `Cr16R2_M1.jpeg` (minúscula).
- Três combinações têm menos de 3 fotos: CR11 M2, TR02 M2 e TR11 M1.
- As imagens têm dois tamanhos: 217 com 720x1280 e 19 com 900x1600. Atributos de textura mudam com a resolução.
