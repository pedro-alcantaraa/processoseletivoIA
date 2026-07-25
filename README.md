## 📝 Relatório do Candidato

👤 **Nome:** Pedro Henrique Rodrigues De Sá Alcantara
**Curso:** Ciência da Computação
**Instituição:** Faculdade de Petrolina (Facape)

---

### 1️⃣ Resumo da Abordagem

Neste projeto foi utilizado o modelo pré-treinado **YOLO11n**, conforme solicitado no desafio. Essa variante foi escolhida por ser leve e adequada para aplicações de Edge AI, permitindo o treinamento e a execução em CPU sem necessidade de uma GPU dedicada.

O treinamento foi realizado utilizando o dataset disponibilizado no projeto (`dataset/data.yaml`), mantendo a configuração de **20 épocas**, **imgsz=640**, **batch=8** e **device="cpu"**.

A escolha de 20 épocas foi suficiente para realizar o fine-tuning do modelo, já que o YOLO11n parte de pesos previamente treinados e precisa apenas se adaptar ao novo conjunto de dados. O tamanho de imagem foi mantido em 640 pixels por ser o padrão utilizado pela biblioteca Ultralytics, enquanto o batch de 8 foi adotado para reduzir o consumo de memória durante o treinamento em CPU.

Durante o desenvolvimento, o objetivo principal foi implementar corretamente todo o pipeline solicitado pelo desafio:

- Treinamento do modelo;
- Geração do arquivo `model.pt`;
- Exportação para TensorFlow Lite (`model.tflite`);
- Execução de inferência utilizando o modelo otimizado.

Não foram aplicadas técnicas específicas para tratar o desbalanceamento das classes, como aumento de dados direcionado ou ponderação da função de perda. Essa decisão foi tomada porque o foco do desafio era validar o funcionamento completo do pipeline, e não maximizar a precisão da classe minoritária.

Ao final do desenvolvimento, os scripts foram revisados para melhorar a organização do código, utilizando constantes para valores fixos e separando melhor as responsabilidades de cada etapa.

---

### 2️⃣ Bibliotecas Utilizadas

| Biblioteca | Versão | Papel no projeto |
|---|---|---|
| Ultralytics | 8.4.0 | Framework principal — carregamento do YOLO11n, treinamento, validação e exportação |
| PyTorch | 2.13.0 (build CPU) | Backend de treinamento do modelo `.pt` |
| TensorFlow | 2.19.0 | Usado internamente pela Ultralytics na conversão para TensorFlow Lite |
| Python | 3.11.0 | Linguagem de execução dos scripts |

As versões utilizadas foram instaladas automaticamente a partir do arquivo `requirements.txt`, que especifica a dependência `ultralytics>=8.4`. Durante a exportação para TensorFlow Lite, a própria biblioteca instalou automaticamente algumas dependências adicionais necessárias para concluir o processo.

---

### 3️⃣ Técnica de Otimização do Modelo

A exportação é feita com `model.export(format="tflite", imgsz=640)`. Olhando
o log de execução, dá pra ver que a Ultralytics faz essa conversão em duas
etapas internas:

1. **PyTorch → TensorFlow SavedModel**: o modelo `.pt` é convertido primeiro
   para um SavedModel do TensorFlow (25,6 MB) — passo necessário porque o
   TensorFlow Lite não converte diretamente de um modelo PyTorch, precisa
   passar pelo grafo de computação do TensorFlow antes.
2. **SavedModel → TensorFlow Lite**: a partir do SavedModel, é gerado o
   arquivo `model_float32.tflite`, já no formato otimizado para inferência em
   dispositivos de borda, mantendo precisão float32 (sem quantização
   adicional para int8, que reduziria mais o tamanho às custas de precisão).

O motivo de usar TFLite em vez de continuar com o `.pt`: o formato é
desenhado especificamente para inferência (não treino) em ambientes com
recursos limitados, removendo estruturas que só fazem sentido durante o
treinamento e simplificando a API de execução (`Interpreter`) — o cenário
real de um dispositivo embarcado rodando esse modelo sem PyTorch instalado.

Na versão atual do `optimize_model.py`, a cópia do arquivo final para
`model.tflite` passou a usar diretamente o caminho retornado pela própria
função `model.export()`, em vez de um caminho fixo escrito manualmente. Isso
resolveu uma fragilidade da primeira versão: o nome/local exato do arquivo
gerado pode variar entre versões da Ultralytics, então depender do valor
retornado pela função é mais seguro do que presumir a estrutura de pastas.

---

### 4️⃣ Resultados Obtidos

**Validação (conjunto `dataset/images/val/`, 170 imagens, 726 instâncias):**

| Classe | Imagens | Instâncias | Precision | Recall | mAP50 | mAP50-95 |
|---|---|---|---|---|---|---|
| **Todas as classes** | 170 | 726 | 0,698 | 0,714 | **0,690** | 0,469 |
| with_mask | 149 | 593 | 0,888 | 0,949 | 0,966 | 0,680 |
| without_mask | 57 | 114 | 0,745 | 0,772 | 0,802 | 0,516 |
| mask_weared_incorrect | 15 | 19 | 0,461 | 0,421 | 0,302 | 0,211 |

**Tamanho dos artefatos entregues:**
- `model.pt`: 5,3 MB
- `model.tflite`: 10,12 MB (o tamanho maior que o `.pt`, mesmo sendo o
  formato de edge, se explica pela ausência de quantização — o TFLite em
  float32 mantém a mesma precisão numérica dos pesos, sem a compressão que
  int8 traria)

**Velocidade (CPU, Intel Core i3-10110U):** ~3,0 ms de pré-processamento,
~130,4 ms de inferência e ~3,3 ms de pós-processamento por imagem no
conjunto de validação.

---

### 5️⃣ Comentários Adicionais

O desbalanceamento do dataset ficou bem visível nos resultados: with_mask
(classe com mais exemplos) chegou a 0,966 de mAP50, enquanto
mask_weared_incorrect (poucas amostras) ficou em 0,302. Faz sentido, já que
com menos exemplos o modelo tem menos chance de aprender bem as
características dessa classe.

O pipeline completo funcionou de ponta a ponta sem problemas: treino,
exportação para TFLite e inferência rodaram sem erros.

Uma melhoria futura seria tentar reduzir esse desbalanceamento, com data
augmentation focado na classe minoritária ou ajustando os pesos da loss por
classe, pra tentar melhorar o mask_weared_incorrect sem afetar as outras.

---

### 6️⃣ Exemplo de Inferência

Imagem                              Detecções  Detalhes
----------------------------------------------------------------------
maksssksksss105.jpg                         9  [9x with_mask]
maksssksksss107.jpg                         1  [1x with_mask]
maksssksksss11.jpg                         24  [23x with_mask, 1x mask_weared_incorrect]
maksssksksss113.jpg                         4  [4x with_mask]
maksssksksss12.jpg                         13  [11x with_mask, 2x without_mask]
----------------------------------------------------------------------
TOTAL                                      51

Olhando os números por imagem, o padrão bate com o que a validação já
mostrava: `mask_weared_incorrect` aparece só uma vez em todas as 51 detecções
das 5 amostras, e justamente na imagem com mais rostos (`maksssksksss11.jpg`,
24 detecções) — coerente com o recall baixo (0,421) dessa classe, que faz
sentido ela aparecer raramente mesmo em cenas com muitas pessoas. As imagens
mais simples (`maksssksksss105.jpg`, `maksssksksss107.jpg`) tiveram só
detecções de `with_mask`, sem nenhuma classe confundida entre si.

De forma geral, o modelo apresentou um funcionamento consistente, realizando corretamente a detecção dos rostos e a classificação da maioria das máscaras presentes nas imagens utilizadas para teste.

## 📄 Créditos do Dataset

Face Mask Detection Dataset — [Kaggle: andrewmvd/face-mask-detection](https://www.kaggle.com/datasets/andrewmvd/face-mask-detection), licença CC0 1.0 (domínio público).