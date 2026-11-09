# Shelf Intelligence AI

**Detecção de espaços vazios em gôndolas de varejo com visão computacional e YOLOv8**

[streamlit demo](assets/results/streamlit_demo.png)

O **Shelf Intelligence AI** é um projeto de portfólio que utiliza detecção de objetos para identificar gôndolas (*shelf*) e possíveis espaços vazios (*void*) em imagens de prateleiras de lojas. O modelo foi treinado com **YOLOv8n** e disponibilizado em uma aplicação interativa desenvolvida com **Streamlit**.

> **Escopo:** protótipo educacional de visão computacional. As detecções não representam uma medição validada de ruptura de estoque nem substituem uma inspeção operacional.

## 1. Problema e objetivo

A identificação visual de espaços vazios em prateleiras pode apoiar análises de exposição de produtos e reposição. Neste projeto, o objetivo foi construir um fluxo de ponta a ponta: preparar um dataset anotado, treinar um detector, avaliar resultados, investigar erros e disponibilizar uma interface de inferência.

O foco não foi maximizar métricas a qualquer custo, mas demonstrar uma implementação funcional e uma avaliação crítica de seus limites.

## 2. Demonstração da aplicação

A interface Streamlit permite:

- Enviar uma imagem `.jpg`, `.jpeg` ou `.png`;
- Visualizar a imagem original e as detecções lado a lado;
- Ajustar o *confidence threshold* (valor inicial: **0,25**);
- Consultar a quantidade de `shelf` e `void` detectados;
- Baixar a imagem anotada.

## 3. Dataset

Foi utilizado o dataset público [Empty Spaces Detection — Roboflow Universe](https://universe.roboflow.com/bilalnaeem-d2fox/empty-spaces-detection/dataset/1), versão 1, disponibilizado sob licença **CC BY 4.0**.

| Conjunto | Imagens | Anotações `shelf` | Anotações `void` |
|---|---:|---:|---:|
| Treino | 262 | 258 | 1.272 |
| Validação | 93 | 94 | 469 |
| Teste | 110 | 110 | 573 |
| **Total** | **465** | **462** | **2.314** |

O dataset contém duas classes: `shelf` (estrutura da gôndola) e `void` (região anotada como espaço vazio). As imagens e anotações seguem o formato de detecção YOLO.

**Observação de qualidade:** durante a inspeção visual, foram encontrados exemplos de regiões aparentemente vazias sem anotação correspondente. Isso pode fazer com que algumas detecções plausíveis sejam contabilizadas como falsos positivos. Também há erros reais do detector; a revisão visual não quantificou a proporção de cada causa.

## 4. Abordagem técnica

**Tecnologias:** Python, Ultralytics YOLOv8, PyTorch, OpenCV, Pandas, NumPy, Matplotlib e Streamlit.

O modelo escolhido foi o **YOLOv8n**, inicializado com pesos pré-treinados (`yolov8n.pt`) e ajustado para as duas classes do dataset.

### Configuração do treinamento final

| Parâmetro | Valor |
|---|---|
| Modelo | YOLOv8n |
| Épocas | 20 |
| Resolução | 640 × 640 |
| Batch size | 4 |
| Dispositivo | CPU |
| Workers | 0 |
| Pesos selecionados | `best.pt` |

O projeto foi desenvolvido em um ambiente sem GPU CUDA. Foram realizados testes exploratórios com resolução menor antes do treinamento final a 640 × 640.

## 5. Resultados no conjunto de teste

A avaliação final foi realizada no **conjunto de teste (110 imagens)** com a rotina de validação do Ultralytics.

| Classe | Precision | Recall | mAP@50 | mAP@50–95 |
|---|---:|---:|---:|---:|
| `shelf` | 0,9869 | 0,9909 | 0,9895 | 0,8481 |
| `void` | 0,6691 | 0,5672 | 0,6549 | 0,3527 |
| **Geral (média das classes)** | **0,8280** | **0,7790** | **0,8222** | **0,6004** |

O detector apresentou desempenho superior na classe `shelf`. A classe `void` é mais desafiadora, devido à diversidade de tamanhos, formatos, iluminação, oclusões e possíveis inconsistências nas anotações.

**Importante:** as métricas acima são da avaliação oficial do Ultralytics. Elas não devem ser confundidas com as métricas calculadas no experimento personalizado de seleção de threshold descrito a seguir.

## 6. Seleção do confidence threshold

O threshold foi investigado **somente no conjunto de validação**, usando associação individual de caixas previstas e anotadas com **IoU ≥ 0,50**. Para cada valor de confiança, foram calculados TP, FP, FN, precision, recall e F1 da classe `void`.

| Threshold | Precision | Recall | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|
| 0,10 | 0,3579 | 0,7974 | 0,4941 | 374 | 671 | 95 |
| **0,25** | **0,5961** | **0,6546** | **0,6240** | **307** | **208** | **162** |
| 0,35 | 0,7092 | 0,5565 | 0,6237 | 261 | 107 | 208 |
| 0,50 | 0,8750 | 0,3582 | 0,5083 | 168 | 24 | 301 |

O valor **0,25** foi escolhido como configuração inicial por apresentar o maior F1 entre os thresholds avaliados. O threshold **0,35** apresentou F1 muito próximo, com menos falsos positivos, ilustrando que a escolha depende da prioridade operacional.

Na aplicação, o usuário pode alterar o threshold e observar como a quantidade de detecções muda. Por exemplo, em uma das imagens demonstrativas, a contagem caiu de **15** para **8** espaços vazios ao elevar o threshold de **0,25** para **0,50**. Esse é um exemplo visual, não uma estimativa de desempenho global.

> O pareamento utilizado nesse experimento é uma avaliação personalizada com IoU fixo e não reproduz integralmente o cálculo de mAP do Ultralytics.

## 7. Análise de erros e limitações

A inspeção visual das predições permitiu identificar diferentes situações:

1. **Possíveis anotações ausentes:** espaços aparentemente vazios não marcados no ground truth;
2. **Falsos positivos reais:** regiões com produtos ou estruturas interpretadas como espaços vazios;
3. **Falsos negativos:** espaços anotados que não foram detectados;
4. **Localização imprecisa:** caixas próximas à região correta, mas com sobreposição insuficiente;
5. **Generalização limitada:** dataset relativamente pequeno e sem validação em novas lojas ou condições operacionais.

Na análise personalizada com threshold 0,25 no conjunto de validação, foram contabilizados **307 TP, 208 FP e 162 FN** para `void`. Esses números refletem a correspondência com as anotações existentes, não uma auditoria humana de todos os espaços realmente vazios.

### Possíveis melhorias futuras

- Auditar e corrigir anotações incompletas;
- Ampliar a diversidade de lojas, ângulos e iluminação;
- Comparar arquiteturas e tamanhos de modelos;
- Avaliar desempenho em imagens externas ao dataset;
- Definir critérios de negócio para diferenciar espaço vazio de ruptura efetiva de estoque.

Essas melhorias foram registradas como evolução futura e **não fazem parte do escopo do protótipo atual**.

## 8. Estrutura do repositório

```text
shelf-intelligence-ai/
├── app/
│   └── streamlit_app.py
├── assets/
│   └── examples/
├── configs/
│   └── data.yaml
├── data/                      # Dataset local (ignorado pelo Git)
├── models/
│   └── best.pt                # Pesos locais (ignorados pelo Git)
├── src/
│   ├── common.py
│   ├── train.py
│   ├── predict.py
│   ├── evaluate.py
│   └── optimize_threshold.py
├── requirements.txt
├── .gitignore
└── README.md
```

Os diretórios `data/`, `others/` e os arquivos de pesos `.pt` estão fora do versionamento, conforme `.gitignore`. O diretório `others/` mantém experimentos exploratórios locais.

## 9. Como executar localmente

### Pré-requisitos

- Python 3.11 (versão utilizada no desenvolvimento);
- Dependências listadas em `requirements.txt`;
- Pesos treinados em `models/best.pt` para executar a aplicação.

Na raiz do repositório:

```bash
python -m venv .venv
```

Ative o ambiente virtual no PowerShell (Windows):

```powershell
.venv\Scripts\Activate.ps1
```

Instale as dependências:

```bash
python -m pip install -r requirements.txt
```

Coloque o arquivo `best.pt` em `models/best.pt` e inicie a interface:

```bash
python -m streamlit run app/streamlit_app.py
```

O Streamlit normalmente disponibiliza a aplicação em `http://localhost:8501`.

### Inferência por linha de comando

```bash
python src/predict.py
python src/predict.py --limit 0
python src/predict.py --conf 0.35
python src/predict.py --source "assets/examples/example_01.jpg"
```

### Reprodução de treinamento e avaliação

Para reproduzir os experimentos, baixe o dataset da fonte indicada e organize `train/`, `valid/`, `test/` e `data.yaml` em `data/`. A configuração do dataset é resolvida localmente pela função `get_dataset_config()` de `src/common.py`.

```bash
python src/train.py
python src/evaluate.py
python src/optimize_threshold.py
```

> O modelo usado na demonstração é o `best.pt` previamente treinado; executar `train.py` não é necessário para utilizar o Streamlit.

## 10. Principais aprendizados

O projeto permitiu aplicar conceitos de **transfer learning, detecção de objetos, métricas de localização (IoU/mAP), ajuste de confiança, inspeção de falsos positivos e falsos negativos e disponibilização de modelos em uma interface interativa**.

Um dos principais aprendizados foi perceber que **qualidade de anotação e qualidade do modelo são problemas diferentes**, mas ambos afetam as métricas observadas. A documentação dessas limitações faz parte do resultado técnico do projeto.

## 11. Créditos e licença dos dados

Dataset: **Empty Spaces Detection**, workspace `bilalnaeem-d2fox`, versão 1, Roboflow Universe. Licença informada pelo provedor: **CC BY 4.0**. Fonte: https://universe.roboflow.com/bilalnaeem-d2fox/empty-spaces-detection/dataset/1

As imagens e anotações do dataset devem manter atribuição adequada ao serem reutilizadas. A licença dos dados não determina automaticamente a licença do código deste repositório.
