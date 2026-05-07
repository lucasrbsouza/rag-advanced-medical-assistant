# RAG Advanced Medical Assistant

**Laboratório 09 — Arquitetura RAG Avançada (HNSW, HyDE e Cross-Encoders)**
**Disciplina:** Processamento de Linguagem Natural — 7º Período
**Autor:** José Lucas Silva Souza

---

## Contexto do Problema

Quando um paciente digita uma query coloquial como *"dor de cabeça latejante e luz incomodando"*, a similaridade de cosseno pura falha: o espaço vetorial das perguntas informais é geometricamente distante do jargão técnico dos manuais médicos (*"cefaleia pulsátil e fotofobia"*). Este pipeline intercepta a pergunta, usa um LLM para gerar uma ponte semântica (HyDE), busca rapidamente em um grafo hierárquico (HNSW) e refina a precisão com atenção cruzada (Cross-Encoder) antes de entregar os documentos ao LLM gerador.

---

## Arquitetura do Pipeline

```text
Query Coloquial do Paciente
        │
        ▼  [Passo 2] HyDE — Query Transformation
   LLM gera Documento Hipotético técnico
   (llama-3.1-8b via Groq  OU  flan-t5-base local)
        │
        ▼  Embedding local
   (paraphrase-multilingual-MiniLM-L12-v2, dim=384)
   Vetor normalizado — nova âncora geométrica
        │
        ▼  [Passo 3] Bi-Encoder — Busca HNSW (FAISS)
   Top-10 Candidatos  ←  funil largo, O(log N)
        │
        ▼  [Passo 4] Cross-Encoder (ms-marco-MiniLM-L-6-v2)
   Top-3 Documentos Finais  ←  filtro fino, atenção cruzada
        │
        ▼
   Contexto injetado no LLM Gerador
```

---

## Passo 1 — Construção e Indexação do Grafo HNSW

O corpus contém **25 fragmentos de manuais médicos** em português, cobrindo Neurologia, Cardiologia, Endocrinologia, Pneumologia, Reumatologia, entre outras especialidades. Os textos foram gerados com terminologia clínica técnica precisa.

Os fragmentos são convertidos em vetores densos pelo modelo local `paraphrase-multilingual-MiniLM-L12-v2` (sentence-transformers, dimensão 384, suporte nativo ao português) e indexados em um `faiss.IndexHNSWFlat` com produto interno — equivalente a similaridade de cosseno após normalização dos vetores.

```python
index = faiss.IndexHNSWFlat(384, M=16, faiss.METRIC_INNER_PRODUCT)
index.hnsw.efConstruction = 200
index.hnsw.efSearch = 50
```

### Tarefa Analítica: Hiperparâmetros HNSW (M e ef_construction) vs RAM em KNN Exato

#### KNN Exato — Baseline

Na busca K-Nearest Neighbors exata, todos os N vetores ficam em memória e a busca compara a query com **todos** eles:

```text
RAM (KNN exato) = N × D × 4 bytes

Exemplo: N=1.000.000 vetores, D=384
  → 1.000.000 × 384 × 4 = 1.536 MB ≈ 1,5 GB
  Tempo de busca: O(N × D) — escala linearmente, inviável para N grande.
```

#### HNSW — Grafo Hierárquico Multicamada

O HNSW constrói um grafo em camadas: a camada superior tem poucos nós altamente conectados (navegação grossa); as camadas inferiores têm todos os nós com conexões locais (refinamento fino). A busca entra pelo topo e desce em tempo **O(log N)**.

A memória tem dois componentes:

```text
RAM (HNSW) = vetores + overhead do grafo

  Vetores:  N × D × 4 bytes          (igual ao KNN)
  Grafo:    N × M × n_camadas × 4 bytes

Exemplo: N=1.000.000, D=384, M=16, n_camadas≈20
  Vetores:  1.536 MB
  Grafo:    1.000.000 × 16 × 20 × 4 ≈ 1.280 MB
  Total:    ≈ 2.816 MB ≈ 2,8 GB
```

#### Impacto do hiperparâmetro M

`M` define quantas **arestas bidirecionais** cada nó mantém em cada camada do grafo.

| M | Overhead de RAM | Recall@10 | Velocidade de busca |
| --- | --------------- | --------- | ------------------- |
| 8 | +10% | ~95% | Mais rápida |
| **16 (usado)** | **+83%** | **~98%** | **Balanceada** |
| 32 | +166% | ~99,5% | Mais lenta por nó |
| 64 | +332% | ~99,9% | Próxima ao KNN |

M maior → mais conexões → grafo mais denso → mais RAM → maior recall mas busca mais lenta por nó (mais vizinhos a avaliar em cada passo de navegação).

#### Impacto do hiperparâmetro ef_construction

`ef_construction` define o tamanho da **lista de candidatos durante a construção** do grafo. Ele **não afeta o consumo de RAM em produção** — apenas o tempo de build e a qualidade final do grafo:

| ef_construction | Tempo de build | Qualidade do grafo | RAM em produção |
| --- | --- | --- | --- |
| 40 | Rápido | Menor precisão | Igual |
| **200 (usado)** | **Moderado** | **Alta precisão** | **Igual** |
| 800 | Lento | Máxima precisão | Igual |

ef_construction alto garante que cada nó inserido encontrou os melhores vizinhos possíveis, resultando em um grafo de maior qualidade — mas esse custo é pago uma única vez no build, não nas buscas.

#### Comparação Final: HNSW vs KNN Exato

```text
┌──────────────────┬─────────────┬──────────────────┐
│ Métrica          │  KNN Exato  │  HNSW (M=16)     │
├──────────────────┼─────────────┼──────────────────┤
│ RAM base         │  1,5 GB     │  1,5 GB          │
│ Overhead grafo   │  0          │  ~1,3 GB         │
│ RAM total        │  1,5 GB     │  ~2,8 GB         │
│ Tempo de busca   │  O(N×D)     │  O(log N)        │
│ Recall@10        │  100%       │  ~98%            │
│ Escalável?       │  Não        │  Sim             │
└──────────────────┴─────────────┴──────────────────┘
```

**Conclusão:** HNSW usa ~83% a mais de RAM que o armazenamento puro dos vetores (overhead do grafo), mas compra velocidade de busca 100–1000× superior e escalabilidade logarítmica. Para N > 100k documentos, KNN exato torna-se inviável; HNSW é o padrão da indústria em bancos de dados vetoriais de produção.

---

## Passo 2 — HyDE (Hypothetical Document Embeddings)

**Problema:** A query coloquial `"dor de cabeça latejante e luz incomodando"` vive em uma região do espaço vetorial distante dos documentos técnicos `"cefaleia pulsátil e fotofobia"`. Similaridade de cosseno direta falha.

**Solução HyDE:** Em vez de vetorizar a query diretamente, o LLM é instruído a *alucinar* um trecho de manual médico técnico que responderia à pergunta. Esse Documento Hipotético usa a mesma terminologia dos documentos reais, posicionando o vetor de busca na região correta do espaço vetorial:

```text
Query coloquial  →  embedding  →  região "leiga"   ← longe dos manuais
                                                           ↕ gap semântico
Doc. Hipotético  →  embedding  →  região "técnica" ← próximo aos manuais ✓
```

O sistema detecta automaticamente o modo de LLM disponível:

- **Com `GROQ_API_KEY`**: usa `llama-3.1-8b-instant` via Groq (melhor qualidade)
- **Sem chave**: usa `google/flan-t5-base` local (zero configuração)

---

## Passo 3 — Bi-Encoder + HNSW (Funil Largo)

O vetor do Documento Hipotético é usado como query no índice HNSW. O FAISS retorna os **Top-10** documentos mais próximos por similaridade de cosseno em tempo O(log N). Essa etapa prioriza **recall** — não perder documentos relevantes — formando o funil largo para a próxima etapa.

Os resultados são impressos no console com ID, categoria, título e score de cosseno.

---

## Passo 4 — Cross-Encoder (Filtro Fino)

O Cross-Encoder `cross-encoder/ms-marco-MiniLM-L-6-v2` recebe cada par `(query_original, documento)` no formato `[CLS] Query [SEP] Documento` e computa um score de relevância com **atenção cruzada bidirecional**. Diferente do Bi-Encoder (vetores independentes), o Cross-Encoder enxerga query e documento simultaneamente, capturando dependências contextuais mais ricas.

Os 10 candidatos são reordenados pelos novos scores. Os **Top-3** finais são impressos com seus ranks originais do Bi-Encoder, evidenciando as mudanças de posição, e seriam injetados no contexto do LLM gerador.

---

## Como Executar

### Pré-requisito: Python 3.10+

Verifique sua versão:

```bash
python --version
```

---

### Opção 1 — Executar sem API key (recomendado para teste rápido)

Todos os modelos rodam localmente. Nenhuma conta ou chave necessária.

#### 1. Clone o repositório

```bash
git clone https://github.com/lucasrbsouza/rag-advanced-medical-assistant.git
cd rag-advanced-medical-assistant
```

#### 2. Crie e ative um ambiente virtual

```bash
python -m venv .venv
source .venv/bin/activate        # Linux/Mac
# .venv\Scripts\activate         # Windows
```

#### 3. Instale as dependências

```bash
pip install -r requirements.txt
```

#### 4. Execute o pipeline

```bash
python main.py
```

Na primeira execução os modelos são baixados automaticamente do HuggingFace (~1,5 GB total):

- Embeddings: `paraphrase-multilingual-MiniLM-L12-v2` (~450 MB)
- HyDE LLM: `google/flan-t5-base` (~990 MB)
- Cross-Encoder: `ms-marco-MiniLM-L-6-v2` (~91 MB)

O índice HNSW também é construído e salvo em `data/` — execuções seguintes carregam do disco e iniciam em segundos.

---

### Opção 2 — Executar com Groq API (maior qualidade no HyDE)

O HyDE usa `llama-3.1-8b-instant` em vez do flan-t5 local, gerando Documentos Hipotéticos de qualidade significativamente superior. A API é **gratuita** (6.000 requisições/dia, sem cartão de crédito).

Siga os passos 1–3 da Opção 1 e continue:

#### 4. Obtenha uma API key gratuita

Acesse [console.groq.com](https://console.groq.com) → crie uma conta → **API Keys** → **Create API Key**.

#### 5. Configure a chave

```bash
cp .env.example .env
```

Edite o arquivo `.env` e insira sua chave:

```text
GROQ_API_KEY=gsk_...sua-chave-aqui...
```

#### 6. Execute o pipeline

```bash
python main.py
```

O sistema detecta automaticamente a presença da `GROQ_API_KEY` e usa o Groq. Sem a chave, cai para o modelo local sem nenhuma alteração necessária no código.

---

## Estrutura do Projeto

```bash
.
├── main.py                  # Orquestração do pipeline completo
├── requirements.txt         # Dependências Python
├── .env.example             # Template de configuração
├── data/
│   ├── medical_corpus.json  # 25 fragmentos de manuais médicos (corpus)
│   ├── hnsw_index.faiss     # Índice HNSW (gerado na primeira execução)
│   └── documents.pkl        # Cache dos documentos indexados
└── src/
    ├── embedder.py          # Embeddings locais via sentence-transformers
    ├── indexer.py           # Passo 1: construção do índice HNSW com FAISS
    ├── hyde.py              # Passo 2: geração do Documento Hipotético (HyDE)
    ├── retriever.py         # Passo 3: busca Top-10 via Bi-Encoder no HNSW
    └── reranker.py          # Passo 4: re-ranking Cross-Encoder → Top-3
```

---

## Declaração de Uso de IA

Partes deste laboratório foram geradas/complementadas com IA, revisadas e validadas por **José Lucas Silva Souza**.
