# RAG Advanced Medical Assistant

Pipeline de Retrieval-Augmented Generation (RAG) de nível de produção para busca em manuais médicos. Transforma queries coloquiais de pacientes em buscas precisas sobre terminologia clínica técnica, combinando três técnicas avançadas: **HNSW**, **HyDE** e **Cross-Encoder**.

## Arquitetura do Pipeline

```
Query Coloquial do Paciente
        │
        ▼  [Passo 2] HyDE
   LLM (gemini-2.0-flash) gera Documento Hipotético técnico
        │
        ▼  embed (paraphrase-multilingual-MiniLM-L12-v2, local)
   Vetor normalizado do Doc. Hipotético
        │
        ▼  [Passo 3] Bi-Encoder — HNSW (FAISS)
   Top-10 Candidatos (funil largo)
        │
        ▼  [Passo 4] Cross-Encoder (ms-marco-MiniLM-L-6-v2)
   Top-3 Documentos Finais
        │
        ▼
   Contexto injetado no LLM Gerador
```

---

## Passo 1 — Construção e Indexação do Grafo HNSW

O corpus contém **25 fragmentos de manuais médicos** cobrindo especialidades como Neurologia, Cardiologia, Endocrinologia, Pneumologia e outras. Os textos utilizam terminologia clínica técnica (ex: "cefaleia pulsátil", "fotofobia", "fisiopatologia trigeminovascular").

Os fragmentos são convertidos em vetores densos com o modelo `paraphrase-multilingual-MiniLM-L12-v2` (HuggingFace/sentence-transformers, dimensão 384, suporte nativo ao português) e indexados em um `faiss.IndexHNSWFlat` com similaridade de cosseno (produto interno de vetores normalizados). O modelo roda localmente, sem necessidade de API key.

### Análise: Hiperparâmetros HNSW (M e ef_construction) vs RAM em KNN Exato

#### KNN Exato — Baseline de Memória

Na busca KNN exata, todos os N vetores de dimensão D são armazenados em uma matriz densa. A busca compara a query com **todos** os vetores:

```
RAM (KNN exato) = N × D × 4 bytes

Exemplo: N=1.000.000 vetores, D=384 (paraphrase-multilingual-MiniLM-L12-v2)
  → 1.000.000 × 384 × 4 = 1.536 MB ≈ 1,5 GB
  Tempo de busca: O(N×D) — linear, lento para N grande.
```

#### HNSW — Estrutura em Grafo Hierárquico

O HNSW (Hierarchical Navigable Small World) constrói um grafo multicamada. Na camada mais alta há poucos nós altamente conectados (busca grossa); nas camadas inferiores, todos os nós com conexões locais (refinamento). A busca navega do topo ao fundo em tempo **O(log N)**.

A memória do HNSW tem dois componentes:

```
1. Armazenamento dos vetores: N × D × 4 bytes  (igual ao KNN)
2. Overhead do grafo:         N × M × camadas × 4 bytes

RAM (HNSW) ≈ N × (D + M × log(N)) × 4 bytes

Exemplo: N=1.000.000, D=384, M=16
  Vetores:  1.536 MB
  Grafo:    1.000.000 × 16 × 20 camadas × 4 ≈ 1.280 MB
  Total:    ≈ 2.816 MB ≈ 2,8 GB  (+83% sobre KNN exato)
```

#### Impacto de M

| M | Arestas por nó | Overhead RAM | Recall@10 | Velocidade de Busca |
|---|----------------|--------------|-----------|---------------------|
| 8 | 8 × layers | Baixo (+10%) | ~95% | Mais rápida |
| **16** | **16 × layers** | **Médio (+20%)** | **~98%** | **Balanceada** |
| 32 | 32 × layers | Alto (+40%) | ~99.5% | Mais lenta por nó |
| 64 | 64 × layers | Muito alto (+80%) | ~99.9% | Próxima ao KNN |

**M** controla quantas arestas bidirecionais cada nó mantém em cada camada. Valores maiores aumentam o recall (menos falsos negativos) mas elevam o uso de RAM proporcionalmente e aumentam o tempo de busca por nó (mais vizinhos a avaliar).

#### Impacto de ef_construction

**ef_construction** define o tamanho da lista de candidatos durante a construção do grafo. Ele **não afeta o uso de RAM em produção** — apenas o tempo e a qualidade da fase de build:

| ef_construction | Tempo de Build | Qualidade do Grafo | RAM em uso |
|----------------|----------------|---------------------|------------|
| 40 | Rápido | Menor precisão | Igual |
| **200** | **Moderado** | **Alta precisão** | **Igual** |
| 800 | Lento | Máxima precisão | Igual |

#### Conclusão Comparativa

```
                    ┌────────────────────────────────────────────┐
                    │         KNN Exato vs HNSW                  │
                    ├─────────────────┬──────────┬───────────────┤
                    │ Métrica         │  KNN     │  HNSW (M=16)  │
                    ├─────────────────┼──────────┼───────────────┤
                    │ RAM base        │ 1,5 GB   │ 1,5 GB        │
                    │ Overhead grafo  │ 0        │ ~1,3 GB       │
                    │ RAM total       │ 1,5 GB   │ ~2,8 GB       │
                    │ Tempo de busca  │ O(N×D)   │ O(log N)      │
                    │ Recall@10       │ 100%     │ ~98%          │
                    │ Escalável?      │ Não      │ Sim           │
                    └─────────────────┴──────────┴───────────────┘
```

**Trade-off**: HNSW paga ~20% a mais de RAM para comprar velocidade de busca 100–1000× superior e escalabilidade logarítmica. Em produção com N > 100k documentos, KNN exato é inviável; HNSW é o padrão da indústria.

---

## Passo 2 — HyDE (Hypothetical Document Embeddings)

**Problema**: A query coloquial `"dor de cabeça latejante e luz incomodando"` vive em uma região do espaço vetorial distante dos documentos técnicos `"cefaleia pulsátil e fotofobia"`. Similaridade de cosseno direta falha.

**Solução HyDE**: O LLM é instruído a *alucinar* um trecho de manual médico técnico que responderia à pergunta. Esse Documento Hipotético usa a mesma terminologia dos documentos reais, posicionando o vetor de busca na região correta do espaço vetorial.

```
Query coloquial  →  embedding  →  vetor em região "leiga"   ← longe dos manuais
                                                                      ↕ gap semântico
Doc. Hipotético  →  embedding  →  vetor em região "técnica" ← próximo aos manuais ✓
```

---

## Passo 3 — Bi-Encoder + HNSW (Funil Largo)

O vetor do Documento Hipotético é usado como query no índice HNSW. O FAISS retorna os **Top-10** documentos mais próximos por similaridade de cosseno em tempo O(log N). Esse é o "funil largo": priorizamos recall (não perder documentos relevantes) sobre precisão.

---

## Passo 4 — Cross-Encoder (Filtro Fino)

O Cross-Encoder `cross-encoder/ms-marco-MiniLM-L-6-v2` recebe cada par `(query_original, documento)` e computa um score de relevância com **atenção cruzada bidirecional**. Diferente do Bi-Encoder (que gera vetores independentes), o Cross-Encoder enxerga a query e o documento simultaneamente, capturando dependências contextuais mais sutis.

Os 10 candidatos são reordenados pelos novos scores. Os **Top-3** finais são os fragmentos injetados no contexto do LLM gerador.

---

## Como Executar

### 1. Instalar dependências

```bash
pip install -r requirements.txt
```

### 2. Executar o pipeline

```bash
python main.py
```

**Sem nenhuma configuração adicional** — o sistema roda completamente local:
- Embeddings: `paraphrase-multilingual-MiniLM-L12-v2` (HuggingFace, local)
- HyDE LLM: `google/flan-t5-base` (HuggingFace, local)
- Cross-Encoder: `ms-marco-MiniLM-L-6-v2` (HuggingFace, local)

Os modelos são baixados automaticamente na primeira execução (~500 MB total).

### Opcional: Groq API para HyDE de maior qualidade

```bash
cp .env.example .env
# Insira GROQ_API_KEY (gratuito em console.groq.com)
```

Com `GROQ_API_KEY` definida, o HyDE usa `llama-3.1-8b-instant` via Groq — melhor qualidade de geração do Documento Hipotético.

Na primeira execução, o script gera os embeddings e constrói o índice HNSW (salvo em `data/`). Execuções seguintes carregam o índice do disco.

Para forçar reconstrução do índice, chame `build_index(force_rebuild=True)` em `main.py`.

---

## Estrutura do Projeto

```
.
├── main.py                  # Orquestração do pipeline completo
├── requirements.txt
├── .env.example
├── data/
│   ├── medical_corpus.json  # 25 fragmentos de manuais médicos
│   ├── hnsw_index.faiss     # Índice HNSW (gerado na execução)
│   └── documents.pkl        # Documentos indexados (gerado na execução)
└── src/
    ├── embedder.py          # Geração de embeddings (OpenAI)
    ├── indexer.py           # Construção do índice HNSW (Passo 1)
    ├── hyde.py              # Geração do Documento Hipotético (Passo 2)
    ├── retriever.py         # Busca Top-10 no HNSW (Passo 3)
    └── reranker.py          # Re-ranking Cross-Encoder Top-3 (Passo 4)
```

---

## Declaração de Uso de IA

Partes deste laboratório foram geradas/complementadas com IA, revisadas e validadas por **José Lucas Silva Souza**.
