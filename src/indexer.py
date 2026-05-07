"""
Passo 1: Construção e Indexação do Grafo HNSW

Usa faiss.IndexHNSWFlat com METRIC_INNER_PRODUCT. Com vetores normalizados,
produto interno == similaridade de cosseno, permitindo busca aproximada eficiente.
"""

import os
import json
import pickle

import faiss
import numpy as np

from src.embedder import get_embeddings_batch

CORPUS_PATH = os.path.join("data", "medical_corpus.json")
INDEX_PATH = os.path.join("data", "hnsw_index.faiss")
DOCS_PATH = os.path.join("data", "documents.pkl")

EMBEDDING_DIMENSION = 384  # paraphrase-multilingual-MiniLM-L12-v2

# Hiperparâmetros HNSW — ver README.md para análise de impacto em RAM
HNSW_M = 16               # conexões bidirecionais por nó por camada
HNSW_EF_CONSTRUCTION = 200  # candidatos durante construção do grafo
HNSW_EF_SEARCH = 50         # candidatos durante busca (velocidade vs precisão)


def build_index(force_rebuild: bool = False) -> tuple[faiss.Index, list[dict]]:
    if not force_rebuild and os.path.exists(INDEX_PATH) and os.path.exists(DOCS_PATH):
        print("[Indexer] Índice existente encontrado. Carregando do disco...")
        return _load_index()

    print("[Indexer] Construindo índice HNSW...")

    with open(CORPUS_PATH, "r", encoding="utf-8") as f:
        corpus = json.load(f)

    documents: list[dict] = corpus["documents"]
    texts = [doc["content"] for doc in documents]

    print(f"[Indexer] Gerando embeddings para {len(texts)} documentos...")
    vectors = get_embeddings_batch(texts)

    index = faiss.IndexHNSWFlat(EMBEDDING_DIMENSION, HNSW_M, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = HNSW_EF_CONSTRUCTION
    index.hnsw.efSearch = HNSW_EF_SEARCH

    index.add(vectors)

    print(f"[Indexer] Índice construído com {index.ntotal} vetores.")
    print(f"  Hiperparâmetros HNSW: M={HNSW_M}, ef_construction={HNSW_EF_CONSTRUCTION}")
    print(f"  Modelo de embedding: paraphrase-multilingual-MiniLM-L12-v2 (local)")
    print(f"  Dimensão dos vetores: {EMBEDDING_DIMENSION}")

    faiss.write_index(index, INDEX_PATH)
    with open(DOCS_PATH, "wb") as f:
        pickle.dump(documents, f)

    print(f"[Indexer] Índice salvo em '{INDEX_PATH}'")
    return index, documents


def _load_index() -> tuple[faiss.Index, list[dict]]:
    index = faiss.read_index(INDEX_PATH)
    with open(DOCS_PATH, "rb") as f:
        documents = pickle.load(f)
    print(f"[Indexer] Carregado: {index.ntotal} vetores, dimensão {EMBEDDING_DIMENSION}")
    return index, documents
