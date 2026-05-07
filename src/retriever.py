"""
Passo 3: Busca Rápida via Bi-Encoder no Índice HNSW

Usa o vetor do Documento Hipotético (HyDE) como query de busca.
O HNSW retorna os Top-K documentos em tempo sub-linear O(log N),
formando o "funil largo" que alimenta o Cross-Encoder.
"""

import faiss
import numpy as np


def retrieve_top_k(
    query_vector: np.ndarray,
    index: faiss.Index,
    documents: list[dict],
    k: int = 10,
) -> list[dict]:
    """
    Busca os k documentos mais similares ao query_vector no índice HNSW.
    Retorna lista de dicionários com campos do documento + metadados de ranking.
    """
    query_vector = query_vector.reshape(1, -1).astype(np.float32)
    distances, indices = index.search(query_vector, k)

    results = []
    for rank, (dist, idx) in enumerate(zip(distances[0], indices[0])):
        if idx == -1:
            continue
        doc = documents[idx].copy()
        doc["bi_encoder_score"] = float(dist)
        doc["bi_encoder_rank"] = rank + 1
        results.append(doc)

    return results


def print_retrieval_results(results: list[dict]) -> None:
    print(f"\n{'='*60}")
    print(f"[Bi-Encoder / HNSW] Top-{len(results)} documentos recuperados (funil largo):")
    print(f"{'='*60}")
    for doc in results:
        print(
            f"\n  #{doc['bi_encoder_rank']:02d} | Score Cosseno: {doc['bi_encoder_score']:.4f}"
        )
        print(f"       [{doc['category']}] {doc['title']}")
        print(f"       {doc['content'][:120]}...")
    print(f"{'='*60}\n")
