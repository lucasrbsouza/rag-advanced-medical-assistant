"""
RAG Advanced Medical Assistant
Pipeline: HNSW + HyDE + Cross-Encoder

Autor: José Lucas Silva Souza
Lab 09 — NLP — 7º Período

Fluxo:
  Query Coloquial
      │
      ▼ [Passo 2] HyDE: LLM gera Documento Hipotético técnico
  Documento Hipotético
      │
      ▼ embed → vetor normalizado
  Vetor HyDE
      │
      ▼ [Passo 3] Busca HNSW (Bi-Encoder) → Top-10
  10 Candidatos
      │
      ▼ [Passo 4] Cross-Encoder re-ranking → Top-3
  3 Documentos Finais → Contexto do LLM Gerador
"""

from src.indexer import build_index
from src.hyde import generate_hypothetical_document
from src.embedder import get_embedding
from src.retriever import retrieve_top_k, print_retrieval_results
from src.reranker import load_cross_encoder, rerank, print_final_results


def run_rag_pipeline(query: str, top_retrieve: int = 10, top_final: int = 3) -> list[dict]:
    print("\n" + "█" * 60)
    print("  RAG ADVANCED MEDICAL ASSISTANT — Pipeline Completo")
    print("█" * 60)

    # ── Passo 1: Carregar/construir índice HNSW ─────────────────────
    print("\n[Passo 1] Índice HNSW")
    index, documents = build_index()

    # ── Passo 2: HyDE — gerar e vetorizar Documento Hipotético ──────
    print("\n[Passo 2] HyDE — Query Transformation")
    hypothetical_doc = generate_hypothetical_document(query)
    hyde_vector = get_embedding(hypothetical_doc)

    # ── Passo 3: Busca rápida no HNSW ───────────────────────────────
    print("\n[Passo 3] Retrieve via Bi-Encoder (HNSW)")
    candidates = retrieve_top_k(hyde_vector, index, documents, k=top_retrieve)
    print_retrieval_results(candidates)

    # ── Passo 4: Re-ranking com Cross-Encoder ───────────────────────
    print("\n[Passo 4] Re-ranking via Cross-Encoder")
    cross_encoder = load_cross_encoder()
    reranked = rerank(query, candidates, cross_encoder)
    print_final_results(reranked, top_k=top_final)

    return reranked[:top_final]


if __name__ == "__main__":
    # Exemplo do enunciado do laboratório:
    # query coloquial → sistema converte para busca técnica
    query_exemplo = "dor de cabeça latejante e a luz me incomodando muito"

    resultado = run_rag_pipeline(query=query_exemplo)
