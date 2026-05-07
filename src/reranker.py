"""
Passo 4: Filtro Fino via Re-ranking com Cross-Encoder

O Cross-Encoder avalia cada par (query_original, documento) em conjunto,
com atenção cruzada bidirecional — diferente do Bi-Encoder, que gera vetores
independentes. Isso captura dependências contextuais mais ricas, mas é mais lento:
por isso é aplicado apenas nos Top-K do HNSW, não em todo o corpus.

Formato de entrada para o Cross-Encoder:
    [CLS] Query [SEP] Documento [SEP]
"""

from sentence_transformers import CrossEncoder

CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

_cross_encoder: CrossEncoder | None = None


def load_cross_encoder() -> CrossEncoder:
    global _cross_encoder
    if _cross_encoder is None:
        print(f"[Cross-Encoder] Carregando modelo: {CROSS_ENCODER_MODEL}")
        _cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL)
        print("[Cross-Encoder] Modelo carregado.")
    return _cross_encoder


def rerank(query: str, candidates: list[dict], cross_encoder: CrossEncoder) -> list[dict]:
    """
    Recebe a query original (coloquial) e os candidatos do Bi-Encoder.
    Retorna a lista reordenada pelo score do Cross-Encoder (decrescente).
    """
    print(f"[Cross-Encoder] Re-ranking de {len(candidates)} candidatos com atenção profunda...")

    pairs = [(query, doc["content"]) for doc in candidates]
    scores = cross_encoder.predict(pairs)

    reranked = []
    for doc, score in zip(candidates, scores):
        doc_copy = doc.copy()
        doc_copy["cross_encoder_score"] = float(score)
        reranked.append(doc_copy)

    reranked.sort(key=lambda x: x["cross_encoder_score"], reverse=True)

    for rank, doc in enumerate(reranked):
        doc["cross_encoder_rank"] = rank + 1

    return reranked


def print_final_results(results: list[dict], top_k: int = 3) -> None:
    top = results[:top_k]
    print(f"\n{'='*60}")
    print(f"[Cross-Encoder] Top-{top_k} documentos finais → contexto do LLM gerador:")
    print(f"{'='*60}")
    for doc in top:
        bi_rank = doc.get("bi_encoder_rank", "?")
        ce_score = doc["cross_encoder_score"]
        bi_score = doc.get("bi_encoder_score", 0.0)
        print(f"\n  Rank Final #{doc['cross_encoder_rank']:02d} | CE Score: {ce_score:.4f}"
              f"  (era Bi-Encoder #{bi_rank:02d}, score {bi_score:.4f})")
        print(f"  [{doc['category']}] {doc['title']}")
        print(f"  Conteúdo completo:")
        print(f"    {doc['content']}")
    print(f"\n{'='*60}")
    print("→ Estes 3 fragmentos seriam injetados no prompt do LLM gerador.")
    print(f"{'='*60}\n")
