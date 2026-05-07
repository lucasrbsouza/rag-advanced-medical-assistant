"""
Modelo de embedding local via HuggingFace/sentence-transformers.
Usa paraphrase-multilingual-MiniLM-L12-v2 (multilingual, suporte nativo ao português).
Dimensão dos vetores: 384.
"""

import numpy as np
from sentence_transformers import SentenceTransformer

# Modelo multilingual — funciona bem com texto médico em português
EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"

_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        print(f"[Embedder] Carregando modelo local: {EMBEDDING_MODEL}")
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model


def _normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    return vectors / np.where(norms > 0, norms, 1.0)


def get_embedding(text: str) -> np.ndarray:
    model = _get_model()
    vector = model.encode(text, normalize_embeddings=True)
    return vector.astype(np.float32)


def get_embeddings_batch(texts: list[str]) -> np.ndarray:
    model = _get_model()
    vectors = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)
    return vectors.astype(np.float32)
