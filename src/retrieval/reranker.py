"""
Reranker — re-ranks retrieved results using cross-encoder scoring.

After initial retrieval (semantic + BM25 hybrid), a cross-encoder model
scores each (question, chunk) pair more precisely than the bi-encoder
embeddings used for the initial search. This is expensive but can
significantly improve precision@k for the top results.

This module uses a simple cross-encoder interface that can be swapped
for different models (e.g., bge-reranker, Cohere Rerank API).
"""

from abc import ABC, abstractmethod


class Reranker(ABC):
    name: str

    @abstractmethod
    def rerank(
        self,
        question: str,
        chunks: list[dict],
        top_k: int = 5,
    ) -> list[dict]:
        """
        Re-rank chunks based on their relevance to the question.
        
        Args:
            question: the user's query
            chunks: list of retrieved chunks with 'chunk_id', 'text', and existing scores
            top_k: number of top results to return after reranking
            
        Returns:
            Re-ordered list of chunks, with top_k most relevant first
        """
        raise NotImplementedError


class CrossEncoderReranker(Reranker):
    """
    Open-source cross-encoder reranker (e.g., bge-m3-reranker).
    More accurate than bi-encoder embeddings but slower.
    """

    def __init__(self, model_name: str = "BAAI/bge-reranker-base"):
        try:
            from sentence_transformers import CrossEncoder
            self.name = model_name
            self._model = CrossEncoder(model_name)
        except Exception:
            # Fallback to overlap scoring when the package is missing OR the
            # weights can't be downloaded (offline sandbox) — never crash a run.
            self.name = f"{model_name}-fallback"
            self._model = None

    def rerank(
        self,
        question: str,
        chunks: list[dict],
        top_k: int = 5,
    ) -> list[dict]:
        if not chunks:
            return []

        if not question.strip():
            # No query signal: preserve retrieval order instead of letting a
            # cross-encoder score pairs against an empty string.
            return chunks[:top_k]

        if self._model is None:
            # Fallback: simple keyword overlap scoring
            return self._fallback_rerank(question, chunks, top_k)

        # Cross-encoder scoring
        pairs = [[question, chunk.get("raw_text", chunk.get("text", ""))] for chunk in chunks]
        scores = self._model.predict(pairs)

        # Add scores and re-sort
        for chunk, score in zip(chunks, scores):
            chunk["rerank_score"] = float(score)

        reranked = sorted(chunks, key=lambda c: c["rerank_score"], reverse=True)
        return reranked[:top_k]

    def _fallback_rerank(
        self,
        question: str,
        chunks: list[dict],
        top_k: int,
    ) -> list[dict]:
        """
        Simple fallback: score by term overlap between question and chunk.
        Not as good as a real cross-encoder, but better than nothing.
        """
        question_terms = set(question.lower().split())
        
        for chunk in chunks:
            text = chunk.get("raw_text", chunk.get("text", "")).lower()
            chunk_terms = set(text.split())
            overlap = len(question_terms & chunk_terms)
            chunk["rerank_score"] = overlap

        reranked = sorted(chunks, key=lambda c: c["rerank_score"], reverse=True)
        return reranked[:top_k]


class NoOpReranker(Reranker):
    """
    No-op reranker — returns chunks in their original order.
    Used for the "reranking disabled" baseline comparison.
    """

    def __init__(self):
        self.name = "no-op"

    def rerank(
        self,
        question: str,
        chunks: list[dict],
        top_k: int = 5,
    ) -> list[dict]:
        # Just return top_k without reordering
        return chunks[:top_k]
