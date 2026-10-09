"""
Shared dependencies for SANAD API.
Moved here to avoid circular imports.
"""

from src.embeddings.base import EmbeddingModel, HashEmbedding, SentenceTransformerEmbedding
from src.generation.llm_client import AnthropicLLM, FakeLLM, GeminiLLM

# Process-wide cache: loading a 2GB embedding model per request is slow
# and flaky (each load re-validates the HF cache). Load once, reuse.
# Keyed by EMBEDDING_MODEL choice so A/B comparisons stay isolated.
_embedding_singleton: dict[str, EmbeddingModel] = {}


def _get_embedding_model() -> EmbeddingModel:
    """
    Selects the embedding backend. Defaults to HashEmbedding because this
    sandboxed environment has no network access to huggingface.co or an
    OpenAI API key. On Omar's own machine (real network + model access),
    set EMBEDDING_MODEL=multilingual-e5 to use the real model documented
    in docs/research.md.
    """
    import os

    choice = os.environ.get("EMBEDDING_MODEL", "hash-dev")
    if choice not in _embedding_singleton:
        if choice == "multilingual-e5":
            _embedding_singleton[choice] = SentenceTransformerEmbedding()
        elif choice == "bge-m3":
            _embedding_singleton[choice] = SentenceTransformerEmbedding(model_name="BAAI/bge-m3")
        else:
            _embedding_singleton[choice] = HashEmbedding()
    return _embedding_singleton[choice]


def warmup_embedding_model() -> None:
    """Pre-load the embedding model at startup so the first request isn't slow.

    Raises on failure — callers decide whether a missing model is fatal.
    """
    _get_embedding_model()


def reset_embedding_singleton_for_testing() -> None:
    """Clears the cache — used by tests that switch EMBEDDING_MODEL."""
    _embedding_singleton.clear()


def _get_llm_clients():
    """
    Selects primary + fallback LLM backends. Defaults to FakeLLM because
    this sandboxed environment has no ANTHROPIC_API_KEY / network access
    for real generation. On Omar's own machine, set ANTHROPIC_API_KEY or
    GEMINI_API_KEY to get real generation with the FakeLLM kept only as the fallback.
    """
    import os

    if os.environ.get("ANTHROPIC_API_KEY"):
        return AnthropicLLM(), FakeLLM()
    if os.environ.get("GEMINI_API_KEY"):
        return GeminiLLM(), FakeLLM()
    return FakeLLM(), None


def _check_qdrant() -> bool:
    try:
        from src.retrieval.vector_store import get_client

        client = get_client()
        client.get_collections()
        return True
    except Exception:
        return False


def _check_db() -> bool:
    try:
        from src.evaluation.storage import get_connection

        with get_connection() as conn:
            conn.execute("SELECT 1")
        return True
    except Exception:
        return False