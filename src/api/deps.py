"""
Shared dependencies for SANAD API.
Moved here to avoid circular imports.
"""

from src.embeddings.base import HashEmbedding, SentenceTransformerEmbedding
from src.generation.llm_client import AnthropicLLM, FakeLLM, GeminiLLM


def _get_embedding_model():
    """
    Selects the embedding backend. Defaults to HashEmbedding because this
    sandboxed environment has no network access to huggingface.co or an
    OpenAI API key. On Omar's own machine (real network + model access),
    set EMBEDDING_MODEL=multilingual-e5 to use the real model documented
    in docs/research.md.
    """
    import os

    choice = os.environ.get("EMBEDDING_MODEL", "hash-dev")
    if choice == "multilingual-e5":
        return SentenceTransformerEmbedding()
    if choice == "bge-m3":
        return SentenceTransformerEmbedding(model_name="BAAI/bge-m3")
    return HashEmbedding()


def _get_llm_clients():
    """
    Selects primary + fallback LLM backends. Defaults to FakeLLM because
    this sandboxed environment has no ANTHROPIC_API_KEY / network access
    for real generation. On Omar's own machine, set ANTHROPIC_API_KEY or
    GEMINI_API_KEY to get real generation with the FakeLLM kept only as the fallback.
    """
    import os

    print(f"[DEBUG] ANTHROPIC_API_KEY set: {bool(os.environ.get('ANTHROPIC_API_KEY'))}")
    print(f"[DEBUG] GEMINI_API_KEY set: {bool(os.environ.get('GEMINI_API_KEY'))}")
    print(f"[DEBUG] GEMINI_API_KEY value: {os.environ.get('GEMINI_API_KEY', '')[:20]}...")

    if os.environ.get("ANTHROPIC_API_KEY"):
        print("[DEBUG] Using AnthropicLLM")
        return AnthropicLLM(), FakeLLM()
    if os.environ.get("GEMINI_API_KEY"):
        print("[DEBUG] Using GeminiLLM")
        return GeminiLLM(), FakeLLM()
    print("[DEBUG] Using FakeLLM (no API keys found)")
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