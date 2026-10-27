"""Shared model selection for scripts.

The environment decides which embedding model runs; no script silently
substitutes a fake model without saying so. A benchmark is only as honest as
the model name stored next to it.
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def get_embedding_model():
    choice = os.environ.get("EMBEDDING_MODEL", "hash-dev")
    if choice == "multilingual-e5":
        from src.embeddings.base import SentenceTransformerEmbedding

        return SentenceTransformerEmbedding()
    if choice == "bge-m3":
        from src.embeddings.base import SentenceTransformerEmbedding

        return SentenceTransformerEmbedding(model_name="BAAI/bge-m3")
    from src.embeddings.base import HashEmbedding

    print(
        "⚠️  EMBEDDING_MODEL not set to a real model — using HashEmbedding. "
        "Set EMBEDDING_MODEL=multilingual-e5 (or bge-m3) for real numbers."
    )
    return HashEmbedding()
