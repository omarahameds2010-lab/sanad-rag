"""
Pluggable embedding interface.

Why an interface instead of calling one embedding model directly: the whole
point of this project is to A/B compare embedding strategies on the same
Arabic corpus and measure which one actually retrieves better. Hardcoding a
single provider would make that comparison impossible later.
"""

from abc import ABC, abstractmethod


class EmbeddingModel(ABC):
    name: str

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Return one embedding vector per input text (passages/documents)."""
        raise NotImplementedError

    def embed_query(self, text: str) -> list[float]:
        """Embed a search query.

        Models with asymmetric query/passage conventions (e5) override this.
        Symmetric models (bge-m3) use the same encoding for both.
        """
        return self.embed([text])[0]

    @property
    @abstractmethod
    def dimensions(self) -> int:
        raise NotImplementedError


def _is_e5(model_name: str) -> bool:
    return "e5" in model_name.lower()


def _format_passages(model_name: str, texts: list[str]) -> list[str]:
    # e5 expects "passage: " on documents; bge-m3 expects no prefix.
    return [f"passage: {text}" for text in texts] if _is_e5(model_name) else texts


def _format_query(model_name: str, text: str) -> str:
    # e5 expects "query: " on queries. Applying the passage prefix to a query
    # (the old behaviour) silently degrades every e5 retrieval number.
    return f"query: {text}" if _is_e5(model_name) else text


class SentenceTransformerEmbedding(EmbeddingModel):
    """
    Open-source multilingual embedding model — used as the $0-cost default
    during development so experimentation doesn't burn API budget.
    """

    def __init__(self, model_name: str = "intfloat/multilingual-e5-base"):
        from sentence_transformers import SentenceTransformer

        self.name = model_name
        self._model = SentenceTransformer(model_name)

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors = self._model.encode(
            _format_passages(self.name, texts), normalize_embeddings=True
        )
        return vectors.tolist()

    def embed_query(self, text: str) -> list[float]:
        vector = self._model.encode(
            [_format_query(self.name, text)], normalize_embeddings=True
        )[0]
        return vector.tolist()

    @property
    def dimensions(self) -> int:
        get_dim = getattr(self._model, "get_embedding_dimension", None)
        if callable(get_dim):
            return get_dim()
        return self._model.get_sentence_embedding_dimension()  # legacy API


_OPENAI_EMBEDDING_DIMS = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}


class OpenAIEmbedding(EmbeddingModel):
    """
    Comparison arm: OpenAI's embedding API. Kept behind the same interface
    so the evaluation harness can score it against the open-source default
    on identical chunks — this is what makes the comparison legitimate.
    """

    def __init__(self, model_name: str = "text-embedding-3-small"):
        if model_name not in _OPENAI_EMBEDDING_DIMS:
            raise ValueError(
                f"Unknown OpenAI embedding model {model_name!r}; known models: "
                f"{sorted(_OPENAI_EMBEDDING_DIMS)}. Add its dimension explicitly "
                "instead of guessing from the name."
            )
        self.name = model_name
        self._dims = _OPENAI_EMBEDDING_DIMS[model_name]

    def embed(self, texts: list[str]) -> list[list[float]]:
        import os

        from openai import OpenAI

        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        response = client.embeddings.create(model=self.name, input=texts)
        return [item.embedding for item in response.data]

    @property
    def dimensions(self) -> int:
        return self._dims


class HashEmbedding(EmbeddingModel):
    """
    Deterministic, dependency-free pseudo-embedding used ONLY for local
    development and tests where model downloads (Hugging Face) or API keys
    (OpenAI) aren't available — e.g. in a sandboxed CI environment with no
    access to huggingface.co.

    This is NOT a retrieval-quality embedding — it has no semantic meaning,
    just deterministic character n-gram hashing into a fixed-size vector.
    Its only job is to let the ingestion/retrieval *pipeline plumbing* be
    tested end-to-end (chunking -> embedding -> vector store -> retrieval)
    without needing network access or a GPU. Retrieval-quality experiments
    in Phase 5/6 MUST use SentenceTransformerEmbedding or OpenAIEmbedding —
    never this class — or every precision/recall number would be meaningless.
    """

    def __init__(self, dims: int = 128):
        self.name = "hash-embedding-DEV-ONLY"
        self._dims = dims

    def embed(self, texts: list[str]) -> list[list[float]]:
        import hashlib
        import math

        vectors = []
        for text in texts:
            vec = [0.0] * self._dims
            # hash overlapping character trigrams into buckets — gives
            # texts that share substrings a nonzero dot product, which is
            # enough to exercise ranking logic in tests without claiming
            # any real semantic understanding.
            for i in range(len(text) - 2):
                trigram = text[i : i + 3]
                bucket = int(hashlib.md5(trigram.encode("utf-8")).hexdigest(), 16) % self._dims
                vec[bucket] += 1.0
            norm = math.sqrt(sum(v * v for v in vec)) or 1.0
            vectors.append([v / norm for v in vec])
        return vectors

    @property
    def dimensions(self) -> int:
        return self._dims
