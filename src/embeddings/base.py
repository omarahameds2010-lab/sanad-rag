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
        """Return one embedding vector per input text."""
        raise NotImplementedError

    @property
    @abstractmethod
    def dimensions(self) -> int:
        raise NotImplementedError


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
        # e5 models expect a "query: " / "passage: " prefix convention
        prefixed = [f"passage: {t}" for t in texts]
        vectors = self._model.encode(prefixed, normalize_embeddings=True)
        return vectors.tolist()

    @property
    def dimensions(self) -> int:
        return self._model.get_sentence_embedding_dimension()


class OpenAIEmbedding(EmbeddingModel):
    """
    Comparison arm: OpenAI's embedding API. Kept behind the same interface
    so the evaluation harness can score it against the open-source default
    on identical chunks — this is what makes the comparison legitimate.
    """

    def __init__(self, model_name: str = "text-embedding-3-small"):
        self.name = model_name
        self._dims = 1536 if "small" in model_name else 3072

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
