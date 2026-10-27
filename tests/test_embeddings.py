"""Embedding prefix conventions.

e5 is asymmetric: documents take ``passage:`` and queries take ``query:``.
bge-m3 is symmetric and takes no prefix. Applying the passage prefix to a
query, or any prefix to bge-m3, silently degrades retrieval — so the
convention is tested explicitly.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

from src.embeddings.base import (
    HashEmbedding,
    OpenAIEmbedding,
    _format_passages,
    _format_query,
)


def test_e5_uses_asymmetric_prefixes():
    assert _format_passages("intfloat/multilingual-e5-base", ["نص"]) == ["passage: نص"]
    assert _format_query("intfloat/multilingual-e5-base", "سؤال") == "query: سؤال"


def test_bge_m3_uses_no_prefixes():
    assert _format_passages("BAAI/bge-m3", ["نص"]) == ["نص"]
    assert _format_query("BAAI/bge-m3", "سؤال") == "سؤال"


def test_hash_embedding_query_matches_document_embedding():
    model = HashEmbedding(dims=16)
    assert model.embed_query("سؤال") == model.embed(["سؤال"])[0]


def test_openai_embedding_dimensions_are_model_specific():
    assert OpenAIEmbedding("text-embedding-3-small").dimensions == 1536
    assert OpenAIEmbedding("text-embedding-3-large").dimensions == 3072
    # The old "small in name else 3072" guess got ada-002 wrong.
    assert OpenAIEmbedding("text-embedding-ada-002").dimensions == 1536


def test_openai_embedding_rejects_unknown_model():
    with pytest.raises(ValueError, match="Unknown OpenAI embedding model"):
        OpenAIEmbedding("my-custom-embedding")
