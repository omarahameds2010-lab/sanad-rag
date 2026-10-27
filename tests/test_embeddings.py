"""Embedding prefix conventions.

e5 is asymmetric: documents take ``passage:`` and queries take ``query:``.
bge-m3 is symmetric and takes no prefix. Applying the passage prefix to a
query, or any prefix to bge-m3, silently degrades retrieval — so the
convention is tested explicitly.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.embeddings.base import (
    HashEmbedding,
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
