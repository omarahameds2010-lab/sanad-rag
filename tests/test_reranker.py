"""Reranker edge cases.

An empty query is not a query. Scoring every chunk against "" lets a
cross-encoder (or even the overlap fallback) attach meaningless scores and
reorder results. The reranker must preserve retrieval order and leave the
chunks untouched in that case.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.retrieval.reranker import CrossEncoderReranker, NoOpReranker


def _chunks():
    return [
        {"chunk_id": "a", "raw_text": "النحو"},
        {"chunk_id": "b", "raw_text": "الصرف"},
        {"chunk_id": "c", "raw_text": "البلاغة"},
    ]


def test_empty_question_preserves_order_without_scoring():
    reranker = object.__new__(CrossEncoderReranker)
    reranker.name = "test"
    reranker._model = None  # must not be reached for an empty question

    chunks = _chunks()
    result = reranker.rerank("   ", chunks, top_k=2)

    assert [c["chunk_id"] for c in result] == ["a", "b"]
    assert all("rerank_score" not in c for c in result)


def test_empty_chunk_list_is_safe():
    reranker = object.__new__(CrossEncoderReranker)
    reranker.name = "test"
    reranker._model = None
    assert reranker.rerank("سؤال", [], top_k=5) == []


def test_noop_reranker_keeps_existing_order():
    reranker = NoOpReranker()
    chunks = _chunks()
    assert [c["chunk_id"] for c in reranker.rerank("سؤال", chunks, top_k=2)] == ["a", "b"]
