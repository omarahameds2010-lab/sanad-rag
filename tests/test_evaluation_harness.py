import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

from src.embeddings.base import HashEmbedding
from src.generation.llm_client import FakeLLM
from src.evaluation.harness import run_evaluation
from src.evaluation.storage import get_evaluation_run

GOLDEN_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "golden_dataset", "sample_golden.json"
)


@pytest.fixture
def isolated_env(monkeypatch, tmp_path):
    from src.retrieval.vector_store import reset_client_for_testing

    reset_client_for_testing()
    monkeypatch.setattr(
        "src.retrieval.vector_store.DEFAULT_LOCAL_PATH", str(tmp_path / "qdrant_local")
    )
    monkeypatch.setattr("src.evaluation.storage.DEFAULT_DB_PATH", tmp_path / "sanad.db")
    monkeypatch.delenv("QDRANT_URL", raising=False)
    yield
    reset_client_for_testing()


def test_run_evaluation_produces_real_metrics(isolated_env):
    metrics = run_evaluation(
        golden_dataset_path=GOLDEN_PATH,
        embedding_model=HashEmbedding(dims=64),
        llm=FakeLLM(),
    )

    assert metrics["n_examples"] == 4
    assert 0.0 <= metrics["precision_at_k"] <= 1.0
    assert 0.0 <= metrics["recall_at_k"] <= 1.0
    assert metrics["is_dev_only_run"] is True  # honest flag, not hidden


def test_evaluation_run_is_persisted_and_retrievable(isolated_env):
    metrics = run_evaluation(
        golden_dataset_path=GOLDEN_PATH,
        embedding_model=HashEmbedding(dims=64),
        llm=FakeLLM(),
        config_name="test-config",
    )

    stored = get_evaluation_run(metrics["run_id"])
    assert stored is not None
    assert stored["config_name"] == "test-config"
    assert stored["n_examples"] == 4


def test_recall_is_perfect_when_corpus_is_small_and_topk_generous(isolated_env):
    """Sanity check on the sample dataset itself: with only 3 tiny source
    documents and top_k=5, every relevant chunk should be retrievable —
    if this ever drops, the chunker/retriever broke, not the dataset."""
    metrics = run_evaluation(
        golden_dataset_path=GOLDEN_PATH,
        embedding_model=HashEmbedding(dims=64),
        llm=FakeLLM(),
        top_k=5,
    )
    assert metrics["recall_at_k"] == 1.0
