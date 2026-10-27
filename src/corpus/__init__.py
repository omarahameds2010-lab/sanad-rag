"""Corpus governance: only audited sources may enter the index."""

from src.corpus.manifest import (
    DEFAULT_MANIFEST,
    REPO_ROOT,
    CorpusSource,
    load_manifest,
    verify_source,
)

__all__ = [
    "DEFAULT_MANIFEST",
    "REPO_ROOT",
    "CorpusSource",
    "load_manifest",
    "verify_source",
]
