"""
Parent-Child (Hierarchical) Chunking for Arabic RAG.

This module implements a two-level chunking strategy:
- Parent chunks: larger segments (e.g., 1500-2000 chars) that provide full
  context for LLM generation
- Child chunks: smaller segments (e.g., 300-500 chars) indexed in Qdrant
  for high-precision vector matching

Each child chunk knows its parent chunk ID, so at retrieval time we can
map matched children back to their parent for richer LLM context.

Why hierarchical: dense embeddings work better on focused, smaller text
segments (children), but LLMs need broader context (parents) to generate
coherent, grounded answers. This decouples retrieval precision from
generation context quality.
"""

import re
from dataclasses import dataclass, field

from .arabic_chunker import _split_long_sentence

# Reuse the Arabic-aware sentence splitter from arabic_chunker
SENTENCE_BOUNDARY = re.compile(r"(?<=[.!؟?؛۔])\s+")


@dataclass
class ParentChunk:
    """A larger chunk that provides full context for LLM generation."""
    text: str
    parent_id: str
    start_sentence_idx: int
    end_sentence_idx: int
    child_ids: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


@dataclass
class ChildChunk:
    """A smaller chunk indexed in Qdrant for high-precision retrieval."""
    text: str
    child_id: str
    parent_id: str
    start_sentence_idx: int
    end_sentence_idx: int
    metadata: dict = field(default_factory=dict)


def split_sentences(text: str, max_sentence_chars: int | None = None) -> list[str]:
    """Split normalized text into sentences using Arabic-aware boundaries.

    When ``max_sentence_chars`` is set, punctuation-free sentences longer
    than the limit (poetry, lists, OCR) are split at word boundaries.
    """
    sentences = [s.strip() for s in SENTENCE_BOUNDARY.split(text.strip()) if s.strip()]
    if max_sentence_chars is None:
        return sentences

    expanded: list[str] = []
    for sentence in sentences:
        if len(sentence) <= max_sentence_chars:
            expanded.append(sentence)
        else:
            expanded.extend(
                _split_long_sentence(
                    sentence,
                    max_sentence_chars,
                    overlap_chars=min(100, max_sentence_chars // 8),
                )
            )
    return expanded


def build_parent_chunks(
    text: str,
    max_parent_chars: int = 1800,
    overlap_sentences: int = 1,
    metadata: dict | None = None,
) -> list[ParentChunk]:
    """
    Build parent chunks from text using sentence-boundary aware chunking.
    These are the chunks that will be sent to the LLM for context.
    """
    sentences = split_sentences(text, max_sentence_chars=max_parent_chars)
    if not sentences:
        return []

    parents: list[ParentChunk] = []
    current: list[str] = []
    current_len = 0
    start_idx = 0
    parent_idx = 0

    for i, sentence in enumerate(sentences):
        sentence_len = len(sentence)

        if current_len + sentence_len > max_parent_chars and current:
            parent_text = " ".join(current)
            parent_id = f"parent_{parent_idx}"
            parents.append(
                ParentChunk(
                    text=parent_text,
                    parent_id=parent_id,
                    start_sentence_idx=start_idx,
                    end_sentence_idx=i - 1,
                    metadata=dict(metadata or {}),
                )
            )
            # carry over overlap sentences
            overlap = current[-overlap_sentences:] if overlap_sentences else []
            current = list(overlap)
            current_len = sum(len(s) for s in current)
            start_idx = i - len(overlap)
            parent_idx += 1

        current.append(sentence)
        current_len += sentence_len

    if current:
        parent_text = " ".join(current)
        parent_id = f"parent_{parent_idx}"
        parents.append(
            ParentChunk(
                text=parent_text,
                parent_id=parent_id,
                start_sentence_idx=start_idx,
                end_sentence_idx=len(sentences) - 1,
                metadata=dict(metadata or {}),
            )
        )

    return parents


def build_child_chunks(
    parent: ParentChunk,
    max_child_chars: int = 400,
    overlap_sentences: int = 0,
) -> list[ChildChunk]:
    """
    Split a parent chunk into smaller child chunks for indexing.
    Children stay within parent sentence boundaries.
    """
    # Get the sentences that belong to this parent
    # We need the full text sentences - for simplicity, we split the parent text
    sentences = split_sentences(parent.text, max_sentence_chars=max_child_chars)
    if not sentences:
        return []

    children: list[ChildChunk] = []
    current: list[str] = []
    current_len = 0
    start_idx = 0
    child_idx = 0

    for i, sentence in enumerate(sentences):
        sentence_len = len(sentence)

        if current_len + sentence_len > max_child_chars and current:
            child_text = " ".join(current)
            child_id = f"{parent.parent_id}_child_{child_idx}"
            children.append(
                ChildChunk(
                    text=child_text,
                    child_id=child_id,
                    parent_id=parent.parent_id,
                    start_sentence_idx=start_idx,
                    end_sentence_idx=i - 1,
                    metadata={},
                )
            )
            # carry over overlap sentences (usually 0 for children to keep them focused)
            overlap = current[-overlap_sentences:] if overlap_sentences else []
            current = list(overlap)
            current_len = sum(len(s) for s in current)
            start_idx = i - len(overlap)
            child_idx += 1

        current.append(sentence)
        current_len += sentence_len

    if current:
        child_text = " ".join(current)
        child_id = f"{parent.parent_id}_child_{child_idx}"
        children.append(
            ChildChunk(
                text=child_text,
                child_id=child_id,
                parent_id=parent.parent_id,
                start_sentence_idx=start_idx,
                end_sentence_idx=len(sentences) - 1,
                metadata={},
            )
        )

    return children


def hierarchical_chunk_text(
    text: str,
    max_parent_chars: int = 1800,
    max_child_chars: int = 400,
    parent_overlap_sentences: int = 1,
    child_overlap_sentences: int = 0,
    metadata: dict | None = None,
) -> tuple[list[ParentChunk], list[ChildChunk]]:
    """
    Main entry point: chunk text into parent and child chunks.

    Returns:
        (parents, children) where:
        - parents: list of ParentChunk objects for LLM context
        - children: list of ChildChunk objects for Qdrant indexing
    """
    parents = build_parent_chunks(
        text,
        max_parent_chars=max_parent_chars,
        overlap_sentences=parent_overlap_sentences,
        metadata=metadata,
    )

    all_children: list[ChildChunk] = []
    for parent in parents:
        children = build_child_chunks(
            parent,
            max_child_chars=max_child_chars,
            overlap_sentences=child_overlap_sentences,
        )
        # Link children to parent
        parent.child_ids = [c.child_id for c in children]
        all_children.extend(children)

    return parents, all_children


def chunk_text(
    text: str,
    max_chunk_chars: int = 800,
    overlap_sentences: int = 1,
    metadata: dict | None = None,
) -> list:
    """
    Backward-compatible interface that returns only parent chunks.
    This allows the hierarchical chunker to be used as a drop-in replacement
    for the existing arabic_chunker interface when parent-child is not needed.
    """
    parents, _ = hierarchical_chunk_text(
        text,
        max_parent_chars=max_chunk_chars,
        max_child_chars=max_chunk_chars // 2,
        parent_overlap_sentences=overlap_sentences,
        metadata=metadata,
    )
    # Return in a format compatible with the old Chunk interface
    return parents