"""
Arabic-aware chunking.

The naive approach most RAG tutorials use is: split every N characters, or
split on English-style sentence punctuation (". "). This breaks on Arabic
text because:

1. Arabic sentence-final punctuation includes "،" (Arabic comma), "؛"
   (Arabic semicolon), and "۔"/"." — a splitter tuned only for "." misses
   these and produces chunks that cut mid-sentence.
2. Word boundaries in Arabic can be ambiguous around attached prefixes/
   suffixes (و، ف، ب، ل، ك) — splitting on raw whitespace when trimming to
   a size limit can cut a word's grammatical prefix from its root.
3. Arabic text is often mixed with embedded English/numeric spans (dates,
   technical terms) that should not be split across chunk boundaries.

This module produces chunks that:
- Prefer to break on sentence boundaries (Arabic + Latin punctuation)
- Fall back to word boundaries (never mid-word) when a sentence exceeds
  the max chunk size
- Support configurable overlap between chunks (for retrieval context
  continuity), measured in whole sentences, not raw characters
"""

import re
from dataclasses import dataclass, field

# Sentence-final punctuation used in Arabic and mixed Arabic/Latin text.
SENTENCE_BOUNDARY = re.compile(r"(?<=[.!؟?؛۔])\s+")


@dataclass
class Chunk:
    text: str
    start_sentence_idx: int
    end_sentence_idx: int
    metadata: dict = field(default_factory=dict)


def split_sentences(text: str) -> list[str]:
    """Split normalized text into sentences using Arabic-aware boundaries."""
    sentences = SENTENCE_BOUNDARY.split(text.strip())
    return [s.strip() for s in sentences if s.strip()]


def chunk_text(
    text: str,
    max_chunk_chars: int = 800,
    overlap_sentences: int = 1,
    metadata: dict | None = None,
) -> list[Chunk]:
    """
    Build chunks from Arabic (or mixed) text, respecting sentence boundaries.

    Args:
        text: normalized document text (already run through
              ingestion.normalize.clean_for_display).
        max_chunk_chars: soft limit on characters per chunk. A single
              sentence longer than this limit is kept whole rather than
              being cut mid-word.
        overlap_sentences: number of trailing sentences from the previous
              chunk repeated at the start of the next chunk, to preserve
              context continuity across chunk boundaries.
        metadata: optional metadata to attach to every chunk (source_id,
              page_number, etc.) for citation tracking downstream.
    """
    sentences = split_sentences(text)
    if not sentences:
        return []

    chunks: list[Chunk] = []
    current: list[str] = []
    current_len = 0
    start_idx = 0

    for i, sentence in enumerate(sentences):
        sentence_len = len(sentence)

        if current_len + sentence_len > max_chunk_chars and current:
            chunk_text_value = " ".join(current)
            chunks.append(
                Chunk(
                    text=chunk_text_value,
                    start_sentence_idx=start_idx,
                    end_sentence_idx=i - 1,
                    metadata=dict(metadata or {}),
                )
            )
            # carry over the last `overlap_sentences` sentences for context
            overlap = current[-overlap_sentences:] if overlap_sentences else []
            current = list(overlap)
            current_len = sum(len(s) for s in current)
            start_idx = i - len(overlap)

        current.append(sentence)
        current_len += sentence_len

    if current:
        chunks.append(
            Chunk(
                text=" ".join(current),
                start_sentence_idx=start_idx,
                end_sentence_idx=len(sentences) - 1,
                metadata=dict(metadata or {}),
            )
        )

    return chunks
