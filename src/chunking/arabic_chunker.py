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


def _split_long_sentence(
    sentence: str,
    max_chars: int,
    overlap_chars: int = 100,
) -> list[str]:
    """Split a punctuation-free sentence at word boundaries.

    Poetry, lists, and raw OCR often contain thousands of characters without
    a single sentence-final mark. Keeping that as one chunk defeats retrieval
    and bloats the embedding input, so split on whitespace while never
    cutting inside a word. A small trailing-word overlap preserves context.
    """
    if len(sentence) <= max_chars:
        return [sentence]

    words = sentence.split()
    parts: list[str] = []
    current: list[str] = []
    current_len = 0

    for word in words:
        addition = len(word) + (1 if current else 0)
        if current and current_len + addition > max_chars:
            parts.append(" ".join(current))

            carry: list[str] = []
            carry_len = 0
            for previous in reversed(current):
                carry_addition = len(previous) + (1 if carry else 0)
                if carry_len + carry_addition > overlap_chars:
                    break
                carry.insert(0, previous)
                carry_len += carry_addition

            current = carry
            current_len = carry_len
            if current_len + len(word) + (1 if current else 0) > max_chars:
                current = []
                current_len = 0

        current.append(word)
        current_len += len(word) + (1 if len(current) > 1 else 0)

    if current:
        parts.append(" ".join(current))

    return parts


def _prepare_sentences(text: str, max_chars: int) -> list[tuple[str, bool]]:
    """Return (text, is_long_part) entries, splitting oversized sentences."""
    entries: list[tuple[str, bool]] = []
    for sentence in split_sentences(text):
        if len(sentence) <= max_chars:
            entries.append((sentence, False))
        else:
            parts = _split_long_sentence(
                sentence,
                max_chars,
                overlap_chars=min(100, max_chars // 8),
            )
            entries.extend((part, True) for part in parts)
    return entries


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
        max_chunk_chars: soft limit on characters per chunk. Multi-sentence
              chunks can exceed it slightly by the spaces between sentences;
              a single sentence longer than the limit is split at word
              boundaries (never mid-word) with a small word overlap.
        overlap_sentences: number of trailing sentences from the previous
              chunk repeated at the start of the next chunk, to preserve
              context continuity across chunk boundaries.
        metadata: optional metadata to attach to every chunk (source_id,
              page_number, etc.) for citation tracking downstream.
    """
    entries = _prepare_sentences(text, max_chunk_chars)
    if not entries:
        return []

    chunks: list[Chunk] = []
    current: list[tuple[str, bool]] = []
    current_len = 0
    start_idx = 0

    for i, (sentence, is_long_part) in enumerate(entries):
        sentence_len = len(sentence)

        if current_len + sentence_len > max_chunk_chars and current:
            chunks.append(
                Chunk(
                    text=" ".join(s for s, _ in current),
                    start_sentence_idx=start_idx,
                    end_sentence_idx=i - 1,
                    metadata=dict(metadata or {}),
                )
            )
            # carry over the last `overlap_sentences` sentences for context
            overlap = current[-overlap_sentences:] if overlap_sentences else []
            # Long-sentence parts already carry a word overlap; copying a
            # whole 800-char part would double the chunk size for no gain.
            if is_long_part or any(is_long for _, is_long in overlap):
                overlap = []
            current = list(overlap)
            current_len = sum(len(s) for s, _ in current)
            start_idx = i - len(overlap)

        current.append((sentence, is_long_part))
        current_len += sentence_len

    if current:
        chunks.append(
            Chunk(
                text=" ".join(s for s, _ in current),
                start_sentence_idx=start_idx,
                end_sentence_idx=len(entries) - 1,
                metadata=dict(metadata or {}),
            )
        )

    return chunks
