"""
Baseline chunker — naive character/whitespace splitting.

This is the "wrong" approach most RAG tutorials use: split every N
characters or on simple whitespace, with no awareness of Arabic sentence
boundaries or word structure. It exists solely as a baseline to measure
how much the Arabic-aware chunker improves retrieval quality.

This chunker will:
- Split mid-sentence on Arabic text
- Cut words at arbitrary character positions
- Not respect Arabic punctuation (،، ؛، ۔)
"""

from dataclasses import dataclass, field


@dataclass
class Chunk:
    text: str
    start_char_idx: int
    end_char_idx: int
    metadata: dict = field(default_factory=dict)


def chunk_text(
    text: str,
    max_chunk_chars: int = 800,
    overlap_chars: int = 100,
    metadata: dict | None = None,
) -> list[Chunk]:
    """
    Naive character-based chunking — splits text into fixed-size chunks
    with character overlap, ignoring sentence boundaries and word structure.

    This is intentionally simple and wrong for Arabic — it's the baseline
    against which we measure the Arabic-aware chunker's improvement.
    """
    if not text.strip():
        return []

    chunks: list[Chunk] = []
    start = 0
    chunk_idx = 0

    while start < len(text):
        end = min(start + max_chunk_chars, len(text))
        chunk_text_value = text[start:end].strip()
        
        if chunk_text_value:
            chunks.append(
                Chunk(
                    text=chunk_text_value,
                    start_char_idx=start,
                    end_char_idx=end,
                    metadata=dict(metadata or {}),
                )
            )
        
        # Move start with overlap
        start = end - overlap_chars if end < len(text) else end
        chunk_idx += 1

    return chunks
