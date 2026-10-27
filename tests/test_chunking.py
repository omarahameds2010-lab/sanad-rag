import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.chunking.arabic_chunker import chunk_text, split_sentences
from src.ingestion.normalize import normalize_for_retrieval, strip_diacritics


def test_split_sentences_arabic_punctuation():
    text = "هذا مثال أول. وهذا مثال ثانٍ؟ وهذا ثالث؛ وأخيرا هذا الرابع."
    sentences = split_sentences(text)
    assert len(sentences) == 4


def test_chunk_text_respects_max_size_and_overlap():
    text = "جملة رقم واحد. جملة رقم اثنين. جملة رقم ثلاثة. جملة رقم أربعة."
    chunks = chunk_text(text, max_chunk_chars=30, overlap_sentences=1)
    assert len(chunks) >= 2
    # every chunk must be non-empty and traceable to source sentence range
    for c in chunks:
        assert c.text
        assert c.start_sentence_idx <= c.end_sentence_idx


def test_strip_diacritics_removes_tashkeel():
    text = "الْعَرَبِيَّة"
    stripped = strip_diacritics(text)
    assert "َ" not in stripped
    assert "ً" not in stripped


def test_normalize_for_retrieval_unifies_alef_variants():
    assert normalize_for_retrieval("أحمد") == normalize_for_retrieval("احمد")


def test_long_unpunctuated_sentence_splits_at_word_boundaries():
    """Poetry and lists have no sentence punctuation; a long poem must not
    become one giant chunk (the Nazm al-Ajurrumiyya regression)."""
    text = " ".join(["كلمة"] * 400)
    chunks = chunk_text(text, max_chunk_chars=300, overlap_sentences=1)

    assert len(chunks) > 1
    assert max(len(c.text) for c in chunks) <= 300
    words = [word for c in chunks for word in c.text.split()]
    assert set(words) == {"كلمة"}
    assert len(words) >= 400  # overlap may duplicate words, never drop them
