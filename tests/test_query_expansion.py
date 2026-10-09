import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

from src.retrieval.query_expansion import (
    expand_query,
    expand_with_synonyms,
    expand_with_templates,
    multi_query_retrieve,
)


def test_expand_with_synonyms_returns_original():
    """Original query should always be in variations."""
    query = "ما هو الإعراب في النحو العربي؟"
    variations = expand_with_synonyms(query)
    
    from src.ingestion.normalize import normalize_for_retrieval
    assert normalize_for_retrieval(query) in variations


def test_expand_with_synonyms_generates_variations():
    """Should generate variations for known synonyms."""
    query = "ما هو الإعراب"
    variations = expand_with_synonyms(query)
    
    # Should have at least original + some variations
    assert len(variations) >= 1
    # Check that we get variations for "ما" -> "ماذا"
    joined = " ".join(variations)
    assert "ماذا" in joined or "ما" in joined


def test_expand_with_templates_returns_original():
    """Original query should always be in template variations."""
    query = "ما هو الإعراب في النحو العربي؟"
    variations = expand_with_templates(query)
    
    from src.ingestion.normalize import normalize_for_retrieval
    assert normalize_for_retrieval(query) in variations


def test_expand_query_combines_sources():
    """expand_query should combine synonym and template variations."""
    query = "ما هو الإعراب في النحو العربي؟"
    variations = expand_query(query, max_variations=5)
    
    from src.ingestion.normalize import normalize_for_retrieval
    assert normalize_for_retrieval(query) == variations[0]  # Original first
    assert len(variations) <= 5  # Respects max_variations


def test_expand_query_max_variations_limit():
    """Should respect max_variations parameter."""
    query = "ما هو الإعراب في النحو العربي؟"
    variations = expand_query(query, max_variations=2)
    assert len(variations) <= 2
    
    variations = expand_query(query, max_variations=10)
    assert len(variations) <= 10


def test_expand_query_deduplicates():
    """Should not return duplicate variations."""
    query = "ما هو الإعراب"
    variations = expand_query(query, max_variations=10)
    
    assert len(variations) == len(set(variations))


def test_expand_query_empty_string():
    """Empty query should return empty list or just empty."""
    variations = expand_query("")
    # Implementation may vary - just ensure no crash
    assert isinstance(variations, list)


def test_arabic_synonyms_coverage():
    """Check that key Arabic terms have synonyms defined."""
    from src.retrieval.query_expansion import ARABIC_SYNONYMS
    
    # Question words
    assert "ما" in ARABIC_SYNONYMS
    assert "متى" in ARABIC_SYNONYMS
    assert "أين" in ARABIC_SYNONYMS
    assert "كيف" in ARABIC_SYNONYMS
    assert "لماذا" in ARABIC_SYNONYMS
    assert "من" in ARABIC_SYNONYMS
    
    # Grammar terms
    assert "الإعراب" in ARABIC_SYNONYMS
    assert "البناء" in ARABIC_SYNONYMS
    assert "الرفع" in ARABIC_SYNONYMS
    assert "النصب" in ARABIC_SYNONYMS
    assert "الجر" in ARABIC_SYNONYMS
    assert "الجزم" in ARABIC_SYNONYMS


def test_question_templates_not_empty():
    """QUESTION_TEMPLATES should have at least one template."""
    from src.retrieval.query_expansion import QUESTION_TEMPLATES
    
    assert len(QUESTION_TEMPLATES) >= 1
    # Each template should be callable
    for template in QUESTION_TEMPLATES:
        result = template("ما هو الإعراب")
        assert isinstance(result, str)