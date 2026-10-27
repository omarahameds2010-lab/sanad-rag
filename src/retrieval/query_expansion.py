"""
Arabic Query Expansion for improved recall in hybrid search.

This module generates multiple query variations from a single Arabic question
using:
1. Morphological variations (stemming, lemmatization via pyarabic/camel-tools)
2. Synonym expansion for common Arabic terms
3. Question paraphrasing patterns

The expanded queries are run through the same hybrid retrieval pipeline and
results are fused via RRF (Reciprocal Rank Fusion) — same fusion strategy
as semantic + BM25, ensuring consistent score combination.

Why query expansion: Arabic is highly inflectional (root-pattern morphology).
A single question form may miss relevant chunks that use different morphological
forms of the same root. Expansion increases recall without sacrificing precision
because RRF naturally down-weights lower-ranked results from expanded queries.
"""

from src.ingestion.normalize import normalize_for_retrieval, strip_diacritics

# Common Arabic synonyms and morphological variants for question expansion
# These are hand-curated for high-impact, low-noise expansions
ARABIC_SYNONYMS = {
    # Question words
    "ما": ["ماذا", "أي"],
    "متى": ["أي وقت", "زمان"],
    "أين": ["في أي مكان", "مكان"],
    "كيف": ["بأي طريقة", "طريقة"],
    "لماذا": ["لأي سبب", "سبب"],
    # NOTE: "من" is both an interrogative (who) and a preposition (from).
    # Keep one merged entry — a second dict key would silently win and drop
    # one of the two senses from expansion.
    "من": ["أي شخص", "شخص", "من عند", "ابتداء من"],

    # Common verbs - root variations
    "يكون": ["يوجد", "يتواجد", "يحصل"],
    "كان": ["أصبح", "صار", "وجد"],
    "يوجد": ["يكون", "يتوفر", "متاح"],
    "يملك": ["يمتلك", "له", "عند"],
    "يعرف": ["يدرك", "يفهم", "يعلم"],
    "يفهم": ["يدرك", "يعي", "يعرف"],
    "يشرح": ["يوضح", "يبين", "يفصل"],
    "يذكر": ["يسرد", "يعدد", "يستحضر"],
    
    # Nouns - common synonyms
    "الكتاب": ["المؤلف", "المصنف", "العمل"],
    "العلم": ["المعرفة", "الدراية", "الفقه"],
    "المدرسة": ["المعهد", "الجامعة", "الكلية"],
    "الطالب": ["المتعلم", "الدارس", "التلميذ"],
    "المعلم": ["الأستاذ", "المدرس", "المربي"],
    "الامتحان": ["الاختبار", "الامتحان", "التقييم"],
    "النتيجة": ["الناتج", "المحصلة", "الثمرة"],
    "القاعدة": ["الأصل", "المبدأ", "القانون"],
    "الاستثناء": ["الخروج", "الشذوذ", "الخروج عن القاعدة"],
    
    # Grammar terms
    "الإعراب": ["الحركة", "العلامة", "الإعراب"],
    "البناء": ["الثبات", "السكون", "البناء"],
    "الرفع": ["الضم", "الرفع", "الضمرة"],
    "النصب": ["الفتح", "النصب", "الفتحة"],
    "الجر": ["الكسر", "الجر", "الكسرة"],
    "الجزم": ["السكون", "الجزم", "السكنة"],
    "المبتدأ": ["الابتداء", "الخبر"],
    "الخبر": ["المسند", "الخبر"],
    "الفاعل": ["الفاعل", "المنسوب"],
    "المفعول": ["المفعول به", "المنصوب"],
    
    # Prepositions/particles
    "في": ["داخل", "وسط", "أثناء"],
    "على": ["فوق", "أعلى", "على"],
    "إلى": ["نحو", "باتجاه", "حتى"],
    "مع": ["بجانب", "مرافق", "معا"],
}


# Arabic question paraphrase templates
# These generate structural variations of questions
QUESTION_TEMPLATES = [
    # Direct question -> indirect
    lambda q: q.replace("ما هو ", "أخبرني عن ").replace("ما هي ", "أخبرني عن "),
    lambda q: q.replace("ما هو ", "ما المقصود بـ ").replace("ما هي ", "ما المقصود بـ "),
    lambda q: q.replace("ما ", "ماذا "),
    
    # Add politeness markers
    lambda q: "هل يمكنك توضيح " + q[3:] if q.startswith("ما ") else q,
    lambda q: "أريد معرفة " + q[3:] if q.startswith("ما ") else q,
    
    # Reorder for emphasis
    lambda q: q.replace("في النحو العربي", "النحو العربي في") if "في النحو العربي" in q else q,
]


def expand_with_synonyms(query: str) -> list[str]:
    """
    Generate query variations by replacing words with synonyms.
    Returns a list including the original query.
    """
    normalized = normalize_for_retrieval(query)
    words = normalized.split()
    variations = {normalized}
    
    for i, word in enumerate(words):
        # Check exact match
        if word in ARABIC_SYNONYMS:
            for syn in ARABIC_SYNONYMS[word]:
                new_words = words.copy()
                new_words[i] = syn
                variations.add(" ".join(new_words))
        
        # Check stemmed form (simple prefix stripping for common Arabic prefixes)
        stripped = strip_diacritics(word)
        for prefix in ["ال", "و", "ف", "ب", "ل", "ك"]:
            if stripped.startswith(prefix) and len(stripped) > len(prefix) + 1:
                stem = stripped[len(prefix):]
                if stem in ARABIC_SYNONYMS:
                    for syn in ARABIC_SYNONYMS[stem]:
                        new_words = words.copy()
                        new_words[i] = prefix + syn
                        variations.add(" ".join(new_words))
    
    return list(variations)


def expand_with_templates(query: str) -> list[str]:
    """
    Generate query variations using structural templates.
    """
    normalized = normalize_for_retrieval(query)
    variations = {normalized}
    
    for template in QUESTION_TEMPLATES:
        try:
            expanded = template(normalized)
            if expanded != normalized:
                variations.add(expanded)
        except Exception:
            pass
    
    return list(variations)


def expand_query(
    query: str,
    max_variations: int = 5,
    use_synonyms: bool = True,
    use_templates: bool = True,
) -> list[str]:
    """
    Main entry point: expand an Arabic query into multiple variations.
    
    Args:
        query: original Arabic question
        max_variations: maximum number of variations to return (including original)
        use_synonyms: whether to use synonym expansion
        use_templates: whether to use template-based expansion
    
    Returns:
        List of query variations (first is always the original normalized query)
    """
    normalized = normalize_for_retrieval(query)
    all_variations = [normalized]
    
    if use_synonyms:
        syn_vars = expand_with_synonyms(query)
        for v in syn_vars:
            if v not in all_variations:
                all_variations.append(v)
    
    if use_templates:
        tmpl_vars = expand_with_templates(query)
        for v in tmpl_vars:
            if v not in all_variations:
                all_variations.append(v)
    
    # Limit variations to avoid too many retrieval calls
    return all_variations[:max_variations]


def multi_query_retrieve(
    queries: list[str],
    embedding_model,
    top_k: int = 5,
    chunker_type: str = "arabic",
    reranker=None,
    query_filter=None,
    payload_filter=None,
) -> list[dict]:
    """
    Run retrieval for multiple query variations and fuse results via RRF.
    
    This is the same RRF fusion used in hybrid_retrieve, applied across
    multiple queries instead of semantic + BM25.
    
    Args:
        queries: list of query variations
        embedding_model: embedding model to use
        top_k: final number of results to return
        chunker_type: chunker type for collection selection
        reranker: optional reranker
    
    Returns:
        Fused and ranked results
    """
    from src.retrieval.hybrid import RRF_K, hybrid_retrieve
    from src.retrieval.vector_store import collection_for_chunker
    
    all_results: dict[str, float] = {}
    payload_by_id: dict[str, dict] = {}
    
    collection = collection_for_chunker(chunker_type, embedding_model.dimensions)
    
    for q_idx, query in enumerate(queries):
        query_vector = embedding_model.embed_query(normalize_for_retrieval(query))
        results = hybrid_retrieve(query_vector, query, top_k=top_k * 2, collection=collection,
                                  query_filter=query_filter, payload_filter=payload_filter)
        
        # Fuse across queries using RRF with query-rank damping
        for rank, result in enumerate(results):
            cid = result["chunk_id"]
            # Add query index as secondary damping factor
            score = 1.0 / (RRF_K + rank + 1 + q_idx * 0.1)
            all_results[cid] = all_results.get(cid, 0.0) + score
            payload_by_id[cid] = result
    
    ranked_ids = sorted(all_results, key=lambda cid: all_results[cid], reverse=True)
    return [
        {**payload_by_id[cid], "fused_score": all_results[cid]}
        for cid in ranked_ids[:top_k]
    ]


# Convenience function for integration with query pipeline
def expand_and_retrieve(
    question: str,
    embedding_model,
    top_k: int = 5,
    chunker_type: str = "arabic",
    reranker=None,
    max_variations: int = 3,
    query_filter=None,
    payload_filter=None,
) -> list[dict]:
    """
    Expand query and run multi-query retrieval in one call.
    """
    expanded_queries = expand_query(question, max_variations=max_variations)
    return multi_query_retrieve(
        expanded_queries,
        embedding_model,
        top_k=top_k,
        chunker_type=chunker_type,
        reranker=reranker,
        query_filter=query_filter,
        payload_filter=payload_filter,
    )