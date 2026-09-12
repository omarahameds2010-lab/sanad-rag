"""
Arabic text normalization for ingestion.

Why this exists:
Most generic text pipelines (built for English) don't correctly handle:
- Diacritics (تشكيل) which add noise to embeddings if not stripped consistently
- Alef/Ya/Ta-Marbuta variants that should often be unified for retrieval
- Tatweel (kashida) elongation characters used for justification, which are
  visual-only and carry no semantic value

This module is intentionally separate from chunking: normalization changes
character content, chunking changes segmentation. Keeping them separate makes
it possible to run controlled experiments (e.g. "chunking with vs without
normalization") without conflating the two variables.
"""

import re
import unicodedata

# Arabic diacritics (tashkeel) Unicode range
ARABIC_DIACRITICS = re.compile(r"[\u064B-\u0652\u0670\u0640]")

# Common character variant mappings used for retrieval-time normalization.
# NOTE: these are applied to a *copy* used for embedding/matching — the
# original text is preserved for display and citation purposes.
CHAR_VARIANTS = {
    "أ": "ا", "إ": "ا", "آ": "ا",
    "ى": "ي",
    "ة": "ه",
}


def strip_diacritics(text: str) -> str:
    """Remove Arabic diacritics and tatweel elongation characters."""
    return ARABIC_DIACRITICS.sub("", text)


def normalize_variants(text: str) -> str:
    """Unify common Arabic character variants for retrieval matching."""
    for src, dst in CHAR_VARIANTS.items():
        text = text.replace(src, dst)
    return text


def normalize_for_retrieval(text: str) -> str:
    """
    Full normalization pipeline used to build the *matching* representation
    of a passage. The original (un-normalized) text is what gets shown to
    the user and cited — never the normalized version.
    """
    text = unicodedata.normalize("NFKC", text)
    text = strip_diacritics(text)
    text = normalize_variants(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def clean_for_display(text: str) -> str:
    """Light cleanup only — preserves diacritics and original characters."""
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()
