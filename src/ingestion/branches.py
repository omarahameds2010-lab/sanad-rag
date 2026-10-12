"""Branch taxonomy for the SANAD knowledge library.

Each ingested source gets a `branch` payload field (e.g. "nahw", "adab")
so the Evidence Explorer can offer sub-filters inside one subject instead
of separate subjects per science.

Rules are data-only (no imports) so this module can be imported from
anywhere without circular-import risk. First matching rule wins; unknown
stems fall back to "misc" (honest default, never a wrong science).
"""

# Canonical branch ids (stable API contract — do not rename).
BRANCHES = (
    "nahw",    # النحو والإعراب
    "sarf",    # الصرف والإملاء
    "balagha", # البلاغة والنقد
    "arood",   # العروض والقوافي
    "adab",    # الأدب والنصوص
    "lugha",   # اللغة والمفردات
    "quran",   # القرآن وعلومه
    "hadith",  # الحديث وعلومه
    "sira",    # السيرة والتاريخ
    "misc",    # عام / غير مصنف
)

# (branch, filename keywords). Ordered: specific sciences before general ones.
BRANCH_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("sarf", ("sarf", "emlaa")),
    ("nahw", ("nahw", "irab", "alfiyya", "ajurrumiyya", "huruf")),
    ("balagha", ("balagha", "fonoun")),
    ("arood", ("arood", "muallaqat")),
    ("quran", ("juzamma", "tafsir", "duaa_quran", "asbab")),
    ("hadith", ("hadith", "arbaeen", "qudsiya", "ahadith")),
    ("sira", ("seera", "tarikh")),
    ("adab", ("adab", "diwan", "nahj", "hikam", "amthal", "taabeer",
               "insha", "nosous", "qiraa", "aalam")),
    ("lugha", ("vocab", "fiqh_lugha")),
)


def infer_branch(source_id: str) -> str:
    """Infer a branch from a source_id / file stem. Never raises."""
    stem = (source_id or "").lower()
    for branch, keywords in BRANCH_RULES:
        for kw in keywords:
            if kw in stem:
                return branch
    return "misc"
