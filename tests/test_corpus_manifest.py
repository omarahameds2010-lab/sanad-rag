"""The corpus is a promise: these tests make it executable.

A benchmark is meaningless if the corpus silently contains filler. The
manifest is the single list of indexable sources; every file must exist,
match its recorded SHA-256, and pass the objective filler audit.
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.corpus.manifest import (  # noqa: E402
    load_manifest,
    manifest_sources,
    verify_source,
)

FILLER_FILES = [
    *(f"data/knowledge/adab_{i}.txt" for i in range(1, 9)),
    "data/knowledge/adab_al_talib.txt",
    "data/knowledge/hadith_qudsiya_large.txt",
]

BRANCHES = {"nahw", "sarf", "balagha", "arood", "adab", "lugha", "quran", "sira", "misc"}


def test_manifest_has_unique_audited_sources():
    sources = manifest_sources(load_manifest())
    assert len(sources) == 23
    assert len({s.source_id for s in sources}) == 23
    assert len({s.file for s in sources}) == 23
    for source in sources:
        assert source.branch in BRANCHES
        assert source.title.strip()
        assert source.sha256 and len(source.sha256) == 64


def test_manifest_hashes_match_committed_files():
    for source in manifest_sources(load_manifest()):
        text = verify_source(source)
        assert text.strip(), f"{source.file} is empty"


def test_manifest_never_includes_filler():
    files = {s.file for s in manifest_sources(load_manifest())}
    for filler in FILLER_FILES:
        assert filler not in files


def test_filler_files_are_gone_from_disk():
    for filler in FILLER_FILES:
        assert not (ROOT / filler).exists(), f"{filler} is filler and must not ship"


def test_excluded_patterns_are_explicit():
    excluded = load_manifest()["excluded"]
    patterns = " ".join(item.get("pattern", "") for item in excluded)
    assert "seeds_" in patterns
    assert "gen_supreme_core" in patterns
    assert "adab_" in patterns


def test_corpus_audit_passes_strict():
    result = subprocess.run(
        [sys.executable, "scripts/audit_corpus.py", "--strict"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
