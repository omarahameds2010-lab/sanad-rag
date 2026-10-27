"""Executable corpus manifest.

SANAD's content rule is not a README promise: ingestion reads
``data/knowledge/corpus_manifest.json`` and verifies each file's SHA-256
before embedding it. A file that is not in the manifest cannot be indexed
by ``scripts/ingest_corpus.py`` or ``scripts/ingest_library.py``.

Hashes are computed over the universal-newline text (LF), so a Windows
checkout with ``core.autocrlf=true`` and a Linux CI checkout produce the
same fingerprint.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = REPO_ROOT / "data" / "knowledge" / "corpus_manifest.json"


@dataclass(frozen=True)
class CorpusSource:
    source_id: str
    file: str
    title: str
    branch: str
    chars: int
    sha256: str


def load_manifest(path: str | Path = DEFAULT_MANIFEST) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def manifest_sources(manifest: dict) -> list[CorpusSource]:
    return [CorpusSource(**item) for item in manifest["sources"]]


def verify_source(source: CorpusSource, root: Path = REPO_ROOT) -> str:
    """Return the file text, or raise if it no longer matches the manifest.

    A mismatch means the corpus changed without the manifest being updated —
    exactly the situation that must stop a benchmark before it produces
    untraceable numbers.
    """
    path = root / source.file
    text = path.read_text(encoding="utf-8")
    digest = sha256(text.encode("utf-8")).hexdigest()
    if digest != source.sha256:
        raise ValueError(
            f"{source.file}: sha256 mismatch (manifest {source.sha256[:12]}…, "
            f"file {digest[:12]}…). Update the manifest deliberately after "
            "re-auditing the content."
        )
    if len(text) != source.chars:
        raise ValueError(
            f"{source.file}: expected {source.chars} chars, found {len(text)}"
        )
    if source.source_id != path.stem + "_v1":
        raise ValueError(
            f"{source.source_id}: source_id must be the file stem + '_v1' "
            f"(expected {path.stem}_v1)"
        )
    return text
