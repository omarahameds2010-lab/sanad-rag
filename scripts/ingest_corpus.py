"""Ingest exactly the audited corpus manifest — nothing else.

This is the canonical direct-ingestion entry point used for benchmarks. It
verifies every file's SHA-256 against ``data/knowledge/corpus_manifest.json``
before embedding anything, so filler, seeds, and local-only shards cannot
silently enter the index.

Usage:
    python scripts/ingest_corpus.py --dry-run
    EMBEDDING_MODEL=bge-m3 QDRANT_URL=http://localhost:6333 \\
        python scripts/ingest_corpus.py --fresh

The embedding model and Qdrant target come from ``EMBEDDING_MODEL`` and
``QDRANT_URL``; the model name is printed and recorded with every source.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from _models import get_embedding_model

from src.corpus.manifest import (
    DEFAULT_MANIFEST,
    CorpusSource,
    load_manifest,
    manifest_sources,
    verify_source,
)
from src.ingestion.pipeline import ingest_document
from src.retrieval.vector_store import collection_for_chunker, get_client


def target_collections(chunker: str, dims: int) -> list[str]:
    if chunker == "hierarchical":
        return [
            collection_for_chunker("hierarchical_children", dims),
            collection_for_chunker("hierarchical_parents", dims),
        ]
    return [collection_for_chunker(chunker, dims)]


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument(
        "--chunker",
        default="arabic",
        choices=["arabic", "baseline", "hierarchical"],
    )
    parser.add_argument(
        "--only",
        nargs="*",
        default=None,
        help="Ingest only these source_ids (still manifest-verified).",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Delete the target collection(s) before ingesting.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    manifest = load_manifest(args.manifest)
    sources = manifest_sources(manifest)
    if args.only:
        wanted = set(args.only)
        sources = [s for s in sources if s.source_id in wanted]
        missing = wanted - {s.source_id for s in sources}
        if missing:
            print(f"Unknown source_id(s): {sorted(missing)}", file=sys.stderr)
            return 2
    if not sources:
        print("No sources selected.", file=sys.stderr)
        return 2

    # Verify the whole selection before touching Qdrant: a half-ingested corpus
    # is worse than no corpus.
    verified: list[tuple[CorpusSource, str]] = []
    for source in sources:
        try:
            text = verify_source(source)
        except (OSError, ValueError) as exc:
            print(f"✗ {source.source_id}: {exc}", file=sys.stderr)
            return 1
        verified.append((source, text))

    print(
        f"Manifest {args.manifest}: {len(verified)} source(s), "
        f"{sum(len(t) for _, t in verified):,} chars"
    )
    if args.dry_run:
        for source, text in verified:
            print(
                f"  {source.source_id:28} {source.branch:8} "
                f"{len(text):7,} chars  {source.title}"
            )
        return 0

    embedding_model = get_embedding_model()
    print(f"Embedding model: {embedding_model.name} ({embedding_model.dimensions}d)")

    if args.fresh:
        client = get_client()
        for name in target_collections(args.chunker, embedding_model.dimensions):
            if client.collection_exists(name):
                client.delete_collection(name)
                print(f"Deleted collection {name}")

    started = time.perf_counter()
    total_chunks = 0
    for index, (source, text) in enumerate(verified, start=1):
        t0 = time.perf_counter()
        result = ingest_document(
            source_id=source.source_id,
            raw_text=text,
            embedding_model=embedding_model,
            metadata={
                "dataset": "sanad-library",
                "file": Path(source.file).name,
                "title": source.title,
                "branch": source.branch,
            },
            chunker_type=args.chunker,
        )
        elapsed = time.perf_counter() - t0
        total_chunks += result.get("n_chunks", 0)
        print(
            f"[{index:2}/{len(verified)}] {source.source_id:28} "
            f"chunks={result.get('n_chunks', 0):4d}  {elapsed:6.1f}s"
        )

    total_elapsed = time.perf_counter() - started
    print(
        f"\nIngested {len(verified)} sources / {total_chunks} chunks in "
        f"{total_elapsed:.1f}s with {embedding_model.name}."
    )
    print("Target collection(s): " + ", ".join(target_collections(args.chunker, embedding_model.dimensions)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
