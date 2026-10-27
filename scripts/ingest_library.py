"""Ingest the audited SANAD corpus through the running API.

Only sources listed in ``data/knowledge/corpus_manifest.json`` are sent, and
each file is SHA-256 verified first. Seeds, filler, and local-only shards are
not reachable from this script.

Usage:
    python scripts/ingest_library.py --dry-run
    SANAD_API_KEY=... python scripts/ingest_library.py --chunker hierarchical
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.corpus.manifest import (
    DEFAULT_MANIFEST,
    load_manifest,
    manifest_sources,
    verify_source,
)

BASE = os.environ.get("SANAD_API_URL", "http://localhost:8000")
HEADERS = {
    "Content-Type": "application/json",
    "X-API-Key": os.environ.get("SANAD_API_KEY", "demo-key-123"),
}


def call(method, path, payload=None, timeout=300):
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8") if payload is not None else None
    req = urllib.request.Request(BASE + path, data=data, headers=HEADERS, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", "replace")


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument(
        "--chunker",
        default="hierarchical",
        choices=["arabic", "baseline", "hierarchical"],
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    sources = manifest_sources(load_manifest(args.manifest))
    print(f"Manifest: {len(sources)} approved source(s)")

    failed = 0
    for source in sources:
        try:
            text = verify_source(source)
        except (OSError, ValueError) as exc:
            print(f"✗ {source.source_id}: {exc}", file=sys.stderr)
            failed += 1
            continue
        if args.dry_run:
            print(f"  {source.source_id:28} {len(text):7,} chars  {source.title}")
            continue
        status, body = call(
            "POST",
            "/ingest",
            {
                "source_id": source.source_id,
                "text": text,
                "metadata": {
                    "dataset": "sanad-library",
                    "file": Path(source.file).name,
                    "title": source.title,
                    "branch": source.branch,
                },
                "chunker_type": args.chunker,
            },
            timeout=600,
        )
        n = body.get("n_chunks") if isinstance(body, dict) else body
        print(f"{source.source_id}: {status} chunks={n}")
        if status != 200:
            failed += 1

    if failed:
        print(f"{failed} source(s) failed", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
