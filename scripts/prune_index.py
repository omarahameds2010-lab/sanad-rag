"""Quarantine filler sources from the Qdrant index + SQLite bookkeeping.

Filler = content verified as padding, not knowledge:
- adab_1..8: 1000 repetitions of one identical line each
- final_touches / reach_25k / *touch: English filler or mojibake markers

MUST run with the API server STOPPED (embedded Qdrant takes an
exclusive file lock -- see src/retrieval/vector_store.get_client).

Usage:
  .venv/Scripts/python scripts/prune_index.py --dry-run   # count only
  .venv/Scripts/python scripts/prune_index.py --apply     # delete
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.retrieval.vector_store import delete_by_sources
from src.evaluation.storage import delete_ingested_document

# Audited 2026-10-13 by reading chunk text: padding, not knowledge.
FILLER_SOURCES = (
    [f"adab_{i}_v1" for i in range(1, 9)]
    + ["final_touches_v1", "reach_25k_v1"]
)


def main() -> None:
    ap = argparse.ArgumentParser(description="Quarantine filler sources")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--dry-run", action="store_true")
    g.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    if args.dry_run:
        # Count without deleting: scroll per collection directly.
        from src.retrieval.vector_store import get_client, COLLECTION_NAME
        from qdrant_client.models import FieldCondition, Filter, MatchAny

        client = get_client()
        total = 0
        for c in [c.name for c in client.get_collections().collections
                  if c.name == COLLECTION_NAME or c.name.startswith(COLLECTION_NAME + "_")]:
            pts, _ = client.scroll(
                collection_name=c,
                scroll_filter=Filter(must=[FieldCondition(
                    key="source_id", match=MatchAny(any=list(FILLER_SOURCES)))]),
                limit=100_000,
                with_payload=False,
            )
            print(f"  {c}: {len(pts)} filler points")
            total += len(pts)
        print(f"TOTAL filler points: {total} across {len(FILLER_SOURCES)} sources")
        print("dry-run only: nothing deleted.")
        return

    deleted = delete_by_sources(list(FILLER_SOURCES))
    for collection, n in deleted.items():
        print(f"  {collection}: deleted {n}")
    for sid in FILLER_SOURCES:
        delete_ingested_document(sid)
    print(f"APPLIED: quarantined {len(FILLER_SOURCES)} sources.")


if __name__ == "__main__":
    main()
