"""Ingest the full SANAD knowledge library (committed corpus only).

Skips seeds_*.txt (raw generator input — the expanded gen_*.txt files
are ingested instead, to avoid duplicate content).

Usage:
    python scripts/ingest_library.py
"""

import json
import urllib.request
from pathlib import Path

BASE = "http://localhost:8000"
HEADERS = {"Content-Type": "application/json", "X-API-Key": "demo-key-123"}
ROOT = Path(__file__).parent.parent
KNOWLEDGE = ROOT / "data" / "knowledge"


def call(method, path, payload=None, timeout=300):
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8") if payload is not None else None
    req = urllib.request.Request(BASE + path, data=data, headers=HEADERS, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", "replace")


def main() -> None:
    files = sorted(
        p for p in KNOWLEDGE.glob("*.txt")
        if not p.name.startswith("seeds_") and not p.name.startswith(".")
    )
    print(f"Found {len(files)} knowledge files")
    for path in files:
        text = path.read_text(encoding="utf-8")
        source_id = path.stem + "_v1"
        status, body = call("POST", "/ingest", {
            "source_id": source_id,
            "text": text,
            "metadata": {"dataset": "sanad-library", "file": path.name},
            "chunker_type": "hierarchical",
        }, timeout=600)
        n = body.get("n_chunks") if isinstance(body, dict) else body
        print(f"{source_id}: {status} chunks={n}")


if __name__ == "__main__":
    main()
